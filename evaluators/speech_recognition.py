# ─── evaluators/speech_recognition.py ──────────────────────────
import os, random, time, math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import evaluate

from rich.console  import Console
from rich.table    import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich          import box

# ─── CONSTANTS ─────────────────────────────────────────────────
SUBSET_RATIO   = 0.50  # evaluate on 50 % of pairs each loop
RESULTS_ROOT   = Path(__file__).parent.parent / "results" / "speech"
CSV_PATH       = RESULTS_ROOT / "speech_recognition_evaluation_results.csv"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 8
# ───────────────────────────────────────────────────────────────

console = Console()

# --------------------------- helpers ---------------------------
def build_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def _make_progress(desc: str) -> Progress:
    return Progress(
        f"[bold cyan]{desc}[/bold cyan]",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
        transient=False,
    )

def find_file(name: str, ext: str, root: Path) -> Path:
    fname   = name + ext if not name.endswith(ext) else name
    matches = list(root.rglob(fname))
    if not matches:
        console.print(f"[red]Error:[/red] Could not find '{fname}' under {root}")
        raise SystemExit(1)
    if len(matches) > 1:
        console.print(f"[yellow]Warning:[/yellow] Multiple '{fname}' found. Using {matches[0]}")
    return matches[0]

# ------------------- data collection utilities ----------------
def _collect_pairs(split_root: Path):
    pairs = []
    for spk_dir in split_root.iterdir():
        if not spk_dir.is_dir(): continue
        for book_dir in spk_dir.iterdir():
            if not book_dir.is_dir(): continue
            tfile = book_dir / f"{spk_dir.name}-{book_dir.name}.trans.txt"
            if not tfile.exists(): continue
            mapping = {l.split(' ', 1)[0]: l.split(' ', 1)[1].strip().upper()
                       for l in tfile.read_text(encoding="utf-8").splitlines()}
            for flac in book_dir.glob("*.flac"):
                if flac.stem in mapping:
                    pairs.append((flac, mapping[flac.stem]))
    return pairs

class ASRDataset(Dataset):
    def __init__(self, pairs, processor):
        self.pairs = pairs
        self.proc  = processor
    def __len__(self):  return len(self.pairs)
    def __getitem__(self, idx):
        path, text = self.pairs[idx]
        wav, sr    = sf.read(str(path))
        inp = self.proc.feature_extractor(
            wav, sampling_rate=sr, return_tensors="pt").input_values[0]
        labels = self.proc.tokenizer(text, add_special_tokens=False).input_ids
        return inp, torch.tensor(labels, dtype=torch.long)

def _collate(batch, processor):
    inputs, labels = zip(*batch)
    ivals = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    amask = (ivals != 0.0).long()
    labs  = pad_sequence(labels, batch_first=True,
                         padding_value=processor.tokenizer.pad_token_id)
    labs  = labs.masked_fill(labs == processor.tokenizer.pad_token_id, -100)
    return {"input_values": ivals, "attention_mask": amask, "labels": labs}

# -------------------- single-run evaluator --------------------
def _evaluate_once(
    model_path: Path,
    dataset: ASRDataset,
    subset_idx: np.ndarray,
    out_dir: Path,
    run_idx: int,
    total_runs: int,
) -> float:
    loader = DataLoader(
        Subset(dataset, subset_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: _collate(b, dataset.proc)
    )

    model = Wav2Vec2ForCTC.from_pretrained(str(model_path)).to(DEVICE).eval()
    wer_metric = evaluate.load("wer")

    all_refs, all_hyps, scores = [], [], []

    with torch.no_grad(), _make_progress(f"Inferencing {run_idx}/{total_runs}") as prog:
        task = prog.add_task("", total=math.ceil(len(subset_idx) / BATCH_SIZE))
        for batch in loader:
            iv = batch["input_values"].to(DEVICE)
            am = batch["attention_mask"].to(DEVICE)
            logits = model(iv, attention_mask=am).logits.cpu().numpy()
            pred_ids = logits.argmax(axis=-1)
            hyps = dataset.proc.batch_decode(pred_ids)

            labs = batch["labels"].cpu().numpy()
            labs[labs == -100] = dataset.proc.tokenizer.pad_token_id
            refs = dataset.proc.batch_decode(labs, group_tokens=False)

            all_hyps.extend(hyps)
            all_refs.extend(refs)
            scores.extend([wer_metric.compute(predictions=[h], references=[r])
                           for h, r in zip(hyps, refs)])
            prog.update(task, advance=1)

    test_wer = wer_metric.compute(predictions=all_hyps, references=all_refs)
    accuracy = (1.0 - test_wer) * 100

    # ------- save artefacts -------
    build_dirs(out_dir)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Test WER: {test_wer:.4f}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Median per-utterance WER: {np.median(scores):.4f}\n")
        f.write(f"95th-percentile WER: {np.percentile(scores,95):.4f}\n")

    with open(out_dir / "examples.txt", "w", encoding="utf-8") as f:
        f.write("First 10 REF vs HYP pairs:\n\n")
        for i in range(min(10, len(all_refs))):
            f.write(f"REF: {all_refs[i]}\nHYP: {all_hyps[i]}\n{'-'*30}\n")

    pd.DataFrame({
        "audio_path": [dataset.pairs[i][0] for i in subset_idx],
        "reference":  all_refs,
        "hypothesis": all_hyps,
        "wer":        scores
    }).to_csv(out_dir / "results.csv", index=False)

    # plots
    plt.figure(); plt.hist(scores, bins=20)
    plt.title("Per-utterance WER Distribution")
    plt.xlabel("WER"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out_dir / "wer_histogram.png"); plt.close()

    lengths = [len(r.split()) for r in all_refs]
    plt.figure(); plt.scatter(lengths, scores)
    plt.title("Utterance Length vs WER")
    plt.xlabel("Utterance Length (words)"); plt.ylabel("WER")
    plt.tight_layout(); plt.savefig(out_dir / "length_vs_wer.png"); plt.close()

    # Rich summary
    summary_lines = (out_dir / "summary.txt").read_text().splitlines()
    table = Table(title="Speech Recognition Model Evaluation Summary",
                  box=box.MINIMAL)
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="magenta")
    for line in summary_lines:
        if ": " in line:
            k, v = line.split(": ", 1)
            table.add_row(k, v)
    console.print(table)

    return accuracy  # %

# ---------------------- batch driver ---------------------------
def _batch_evaluate(model_name: str, test_split_name: str, iterations: int):
    proj_root  = Path(__file__).parent.parent
    models_dir = proj_root / "models"
    data_dir   = proj_root / "data"

    model_path = find_file(model_name, "", models_dir)
    test_root  = find_file(test_split_name, "", data_dir)

    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
    pairs     = _collect_pairs(test_root)
    if not pairs:
        console.print(f"[red]Error:[/red] No data found under {test_root}")
        raise SystemExit(1)

    dataset = ASRDataset(pairs, processor)
    build_dirs(RESULTS_ROOT)
    records = []

    console.print("[bold green]==== Starting batch evaluation ====[/bold green]")
    for i in range(1, iterations + 1):
        start = time.time()
        subset_len = max(1, int(len(dataset) * SUBSET_RATIO))
        subset_idx = np.random.choice(len(dataset), subset_len, replace=False)

        out_dir = RESULTS_ROOT / f"result_{i}"
        acc = _evaluate_once(model_path, dataset, subset_idx,
                             out_dir, run_idx=i, total_runs=iterations)

        now = datetime.now().strftime("%H:%M:%S")
        console.print(
            f"[bold yellow][ITER {i:03d}/{iterations}] "
            f"Accuracy: {acc:.2f}%   Time: {now} "
            f"({time.time() - start:.1f}s)[/bold yellow]"
        )
        records.append({
            "number_of_eval": i,
            "model_name":     model_path.name,
            "test_data_name": test_root.name,
            "accuracy":       round(acc, 2),
            "time":           now
        })

    # stability row
    # ---------- reproducibility accuracy (±2 pts) ----------
    accs       = [r["accuracy"] for r in records]   # each is 0-100 %
    mean_acc   = sum(accs) / len(accs)              # reference value
    tolerance  = 2.0                                # ±2 percentage points

    accurate   = sum(abs(a - mean_acc) <= tolerance for a in accs)
    overall    = accurate / len(accs) * 100         # % of runs within the band

    records.append({
    "number_of_eval": "summary",
    "model_name":     model_path.name,
    "test_data_name": test_root.name,
    "accuracy":       round(overall, 2),        # e.g. 98.33
    "time":           datetime.now().strftime("%H:%M:%S")
})


    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    console.print(f"\n[green]Saved aggregated CSV →[/green] {CSV_PATH}")
    console.print("[bold green]==== Evaluation complete! ====[/bold green]")

# ----------------------- public entry --------------------------
def run_speech_recognition(model_name: str, test_split_name: str,
                           loops: int | None = None):
    if loops is None:
        loops = int(console.input("[cyan]Enter number of evaluation loops ➜ [/cyan]"))
    _batch_evaluate(model_name, test_split_name, iterations=loops)

# -------------------------- CLI (stand-alone) ------------------
if __name__ == "__main__":
    console.print("[bold]Speech Recognition Evaluation[/bold]\n")
    m = console.input("Model folder name under 'models': ")
    d = console.input("Test split folder under 'data': ")
    loops = int(console.input("Number of evaluation loops: ").strip())
    run_speech_recognition(m, d, loops)
# ───────────────────────────────────────────────────────────────
