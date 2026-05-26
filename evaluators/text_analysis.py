# ─── evaluators/text_analysis.py ───────────────────────────────
import os, math, time, random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

from rich.console  import Console
from rich.table    import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich          import box

# ─── CONSTANTS ─────────────────────────────────────────────────
BATCH_SIZE     = 32
MAX_LENGTH     = 128
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSET_RATIO   = 0.50          # evaluate on 50 % of the test split
RESULTS_ROOT   = Path(__file__).parent.parent / "results" / "text"
CSV_PATH       = RESULTS_ROOT / "text_analysis_evaluation_results.csv"
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

# ----------------------- single-run evaluator ------------------
def _evaluate_once(
    model_dir: Path,
    test_csv: Path,
    out_dir: Path,
    run_idx: int,
    total_runs: int,
    seed: int,
) -> float:
    """
    Runs ONE evaluation pass on a 50 % random subset and saves
    all artefacts under `out_dir`. Returns accuracy (percentage).
    """
    rng = np.random.default_rng(seed)

    # ------------ load data ------------
    ds      = load_dataset("csv", data_files={"test": str(test_csv)})["test"]
    texts   = list(ds["text"])
    labels  = np.asarray(ds["label"])
    n_total = len(texts)

    n_subset   = max(1, int(n_total * SUBSET_RATIO))
    subset_idx = rng.choice(n_total, n_subset, replace=False)
    texts      = [texts[i] for i in subset_idx]
    labels     = labels[subset_idx]

    # ------------ model / tokenizer ------------
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE).eval()

    # ------------ inference ------------
    all_probs, all_preds, all_embs = [], [], []

    with _make_progress(f"Inferencing {run_idx}/{total_runs}") as progress:
        task = progress.add_task("", total=math.ceil(n_subset / BATCH_SIZE))
        with torch.no_grad():
            for i in range(0, n_subset, BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                enc   = tokenizer(
                    batch, truncation=True, padding="max_length",
                    max_length=MAX_LENGTH, return_tensors="pt"
                )
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True)

                probs = torch.softmax(out.logits, dim=-1)
                preds = probs.argmax(dim=-1)
                emb   = out.hidden_states[-1].mean(dim=1)  # mean-pooled emb

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_embs.append(emb.cpu().numpy())
                progress.update(task, advance=1)

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_embs  = np.vstack(all_embs)

    # ------------ metrics ------------
    acc   = accuracy_score(labels, all_preds) * 100
    prec  = precision_score(labels, all_preds, average="weighted", zero_division=0)
    rec   = recall_score(labels,  all_preds, average="weighted", zero_division=0)
    f1_w  = f1_score(labels,     all_preds, average="weighted", zero_division=0)

    # ------------ saving ------------
    build_dirs(out_dir, out_dir / "plots")

    with open(out_dir / "overall_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {acc/100:.4f}\n"
                f"Precision: {prec:.4f}\n"
                f"Recall   : {rec:.4f}\n"
                f"F1-score : {f1_w:.4f}\n")

    rep_dict = classification_report(labels, all_preds, digits=4,
                                     output_dict=True, zero_division=0)
    rep_str  = classification_report(labels, all_preds, digits=4,
                                     zero_division=0)
    with open(out_dir / "class_metrics.txt", "w", encoding="utf-8") as f:
        f.write(rep_str)
    pd.DataFrame(rep_dict).transpose().to_csv(out_dir / "classification_report.csv")

    cm = confusion_matrix(labels, all_preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "confusion_matrix.png")
    plt.close()

    classes    = sorted(set(int(x) for x in labels))
    y_true_bin = label_binarize(labels, classes=classes)

    plt.figure()
    for i, cls in enumerate(classes):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "roc_curve.png")
    plt.close()

    plt.figure()
    for i, cls in enumerate(classes):
        if y_true_bin[:, i].sum() == 0:
            continue
        p, r, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        plt.plot(r, p, label=f"Class {cls} (AUC={auc(r, p):.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "pr_curve.png")
    plt.close()

    pca = PCA(n_components=2, random_state=42).fit_transform(all_embs)
    plt.figure(figsize=(6, 5))
    plt.scatter(pca[:, 0], pca[:, 1], c=labels, cmap='tab10', s=5)
    plt.title("PCA of Embeddings")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "embedding_pca.png")
    plt.close()

    n_tsne = min(1000, all_embs.shape[0])
    em_tsne = TSNE(n_components=2, random_state=42).fit_transform(
        all_embs[np.random.choice(all_embs.shape[0], n_tsne, replace=False)]
    )
    plt.figure(figsize=(6, 5))
    plt.scatter(em_tsne[:, 0], em_tsne[:, 1], c=labels[:n_tsne], cmap='tab10', s=5)
    plt.title("t-SNE of Embeddings")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "embedding_tsne.png")
    plt.close()

    # ------------ Rich summary ------------
    table = Table(title="Text Analysis Model Evaluation Summary",
                  box=box.MINIMAL)
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="magenta")
    table.add_row("Accuracy",   f"{acc:.2f}%")
    table.add_row("Weighted F1",f"{f1_w:.4f}")
    console.print(table)

    return acc  # %

# ----------------------- batch driver --------------------------
def _batch_evaluate(model_name: str, split_name: str, iterations: int):
    """
    Repeats `_evaluate_once` *iterations* times on fresh 50 % subsets,
    storing artefacts in results/text/result_ⓝ sub-folders and
    writing an aggregated CSV (with stability row).
    """
    project_root = Path(__file__).parent.parent
    models_dir   = project_root / "models"
    data_dir     = project_root / "data"

    model_dir = models_dir / model_name
    test_csv  = data_dir / f"{split_name}.csv"

    if not model_dir.is_dir():
        console.print(f"[red]Error:[/red] model folder not found → {model_dir}")
        raise SystemExit(1)
    if not test_csv.is_file():
        console.print(f"[red]Error:[/red] test CSV not found → {test_csv}")
        raise SystemExit(1)

    build_dirs(RESULTS_ROOT)
    records = []

    console.print("[bold green]==== Starting batch evaluation ====[/bold green]")
    for i in range(1, iterations + 1):
        start = time.time()
        seed  = random.randint(0, 2**32 - 1)

        out_dir = RESULTS_ROOT / f"result_{i}"
        acc = _evaluate_once(model_dir, test_csv, out_dir,
                             run_idx=i, total_runs=iterations, seed=seed)

        now = datetime.now().strftime("%H:%M:%S")
        console.print(
            f"[bold yellow][ITER {i:03d}/{iterations}] "
            f"Accuracy: {acc:.2f}%   Time: {now} "
            f"({time.time() - start:.1f}s)[/bold yellow]"
        )
        records.append({
            "number_of_eval": i,
            "model_name":     model_dir.name,
            "test_data_name": test_csv.name,
            "accuracy":       round(acc, 2),
            "time":           now
        })

    # -------- stability row --------
    accs       = [r["accuracy"] for r in records]   # each is 0-100 %
    mean_acc   = sum(accs) / len(accs)              # reference value
    tolerance  = 2.0                                # ±2 percentage points

    accurate   = sum(abs(a - mean_acc) <= tolerance for a in accs)
    overall    = accurate / len(accs) * 100    
    records.append({
        "number_of_eval": "summary",
        "model_name":     model_dir.name,
        "test_data_name": test_csv.name,
        "accuracy":       round(overall, 2),
        "time":           datetime.now().strftime("%H:%M:%S")
    })

    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    console.print(f"\n[green]Saved aggregated CSV →[/green] {CSV_PATH}")
    console.print("[bold green]==== Evaluation complete! ====[/bold green]")

# ----------------------- public wrapper ------------------------
def run_text_analysis(model_name: str, split_name: str, loops: int):
    """
    Public entry-point used by cli.py.
    Simply forwards to the batch driver with the given *loops*.
    """
    _batch_evaluate(model_name, split_name, iterations=loops)

# -------------------------- CLI (stand-alone) ------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Batch-evaluate a HuggingFace text classifier")
    ap.add_argument("--model", required=True,
                    help="folder name inside ./models/")
    ap.add_argument("--split", required=True,
                    help="CSV file name (without .csv) inside ./data/")
    ap.add_argument("--loops", type=int, default=None,
                    help="number of evaluation iterations")
    args = ap.parse_args()

    loops = args.loops or int(
        console.input("[cyan]Enter number of evaluation loops ➜ [/cyan]"))
    run_text_analysis(args.model, args.split, loops)
# ───────────────────────────────────────────────────────────────
