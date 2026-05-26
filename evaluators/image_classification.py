# ─── evaluators/image_classification.py ────────────────────────
import os, math, time, random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50

from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    roc_curve, auc
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

from rich.console  import Console
from rich.table    import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich          import box
from tqdm.auto     import tqdm

# ─── CONSTANTS ─────────────────────────────────────────────────
BATCH_SIZE     = 64
SUBSET_RATIO   = 0.50   # evaluate on 50 % random subset
RESULTS_ROOT   = Path(__file__).parent.parent / "results" / "classification"
CSV_PATH       = RESULTS_ROOT / "classification_evaluation_results.csv"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# ------------------------- dataset ----------------------------
class CIFAR10Folder(Dataset):
    CIFAR10_CLASSES = [str(i) for i in range(10)]
    IDX_TO_CLASS = {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
        4: "deer",     5: "dog",        6: "frog", 7: "horse",
        8: "ship",     9: "truck"
    }

    def __init__(self, root_dir: str):
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247,  0.243,  0.261))
        ])
        self.paths, self.labels = [], []
        for idx, cls in enumerate(self.CIFAR10_CLASSES):
            cls_path = Path(root_dir) / cls
            for f in cls_path.glob("*.[jp][pn]g"):
                self.paths.append(str(f))
                self.labels.append(idx)
        self.transform = tfm

    def __len__(self):  return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]

# --------------------- single-run evaluator -------------------
def _evaluate_once(
    model_path: str,
    dataset: CIFAR10Folder,
    subset_idx: np.ndarray,
    out_dir: Path,
    run_idx: int,
    total_runs: int,
):
    """Evaluate on a subset and write all artefacts under `out_dir`."""
    loader = DataLoader(Subset(dataset, subset_idx),
                        batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

    # ------- model -------
    model = resnet50(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                            padding=1, bias=False)
    model.maxpool = nn.Identity()
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # ------- inference -------
    all_labels, all_preds, all_probs = [], [], []
    with _make_progress(f"Inferencing {run_idx}/{total_runs}") as progress:
        task = progress.add_task("", total=math.ceil(len(subset_idx) / BATCH_SIZE))
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), torch.tensor(lbls).to(DEVICE)
            with torch.no_grad():
                out   = model(imgs)
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)
            all_labels.extend(lbls.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            progress.update(task, advance=1)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.concatenate(all_probs, axis=0)

    # ------- metrics -------
    acc      = accuracy_score(all_labels, all_preds) * 100
    top5     = np.argsort(-all_probs, axis=1)[:, :5]
    top5_acc = np.mean([lbl in preds for lbl, preds
                        in zip(all_labels, top5)]) * 100

    # ------- save artefacts -------
    build_dirs(out_dir) 

    with open(out_dir / "results.txt", "w") as f:
        f.write(f"Top-1 Accuracy: {acc:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")
        f.write(f"Total images tested: {len(all_labels)}\n")

    report = classification_report(
        all_labels, all_preds,
        target_names=[CIFAR10Folder.IDX_TO_CLASS[i] for i in range(10)],
        digits=4
    )
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    with open(out_dir / "top5_accuracy.txt", "w") as f:
        f.write(f"Top-1 Accuracy: {acc:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")

    with open(out_dir / "results_by_class.txt", "w") as f:
        for i in range(10):
            mask = (all_labels == i)
            n    = mask.sum()
            cls_acc = (all_preds[mask] == i).sum() / n * 100 if n else 0
            f.write(f"{CIFAR10Folder.IDX_TO_CLASS[i]}: "
                    f"{cls_acc:.2f}% ({n} samples)\n")

    # confusion matrices
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), CIFAR10Folder.IDX_TO_CLASS.values(),
               rotation=45, ha="right")
    plt.yticks(range(10), CIFAR10Folder.IDX_TO_CLASS.values())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "cm_results.png")
    plt.close()

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), CIFAR10Folder.IDX_TO_CLASS.values(),
               rotation=45, ha="right")
    plt.yticks(range(10), CIFAR10Folder.IDX_TO_CLASS.values())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "cm_normalized.png")
    plt.close()

    # confidence histogram
    plt.figure()
    plt.hist(all_probs.max(axis=1), bins=10)
    plt.title("Confidence Histogram")
    plt.xlabel("Max Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_histogram.png")
    plt.close()

    # calibration curves
    plt.figure()
    y_bin = label_binarize(all_labels, classes=list(range(10)))
    for i in range(10):
        frac_pos, mean_pred = calibration_curve(y_bin[:, i],
                                                all_probs[:, i],
                                                n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o",
                 label=CIFAR10Folder.IDX_TO_CLASS[i])
    plt.plot([0, 1], [0, 1], "--")
    plt.title("Calibration Curves")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png")
    plt.close()

    # ROC curves
    plt.figure()
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr,
                 label=f"{CIFAR10Folder.IDX_TO_CLASS[i]} "
                       f"(AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png")
    plt.close()

    # ------- Rich summary -------
    table = Table(title="Image Classification Model Evaluation Summary",
                  box=box.MINIMAL)
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="magenta")
    table.add_row("Top-1 Accuracy", f"{acc:.2f}%")
    table.add_row("Top-5 Accuracy", f"{top5_acc:.2f}%")
    table.add_row("Images tested",  str(len(all_labels)))
    console.print(table)

    return acc  # %

# ----------------------- batch driver --------------------------
def _batch_evaluate(model_path: str, test_dir: str, iterations: int):
    """
    Repeats _evaluate_once `iterations` times on fresh 50 % subsets.
    Saves per-run folders results/classification/result_ⓝ/ and
    aggregates accuracies into evaluation_results.csv.
    """
    # ---- validate folder structure once ----
    expected = [Path(test_dir) / str(i) for i in range(10)]
    if any(not p.is_dir() for p in expected):
        console.print("[red]Folder structure incorrect for CIFAR-10 test data[/red]")
        raise SystemExit(1)

    dataset = CIFAR10Folder(test_dir)
    build_dirs(RESULTS_ROOT)

    records = []
    console.print("[bold green]==== Starting batch evaluation ====[/bold green]")
    for i in range(1, iterations + 1):
        start = time.time()
        subset_size = max(1, int(len(dataset) * SUBSET_RATIO))
        subset_idx  = np.random.choice(len(dataset), subset_size, replace=False)

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
            "model_name":     Path(model_path).name,
            "test_data_name": Path(test_dir).name,
            "accuracy":       round(acc, 2),
            "time":           now
        })

    # ---- stability row ----
    accs      = [r["accuracy"] for r in records]
    mean_acc = sum(accs) / len(accs)
    tolerance = 2.0

    accurate   = sum(abs(a - mean_acc) <= tolerance for a in accs)
    overall    = accurate / len(accs) * 100 
    records.append({
        "number_of_eval": "summary",
        "model_name":     Path(model_path).name,
        "test_data_name": Path(test_dir).name,
        "accuracy":       round(overall, 2),
        "time":           datetime.now().strftime("%H:%M:%S")
    })

    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    console.print(f"\n[green]Saved aggregated CSV →[/green] {CSV_PATH}")
    console.print("[bold green]==== Evaluation complete! ====[/bold green]")

# ----------------------- public wrapper ------------------------
def run_classification(model_path: str, test_dir: str, loops: int | None = None):
    """
    Public entry-point used by cli.py.
    If `loops` is None, prompt the user for a number.
    """
    if loops is None:
        loops = int(console.input("[cyan]Enter number of evaluation loops ➜ [/cyan]"))
    _batch_evaluate(model_path, test_dir, iterations=loops)

# -------------------------- CLI (stand-alone) ------------------
if __name__ == "__main__":
    mp = console.input("Enter path to your ResNet-50 CIFAR-10 model (.pth): ").strip()
    td = console.input("Enter path to your CIFAR-10 test data folder: ").strip()
    loops = int(console.input("Enter number of evaluation loops: ").strip())
    run_classification(mp, td, loops)
# ───────────────────────────────────────────────────────────────
