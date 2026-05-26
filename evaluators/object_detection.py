# ─── evaluators/object_detection.py ────────────────────────────
import os, random, time, math
from datetime import datetime
from pathlib import Path
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from rich.console  import Console
from rich.table    import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich          import box
import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings("ignore",
                        message="A single label was found.*",
                        category=UserWarning)

# ─── CONSTANTS ─────────────────────────────────────────────────
SUBSET_RATIO   = 0.50
RESULTS_ROOT   = Path(__file__).parent.parent / "results" / "detection"
CSV_PATH       = RESULTS_ROOT / "detection_evaluation_results.csv"
DEVICE_INDEX   = 0 if torch.cuda.is_available() else "cpu"          # change if you want a different GPU id
IMGSZ          = 640
BATCH_VAL      = 4
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
    fname   = name if name.endswith(ext) else name + ext
    matches = list(root.rglob(fname))
    if not matches:
        console.print(f"[red]Error:[/red] Could not find '{fname}' under {root}")
        raise SystemExit(1)
    if len(matches) > 1:
        console.print(f"[yellow]Warning:[/yellow] Multiple '{fname}' found – using {matches[0]}")
    return matches[0]

# ----------------------- IoU helper ---------------------------
def box_iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter    = max(xi2-xi1, 0) * max(yi2-yi1, 0)
    area1    = max(b1[2]-b1[0], 0) * max(b1[3]-b1[1], 0)
    area2    = max(b2[2]-b2[0], 0) * max(b2[3]-b2[1], 0)
    union    = area1 + area2 - inter
    return inter / union if union else 0

# ------------------- single-run evaluator ---------------------
def _evaluate_once(
    model_path: Path,
    data_spec: dict,
    img_paths: list[Path],
    out_dir: Path,
    run_idx: int,
    total_runs: int,
) -> float:
    """
    Evaluates on img_paths subset and saves artefacts under out_dir.
    Returns mAP50 (if YAML provided) else 0.0.
    """
    build_dirs(out_dir, out_dir / "examples")

    model = YOLO(str(model_path))

    test_labels_dir = data_spec["labels_dir"]

    # ---------------- per-image evaluation --------------------
    rows, iou_per_img = [], []
    y_true_img, y_pred_img = [], []
    confs, pred_counts     = [], []

    with _make_progress(f"Inferencing {run_idx}/{total_runs}") as prog:
        task = prog.add_task("", total=len(img_paths))
        for img_path in img_paths:
            img = cv2.imread(str(img_path)); h, w = img.shape[:2]

            # Ground-truth boxes
            gt_boxes = []
            gt_file  = test_labels_dir / f"{img_path.stem}.txt"
            if gt_file.exists():
                for ln in open(gt_file):
                    _, xc, yc, bw, bh = map(float, ln.split())
                    xc, yc = xc*w, yc*h
                    bw, bh = bw*w, bh*h
                    x1, y1 = int(xc-bw/2), int(yc-bh/2)
                    x2, y2 = int(xc+bw/2), int(yc+bh/2)
                    gt_boxes.append([x1, y1, x2, y2])

            # Predictions
            preds = []
            res = model.predict(str(img_path), imgsz=IMGSZ,
                                conf=0.001, save=False, verbose=False)[0]
            if res.boxes is not None:
                for b in res.boxes:
                    coords = [int(x) for x in b.xyxy[0].tolist()]
                    conf   = float(b.conf[0])
                    preds.append(coords)
                    confs.append(conf)
                    rows.append({
                        "image":      img_path.name,
                        "class":      int(b.cls[0]),
                        "confidence": conf,
                        "xmin":       coords[0], "ymin": coords[1],
                        "xmax":       coords[2], "ymax": coords[3],
                    })
                pred_counts.append(len(preds))
            else:
                pred_counts.append(0)

            y_true_img.append(int(bool(gt_boxes)))
            y_pred_img.append(int(bool(preds)))

            # IoU computation
            if preds and gt_boxes:
                ious = [max(box_iou(p, g) for g in gt_boxes) for p in preds]
            else:
                ious = [0.0] * len(preds) if preds else []
            iou_per_img.append({
                "image": img_path.name,
                "mean_iou": np.mean(ious) if ious else 0.0
            })
            prog.update(task, advance=1)

        # ---------------- per-image evaluation done -----------------
    # ---------- YOLO metrics on THE SAME subset -----------
    # ---- create <out_dir>/subset.yaml + subset.txt ----------------
    subset_txt  = out_dir / "subset.txt"
    subset_yaml = out_dir / "subset.yaml"

# write list-of-image paths (one per line) for the subset
    subset_txt  = out_dir / "subset.txt"
    subset_txt.write_text("\n".join(str(p) for p in img_paths))   # 50 % image list

    yaml_dict = {
    "path": ".",                 # root (unused because paths are absolute)
    "train": str(subset_txt),    # ← DUMMY entries required by Ultralytics
    "val":   str(subset_txt),    # ← DUMMY entries required by Ultralytics
    "test":  str(subset_txt),    # our actual subset
    "nc": 1,
    "names": ["License_Plate"]   # your single class name
}
    subset_yaml = out_dir / "subset.yaml"
    subset_yaml.write_text(yaml.safe_dump(yaml_dict))

# ---- run val() on THIS subset --------------------------------
    val_res     = model.val(
    data=str(subset_yaml),       # <- give the file path, not a dict
    imgsz=IMGSZ,
    device=DEVICE_INDEX,
    batch=BATCH_VAL,
    verbose=False,
    save=False,
    save_txt=False,
    save_json=False,
    save_conf=False,
    plots=False,
    project=str(RESULTS_ROOT),
    name=".",
    exist_ok=True
    )
    val_metrics = val_res.results_dict
    map50       = val_metrics.get("metrics/mAP50(B)", 0.0)


    mean_iou = np.mean([d["mean_iou"] for d in iou_per_img])

    # include all YOLO validation metrics when available
    summary = {
    "precision":  val_metrics.get("metrics/precision(B)", 0),
    "recall":     val_metrics.get("metrics/recall(B)",    0),
    "mAP50":      map50,
    "mAP50-95":   val_metrics.get("metrics/mAP50-95(B)", 0),
    "meanIoU":    mean_iou,
}
    # ---------------- save artefacts --------------------------
    pd.DataFrame(iou_per_img).to_csv(out_dir / "per_image_iou.csv", index=False)
    plt.figure(); plt.hist([d["mean_iou"] for d in iou_per_img], bins=20)
    plt.title("IoU per image"); plt.tight_layout()
    plt.savefig(out_dir / "iou_histogram.png"); plt.close()

    plt.figure(); plt.hist(confs, bins=20)
    plt.title("Confidence histogram"); plt.tight_layout()
    plt.savefig(out_dir / "histogram_confidences.png"); plt.close()

    pd.DataFrame({
        "image": [p.name for p in img_paths],
        "num_predictions": pred_counts
    }).to_csv(out_dir / "per_image_pred_count.csv", index=False)

    cm = confusion_matrix(y_true_img, y_pred_img, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["No GT", "Has GT"]) \
        .plot(cmap="Blues")
    plt.title("Image-level Confusion Matrix")
    plt.savefig(out_dir / "confusion_matrix.png"); plt.close()

    pd.DataFrame(rows).to_csv(out_dir / "all_predictions.csv", index=False)

    with open(out_dir / "summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v:.4f}\n")

    # 10 examples (side-by-side) — unchanged
    for img_path in random.sample(img_paths, min(10, len(img_paths))):
        img    = cv2.imread(str(img_path)); h, w = img.shape[:2]
        gt_img = img.copy(); pr_img = img.copy()

        # draw GT
        gt_file = test_labels_dir / f"{img_path.stem}.txt"
        if gt_file.exists():
            for ln in open(gt_file):
                _, xc, yc, bw, bh = map(float, ln.split())
                xc, yc = xc*w, yc*h; bw, bh = bw*w, bh*h
                x1,y1 = int(xc-bw/2), int(yc-bh/2)
                x2,y2 = int(xc+bw/2), int(yc+bh/2)
                cv2.rectangle(gt_img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(gt_img, "GROUND TRUTH", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        pr = model.predict(str(img_path), imgsz=IMGSZ, conf=0.001,
                           save=False, verbose=False)[0]
        for b in pr.boxes:
            c = [int(x) for x in b.xyxy[0].tolist()]
            cv2.rectangle(pr_img, (c[0],c[1]), (c[2],c[3]), (0,0,255), 2)
        cv2.putText(pr_img, "PREDICTION", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imwrite(str((out_dir/"examples") / f"{img_path.stem}_example.jpg"),
                    np.concatenate([gt_img, pr_img], axis=1))

    # Rich summary
    table = Table(title="Object Detection Model Evaluation Summary",
                  box=box.MINIMAL)
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="magenta")
    for k, v in summary.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    return summary["mAP50"]  # may be 0.0 when YAML absent

# --------------------- data-spec builder -----------------------
def _prepare_data(data_input: str) -> dict:
    """
    Returns a dict with:
        yaml_path (Path | None), images_dir (Path), labels_dir (Path)
    """
    proj_root = Path(__file__).parent.parent
    data_dir  = proj_root / "data"
    data_path = Path(data_input)

    if data_path.suffix in (".yaml", ".yml"):
        if not data_path.is_file():
            data_path = find_file(data_input, ".yaml", proj_root)
        cfg  = yaml.safe_load(open(data_path, "r"))
        base = data_path.parent
        images_dir = (base / cfg["test"]).resolve()
        labels_dir = (base / "labels").resolve()
        return {"yaml_path": data_path, "images_dir": images_dir,
                "labels_dir": labels_dir}
    else:
        base = (data_dir / data_input).resolve()
        return {"yaml_path": None,
                "images_dir": base / "images",
                "labels_dir": base / "labels"}

# ---------------------- batch driver --------------------------
def _batch_evaluate(model_name: str, data_input: str, iterations: int):
    proj_root  = Path(__file__).parent.parent
    model_path = find_file(model_name, ".pt", proj_root / "models")
    data_spec  = _prepare_data(data_input)

    if not data_spec["images_dir"].is_dir():
        console.print(f"[red]Error:[/red] missing folder {data_spec['images_dir']}")
        raise SystemExit(1)

    img_paths_all = sorted(data_spec["images_dir"].glob("*.jpg"))
    if not img_paths_all:
        console.print(f"[red]Error:[/red] no .jpg images in {data_spec['images_dir']}")
        raise SystemExit(1)

    build_dirs(RESULTS_ROOT)
    records = []

    console.print(f"[green]Model:[/green] {model_path}")
    console.print(f"[green]Images:[/green] {data_spec['images_dir']}\n")
    console.print("[bold green]==== Starting batch evaluation ====[/bold green]")

    for i in range(1, iterations + 1):
        start = time.time()

        subset_len = max(1, int(len(img_paths_all) * SUBSET_RATIO))
        subset_idx = np.random.choice(len(img_paths_all), subset_len,
                                      replace=False)
        subset_paths = [img_paths_all[j] for j in subset_idx]

        out_dir = RESULTS_ROOT / f"result_{i}"
        mAP50 = _evaluate_once(model_path, data_spec,
                               subset_paths, out_dir,
                               run_idx=i, total_runs=iterations)

        now = datetime.now().strftime("%H:%M:%S")
        console.print(
            f"[bold yellow][ITER {i:03d}/{iterations}] "
            f"mAP50: {mAP50:.4f}   Time: {now} "
            f"({time.time() - start:.1f}s)[/bold yellow]"
        )
        records.append({
            "number_of_eval": i,
            "model_name":     model_path.name,
            "test_data_name": Path(data_input).name,
            "mAP50":          round(mAP50, 4),
            "time":           now
        })

    # stability row
    maps       = [r["mAP50"] for r in records]      # list of floats 0-1
    mean_map   = sum(maps) / len(maps)              # reference value
    tolerance  = 0.02                               # ±2 % band

    accurate   = sum(abs(m - mean_map) <= tolerance for m in maps)
    overall    = accurate / len(maps) * 100         # reproducibility accuracy %

    records.append({
    "number_of_eval": "summary",
    "model_name":     model_path.name,
    "test_data_name": Path(data_input).name,
    "mAP50":          round(overall, 2),        # e.g. 98.33
    "time":           datetime.now().strftime("%H:%M:%S")
})

    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    console.print(f"\n[green]Saved aggregated CSV →[/green] {CSV_PATH}")
    console.print("[bold green]==== Evaluation complete! ====[/bold green]")

# ---------------------- public wrapper ------------------------
def run_detection(model_name: str, data_input: str, loops: int | None = None):
    if loops is None:
        loops = int(console.input("[cyan]Enter number of evaluation loops ➜ [/cyan]"))
    _batch_evaluate(model_name, data_input, iterations=loops)

# -------------------------- CLI (stand-alone) -----------------
if __name__ == "__main__":
    m = console.input("Model .pt filename under 'models': ")
    d = console.input("YAML path OR folder name under 'data': ")
    loops = int(console.input("Number of evaluation loops: ").strip())
    run_detection(m, d, loops)
# ───────────────────────────────────────────────────────────────
