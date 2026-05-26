# ─── evaluators/segmentation_eval.py ───────────────────────────
import os, time, random, cv2, torch, numpy as np, math
from glob import glob
from datetime import datetime
from pathlib import Path

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex
from sklearn.metrics import precision_recall_curve, f1_score, auc
from skimage.metrics import hausdorff_distance

from rich.console  import Console
from rich.table    import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich          import box

CROP_SIZE     = 256
MEAN, STD     = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FONT, FONT_SCALE, THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
COLOR_WHITE   = (255, 255, 255)

SUBSET_RATIO  = 0.50
RESULTS_ROOT  = Path(__file__).parent.parent / "results" / "segmentation"
CSV_PATH      = RESULTS_ROOT / "segmentation_evaluation_results.csv"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def compute_dice(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    return (2 * inter) / (gt.sum() + pred.sum() + 1e-7)

def load_model(weights: Path):
    model = smp.Unet("resnet34", encoder_weights=None,
                     in_channels=3, classes=1)
    state = torch.load(str(weights), map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

# ------------------- single-run evaluator ---------------------
def _evaluate_once(
    model_path: Path,
    img_paths: list[Path],
    mask_suffix: str,
    out_dir: Path,
    run_idx: int,
    total_runs: int,
) -> float:
    build_dirs(out_dir, out_dir / "vis_side_by_side", out_dir / "vis_overlays")

    HEATMAP_PATH = out_dir / "error_heatmap.png"
    HIST_PATH    = out_dir / "iou_histogram.png"
    PRC_PATH     = out_dir / "precision_recall_curve.png"
    METRICS_CSV  = out_dir / "per_image_metrics.csv"
    SUMMARY_TXT  = out_dir / "summary.txt"

    model   = load_model(model_path)
    jaccard = JaccardIndex(task="binary").to(DEVICE)
    val_tf  = A.Compose([
        A.CenterCrop(CROP_SIZE, CROP_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    records, ious, dices, hausds, bf1s = [], [], [], [], []
    all_probs, all_gts = [], []
    heatmap = None

    with _make_progress(f"Inferencing {run_idx}/{total_runs}") as prog:
        task = prog.add_task("", total=len(img_paths))
        t_start = time.time()

        for idx, p in enumerate(img_paths):
            img  = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            mask_path = p.with_name(p.stem.replace("_sat", mask_suffix))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


            h, w = img.shape[:2]
            top, left = (h-CROP_SIZE)//2, (w-CROP_SIZE)//2
            crop = img[top:top+CROP_SIZE, left:left+CROP_SIZE]
            gt_np = (mask[top:top+CROP_SIZE,
                          left:left+CROP_SIZE] > 127).astype(np.uint8)

            aug   = val_tf(image=crop, mask=gt_np)
            inp   = aug["image"].unsqueeze(0).to(DEVICE)
            gt_np = np.squeeze(gt_np)
            pred_gt_t = torch.from_numpy(gt_np).bool().unsqueeze(0).to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                logits = model(inp)
                probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()
            t1 = time.time()

            pred_np = (probs > 0.5).astype(np.uint8)
            pred_t  = torch.from_numpy(pred_np).bool().unsqueeze(0).to(DEVICE)

            # metrics
            if pred_np.sum() == 0 and gt_np.sum() == 0:
                iou_val, dice_val, haus_val, bf1_val = 1.0, 1.0, 0.0, 1.0
            else:
                iou_val  = jaccard(pred_t, pred_gt_t).item()
                dice_val = compute_dice(gt_np, pred_np)
                haus_val = hausdorff_distance(gt_np, pred_np)
                eg, ep   = cv2.Canny(gt_np*255,100,200)>0, cv2.Canny(pred_np*255,100,200)>0
                bf1_val  = f1_score(eg.flatten(), ep.flatten(), zero_division=0)

            records.append({
                "file": p.name,
                "iou": iou_val,
                "dice": dice_val,
                "hausdorff": haus_val,
                "boundary_f1": bf1_val,
                "time": t1 - t0,
            })
            ious.append(iou_val); dices.append(dice_val)
            hausds.append(haus_val); bf1s.append(bf1_val)
            all_probs.extend(probs.flatten().tolist())
            all_gts.extend(gt_np.flatten().tolist())

            if heatmap is None:
                heatmap = np.zeros_like(pred_np, dtype=np.uint32)
            heatmap += (pred_np != gt_np).astype(np.uint32)

            # visualisations
            orig_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            pr_bgr   = cv2.cvtColor(pred_np*255, cv2.COLOR_GRAY2BGR)
            gt_bgr   = cv2.cvtColor(gt_np*255,   cv2.COLOR_GRAY2BGR)
            vis = np.hstack([orig_bgr, pr_bgr, gt_bgr])
            for col, txt in enumerate(["Original", "Prediction", "Ground Truth"]):
                cv2.putText(vis, txt, (col*CROP_SIZE+10, 30),
                            FONT, FONT_SCALE, COLOR_WHITE, THICKNESS)
            cv2.imwrite(str(out_dir / "vis_side_by_side" / f"vis_{idx:03d}.png"), vis)

            overlay = orig_bgr.copy()
            overlay[pred_np == 1] = (0, 255, 0)
            blend = cv2.addWeighted(orig_bgr, 0.7, overlay, 0.3, 0)
            cv2.imwrite(str(out_dir / "vis_overlays" / f"overlay_{idx:03d}.png"), blend)

            prog.update(task, advance=1)

    total = time.time() - t_start
    fps   = len(img_paths) / total if total else 0.0

    df = pd.DataFrame(records)
    df["fps"] = fps
    df.to_csv(METRICS_CSV, index=False)

    # plots
    plt.figure(); plt.hist(ious, bins=20, edgecolor="k")
    plt.title("IoU Distribution"); plt.xlabel("IoU"); plt.ylabel("Count")
    plt.savefig(HIST_PATH); plt.close()

    prec, rec, _ = precision_recall_curve(all_gts, all_probs)
    pr_auc = auc(rec, prec)
    plt.figure(); plt.plot(rec, prec, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.legend(); plt.savefig(PRC_PATH); plt.close()

    hm = (heatmap.astype(float) / len(img_paths) * 255).astype(np.uint8)
    cv2.imwrite(HEATMAP_PATH, cv2.applyColorMap(hm, cv2.COLORMAP_JET))

    mean_iou, mean_dice = np.mean(ious), np.mean(dices)
    with open(SUMMARY_TXT, "w") as f:
        f.write(f"meanIoU:  {mean_iou:.4f}\n")
        f.write(f"meanDice: {mean_dice:.4f}\n")
        f.write(f"FPS:      {fps:.2f}\n")

    table = Table(title="Segmentation Model Evaluation Summary",
            box=box.MINIMAL)
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="magenta")
    table.add_row("meanIoU",  f"{mean_iou:.4f}")
    table.add_row("meanDice", f"{mean_dice:.4f}")
    table.add_row("FPS",      f"{fps:.2f}")
    console.print(table)


    return mean_iou  # %

# ----------------------- batch driver --------------------------
def _batch_evaluate(model_name: str, test_split: str, iterations: int):
    proj_root  = Path(__file__).parent.parent
    models_dir = proj_root / "models"
    data_dir   = proj_root / "data"

    model_path = models_dir / model_name
    if not model_path.is_file():
        console.print(f"[red]Error:[/red] model not found → {model_path}")
        raise SystemExit(1)

    test_dir   = data_dir / test_split
    if not test_dir.is_dir():
        console.print(f"[red]Error:[/red] test folder not found → {test_dir}")
        raise SystemExit(1)

    all_imgs = sorted(Path(test_dir).glob("*_sat.jpg"))
    if not all_imgs:
        console.print(f"[red]Error:[/red] no *_sat.jpg images in {test_dir}")
        raise SystemExit(1)

    mask_suffix = "_mask.png"
    build_dirs(RESULTS_ROOT)
    records = []

    console.print("[bold green]==== Starting batch evaluation ====[/bold green]")
    for i in range(1, iterations + 1):
        start = time.time()
        subset_len = max(1, int(len(all_imgs) * SUBSET_RATIO))
        subset_imgs = random.sample(all_imgs, subset_len)

        out_dir = RESULTS_ROOT / f"result_{i}"
        miou = _evaluate_once(model_path, subset_imgs, mask_suffix,
                              out_dir, run_idx=i, total_runs=iterations)

        now = datetime.now().strftime("%H:%M:%S")
        console.print(
            f"[bold yellow][ITER {i:03d}/{iterations}] "
            f"meanIoU: {miou:.4f}   Time: {now} "
            f"({time.time() - start:.1f}s)[/bold yellow]"
        )
        records.append({
            "number_of_eval": i,
            "model_name":     model_path.name,
            "test_data_name": test_dir.name,
            "meanIoU":        round(miou, 4),
            "time":           now
        })

    # stability row
    ious       = [r["meanIoU"] for r in records]   # list of meanIoU values (0-1 range)
    mean_iou   = sum(ious) / len(ious)             # reference value
    tolerance  = 0.02

    accurate   = sum(abs(i - mean_iou) <= tolerance for i in ious)
    overall    = accurate / len(ious) * 100        # % of runs that are “accurate”

    records.append({
    "number_of_eval": "summary",
    "model_name":     model_path.name,
    "test_data_name": test_dir.name,
    "meanIoU":        round(overall, 2),       # e.g. 98.33 means 98 % of runs pass
    "time":           datetime.now().strftime("%H:%M:%S")
})

    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    console.print(f"\n[green]Saved aggregated CSV →[/green] {CSV_PATH}")
    console.print("[bold green]==== Evaluation complete! ====[/bold green]")

# ----------------------- public wrapper ------------------------
def run_segmentation(model_name: str, test_split: str, loops: int | None = None):
    if loops is None:
        loops = int(console.input("[cyan]Enter number of evaluation loops ➜ [/cyan]"))
    _batch_evaluate(model_name, test_split, iterations=loops)

# -------------------------- CLI (stand-alone) ------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate segmentation model")
    ap.add_argument("--model", required=True,
                    help="file name inside ./models/")
    ap.add_argument("--split", required=True,
                    help="sub-folder inside ./data/")
    ap.add_argument("--loops", type=int, default=None)
    args = ap.parse_args()
    run_segmentation(args.model, args.split, loops=args.loops)
# ───────────────────────────────────────────────────────────────
