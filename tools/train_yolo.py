"""Fine-tune YOLO11s on the 15-class puzzle catalog.

Why yolo11s-cls: the solver needs "which class is this crop", so a classifier head
is the right tool; the site's reference grid shows a different photo of the same
class, which rules out pixel matching entirely.

Two runs are kept so the report can compare them:
  1. yolo11s-cls.pt warm start  (pretrained ImageNet weights)  -> runs/cls
  2. from scratch (no pretrained)                             -> runs/cls_scratch
The pretrained run is what runs/cls holds and what the solver loads; the scratch run documents why
pretraining matters for the three classes that are not in ImageNet-1k either.

CPU-friendly settings: imgsz 224, batch 16, cosine LR, mild augmentation
(the site crops pieces out of these same photos, so heavy aug hurts).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(r"F:\exam_next")
DATA = ROOT / "dataset" / "cls"
RUNS = ROOT / "runs"


def run(name: str, pretrained: bool, epochs: int, imgsz: int, batch: int) -> dict:
    from ultralytics import YOLO

    if pretrained:
        model = YOLO("yolo11s-cls.pt")          # downloads official weights once
    else:
        model = YOLO("yolo11s-cls.yaml")        # random init, same architecture
    t0 = time.time()
    model.train(
        data=str(DATA), epochs=epochs, imgsz=imgsz, batch=batch, device="cpu",
        project=str(RUNS), name=name, exist_ok=True, seed=20261117,
        workers=2, patience=12, cos_lr=True,
        fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
        scale=0.35, translate=0.08, erasing=0.15,
        verbose=True,
    )
    metrics = {}
    try:
        m = model.val(data=str(DATA), imgsz=imgsz, batch=batch, device="cpu", project=str(RUNS),
                      name=f"{name}_val", exist_ok=True)
        metrics = {"top1": float(getattr(m, "top1", 0.0)),
                   "top5": float(getattr(m, "top5", 0.0))}
    except Exception as exc:                    # noqa: BLE001
        metrics = {"error": str(exc)}
    return {"run": name, "pretrained": pretrained, "epochs": epochs,
            "seconds": round(time.time() - t0, 1), "val": metrics,
            "weights": str(RUNS / name / "weights" / "best.pt")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--only", default="", help="pretrained | scratch")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    report = []
    if args.only in ("", "pretrained"):
        report.append(run("cls", True, args.epochs, args.imgsz, args.batch))
        print("PRETRAINED DONE", report[-1], flush=True)
    if args.only in ("", "scratch"):
        report.append(run("cls_scratch", False, args.epochs, args.imgsz, args.batch))
        print("SCRATCH DONE", report[-1], flush=True)
    (RUNS / "train_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL DONE", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
