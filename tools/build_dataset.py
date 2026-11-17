"""Build the training dataset from the two image sources.

Sources
  raw_assets/<class>/<class>-<n>.jpg   photos shipped by the exam site itself (388)
  raw_assets/<class>/web/*.jpg         extra photos collected from Bing Images (360)

Output (classification dataset, ImageNet folder layout - what YOLO11s-cls eats)
  dataset/cls/train/<class>/*.jpg
  dataset/cls/val/<class>/*.jpg
  dataset/manifest.json                 per-class counts + split + where each file came from

Why classification and not detection: the exam gives you one piece per cell, so the
question is only "which class is this crop", never "where is the object".  The site
itself proves it - the reference grid shows a *different photo of the same class*,
so a detector would have to localise a subject that may be cropped into pieces.  A
classifier trained on the same crops answers the actual question and is far more
forgiving about framing.

Split: 80/20 per class, deterministic seed.  Site photos and web photos are shuffled
together first, so a class never has "all real photos in train, all web photos in
val" (that would flatter the validation number).
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

ROOT = Path(r"F:\exam_next")
RAW = ROOT / "raw_assets"
OUT = ROOT / "dataset" / "cls"
MANIFEST = ROOT / "dataset" / "manifest.json"

CLASSES = ["dog", "cat", "elephant", "desktop", "bicycle", "car", "chair", "cup",
           "airplane", "backpack", "umbrella", "suitcase",
           "penguin", "tafei", "nailong"]
CUSTOM = {"penguin", "tafei", "nailong"}
VAL_FRACTION = 0.2
SEED = 20261117


def collect(key: str) -> list[Path]:
    files = sorted((RAW / key).glob("*.jpg"))
    web = sorted((RAW / key / "web").glob("*.jpg"))
    return files + web


def main() -> None:
    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / split).mkdir(parents=True, exist_ok=True)

    manifest = {"seed": SEED, "val_fraction": VAL_FRACTION, "classes": {}}
    total_train = total_val = 0
    for key in CLASSES:
        files = collect(key)
        if not files:
            print(f"!! {key}: no images")
            continue
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * VAL_FRACTION)))
        # never let val eat the whole class for the tiny classes
        n_val = min(n_val, max(1, len(files) // 4))
        splits = {"val": files[:n_val], "train": files[n_val:]}
        for split, group in splits.items():
            dst = OUT / split / key
            dst.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(group):
                shutil.copy2(src, dst / f"{key}-{split}-{i:04d}.jpg")
        manifest["classes"][key] = {
            "total": len(files),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "from_site": len([p for p in files if p.parent.name == key]),
            "from_web": len([p for p in files if p.parent.name == "web"]),
            "needs_finetune": key in CUSTOM,
        }
        total_train += len(splits["train"])
        total_val += len(splits["val"])
        print(f"{key:10s} total {len(files):3d} -> train {len(splits['train']):3d} val {len(splits['val']):3d}")
    manifest["totals"] = {"train": total_train, "val": total_val}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOTAL train {total_train} val {total_val}")


if __name__ == "__main__":
    main()
