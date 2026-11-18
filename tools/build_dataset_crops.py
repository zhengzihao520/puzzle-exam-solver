"""生成"拼图块状"数据集：让训练分布和运行时一致。

运行时模型看到的是「从照片里切出来的一小块（约 112×83，带切边）」，
而之前训练用的是整张照片，所以块状裁片上的准确率明显下降
（实测：飞机→奶龙、狗→猫、企鹅→desktop）。
本脚本把每张图库照片切成 5×5 的格子裁片（与运行时同比例），
再对每片做轻微抖动与亮度扰动，得到"块状"样本。

输出：
  dataset/crops/train/<class>/*.jpg
  dataset/crops/val/<class>/*.jpg
  dataset/crops_manifest.json
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"F:\exam_next")
RAW = ROOT / "raw_assets"
OUT = ROOT / "dataset" / "crops"
MANIFEST = ROOT / "dataset" / "crops_manifest.json"

CLASSES = ["dog", "cat", "elephant", "desktop", "bicycle", "car", "chair", "cup",
           "airplane", "backpack", "umbrella", "suitcase", "penguin", "tafei", "nailong"]
GRID = 5                 # 与 5×5 难度的格子数一致
JITTER = 0.06            # 切窗抖动（相对格子尺寸）
VAL_FRACTION = 0.15
SEED = 20261117


def crops_of(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    ch, cw = h / GRID, w / GRID
    out = []
    for r in range(GRID):
        for c in range(GRID):
            jx = (random.random() - 0.5) * JITTER * cw
            jy = (random.random() - 0.5) * JITTER * ch
            x0 = int(max(0, c * cw + jx))
            y0 = int(max(0, r * ch + jy))
            x1 = int(min(w, x0 + cw))
            y1 = int(min(h, y0 + ch))
            if x1 - x0 < 24 or y1 - y0 < 18:
                continue
            crop = img[y0:y1, x0:x1]
            crop = cv2.resize(crop, (112, 83), interpolation=cv2.INTER_AREA)
            if random.random() < 0.5:
                f = 1.0 + (random.random() - 0.5) * 0.35
                crop = np.clip(crop.astype(np.float32) * f, 0, 255).astype(np.uint8)
            out.append(crop)
    return out


def main() -> None:
    random.seed(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        for key in CLASSES:
            (OUT / split / key).mkdir(parents=True, exist_ok=True)

    manifest = {"seed": SEED, "grid": GRID, "classes": {}}
    tot_tr = tot_va = 0
    for key in CLASSES:
        files = sorted((RAW / key).glob("*.jpg")) + sorted((RAW / key / "web").glob("*.jpg"))
        if not files:
            print(f"!! {key} 没有原图")
            continue
        random.shuffle(files)
        n_val = max(1, int(round(len(files) * VAL_FRACTION)))
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None:
                continue
            split = "val" if i < n_val else "train"
            for j, crop in enumerate(crops_of(img)):
                p = OUT / split / key / f"{key}-{i:03d}-{j:02d}.jpg"
                cv2.imwrite(str(p), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        n_tr = len(list((OUT / "train" / key).glob("*.jpg")))
        n_va = len(list((OUT / "val" / key).glob("*.jpg")))
        tot_tr += n_tr
        tot_va += n_va
        manifest["classes"][key] = {"photos": len(files), "train": n_tr, "val": n_va}
        print(f"{key:10s} 原图 {len(files):3d} -> 块裁片 train {n_tr:5d} val {n_va:4d}", flush=True)

    manifest["totals"] = {"train": tot_tr, "val": tot_va}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOTAL train {tot_tr} val {tot_va}")


if __name__ == "__main__":
    main()
