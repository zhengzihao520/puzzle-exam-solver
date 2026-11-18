"""用 30 块真实拼图块 + 合成块裁片混合微调分类器。

为什么需要这一步：合成裁片是"干净矩形局部"，而运行时块带凸台、黑底、相邻块边缘，
分布不同导致实测大量误判（飞机→奶龙、狗→猫、企鹅→desktop、甚至类别塌缩到 tafei）。
这次把用户逐块确认的 30 个真实块以高倍重复采样注入训练集，让模型见到真实分布。

真实块来源：用户提供的两条托盘带截图（band_left.png / band_right.png），
按用户给的行列布局与固定间距切出，标签由用户逐块确认。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"F:\exam_next")
CROPS = ROOT / "dataset" / "crops"
REAL = ROOT / "real_pieces"
SHOTS = ROOT / "shots"

PW, PH = 112, 85
SX, SY = 129.0, 97.5
LEFT_MISS = {(5, 2)}
RIGHT_MISS = {(3, 2), (3, 3), (4, 1), (5, 3), (6, 2)}

# 用户逐块确认的标签（#0..#29 顺序）
LABELS = [
    "penguin", "cat", "penguin", "dog", "dog", "dog", "airplane", "cup", "bicycle",
    "suitcase", "chair", "dog", "cup", "cat", "dog", "backpack", "airplane",
    "nailong", "tafei", "suitcase", "dog", "backpack", "umbrella", "dog",
    "penguin", "bicycle", "chair", "car", "dog", "dog",
]
REPEAT = 60          # 真实样本在训练集里的重复次数（对冲合成数据的分布差异）


def cut_all() -> list[tuple[int, str, np.ndarray]]:
    out = []
    idx = 0
    for name, miss, x0, y0 in (("left", LEFT_MISS, 66, 58), ("right", RIGHT_MISS, 79, 58)):
        im = cv2.imread(str(SHOTS / f"band_{name}.png"))
        H, W = im.shape[:2]
        for r in range(1, 7):
            for c in range(1, 4):
                if (r, c) in miss:
                    continue
                cx = int(x0 + (c - 1) * SX)
                cy = int(y0 + (r - 1) * SY)
                if cx - PW // 2 < 0 or cy - PH // 2 < 0 or cx + PW // 2 > W or cy + PH // 2 > H:
                    continue
                crop = im[cy - PH // 2:cy - PH // 2 + PH, cx - PW // 2:cx - PW // 2 + PW]
                if crop.shape[:2] != (PH, PW):
                    continue
                if float((crop.mean(axis=2) < 35).mean()) > 0.55:
                    continue
                out.append((idx, LABELS[idx], crop))
                idx += 1
    return out


def main() -> None:
    pieces = cut_all()
    assert len(pieces) == len(LABELS), f"切出 {len(pieces)} 块，标签 {len(LABELS)} 个"
    if REAL.exists():
        shutil.rmtree(REAL)
    for i, lab, crop in pieces:
        d = REAL / lab
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{lab}-real-{i:02d}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    (ROOT / "real_pieces_labels.json").write_text(
        json.dumps({i: lab for i, lab, _ in pieces}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 注入训练集（高倍重复）+ 验证集（每类各 1 张，只做冒烟检查，最终以实机验证为准）
    added = 0
    for i, lab, crop in pieces:
        tr = CROPS / "train" / lab
        tr.mkdir(parents=True, exist_ok=True)
        for k in range(REPEAT):
            cv2.imwrite(str(tr / f"{lab}-real{i:02d}-r{k:02d}.jpg"), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            added += 1
    print(f"真实块 {len(pieces)} 个，注入训练集副本 {added} 张", flush=True)
    from collections import Counter
    print("真实块类别分布:", Counter([l for _, l, _ in pieces]).most_common(), flush=True)

    # 微调
    import time
    from ultralytics import YOLO
    base = ROOT / "runs" / "cls_crops12" / "weights" / "best.pt"
    m = YOLO(str(base if base.exists() else ROOT / "yolo11s-cls.pt"))
    t0 = time.time()
    m.train(data=str(CROPS), epochs=10, imgsz=96, batch=32, device="cpu",
            project=str(ROOT / "runs"), name="cls_real", exist_ok=True, seed=20261117,
            workers=4, patience=5, lr0=0.0006, cos_lr=True,
            fliplr=0.3, flipud=0.0, hsv_h=0.01, hsv_s=0.3, hsv_v=0.25,
            scale=0.15, translate=0.04, erasing=0.05, verbose=True)
    r = m.val(data=str(CROPS), imgsz=96, batch=32, device="cpu",
              project=str(ROOT / "runs"), name="cls_real_val", exist_ok=True)
    info = {"run": "cls_real", "real_pieces": len(pieces), "repeat": REPEAT,
            "seconds": round(time.time() - t0, 1),
            "val": {"top1": float(r.top1), "top5": float(r.top5)}}
    (ROOT / "runs" / "cls_real_summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", info, flush=True)


if __name__ == "__main__":
    main()
