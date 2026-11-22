"""Offline: compare a real parked-piece silhouette with the engine template profile.

Phase 1 (`--grab`) lifts one piece, parks it over the flat dark board area, saves the raw
screenshot and puts the piece back -- a single drag, only to obtain a real silhouette.

Phase 2 (no `--grab`) analyses the saved screenshot: it prints, per side, how far the real
silhouette deviates from the straight body edge, next to the same number for the best
template.  That separates "my template is scaled wrong" from "the real piece is a different
curve", which is the question blocking shape recognition.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import piececutter as pc
import vision
from solve_live import Mouse, check_foreground, _enable_dpi

ROOT = vision.ROOT
SAMPLE = ROOT / "shots" / "park_raw.png"
PARK = (560, 620)
HALF = (115, 100)


def mask_from(shot: np.ndarray) -> np.ndarray:
    x, y = PARK
    hw, hh = HALF
    win = cv2.cvtColor(shot[y - hh:y + hh, x - hw:x + hw], cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = float(np.percentile(win, 20))
    m = (win > bg + 22).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def side_profile(mask: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """(min, median, max) of the silhouette extreme along each side, in px."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 500:
        return {}
    out = {}
    tops, bots, lefts, rights = [], [], [], []
    for x in range(int(xs.min()), int(xs.max()) + 1):
        col = np.nonzero(mask[:, x])[0]
        if col.size:
            tops.append(col.min())
            bots.append(col.max())
    for y in range(int(ys.min()), int(ys.max()) + 1):
        row = np.nonzero(mask[y, :])[0]
        if row.size:
            lefts.append(row.min())
            rights.append(row.max())
    for side, vals in (("top", tops), ("bottom", bots), ("left", lefts), ("right", rights)):
        v = np.array(vals, dtype=float)
        out[side] = (float(v.min()), float(np.median(v)), float(v.max()))
    return out


def report(tag: str, prof: dict[str, tuple[float, float, float]]) -> None:
    print(f"  {tag}")
    for side, (lo, med, hi) in prof.items():
        dev = max(med - lo, hi - med)
        print(f"    {side:6s} min={lo:6.1f} median={med:6.1f} max={hi:6.1f}  deviation={dev:5.1f}px")


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()

    if args.grab:
        if args.delay:
            print(f"waiting {args.delay:.0f}s ...", flush=True)
            time.sleep(args.delay)
        ok, why = check_foreground()
        print(f"foreground: {why}")
        if not ok:
            return 2
        img = vision.grab()
        x, y = PARK
        hw, hh = HALF
        g = cv2.cvtColor(vision.sub(img, (x - hw, y - hh, hw * 2, hh * 2)), cv2.COLOR_BGR2GRAY)
        if g.std() > 8:
            print(f"ABORT: parking area not empty (mean {g.mean():.0f} std {g.std():.0f})")
            return 3
        rows = vision.tray_rows(img)
        piece = None
        for band in ("left", "right"):
            for ri, ci, box in vision.tray_slots(img, band, rows):
                if not vision.is_empty(vision.sub(img, box)):
                    piece = (band, ri, ci, box)
                    break
            if piece:
                break
        if piece is None:
            print("ABORT: no piece found in the tray")
            return 3
        band, ri, ci, box = piece
        print(f"sampling piece {band} r{ri + 1}c{ci + 1} at {box}")
        mouse = Mouse()
        src = (box[0] + box[2] // 2, box[1] + box[3] // 2)
        mouse.grab_at(src)
        mouse.glide((src[0] + 16, src[1] + 10))
        time.sleep(mouse.settle)
        mouse.glide(PARK)
        time.sleep(mouse.settle * 2)
        # drop it first: a piece that is still held carries the blue "selected" glow
        # (shadowColor rgba(99,102,241,.75), blur 18), which is brighter than the canvas
        # and would be merged into the silhouette
        mouse.release()
        time.sleep(0.45)
        shot = mouse.shot()
        cv2.imwrite(str(SAMPLE), shot)
        print(f"saved {SAMPLE} (piece dropped, dark shadow only)")
        mouse.grab_at(PARK)                    # pick it up again and put it back
        mouse.glide((PARK[0] + 10, PARK[1] + 8))
        time.sleep(mouse.settle)
        mouse.glide(src)
        time.sleep(mouse.settle)
        mouse.release()
        time.sleep(0.2)

    shot = cv2.imread(str(SAMPLE))
    if shot is None:
        print(f"no sample at {SAMPLE}; run with --grab first")
        return 4
    mask = mask_from(shot)
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    print(f"\nmask area {area}px, bbox {xs.max() - xs.min() + 1}x{ys.max() - ys.min() + 1}")
    real = side_profile(mask)
    report("REAL parked piece:", real)
    edges, iou, top = pc.classify(mask)
    print(f"  best fit: {'/'.join(edges[s] for s in pc.SIDE_NAMES)} IoU={iou:.3f} "
          f"(runner-up {top[1][1]} {top[1][0]:.3f})")
    tpl = pc.piece_mask(tuple(pc.KEY_TO_VALUE[edges[s]] for s in pc.SIDE_NAMES))
    report("best TEMPLATE:", side_profile(tpl))
    print("\nIf the real deviations are clearly smaller than the template's, the on-screen "
          "piece is drawn at a different tab scale than the ported math assumes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
