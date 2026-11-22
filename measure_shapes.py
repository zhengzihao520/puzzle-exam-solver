"""Read every tray piece's shape by the engine's own templates.

Why parking under the board
---------------------------
Pieces sit 23 px apart with 18 px tabs, so in the tray a neighbour's tab lands inside the
measuring window and no clean silhouette can be cut.  A piece held over a flat dark area
has nothing else in view, so its mask is unambiguous.

The mask is compared against all 81 edge combinations rendered by `piececutter` (the site's
own Bezier geometry), which needs no knowledge of the grab offset: `classify` normalises by
bounding box and scale, so only the outline matters.

Acceptance for a 5x5: exactly 4 pieces with two flats, 12 with one, 9 with none.

  python measure_shapes.py --delay 12        measure every piece on the tray
  python measure_shapes.py --dry             show the plan only
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

import piececutter as pc
import vision
from solve_live import Mouse, check_foreground, _enable_dpi

ROOT = vision.ROOT
PARK = (560, 620)          # flat dark area inside the empty board (verified at run time)
PARK_HALF = (110, 95)
OUT = ROOT / "logs" / "shapes.json"


def park_mask(shot: np.ndarray) -> np.ndarray:
    x, y = PARK
    hw, hh = PARK_HALF
    win = cv2.cvtColor(shot[y - hh:y + hh, x - hw:x + hw], cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = float(np.percentile(win, 20))
    mask = (win > bg + 22).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def park_is_clean(img: np.ndarray) -> tuple[bool, str]:
    x, y = PARK
    hw, hh = PARK_HALF
    g = cv2.cvtColor(vision.sub(img, (x - hw, y - hh, hw * 2, hh * 2)), cv2.COLOR_BGR2GRAY)
    return float(g.std()) < 8, f"park area mean {g.mean():.0f} std {g.std():.0f}"


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.delay:
        print(f"waiting {args.delay:.0f}s for you to focus the exam page ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok and not args.dry:
        return 2

    img = vision.grab()
    clean, info = park_is_clean(img)
    print(f"{info} -> {'clean' if clean else 'NOT CLEAN'}")
    rows = vision.tray_rows(img)
    slots = [(f"{band[0]}{ri + 1}{ci + 1}", box)
             for band in ("left", "right")
             for ri, ci, box in vision.tray_slots(img, band, rows)]
    todo = [(k, b) for k, b in slots if not vision.is_empty(vision.sub(img, b))]
    print(f"tray slots {len(slots)}, occupied {len(todo)}")
    if args.dry:
        print("first few:", [k for k, _ in todo[:8]])
        return 0
    if not clean:
        print("ABORT: the parking area is not an empty dark region; the board must be empty")
        return 3
    if len(todo) < 8:
        print(f"ABORT: only {len(todo)} pieces on the tray; reset the game")
        return 3
    # infer the grid from the piece count: N*N cells + round(N*N*0.2) decoys
    grid = next((n for n in (3, 4, 5, 6) if n * n + round(n * n * 0.2) == len(todo)), None)
    print(f"{len(todo)} pieces -> grid {grid}x{grid}" if grid else
          f"{len(todo)} pieces -> grid unknown, will just report the histogram")

    mouse = Mouse()
    out = {}
    hist = {0: 0, 1: 0, 2: 0}
    for key, box in todo:
        src = (box[0] + box[2] // 2, box[1] + box[3] // 2)
        before = float(cv2.cvtColor(vision.sub(mouse.shot(), box), cv2.COLOR_BGR2GRAY).mean())
        mouse.grab_at(src)
        mouse.glide((src[0] + 16, src[1] + 10))
        time.sleep(mouse.settle)
        lifted = float(cv2.cvtColor(vision.sub(mouse.shot(), box), cv2.COLOR_BGR2GRAY).mean())
        if abs(lifted - before) < 12:
            mouse.release()
            print(f"  {key}: could not pick up")
            continue
        mouse.glide(PARK)
        time.sleep(mouse.settle * 2)
        shot = mouse.shot()
        mask = park_mask(shot)
        area = int(mask.sum())
        edges, iou, top = pc.classify(mask)
        flats = pc.flats_of(edges)
        mouse.glide(src)                       # put it back before anything else
        time.sleep(mouse.settle)
        mouse.release()
        time.sleep(0.15)
        out[key] = {"box": list(box), "edges": edges, "flats": flats,
                    "iou": round(iou, 3), "margin": round(top[0][0] - top[1][0], 3),
                    "runner_up": top[1][1]}
        hist[min(len(flats), 2)] = hist.get(min(len(flats), 2), 0) + 1
        if len(out) == 0:
            import cv2 as _cv2
            _cv2.imwrite(str(ROOT / "shots" / "mask_sample.png"), mask * 255)
        print(f"  {key}: {'/'.join(edges[s] for s in pc.SIDE_NAMES):30s} flats={len(flats)} "
              f"IoU={iou:.3f} margin={top[0][0] - top[1][0]:.3f} maskarea={area}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{len(out)} pieces measured -> {OUT}")
    if grid:
        want = {2: 4, 1: 4 * (grid - 2), 0: (grid - 2) ** 2}
        print(f"flat histogram: {dict(sorted(hist.items()))}   expected for {grid}x{grid}: {want}")
        ok = all(hist.get(k, 0) == v for k, v in want.items())
        print("VALIDATION:", "PASS" if ok else "FAIL")
    else:
        print(f"flat histogram: {dict(sorted(hist.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
