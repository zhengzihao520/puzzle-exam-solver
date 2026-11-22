"""Find the real snap offset on one cell, with one piece, by sweeping a line.

The user reports two symptoms: pieces land slightly right of the cell (the left column never
touches the frame) and the placed photo never matches the reference.  Both can be tested in
one shot: drag a single piece across a row of positions and see whether the engine ever
locks it.  A lock proves the piece fits that cell and gives the offset; no lock at any
position proves the piece is the wrong one.

  python snap_probe.py --delay 10 --row 0 --col 0
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, check_foreground, _enable_dpi

ROOT = vision.ROOT


def cell_mean(img, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--pieces", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--from", dest="dfrom", type=int, default=-20)
    ap.add_argument("--to", dest="dto", type=int, default=9)
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    img = vision.grab()
    ref, board, _ = vision.layout(img)
    n = args.grid
    _r, _c, cell = board.cells(n)[args.row * n + args.col]
    cx, cy = cell[0] + cell[2] // 2, cell[1] + cell[3] // 2
    print(f"cell({args.row + 1},{args.col + 1}) box={cell} centre=({cx},{cy})")
    if cell_mean(img, cell) >= vision.BOARD_FILLED_LEVEL:
        print("ABORT: that cell already holds something")
        return 3

    rows = vision.tray_rows(img)
    tray = [(f"{b[0]}{ri + 1}{ci + 1}", box) for b in ("left", "right")
            for ri, ci, box in vision.tray_slots(img, b, rows)
            if not vision.is_empty(vision.sub(img, box))]
    print(f"tray pieces: {len(tray)}")

    mouse = Mouse()
    for key, box in tray[:args.pieces]:
        src = (box[0] + box[2] // 2, box[1] + box[3] // 2)
        print(f"\n--- piece {key} at {src}")
        mouse.grab_at(src)
        mouse.glide((src[0] + 16, src[1] + 10))
        time.sleep(mouse.settle)
        if cell_mean(mouse.shot(), box) > 0.95 * cell_mean(vision.grab(), box):
            pass
        hit = None
        for dx in range(args.dfrom, args.dto, args.step):
            mouse.glide((cx + dx, cy))
            time.sleep(mouse.settle * 1.2)
            # did it stop following?  nudge and look at the cell
            mouse.glide((cx + dx + 62, cy + 46))
            time.sleep(mouse.settle * 1.2)
            if cell_mean(mouse.shot(), cell) >= vision.BOARD_FILLED_LEVEL * 0.9:
                hit = dx
                break
            mouse.glide((cx + dx, cy))
            time.sleep(mouse.settle * 0.8)
        if hit is not None:
            mouse.glide((cx + hit, cy))
            time.sleep(mouse.settle)
            mouse.release()
            print(f"   SNAPPED at dx={hit:+d} (cell centre was {cx}, true spot x={cx + hit})")
            after = vision.grab()
            cv2.imwrite(str(ROOT / "shots" / "snap_probe_hit.png"), after)
            return 0
        # nothing snapped: put it back and try the next piece
        mouse.glide(src)
        time.sleep(mouse.settle)
        mouse.release()
        time.sleep(0.2)
        print("   no snap anywhere on this line -> wrong piece for this cell")
    print("\nno piece of the first few snapped: either all wrong, or the y offset is off too")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
