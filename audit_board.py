"""Audit the board: for every cell that shows a photo, is it locked in or just lying there?

The single-point lock probe used in place.py can be fooled: if the press lands on nothing
(the piece sits a few pixels away, or a neighbour covers that exact spot) the picture does
not change either, and "nothing moved" is then read as "locked".  This audits with five
press points -- the cell centre and its four quarter points, all inside the piece box -- so
a false "locked" needs five misses in a row.

A cell that turns out to be loose is carried straight to a free tray slot: the board must
never hold a loose piece, otherwise a later press can lift the wrong one.

  python audit_board.py [--grid 5] [--delay 3]
"""
from __future__ import annotations

import argparse
import time

import cv2

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"
PROBES = [(0, 0), (-30, -20), (30, 20), (-30, 20), (30, -20)]


def mean_at(img, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    mouse = Mouse()
    img = vision.grab()
    cv2.imwrite(str(SHOTS / "audit_start.png"), img)
    ref, board, _ = vision.layout(img)
    n = args.grid
    boxes = {f"{r},{c}": b for r, c, b in board.cells(n)}
    print(f"board {board.x},{board.y} {board.w}x{board.h}  cell {board.w / n:.0f}x{board.h / n:.0f}")

    slots = [b for band in ("left", "right")
             for _r, _c, b in vision.lattice_slots(img, band)]

    def free_slots():
        shot = mouse.shot()
        return [b for b in slots if vision.is_empty(vision.sub(shot, b))]

    bright = {k: mean_at(img, b) for k, b in boxes.items() if mean_at(img, b) >= 45}
    print(f"cells showing a photo: {len(bright)}")
    for k, v in sorted(bright.items()):
        print(f"  ({k}) mean {v:.0f}")

    locked, loose, moves = [], [], 0
    for key in sorted(bright, key=lambda k: (int(k.split(",")[0]), int(k.split(",")[1]))):
        box = boxes[key]
        cx, cy = centre(box)
        verdict = "locked"
        for dx, dy in PROBES:
            pt = (cx + dx, cy + dy)
            before = mean_at(mouse.shot(), box)
            mouse.grab_at(pt)
            mouse.glide((pt[0] + 62, pt[1] + 38))
            time.sleep(0.2)
            after = mean_at(mouse.shot(), box)
            if after < before - 25 or after < 35:
                verdict = "loose"
                break
            mouse.release()          # nothing was selected: no move is counted
        if verdict == "locked":
            locked.append(key)
            print(f"  ({key})  locked")
            continue
        free = free_slots()
        if free:
            mouse.glide(centre(free[0]))
            mouse.release()
            moves += 1
            print(f"  ({key})  LOOSE -> moved to tray slot ({free[0][0]},{free[0][1]})")
        else:
            mouse.release()
            print(f"  ({key})  LOOSE but no free slot")
        loose.append(key)

    print(f"\nlocked {len(locked)}: {', '.join(sorted(locked))}")
    print(f"loose  {len(loose)}: {', '.join(sorted(loose))}")
    print(f"extra moves spent putting loose pieces away: {moves}")
    shot = mouse.shot()
    cv2.imwrite(str(SHOTS / "audit_end.png"), shot)
    print(f"cells still showing a photo: {sum(1 for b in boxes.values() if mean_at(shot, b) >= 45)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
