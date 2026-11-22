"""Is that piece on the board locked in, or just lying there?

The engine refuses to pick up a placed piece (getPieceAt skips isPlaced), and a placed piece
is drawn exactly at spot.x/spot.y.  A loose piece dropped with its box centre on the cell
centre is drawn in the same place, so the picture cannot tell them apart -- the drag can.
A locked piece also costs no move, because mousedown hits nothing and the engine returns
before incrementing its counter.

  python probe_locked.py 0,4 [--delay 3]
"""
from __future__ import annotations

import argparse
import time

import cv2

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"


def mean_at(img, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", nargs="?", default="0,4")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--grid", type=int, default=5)
    args = ap.parse_args()
    if args.delay:
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    mouse = Mouse()
    img = vision.grab()
    ref, board, _ = vision.layout(img)
    n = args.grid
    boxes = {f"{r},{c}": b for r, c, b in board.cells(n)}
    r, c = (int(v) for v in args.cell.split(","))
    box = boxes[args.cell]
    before = mean_at(img, box)
    print(f"cell({r},{c}) at {box}  mean {before:.1f}")
    if before < 45:
        print("nothing on that cell")
        return 0

    pt = centre(box)
    mouse.grab_at(pt)
    mouse.glide((pt[0] + 62, pt[1] + 38))
    time.sleep(0.25)
    held = mouse.shot()
    cv2.imwrite(str(SHOTS / "locked_probe.png"), held)
    after = mean_at(held, box)
    print(f"after a 62,38 nudge the same cell reads {after:.1f}")

    if after < 35:
        print("verdict: LOOSE -- the piece followed the pointer")
        mouse.glide(pt)          # put it back where it was
        mouse.release()
        time.sleep(0.2)
    else:
        print("verdict: LOCKED -- the engine would not lift it (mousedown found nothing)")
        mouse.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
