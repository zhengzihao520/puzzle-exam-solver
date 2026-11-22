"""Find pieces lying anywhere on the page and put them back in the tray.

A refused drop is released wherever the pointer was, which may be outside every cell box
(the bottom-right corner of the board is a favourite), and a tray grid measured from dark
gaps can leave pieces parked *between* slots.  Neither shows up as a cell, so this scans the
whole window for piece-sized bright blobs while ignoring the reference panel, the page chrome
and the tray slots that legitimately hold a piece.

Uses vision.lattice_slots(): the tray lattice is a constant of the layout
(stepX = pieceWidth*1.16, stepY = pieceHeight*1.16), so it stays correct after most pieces
have been placed -- unlike gap detection, which collapsed to three half-height rows once
the tray emptied.

  python sweep_loose.py [--delay 3] [--dry]
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"
PIECE_AREA = (3500, 26000)      # one tray piece is ~112x85, tabs included


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--rounds", type=int, default=6)
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
    ref, board, _ = vision.layout(img)
    slot_boxes = [b for band in ("left", "right")
                  for _r, _c, b in vision.lattice_slots(img, band)]
    occupied = [b for b in slot_boxes if not vision.is_empty(vision.sub(img, b))]
    print(f"tray lattice {len(slot_boxes)} slots, {len(occupied)} holding a piece")
    print(f"reference panel ({ref.x},{ref.y}) {ref.w}x{ref.h}")

    for rnd in range(args.rounds):
        shot = mouse.shot()
        cv2.imwrite(str(SHOTS / f"sweep_{rnd}.png"), shot)
        g = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        mask = (g > 60).astype(np.uint8)
        mask[:300, :] = 0                                     # page header and controls
        mask[ref.y:ref.y + ref.h, ref.x:ref.x + ref.w] = 0     # the reference thumbnails
        # Only slots that legitimately hold a piece are excluded: excluding whole bands is
        # what hid the pieces parked between slots by an earlier, broken tray grid.
        for x, y, w, h in occupied:
            mask[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, _lab, stats, _ce = cv2.connectedComponentsWithStats(mask)
        found = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i][:5])
            if PIECE_AREA[0] <= area <= PIECE_AREA[1] and 80 <= w <= 220 and 60 <= h <= 180:
                found.append((x, y, w, h, area))
        print(f"round {rnd}: {len(found)} loose piece(s)")
        if not found:
            break
        for x, y, w, h, area in found:
            print(f"  blob ({x},{y}) {w}x{h} area {area}")
            if args.dry:
                continue
            pt = (x + w // 2, y + h // 2)
            mouse.grab_at(pt)
            mouse.glide((pt[0] + 40, pt[1] + 30))
            time.sleep(0.15)
            free = [b for b in slot_boxes if vision.is_empty(vision.sub(mouse.shot(), b))]
            if not free:
                print("    no free slot -- released where it is")
                mouse.release()
                continue
            mouse.glide(centre(free[0]))
            mouse.release()
            time.sleep(0.2)
            occupied.append(free[0])
            print(f"    -> tray slot ({free[0][0]},{free[0][1]})")

    final = mouse.shot()
    cv2.imwrite(str(SHOTS / "sweep_end.png"), final)
    filled = sum(1 for _r, _c, b in board.cells(5)
                 if float(cv2.cvtColor(vision.sub(final, b), cv2.COLOR_BGR2GRAY).mean()) >= 45)
    tray = sum(1 for b in slot_boxes if not vision.is_empty(vision.sub(final, b)))
    print(f"board cells showing a photo: {filled}")
    print(f"tray slots holding a piece:  {tray}  (a 5x5 game deals 30)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
