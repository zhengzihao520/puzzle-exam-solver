"""The board never keeps a wrong piece.

A refused drop leaves the piece lying on the cell.  The rule for this project is that it goes
back to the tray immediately, so the board only ever shows correct pieces -- and so the next
attempt lands on an empty cell instead of on top of somebody else.

Finding a loose piece needs a *dense* press grid, not five points: a piece offset by 40 px from
a cell centre is missed by the five-point probe (that is how the same board measured 19, 16 and
8 locked in successive audits).  Nine points spanning the whole cell catch anything that
overlaps the cell at all.

  python clean_board.py [--dry]
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"
GRID = ((0, 0), (-35, -26), (0, -26), (35, -26), (-35, 0), (35, 0),
        (-35, 26), (0, 26), (35, 26))


def grey(img, box) -> np.ndarray:
    return cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY)


def snap(mouse):
    img = mouse.shot()
    ref, board, _ = vision.layout(img)
    cells = {f"{r},{c}": b for r, c, b in board.cells(5)}
    slots, rows, top, occ = vision.calibrate_tray(img, expected=30)
    return img, ref, board, cells, slots


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    mouse = Mouse()
    parked = 0
    for rnd in range(8):
        img, ref, board, cells, slots = snap(mouse)
        cv2.imwrite(str(SHOTS / f"clean_{rnd}.png"), img)
        acted = 0
        for key, box in cells.items():
            # re-read before every cell: parking changes the tray, and the previous cell may
            # have moved a piece onto this one
            img = mouse.shot()
            if not vision.has_piece(vision.sub(img, box), 30):
                continue
            cx, cy = centre(box)
            lifted = False
            for dx, dy in GRID:
                pt = (cx + dx, cy + dy)
                before = float(grey(mouse.shot(), box).mean())
                mouse.grab_at(pt)
                mouse.glide((pt[0] + 58, pt[1] + 40))
                time.sleep(0.12)
                after = float(grey(mouse.shot(), box).mean())
                if after < before - 20 or after < 28:
                    lifted = True
                    break
                mouse.release()
            if not lifted:
                continue                       # correct piece (or nothing reachable)
            if args.dry:
                print(f"  cell({key}) loose")
                mouse.glide((cx, cy))
                mouse.release()
                break
            now = mouse.shot()
            slots_now, _r, _t, _o = vision.calibrate_tray(now, expected=30)
            free = [b for b in slots_now if not vision.has_piece(vision.sub(now, b), 20)]
            if not free:
                print("  no empty slot -- putting it back where it was")
                mouse.glide((cx, cy))
                mouse.release()
                continue
            mouse.glide(centre(free[0]))
            mouse.release()
            parked += 1
            acted += 1
            time.sleep(0.18)
            print(f"  cell({key}) loose -> back to slot ({free[0][0]},{free[0][1]})")
        print(f"round {rnd}: {acted} piece(s) taken back")
        if acted == 0:
            break

    img, ref, board, cells, slots = snap(mouse)
    cv2.imwrite(str(SHOTS / "clean_end.png"), img)
    print(f"\npieces taken back: {parked}")
    holding = [k for k, b in cells.items() if vision.has_piece(vision.sub(img, b), 30)]
    print(f"board cells still holding something: {len(holding)} -> {sorted(holding)}")
    print("(everything left there is either correct or was not reachable this round)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
