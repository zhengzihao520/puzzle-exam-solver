"""Put every loose piece back in a slot -- proven by the pickup itself, not by memory.

Why this exists: the placement loop tracks a piece by the slot it came from.  If a release
does not land the piece back in that slot, the bookkeeping is wrong from then on and the
piece is never found again -- which is exactly what happened: one piece was left out of
place, its cell stayed empty, and every later attempt to fill that cell failed.

So this does not trust any remembered position.  It scans the whole window for piece-sized
blobs, presses each one, and asks the screen whether the piece came away:

    blob went dark after the press  -> it was a loose piece, carry it to a free slot
    blob unchanged                  -> it is a locked piece (the engine will not lift it)
                                       or page furniture; release and move on

The slot it lands in is verified empty on a fresh screenshot first, so two pieces can never
end up stacked in one slot -- the failure that made the tray unreadable last time.

  python recover_pieces.py [--dry]
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"
AREA = (3000, 27000)          # one piece is ~112x85 = 9520 px plus tabs


def grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blobs(img, ref, board):
    """Piece-sized bright blobs over the whole window, minus the UI and the reference panel."""
    g = grey(img)
    m = (g > 55).astype(np.uint8)
    m[:150, :] = 0
    m[ref.y - 10:ref.y + ref.h + 10, ref.x - 10:ref.x + ref.w + 10] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _lab, st, _ce = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i][:5])
        if AREA[0] <= a <= AREA[1] and 70 <= w <= 220 and 50 <= h <= 180:
            out.append((x, y, w, h))
    return out


def mean_of(img, box) -> float:
    return float(grey(vision.sub(img, box)).mean())


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    mouse = Mouse()
    ref, board, _ = vision.layout(mouse.shot())
    print(f"board {board.x},{board.y} {board.w}x{board.h}, reference {ref.x},{ref.y}")

    moved = 0
    for rnd in range(args.rounds):
        shot = mouse.shot()
        cv2.imwrite(str(SHOTS / f"recover_{rnd}.png"), shot)
        found = blobs(shot, ref, board)
        print(f"\nround {rnd}: {len(found)} blob(s) (board included: a locked piece cannot be lifted, so pressing it costs nothing)")
        if not found:
            break
        acted = 0
        for x, y, w, h in found:
            pt = (x + w // 2, y + h // 2)
            here = (x, y, w, h)
            before = mean_of(mouse.shot(), here)
            mouse.grab_at(pt)
            mouse.glide((pt[0] + 70, pt[1] + 50))
            time.sleep(0.18)
            after = mean_of(mouse.shot(), here)
            if after > before - 20:
                # nothing came away: a locked piece or page furniture
                mouse.release()
                print(f"  ({x},{y}) stayed put  (before {before:.0f} after {after:.0f})")
                continue
            # it is in hand: find a slot that is empty *now*
            shot2 = mouse.shot()
            slots = [b for band in ("left", "right")
                     for _r, _c, b in vision.lattice_slots(shot2, band, rows=6, top_rel=-182)]
            free = [b for b in slots if not vision.has_piece(vision.sub(shot2, b), 20)]
            if not free:
                print("  no empty slot -- releasing where it is")
                mouse.release()
                break
            print(f"  ({x},{y}) lifted; parking in empty slot ({free[0][0]},{free[0][1]})")
            if args.dry:
                mouse.glide(centre(free[0]))
                mouse.release()
                continue
            mouse.glide(centre(free[0]))
            mouse.release()
            time.sleep(0.25)
            check = mouse.shot()
            landed = vision.has_piece(vision.sub(check, free[0]), 20)
            print(f"    verified: slot holds a piece = {landed}")
            moved += 1
            acted += 1
        if acted == 0:
            print("  nothing could be lifted this round")
            break

    final = mouse.shot()
    cv2.imwrite(str(SHOTS / "recover_end.png"), final)
    slots = [b for band in ("left", "right")
             for _r, _c, b in vision.lattice_slots(final, band, rows=6, top_rel=-182)]
    occ = sum(1 for b in slots if vision.has_piece(vision.sub(final, b), 20))
    filled = sum(1 for _r, _c, b in board.cells(5) if vision.has_piece(vision.sub(final, b), 20))
    print(f"\npieces parked: {moved}")
    print(f"tray slots holding a piece: {occ} of {len(slots)}")
    print(f"board cells holding a piece: {filled} of 25")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
