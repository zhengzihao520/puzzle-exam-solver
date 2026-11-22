"""Place pieces by hunting each piece's own cell, with the corrected 112x85 grid.

Why piece-centric
-----------------
`checkSnap` locks a piece only on its OWN cell, so trying many pieces on one cell wastes
drags: for each piece there is exactly one cell that can accept it.  Visiting the few
cells whose reference class matches the piece is therefore the shortest path, and every
attempt is verified twice -- the piece stops following the cursor, and the progress
counter moves.

Geometry, measured from a hand-placed piece: grid origin (433, 453), cell 112 x 85,
so cell centres are at 433 + c*112 + 56, 453 + r*85 + 42.

  python solve4.py --delay 10 --moves 3
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, check_foreground, _enable_dpi

N = 5
SNAP_NUDGE = (125, 95)          # far enough that a loose piece leaves the cell


def centre(box) -> tuple[int, int]:
    return box[0] + box[2] // 2, box[1] + box[3] // 2


def cell_mean(img, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


def ring_first(cells):
    """Outer ring first: only ring cells (or cells next to placed ones) can accept a piece."""
    return sorted(cells, key=lambda t: (0 if t[0] in (0, N - 1) or t[1] in (0, N - 1) else 1,
                                        t[0] * N + t[1]))


def try_place(mouse: Mouse, piece_box, cell_box) -> bool:
    """Drag the piece onto the cell centre and report whether the engine locked it."""
    src, dst = centre(piece_box), centre(cell_box)
    mouse.grab_at(src)
    mouse.glide((src[0] + 18, src[1] + 12))
    time.sleep(mouse.settle)
    mouse.glide(dst)
    time.sleep(mouse.settle * 1.5)
    mouse.glide((dst[0] + SNAP_NUDGE[0], dst[1] + SNAP_NUDGE[1]))
    time.sleep(mouse.settle * 1.5)
    held = cell_mean(mouse.shot(), cell_box)
    if held >= vision.BOARD_FILLED_LEVEL:
        # it stayed behind: locked.  Put the pointer back and let go.
        mouse.glide(dst)
        time.sleep(mouse.settle)
        mouse.release()
        return True
    mouse.glide(src)
    time.sleep(mouse.settle)
    mouse.release()
    time.sleep(0.15)
    return False


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--moves", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    global N
    N = args.grid

    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    clf = vision.Classifier()
    img = vision.grab()
    ref, board, _ = vision.layout(img)
    print(f"board {board}  cell {board.w / N:.0f}x{board.h / N:.0f}")
    filled = vision.board_filled(img, N)
    print("filled:", sum(filled), "/", N * N)

    need = []
    for cell in [vision.sub(img, b) for _r, _c, b in ref.cells(N)]:
        need.append("blank" if vision.is_blank(cell) else
                    str(clf.names[int(clf.probs(cell).argmax())]))
    rows = vision.tray_rows(img)
    pieces = []
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            cell = vision.sub(img, box)
            if vision.is_empty(cell):
                continue
            p = clf.probs(cell)
            label = "blank" if vision.is_white_piece(cell) else str(clf.names[int(p.argmax())])
            pieces.append({"key": f"{band[0]}{ri + 1}{ci + 1}", "box": box,
                           "label": label, "probs": p})
    print(f"tray pieces {len(pieces)}")
    if not pieces:
        print("ABORT: no pieces read from the tray")
        return 3

    free = [(r, c) for r in range(N) for c in range(N) if not filled[r * N + c]]
    mouse = Mouse()
    placed = 0
    for piece in pieces:
        if args.moves and placed >= args.moves:
            break
        same = [rc for rc in free if need[rc[0] * N + rc[1]] == piece["label"]]
        order = ring_first(same) if same else []
        if not order:
            print(f"{piece['key']} {piece['label']}: no matching cell left, skipped")
            continue
        print(f"{piece['key']} {piece['label']}: {len(order)} candidate cell(s)", flush=True)
        done = False
        for r, c in order:
            _r, _c, cell_box = board.cells(N)[r * N + c]
            print(f"   -> cell({r + 1},{c + 1})", flush=True)
            if try_place(mouse, piece["box"], cell_box):
                print("      locked in")
                placed += 1
                free.remove((r, c))
                done = True
                break
            print("      no snap")
        if done:
            continue
        print(f"   {piece['key']} found no home among its class cells")

    final = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "solve4_end.png"), final)
    print(f"\nplaced {placed}; board {sum(vision.board_filled(final, N))}/{N * N}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
