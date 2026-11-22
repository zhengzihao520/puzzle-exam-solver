"""Brute-force placement: no class filtering, only the engine's answer.

A piece can lock on exactly one cell, so visiting empty cells until it stops following the
cursor is deterministic -- it cannot place a piece wrongly, it can only take time.  Ring
cells go first because `hasAnchor` refuses an interior piece whose neighbours are not in
place yet, and a second pass picks up whatever the first pass could not anchor.

Geometry (measured from a hand-placed piece): grid origin (433,453), cell 112 x 85.

  python solve5.py --delay 10          whole board
  python solve5.py --passes 1 --max-seconds 600
"""
from __future__ import annotations

import argparse
import time

import cv2

import vision
from solve_live import Mouse, check_foreground, _enable_dpi

N = 5
SNAP_NUDGE = (125, 95)


def centre(box):
    return box[0] + box[2] // 2, box[1] + box[3] // 2


def cell_mean(img, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


def try_place(mouse: Mouse, piece_box, cell_box) -> bool:
    src, dst = centre(piece_box), centre(cell_box)
    mouse.grab_at(src)
    mouse.glide((src[0] + 18, src[1] + 12))
    time.sleep(mouse.settle)
    mouse.glide(dst)
    time.sleep(mouse.settle * 1.2)
    mouse.glide((dst[0] + SNAP_NUDGE[0], dst[1] + SNAP_NUDGE[1]))
    time.sleep(mouse.settle * 1.2)
    held = cell_mean(mouse.shot(), cell_box)
    if held >= vision.BOARD_FILLED_LEVEL:
        mouse.glide(dst)
        time.sleep(mouse.settle)
        mouse.release()
        return True
    mouse.glide(src)
    time.sleep(mouse.settle)
    mouse.release()
    time.sleep(0.12)
    return False


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=2400)
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

    img = vision.grab()
    ref, board, _ = vision.layout(img)
    print(f"board {board}  cell {board.w / N:.0f}x{board.h / N:.0f}")
    rows = vision.tray_rows(img)
    pieces = []
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            cell_img = vision.sub(img, box)
            if vision.is_empty(cell_img):
                continue
            if vision.is_white_piece(cell_img):
                continue          # decoy: no SOLUTIONS entry, it can never snap
            pieces.append({"key": f"{band[0]}{ri + 1}{ci + 1}", "box": box})
    print(f"tray pieces {len(pieces)}")
    if not pieces:
        print("ABORT: no pieces read from the tray")
        return 3

    mouse = Mouse()
    started = time.time()
    placed_total = 0
    for p in range(args.passes):
        if not pieces or time.time() - started > args.max_seconds:
            break
        filled = vision.board_filled(vision.grab(), N)
        empty = [(r, c) for r in range(N) for c in range(N) if not filled[r * N + c]]
        if not empty:
            break
        ring = [rc for rc in empty if rc[0] in (0, N - 1) or rc[1] in (0, N - 1)]
        inner = [rc for rc in empty if rc not in ring]
        order_cells = ring + inner
        print(f"\n=== pass {p + 1}: {len(pieces)} pieces, {len(empty)} empty cells "
              f"({len(ring)} ring) ===", flush=True)
        progressed = 0
        for piece in list(pieces):
            if time.time() - started > args.max_seconds:
                print("time budget reached")
                break
            hit = None
            for r, c in order_cells:
                if time.time() - started > args.max_seconds:
                    break
                _r, _c, cell_box = board.cells(N)[r * N + c]
                if cell_mean(vision.grab(), cell_box) >= vision.BOARD_FILLED_LEVEL:
                    order_cells.remove((r, c))
                    continue
                if try_place(mouse, piece["box"], cell_box):
                    hit = (r, c)
                    break
            if hit:
                print(f"  {piece['key']} -> cell({hit[0] + 1},{hit[1] + 1}) OK", flush=True)
                placed_total += 1
                progressed += 1
                pieces.remove(piece)
                order_cells.remove(hit)
            else:
                print(f"  {piece['key']} no cell accepted it this pass", flush=True)
            if not order_cells:
                break
        if progressed == 0:
            print("no progress in this pass, stopping")
            break

    final = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "solve5_end.png"), final)
    print(f"\nplaced {placed_total}; board {sum(vision.board_filled(final, N))}/{N * N}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
