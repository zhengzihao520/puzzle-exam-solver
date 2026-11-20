"""Solve with the game's own verdict: the progress counter only moves on a correct drop.

Every attempt is: aim with a short search, drop, then compare the "进度: k/N" crop with
the crop taken before the drop.  A change is proof the piece belonged to that cell; no
change means the piece is lying loose, so it is dragged straight back to its slot and the
next candidate is tried.  Nothing is left on the board and no placement is assumed.

  python solve3.py --dry           show the plan
  python solve3.py --moves 3       stop after 3 confirmed placements
  python solve3.py                 whole board
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, check_foreground, _enable_dpi

N = 5
NEAR = [(0, 0), (-6, 0), (6, 0), (0, -6), (0, 6), (-6, -6), (6, -6), (-6, 6), (6, 6),
        (0, -13), (0, 13), (-13, 0), (13, 0)]


def spiral(n: int) -> list[int]:
    order: list[int] = []
    top, left, bottom, right = 0, 0, n - 1, n - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            order.append(top * n + c)
        top += 1
        for r in range(top, bottom + 1):
            order.append(r * n + right)
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                order.append(bottom * n + c)
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                order.append(r * n + left)
            left += 1
    return order


def centre(box) -> tuple[int, int]:
    return box[0] + box[2] // 2, box[1] + box[3] // 2


def cell_mean(img, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


def attempt(mouse: Mouse, piece_box, cell_box, slot_centre) -> str:
    """Drag a candidate to the cell and let the progress counter judge it.

    Returns 'ok' (piece belongs there), 'returned' (dragged back to its slot) or
    'failed' (could not grab the piece).
    """
    src, dst = centre(piece_box), centre(cell_box)
    before = vision.progress_crop(vision.grab())

    mouse.grab_at(src)
    mouse.glide((src[0] + 18, src[1] + 12))
    time.sleep(mouse.settle)
    slot_before = cell_mean(mouse.shot(), piece_box)
    for dx, dy in NEAR:
        mouse.glide((dst[0] + dx, dst[1] + dy))
        time.sleep(mouse.settle)
    mouse.release()
    # the counter updates with a short delay: poll instead of reading once
    for _wait in range(10):
        time.sleep(0.3)
        if not vision.progress_same(before, vision.progress_crop(vision.grab())):
            return "ok"

    # no progress: the piece is only lying there -> put it back
    mouse.grab_at(dst)
    mouse.glide((dst[0] + 40, dst[1] + 30))
    time.sleep(mouse.settle)
    mouse.glide(slot_centre)
    time.sleep(mouse.settle)
    mouse.release()
    time.sleep(0.2)
    return "returned"


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--moves", type=int, default=0)
    args = ap.parse_args()

    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok and not args.dry:
        return 2

    clf = vision.Classifier()
    img = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "solve3_start.png"), img)
    ref, board, _ = vision.layout(img)
    print(f"progress box: {vision.progress_box(img)}")

    need = []
    for cell in [vision.sub(img, b) for _r, _c, b in ref.cells(N)]:
        need.append("blank" if vision.is_blank(cell) else
                    str(clf.names[int(clf.probs(cell).argmax())]))
    rows = vision.tray_rows(img)
    pieces = []
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            cell = vision.sub(img, box)
            if vision.is_empty(cell) or not vision.is_real_piece(img, box):
                continue
            p = clf.probs(cell)
            label = "blank" if vision.is_white_piece(cell) else str(clf.names[int(p.argmax())])
            pieces.append({"key": f"{band[0]}{ri + 1}{ci + 1}", "box": box,
                           "label": label, "probs": p})
    filled = vision.board_filled(img, N)
    rows_now = len(rows)
    print(f"board {sum(filled)}/{N * N}, pieces {len(pieces)}, tray rows {rows_now}")
    if rows_now < 4 or len(pieces) < 18:
        print("ABORT: the tray is not fully on screen (page scrolled?), "
              "scroll so the whole tray is visible and rerun")
        return 3

    order = [i for i in spiral(N) if not filled[i]]
    plan = []
    for idx in order:
        want = need[idx]
        same = [p for p in pieces if p["label"] == want]
        others = [p for p in pieces if p["label"] != want]
        plan.append({"idx": idx, "want": want, "cands": same + others[:2]})
    print(f"plan: {len(plan)} cells")
    if args.dry:
        for s in plan:
            names = ", ".join(f"{p['key']}/{p['label']}" for p in s["cands"][:4])
            print(f"  cell({s['idx'] // N + 1},{s['idx'] % N + 1}) want {s['want']:9s} <- {names}")
        return 0

    print("\nstarting in 4s -- pointer into a screen corner aborts")
    time.sleep(4)
    mouse = Mouse()
    done = 0
    for step in plan:
        if args.moves and done >= args.moves:
            break
        idx = step["idx"]
        _r, _c, cell_box = board.cells(N)[idx]
        if cell_mean(vision.grab(), cell_box) >= vision.BOARD_FILLED_LEVEL:
            continue
        for cand in step["cands"]:
            if cand not in pieces:
                continue
            print(f"cell({idx // N + 1},{idx % N + 1}) want {step['want']:9s} "
                  f"try {cand['key']} {cand['label']}", flush=True)
            verdict = attempt(mouse, cand["box"], cell_box, centre(cand["box"]))
            if verdict == "ok":
                print("   progress moved -> placed")
                done += 1
                pieces = [p for p in pieces if p is not cand]
                break
            print("   progress unchanged -> returned to slot")
        else:
            print(f"   no candidate fit cell({idx // N + 1},{idx % N + 1})")

    final = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "solve3_end.png"), final)
    print(f"\nconfirmed placements: {done}; board reads {sum(vision.board_filled(final, N))}/{N * N}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
