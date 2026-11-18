"""Dry run: read one screenshot, print the full piece -> board-cell plan. No mouse.

Usage:
  python plan.py [path/to/screenshot.png]      (default: shots/now_state.png)

What it does
  1. classifies the reference panel  -> which class each board cell needs
  2. classifies every tray piece     -> what we actually have
  3. checks quotas: every non-blank class on the board must be available in the tray
  4. walks the board outer ring -> inner, clockwise, and pairs each empty cell with a
     piece of the required class
  5. writes an annotated screenshot to shots/plan_dryrun.png and prints the plan

Nothing is dragged here, so it is safe to run against a live page.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

import vision

ROOT = Path(__file__).resolve().parent


def spiral(n: int) -> list[int]:
    """Board cells outer ring -> inner, clockwise, as row-major indices."""
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


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "shots" / "now_state.png"
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    n = 5
    clf = vision.Classifier()
    ref = vision.find_reference(img)
    ref_cells = [vision.sub(img, box) for _r, _c, box in ref.cells(n)]
    piece_cells = []
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band):
            cell = vision.sub(img, box)
            if not vision.is_empty(cell):
                piece_cells.append((band, f"{band[0]}{ri + 1}{ci + 1}", ri, ci, box, cell))
    filled = vision.board_filled(img, n)

    print(f"image      : {path.name}  {img.shape[1]}x{img.shape[0]}")
    print(f"reference  : {ref}  ({ref.w / n:.1f}x{ref.h / n:.1f} per cell)")
    print(f"board      : {sum(filled)}/{n * n} cells already filled")
    print(f"tray       : {len(piece_cells)} pieces in "
          f"{len(vision.tray_rows(img))}x3 slots per band\n")

    # ---- probabilities: pieces x reference cells -------------------------
    ref_probs, ref_txt = [], []
    for cell in ref_cells:
        if vision.is_blank(cell):
            ref_probs.append(None)
            ref_txt.append("blank")
        else:
            p = clf.probs(cell)
            ref_probs.append(p)
            ref_txt.append(str(clf.names[int(p.argmax())]))
    piece_probs, piece_txt = [], []
    for item in piece_cells:
        cell = item[5]
        if vision.is_blank(cell):
            piece_probs.append(None)
            piece_txt.append("blank")
        else:
            p = clf.probs(cell)
            piece_probs.append(p)
            piece_txt.append(str(clf.names[int(p.argmax())]))

    def score(ci: int, pi: int) -> float:
        """How well piece pi fills reference cell ci."""
        if ref_txt[ci] == "blank" or piece_txt[pi] == "blank":
            return 1.0 if ref_txt[ci] == piece_txt[pi] else -1.0
        rp, pp = ref_probs[ci], piece_probs[pi]
        return float(np.dot(rp, pp))          # probability that both mean the class

    # ---- global greedy assignment, best pair first ----------------------
    pairs = sorted(((score(ci, pi), ci, pi)
                    for ci in range(len(ref_cells))
                    for pi in range(len(piece_cells))), reverse=True)
    cell_of = {}
    taken = set()
    for s, ci, pi in pairs:
        if s <= 0 or ci in cell_of or pi in taken:
            continue
        cell_of[ci] = pi
        taken.add(pi)

    need = Counter(ref_txt)
    have = Counter(piece_txt)
    print("class        need  have")
    for cls in sorted(set(need) | set(have), key=lambda k: (-need[k], k)):
        flag = "" if have[cls] >= need[cls] else "   <-- short"
        print(f"  {cls:10s} {need[cls]:4d} {have[cls]:5d}{flag}")

    plan = []
    for idx in spiral(n):
        ci = idx
        if filled[idx] or ci not in cell_of:
            continue
        pi = cell_of[ci]
        band, key, pri, pci, box, _cell = piece_cells[pi]
        r, c = divmod(idx, n)
        _rr, _cc, (bx, by, bw, bh) = vision.Rect(*vision.BOARD).cells(n)[idx]
        plan.append((idx, r, c, ref_txt[ci], piece_txt[pi], key, box,
                     (bx + bw // 2, by + bh // 2)))

    print(f"\nplan: {len(plan)} moves")
    for idx, r, c, want, got, key, box, (tx, ty) in plan:
        sx, sy = box[0] + box[2] // 2, box[1] + box[3] // 2
        mark = "" if want == got else "  (labels differ)"
        print(f"  cell({r + 1},{c + 1}) need {want:9s} <- {key:4s} {got:9s} "
              f"({sx:4d},{sy:4d}) -> ({tx:4d},{ty:4d}){mark}")

    unused = [piece_cells[i][1] for i in range(len(piece_cells)) if i not in taken]
    unfilled = [f"({i // n + 1},{i % n + 1})" for i in spiral(n)
                if not filled[i] and i not in cell_of]
    print(f"\nunused pieces: {len(unused)} {unused}")
    if unfilled:
        print(f"cells left without a piece: {unfilled}")

    # annotated copy for eyeballing
    vis = img.copy()
    for idx, r, c, want, got, key, box, (tx, ty) in plan:
        sx, sy = box[0] + box[2] // 2, box[1] + box[3] // 2
        cv2.rectangle(vis, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      (0, 255, 0), 2)
        cv2.putText(vis, f"{idx + 1}", (box[0] + 3, box[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.line(vis, (sx, sy), (tx, ty), (0, 255, 0), 1)
        cv2.circle(vis, (tx, ty), 4, (0, 200, 255), -1)
        cv2.putText(vis, f"{idx + 1}", (tx - 8, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    out = ROOT / "shots" / "plan_dryrun.png"
    cv2.imwrite(str(out), vis)
    print(f"\nannotated: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
