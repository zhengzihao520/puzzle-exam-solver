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
from collections import Counter, defaultdict
from pathlib import Path

import cv2

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

    needed = vision.reference_grid(img, n, clf)
    pieces = vision.tray_cells(img, clf)
    filled = vision.board_filled(img, n)

    print(f"image      : {path.name}  {img.shape[1]}x{img.shape[0]}")
    print(f"board      : {sum(filled)}/{n * n} cells already filled")
    print(f"reference  : {len(needed)} cells classified")
    print(f"tray       : {len(pieces)} pieces classified\n")

    quota = Counter(needed)
    have = Counter(p[1] for p in pieces)
    print("class        need  have")
    for cls in sorted(set(quota) | set(have), key=lambda k: (-quota[k], k)):
        flag = "" if have[cls] >= quota[cls] else "   <-- short"
        print(f"  {cls:10s} {quota[cls]:4d} {have[cls]:5d}{flag}")
    print()

    pool: dict[str, list] = defaultdict(list)
    for piece in pieces:
        pool[piece[1]].append(piece)

    plan = []
    missing = []
    for idx in spiral(n):
        if filled[idx]:
            continue
        cls = needed[idx]
        if not pool[cls]:
            missing.append((idx, cls))
            continue
        piece = pool[cls].pop(0)
        r, c = divmod(idx, n)
        bx, by, bw, bh = vision.Rect(*vision.BOARD).cells(n)[idx][2]
        plan.append((idx, r, c, cls, piece, (bx + bw // 2, by + bh // 2)))

    print(f"plan: {len(plan)} moves")
    for idx, r, c, cls, piece, (tx, ty) in plan:
        band, plabel, pri, pci, key, box = piece
        sx, sy = box[0] + box[2] // 2, box[1] + box[3] // 2
        print(f"  cell({r + 1},{c + 1}) <- {cls:10s} {key:4s} {plabel:9s} "
              f"({sx:4d},{sy:4d}) -> ({tx:4d},{ty:4d})")

    left_over = [p for cls in pool for p in pool[cls]]
    print(f"\nunused pieces: {len(left_over)} "
          f"{Counter(p[1] for p in left_over).most_common()}")
    if missing:
        print(f"cells without a usable piece: {[(i, k) for i, k in missing]}")

    # annotated copy for eyeballing
    vis = img.copy()
    for idx, r, c, cls, piece, (tx, ty) in plan:
        band, plabel, pri, pci, key, box = piece
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
