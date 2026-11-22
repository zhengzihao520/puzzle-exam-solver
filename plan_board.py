"""Read the whole board and print the plan: which cell needs which class, which piece is
which class, and -- where the answer is unambiguous -- which piece belongs to which cell.

Read-only: nothing is dragged, nothing is placed.

The reference panel deliberately shows a *different photo of the same class* for every cell
(recon/catalog.js, alternatePhoto), so the cell can only be named by class, never matched as
a template.  That leaves a within-class ambiguity, and the engine resolves it silently: a
piece snaps only into its own cell (SOLUTIONS is per piece).  This script makes the ambiguity
explicit -- it names the classes, lists the pieces per class and prints the assignments that
follow without any guessing.

  python plan_board.py [--grid 5] [--delay 10]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import cv2

import vision
from identify import Identifier, BLANK
from solve_live import _enable_dpi, check_foreground

OUT = vision.ROOT / "logs" / "plan_board.json"


def order_ring(n: int) -> list[tuple[int, int]]:
    """Outer ring clockwise, then the next ring inwards: the engine's anchor rule means a
    cell is only eligible once its own row/col touches the frame or a neighbour is placed."""
    out = []
    for k in range((n + 1) // 2):
        top, bottom, left, right = k, n - 1 - k, k, n - 1 - k
        if top == bottom:
            out.extend((top, c) for c in range(left, right + 1))
            continue
        out.extend((top, c) for c in range(left, right))
        out.extend((r, right) for r in range(top, bottom))
        out.extend((bottom, c) for c in range(right, left, -1))
        out.extend((r, left) for r in range(bottom, top, -1))
    return out


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    n = args.grid
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    img = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "plan_board.png"), img)
    ref, board, _ = vision.layout(img)
    idf = Identifier()

    print(f"\nreference panel ({ref.x},{ref.y}) {ref.w}x{ref.h}")
    need: dict[tuple[int, int], object] = {}
    for r, c, box in ref.cells(n):
        cell = vision.sub(img, box)
        if vision.is_blank(cell):
            print(f"  ({r},{c})  BLANK (empty thumbnail)")
            need[(r, c)] = None
            continue
        res = idf.identify(cell)
        need[(r, c)] = res
        print(f"  ({r},{c})  {res}")

    print("\ntray")
    rows = vision.tray_rows(img)
    pieces: dict[str, object] = {}
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            cell = vision.sub(img, box)
            if vision.is_empty(cell):
                continue
            key = f"{band[0]}{ri + 1}{ci + 1}"
            res = idf.identify(cell)
            pieces[key] = (res, box)
            print(f"  {key}  {res}")

    need_cnt = Counter(v.label for v in need.values() if v)
    pool: dict[str, list[str]] = defaultdict(list)
    for key, (res, _box) in pieces.items():
        if res.label != BLANK:
            pool[res.label].append(key)

    print(f"\n{'class':10s} {'cells':>5s} {'pieces':>6s}   pieces")
    for cls in sorted(set(need_cnt) | set(pool), key=lambda k: (-need_cnt[k], -len(pool[k]), k)):
        mark = ""
        if len(pool[cls]) < need_cnt[cls]:
            mark = "  SHORT"
        elif len(pool[cls]) > need_cnt[cls]:
            mark = f"  +{len(pool[cls]) - need_cnt[cls]}"
        print(f"  {cls:10s} {need_cnt[cls]:5d} {len(pool[cls]):6d}{mark}   {', '.join(pool[cls])}")

    # cells that can be filled with no guesswork: the class occurs exactly once on the
    # board and exactly once on the tray
    unique = {c: v.label for c, v in need.items()
              if v and need_cnt[v.label] == 1 and len(pool.get(v.label, [])) == 1}
    print(f"\nunambiguous cells: {len(unique)} of {sum(1 for v in need.values() if v)}")
    for (r, c), cls in sorted(unique.items()):
        print(f"  cell({r},{c}) <- {pool[cls][0]}  ({cls})")

    ring = order_ring(n)
    ambiguous = [(r, c) for r, c in ring if (r, c) in need and need[(r, c)]
                 and need_cnt[need[(r, c)].label] > 1]
    print(f"\nambiguous cells (class repeats): {len(ambiguous)}")
    for (r, c) in ambiguous:
        cls = need[(r, c)].label
        print(f"  cell({r},{c}) {cls:9s} <- any of {', '.join(pool.get(cls, [])) or 'NONE'}"
              f"   ({len(pool.get(cls, []))} candidates)")

    OUT.write_text(json.dumps({
        "cells": {f"{r},{c}": ({"class": v.label, "confident": v.confident,
                                "alts": v.alternatives} if v else None)
                  for (r, c), v in need.items()},
        "pieces": {k: {"class": res.label, "confident": res.confident, "alts": res.alternatives,
                       "box": box} for k, (res, box) in pieces.items()},
        "unique": {f"{r},{c}": pool[cls][0] for (r, c), cls in unique.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
