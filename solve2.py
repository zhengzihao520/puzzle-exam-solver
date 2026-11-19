"""Corner -> edge -> interior solver driven by the reference panel and piece shape.

Rule from the site (recon/puzzle_board.js, engine_PieceCutter.js)
----------------------------------------------------------------
* The panel on the right shows, per grid position, a *different* photo of the same
  class, so it tells you which class belongs to each cell -- not which photo.
* `SOLUTIONS.set(piece, {row, col})` maps every piece to exactly one cell, so two
  pieces of the same class are not interchangeable.
* Pieces are real jigsaw pieces (`tabSize = 0.22*min(w,h)`), therefore:
      corner cell -> exactly two flats, the two board sides it touches
      edge cell   -> exactly one flat, the board side it touches
      inner cell  -> no flats
  The flat-side set pins a piece to a *set of cells*; the reference class then pins
  the cell among those, and the engine's snap confirms it.

Order: 4 corners, then the 12 edge cells clockwise, then the 9 inner cells.  Every drop
is confirmed with the lock probe (a placed piece cannot be picked up again), so a wrong
candidate is returned to its slot instead of being left on the board.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import vision
from solve_live import Mouse, attempt, cell_mean, check_foreground, centre, _enable_dpi

ROOT = vision.ROOT
N = 5
SIDES = ("top", "bottom", "left", "right")


def cell_pattern(r: int, c: int) -> frozenset[str]:
    """Board sides this cell touches: those must be flat on the piece."""
    pat = []
    if r == 0:
        pat.append("top")
    if r == N - 1:
        pat.append("bottom")
    if c == 0:
        pat.append("left")
    if c == N - 1:
        pat.append("right")
    return frozenset(pat)


def cached_slots(shot: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Slot boxes from a screenshot of the full tray (slots never move during a game)."""
    img = cv2.imread(str(shot))
    if img is None:
        raise SystemExit(f"missing {shot}")
    rows = vision.tray_rows(img)
    out = []
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            out.append((f"{band[0]}{ri + 1}{ci + 1}", box))
    if len(out) < N * N:
        raise SystemExit(f"only {len(out)} slots found in {shot}")
    return out


def read_table(img: np.ndarray, slots) -> list[dict]:
    """Pieces still on the table, with class and flat-side signature."""
    clf = vision.Classifier()
    out = []
    for key, box in slots:
        if vision.is_empty(vision.sub(img, box)) or not vision.is_real_piece(img, box):
            continue
        flats = frozenset(s for s, k in vision.flat_sides(img, box).items() if k == "flat")
        p = clf.probs(vision.sub(img, box))
        out.append({"key": key, "box": box, "flats": flats,
                    "label": str(clf.names[int(p.argmax())]), "probs": p})
    return out, clf


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--moves", type=int, default=0)
    ap.add_argument("--shot", default="", help="screenshot of the current state")
    ap.add_argument("--slots", default="shots/s1.png", help="screenshot of the full tray")
    args = ap.parse_args()

    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok and not args.dry:
        return 2

    img = cv2.imread(args.shot) if args.shot else vision.grab()
    cv2.imwrite(str(ROOT / "shots" / "solve2_start.png"), img)
    ref, board, _ = vision.layout(img)
    filled = vision.board_filled(img, N)
    slots = cached_slots(ROOT / args.slots)
    pieces, clf = read_table(img, slots)
    print(f"board filled {sum(filled)}/{N * N}, pieces on table {len(pieces)}")

    ref_cells = [vision.sub(img, b) for _r, _c, b in ref.cells(N)]
    need, ref_probs = [], []
    for cell in ref_cells:
        if vision.is_blank(cell):
            need.append("blank")
            ref_probs.append(None)
        else:
            p = clf.probs(cell)
            ref_probs.append(p)
            need.append(str(clf.names[int(p.argmax())]))

    print("flat signatures:", dict(Counter("".join(sorted(p["flats"])) or "-" for p in pieces)))

    cells = []
    for r in range(N):
        for c in range(N):
            idx = r * N + c
            if filled[idx]:
                continue
            cells.append({"idx": idx, "r": r, "c": c, "pattern": cell_pattern(r, c),
                          "want": need[idx], "want_probs": ref_probs[idx]})
    cells.sort(key=lambda x: (-len(x["pattern"]), x["r"], x["c"]))   # corners, edges, inner

    plan = []
    remaining = list(pieces)
    for cell in cells:
        exact = [p for p in remaining if p["flats"] == cell["pattern"]]
        loose = [p for p in remaining if p["label"] == cell["want"] and p not in exact]

        def rank(p):
            if cell["want_probs"] is None:
                return 0.0
            return float(np.dot(p["probs"], cell["want_probs"]))

        exact.sort(key=rank, reverse=True)
        loose.sort(key=rank, reverse=True)
        cands = exact[:3] + loose[:2]
        if not cands:
            cands = sorted(remaining, key=rank, reverse=True)[:2]
        plan.append({**cell, "cands": cands})

    print(f"\nplan: {len(plan)} cells")
    for step in plan:
        names = ", ".join(f"{p['key']}[{'/'.join(sorted(p['flats'])) or '-'}]{p['label']}"
                          for p in step["cands"])
        print(f"  cell({step['r'] + 1},{step['c'] + 1}) {'/'.join(sorted(step['pattern'])) or 'inner':11s}"
              f" want {step['want']:9s} <- {names}")
    if args.dry:
        return 0

    print("\nstarting in 4s -- pointer into a screen corner aborts")
    time.sleep(4)
    mouse = Mouse()
    placed = 0
    for step in plan:
        if args.moves and placed >= args.moves:
            break
        _r, _c, cell_box = board.cells(N)[step["idx"]]
        if cell_mean(vision.grab(), cell_box) >= vision.BOARD_FILLED_LEVEL:
            continue
        for cand in step["cands"]:
            print(f"cell({step['r'] + 1},{step['c'] + 1}) want {step['want']:9s} "
                  f"try {cand['key']} {cand['label']}", flush=True)
            if attempt(mouse, cand["box"], cell_box):
                print("   locked in")
                placed += 1
                remaining = [p for p in remaining if p is not cand]
                break
            print("   no snap, returned to slot")
        else:
            print(f"   no candidate snapped on cell({step['r'] + 1},{step['c'] + 1})")

    final = vision.grab()
    cv2.imwrite(str(ROOT / "shots" / "solve2_end.png"), final)
    print(f"\nplaced {placed}; board {sum(vision.board_filled(final, N))}/{N * N}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
