"""Check the assignment solver in panel_match against brute force.

The solver decides which piece goes to which cell, and a 25-move game needs every one of those
25 decisions to be right, so it is verified rather than trusted: 300 random matrices up to 5x6 are
solved both ways, and the totals must agree exactly.  Also checks the two shapes it is used in --
30 pieces for 25 cells (the deal) and 6 pieces for 9 cells (the endgame, when more cells than
pieces are open).

    python tools/assign_test.py
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import panel_match as pm  # noqa: E402


def brute(rows, cols, w):
    """Best total weight over every one-to-one assignment, by trying all of them.

    Written for both shapes: pieces are usually more numerous than cells (30 for 25) but the
    endgame turns that around, so whichever side is smaller is the one being permuted.
    """
    best = None
    if len(rows) <= len(cols):
        for perm in itertools.permutations(range(len(cols)), len(rows)):
            tot = sum(w.get((rows[i], cols[perm[i]]), 0.0) for i in range(len(rows)))
            if best is None or tot > best:
                best = tot
    else:
        for perm in itertools.permutations(range(len(rows)), len(cols)):
            tot = sum(w.get((rows[perm[j]], cols[j]), 0.0) for j in range(len(cols)))
            if best is None or tot > best:
                best = tot
    return best


def total(rows, cols, w, plan) -> float:
    return sum(w.get((plan[ck], ck), 0.0) for ck in plan)


def main() -> int:
    random.seed(11)
    bad = 0
    for _ in range(300):
        n_row = random.randint(1, 5)                 # pieces
        n_col = random.randint(1, 6)                 # cells
        rows = [f"p{i}" for i in range(n_row)]
        cols = [f"c{j}" for j in range(n_col)]
        w = {(r, c): round(random.uniform(-1, 1), 3) for r in rows for c in cols}
        plan = pm._assign_max(rows, cols, w)
        if len(plan) != min(n_row, n_col) or len(set(plan.values())) != len(plan):
            bad += 1
            print(f"bad shape: {n_row}x{n_col} -> {len(plan)} pairs, "
                  f"{len(set(plan.values()))} distinct pieces")
            continue
        got, want = total(rows, cols, w, plan), brute(rows, cols, w)
        if abs(got - want) > 1e-9:
            bad += 1
            print(f"suboptimal: {n_row}x{n_col} got {got:.4f} want {want:.4f}")
            if bad > 3:
                return 1

    # the two shapes the deal and the endgame actually produce
    for n_row, n_col, label in ((30, 25, "deal"), (6, 9, "endgame"), (25, 25, "balanced")):
        rows = [f"p{i}" for i in range(n_row)]
        cols = [f"c{j}" for j in range(n_col)]
        w = {(r, c): random.random() for r in rows for c in cols}
        plan = pm._assign_max(rows, cols, w)
        print(f"  {label:9s} {n_row} pieces x {n_col} cells -> {len(plan)} pairs, "
              f"total {total(rows, cols, w, plan):.3f}")
    print("brute-force agreement:", "all 300" if not bad else f"{bad} disagreement(s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
