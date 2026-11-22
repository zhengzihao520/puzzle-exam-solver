"""Validate the assignment solver offline, without touching the site.

The generator follows recon/puzzle_board.js exactly:

    top    = (r === 0)      ? 0 : -hEdges[r-1][c]
    right  = (c === cols-1) ? 0 :  vEdges[r][c]
    bottom = (r === rows-1) ? 0 :  hEdges[r][c]
    left   = (c === 0)      ? 0 : -vEdges[r][c-1]

and decoys are cut with `Math.random() < 0.5 ? 1 : -1` on all four sides, so they never carry
a flat edge -- which is itself a discriminator, because every frame cell needs at least one.

Checks:
  * a clean board must solve to exactly one solution, equal to the truth
  * with a fraction of the edges misread, how often is it still right?
  * with a fraction of the edges unreadable, how often is it still right?

  python test_jigsaw.py [--trials 500]
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

import jigsaw

CLASSES = ["dog", "cat", "elephant", "desktop", "bicycle", "car", "chair", "cup",
           "airplane", "backpack", "umbrella", "suitcase", "penguin", "tafei", "nailong"]


def make_game(n: int, rng: random.Random, decoys: int):
    h = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n - 1)]
    v = [[rng.choice((-1, 1)) for _ in range(n - 1)] for _ in range(n)]

    cells: dict[tuple[int, int], str] = {}
    truth_edges: dict[tuple[int, int], dict[str, int]] = {}
    for r in range(n):
        for c in range(n):
            cells[(r, c)] = rng.choice(CLASSES)
            truth_edges[(r, c)] = {
                "top": 0 if r == 0 else -h[r - 1][c],
                "right": 0 if c == n - 1 else v[r][c],
                "bottom": 0 if r == n - 1 else h[r][c],
                "left": 0 if c == 0 else -v[r][c - 1],
            }

    pieces: dict[str, jigsaw.Piece] = {}
    truth: dict[tuple[int, int], str] = {}
    for (r, c), cls in cells.items():
        key = f"p{r}{c}"
        pieces[key] = jigsaw.Piece(key, cls, dict(truth_edges[(r, c)]))
        truth[(r, c)] = key
    for i in range(decoys):
        pieces[f"d{i}"] = jigsaw.Piece(
            f"d{i}", rng.choice(CLASSES),
            {s: rng.choice((-1, 1)) for s in jigsaw.SIDES})
    return cells, pieces, truth


def corrupt(pieces, rng, wrong=0.0, unknown=0.0):
    for p in pieces.values():
        for s in jigsaw.SIDES:
            roll = rng.random()
            if roll < unknown:
                p.edges[s] = None
            elif roll < unknown + wrong:
                p.edges[s] = -p.edges[s]
    return pieces


def run(n: int = 5, trials: int = 200, decoys: int = 5, wrong: float = 0.0,
        unknown: float = 0.0) -> dict:
    rng = random.Random(20261117)
    stats = Counter()
    nodes = []
    for _ in range(trials):
        cells, pieces, truth = make_game(n, rng, decoys)
        if wrong or unknown:
            corrupt(pieces, rng, wrong=wrong, unknown=unknown)
        order = list(pieces)
        rng.shuffle(order)
        shuffled = {k: pieces[k] for k in order}
        res = jigsaw.solve(cells, shuffled, n=n)
        nodes.append(res.nodes)
        stats["trials"] += 1
        if res.solutions == 1:
            stats["unique"] += 1
        if res.assignment == truth:
            stats["exact"] += 1
        elif res.solutions == 1:
            stats["unique_but_wrong"] += 1
        if res.solutions == 0:
            stats["no_solution"] += 1
        bad = jigsaw.consistent(cells, shuffled, res.assignment, n=n) if res.assignment else []
        if bad:
            stats["self_inconsistent"] += 1
    stats["nodes_max"] = max(nodes)
    stats["nodes_avg"] = round(sum(nodes) / len(nodes), 1)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    print("clean board")
    print("  ", dict(run(trials=args.trials)))
    for w, u in ((0.05, 0.0), (0.10, 0.0), (0.20, 0.0), (0.0, 0.25), (0.05, 0.25), (0.10, 0.50)):
        print(f"wrong {w:.0%}  unread {u:.0%}")
        print("  ", dict(run(trials=args.trials, wrong=w, unknown=u)))
    print("4x4 board, clean")
    print("  ", dict(run(n=4, trials=args.trials, decoys=3)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
