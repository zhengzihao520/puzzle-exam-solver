"""Solve the assignment: which piece belongs to which cell, without trying.

A piece is interchangeable with another piece of the same class as far as the reference
panel is concerned, because the panel shows a different photo of the same class for every
cell.  What is *not* interchangeable is the cut: the site's edge matrix is globally
consistent (recon/puzzle_board.js),

    piece(r,c).right === -piece(r,c+1).left
    piece(r,c).bottom === -piece(r+1,c).top

and every side on the frame is flat.  Two constraints together -- the class the cell asks
for, and the edges that fit its neighbours -- pin the assignment down; in practice they
leave exactly one solution for a 5x5 board.

Edges are given per side as +1 (tab), -1 (notch), 0 (flat) or None (not read).  A missing
edge simply imposes no constraint, so partial reads still narrow the field.

  from jigsaw import solve
  sol = solve(cells, pieces, n=5)      # -> SolveResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DELTA = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}


@dataclass
class Piece:
    key: str
    cls: str
    edges: dict[str, int | None] = field(default_factory=dict)

    def edge(self, side: str) -> int | None:
        return self.edges.get(side)


@dataclass
class SolveResult:
    assignment: dict[tuple[int, int], str]      # cell -> piece key
    unused: list[str]                           # decoys, or pieces that do not fit
    solutions: int                              # how many were found (capped)
    nodes: int                                  # search effort


def ring_order(n: int) -> list[tuple[int, int]]:
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


def _fits(cell, piece: Piece, n: int, placed: dict, pieces: dict[str, Piece]) -> bool:
    """Does this piece's cut match the frame and every already-placed neighbour?"""
    r, c = cell
    for side in SIDES:
        e = piece.edge(side)
        if e is None:
            continue
        dr, dc = DELTA[side]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n and 0 <= nc < n):
            if e != 0:                       # the frame is straight
                return False
            continue
        if e == 0:                           # an interior side is never flat
            return False
        nb = placed.get((nr, nc))
        if nb is None:
            continue
        other = pieces[nb].edge(OPPOSITE[side])
        if other is not None and e != -other:
            return False
    return True


def solve(cells: dict[tuple[int, int], str], pieces: dict[str, Piece], n: int = 5,
          limit: int = 2) -> SolveResult:
    """Backtracking with the class and cut constraints; stops at `limit` solutions."""
    order = ring_order(n)
    by_class: dict[str, list[str]] = {}
    for key, p in pieces.items():
        by_class.setdefault(p.cls, []).append(key)

    solutions: list[dict[tuple[int, int], str]] = []
    placed: dict[tuple[int, int], str] = {}
    used: set[str] = set()
    nodes = count()

    def step(i: int) -> None:
        if len(solutions) >= limit:
            return
        next(nodes)
        if i == len(order):
            solutions.append(dict(placed))
            return
        cell = order[i]
        need = cells.get(cell)
        if need is None:                      # nothing to place here
            step(i + 1)
            return
        for key in by_class.get(need, ()):
            if key in used:
                continue
            if not _fits(cell, pieces[key], n, placed, pieces):
                continue
            placed[cell] = key
            used.add(key)
            step(i + 1)
            used.discard(key)
            del placed[cell]
            if len(solutions) >= limit:
                return

    step(0)
    best = solutions[0] if solutions else {}
    return SolveResult(assignment=best,
                       unused=sorted(set(pieces) - set(best.values())),
                       solutions=len(solutions),
                       nodes=next(nodes, 0))


def consistent(cells: dict[tuple[int, int], str], pieces: dict[str, Piece],
               assignment: dict[tuple[int, int], str], n: int = 5) -> list[str]:
    """List every place an assignment breaks the rules -- used by the tests."""
    bad = []
    for (r, c), key in assignment.items():
        p = pieces[key]
        if cells[(r, c)] != p.cls:
            bad.append(f"({r},{c}) wants {cells[(r, c)]} but {key} is {p.cls}")
        for side in SIDES:
            e = p.edge(side)
            dr, dc = DELTA[side]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                if e not in (0, None):
                    bad.append(f"({r},{c}) {side} must be flat on the frame, reads {e}")
                continue
            if e == 0:
                bad.append(f"({r},{c}) {side} is flat inside the board")
                continue
            nb = assignment.get((nr, nc))
            if nb is None:
                continue
            other = pieces[nb].edge(OPPOSITE[side])
            if other is not None and e != -other:
                bad.append(f"({r},{c}).{side}={e} but neighbour {nb}."
                           f"{OPPOSITE[side]}={other}")
    return bad

def violations(cells, pieces, assignment, n=5) -> int:
    """How many rules the arrangement breaks: class, frame, and each shared edge."""
    bad = 0
    for (r, c), key in assignment.items():
        p = pieces[key]
        if cells.get((r, c)) != p.cls:
            bad += 1
        for side in SIDES:
            e = p.edge(side)
            if e is None:
                continue
            dr, dc = DELTA[side]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                if e != 0:
                    bad += 1
                continue
            nb = assignment.get((nr, nc))
            if nb is None:
                continue
            other = pieces[nb].edge(OPPOSITE[side])
            if other is not None and e != -other:
                bad += 1
    return bad


def solve_soft(cells: dict, pieces: dict, n: int = 5, rounds: int = 400) -> SolveResult:
    """Choose an assignment that breaks as few rules as possible.

    Exact solving is too brittle: one misread edge on one piece makes the whole board
    unsatisfiable (measured: 5% wrong edges -> no solution in 195 of 200 synthetic games).
    The real question is not "is the arrangement perfect" but "which piece is each cell's
    best partner", so this minimises broken rules instead of demanding zero.

    Seeded with a greedy pass (ring order, fewest broken rules first), then improved by
    swapping pieces inside a class group -- the only swaps that can help, because the class
    constraint is strict.
    """
    order = ring_order(n)
    by_class: dict[str, list[str]] = {}
    for key, p in pieces.items():
        by_class.setdefault(p.cls, []).append(key)

    assignment: dict[tuple[int, int], str] = {}
    used: set[str] = set()
    for cell in order:
        need = cells.get(cell)
        if need is None:
            continue
        best, best_bad = None, None
        for key in by_class.get(need, ()):
            if key in used:
                continue
            trial = dict(assignment)
            trial[cell] = key
            bad = violations(cells, pieces, trial, n)
            if best_bad is None or bad < best_bad:
                best, best_bad = key, bad
        if best is not None:
            assignment[cell] = best
            used.add(best)

    current = violations(cells, pieces, assignment, n)
    for _ in range(rounds):
        improved = False
        cells_by_class: dict[str, list[tuple[int, int]]] = {}
        for cell, key in assignment.items():
            cells_by_class.setdefault(pieces[key].cls, []).append(cell)
        for cls, group in cells_by_class.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    ka, kb = assignment[a], assignment[b]
                    if ka == kb:
                        continue
                    assignment[a], assignment[b] = kb, ka
                    bad = violations(cells, pieces, assignment, n)
                    if bad < current:
                        current = bad
                        improved = True
                    else:
                        assignment[a], assignment[b] = ka, kb
        if not improved:
            break
    return SolveResult(assignment=dict(assignment),
                       unused=sorted(set(pieces) - set(assignment.values())),
                       solutions=1 if current == 0 else 0,
                       nodes=current)


def candidates_for(cells: dict, pieces: dict, cell, n: int = 5) -> list[tuple[str, int]]:
    """Pieces that could take this cell, best first, with the rules they break."""
    need = cells.get(cell)
    out = []
    for key, p in pieces.items():
        if p.cls != need:
            continue
        trial = {cell: key}
        out.append((key, violations(cells, pieces, trial, n)))
    return sorted(out, key=lambda t: t[1])
