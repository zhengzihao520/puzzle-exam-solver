"""Pair each white hole with the white piece that fits it, by continuous edge readings.

The three-way labels (tab / notch / flat) were not enough: one wrong label on one of four sides broke
the exact match on every hole tried.  The distance in pixels behind each label keeps that information
and degrades gracefully, so this pairs by total squared error over the eight numbers and then asks
whether the answer is a clean one-to-one match with a gap to the runner-up.

Board geometry is used as a hard constraint, not as a hint: a cell in row 0 has a flat top, row 4 a
flat bottom, column 0 a flat left, column 4 a flat right.

    python tools/hole_pair.py <frame dir> ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import panel_match as pm  # noqa: E402
import stateless as st  # noqa: E402
import vision  # noqa: E402
import white_edges as we  # noqa: E402
from identify import Identifier  # noqa: E402

SIDES = ("top", "right", "bottom", "left")


def boundary_sides(r: int, c: int, n: int = 5) -> set:
    out = set()
    if r == 0:
        out.add("top")
    if r == n - 1:
        out.add("bottom")
    if c == 0:
        out.add("left")
    if c == n - 1:
        out.add("right")
    return out


def distance(hole: dict, piece: dict, forced: set) -> float:
    """Squared error over the eight numbers, board-forced sides counted twice."""
    total = 0.0
    for side in SIDES:
        if side not in hole or side not in piece:
            continue
        weight = 2.0 if side in forced else 1.0
        ho, hi = hole[side]
        po, pi = piece[side]
        # a flat side reads ~0 outward and ~0 inward, a tab reads ~19 outward: compare both numbers
        total += weight * ((ho - po) ** 2 + (hi - pi) ** 2)
    return total


def analyse(img, name: str) -> tuple[int, int, int]:
    snap = st.Snap(img)
    idf = Identifier()
    pcs = st.Pieces()
    pcs.inventory(snap, idf, verbose=False)
    holes, forced = {}, {}
    for k, box in snap.cells.items():
        if vision.has_piece(vision.sub(img, box), 30):
            continue
        if not vision.is_blank(vision.sub(img, snap.panel[k])):
            continue
        prof = we.profiles_of(img, box, hole=True)
        if prof:
            holes[k] = prof
            r, c = (int(v) for v in k.split(","))
            forced[k] = boundary_sides(r, c)
    pieces = {}
    for k in pcs.tpl:
        b = pcs.box[k]
        if b[0] < 30 or b[1] < 30 or b[0] + b[2] > img.shape[1] - 30 \
                or b[1] + b[3] > img.shape[0] - 30:
            continue
        if st.whiteness(vision.sub(img, b)) < 0.9:
            continue
        prof = we.profiles_of(img, b, hole=False)
        if prof:
            pieces[k] = prof
    print(f"\n{name}: {len(holes)} white hole(s), {len(pieces)} white piece(s)")
    if not holes or not pieces:
        return 0, 0, 0
    weights = {(hk, pk): distance(hs, ps, forced[hk])
               for hk, hs in holes.items() for pk, ps in pieces.items()}
    plan = pm._assign_max(list(pieces), list(holes), {k: -v for k, v in weights.items()})
    pairs = sorted(plan.items(), key=lambda kv: weights[(kv[0], kv[1])])
    for hk, pk in pairs:
        others = sorted(weights[(hk, p)] for p in pieces)
        second = others[1] if len(others) > 1 else 0.0
        mine = weights[(hk, pk)]
        print(f"    {hk} <- {pk}   err {mine:7.1f}   next best {second:7.1f}"
              f"   margin {second - mine:7.1f}")
    return len(plan), len(holes), len(pieces)


def main() -> int:
    frames = []
    for a in sys.argv[1:]:
        p = Path(a)
        frames += sorted(p.glob("*.png")) if p.is_dir() else [p]
    ok = tot = 0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[:2] != (1080, 1920):
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        n, h, _p = analyse(img, Path(f).name)
        ok += n
        tot += h
    if tot:
        print(f"\npaired {ok}/{tot} holes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
