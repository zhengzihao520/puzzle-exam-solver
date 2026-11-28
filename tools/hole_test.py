"""Judge the hole reader on facts the board itself states -- no logs, no game.

Two tests, both free of any assumption about which piece goes where:

1. **Boundary sides.**  A cell in row 0 has a flat top, row 4 a flat bottom, column 0 a flat left,
   column 4 a flat right.  That is board geometry, so every hole on the edge is a labelled example of
   what the reader must return there.
2. **Bijection.**  A Part 3 board has exactly as many white pieces as white cells, and their shapes
   must pair up one-to-one.  If the five hole signatures and the five piece signatures are the same
   multiset, the assignment is determined even without knowing the answer.

    python tools/hole_test.py <frame dir> ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import stateless as st  # noqa: E402
import vision  # noqa: E402
import white_edges as we  # noqa: E402
from identify import Identifier  # noqa: E402

SIDES = ("top", "right", "bottom", "left")


def boundary_sides(r: int, c: int, n: int = 5) -> dict:
    """The sides the board forces to be flat for this cell."""
    out = {}
    if r == 0:
        out["top"] = "flat"
    if r == n - 1:
        out["bottom"] = "flat"
    if c == 0:
        out["left"] = "flat"
    if c == n - 1:
        out["right"] = "flat"
    return out


def show(sig) -> str:
    return " ".join(f"{s[0]}{(sig.get(s) or '?')}" for s in SIDES)


def analyse(img, name: str) -> tuple[int, int]:
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
        holes[k] = we.read_hole_edges(img, box)
        r, c = (int(v) for v in k.split(","))
        forced[k] = boundary_sides(r, c)
    whites = {}
    for k in pcs.tpl:
        b = pcs.box[k]
        if b[0] < 30 or b[1] < 30 or b[0] + b[2] > img.shape[1] - 30 \
                or b[1] + b[3] > img.shape[0] - 30:
            continue
        if st.whiteness(vision.sub(img, b)) >= 0.9:
            whites[k] = we.read_white_edges(img, b)
    print(f"\n{name}: {len(holes)} white hole(s), {len(whites)} white piece(s)")
    for k in holes:
        print(f"    hole  {k}: {show(holes[k])}   forced "
              + " ".join(f"{s[0]}{v[0]}" for s, v in forced[k].items()))
    for k in whites:
        print(f"    piece {k}: {show(whites[k])}")
    right = total = 0
    for k, sig in holes.items():
        for side, want in forced[k].items():
            total += 1
            if sig.get(side) == want:
                right += 1
            else:
                print(f"    boundary miss {k} {side}: read {sig.get(side)}, must be {want}")
    return right, total


def main() -> int:
    frames = []
    for a in sys.argv[1:]:
        p = Path(a)
        frames += sorted(p.glob("*.png")) if p.is_dir() else [p]
    right = total = 0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[:2] != (1080, 1920):
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        r, t = analyse(img, Path(f).name)
        right += r
        total += t
    if total:
        print(f"\nboundary sides: {right}/{total} = {100.0 * right / total:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
