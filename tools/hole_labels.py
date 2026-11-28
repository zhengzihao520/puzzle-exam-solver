"""Can a white cell's cut signature be read from its *neighbours*?  Measure before building.

The board renders nothing in an empty cell (grey 23.0, same as the gaps), so the cell's own shape
cannot be seen.  But it does not need to be: the cell's cut on a side is the complement of the
piece already placed on that side, and that piece's facing edge is a bright silhouette against the
dark hole -- the same measurement `white_edges` makes on a piece in the tray, just aimed inward.

For a 5x5 the five white cells must be filled by the five white pieces, so the multiset of the five
cell signatures has to equal the multiset of the five piece signatures.  That equality is the test:
if it holds on real frames, the assignment is determined and one move per white cell is enough.

    python probe_white_cells.py <frame dir or single png> ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"F:\exam_next")
sys.path.insert(0, str(ROOT))

import stateless as st  # noqa: E402
import vision  # noqa: E402
import white_edges as we  # noqa: E402
from identify import BLANK, Identifier  # noqa: E402

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
COMPLEMENT = {"tab": "notch", "notch": "tab", "flat": "flat", None: None}
NEIGHBOUR = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}


def facing_side(img, cell_box, side: str, reach: int = 46):
    """The neighbour's edge that faces into `cell_box`, read as if it were a piece's own side.

    Crops a band on `side` of the empty cell, takes the bright (piece) mask, and for each line
    across the band measures where the silhouette sits.  A bulge towards the cell is the
    neighbour's tab (our piece needs a notch there); an inward step is its notch.
    """
    x, y, w, h = cell_box
def facing_side(img, cell_box, side: str, reach: int = 46):
    """The neighbour's edge that faces into `cell_box`, read the way a piece's own side is read.

    A band on `side` of the empty cell holds the neighbour's half of the boundary.  The neighbour
    is a photo (bright) and the hole is tray grey (23), so for every line across the band the
    boundary is simply where the bright run ends:

      side "left"   the neighbour's RIGHT edge, so per row take the rightmost bright column
      side "right"  the neighbour's LEFT  edge, so per row take the leftmost bright column
      side "top"    the neighbour's BOTTOM edge, so per column take the lowest bright row
      side "bottom" the neighbour's TOP   edge, so per column take the highest bright row

    A bulge towards the cell is the neighbour's tab -- and therefore a notch in the piece that
    belongs in this cell; an indent away from it is the neighbour's notch.
    """
    x, y, w, h = cell_box
    if side == "left":
        band = (x - reach, y, reach, h)
    elif side == "right":
        band = (x + w, y, reach, h)
    elif side == "top":
        band = (x, y - reach, w, reach)
    else:
        band = (x, y + h, w, reach)
    crop = vision.sub(img, band)
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    m = (g > 60).astype(np.uint8)          # a piece is a photo: far brighter than the 23 hole
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if int(m.sum()) < 400:
        return None

    if side in ("left", "right"):
        rows = np.nonzero(m.any(axis=1))[0]
        if len(rows) < 20:
            return None
        prof = np.empty(len(rows), np.float32)
        for i, r in enumerate(rows):
            cols = np.nonzero(m[r])[0]
            prof[i] = cols.max() if side == "left" else cols.min()
        toward_cell_grows = side == "left"        # for "left", a bigger column is nearer the cell
    else:
        cols = np.nonzero(m.any(axis=0))[0]
        if len(cols) < 20:
            return None
        prof = np.empty(len(cols), np.float32)
        for i, c in enumerate(cols):
            rr = np.nonzero(m[:, c])[0]
            prof[i] = rr.max() if side == "top" else rr.min()
        toward_cell_grows = side == "top"          # for "top", a bigger row is nearer the cell

    n = len(prof)
    mid = prof[n // 3: 2 * n // 3]
    core = float(np.median(prof))
    outward = (float(mid.max()) - core) if toward_cell_grows else (core - float(mid.min()))
    inward = (core - float(mid.min())) if toward_cell_grows else (float(mid.max()) - core)
    if outward > we.TAB_PX:
        return "tab"
    if inward > we.TAB_PX:
        return "notch"
    if abs(outward) < we.FLAT_PX and abs(inward) < we.FLAT_PX:
        return "flat"
    return None


def analyse(img) -> None:
    snap = st.Snap(img)
    idf = Identifier()
    white_cells, empty_cells = [], []
    for key, box in snap.cells.items():
        if not vision.has_piece(vision.sub(img, box), 30):
            empty_cells.append(key)
            if vision.is_blank(vision.sub(img, snap.panel[key])):
                white_cells.append(key)
    print(f"  filled {25 - len(empty_cells)}/25   empty {len(empty_cells)}"
          f"   of which white {len(white_cells)}")
    if not white_cells:
        return
    pieces = st.Pieces()
    pieces.inventory(snap, idf, verbose=False)
    whites = []
    for k in pieces.tpl:
        box = pieces.box[k]
        if box[0] < 30 or box[1] < 30 or box[0] + box[2] > img.shape[1] - 30 \
                or box[1] + box[3] > img.shape[0] - 30:
            continue                      # a slot at the frame edge: the crop would be short
        if st.whiteness(vision.sub(img, box)) >= 0.9:
            whites.append(k)
    sigs = {k: we.read_white_edges(img, pieces.box[k]) for k in whites}
    print(f"  white pieces in the tray: {len(whites)}")
    for k, v in sigs.items():
        print(f"    piece {k}: " + " ".join(f"{s[0]}{v.get(s) or '?'}" for s in SIDES))
    req = {}
    for key in white_cells:
        r, c = (int(v) for v in key.split(","))
        got = {}
        for side in SIDES:
            dr, dc = NEIGHBOUR[side]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 5 and 0 <= nc < 5):
                got[side] = "flat"                 # board edge
                continue
            nb = f"{nr},{nc}"
            if nb in empty_cells:
                got[side] = None                   # neighbour not placed yet: no information
                continue
            got[side] = COMPLEMENT[facing_side(img, snap.cells[key], side)]
        req[key] = got
        print(f"    cell {key}: " + " ".join(f"{s[0]}{got.get(s) or '?'}" for s in SIDES))
    # the test: the multiset of readable cell signatures must be subsumed by the pieces'
    piece_pool = [dict(sigs[k]) for k in whites]
    matched = 0
    for key, need in req.items():
        for i, sig in enumerate(piece_pool):
            if all(need[s] is None or sig[s] == need[s] for s in SIDES):
                matched += 1
                piece_pool.pop(i)
                break
    print(f"  signatures: {matched}/{len(white_cells)} white cell(s) have a piece that fits on "
          f"every readable side")


def main() -> int:
    args = sys.argv[1:]
    frames = []
    for a in args:
        p = Path(a)
        frames += sorted(p.glob("*.png")) if p.is_dir() else [p]
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[:2] != (1080, 1920):
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        print(f"\n{Path(f).name}")
        analyse(img)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
