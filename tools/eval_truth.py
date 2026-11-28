"""Judge the hole/piece descriptor against the engine's own answers.

Each captured frame is named after the cell and the piece the loop was about to offer, and the run's
log says whether the engine took it.  So a frame where the cell's true piece is known gives a
labelled example for the interior sides too -- which board geometry cannot supply.

    python eval_truth.py <log> <frame dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2

ROOT = Path(r"F:\exam_next")
sys.path.insert(0, str(ROOT))

import panel_match as pm  # noqa: E402
import stateless as st  # noqa: E402
import vision  # noqa: E402
import white_edges as we  # noqa: E402
from identify import Identifier  # noqa: E402

SIDES = ("top", "right", "bottom", "left")
NEIGHBOUR = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
REACH = 20


def parse_log(text: str) -> tuple[dict, set]:
    locked, refused = {}, set()
    for m in re.finditer(r"(s\d\d) \(\w+\) -> cell\((\d),(\d)\)(.*?)(LOCKED|refused)", text):
        cell = f"{m.group(2)},{m.group(3)}"
        if m.group(5) == "LOCKED":
            locked[cell] = m.group(1)
        else:
            refused.add((m.group(1), cell))
    return locked, refused


def hole_and_sides(img, box, placed: set, key: str):
    mask = we.read_hole_mask(img, box, 24, "drop")
    if mask.shape[0] < 20:
        return {}, set()
    x, y, w, h = box
    pad = 24
    r, c = (int(v) for v in key.split(","))
    keep = mask.copy() * 0
    left = pad - (REACH if f"{r},{c - 1}" in placed else 0)
    right = pad + w + (REACH if f"{r},{c + 1}" in placed else 0)
    top = pad - (REACH if f"{r - 1},{c}" in placed else 0)
    bottom = pad + h + (REACH if f"{r + 1},{c}" in placed else 0)
    keep[max(0, top):bottom, max(0, left):right] = 1
    m = (mask & keep).astype("uint8")
    ys, xs = m.nonzero()
    if len(ys) < 200:
        return {}, set()
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    good = {s for s, (dr, dc) in NEIGHBOUR.items() if f"{r + dr},{c + dc}" in placed}
    good |= {s for s, on in (("top", r == 0), ("bottom", r == 4), ("left", c == 0), ("right", c == 4)) if on}
    return we._profiles_of_mask(m), good


def dist(hole: dict, piece: dict, sides: set) -> float:
    total = 0.0
    for side in sides & set(hole) & set(piece):
        ho, hi = hole[side]
        po, pi = piece[side]
        total += (ho - po) ** 2 + (hi - pi) ** 2
    return total


def main() -> int:
    log_text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    locked, _refused = parse_log(log_text)
    frames = sorted(Path(sys.argv[2]).glob("*.png"))
    print(f"{len(frames)} frame(s); the log names {len(locked)} locked cell(s)")
    rank1 = total = 0
    for f in frames:
        m = re.match(r"w_(\d)_(\d)_(s\d\d)_(\d+)", f.name)
        if not m:
            continue
        key = f"{m.group(1)},{m.group(2)}"
        offered = m.group(3)
        truth = locked.get(key)
        if truth is None:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[:2] != (1080, 1920):
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        snap = st.Snap(img)
        placed = {k for k, b in snap.cells.items() if vision.has_piece(vision.sub(img, b), 30)}
        idf = Identifier()
        pcs = st.Pieces()
        pcs.inventory(snap, idf, verbose=False)
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
        if truth not in pieces or len(pieces) < 2:
            continue
        hole, sides = hole_and_sides(img, snap.cells[key], placed, key)
        if not hole or not sides:
            continue
        rows = sorted(((dist(hole, p, sides), k) for k, p in pieces.items()))
        mine = dict((k, d) for d, k in rows)[truth]
        better = [d for d, k in rows if d < mine]
        total += 1
        rank1 += 1 if not better else 0
        print(f"  {f.name}: offered {offered}, truth {truth} -> err {mine:7.1f}, "
              f"rank {1 + len(better)}/{len(rows)}"
              f"{'' if not better else '   beaten by ' + ', '.join(k for d, k in rows if d < mine)}")
    if total:
        print(f"\ntruth ranked first in {rank1}/{total} frame(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
