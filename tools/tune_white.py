"""Rank variants of the white-hole/piece matcher against labelled games.

Each game contributes (log, frame dir): the log names the piece the engine accepted into each white
cell, the frames are what the loop saw when it was deciding.  A variant is judged by the rank it
gives the true piece -- rank 1 means the loop would have placed it first and the engine would have
taken it, which is the whole point.

Variants:

  sum       the current one: squared error over the informative sides' continuous readings
  hard      the same, after dropping pieces that contradict the informative sides' labels
  hard+sum  hard filter, then the continuous distance as the tie-break (what the loop would use)
  labels    count of agreeing labels only

    python tune_white.py <log> <frame dir> [<log> <frame dir> ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2

ROOT = Path(r"F:\exam_next")
sys.path.insert(0, str(ROOT))

import stateless as st  # noqa: E402
import vision  # noqa: E402
import white_edges as we  # noqa: E402
from identify import Identifier  # noqa: E402

SIDES = ("top", "right", "bottom", "left")
NEIGHBOUR = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
REACH = 20


def parse_log(text: str) -> dict:
    locked = {}
    for m in re.finditer(r"(s\d\d) \(\w+\) -> cell\((\d),(\d)\)(.*?)(LOCKED|refused)", text):
        if m.group(5) == "LOCKED":
            locked[f"{m.group(2)},{m.group(3)}"] = m.group(1)
    return locked


def hole_read(img, box, placed: set, key: str) -> tuple[dict, dict, set]:
    """(continuous profiles, labels, informative sides) for one empty cell."""
    prof = we.profiles_of(img, box, hole=True, mode="drop")
    lab = we.read_hole_edges(img, box, mode="drop")
    r, c = (int(v) for v in key.split(","))
    good = {s for s, (dr, dc) in NEIGHBOUR.items() if f"{r + dr},{c + dc}" in placed}
    good |= {s for s, on in (("top", r == 0), ("bottom", r == 4),
                             ("left", c == 0), ("right", c == 4)) if on}
    return prof, lab, good


def variants(hole_p, hole_l, sides, pieces_p, pieces_l, truth):
    """rank of `truth` under each variant."""
    keys = list(pieces_p)
    if truth not in keys:
        return {}

    def sse(pk):
        t = 0.0
        for s in sides & set(hole_p) & set(pieces_p[pk]):
            ho, hi = hole_p[s]
            po, pi = pieces_p[pk][s]
            t += (ho - po) ** 2 + (hi - pi) ** 2
        return t

    def agree(pk):
        n = 0
        for s in sides & set(hole_l) & set(pieces_l[pk]):
            if hole_l[s] and pieces_l[pk][s] and hole_l[s] == pieces_l[pk][s]:
                n += 1
        return n

    def known(pk):
        return len([s for s in sides & set(hole_l) & set(pieces_l[pk])
                    if hole_l[s] and pieces_l[pk][s]])

    out = {}
    by_sum = sorted(keys, key=sse)
    out["sum"] = 1 + by_sum.index(truth)

    # a piece contradicts the hole when a readable side disagrees; boundary flats are the strictest
    def contradicts(pk):
        for s in sides & set(hole_l) & set(pieces_l[pk]):
            if s in ("top", "bottom", "left", "right") and s in sides:
                if hole_l[s] and pieces_l[pk][s] and hole_l[s] != pieces_l[pk][s]:
                    return True
        return False

    clean = [k for k in keys if not contradicts(k)] or keys
    by_hard = sorted(clean, key=sse)
    out["hard+sum"] = 1 + by_hard.index(truth) if truth in by_hard else len(keys)
    by_hard_only = sorted(clean, key=lambda k: -agree(k))
    out["hard"] = 1 + by_hard_only.index(truth) if truth in by_hard_only else len(keys)
    by_lab = sorted(keys, key=lambda k: (-agree(k), known(k)))
    out["labels"] = 1 + by_lab.index(truth)
    return out


def main() -> int:
    args = sys.argv[1:]
    pairs = [(Path(args[i]), Path(args[i + 1])) for i in range(0, len(args) - 1, 2)]
    tally = {"sum": [], "hard": [], "hard+sum": [], "labels": []}
    for log_path, frame_dir in pairs:
        locked = parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
        frames = sorted(frame_dir.glob("*.png"))
        print(f"\n{log_path.name} + {frame_dir.name}: {len(frames)} frame(s), "
              f"{len(locked)} locked cell(s)")
        for f in frames:
            m = re.match(r"w_(\d)_(\d)_(s\d\d)_(\d+)", f.name)
            if not m:
                continue
            key = f"{m.group(1)},{m.group(2)}"
            truth = locked.get(key)
            if truth is None:
                continue
            img = cv2.imread(str(f))
            if img is None:
                continue
            if img.shape[:2] != (1080, 1920):
                img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            snap = st.Snap(img)
            placed = {k for k, b in snap.cells.items()
                      if vision.has_piece(vision.sub(img, b), 30)}
            idf = Identifier()
            pcs = st.Pieces()
            pcs.inventory(snap, idf, verbose=False)
            pieces_p, pieces_l = {}, {}
            for k in pcs.tpl:
                b = pcs.box[k]
                if b[0] < 30 or b[1] < 30 or b[0] + b[2] > img.shape[1] - 30 \
                        or b[1] + b[3] > img.shape[0] - 30:
                    continue
                if st.whiteness(vision.sub(img, b)) < 0.9:
                    continue
                p = we.profiles_of(img, b, hole=False)
                if p:
                    pieces_p[k] = p
                    pieces_l[k] = we.read_white_edges(img, b)
            if truth not in pieces_p or len(pieces_p) < 2:
                continue
            hole_p, hole_l, sides = hole_read(img, snap.cells[key], placed, key)
            if not hole_p or not sides:
                continue
            got = variants(hole_p, hole_l, sides, pieces_p, pieces_l, truth)
            if not got:
                continue
            print(f"  {f.name}: truth {truth}  " + "  ".join(f"{k} {v}/{len(pieces_p)}"
                                                             for k, v in got.items()))
            for k, v in got.items():
                tally[k].append(v)
    print("\nsummary (rank 1 = the loop would have placed the true piece first)")
    for k, vals in tally.items():
        if vals:
            print(f"  {k:9s}: rank 1 in {sum(1 for v in vals if v == 1)}/{len(vals)}"
                  f"   mean rank {sum(vals) / len(vals):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
