"""Judge the picture matcher against games that were actually played.

The log of a finished game is a truth table: every line "s26 (chair) -> cell(3,3) ... LOCKED" is
the engine accepting that piece in that cell, and every "refused" line is the engine saying no.  So
a recorded deal can be re-scored offline -- no mouse, no game -- and a change to the matcher can be
judged on the same deals instead of on fresh luck.

Usage:
    python tune_panel.py <batch dir with part1_N.log and part1_N_start.png> [...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import panel_match as pm  # noqa: E402
import stateless as st  # noqa: E402
import vision  # noqa: E402


class Fake:
    """Just what panel_match needs from a Pieces: where each piece is."""

    def __init__(self, boxes: dict[str, tuple]) -> None:
        self.box = boxes
        self.used: set[str] = set()


def parse(log_text: str):
    """(piece boxes from the log, locked pairs, refused pairs)."""
    slots = {}
    for m in re.finditer(r"^\s+(s\d\d) at \((\d+),(\d+)\)", log_text, re.M):
        slots[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    locked, refused = {}, set()
    for m in re.finditer(r"(s\d\d) \(\w+\) -> cell\((\d),(\d)\).*?(LOCKED|refused)", log_text):
        pair = (m.group(1), f"{m.group(2)},{m.group(3)}")
        if m.group(4) == "LOCKED":
            locked[pair[0]] = pair[1]
        else:
            refused.add(pair)
    return slots, locked, refused


def judge(start_png: Path, log_text: str, size: tuple[int, int], extra: int) -> dict:
    slots, locked, refused = parse(log_text)
    img = cv2.imread(str(start_png))
    if img is None:
        raise SystemExit(f"cannot read {start_png}")
    old = (pm.SIZE, pm.EXTRA)
    pm.SIZE, pm.EXTRA = size, extra
    try:
        snap = st.Snap(img)
        # log coordinates are the top-left of each slot; the slot size follows from the board
        piece_w, piece_h = snap.piece
        boxes = {k: (x, y, int(round(piece_w)), int(round(piece_h))) for k, (x, y) in slots.items()}
        pieces = Fake(boxes)
        cells = {k: snap.panel[k] for k in snap.cells}
        scores = pm.scores(img, pieces, cells)
        plan, conf, _m = pm.plan_and_margins(scores, cells)
    finally:
        pm.SIZE, pm.EXTRA = old
    right = sum(1 for ck, pk in plan.items() if locked.get(pk) == ck)
    wrong = sum(1 for ck, pk in plan.items() if (pk, ck) in refused)
    return {"cells": len(plan), "right": right, "wrong": wrong,
            "truth": len(locked), "pieces": len(boxes)}


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT.parent / "scratch"]
    games = []
    for d in dirs:
        for png in sorted(d.glob("part*_start.png")):
            log = png.with_name(png.name.replace("_start.png", ".log"))
            if log.exists():
                games.append((png, log.read_text(encoding="utf-8", errors="replace")))
    if not games:
        print("no (start.png, log) pairs found -- run a batch first")
        return 1
    print(f"{len(games)} recorded deal(s)\n")
    for size, extra in (((96, 72), 9), ((128, 96), 9), ((128, 96), 12), ((160, 120), 9)):
        tot_r = tot_n = tot_w = 0
        line = []
        for png, text in games:
            r = judge(png, text, size, extra)
            tot_r += r["right"]
            tot_w += r["wrong"]
            tot_n += r["truth"] or r["cells"]
            line.append(f"{r['right']}/{r['truth']}")
        print(f"  SIZE {size} EXTRA {extra:2d}: own cell {tot_r}/{tot_n} "
              f"({100.0 * tot_r / max(1, tot_n):.1f}%), refused pair chosen {tot_w}   "
              f"per game: {', '.join(line)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
