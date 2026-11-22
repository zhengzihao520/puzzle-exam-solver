"""Read a piece's four edge types from its silhouette -- the missing piece of the puzzle.

The site cuts each piece with tabs and notches, and the edge matrix is globally consistent
(recon/puzzle_board.js):

    piece(r,c).right === -piece(r,c+1).left
    piece(r,c).bottom === -piece(r+1,c).top

so knowing every piece's four edge signs turns "which of the three chairs belongs to which
of the three chair cells" from a guess into a constraint solve.  In the tray the pieces sit
13-18 px apart, which is narrower than a tab (19 px), so the silhouette is unreadable there.
But a piece held in the hand can be parked over blank canvas: then the whole outline stands
alone on a known background and all four sides can be measured.

The measurement is a background-difference mask, not a brightness threshold, so dark photos
work as well as bright ones:

    protrusion on a side = how far the mask reaches outward over the middle half of the side
    notch     on a side = how far the background reaches inward over the middle half

  python probe_edges.py [--slot 0] [--delay 3]      # one piece, report and put it back
  python probe_edges.py --all                       # every piece in the tray (1 move each)
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"
OUT = vision.ROOT / "logs" / "edges.json"
BLANK = (760, 950)          # blank canvas below the board: clear by 45 px on every side
TAB = 19                    # PieceCutter: tabSize = 0.22 * min(pieceWidth, pieceHeight)
CUT = 8                     # how many pixels count as "clearly protruding / clearly notched"


def bg_colour(img: np.ndarray, box, pad: int = 26) -> np.ndarray:
    """Median colour of a ring around the piece box: what the piece is standing on."""
    x, y, w, h = box
    outer = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    inner = img[y:y + h, x:x + w]
    mask = np.ones(outer.shape[:2], bool)
    oy, ox = y - max(0, y - pad), x - max(0, x - pad)
    mask[oy:oy + h, ox:ox + w] = False
    if not mask.any():
        return np.median(inner.reshape(-1, 3), axis=0)
    return np.median(outer[mask], axis=0)


def piece_mask(img: np.ndarray, box, pad: int = 28, tol: int = 16):
    """Binary silhouette: pixels that differ from the background the piece stands on."""
    x, y, w, h = box
    bg = bg_colour(img, box)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    win = img[y0:y1, x0:x1].astype(np.int16)
    diff = np.abs(win - bg.astype(np.int16)).max(axis=2)
    m = (diff > tol).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m, (x0, y0), bg


def read_edges(img: np.ndarray, box, verbose: bool = False) -> dict[str, str]:
    """{'top','right','bottom','left'} -> 'tab' | 'notch' | 'flat'."""
    x, y, w, h = box
    m, (x0, y0), bg = piece_mask(img, box)
    bx, by = x - x0, y - y0
    band = 0.5                                  # measure the middle half of each side
    prof: dict[str, float] = {}

    def outward(profiles, name):
        depth = 0
        for d in range(1, 34):
            if d > len(profiles) - 1:
                break
            if profiles[d] > 0.5:
                depth = d
        prof[name] = depth
        return depth

    def inward(name):
        depth = 0
        for d in range(1, 34):
            idx = d - 1
            if idx >= (w if name in ("top", "bottom") else h):
                break
            if prof_slices[name][idx] < 0.5:
                depth = d
            else:
                break
        prof[name + "_in"] = depth
        return depth

    # outward profiles per side, middle half of the side only
    cols = slice(bx + int(w * (0.5 - band / 2)), bx + int(w * (0.5 + band / 2)))
    rows = slice(by + int(h * (0.5 - band / 2)), by + int(h * (0.5 + band / 2)))
    sizes = {
        "left": [float(m[by:by + h, bx - d].mean()) if bx - d >= 0 else 0.0 for d in range(34)],
        "right": [float(m[by:by + h, bx + w + d].mean()) if bx + w + d < m.shape[1] else 0.0
                  for d in range(34)],
        "top": [float(m[by - d, cols].mean()) if by - d >= 0 else 0.0 for d in range(34)],
        "bottom": [float(m[by + h + d, cols].mean()) if by + h + d < m.shape[0] else 0.0
                   for d in range(34)],
    }
    prof_slices = {}
    out = {}
    for name, p in sizes.items():
        out[name] = outward(p, name)
    # inward: the first rows/cols of the box, does the background cut into it?
    prof_slices["top"] = [float(m[by + i, cols].mean()) for i in range(min(34, h))]
    prof_slices["bottom"] = [float(m[by + h - 1 - i, cols].mean()) for i in range(min(34, h))]
    prof_slices["left"] = [float(m[rows, bx + i].mean()) for i in range(min(34, w))]
    prof_slices["right"] = [float(m[rows, bx + w - 1 - i].mean()) for i in range(min(34, w))]
    cuts = {name: inward(name) for name in ("top", "right", "bottom", "left")}

    edges = {}
    for name in ("top", "right", "bottom", "left"):
        if out[name] >= CUT:
            edges[name] = "tab"
        elif cuts[name] >= CUT:
            edges[name] = "notch"
        else:
            edges[name] = "flat"
    if verbose:
        print(f"  background {bg}")
        for name in ("top", "right", "bottom", "left"):
            print(f"  {name:6s} out {out[name]:2.0f}  cut {cuts[name]:2.0f}  -> {edges[name]}")
    return edges


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    mouse = Mouse()
    img = vision.grab()
    ref, board, _ = vision.layout(img)
    slots = [box for band in ("left", "right") for _r, _c, box in vision.lattice_slots(img, band)]
    occupied = [b for b in slots if not vision.is_empty(vision.sub(img, b))]
    print(f"tray: {len(occupied)} of {len(slots)} slots hold a piece")
    if not occupied:
        return 3

    todo = occupied if args.all else [occupied[args.slot % len(occupied)]]
    hw = int(vision.BOARD[2] / 5), int(vision.BOARD[3] / 5)
    blank_box = (BLANK[0] - hw[0] // 2, BLANK[1] - hw[1] // 2, hw[0], hw[1])
    print(f"holding each piece at {BLANK}, box {blank_box}")

    result: dict[str, dict] = {}
    for i, slot in enumerate(todo):
        mouse.grab_at(centre(slot))
        mouse.glide(BLANK)
        time.sleep(0.25)
        shot = mouse.shot()
        cv2.imwrite(str(SHOTS / f"edges_{i:02d}.png"), shot)
        print(f"\nslot ({slot[0]},{slot[1]})")
        edges = read_edges(shot, blank_box, verbose=True)
        result[f"{slot[0]},{slot[1]}"] = edges
        mouse.glide(centre(slot))
        mouse.release()
        time.sleep(0.2)

    for key, e in result.items():
        flats = sum(1 for v in e.values() if v == "flat")
        print(f"  ({key}) {e}  flats={flats}")
    if args.all:
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        flats = [sum(1 for v in e.values() if v == "flat") for e in result.values()]
        from collections import Counter
        print("\nflat-count histogram (a 5x5 game must show 4 corners=2, 12 edges=1, 9 inner=0):")
        print(f"  {Counter(flats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
