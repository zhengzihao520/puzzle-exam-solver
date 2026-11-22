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
# The cleanest background available is an empty board cell: its placeholder is flat dark,
# and the cell separators sit exactly on the piece box edges, ~19 px away from where a tab
# reaches.  Parking the piece on the canvas below the tray looked blank but overlapped the
# tray's slot borders, which showed up as a 5-6 px "protrusion" on sides that are flat.
BLANK = (713, 666)          # centre of board cell (2,2); replaced by an empty cell at run time
CUT = 10                    # protrusion/notch depth that counts as real: a tab reads ~15,
CUT = 10                    # unused depth threshold, kept so old logs make sense
SHAPE_CUT = 0.35            # middle-vs-ends coverage gap that counts as tab / notch

def bg_colour(img: np.ndarray, box, pad: int = 34) -> np.ndarray:
    """What the piece is standing on, sampled from the four window corners.

    A ring around the box is not usable: the tabs reach 19 px into it and pull the median
    towards the piece itself (measured: 51,26,20 instead of the board's 23,23,23), which
    then makes flat sides look like they protrude.
    """
    x, y, w, h = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    k = max(4, pad // 3)
    patches = [img[y0:y0 + k, x0:x0 + k], img[y0:y0 + k, x1 - k:x1],
               img[y1 - k:y1, x0:x0 + k], img[y1 - k:y1, x1 - k:x1]]
    patches = [p.reshape(-1, 3) for p in patches if p.size]
    if not patches:
        return np.median(img[y:y + h, x:x + w].reshape(-1, 3), axis=0)
    return np.median(np.concatenate(patches), axis=0)


def piece_mask(img: np.ndarray, box, pad: int = 28, tol: int = 16):
    """Binary silhouette: pixels whose colour differs from what the piece stands on.

    A gradient-based silhouette was tried on the twenty-five real captures and looked
    better there (8 impossible readings down to 1), but it fails badly on synthetic pieces
    with crisp edges (58% of sides right versus 100% here) because a photo's own texture
    also produces strong edges.  This mask keeps the behaviour that is measured at 800/800
    on ground truth; the eight real pieces it cannot read are reported as *unknown* rather
    than guessed, and jigsaw.solve_soft accepts unknown sides.

    Known limitation: the engine draws a ~20 px drop shadow under each piece, and a dark
    photo sits in the same brightness band, so on those pieces the shadow merges with the
    piece and the tabs are lost.  That is why the caller rejects impossible readings.
    """
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


def side_profile(m: np.ndarray, bx: int, by: int, w: int, h: int,
                 side: str, outward: bool, depth: int = 24):
    """(depth+1) x side_length of mask values stepping out of (or into) one side.

    d = 0 is the first pixel just outside the box (outward) or the box edge itself
    (inward).  The four sides need four different offsets: measuring the right side with
    the left side's formula samples *inside* the piece, where everything is covered, and
    that silently reported zero for every tab on the right and the bottom.
    """
    rows = []
    for d in range(depth + 1):
        if side == "top":
            yy = by - 1 - d if outward else by + d
            if not (0 <= yy < m.shape[0]):
                rows.append(np.zeros(w))
                continue
            rows.append(m[yy, bx:bx + w].astype(float))
        elif side == "bottom":
            yy = by + h + d if outward else by + h - 1 - d
            if not (0 <= yy < m.shape[0]):
                rows.append(np.zeros(w))
                continue
            rows.append(m[yy, bx:bx + w].astype(float))
        elif side == "left":
            xx = bx - 1 - d if outward else bx + d
            if not (0 <= xx < m.shape[1]):
                rows.append(np.zeros(h))
                continue
            rows.append(m[by:by + h, xx].astype(float))
        else:                                   # right
            xx = bx + w + d if outward else bx + w - 1 - d
            if not (0 <= xx < m.shape[1]):
                rows.append(np.zeros(h))
                continue
            rows.append(m[by:by + h, xx].astype(float))
    return np.array(rows)


def middle_vs_ends(row: np.ndarray) -> float:
    """How much more of the middle of a side is covered than its two ends."""
    n = len(row)
    if n < 9:
        return 0.0
    third = n // 3
    mid = row[third:n - third].mean()
    ends = np.concatenate([row[:third], row[n - third:]]).mean()
    return float(mid - ends)


def read_edges(img: np.ndarray, box, verbose: bool = False) -> dict[str, str]:
    """{'top','right','bottom','left'} -> 'tab' | 'notch' | 'flat'.

    The discriminating feature is *where* the extra material sits, not how far it reaches.
    A tab is a circle centred on the middle of a side, so it covers the middle of the side
    and none of its ends; a notch is the same but cut inward, so the background covers the
    middle and the piece covers the ends.  Comparing the middle third against the two end
    thirds cancels the uniform blur that made an absolute depth test unreliable (flat sides
    measured 6 px, the weakest tab 11 px -- far too close).
    """
    x, y, w, h = box
    m, (x0, y0), bg = piece_mask(img, box)
    bx, by = x - x0, y - y0

    tab_score, notch_score = {}, {}
    for side in ("top", "right", "bottom", "left"):
        out_prof = side_profile(m, bx, by, w, h, side, outward=True)
        in_prof = side_profile(m, bx, by, w, h, side, outward=False)
        tab_score[side] = max((middle_vs_ends(out_prof[d]) for d in range(1, len(out_prof))),
                              default=0.0)
        # inward the background is what "bulges": mask 0 in the middle of the side
        notch_score[side] = max((middle_vs_ends(1.0 - in_prof[d]) for d in range(1, len(in_prof))),
                                default=0.0)

    edges: dict[str, str | None] = {}
    for side in ("top", "right", "bottom", "left"):
        if tab_score[side] >= SHAPE_CUT:
            edges[side] = "tab"
        elif notch_score[side] >= SHAPE_CUT:
            edges[side] = "notch"
        else:
            edges[side] = "flat"

    # Self-check on the reading itself: no piece has three flat sides (a corner has two, a
    # frame piece one, an interior piece none, a decoy none).  Three or more therefore means
    # the silhouette was misread -- almost always a dark photo whose tabs merge with the
    # engine's drop shadow.  Report those as unknown so the solver treats them as free
    # instead of believing a wrong cut.
    if sum(1 for v in edges.values() if v == "flat") >= 3:
        if verbose:
            print("  reading rejected: three or more flat sides")
        edges = {side: None for side in ("top", "right", "bottom", "left")}

    if verbose:
        print(f"  background {bg}")
        for side in ("top", "right", "bottom", "left"):
            print(f"  {side:6s} tab {tab_score[side]:.2f}  notch {notch_score[side]:.2f}"
                  f"  -> {edges[side]}")
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
    # Hold the piece in the emptiest board cell *closest to the board centre*.  Distance
    # matters: a corner cell's sampling window (box +/- 34 px) reaches past the board edge
    # into the tray bands, where the neighbours are photos, and the background estimate
    # came back as (51,42,40) instead of the board's (23,23,23) -- which made all four
    # sides read as tabs.
    bx0, by0 = board.x + board.w / 2, board.y + board.h / 2
    empties = [(float(cv2.cvtColor(vision.sub(img, b), cv2.COLOR_BGR2GRAY).mean()),
                float((centre(b)[0] - bx0) ** 2 + (centre(b)[1] - by0) ** 2), r, c, b)
               for r, c, b in board.cells(5)]
    empties = [e for e in empties if e[0] < 45] or [min(empties)]
    empties.sort(key=lambda t: (t[1], t[0]))
    spot = centre(empties[0][4])
    blank_box = (spot[0] - hw[0] // 2, spot[1] - hw[1] // 2, hw[0], hw[1])
    print(f"holding each piece at {spot} (empty board cell), box {blank_box}")

    result: dict[str, dict] = {}
    missed: list[str] = []
    for i, slot in enumerate(todo):
        # Prove the pickup before measuring.  A press that lands on nothing leaves the slot
        # full, and the "silhouette" is then whatever is at the blank spot -- all four sides
        # read flat, which is impossible, and eight of twenty-five readings came back that
        # way before this check existed.
        mouse.grab_at(centre(slot))
        sx, sy = centre(slot)
        mouse.glide((sx + 96, sy + 74))
        time.sleep(0.18)
        if not vision.is_empty(vision.sub(mouse.shot(), slot)):
            mouse.release()
            print(f"\nslot ({slot[0]},{slot[1]}): nothing there -- skipped")
            missed.append(f"{slot[0]},{slot[1]}")
            continue
        mouse.glide(spot)
        time.sleep(0.25)
        shot = mouse.shot()
        cv2.imwrite(str(SHOTS / f"edges_{i:02d}.png"), shot)
        print(f"\nslot ({slot[0]},{slot[1]})")
        edges = read_edges(shot, blank_box, verbose=True)
        result[f"{slot[0]},{slot[1]}"] = edges
        mouse.glide(centre(slot))
        mouse.release()
        time.sleep(0.2)
    if missed:
        print(f"\n{len(missed)} slot(s) read as occupied but held nothing: {', '.join(missed)}")

    for key, e in result.items():
        flats = sum(1 for v in e.values() if v == "flat")
        flag = "   <-- impossible: no piece has three flat sides" if flats >= 3 else ""
        print(f"  ({key}) {e}  flats={flats}{flag}")
    if args.all:
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        from collections import Counter
        flats = [sum(1 for v in e.values() if v == "flat") for e in result.values()]
        hist = Counter(flats)
        print("\nflat-count histogram.  A 5x5 game must show, over its 25 real pieces:")
        print("  4 with 2 flat sides (corners), 12 with 1 (edge cells), 9 with 0 (interior)")
        print("  and every decoy piece has 0 (its four sides are all +/-1).")
        print(f"  measured: {dict(sorted(hist.items()))}")
        bad = hist.get(3, 0) + hist.get(4, 0)
        print(f"  impossible readings: {bad}")
        if hist.get(2, 0) and hist.get(1, 0):
            print(f"  corners read {hist.get(2, 0)}/4, edges {hist.get(1, 0)}/12 "
                  f"(the rest are hidden under a wrong reading or a decoy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
