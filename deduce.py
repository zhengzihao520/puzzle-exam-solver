"""Know first, place once.

The reference panel gives the class of every cell but shows a *different* photo of that
class, so two pieces of the same class cannot be told apart by picture.  What tells them
apart is the jigsaw shape: a corner cell needs a piece with exactly the two board flats,
an edge cell one, an inner cell none, and neighbouring inner edges must interlock.

Shape cannot be read while a piece sits in the tray: slots are 23 px apart and tabs reach
18 px, so a neighbour's tab lands inside the measuring strip.  Phase A therefore lifts
each piece, holds it over the empty canvas below the board, measures it there, and puts it
back -- a measurement, not a trial.

  python deduce.py --measure    phase A: read every piece's four sides, save JSON
  python deduce.py --solve      phase B: build the full piece -> cell assignment
  python deduce.py --place      phase C: place each piece once, in order
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np

import vision
from solve_live import Mouse, check_foreground, _enable_dpi

ROOT = vision.ROOT
N = 5
SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
SIG_FILE = ROOT / "logs" / "piece_shapes.json"
PARK = (760, 985)                # empty canvas below the board, verified at run time
BODY_W, BODY_H = 106, 84


def tray_slots_from_full(shot: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    img = cv2.imread(str(shot))
    rows = vision.tray_rows(img)
    out = []
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            out.append((f"{band[0]}{ri + 1}{ci + 1}", box))
    return out


def side_of(mask: np.ndarray, side: str, body: tuple[int, int]) -> str:
    """flat / tab / notch for one side of an isolated piece silhouette.

    mask == 1 means "piece pixel".  A tab shows piece pixels *outside* the body edge in
    the middle third of that side; a notch shows background *inside* it; a flat side
    shows neither.
    """
    h, w = mask.shape
    bw, bh = body
    x0, y0 = int((w - bw) / 2), int((h - bh) / 2)
    x1, y1 = x0 + bw, y0 + bh
    strip = 12
    mid_x = slice(x0 + bw // 3, x0 + 2 * bw // 3)
    mid_y = slice(y0 + bh // 3, y0 + 2 * bh // 3)
    if side == "top":
        outside = mask[max(0, y0 - strip):y0, mid_x]
        inside = mask[y0:y0 + strip, mid_x]
    elif side == "bottom":
        outside = mask[y1:min(h, y1 + strip), mid_x]
        inside = mask[max(0, y1 - strip):y1, mid_x]
    elif side == "left":
        outside = mask[mid_y, max(0, x0 - strip):x0]
        inside = mask[mid_y, x0:x0 + strip]
    else:
        outside = mask[mid_y, x1:min(w, x1 + strip)]
        inside = mask[mid_y, max(0, x1 - strip):x1]
    piece_out = float(np.asarray(outside, float).mean()) if outside.size else 0.0
    piece_in = float(np.asarray(inside, float).mean()) if inside.size else 1.0
    if piece_out > 0.5:
        return "tab"
    if piece_in < 0.5:
        return "notch"
    return "flat"


def silhouette(shot: np.ndarray, centre_pt: tuple[int, int],
               half: tuple[int, int] = (BODY_W // 2 + 24, BODY_H // 2 + 24)) -> np.ndarray:
    hw, hh = half
    x0, y0 = max(0, centre_pt[0] - hw), max(0, centre_pt[1] - hh)
    win = cv2.cvtColor(shot[y0:centre_pt[1] + hh, x0:centre_pt[0] + hw],
                       cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = float(np.median(win))
    return (win > bg + 18).astype(np.uint8)


def measure(mouse: Mouse, box) -> dict[str, str] | None:
    """Lift one piece over the empty canvas and read its four sides there."""
    src = (box[0] + box[2] // 2, box[1] + box[3] // 2)
    mouse.grab_at(src)
    mouse.glide(PARK)
    time.sleep(mouse.settle * 2)
    shot = mouse.shot()
    mask = silhouette(shot, PARK)
    sig = {s: side_of(mask, s, (BODY_W, BODY_H)) for s in SIDES}
    mouse.glide(src)
    mouse.release()
    return sig


def phase_measure(slots_shot: Path) -> int:
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2
    img = vision.grab()
    park_g = cv2.cvtColor(vision.sub(img, (PARK[0] - 80, PARK[1] - 70, 160, 140)),
                          cv2.COLOR_BGR2GRAY)
    if float(park_g.std()) > 12:
        print(f"ABORT: the parking area under the board is not empty "
              f"(mean {park_g.mean():.0f}, std {park_g.std():.0f}); reset the game first")
        return 3
    slots = tray_slots_from_full(slots_shot)
    mouse = Mouse()
    out = {}
    for key, box in slots:
        if vision.is_empty(vision.sub(img, box)):
            continue
        sig = measure(mouse, box)
        if sig is None:
            continue
        out[key] = {"box": list(box), "sides": sig,
                    "flats": sorted(s for s, k in sig.items() if k == "flat")}
        print(f"  {key}: {sig}  flats={out[key]['flats']}")
    SIG_FILE.parent.mkdir(exist_ok=True)
    SIG_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    flats = Counter(len(v["flats"]) for v in out.values())
    print(f"\n{len(out)} pieces measured -> {SIG_FILE}")
    print(f"flat-count histogram: {dict(sorted(flats.items()))} (5x5 wants 2:4, 1:12, 0:9)")
    return 0


def cell_pattern(r: int, c: int) -> frozenset[str]:
    pat = set()
    if r == 0:
        pat.add("top")
    if r == N - 1:
        pat.add("bottom")
    if c == 0:
        pat.add("left")
    if c == N - 1:
        pat.add("right")
    return frozenset(pat)


def phase_solve(shot: Path | None) -> int:
    sig = json.loads(SIG_FILE.read_text(encoding="utf-8"))
    img = vision.grab() if shot is None else cv2.imread(str(shot))
    clf = vision.Classifier()
    ref, board, _ = vision.layout(img)
    need = []
    for cell in [vision.sub(img, b) for _r, _c, b in ref.cells(N)]:
        need.append("blank" if vision.is_blank(cell) else
                    str(clf.names[int(clf.probs(cell).argmax())]))
    label = {}
    for key, v in sig.items():
        cell = vision.sub(img, tuple(v["box"]))
        p = clf.probs(cell)
        label[key] = "blank" if vision.is_white_piece(cell) else str(clf.names[int(p.argmax())])

    print("class balance")
    n, h = Counter(need), Counter(label.values())
    for k in sorted(set(n) | set(h), key=lambda k: -n[k]):
        print(f"  {k:10s} need {n[k]:3d}  have {h[k]:3d}{'   <-- mismatch' if n[k] != h[k] else ''}")

    order = sorted(range(N * N), key=lambda i: (-len(cell_pattern(i // N, i % N)), i))
    assign: dict[int, str] = {}
    used: set[str] = set()
    for idx in order:
        pat = cell_pattern(idx // N, idx % N)
        cands = [k for k, v in sig.items() if k not in used and frozenset(v["flats"]) == pat]
        same = [k for k in cands if label[k] == need[idx]]
        pick = (same or cands or [None])[0]
        if pick is None:
            print(f"  cell({idx // N + 1},{idx % N + 1}) {sorted(pat)} want {need[idx]}: "
                  f"no shape-matching piece left")
            continue
        assign[idx] = pick
        used.add(pick)
        tag = "" if label[pick] == need[idx] else f"  (class {label[pick]} vs {need[idx]})"
        print(f"  cell({idx // N + 1},{idx % N + 1}) {sorted(pat) or 'inner'} "
              f"<- {pick} {sorted(sig[pick]['flats'])}{tag}")
    missing = [f"({i // N + 1},{i % N + 1})" for i in range(N * N) if i not in assign]
    print(f"\nassigned {len(assign)}/{N * N}; unassigned: {missing}")
    print(f"pieces unused: {[k for k in sig if k not in used]}")
    (ROOT / "logs" / "deduced_plan.json").write_text(
        json.dumps({"assign": {str(k): v for k, v in assign.items()}, "need": need},
                   indent=2), encoding="utf-8")
    return 0


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--slots", default="shots/s1.png")
    ap.add_argument("--shot", default="")
    args = ap.parse_args()
    if args.measure:
        return phase_measure(ROOT / args.slots)
    if args.solve:
        return phase_solve(Path(args.shot) if args.shot else None)
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
