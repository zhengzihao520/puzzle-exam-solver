"""Read-only preflight: is the page a real game, and what does the vision actually say?

Nothing is dragged here.  It refuses to report classes on a page that is not ready (blank
reference panel, sparse tray), because feeding flat empty cells to the classifier produces
confident nonsense -- which is exactly what happened when 21 of 25 panel cells had std 0 and
the model answered "desktop" for all of them.

  python preflight.py --delay 10
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import cv2
import numpy as np

import vision
from solve_live import check_foreground, _enable_dpi

N = 5


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    global N
    N = args.grid
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    img = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "preflight.png"), img)
    ref, board, _ = vision.layout(img)
    print(f"reference panel {ref}\nboard {board}  cell {board.w / N:.0f}x{board.h / N:.0f}")

    cells = [vision.sub(img, b) for _r, _c, b in ref.cells(N)]
    stats = [(float(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean()),
              float(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).std())) for c in cells]
    filled = [s[1] > 15 for s in stats]
    print(f"reference cells with content: {sum(filled)}/{N * N}")

    rows = vision.tray_rows(img)
    piece_boxes = []
    whites = 0
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            cell = vision.sub(img, box)
            if vision.is_empty(cell):
                continue
            if vision.is_white_piece(cell):
                whites += 1
                continue
            piece_boxes.append((f"{band[0]}{ri + 1}{ci + 1}", box, cell))
    print(f"tray pieces: {len(piece_boxes)} (+{whites} white decoys)")

    if sum(filled) < 20 or len(piece_boxes) < 20:
        print("\nPAGE NOT READY -- refusing to report classes.")
        print("  the reference panel must show N*N photos and the tray must be full")
        return 3

    clf = vision.Classifier()
    need, need_conf = [], []
    for cell in cells:
        p = clf.probs(cell)
        need.append(str(clf.names[int(p.argmax())]))
        need_conf.append(float(p.max()))
    have, have_conf = [], []
    for _key, _box, cell in piece_boxes:
        p = clf.probs(cell)
        have.append(str(clf.names[int(p.argmax())]))
        have_conf.append(float(p.max()))

    print("\nwhat the reference asks for (rows 1-5, left to right):")
    for r in range(N):
        print("   " + " ".join(f"{need[r * N + c][:8]:8s}" for c in range(N)))
    print(f"   confidence: mean {np.mean(need_conf):.2f} min {min(need_conf):.2f}")

    print(f"\nwhat the tray holds ({len(have)} pieces):")
    for i in range(0, len(have), 8):
        print("   " + " ".join(f"{h[:8]:8s}" for h in have[i:i + 8]))
    print(f"   confidence: mean {np.mean(have_conf):.2f} min {min(have_conf):.2f}")

    a, b = Counter(need), Counter(have)
    keys = sorted(set(a) | set(b), key=lambda k: -(a[k] + b[k]))
    print(f"\n{'class':10s} {'needed':>7s} {'in tray':>8s}")
    for k in keys:
        flag = "" if a[k] == b[k] else "   <-- mismatch"
        print(f"  {k:10s} {a[k]:7d} {b[k]:8d}{flag}")
    l1 = sum(abs(a[k] - b[k]) for k in keys)
    print(f"\nmultiset L1 distance: {l1}  (0 = the classes agree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
