"""Measure the pointer -> piece transform.  No drops, nothing to snap, no progress change.

The failed placements are consistent with a fixed aim error of roughly (+27, +16) px, which
is outside the engine's 15.7 px snap radius.  Before blaming the aim, check the transform:

    hold a piece, move the pointer by a known delta, and see how far the piece moved.

If the ratio is not 1:1 the browser's bitmap space is not the screen's pixel space (page
zoom, DPI virtualisation, or a stretched canvas), and every absolute coordinate this project
uses needs that factor applied.

  python probe_map.py [--delay 5] [--dx 100] [--dy 100]
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"


def white_core(img: np.ndarray, near, radius: int = 160):
    """Bounding box of the bright core of the piece, expressed as (x, y, w, h) of its box."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x0, y0 = max(0, near[0] - radius), max(0, near[1] - radius)
    x1, y1 = min(g.shape[1], near[0] + radius), min(g.shape[0], near[1] + radius)
    m = (g[y0:y1, x0:x1] > 200).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        bx, by, bw, bh, area = (int(v) for v in stats[i][:5])
        if area < 3000:
            continue
        cx, cy = x0 + bx + bw / 2, y0 + by + bh / 2
        d = float(np.hypot(cx - near[0], cy - near[1]))
        if best is None or d < best[0]:
            best = (d, (x0 + bx, y0 + by, bw, bh))
    return None if best is None else best[1]


def largest_white(img: np.ndarray):
    """The biggest bright blob on the page: the piece we are holding stands on dark ground."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (g > 200).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))
    out = [tuple(int(v) for v in stats[i][:5]) for i in range(1, n) if 6000 < stats[i][4] < 40000]
    if not out:
        return None
    return max(out, key=lambda t: t[2] * t[3])


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--dx", type=int, default=100)
    ap.add_argument("--dy", type=int, default=100)
    args = ap.parse_args()
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground : {why}")
    if not ok:
        return 2

    mouse = Mouse()
    img = vision.grab()
    ref, board, _ = vision.layout(img)
    print(f"board      : ({board.x},{board.y}) {board.w}x{board.h}   cell {board.w / 5:.1f}x{board.h / 5:.1f}")
    rows = vision.tray_rows(img)
    slots = [(b, box) for b in ("left", "right") for _r, _c, box in vision.tray_slots(img, b, rows)]
    pieces = [(b, box) for b, box in slots if not vision.is_empty(vision.sub(img, box))]
    print(f"tray       : {len(pieces)} occupied")

    # prefer the whitest slot: its core is easy to isolate against any background
    def whiteness(box):
        return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())
    band, src = max(pieces, key=lambda t: whiteness(t[1]))
    print(f"piece      : {band} slot ({src[0]},{src[1]}) {src[2]}x{src[3]}  mean {whiteness(src):.0f}")

    start = centre(src)
    # a calm dark spot in the middle of the board: no cell boundary, nothing to snap
    place = (board.x + board.w // 2, board.y + board.h // 2 + 160)
    print(f"grab at    : {start}   hold over {place}")

    mouse.grab_at(start)
    mouse.glide(place)
    time.sleep(0.25)
    a = mouse.shot()
    cv2.imwrite(str(SHOTS / "map_a.png"), a)
    ba = white_core(a, place)

    p2 = (place[0] + args.dx, place[1])
    mouse.glide(p2)
    time.sleep(0.25)
    b = mouse.shot()
    cv2.imwrite(str(SHOTS / "map_b.png"), b)
    bb = white_core(b, p2)

    p3 = (p2[0], p2[1] + args.dy)
    mouse.glide(p3)
    time.sleep(0.25)
    c = mouse.shot()
    cv2.imwrite(str(SHOTS / "map_c.png"), c)
    bc = white_core(c, p3)

    print(f"bbox at {place}: {ba}")
    print(f"bbox at {p2}: {bb}")
    print(f"bbox at {p3}: {bc}")

    if ba and bb and bc:
        mx = bb[0] - ba[0]
        my = bc[1] - bb[1]
        print(f"\npointer moved ({args.dx:+d},0) -> piece moved ({mx:+d},0)   "
              f"ratio x {mx / args.dx:.3f}")
        print(f"pointer moved (0,{args.dy:+d}) -> piece moved (0,{my:+d})   "
              f"ratio y {my / args.dy:.3f}")
        if abs(mx - args.dx) <= 3 and abs(my - args.dy) <= 3:
            print("           the transform is 1:1 -- absolute coordinates are sound")
        else:
            print("           THE TRANSFORM IS NOT 1:1 -- every absolute coordinate needs scaling")
    else:
        print("\ncould not isolate the piece core in every frame (move it over a darker spot)")

    # return the piece to its slot and let go
    mouse.glide(start)
    time.sleep(0.2)
    mouse.release()
    time.sleep(0.3)
    final = mouse.shot()
    cv2.imwrite(str(SHOTS / "map_restored.png"), final)
    g = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    print(f"restored   : slot mean {float(g[src[1]:src[1] + src[3], src[0]:src[0] + src[2]].mean()):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
