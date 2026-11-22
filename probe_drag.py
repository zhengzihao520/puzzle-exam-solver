"""Where does a dragged piece actually land?  Measure the residual, not the snap.

The engine only locks a piece into *its own* cell (SOLUTIONS is per piece), so a random
piece dropped on a random cell is refused even when every gate is passed.  Testing the
mechanism by "did it snap" therefore proves nothing.

What can be tested with any piece is the geometry:

    residual = (bbox after the glide - bbox at the slot) - (cell top-left - slot top-left)

Both bboxes are of the piece's own drawn content, so the tab margins cancel exactly; a
zero residual means the piece box landed on the cell box, i.e. the engine's

    dist = hypot(piece.x - spot.x, piece.y - spot.y)   <   pieceWidth * 0.14

is satisfied to within the pixel.  Together with the pickup test (does mousedown hit the
box and set `dragging`) that covers two of the three gates.  The third, hasAnchor, is
satisfied by construction by aiming at a ring cell.

  python probe_drag.py [--delay 5] [--cell 0,0] [--piece 0] [--scan]
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def region_mean(img: np.ndarray, box) -> float:
    return float(gray(vision.sub(img, box)).mean())


def shifted(box, dx: int, dy: int) -> tuple[int, int, int, int]:
    return box[0] + dx, box[1] + dy, box[2], box[3]


def blob_at(img: np.ndarray, pt, radius: int = 110):
    """Bounding box of the drawn piece containing `pt`, or None.

    The piece's own art is bright against both the tray background and the dark board
    placeholder, so a plain brightness threshold isolates it.  The component containing
    the point is taken, which ignores neighbouring pieces and stray pixels.
    """
    h, w = gray(img).shape
    x0, y0 = max(0, pt[0] - radius), max(0, pt[1] - radius)
    x1, y1 = min(w, pt[0] + radius), min(h, pt[1] + radius)
    m = (gray(img)[y0:y1, x0:x1] > 40).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    lx, ly = pt[0] - x0, pt[1] - y0
    if not (0 <= lx < m.shape[1] and 0 <= ly < m.shape[0]):
        return None
    tag = lab[ly, lx]
    if tag == 0:
        # the point missed the art by a pixel or two: take the nearest big component
        best, tag = 0, 0
        for i in range(1, n):
            bx, by, bw, bh, area = (int(v) for v in stats[i][:5])
            if area < 2000:
                continue
            dx = max(bx - lx, 0, lx - (bx + bw))
            dy = max(by - ly, 0, ly - (by + bh))
            if best == 0 or dx + dy < best:
                best, tag = dx + dy, i
        if tag == 0:
            return None
    bx, by, bw, bh, _area = (int(v) for v in stats[tag][:5])
    return x0 + bx, y0 + by, bw, bh


def idle_columns(mouse: Mouse, frames: int = 3, wait: float = 0.35) -> set[int]:
    prev = vision.progress_crop(mouse.shot())
    ignore: set[int] = set()
    for _ in range(frames):
        time.sleep(wait)
        cur = vision.progress_crop(mouse.shot())
        ignore |= vision.ticking_columns(prev, cur)
        prev = cur
    return ignore


def pickup(mouse: Mouse, src, dx: int = 24, dy: int = 14):
    """Press on the slot centre, nudge, and report whether the piece followed."""
    p0 = centre(src)
    mouse.grab_at(p0)
    mouse.glide((p0[0] + dx, p0[1] + dy))
    time.sleep(0.2)
    held = mouse.shot()
    home = region_mean(held, src)
    follow = region_mean(held, shifted(src, dx, dy))
    return home, follow, held


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--cell", default="0,0")
    ap.add_argument("--piece", type=int, default=0)
    ap.add_argument("--scan", action="store_true", help="try nine grab offsets if the first misses")
    args = ap.parse_args()
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)

    ok, why = check_foreground()
    print(f"foreground : {why}")
    if not ok:
        return 2

    tr, tc = (int(v) for v in args.cell.split(","))
    mouse = Mouse()
    img = vision.grab()
    cv2.imwrite(str(SHOTS / "probe_start.png"), img)
    ref, board, _ = vision.layout(img)
    cells = board.cells(5)
    target = cells[tr * 5 + tc][2]
    pw, ph = board.w / 5, board.h / 5
    print(f"reference  : ({ref.x},{ref.y}) {ref.w}x{ref.h}")
    print(f"board      : ({board.x},{board.y}) {board.w}x{board.h}   piece {pw:.1f}x{ph:.1f}")
    print(f"target     : cell({tr},{tc}) at ({target[0]},{target[1]})")
    print(f"snap radius: {max(pw * 0.14, 10):.1f} px")

    rows = vision.tray_rows(img)
    slots = [(b, box) for b in ("left", "right") for _r, _c, box in vision.tray_slots(img, b, rows)]
    pieces = [(b, box) for b, box in slots if not vision.is_empty(vision.sub(img, box))]
    print(f"tray       : {len(slots)} slots, {len(pieces)} occupied   rows {rows}")
    if len(pieces) <= args.piece:
        print("ABORT: not enough pieces on the tray")
        return 3
    band, src = pieces[args.piece]
    print(f"piece      : {band} slot ({src[0]},{src[1]}) {src[2]}x{src[3]}")

    home, follow, held = pickup(mouse, src)
    picked = home < 30 and follow > 40
    print(f"\npickup     : home {home:5.1f}  follow {follow:5.1f}  -> "
          f"{'picked up (dragging set)' if picked else 'MISSED: mousedown did not hit the box'}")

    if not picked and args.scan:
        print("           scanning grab offsets ...")
        for dy in (-20, 0, 20):
            for dx in (-20, 0, 20):
                if (dx, dy) == (0, 0):
                    continue
                p = centre(src)
                mouse.release()
                mouse.gui.moveTo(*mouse._safe(p))
                time.sleep(0.1)
                mouse.grab_at((p[0] + dx, p[1] + dy))
                mouse.glide((p[0] + dx + 24, p[1] + dy + 14))
                time.sleep(0.18)
                s = mouse.shot()
                h_ = region_mean(s, src)
                f_ = region_mean(s, shifted(src, 24, 14))
                print(f"             offset {dx:+3d},{dy:+3d}: home {h_:5.1f} follow {f_:5.1f}"
                      f"{'   <-- hits' if h_ < 30 and f_ > 40 else ''}")
                mouse.glide(p)
                mouse.release()
                time.sleep(0.2)

    # put it back where it started, then re-read the slot so the two bboxes line up
    if picked:
        mouse.glide(centre(src))
    mouse.release()
    time.sleep(0.25)
    fresh = mouse.shot()
    at_slot = blob_at(fresh, centre(src))
    print(f"bbox@slot  : {at_slot}")

    src_c = centre(src)
    mouse.grab_at(src_c)
    dst = centre(target)
    mouse.glide(dst)
    time.sleep(0.3)
    hover = mouse.shot()
    cv2.imwrite(str(SHOTS / "probe_hover.png"), hover)
    at_target = blob_at(hover, dst)
    print(f"bbox@target: {at_target}")

    if at_slot and at_target:
        dx = (at_target[0] - at_slot[0]) - (target[0] - src[0])
        dy = (at_target[1] - at_slot[1]) - (target[1] - src[1])
        dist = float(np.hypot(dx, dy))
        print(f"residual   : dx {dx:+d}  dy {dy:+d}  |{dist:.1f}| px  "
              f"(piece box vs cell box)")
        print(f"             {'LANDS ON THE CELL' if dist < 8 else 'OFF THE CELL by %.1f px' % dist}")
    else:
        print("residual   : could not isolate the piece blob in one of the two frames")
        dx = dy = 0

    mouse.release()
    time.sleep(0.4)
    dropped = mouse.shot()
    cv2.imwrite(str(SHOTS / "probe_dropped.png"), dropped)
    print(f"drop       : cell mean {region_mean(dropped, target):5.1f}  "
          f"(placeholder ~25, a piece ~70+)")

    ignore = idle_columns(mouse)
    baseline = vision.progress_crop(mouse.shot())
    changed = False
    for _ in range(8):
        if vision.progress_changed(baseline, vision.progress_crop(mouse.shot()), ignore):
            changed = True
            break
        time.sleep(0.35)
    print(f"progress   : {'changed' if changed else 'unchanged'}  "
          f"(expected: unchanged, this piece does not belong to that cell)")

    # leave the board clean: pick the piece back up and return it to its slot
    mouse.grab_at(dst)
    mouse.glide((dst[0] + 40, dst[1] + 30))
    time.sleep(0.2)
    if region_mean(mouse.shot(), target) < 35:
        print("relift     : the piece left the cell (loose, as expected)")
        mouse.glide(src_c)
        mouse.release()
    else:
        print("relift     : the piece is locked in -- it was the cell's own piece")
        mouse.glide(dst)
        mouse.release()
    time.sleep(0.3)
    print(f"restored   : slot mean {region_mean(mouse.shot(), src):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
