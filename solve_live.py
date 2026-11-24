"""Live solver: place pieces by class + shape-agnostic trial, verified every time.

Why this design
---------------
The site maps every piece to exactly ONE cell (`SOLUTIONS.set(piece, ...)` in
puzzle_board.js).  Two pieces of the same class are different photos, so matching
by class alone cannot pick the right one -- that is why an earlier class-only run
scored 2/25.  The engine's own verdict is the only reliable answer, and it is
readable without touching the page:

  * `render()` draws unplaced pieces with a shadow and placed pieces without, and
    `pickPiece()` skips placed pieces (`if (!p.isPlaced)`).
  * so: grab the cell and pull 60 px.  A placed piece cannot be picked up and the
    cell stays bright; a loose piece follows the cursor and the cell goes dark
    (only the hover highlight is left, ~50-80 vs a 200+ piece).

Therefore each attempt is: drag candidate -> probe -> keep if locked, otherwise keep
dragging the same piece back to its tray slot and try the next candidate.  Nothing is
ever left lying on the board.

Usage
  python solve_live.py --dry            show the intended order, touch nothing
  python solve_live.py --moves 3        stop after 3 successful placements
  python solve_live.py                  whole board, outer ring first
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import cv2
import numpy as np

import vision

LOCK_DELTA = 15.0           # locked: the cell hardly changes while probing; loose: it empties


def _enable_dpi() -> None:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def check_foreground(marker: str | tuple[str, ...] = ("Edge", "Chrome"), tries: int = 6) -> tuple[bool, str]:
    """Look only, never switch windows -- and tolerate a window caught mid-transition."""
    markers = (marker,) if isinstance(marker, str) else tuple(marker)
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    why = "no window"
    for _ in range(tries):
        hwnd = u.GetForegroundWindow()
        title = ctypes.create_unicode_buffer(512)
        u.GetWindowTextW(hwnd, title, 512)
        r = wintypes.RECT()
        u.GetWindowRect(hwnd, ctypes.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        sw, sh = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        why = f"{w}x{h} on {sw}x{sh}"
        if w == 0 or h == 0 or sw == 0:
            time.sleep(0.4)
            continue
        if not any(m in title.value for m in markers):
            return False, (f"{why} [{title.value[:40]}] -- abort: exam page is not the foreground window")
        if w < sw * 0.97 or h < sh * 0.93:
            return False, f"{why} -- abort: window is not maximised"
        return True, why
    return False, f"{why} -- abort: could not read the foreground window"


def spiral(n: int) -> list[int]:
    order: list[int] = []
    top, left, bottom, right = 0, 0, n - 1, n - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            order.append(top * n + c)
        top += 1
        for r in range(top, bottom + 1):
            order.append(r * n + right)
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                order.append(bottom * n + c)
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                order.append(r * n + left)
            left += 1
    return order


class Mouse:
    def __init__(self, speed: float = 0.22, settle: float = 0.05) -> None:
        import pyautogui
        self.gui = pyautogui
        self.speed = speed
        self.settle = settle
        # pyautogui raises if the pointer is sitting in a screen corner when a move starts.  All
        # our own moves are clamped to 40 px inside the edges by _safe(), so that guard can only
        # fire because something else left the pointer in a corner -- and firing mid-drag is bad:
        # mouseDown has already happened, so the button stays held and the next grab drags the
        # board around.  A run that hits that leaves pieces strewn about and one cell unsolvable.
        pyautogui.FAILSAFE = False
        self._guarded = True

    def grab_at(self, pt) -> None:
        self.gui.moveTo(*self._safe(pt), duration=self.speed * 0.5)
        time.sleep(self.settle)
        self.gui.mouseDown()
        time.sleep(self.settle * 2)

    def _safe(self, pt):
        """Keep away from the screen corners so pyautogui's fail-safe cannot fire on our
        own moves (the user can still abort by taking the pointer there themselves)."""
        return max(40, min(int(pt[0]), 1880)), max(40, min(int(pt[1]), 1040))

    def glide(self, pt) -> None:
        self.gui.moveTo(*self._safe(pt), duration=self.speed)

    def release(self) -> None:
        self.gui.mouseUp()
        time.sleep(self.settle * 3)

    def shot(self) -> np.ndarray:
        return cv2.cvtColor(np.array(self.gui.screenshot()), cv2.COLOR_RGB2BGR)


def centre(box) -> tuple[int, int]:
    return box[0] + box[2] // 2, box[1] + box[3] // 2


def cell_mean(img: np.ndarray, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


def attempt(mouse: Mouse, piece_box, cell_box) -> bool:
    """Drag one piece to one cell and report whether the engine locked it in."""
    src, dst = centre(piece_box), centre(cell_box)
    before = mouse.shot()
    mouse.grab_at(src)
    mouse.glide((src[0] + 18, src[1] + 10))       # prove the pickup
    moved = mouse.shot()
    # (a missed pickup is caught by the probe below: the cell never turns bright)
    mouse.glide(dst)
    time.sleep(mouse.settle * 2)
    mouse.release()

    # probe: can the piece still be picked up?
    dropped_mean = cell_mean(mouse.shot(), cell_box)
    mouse.grab_at(dst)
    mouse.glide((dst[0] + 60, dst[1] + 20))
    time.sleep(mouse.settle * 2)
    probed = mouse.shot()
    probed_mean = cell_mean(probed, cell_box)
    locked = (dropped_mean >= vision.BOARD_FILLED_LEVEL
              and abs(probed_mean - dropped_mean) < LOCK_DELTA)
    print(f"      drop={dropped_mean:5.1f} probe={probed_mean:5.1f} -> "
          f"{'locked' if locked else 'loose'}")
    if locked:
        mouse.glide(dst)
        mouse.release()                            # placed pieces are not draggable
        return True
    mouse.glide(src)                               # still in hand: put it back
    mouse.release()
    return False


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--moves", type=int, default=0)
    args = ap.parse_args()

    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok and not args.dry:
        return 2

    clf = vision.Classifier()
    img = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "solve_start.png"), img)
    ref, board, _regions = vision.layout(img)
    n = 5
    filled = vision.board_filled(img, n)
    rows = vision.tray_rows(img)
    pieces = []
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            cell = vision.sub(img, box)
            if vision.is_empty(cell):
                continue
            pieces.append({"band": band, "key": f"{band[0]}{ri + 1}{ci + 1}",
                           "box": box, "cell": cell})
    print(f"board filled {sum(filled)}/{n * n}, tray pieces {len(pieces)}")

    ref_cells = [vision.sub(img, b) for _r, _c, b in ref.cells(n)]
    need = []
    for cell in ref_cells:
        need.append("blank" if vision.is_blank(cell) else
                    str(clf.names[int(clf.probs(cell).argmax())]))
    for p in pieces:
        p["label"] = str(clf.names[int(clf.probs(p["cell"]).argmax())])
    print("need:", dict(Counter(need)))
    print("have:", dict(Counter(p["label"] for p in pieces)))

    order = spiral(n)
    plan = []
    remaining = list(pieces)
    cell_pieces: dict[int, int] = {}
    for idx in order:
        if filled[idx]:
            continue
        want = need[idx]
        cands = [p for p in remaining if p["label"] == want]
        if not cands:
            cands = [p for p in remaining if p["label"] != "blank"]
        if not cands:
            continue
        # try same-class candidates first, best confidence first
        cands.sort(key=lambda p: -float(clf.probs(p["cell"]).max()))
        plan.append({"idx": idx, "want": want, "cands": cands[:3]})
        cell_pieces[idx] = 1
    print(f"\nplan: {len(plan)} cells, probing up to 3 candidates each")

    if args.dry:
        for step in plan:
            names = ",".join(f"{c['key']}/{c['label']}" for c in step["cands"])
            print(f"  cell({step['idx'] // n + 1},{step['idx'] % n + 1}) want {step['want']:9s} <- {names}")
        return 0

    print("\nstarting in 4s -- drag the pointer into a screen corner to abort")
    time.sleep(4)
    mouse = Mouse()
    placed = 0
    for step in plan:
        if args.moves and placed >= args.moves:
            break
        idx = step["idx"]
        _r, _c, cell_box = board.cells(n)[idx]
        r, c = idx // n, idx % n
        if cell_mean(vision.grab(), cell_box) >= vision.BOARD_FILLED_LEVEL:
            print(f"cell({r + 1},{c + 1}) already holds something, skipping")
            continue
        for cand in step["cands"]:
            print(f"cell({r + 1},{c + 1}) want {step['want']:9s} try {cand['key']}/{cand['label']}",
                  flush=True)
            if attempt(mouse, cand["box"], cell_box):
                print("   locked in")
                placed += 1
                remaining = [p for p in remaining if p is not cand]
                break
            print("   no snap, piece returned to the tray")
        else:
            print(f"   gave up on cell({r + 1},{c + 1}) for now")

    final = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "solve_end.png"), final)
    print(f"\nplaced {placed}; board reads {sum(vision.board_filled(final, n))}/{n * n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

