"""Live runner: drags planned pieces onto the board with real mouse events.

Usage:
  python run.py --dry               print the moves, touch nothing
  python run.py --ring              only the outer ring (16 cells)
  python run.py --ring --moves 4    stop after 4 moves
  python run.py                     whole plan

Safety and honesty rules
  * every move is verified from a fresh screenshot: the board cell must turn bright
    *and* the source tray slot must go flat dark; anything else counts as a failure
  * a failed move is dragged straight back to its slot, then the run stops -- pieces
    are never left lying on the board
  * pyautogui fail-safe stays on: slam the pointer into a screen corner to abort
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import vision

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs"


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


def reading(img: np.ndarray, clf: vision.Classifier, n: int = 5):
    ref = vision.find_reference(img)
    ref_cells = [vision.sub(img, box) for _r, _c, box in ref.cells(n)]
    pieces = []
    rows = vision.tray_rows(img)
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            cell = vision.sub(img, box)
            if not vision.is_empty(cell):
                pieces.append({"band": band, "row": ri, "col": ci,
                               "key": f"{band[0]}{ri + 1}{ci + 1}",
                               "box": box, "cell": cell})
    ref_probs, ref_txt = [], []
    for cell in ref_cells:
        if vision.is_blank(cell):
            ref_probs.append(None)
            ref_txt.append("blank")
        else:
            p = clf.probs(cell)
            ref_probs.append(p)
            ref_txt.append(str(clf.names[int(p.argmax())]))
    piece_probs, piece_txt = [], []
    for item in pieces:
        if vision.is_blank(item["cell"]):
            piece_probs.append(None)
            piece_txt.append("blank")
        else:
            p = clf.probs(item["cell"])
            piece_probs.append(p)
            piece_txt.append(str(clf.names[int(p.argmax())]))
    return ref, ref_probs, ref_txt, pieces, piece_probs, piece_txt


def make_plan(img: np.ndarray, clf: vision.Classifier, n: int = 5):
    ref, ref_probs, ref_txt, pieces, piece_probs, piece_txt = reading(img, clf, n)
    filled = vision.board_filled(img, n)
    if not pieces:
        raise SystemExit("no tray pieces detected")

    def score(ci: int, pi: int) -> float:
        if ref_txt[ci] == "blank" or piece_txt[pi] == "blank":
            return 1.0 if ref_txt[ci] == piece_txt[pi] else -1.0
        return float(np.dot(ref_probs[ci], piece_probs[pi]))

    pairs = sorted(((score(ci, pi), ci, pi)
                    for ci in range(len(ref_txt))
                    for pi in range(len(pieces))), reverse=True)
    cell_piece, taken = {}, set()
    for s, ci, pi in pairs:
        if s <= 0 or ci in cell_piece or pi in taken:
            continue
        cell_piece[ci] = pi
        taken.add(pi)

    board_cells = vision.layout(img)[1].cells(n)
    plan = []
    for idx in spiral(n):
        if filled[idx] or idx not in cell_piece:
            continue
        pi = cell_piece[idx]
        _r, _c, (bx, by, bw, bh) = board_cells[idx]
        plan.append({"idx": idx, "r": idx // n, "c": idx % n,
                     "want": ref_txt[idx], "got": piece_txt[pi],
                     "piece": pieces[pi],
                     "target": (bx + bw // 2, by + bh // 2),
                     "board_box": (bx, by, bw, bh)})
    unused = [pieces[i]["key"] for i in range(len(pieces)) if i not in taken]
    missing = [f"({i // n + 1},{i % n + 1})" for i in spiral(n)
               if not filled[i] and i not in cell_piece]
    return plan, unused, missing


class Mouse:
    def __init__(self, speed: float = 0.30, settle: float = 0.06) -> None:
        import pyautogui
        self.gui = pyautogui
        self.speed = speed
        self.settle = settle

    def shot(self) -> np.ndarray:
        return cv2.cvtColor(np.array(self.gui.screenshot()), cv2.COLOR_RGB2BGR)

    def drag(self, src, dst, probe: tuple[int, int, int, int] | None = None) -> bool:
        """Drag with a pickup check: abort (and report False) if the piece never lifted."""
        g = self.gui
        g.moveTo(int(src[0]), int(src[1]), duration=self.speed * 0.5)
        time.sleep(self.settle)
        g.mouseDown()
        time.sleep(self.settle * 2)
        # a short sideways tug proves the engine actually grabbed a piece
        g.moveTo(int(src[0]) + 18, int(src[1]) + 8, duration=0.12)
        time.sleep(self.settle)
        if probe is not None:
            lifted = cv2.cvtColor(vision.sub(self.shot(), probe), cv2.COLOR_BGR2GRAY)
            was = cv2.cvtColor(vision.sub(self._first, probe), cv2.COLOR_BGR2GRAY)
            if abs(float(lifted.mean()) - float(was.mean())) < 10 and float(lifted.std()) > 6:
                g.mouseUp()
                time.sleep(self.settle * 2)
                return False
        g.moveTo(int(src[0] + (dst[0] - src[0]) * 0.75), int(src[1] + (dst[1] - src[1]) * 0.75),
                 duration=self.speed)
        g.moveTo(int(dst[0]), int(dst[1]), duration=self.speed * 0.6)
        time.sleep(self.settle * 2)
        g.mouseUp()
        time.sleep(self.settle * 3)
        return True

    _first: np.ndarray | None = None


FULL_W, FULL_H = 1750, 950          # a real fullscreen page on the 1920x1080 screen


def check_foreground(marker: str = "鎷煎浘") -> tuple[bool, str]:
    """Only look, never switch: the page must already be the maximised foreground window."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, title, 512)
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if marker not in title.value:
        return False, f"{title.value!r} {w}x{h} -- abort: the exam page is not in the foreground"
    if w < FULL_W or h < FULL_H:
        return False, (f"{title.value!r} {w}x{h} -- abort: the window is not fullscreen, "
                       f"the layout would be wrong")
    return True, f"{title.value!r} {w}x{h}"

def centre(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[0] + box[2] // 2, box[1] + box[3] // 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--ring", action="store_true", help="outer ring cells only")
    ap.add_argument("--moves", type=int, default=0)
    ap.add_argument("--shot", default="", help="use a saved screenshot instead of the screen")
    args = ap.parse_args()

    clf = vision.Classifier()
    if not args.shot:
        ok, why = check_foreground()
        print(f"foreground: {why}")
        if not ok:
            return 2
    if args.shot:
        img = cv2.imread(args.shot)
        if img is None:
            raise SystemExit(f"cannot read {args.shot}")
    else:
        img = vision.grab()
    cv2.imwrite(str(LOG / "run_start.png") if LOG.is_dir() else str(ROOT / "shots" / "run_start.png"), img)

    plan, unused, missing = make_plan(img, clf)
    if args.ring:
        plan = [m for m in plan if m["r"] in (0, 4) or m["c"] in (0, 4)]
    if args.moves:
        plan = plan[:args.moves]

    print(f"plan: {len(plan)} moves   unused pieces: {len(unused)} {unused}")
    if missing:
        print(f"cells with no piece: {missing}")
    for m in plan:
        sx, sy = centre(m["piece"]["box"])
        print(f"  cell({m['r'] + 1},{m['c'] + 1}) need {m['want']:9s} <- "
              f"{m['piece']['key']:4s} {m['got']:9s} ({sx:4d},{sy:4d}) -> "
              f"({m['target'][0]:4d},{m['target'][1]:4d})")
    if args.dry or not plan:
        return 0

    print("\nstarting in 4s -- move the pointer into a screen corner to abort")
    time.sleep(4)
    mouse = Mouse()
    # wake the page: one harmless click on a cell that is still empty
    empty_cell = next((box for i, box in enumerate(
        [b for _r, _c, b in vision.layout(img)[1].cells(5)])
        if not vision.board_filled(img, 5)[i]), None)
    if empty_cell is not None:
        mouse.gui.click(empty_cell[0] + empty_cell[2] // 2, empty_cell[1] + empty_cell[3] // 2)
        time.sleep(0.3)
    done = 0
    for i, m in enumerate(plan, 1):
        key, want = m["piece"]["key"], m["want"]
        print(f"[{i}/{len(plan)}] {key} {m['got']} -> cell({m['r'] + 1},{m['c'] + 1}) {want}",
              flush=True)
        before = vision.grab()
        now_g = cv2.cvtColor(vision.sub(before, m["board_box"]), cv2.COLOR_BGR2GRAY)
        if float(now_g.mean()) >= vision.BOARD_FILLED_LEVEL:
            print("   cell already looks filled, skipping")
            continue
        src = centre(m["piece"]["box"])
        mouse._first = before
        picked = mouse.drag(src, m["target"], probe=m["piece"]["box"])
        after = vision.grab()
        if not picked:
            print("   pickup failed (the engine never grabbed the piece) -- retrying once")
            time.sleep(0.4)
            before = vision.grab()
            mouse._first = before
            picked = mouse.drag(src, m["target"], probe=m["piece"]["box"])
            after = vision.grab()
            if not picked:
                print("   pickup failed twice -- stopping, nothing was moved")
                return 1
        cell_ok = float(cv2.cvtColor(vision.sub(after, m["board_box"]),
                                     cv2.COLOR_BGR2GRAY).mean()) >= vision.BOARD_FILLED_LEVEL
        slot_ok = vision.is_empty(vision.sub(after, m["piece"]["box"]))
        if cell_ok and slot_ok:
            done += 1
            print("   ok: board cell filled, tray slot empty")
            continue
        print(f"   FAILED (cell_filled={cell_ok}, slot_empty={slot_ok}) -- dragging back")
        mouse.drag(m["target"], src)
        cv2.imwrite(str(ROOT / "shots" / "run_fail.png"), after)
        print(f"   stopped after {done} placement(s)")
        return 1
    after = vision.grab()
    cv2.imwrite(str(ROOT / "shots" / "run_end.png"), after)
    print(f"\nplaced {done}/{len(plan)} mov(es); board now "
          f"{sum(vision.board_filled(after, 5))}/25 filled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

