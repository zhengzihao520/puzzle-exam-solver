"""One run: read the board, read every piece, solve the assignment, then place it.

The four things this relies on, all measured in this repo:

  1. Cells can only be named by *class*: the reference panel shows a different photo of the
     same class for every cell (recon/catalog.js alternatePhoto).
  2. Pieces of one class are told apart by their *cut*: the engine's edge matrix is globally
     consistent, so class + four edge signs per piece pins the assignment down
     (jigsaw.solve_soft: 95.7% per piece on ambiguous cells, 91% with 5% edge noise).
  3. Edges can only be read while a piece is isolated: in the tray they sit 13-18 px apart,
     less than a tab (19 px).  So each piece is lifted, parked on an empty board cell, read,
     and put back -- one move per piece.  Readings that break the rules (three flat sides)
     come back as unknown rather than wrong.
  4. A drop only sticks when the piece's box lands within 15.7 px of its own cell, so grab at
     the slot centre and aim at the cell centre (residual 0-6 px measured); then ask the
     engine whether it took: a placed piece cannot be picked up again.  That probe is free
     when it passes.

  python solve_run.py --dry       # read-only: identify everything, print the plan
  python solve_run.py --go        # do it
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import cv2

import jigsaw
import probe_edges as pe
import vision
from identify import BLANK, Identifier
from solve_live import Mouse, _enable_dpi, check_foreground, centre

SHOTS = vision.ROOT / "shots"
LOG = vision.ROOT / "logs" / "solve_run.json"
SIDES = ("top", "right", "bottom", "left")
EDGE_SIGN = {"tab": 1, "notch": -1, "flat": 0, None: None}
N = 5
PROBE_POINTS = ((0, 0), (-30, -20), (30, 20), (-30, 20), (30, -20))


def order_ring(n: int) -> list[tuple[int, int]]:
    out = []
    for k in range((n + 1) // 2):
        top, bottom, left, right = k, n - 1 - k, k, n - 1 - k
        if top == bottom:
            out.extend((top, c) for c in range(left, right + 1))
            continue
        out.extend((top, c) for c in range(left, right))
        out.extend((r, right) for r in range(top, bottom))
        out.extend((bottom, c) for c in range(right, left, -1))
        out.extend((r, left) for r in range(bottom, top, -1))
    return out


def cell_mean(img, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


class Run:
    def __init__(self, mouse: Mouse) -> None:
        self.mouse = mouse
        self.img = mouse.shot()
        self.ref, self.board, _ = vision.layout(self.img)
        self.cells = {f"{r},{c}": b for r, c, b in self.board.cells(N)}
        # the reference panel and the board share cell keys but are different rectangles:
        # the panel says *what* belongs in a cell, the board is *where* it goes
        self.panel = {f"{r},{c}": b for r, c, b in self.ref.cells(N)}
        self.slots, self.rows, self.top_rel, self.occ = vision.calibrate_tray(
            self.img, expected=N * N + round(N * N * 0.2), verbose=True)
        # Park a piece for measurement in the empty board cell nearest the board centre: the
        # sampling window then stays inside the board, where nothing bright can leak in.
        bx0, by0 = self.board.x + self.board.w / 2, self.board.y + self.board.h / 2
        cand = [(cell_mean(self.img, b),
                 (centre(b)[0] - bx0) ** 2 + (centre(b)[1] - by0) ** 2, b)
                for b in self.cells.values()]
        empt = [c for c in cand if c[0] < 45] or cand
        empt.sort(key=lambda t: (t[1], t[0]))
        self.spot = centre(empt[0][2])
        self.hw, self.hh = int(vision.BOARD[2] / N), int(vision.BOARD[3] / N)
        self.blank_box = (self.spot[0] - self.hw // 2, self.spot[1] - self.hh // 2,
                          self.hw, self.hh)
        self.moves = 0
        self.placed: set[str] = set()      # cells confirmed locked
        self.used: set[str] = set()        # pieces already delivered

    # ---- reading -----------------------------------------------------------
    def read_cells(self, idf: Identifier) -> dict[str, str]:
        """Name the cell classes from the reference panel (never from the board)."""
        out, blank = {}, 0
        for key, box in self.panel.items():
            crop = vision.sub(self.img, box)
            if vision.is_blank(crop):
                out[key] = BLANK
                blank += 1
                print(f"  cell({key}) white thumbnail -> {BLANK}")
                continue
            res = idf.identify(crop)
            out[key] = res.label
            print(f"  cell({key}) {res}")
        if blank >= N * N - 5:
            print("ABORT: the reference panel is not rendered (every thumbnail blank)")
            raise SystemExit(4)
        return out

    def read_pieces(self, idf: Identifier) -> tuple[dict[str, str], dict[str, tuple]]:
        cls, box = {}, {}
        for slot in self.slots:
            if vision.is_empty(vision.sub(self.img, slot)):
                continue
            key = f"{slot[0]},{slot[1]}"
            res = idf.identify(vision.sub(self.img, slot))
            cls[key] = res.label
            box[key] = slot
            print(f"  piece({key}) {res}")
        return cls, box

    def read_edges(self, keys: list[str], boxes: dict[str, tuple]) -> dict[str, dict]:
        """One move per piece: lift it, park it on the blank cell, read, put it back."""
        out: dict[str, dict] = {}
        for i, key in enumerate(keys):
            slot = boxes[key]
            sx, sy = centre(slot)
            self.mouse.grab_at((sx, sy))
            self.mouse.glide((sx + 96, sy + 74))
            time.sleep(0.15)
            if not vision.is_empty(vision.sub(self.mouse.shot(), slot)):
                self.mouse.release()
                out[key] = {s: None for s in SIDES}
                print(f"  edges({key}) slot held nothing -- unknown")
                continue
            self.mouse.glide(self.spot)
            time.sleep(0.22)
            shot = self.mouse.shot()
            cv2.imwrite(str(SHOTS / f"run_edge_{i:02d}.png"), shot)
            edges = pe.read_edges(shot, self.blank_box)
            out[key] = edges
            if all(v is None for v in edges.values()):
                tag = "unknown (reading broke the rules)"
            else:
                tag = " ".join(f"{s}:{edges[s]}" for s in SIDES)
            print(f"  edges({key}) {tag}   [{i + 1}/{len(keys)}]")
            self.mouse.glide((sx, sy))
            self.mouse.release()
            self.moves += 1
            time.sleep(0.12)
        return out

    def scan_locked(self) -> None:
        """Seed `placed` with cells that already hold a locked piece.

        Without this a resumed run believes every cell is empty, drops onto occupied cells
        (the engine refuses) and, worse, reads the piece that is already there as its own
        success.  The probe costs no move on a locked cell.
        """
        for key, box in self.cells.items():
            inner = vision.sub(self.img, box)
            h, w = inner.shape[:2]
            middle = inner[h // 4:h - h // 4, w // 4:w - w // 4]
            # the middle half is what a cell's own piece covers; a neighbour's tab only
            # reaches the edge, and testing the whole box called empty cells occupied
            if float(cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY).mean()) < 45:
                continue
            if self.locked(key):
                self.placed.add(key)
                print(f"  cell({key}) already holds a locked piece")
            else:
                self.mouse.glide(centre(box))
                self.mouse.release()

    # ---- placing -----------------------------------------------------------
    def anchored(self, cell: str) -> bool:
        r, c = (int(v) for v in cell.split(","))
        if r in (0, N - 1) or c in (0, N - 1):
            return True
        return any(f"{r + dr},{c + dc}" in self.placed
                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))

    def locked(self, cell: str) -> bool:
        """Five presses inside the piece box: a loose piece cannot survive five misses."""
        cx, cy = centre(self.cells[cell])
        for dx, dy in PROBE_POINTS:
            pt = (cx + dx, cy + dy)
            before = cell_mean(self.mouse.shot(), self.cells[cell])
            self.mouse.grab_at(pt)
            self.mouse.glide((pt[0] + 62, pt[1] + 38))
            time.sleep(0.16)
            after = cell_mean(self.mouse.shot(), self.cells[cell])
            if after < before - 25 or after < 35:
                return False                  # still holding it
            self.mouse.release()
        return True

    def deliver(self, piece: str, cell: str, boxes: dict[str, tuple]) -> bool:
        slot = boxes[piece]
        sx, sy = centre(slot)
        self.mouse.grab_at((sx, sy))
        self.mouse.glide((sx + 96, sy + 74))
        time.sleep(0.15)
        if not vision.is_empty(vision.sub(self.mouse.shot(), slot)):
            self.mouse.release()
            print(f"    {piece}: not in its slot any more")
            return False
        self.mouse.glide(centre(self.cells[cell]))
        time.sleep(0.16)
        self.mouse.release()
        self.moves += 1
        if self.locked(cell):
            self.placed.add(cell)
            self.used.add(piece)
            print(f"    {piece} -> cell({cell})   [move {self.moves}]  locked  "
                  f"{len(self.placed)}/{N * N}")
            return True
        print(f"    {piece} -> cell({cell})   [move {self.moves}]  refused")
        return False

    def park(self, boxes: dict[str, tuple], piece: str) -> None:
        """A refused piece goes back to its own slot, never left on the board."""
        self.mouse.glide(centre(boxes[piece]))
        self.mouse.release()
        self.moves += 1


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--max-moves", type=int, default=90)
    args = ap.parse_args()
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    mouse = Mouse()
    run = Run(mouse)
    print(f"board {run.board.x},{run.board.y} {run.board.w}x{run.board.h}; "
          f"measuring spot {run.spot}")

    if args.go and not args.dry:
        print("\nexisting board")
        run.scan_locked()
        print(f"  {len(run.placed)} cell(s) already locked")

    idf = Identifier()
    print("\nreference panel")
    cells = run.read_cells(idf)
    print("\ntray")
    cls, boxes = run.read_pieces(idf)

    need = Counter(v for k, v in cells.items() if k not in run.placed)
    have = Counter(cls.values())
    print(f"\n{'class':10s} cells pieces")
    for c in sorted(set(need) | set(have), key=lambda k: -(need[k] + have[k])):
        flag = "" if have[c] >= need[c] else "   SHORT"
        print(f"  {c:10s} {need[c]:5d} {have[c]:6d}{flag}")

    edges: dict[str, dict] = {k: {s: None for s in SIDES} for k in cls}
    if args.dry or args.no_edges:
        print("\ncuts not read in this mode (every side unknown)")
    else:
        print(f"\nreading the cut of {len(cls)} pieces ({len(cls)} moves)")
        edges = run.read_edges(list(cls), boxes)

    pieces = {k: jigsaw.Piece(k, cls[k],
                              {s: EDGE_SIGN[edges[k].get(s)] for s in SIDES})
              for k in cls}
    # only the cells that still need a piece take part in the assignment
    cell_cls = {tuple(int(v) for v in k.split(",")): v
                for k, v in cells.items() if k not in run.placed}
    sol = jigsaw.solve_soft(cell_cls, pieces, n=N)
    broken = jigsaw.violations(cell_cls, pieces, sol.assignment, n=N)
    print(f"\nassignment: {len(sol.assignment)} cells, {broken} rule(s) broken, "
          f"{len(sol.unused)} piece(s) left over")
    ring = order_ring(N)
    for cell in ring:
        key = f"{cell[0]},{cell[1]}"
        piece = sol.assignment.get(cell)
        if piece is None:
            print(f"  cell({key}) {cells[key]:9s} <- nothing")
            continue
        cut = " ".join(f"{s[0]}{pieces[piece].edges[s] if pieces[piece].edges[s] is not None else '?'}"
                       for s in SIDES)
        flag = "" if pieces[piece].cls == cells[key] else "   <-- class mismatch!"
        print(f"  cell({key}) {cells[key]:9s} <- {piece}  {cut}{flag}")

    LOG.write_text(json.dumps({
        "cells": cells, "pieces": cls, "edges": edges,
        "assignment": {f"{r},{c}": p for (r, c), p in sol.assignment.items()},
        "violations": broken}, indent=2), encoding="utf-8")
    print(f"\nwritten to {LOG}")

    if args.dry or not args.go:
        print("\n--dry: nothing was touched (plan only)")
        return 0

    print("\nplacing")
    for attempt in range(3):
        todo = [c for c in ring if f"{c[0]},{c[1]}" not in run.placed
                and run.anchored(f"{c[0]},{c[1]}")]
        if not todo:
            break
        progress = 0
        print(f"\n-- pass {attempt + 1}: {len(todo)} candidate cell(s)")
        for (r, c) in todo:
            cell = f"{r},{c}"
            if cell in run.placed or run.moves >= args.max_moves:
                continue
            if not run.anchored(cell):
                continue
            cands = [(p, sc) for p, sc in
                     jigsaw.candidates_for(cell_cls, pieces, (r, c), n=N)
                     if p not in run.used]
            if not cands:
                # The class filter found nothing, which means this cell's piece is one the
                # classifier filed under the wrong class.  Fall back to trying the leftovers
                # -- but only when few are left, so this never becomes a brute force.
                spare = [p for p in pieces if p not in run.used]
                if len(spare) <= 10:
                    cands = [(p, 99) for p in spare]
                    print(f"\ncell({cell}) wants {cells[cell]} -- no class match, "
                          f"trying the {len(spare)} leftovers")
                else:
                    continue
            else:
                print(f"\ncell({cell}) wants {cells[cell]}")
            placed_here = False
            for piece, _sc in cands[:9]:
                if run.moves >= args.max_moves:
                    break
                if run.deliver(piece, cell, boxes):
                    progress += 1
                    placed_here = True
                    break
                run.park(boxes, piece)
            # Even a class match can be wrong: the classifier can put a cell's piece under
            # another class, in which case the right piece is sitting among the leftovers.
            # Try those too, but only when few are left, so this never becomes brute force.
            if not placed_here and run.moves < args.max_moves:
                spare = [p for p in pieces if p not in run.used]
                if 0 < len(spare) <= 10:
                    print(f"  no class match worked; trying {len(spare)} leftovers")
                    for piece in spare:
                        if run.moves >= args.max_moves:
                            break
                        if run.deliver(piece, cell, boxes):
                            progress += 1
                            break
                        run.park(boxes, piece)
        if progress == 0:
            print("  no progress in this pass, stopping")
            break

    shot = mouse.shot()
    cv2.imwrite(str(SHOTS / "run_final.png"), shot)
    print(f"\nmoves used: {run.moves}")
    print(f"board shows {sum(vision.board_filled(shot, N))}/{N * N} cells with a photo, "
          f"{len(run.placed)} confirmed locked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
