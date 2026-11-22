"""Put the pieces where they belong, one *verified* move at a time.

What the engine actually does (recon/puzzle_board.js):

    checkSnap(piece):
        if (this.dragging !== piece) return false;        mousedown must hit the piece box
        if (!SOLUTIONS.has(piece)) return false;           decoys never snap
        if (!this.hasAnchor(piece)) return false;          ring cell, or a placed neighbour
        if (dist(piece, spot) < pieceWidth * 0.14) { place it, placedCount++ }

so a piece snaps only into its own cell, and the only cheap proof that a drop was right is
what the engine does next: `getPieceAt` skips placed pieces, so **a placed piece cannot be
picked up again**.  The progress label is unreliable (its position moves with the page
state), while that behaviour is a property of the engine itself.  Measuring it costs no
move when the piece is locked, because mousedown hits nothing and the engine returns before
touching its move counter.

The placement itself is pure geometry, measured in probe_map.py: the pointer->piece
transform is exactly 1:1, so grabbing a box at its centre lands that centre on the aim
point.  Grab at the slot centre, aim at the cell centre, residual 0-6 px against a 15.7 px
snap radius.

`moveTarget` is ceil(pieces * 1.3) -- a rubric target, not a limit (recon/exame_mode.js),
so a refused piece is retried on its next candidate cell while still in hand, at one move
per try.

  python place.py                                      # print the plan, touch nothing
  python place.py --cleanup-only                       # move loose pieces off the board
  python place.py --go --max-moves 40
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

import vision
from solve_live import Mouse, _enable_dpi, check_foreground, centre

PLAN = vision.ROOT / "logs" / "plan_board.json"
N = 5
BLANK = "blank"


def ring_order(n: int) -> list[tuple[int, int]]:
    """Outer ring clockwise, then inwards.  The engine needs an anchor, so a cell that
    touches the frame is always safe to try, and the ring around an untouched interior
    cell always provides one."""
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


def cell_mean(img: np.ndarray, box) -> float:
    return float(cv2.cvtColor(vision.sub(img, box), cv2.COLOR_BGR2GRAY).mean())


class Solver:
    def __init__(self, mouse: Mouse, plan: dict) -> None:
        self.mouse = mouse
        self.img = mouse.shot()
        self.ref, self.board, _ = vision.layout(self.img)
        self.box = {f"{r},{c}": b for r, c, b in self.board.cells(N)}
        self.cells = {k: (v if v else {"class": BLANK, "confident": True, "alts": []})
                      for k, v in plan["cells"].items()}
        self.pieces = {k: dict(v) for k, v in plan["pieces"].items()}
        self.placed: set[str] = set()
        self.used: set[str] = set()
        self.moves = 0
        self.refused = 0

    # ---- tray bookkeeping --------------------------------------------------
    def slots(self):
        # the analytic lattice, not gap detection: the grid must stay right after most
        # pieces have left the tray, and it must not depend on how many are left
        return [box for b in ("left", "right")
                for _r, _c, box in vision.lattice_slots(self.img, b)]

    def free_slots(self):
        shot = self.mouse.shot() if self.moves else self.img
        return [b for b in self.slots() if vision.is_empty(vision.sub(shot, b))]

    def anchored(self, key: str) -> bool:
        r, c = (int(v) for v in key.split(","))
        if r in (0, N - 1) or c in (0, N - 1):
            return True
        return any(f"{r + dr},{c + dc}" in self.placed
                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))

    def scan_locked(self) -> None:
        """Find cells that already hold a locked piece (an interrupted run).

        The probe is the one `locked()` uses, so a locked cell costs no move; a loose piece
        found this way is put straight back where it was.
        """
        for key, box in self.box.items():
            if cell_mean(self.img, box) < 45:
                continue
            if self.locked(key):
                self.placed.add(key)
                print(f"  cell({key}) already holds a locked piece")
            else:
                self.mouse.glide(centre(box))
                self.mouse.release()
                self.moves += 1

    # ---- the two primitives that talk to the engine ------------------------

    # ---- the two primitives that talk to the engine ------------------------
    def locked(self, cell: str) -> bool:
        """Drop done: ask the engine whether it will let go of the piece.

        A placed piece is skipped by hit testing, so pressing on it and nudging changes
        nothing -- and costs no move, because the engine's mouseup handler returns before
        counting when no piece is selected.  If the piece does move it is still in hand
        when this returns, ready to be carried to the next candidate cell.

        Five press points, not one: a single press that happens to land on nothing changes
        nothing either, and reading that as "locked" is exactly how an earlier version
        reported 17 placed cells when the game said 10.  Five misses in a row inside the
        piece box is not something a loose piece survives.
        """
        cx, cy = centre(self.box[cell])
        for dx, dy in ((0, 0), (-30, -20), (30, 20), (-30, 20), (30, -20)):
            pt = (cx + dx, cy + dy)
            before = cell_mean(self.mouse.shot(), self.box[cell])
            self.mouse.grab_at(pt)
            self.mouse.glide((pt[0] + 62, pt[1] + 38))
            time.sleep(0.18)
            after = cell_mean(self.mouse.shot(), self.box[cell])
            if after < before - 25 or after < 35:
                return False              # still holding it
            self.mouse.release()          # nothing selected: no move is counted
        return True

    def grab(self, piece: str) -> bool:
        """Pick a piece up from its recorded slot and prove that it was picked up.

        The box is remembered from the plan, and a plan goes stale as soon as a piece has
        been moved, so a grab that finds nothing must not be followed by a drop: an empty
        hand over an occupied cell looks exactly like a successful placement.
        A wide nudge is used so the slot is really clear when it is measured; the drag
        offset is fixed at the moment of the press, so the path does not matter.
        """
        src = self.pieces[piece]["box"]
        cx, cy = centre(src)
        self.mouse.grab_at((cx, cy))
        self.mouse.glide((cx + 96, cy + 74))
        time.sleep(0.18)
        if vision.is_empty(vision.sub(self.mouse.shot(), src)):
            return True
        self.mouse.release()
        print(f"    {piece}: slot ({src[0]},{src[1]}) is empty -- that piece is not there")
        self.used.add(piece)
        return False

    def place(self, piece: str, cell: str) -> bool:
        """Carry one piece to one cell, let go, and report whether it stuck."""
        if not self.grab(piece):
            return False
        self.mouse.glide(centre(self.box[cell]))
        time.sleep(0.18)
        self.mouse.release()
        self.moves += 1
        if self.locked(cell):
            self.placed.add(cell)
            self.used.add(piece)
            print(f"    {piece} -> cell({cell})   [move {self.moves}]  locked  "
                  f"{len(self.placed)}/{N * N}")
            return True
        self.refused += 1
        print(f"    {piece} -> cell({cell})   [move {self.moves}]  refused")
        return False

    def carry_to(self, cell: str) -> None:
        """Still holding a refused piece: take it straight to another cell."""
        self.mouse.glide(centre(self.box[cell]))

    def park(self, piece: str) -> None:
        """Anything the engine refused goes back to the tray, never left on the board."""
        free = self.free_slots()
        if not free:
            self.mouse.release()
            return
        self.mouse.glide(centre(free[0]))
        self.mouse.release()
        self.moves += 1
        self.pieces[piece]["box"] = free[0]

    # ---- opening state -----------------------------------------------------
    def cleanup(self) -> None:
        """Move anything loose off the board before we start.

        A piece left by an earlier run covers the placeholder cells, and it is drawn
        outside the cell boxes, so it has to be found as a bright blob inside the board
        rectangle rather than from a cell box.
        """
        for _ in range(8):
            shot = self.mouse.shot()
            g = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
            b = self.board
            m = (g[b.y:b.y + b.h, b.x:b.x + b.w] > 60).astype(np.uint8)
            n, _lab, stats, _ce = cv2.connectedComponentsWithStats(
                cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)))
            blobs = [tuple(int(v) for v in stats[i][:4]) for i in range(1, n)
                     if stats[i][4] > 1500]
            if not blobs:
                return
            for bx, by, bw, bh in blobs:
                pt = (b.x + bx + bw // 2, b.y + by + bh // 2)
                self.mouse.grab_at(pt)
                self.mouse.glide((pt[0] + 30, pt[1] + 20))
                time.sleep(0.15)
                free = self.free_slots()
                if not free:
                    self.mouse.release()
                    return
                self.mouse.glide(centre(free[0]))
                self.mouse.release()
                self.moves += 1
                print(f"    loose piece moved off the board  [move {self.moves}]")


def candidates(piece_class: str, alts: list[str], cells: dict[str, dict]) -> list[str]:
    exact = [k for k, v in cells.items() if v["class"] == piece_class]
    if exact:
        return exact
    return [k for k, v in cells.items() if v["class"] in alts]


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--cleanup-only", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--max-moves", type=int, default=70)
    args = ap.parse_args()
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    plan = json.loads(open(args.plan, encoding="utf-8").read())
    cells = {k: (v if v else {"class": BLANK, "confident": True, "alts": []})
             for k, v in plan["cells"].items()}
    todo = list(plan["pieces"].items())

    need = Counter(v["class"] for v in cells.values())
    have = Counter(v["class"] for _, v in todo)
    print(f"\ncells {len(cells)}, pieces {len(todo)}")
    for cls in sorted(set(need) | set(have), key=lambda k: -(need[k] + have[k])):
        flag = "" if have[cls] >= need[cls] else "   SHORT"
        print(f"  {cls:10s} cells {need[cls]:2d}  pieces {have[cls]:2d}{flag}")

    def rank(kv):
        cands = candidates(kv[1]["class"], kv[1]["alts"], cells)
        return (0 if cands else 1, len(cands), 0 if kv[1]["confident"] else 1)

    ranked = sorted(todo, key=rank)
    print("\nfirst in line:")
    for k, v in ranked[:10]:
        cands = candidates(v["class"], v["alts"], cells)
        print(f"  {k} {v['class']:9s} -> {', '.join(cands) if cands else 'NO CELL'}")
    if len(ranked) > 10:
        print(f"  ... {len(ranked) - 10} more")

    if args.cleanup_only:
        Solver(Mouse(), plan).cleanup()
        print("\nboard cleared")
        return 0
    if args.dry or not args.go:
        print("\n--dry: nothing was touched")
        return 0

    mouse = Mouse()
    s = Solver(mouse, plan)
    if args.cleanup:
        s.cleanup()
    s.scan_locked()

    # ---- pass 1: by class, ring cells first --------------------------------
    for key, info in ranked:
        if key in s.used or s.moves >= args.max_moves:
            continue
        cands = [c for c in candidates(info["class"], info["alts"], cells)
                 if c not in s.placed and s.anchored(c)]
        if not cands:
            continue
        print(f"\n{key} ({info['class']}) -> {cands}")
        for i, cell in enumerate(cands):
            if s.moves >= args.max_moves:
                break
            if i == 0:
                hit = s.place(key, cell)
            else:
                s.carry_to(cell)
                time.sleep(0.18)
                mouse.release()
                s.moves += 1
                hit = s.locked(cell)
                if hit:
                    s.placed.add(cell)
                    s.used.add(key)
                    print(f"    {key} -> cell({cell})   [move {s.moves}]  locked  "
                          f"{len(s.placed)}/{N * N}")
                else:
                    s.refused += 1
                    print(f"    {key} -> cell({cell})   [move {s.moves}]  refused")
            if hit:
                break
        else:
            s.park(key)
        if s.moves >= args.max_moves:
            print(f"\nstopping: move budget spent ({s.moves})")
            break

    # ---- pass 2: sweep the leftovers --------------------------------------
    # A piece the classifier filed under the wrong class has no candidate cell at all,
    # and a cell whose class was misread has no candidate piece.  Only those leftovers are
    # cross-tried here, so the cheap class decisions all happen first.
    if not args.no_sweep and s.moves < args.max_moves:
        spare = [k for k, _v in todo if k not in s.used]
        empty = [f"{r},{c}" for r, c in ring_order(N) if f"{r},{c}" not in s.placed]
        if spare and empty:
            print(f"\nsweep: {len(spare)} spare piece(s) against {len(empty)} empty cell(s)")
        for cell in empty:
            if s.moves >= args.max_moves or not s.anchored(cell):
                continue
            for key in list(spare):
                if s.moves >= args.max_moves:
                    break
                print(f"\nsweep {key} -> cell({cell})")
                if s.place(key, cell):
                    spare.remove(key)
                    break
                s.park(key)

    print(f"\nmoves used: {s.moves}   refused drops: {s.refused}")
    shot = mouse.shot()
    cv2.imwrite(str(vision.ROOT / "shots" / "place_final.png"), shot)
    filled = vision.board_filled(shot, N)
    print(f"board shows {sum(filled)}/{N * N} cells with a photo, "
          f"{len(s.placed)} of them confirmed locked")
    left = [f"{r},{c}" for r, c in ring_order(N) if f"{r},{c}" not in s.placed]
    print(f"unconfirmed cells: {', '.join(left) if left else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
