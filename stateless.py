"""Stateless executor: measure everything, then take one action. Repeat.

Why this replaces the old loop: every bug in this session came from carrying state across
frames -- a remembered slot, a cached cell box, an assumed piece position.  One release that
did not land exactly where assumed poisoned everything downstream, and a piece could be lost
for good.

Here nothing is remembered except two things that cannot rot:

  * the piece templates (what each piece looks like) -- measured stable: the same piece scores
    0.937 on the board after being carried there from the tray,
  * the class of each cell (the reference panel does not change during a game).

After every single action the screen is photographed again and all geometry is re-derived from
that photograph: the reference panel locates the board and the tray lattice, and each piece is
placed by its own fingerprint among those boxes.  There is no parking bookkeeping, because
wherever a refused piece ends up, the next snapshot sees it there.

  python stateless.py --dry     # read-only: derive the state, print the plan
  python stateless.py --go      # act, one deliberate move at a time
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import cv2
import numpy as np

import fingerprint as fp
import jigsaw
import probe_edges as pe
import vision
from identify import BLANK, Identifier
from solve_live import Mouse, _enable_dpi, check_foreground, centre

N = 5
SHOTS = vision.ROOT / "shots"
LOG = vision.ROOT / "logs" / "stateless.json"
EDGE_SIGN = {"tab": 1, "notch": -1, "flat": 0, None: None}
SIDES = ("top", "right", "bottom", "left")


def ring_order(n: int) -> list[tuple[int, int]]:
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


class Snap:
    """One photograph and everything that can be derived from it."""

    def __init__(self, img: np.ndarray) -> None:
        self.img = img
        self.ref, self.board, _ = vision.layout(img)
        self.cells = {f"{r},{c}": b for r, c, b in self.board.cells(N)}
        self.slots, self.rows, self.top_rel, self.occ = vision.calibrate_tray(
            img, expected=N * N + round(N * N * 0.2))
        self.panel = {f"{r},{c}": b for r, c, b in self.ref.cells(N)}

    def cell_has_piece(self, key: str) -> bool:
        return vision.has_piece(vision.sub(self.img, self.cells[key]), 30)

    def slot_has_piece(self, box) -> bool:
        return vision.has_piece(vision.sub(self.img, box), 20)


def locate(snap: Snap, tpl) -> tuple[tuple | None, float]:
    """Where is this piece in *this* photograph?  No memory, no stale box."""
    if tpl is None:
        return None, 0.0
    cands = list(snap.cells.values()) + list(snap.slots)
    best_sc, best_box = 0.0, None
    for b in cands:
        sc = fp.score_at(snap.img, tpl, b)
        if sc > best_sc:
            best_sc, best_box = sc, b
    if best_box is not None and best_sc >= fp.GOOD:
        fine = fp.refine(snap.img, tpl, best_box)
        sc = fp.score_at(snap.img, tpl, fine)
        if sc >= fp.GOOD:
            return fine, sc
    # slipped further than a candidate box: coarse sweep, then the same local gate
    x, y, coarse = fp.find(snap.img, tpl)
    if coarse >= 0.45:
        fine = fp.refine(snap.img, tpl, (x, y, 112, 85), radius=24)
        sc = fp.score_at(snap.img, tpl, fine)
        if sc >= fp.GOOD:
            return fine, sc
    return None, max(best_sc, coarse)


class Pieces:
    """Who each piece is; where it is comes from the latest snapshot."""

    def __init__(self) -> None:
        self.tpl: dict[str, np.ndarray | None] = {}
        self.cls: dict[str, str] = {}
        self.used: set[str] = set()

    def inventory(self, snap: Snap, idf: Identifier, verbose: bool = True) -> None:
        for i, box in enumerate(snap.slots):
            if not snap.slot_has_piece(box):
                continue
            key = f"s{i:02d}"
            tpl = fp.template(snap.img, box)
            if tpl is None:
                continue
            res = idf.identify(vision.sub(snap.img, box))
            self.tpl[key] = tpl
            self.cls[key] = res.label
            if verbose:
                print(f"  {key} at ({box[0]},{box[1]})  {res}")

    def spare(self, cls: str | None = None) -> list[str]:
        return [k for k in self.tpl if k not in self.used and (cls is None or self.cls[k] == cls)]


def press_is_locked(mouse: Mouse, box, presses: int = 5) -> bool:
    """Can the engine lift what is in this box?

    Five press points, not one.  A single press that lands on nothing changes nothing either,
    and reading that as "locked" is exactly how an earlier version reported 17 placed cells
    when the game said 10.  A locked box costs no move (the engine returns before counting),
    so the extra presses are free.
    """
    cx, cy = centre(box)
    offsets = ((0, 0), (-30, -20), (30, 20), (-30, 20), (30, -20))[:presses]
    for dx, dy in offsets:
        pt = (cx + dx, cy + dy)
        before = float(cv2.cvtColor(vision.sub(mouse.shot(), box), cv2.COLOR_BGR2GRAY).mean())
        mouse.grab_at(pt)
        mouse.glide((pt[0] + 62, pt[1] + 38))
        time.sleep(0.14)
        after = float(cv2.cvtColor(vision.sub(mouse.shot(), box), cv2.COLOR_BGR2GRAY).mean())
        if after < before - 25 or after < 30:
            return False                  # it came with the pointer
        mouse.release()
    return True


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--edges", action="store_true", help="also read each piece's cut (one move per piece)")
    ap.add_argument("--max-moves", type=int, default=120)
    args = ap.parse_args()
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    mouse = Mouse()
    snap = Snap(mouse.shot())
    cv2.imwrite(str(SHOTS / "state_start.png"), snap.img)
    print(f"reference ({snap.ref.x},{snap.ref.y}) board ({snap.board.x},{snap.board.y}) "
          f"{snap.board.w}x{snap.board.h}; tray {snap.rows} rows, {snap.occ} slots occupied")

    idf = Identifier()
    print("\ncell classes (reference panel)")
    cells: dict[str, str] = {}
    for key, box in snap.panel.items():
        crop = vision.sub(snap.img, box)
        if vision.is_blank(crop):
            cells[key] = BLANK
            continue
        res = idf.identify(crop)
        cells[key] = res.label
    print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(cells.items())))

    print("\npieces in the tray")
    pieces = Pieces()
    pieces.inventory(snap, idf)
    print(f"  {len(pieces.tpl)} pieces, classes "
          f"{dict(Counter(pieces.cls.values()).most_common())}")

    need = Counter(v for v in cells.values())
    have = Counter(pieces.cls.values())
    print(f"\n{'class':10s} cells pieces")
    for c in sorted(set(need) | set(have), key=lambda k: -(need[k] + have[k])):
        print(f"  {c:10s} {need[c]:5d} {have[c]:6d}"
              f"{'   SHORT' if have[c] < need[c] else ''}")

    edges: dict[str, dict] = {k: {s: None for s in SIDES} for k in pieces.tpl}
    if args.edges and args.go:
        print(f"\nreading {len(list(pieces.tpl))} cuts (one move each: lifted, photographed, put back)")
        for i, key in enumerate(list(pieces.tpl)):
            snap = Snap(mouse.shot())
            box, _sc = locate(snap, pieces.tpl[key])
            if box is None:
                print(f"  {key}: not visible, cut unknown")
                continue
            empty = [k2 for k2 in snap.cells if not snap.cell_has_piece(k2)]
            spot = centre(snap.cells[empty[i % len(empty)]]) if empty else (713, 665)
            sx, sy = centre(box)
            mouse.grab_at((sx, sy))
            mouse.glide((sx + 96, sy + 74))
            time.sleep(0.14)
            if fp.score_at(mouse.shot(), pieces.tpl[key], box) > 0.8:
                mouse.release()
                print(f"  {key}: did not come away, cut unknown")
                continue
            mouse.glide(spot)                      # hold it over an empty cell: isolated
            time.sleep(0.22)
            shot = mouse.shot()
            cv2.imwrite(str(SHOTS / f"cut_{i:02d}.png"), shot)
            edges[key] = pe.read_edges_union(shot, (spot[0] - 56, spot[1] - 42, 112, 85))
            mark = " ".join(f"{s0[0]}{edges[key][s0] or '?'}" for s0 in SIDES)
            print(f"  {key} ({pieces.cls[key]}) {mark}   [{i + 1}/{len(list(pieces.tpl))}]")
            mouse.glide((sx, sy))                  # put it back where it was
            mouse.release()
            time.sleep(0.12)

    # ---- the assignment, from classes (and cuts when they were read) --------
    jp = {k: jigsaw.Piece(k, pieces.cls[k],
                          {s: EDGE_SIGN[edges[k].get(s)] for s in SIDES}) for k in pieces.tpl}
    cell_cls = {tuple(int(v) for v in k.split(",")): v for k, v in cells.items()}
    sol = jigsaw.solve_soft(cell_cls, jp, n=N)
    print(f"\nassignment: {len(sol.assignment)} cells, "
          f"{jigsaw.violations(cell_cls, jp, sol.assignment, n=N)} rule(s) broken")
    plan = {f"{r},{c}": k for (r, c), k in sol.assignment.items()}
    LOG.write_text(json.dumps({"cells": cells, "pieces": pieces.cls, "edges": edges,
                               "plan": plan}, indent=2), encoding="utf-8")

    if args.dry or not args.go:
        for r, c in ring_order(N):
            key = f"{r},{c}"
            print(f"  cell({key}) {cells[key]:9s} <- {plan.get(key, 'nothing')}")
        print("\n--dry: nothing was touched")
        return 0

    # ---- act: one deliberate move at a time, re-measuring after each --------
    print("\nacting (stateless: the screen is re-read before every move)")
    moves = 0
    confirmed: dict[str, str] = {}      # cell -> piece, verified locked
    refused: Counter = Counter()
    tried: set[tuple[str, str]] = set()   # (piece, cell) pairs already attempted
    relocations = 0

    for attempt in range(2):
        progress = 0
        for r, c in ring_order(N):
            if moves >= args.max_moves:
                break
            key = f"{r},{c}"
            if key in confirmed:
                continue
            snap = Snap(mouse.shot())                 # fresh geometry, every time
            anchored = (r in (0, N - 1) or c in (0, N - 1)
                        or any(f"{r + dr},{c + dc}" in confirmed
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))))
            if not anchored:
                continue
            wanted = cells[key]
            # exactly one piece is offered for this cell in this pass: the planned one, or
            # the single best spare of that class.  Trying several per cell is what turned
            # the pass into a sweep.
            order = ([plan[key]] if key in plan and plan[key] not in pieces.used else [])
            order = [k for k in order if (k, key) not in tried]
            if not order:
                order = [k for k in pieces.spare(wanted) if (k, key) not in tried][:1]
            done = False
            for pk in order[:1]:
                if moves >= args.max_moves:
                    break
                box, sc = locate(snap, pieces.tpl[pk])
                if box is None:
                    pieces.used.add(pk)
                    print(f"    {pk}: not visible anywhere (best {sc:.2f}) -- dropped from play")
                    continue
                sx, sy = centre(box)
                mouse.grab_at((sx, sy))
                mouse.glide((sx + 96, sy + 74))
                time.sleep(0.14)
                if fp.score_at(mouse.shot(), pieces.tpl[pk], box) > 0.8:
                    mouse.release()
                    print(f"    {pk}: did not come away from ({box[0]},{box[1]})")
                    continue
                mouse.glide(centre(snap.cells[key]))
                time.sleep(0.14)
                mouse.release()
                moves += 1
                after = Snap(mouse.shot())
                here = fp.score_at(after.img, pieces.tpl[pk], after.cells[key])
                # Credit a placement by the engine's own behaviour (it will not lift a placed
                # piece), not by the fingerprint: the fingerprint told the truth only 2 times
                # out of 8 the engine had in fact locked, which then cost a second attempt on
                # an already-finished cell.
                if press_is_locked(mouse, after.cells[key]):
                    confirmed[key] = pk
                    pieces.used.add(pk)
                    progress += 1
                    print(f"    {pk} ({pieces.cls[pk]}) -> cell({key})  [move {moves}]  "
                          f"LOCKED  {len(confirmed)}/{N * N}")
                    done = True
                    break
                refused[(pieces.cls[pk], wanted)] += 1
                tried.add((pk, key))
                print(f"    {pk} ({pieces.cls[pk]}) -> cell({key}) wanted {wanted}  "
                      f"[move {moves}]  refused (fingerprint {here:.2f})")
                # The user's rule, and the right one: the board never keeps a wrong piece.
                # Take it straight back to a slot that is empty in this very photograph.
                after = Snap(mouse.shot())
                free = [b for b in after.slots if not after.slot_has_piece(b)]
                if free:
                    mouse.grab_at(centre(after.cells[key]))
                    mouse.glide(centre(free[0]))
                    mouse.release()
                    moves += 1
                    time.sleep(0.18)
                    print(f"      -> taken back to slot ({free[0][0]},{free[0][1]})  "
                          f"[move {moves}]")
                else:
                    print("      -> no empty slot to take it back to")
                break            # one piece per cell per pass: no candidate roulette
        if progress == 0:
            print("  no progress in this pass")
            break

    final = Snap(mouse.shot())
    cv2.imwrite(str(SHOTS / "stateless_end.png"), final.img)
    locked = [k for k in final.cells
              if vision.has_piece(vision.sub(final.img, final.cells[k]), 30)
              and press_is_locked(mouse, final.cells[k])]
    print(f"\nmoves: {moves}")
    print(f"confirmed by the loop: {len(confirmed)}   confirmed by a final 5-point audit: "
          f"{len(locked)}")
    print(f"refused attempts by class pair: {dict(refused)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

