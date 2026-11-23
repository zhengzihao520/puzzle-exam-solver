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
CUTLOG = vision.ROOT / "logs" / "cuts.jsonl"
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
        self.slots = vision.tray_slots_all(img)
        self.rows, self.top_rel, self.occ = 0, 0.0, sum(
            1 for b in self.slots if not vision.is_empty(vision.sub(img, b)))
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
        self.box: dict[str, tuple] = {}
        self.cls: dict[str, str] = {}
        self.allow: dict[str, frozenset | None] = {}
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
            self.box[key] = box
            self.cls[key] = res.label
            # A doubtful reading is a two-way vote, and half the tray is doubtful (13 of 30 on
            # the game this was measured on), while the class budget must still be met: five
            # white cells and only three pieces called white.  Keeping both candidates lets the
            # cuts and the seams -- which are colour-blind -- settle it instead of a coin flip.
            self.allow[key] = (frozenset([res.label, *res.alternatives])
                               if not res.confident else frozenset([res.label]))
            if verbose:
                print(f"  {key} at ({box[0]},{box[1]})  {res}")

    def spare(self, cls: str | None = None) -> list[str]:
        """Spare pieces that could be `cls` -- a doubtful piece counts for either reading."""
        if cls is None:
            return [k for k in self.tpl if k not in self.used]
        return [k for k in self.tpl if k not in self.used and
                (self.allow[k] is None or cls in self.allow[k])]


def whiteness(crop: np.ndarray) -> float:
    """Fraction of a piece's middle half that is near-white and unsaturated.

    Measured on one real tray: the five pure-white pieces scored 1.000 1.000 1.000 0.983 0.973
    and the next piece 0.550, so this separates them by a wide margin.  A plain mean brightness
    cannot: the engine's ~20 px drop shadow lands on the piece and puts a white piece's mean
    (239) inside the range of photos taken on white backgrounds (236).
    """
    h, w = crop.shape[:2]
    mid = crop[h // 4:3 * h // 4, w // 4:3 * w // 4]
    g = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)[:, :, 1]     # 1 = saturation, 2 = value
    return float(((g > 225) & (sat < 25)).mean())


def name_blanks(pieces: "Pieces", img: np.ndarray, want: int,
                min_frac: float = 0.9) -> int:
    """Mark the white pieces by rank, since the panel says how many there should be.

    Two of the five white pieces were previously read as desktop and nailong -- both at the
    bottom of the reader's confidence (0.60 and 0.31), and both with a blank-looking body, so
    they won a class by default.  The board already knows the answer: k white cells means k
    white pieces, and the k whitest are the white ones.
    """
    ranked = sorted(((whiteness(vision.sub(img, b)), k) for k, b in pieces.box.items()),
                    reverse=True)
    marked = 0
    for frac, k in ranked[:want]:
        if frac < min_frac:
            break
        pieces.cls[k] = BLANK
        pieces.allow[k] = frozenset([BLANK])
        marked += 1
    if marked:
        print(f"  {marked} piece(s) named {BLANK} by whiteness rank (panel wants {want})")
    return marked


def seam_plan(mouse: Mouse, pieces: "Pieces", cell_cls: dict, edges: dict,
              weight: float = 3.0) -> dict[str, str]:
    """Which piece goes where, read off the tray's pixels plus the cut signs.

    Costs no move: each piece is found by fingerprint, refined onto itself, and its 112x85 body
    is cropped straight from the tray.  4 px of inset clears both the white stroke the engine
    draws and the 1 px by which a neighbour's tab can reach into our box.

    The cut signs carry the decision (colour-blind) and the pixels only break ties -- measured:
    pixels alone get 6/6 games down to 3/6, adding the signs restores 6/6.
    """
    import assemble as asm

    snap = Snap(mouse.shot())
    keys = [k for k in pieces.tpl if k not in pieces.used]
    boxes: dict[str, tuple] = {}
    for k in keys:
        box, _sc = locate(snap, pieces.tpl[k])
        if box is None:
            continue
        boxes[k] = fp.refine(snap.img, pieces.tpl[k], box)
    crops, kept = [], []
    for k in list(boxes):
        crop = vision.sub(snap.img, boxes[k])
        if crop.shape[:2] != (asm.H, asm.W):
            crop = cv2.resize(crop, (asm.W, asm.H), interpolation=cv2.INTER_AREA)
        crops.append(crop)
        kept.append(k)
    if len(kept) < len(cell_cls):
        print(f"  only {len(kept)} of {len(keys)} pieces found; cannot plan from pixels")
        return {}

    cr, cd = asm.cost_matrices(crops)
    def signs_of(k: str) -> tuple:
        """Four signs, with self-contradictory readings thrown out.

        A piece occupies one row and one column, so it can have at most one flat side per axis.
        The reader returned "flat left and flat right" for two pieces on one tray, and three
        pieces claiming to be the bottom-left corner at once -- geometrically impossible, and
        therefore free to reject.  Such a pair is dropped to unknown rather than passed on as
        though it were evidence.
        """
        got = {s: EDGE_SIGN[edges[k].get(s)] for s in SIDES}
        if got["top"] == 0 and got["bottom"] == 0:
            got["top"] = got["bottom"] = None
        if got["left"] == 0 and got["right"] == 0:
            got["left"] = got["right"] = None
        return tuple(got[s] for s in SIDES)

    piece_edges = {i: signs_of(k) for i, k in enumerate(kept)}
    dropped = sum(1 for k in kept
                  if tuple(EDGE_SIGN[edges[k].get(s)] for s in SIDES) != signs_of(k))
    # A doubtful piece may stand in for either of its two readings, but only where that is
    # actually needed.  Left free, the seam cost happily used a doubtful dog as the penguin --
    # penguin was already well supplied -- and the dog cell it should have filled stayed empty.
    wanted = set(cell_cls.values())
    need = Counter(cell_cls.values())
    have = Counter(pieces.cls[k] for k in kept)
    allow = {}
    for i, k in enumerate(kept):
        cand = pieces.allow.get(k)
        label = pieces.cls[k]
        if cand is None:
            allow[i] = None
            continue
        cand = frozenset(cand) & wanted
        # its own class short of pieces -> keep the alternatives; otherwise insist on the reading
        allow[i] = (cand or None) if have[label] < need.get(label, 0) else frozenset([label])
    asm.apply_cut_constraint(cr, cd, piece_edges, weight=weight)

    def misplaced(i: int, cell: tuple[int, int]) -> bool:
        """Is this piece physically unable to sit in this cell?

        Asymmetric on purpose.  A piece that *has* a flat side can only have it facing out, so a
        flat side facing the middle of the board is a contradiction and is refused outright.  The
        reverse is not used: the reader misses flat sides far more often than it invents them
        (25 pieces yielded 3 flat tops and 0 flat lefts), so "the cell is on the border but the
        piece shows no flat there" proves nothing and must not block a placement.

        This is what recovers a cell whose true piece was misread by class: the tray held a piece
        called "car" with flat top and bottom while the empty cell sat on the bottom row, and it
        was the only piece that could physically go in -- but the class filter rejected it and
        the plan sent a suitcase that the engine refused twice.
        """
        r, c = cell
        for side, v in zip(SIDES, piece_edges[i]):
            if v != 0:
                continue
            on_edge = {"top": r == 0, "bottom": r == N - 1,
                       "left": c == 0, "right": c == N - 1}[side]
            if not on_edge:
                return True
        return False

    got = asm.assemble(cell_cls, allow, cr, cd, n=N,
                       cell_ok=lambda i, cell: not misplaced(i, cell))
    known = sum(1 for e in piece_edges.values() if None not in e)
    per_side = Counter()
    for e in piece_edges.values():
        for name, v in zip(SIDES, e):
            if v == 0:
                per_side[name] += 1
    # Cells no piece can legally fill: a cell wanting umbrella when the reader put no umbrella
    # in the tray at all.  Rather than leave it empty, offer the piece whose edges best continue
    # its locked neighbours -- the engine still refuses a wrong one and the piece comes straight
    # back, so a guess costs two moves and a gap costs the cell.
    left_out = [c for c in cell_cls if c not in got]
    for cell in left_out:
        spare = [i for i in range(len(kept)) if i not in got.values()]
        if not spare:
            break
        r, c = cell

        def legal(i: int) -> bool:
            """Does this piece's shape even allow it here?

            A cell on the board's edge needs a flat side facing out, and an interior cell needs
            three or four cut sides -- a fact independent of the picture and of the class label.
            This is what found the last cell of a live game: the tray held a piece read as "car"
            whose shape was flat top and flat bottom while the empty cell was on the bottom row,
            and it was the only piece there that could physically go in.
            """
            for side, v in zip(SIDES, piece_edges[i]):
                if v is None:
                    continue
                on_edge = {"top": r == 0, "bottom": r == N - 1,
                           "left": c == 0, "right": c == N - 1}[side]
                if on_edge != (v == 0):
                    return False
            return True

        def fit(i: int) -> float:
            cost = 0.0
            if (r, c - 1) in got:
                cost += cr[got[(r, c - 1)], i]
            if (r, c + 1) in got:
                cost += cr[i, got[(r, c + 1)]]
            if (r - 1, c) in got:
                cost += cd[got[(r - 1, c)], i]
            if (r + 1, c) in got:
                cost += cd[i, got[(r + 1, c)]]
            return cost

        pool = [i for i in spare if legal(i)] or spare
        got[cell] = min(pool, key=fit)
    print(f"  seam plan: {len(kept)} pieces, {known} with all four cuts readable, "
          f"{dropped} rejected as impossible, {len(got)}/{len(cell_cls)} cells filled"
          + (f" ({len(left_out)} guessed from edges alone)" if left_out else ""))
    print("  flats per side: " + "  ".join(f"{k[:1]} {per_side[k]}" for k in SIDES)
          + "   (a 5x5 needs 5 each)")
    return {f"{r},{c}": kept[i] for (r, c), i in got.items()}


def press_is_locked(mouse: Mouse, box, presses: int = 5) -> bool:
    """Can the engine lift what is in this box?

    Five press points, not one.  A single press that lands on nothing changes nothing either,
    and reading that as "locked" is exactly how an earlier version reported 17 placed cells
    when the game said 10.  A locked box costs no move (the engine returns before counting),
    so the extra presses are free.

    Two things this now does that it did not, both of which made a refused drop look locked:

    * it looks at texture as well as brightness.  A piece leaving a cell takes its texture with
      it, while the brightness test alone missed dark pieces on the dark board -- a cell at 36
      going to 23 is a drop of 13, under the old fixed threshold of 25, so the piece was reported
      as "still there" and the cell was credited as placed when it was not.  That is the loop
      counting one or two more cells than the final audit;
    * it always releases the button and puts the piece back where it found it.  It used to return
      with the button still down when it detected a lift, so the next grab dragged whatever the
      pointer happened to be over.  A test that moves things must undo its own move.
    """
    def signature() -> tuple[float, float]:
        g = cv2.cvtColor(vision.sub(mouse.shot(), box), cv2.COLOR_BGR2GRAY)
        return float(g.mean()), float(g.std())

    cx, cy = centre(box)
    offsets = ((0, 0), (-30, -20), (30, 20), (-30, 20), (30, -20))[:presses]
    for dx, dy in offsets:
        pt = (cx + dx, cy + dy)
        before_mean, before_std = signature()
        mouse.grab_at(pt)
        try:
            mouse.glide((pt[0] + 62, pt[1] + 38))
            time.sleep(0.14)
            after_mean, after_std = signature()
        finally:
            mouse.glide(pt)               # put back whatever this press may have picked up
            mouse.release()
        if after_mean < before_mean - max(8.0, 0.25 * before_mean) or \
                after_std < 0.60 * before_std:
            return False                  # it came with the pointer
    return True


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--edges", action="store_true", help="also read each piece's cut (one move per piece)")
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--passes", type=int, default=4,
                    help="how many times to walk the ring; a later pass may only offer pieces "
                         "and cells not already tried together, so it cannot spin forever")
    args = ap.parse_args()
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2
    mouse = Mouse()
    # A crash between mouseDown and mouseUp would leave the button held, and the next grab then
    # drags whatever the pointer is over -- pieces strewn about, a cell lost for the game.  The
    # live run did exactly that when pyautogui's fail-safe fired mid-drag.
    import atexit
    atexit.register(mouse.release)
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
    name_blanks(pieces, snap.img, want=sum(1 for v in cells.values() if v == BLANK))
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
        # White pieces are skipped.  All five carry the same pure-white picture, so any of them
        # fits any white cell and their cut shape decides nothing -- while a white-on-white edge
        # is exactly what this reader cannot see (one came back with all four sides unknown).
        to_read = [k for k in pieces.tpl
                   if whiteness(vision.sub(snap.img, pieces.box[k])) < 0.9]
        print(f"\nreading {len(to_read)} cuts (one move each: lifted, photographed, put back); "
              f"{len(pieces.tpl) - len(to_read)} white piece(s) skipped")
        # Hold the piece over the empty canvas below the board, not over a cell.  Mid-game the
        # only free cells are surrounded by placed pieces, and a silhouette measured against
        # neighbouring art is worthless: that run produced 7 conflicting edges out of 11 pairs,
        # against 1 out of 12 while the board was still empty.
        dark = (snap.board.x + snap.board.w // 2, snap.board.y + snap.board.h + 45)
        for i, key in enumerate(to_read):
            snap = Snap(mouse.shot())
            box, _sc = locate(snap, pieces.tpl[key])
            if box is None:
                print(f"  {key}: not visible, cut unknown")
                continue
            sx, sy = centre(box)
            mouse.grab_at((sx, sy))
            mouse.glide((sx + 96, sy + 74))
            time.sleep(0.14)
            if fp.score_at(mouse.shot(), pieces.tpl[key], box) > 0.8:
                mouse.release()
                print(f"  {key}: did not come away, cut unknown")
                continue
            mouse.glide(dark)                      # hold it against plain background: isolated
            time.sleep(0.22)
            shot = mouse.shot()
            cv2.imwrite(str(SHOTS / f"cut_{i:02d}.png"), shot)
            crop = (dark[0] - 56, dark[1] - 42, 112, 85)
            edges[key] = pe.read_edges_union(shot, crop)
            with CUTLOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "crop": list(crop),
                                     "signs": edges[key]}) + "\n")
            mark = " ".join(f"{s0[0]}{edges[key][s0] or '?'}" for s0 in SIDES)
            print(f"  {key} ({pieces.cls[key]}) {mark}   [{i + 1}/{len(to_read)}]")
            mouse.glide((sx, sy))                  # put it back where it was
            mouse.release()
            time.sleep(0.12)

    # ---- the assignment -----------------------------------------------------
    # Two independent readings of the same question.  The seam plan compares the whole
    # arrangement against the photograph itself (measured on synthetic games with known
    # answers: 6/6 exact, and still 6/6 with 10% of the cut signs wrong); the rule solver only
    # asks whether each piece's four cuts are legal somewhere.  Both are computed and the more
    # complete one wins, with the seam plan preferred on a tie because it uses more evidence.
    # Seed what is already done first, so the plan is about the cells that still need a piece.
    # Planning all 25 on a half-finished board was rejected outright by the "as many pieces as
    # cells" guard: six pieces left in the tray for one empty cell is not a shortage.
    confirmed: dict[str, str] = {}      # cell -> piece, verified locked
    start_snap = Snap(mouse.shot())
    for k in list(start_snap.cells):
        if vision.has_piece(vision.sub(start_snap.img, start_snap.cells[k]), 30) and \
                press_is_locked(mouse, start_snap.cells[k]):
            confirmed[k] = "(already locked)"
    open_cells = {k: v for k, v in cells.items() if k not in confirmed}
    print(f"\n{len(confirmed)} cell(s) already locked, {len(open_cells)} still to do")
    cell_cls = {tuple(int(v) for v in k.split(",")): v for k, v in open_cells.items()}
    plan = seam_plan(mouse, pieces, cell_cls, edges)
    jp = {k: jigsaw.Piece(k, pieces.cls[k],
                          {s: EDGE_SIGN[edges[k].get(s)] for s in SIDES})
          for k in pieces.tpl}
    sol = jigsaw.solve_soft(cell_cls, jp, n=N)
    rule = {f"{r},{c}": k for (r, c), k in sol.assignment.items()}
    print(f"  rule solver: {len(rule)} cells, "
          f"{jigsaw.violations(cell_cls, jp, sol.assignment, n=N)} rule(s) broken")
    if len(rule) > len(plan):
        plan = rule
    print(f"\nassignment: {len(plan)} cells")
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
    refused: Counter = Counter()
    # (piece, cell) pairs already attempted.  Read from disk and written back, because the pairs a
    # single run can retire are few: the plan keeps proposing the same piece for the same cell,
    # and with a fresh set each time the next run repeats the refusals instead of moving on to the
    # other copy of that class -- of which the exam always gives one that does not fit (a decoy).
    TRIED = vision.ROOT / "logs" / "tried.json"
    tried: dict[tuple[str, str], int] = {}
    if TRIED.exists():
        try:
            raw = json.loads(TRIED.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raw = []
        for row in raw:
            # older runs wrote a bare [piece, cell] pair; newer ones add the refusal count
            piece, cell = row[0], row[1]
            tried[(piece, cell)] = row[2] if len(row) > 2 else 1
        print(f"  {len(tried)} (piece, cell) pair(s) refused in earlier runs")
    # One refusal does not condemn a pair.  The lock test itself is not perfect -- an audit of a
    # finished board counted 1 or 2 fewer cells than the loop had confirmed -- and when it calls a
    # good placement refused, the piece is dragged straight back off the cell.  Banning the pair
    # forever turned that misreading into a cell that could never be filled again.
    TRIES_ALLOWED = 2

    print("walking the ring over the cells still open")
    for attempt in range(args.passes):
        progress = tries = 0
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
            # The planned piece first, then at most two others of the same class that have not
            # already been refused here.  The exam gives a decoy of a class it also needs, so the
            # plan's first choice can be the one that does not fit this cell, and with a single
            # candidate the run would keep re-proposing it instead of trying the other copy.
            order = ([plan[key]] if key in plan and plan[key] not in pieces.used else [])
            order = [k for k in order if tried.get((k, key), 0) < TRIES_ALLOWED]
            if not order:
                order = [k for k in pieces.spare(wanted)
                         if tried.get((k, key), 0) < TRIES_ALLOWED][:2]
            if not order:
                # Last resort for a cell whose own piece the reader labelled as something else:
                # offer *any* untried piece whose shape does not rule the cell out.  Bounded on
                # purpose -- two per cell per pass -- and the refused pairs live on disk, so the
                # next run carries on from where this one stopped instead of repeating them.
                def shape_allows(pk: str) -> bool:
                    for side in SIDES:
                        if EDGE_SIGN[edges[pk].get(side)] != 0:
                            continue
                        on_edge = {"top": r == 0, "bottom": r == N - 1,
                                   "left": c == 0, "right": c == N - 1}[side]
                        if not on_edge:
                            return False
                    return True

                pool = [k for k in pieces.spare()
                        if tried.get((k, key), 0) < TRIES_ALLOWED]
                strict = [k for k in pool if shape_allows(k)]
                # shape first, but never to the point of having nothing to offer: the reader
                # misses flat sides far more often than it invents them (25 pieces produced 0
                # flat lefts), so demanding a flat bottom on a bottom-row cell excluded every
                # candidate and the cell was then declared untryable.
                order = (strict or pool)[:2]
            if not order:
                print(f"    cell({key}) {wanted}: nothing left to try -- cell left empty")
                continue
            done = False
            for pk in order:
                tries += 1
                if moves >= args.max_moves:
                    break
                box, sc = locate(snap, pieces.tpl[pk])
                if box is None:
                    # Do NOT retire the piece.  Not finding it is a failure of the search, or of
                    # where it was last set down -- not proof that it left the game.  Retiring it
                    # is what makes a piece vanish for good: three of thirty were dropped this way
                    # on a live board, and their cells could never be filled again, which is
                    # exactly the "one piece never goes in" symptom.
                    print(f"    {pk}: not found this time (best {sc:.2f}) -- will look again later")
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
                # Let the snap finish before asking whether the piece can be lifted: probing while
                # the engine is still animating the drop gets a "cannot lift" answer for a drop
                # that has not actually taken, which is how a cell got credited without a piece.
                time.sleep(0.28)
                # Press the piece where it actually is, not where the cell is.  A dropped piece
                # that did not snap lies near the cell but not on it, and five presses around the
                # cell's centre then all land on empty board: nothing moves, which reads as
                # "cannot lift" and credits a cell that has no piece in it.  That is the one cell
                # by which the loop kept overrunning its own audit.
                probe = after.cells[key]
                where, _w = locate(after, pieces.tpl[pk])
                if where is not None:
                    probe = where
                locked = press_is_locked(mouse, probe)
                if not locked:
                    # The same probe gives different answers a second apart: the snap animation is
                    # still settling.  Ask again once it has calmed down and take the later answer.
                    time.sleep(0.45)
                    locked = press_is_locked(mouse, probe)
                if locked:
                    confirmed[key] = pk
                    pieces.used.add(pk)
                    progress += 1
                    print(f"    {pk} ({pieces.cls[pk]}) -> cell({key})  [move {moves}]  "
                          f"LOCKED  {len(confirmed)}/{N * N}")
                    done = True
                    break
                refused[(pieces.cls[pk], wanted)] += 1
                tried[(pk, key)] = tried.get((pk, key), 0) + 1
                TRIED.write_text(json.dumps([[a, b, v] for (a, b), v in sorted(tried.items())]),
                                 encoding="utf-8")
                print(f"    {pk} ({pieces.cls[pk]}) -> cell({key}) wanted {wanted}  "
                      f"[move {moves}]  refused (fingerprint {here:.2f})")
                # The user's rule, and the reason it matters beyond tidiness: take the piece back
                # to *its own* slot, not to "some free slot".  A piece parked somewhere else
                # drifts -- the drop lands a few pixels off the lattice, and the only way this
                # program finds a piece again is by fingerprint against the 36 known slots, so a
                # piece nudged away from its slot is a piece it can no longer see.  That is the
                # failure that leaves one cell unsolved for the rest of a game.
                mouse.grab_at(centre(after.cells[key]))
                mouse.glide((sx, sy))              # exactly where it was picked up from
                mouse.release()
                moves += 1
                time.sleep(0.18)
                back = Snap(mouse.shot())
                if not fp.score_at(back.img, pieces.tpl[pk], box) > 0.8:
                    print(f"      -> taken back to ({sx},{sy}), but it is not there any more")
                else:
                    print(f"      -> taken back to its own slot at ({sx},{sy})  [move {moves}]")
                break            # one piece per cell per pass: no candidate roulette
        if progress == 0:
            print(f"  no progress in this pass, {tries} attempt(s) made")
            if tries == 0:
                break          # nothing left to offer anywhere; another pass would be identical

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

