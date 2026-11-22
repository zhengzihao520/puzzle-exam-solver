"""Verified vision layer for puzzle.moyunliuyin.me (exam mode).

Everything here was measured on real screenshots, not guessed:

* board          (463, 460, 530, 418)  -- N x N cells, N comes from the reference panel
* reference      (1436, 509, 451, 338) -- N x N same-class *other* photos (no pixel match)
* progress       (1540, 225, 1790, 285) -- "杩涘害: k/N" text, the only澶栭儴 success truth
* tray           two side bands, 3 columns each, row count varies per part
                  (3x5 = 15 slots per band for part 1/2, 3x6 = 18 slots for part 3)

Tray cells are recovered from the brightness profiles: the gaps between pieces are
the only dark thing on a band, so a cell rect is the span between two dark runs.
A cell is empty when its mean brightness stays below EMPTY_LEVEL.

Classification is a YOLO11s classifier trained on the site's own catalog
(runs/cls, val top1 0.96).  Pieces are whole photos scaled into a 112x85 cell, so
the classifier input is the cell rect itself; a 3-view ensemble (imgsz 160/224 +
hflip) keeps it stable against JPEG/scale noise: 29/30 on the labelled real pieces.
Pure white cells are the decoy class (blank) and are detected by brightness, which
is exact, instead of by the network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "cls" / "weights" / "best.pt"

BOARD = (433, 453, 560, 425)      # measured: 5 cells of 112 x 85, frame included
REFERENCE = (1436, 509, 451, 338)
PROGRESS = (1540, 225, 1790, 285)
TRAY_REGIONS = {"left": (10, 460, 420, 1000), "right": (990, 1440, 420, 1000)}

GAP_LEVEL = 60.0        # darker than this on a band profile = gap between pieces
EMPTY_LEVEL = 45.0      # a tray cell darker than this holds no piece
EMPTY_P99 = 20.0        # ... and its 99th percentile is still near black
BOARD_FILLED_LEVEL = 60.0  # board cell brighter than this shows a placed piece
BLANK_BLOCK_FRACTION = 0.88
WHITE_PIECE_FRACTION = 0.65    # this share of near-white pixels marks the white decoy piece
MIN_RUN = 6
CLASSES = ["dog", "cat", "elephant", "desktop", "bicycle", "car", "chair", "cup",
           "airplane", "backpack", "umbrella", "suitcase", "penguin", "tafei", "nailong"]
BLANK = "blank"


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def cells(self, n: int, pad: int = 1) -> list[tuple[int, int, tuple[int, int, int, int]]]:
        cw, ch = self.w / n, self.h / n
        out = []
        for r in range(n):
            for c in range(n):
                x0, y0 = int(self.x + c * cw) + pad, int(self.y + r * ch) + pad
                x1, y1 = int(self.x + (c + 1) * cw) - pad, int(self.y + (r + 1) * ch) - pad
                out.append((r, c, (x0, y0, x1 - x0, y1 - y0)))
        return out


def grab() -> np.ndarray:
    import pyautogui
    return cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)


def sub(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return img[y:y + h, x:x + w]



# ---- calibrated geometry, expressed relative to the reference panel --------
# Measured on the calibrated screenshot: reference panel at (1435, 517) 451x339,
# board at (463, 460) 530x418, tray bands at x 10..460 / 990..1440, y 420..1000.
# Everything is re-derived from the panel on every grab, so browser scroll and zoom
# no longer break the constants.
CAL_REF = (451.0, 339.0)
BOARD_REL = (-1002.0, -64.0, 560.0, 425.0)
TRAY_REL = {"left": (-1425.0, -975.0, -97.0, 483.0),
            "right": (-445.0, 5.0, -97.0, 483.0)}


def layout(img: np.ndarray) -> tuple[Rect, Rect, dict[str, tuple[int, int, int, int]]]:
    """(reference, board, tray regions) for whatever scroll/zoom the page is at."""
    ref = find_reference(img)
    sx, sy = ref.w / CAL_REF[0], ref.h / CAL_REF[1]

    def board_box() -> Rect:
        dx, dy, dw, dh = BOARD_REL
        return Rect(int(ref.x + dx * sx), int(ref.y + dy * sy),
                    int(dw * sx), int(dh * sy))

    regions = {}
    for band, (dx0, dx1, dy0, dy1) in TRAY_REL.items():
        regions[band] = (int(ref.x + dx0 * sx), int(ref.x + dx1 * sx),
                         int(ref.y + dy0 * sy), int(ref.y + dy1 * sy))
    return ref, board_box(), regions


def tray_region(img: np.ndarray, band: str) -> tuple[int, int, int, int]:
    return layout(img)[2][band]


def _gap_runs(profile: np.ndarray, threshold: float = 0.75) -> list[tuple[int, int]]:
    """Runs where the profile stays above `threshold` (a real gap spans the whole band)."""
    mask = profile > threshold
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= MIN_RUN:
                out.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= MIN_RUN:
        out.append((start, len(mask) - 1))
    return out


def _spans(dark: list[tuple[int, int]], offset: int) -> list[tuple[int, int]]:
    """Cell spans between consecutive gap runs (inclusive bounds, absolute)."""
    return [(a1 + 1 + offset, b0 - 1 + offset)
            for (a0, a1), (b0, _b1) in zip(dark, dark[1:]) if b0 - a1 >= 6]


def _keep_regular(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop UI junk: keep only spans whose length is close to the median length."""
    if not spans:
        return []
    sizes = np.array([b - a + 1 for a, b in spans])
    med = float(np.median(sizes))
    return [s for s, n in zip(spans, sizes) if abs(n - med) <= med * 0.22]


def _dark_frac(g: np.ndarray) -> np.ndarray:
    """Fraction of pixels darker than GAP_LEVEL, per column (axis=0) or row (axis=1)."""
    return np.zeros(0) if g.size == 0 else (g < GAP_LEVEL).astype(np.float32)


def tray_rows(img: np.ndarray) -> list[tuple[int, int]]:
    """Row spans shared by both bands (they sit in one grid, so rows must agree)."""
    boxes = [tray_region(img, b) for b in ("left", "right")]
    y0, y1 = boxes[0][2], boxes[0][3]
    stacked = np.concatenate([cv2.cvtColor(img[b[2]:b[3], b[0]:b[1]], cv2.COLOR_BGR2GRAY)
                              for b in boxes], axis=1)
    dark = _dark_frac(stacked.astype(np.float32))
    return _keep_regular(_spans(_gap_runs(dark.mean(axis=1)), y0))


def tray_cols(img: np.ndarray, band: str) -> list[tuple[int, int]]:
    """Column spans, measured only over the rows that actually hold pieces.

    Using the whole band region breaks as soon as the upper rows are empty: every column
    then looks mostly dark and no gap is ever found.  Restricting to the detected row span
    keeps the column profile meaningful while the tray empties out.
    """
    x0, x1, y0, y1 = tray_region(img, band)
    rows = tray_rows(img)
    if rows:
        y0, y1 = rows[0][0], rows[-1][1]
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    return _keep_regular(_spans(_gap_runs(_dark_frac(g).mean(axis=0)), x0))


# ---- the tray lattice, computed the way the engine computes it -------------
# recon/puzzle_board.js traySlots():
#
#     stepX = pieceWidth * 1.16        slot.x = originX + col * stepX + (stepX - pieceWidth) / 2
#     stepY = pieceHeight * 1.16       slot.y = row * stepY + (stepY - pieceHeight) / 2
#     originX = side === 0 ? 0 : boardX + boardWidth
#     columnsPerSide = floor(boardX / stepX),  slotRows = floor(canvasHeight / stepY)
#
# Measuring that lattice from the dark gaps works only while the tray is full: once most
# pieces are placed the gaps are gone and the grid collapses (it was reading three rows of
# half-height slots).  The pitch and the phase are constants of the layout, so compute them
# instead of detecting them.  Offsets are relative to the reference panel, like everything
# else, so they survive scrolling and zoom.
TRAY_STEP = (129.92, 98.6)          # pieceWidth * 1.16, pieceHeight * 1.16 for a 5x5
TRAY_PIECE = (112.0, 85.0)
TRAY_TOP_REL = -82.0                # first slot row, relative to the reference panel
TRAY_COL_REL = {"left": -1406.0, "right": -433.0}
TRAY_COLS = 3


def tray_slots_all(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Every slot the tray can have, rather than the one geometry that was chosen.

    calibrate_tray picks a single (top offset, row count) by scoring occupied slots, and it can
    lose a whole row: with exactly thirty pieces a five-row layout "holds" thirty, so it was
    preferred over the six-row layout the pieces were actually in, and every piece in the first
    row became invisible -- not in the tray as far as the program could tell, not on the board
    either.  A piece the program cannot see is a cell that can never be filled.

    So enumerate the plausible offsets, take the union, and clip it to above the taskbar: the
    icons there are brighter than any gap and a lattice that reaches into them invents pieces.
    """
    ref = find_reference(img)
    _board = layout(img)[1]
    bottom_limit = _board.y + _board.h + 40
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for band in ("left", "right"):
        for top_rel in (TRAY_TOP_REL, TRAY_TOP_REL - TRAY_STEP[1]):
            for _r, _c, box in lattice_slots(img, band, 6, top_rel):
                x, y, w, h = box
                if x < 0 or y < 0 or x + w > img.shape[1]:
                    continue
                if y + h > bottom_limit:          # the taskbar is not part of the tray
                    continue
                out[(x, y)] = box
    return list(out.values())


def lattice_slots(img: np.ndarray, band: str,
                  rows: int | None = None,
                  top_rel: float | None = None
                  ) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """(row, col, (x, y, w, h)) for a whole tray band, whatever is left in it."""
    ref = find_reference(img)
    sx, sy = ref.w / CAL_REF[0], ref.h / CAL_REF[1]
    x0 = ref.x + TRAY_COL_REL[band] * sx
    y0 = ref.y + (TRAY_TOP_REL if top_rel is None else top_rel) * sy
    rows = 6 if rows is None else rows
    pw, ph = TRAY_PIECE[0] * sx, TRAY_PIECE[1] * sy
    step_x, step_y = TRAY_STEP[0] * sx, TRAY_STEP[1] * sy
    return [(r, c, (round(x0 + c * step_x), round(y0 + r * step_y),
                    round(pw), round(ph)))
            for r in range(rows) for c in range(TRAY_COLS)]


def calibrate_tray(img: np.ndarray, expected: int = 30, verbose: bool = False):
    """Find the tray lattice that can actually hold every piece.

    slotRows = floor(canvasHeight / stepY) depends on the window: this machine shows six
    rows where an earlier session showed five, and a hard-coded five reported five pieces
    missing that were sitting in the row it never looked at.  Rather than trust a constant,
    try the plausible geometries and keep the smallest one that accounts for `expected`
    pieces.
    """
    best = None
    for top_rel in (TRAY_TOP_REL, TRAY_TOP_REL + TRAY_STEP[1], TRAY_TOP_REL - TRAY_STEP[1]):
        for rows in (4, 5, 6, 7, 8):
            slots = [box for band in ("left", "right")
                     for _r, _c, box in lattice_slots(img, band, rows, top_rel)]
            slots = [b for b in slots
                     if b[0] >= 0 and b[1] >= 0
                     and b[0] + b[2] <= img.shape[1] and b[1] + b[3] <= img.shape[0]]
            occ = sum(1 for b in slots if not is_empty(sub(img, b)))
            score = (occ >= expected, occ, -len(slots))
            if best is None or score > best[0]:
                best = (score, top_rel, rows, slots, occ)
    _score, top_rel, rows, slots, occ = best
    if verbose:
        print(f"tray lattice: {rows} rows x {TRAY_COLS} cols per band, "
              f"top offset {top_rel:+.0f} px, {occ} of {len(slots)} slots hold a piece")
    return slots, rows, top_rel, occ


def tray_slots(img: np.ndarray, band: str,
               rows: list[tuple[int, int]] | None = None
               ) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """Deprecated gap-detected slots, kept because older scripts import it.

    It collapses as the tray empties -- use lattice_slots() instead.
    """
    rows = tray_rows(img) if rows is None else rows
    cols = tray_cols(img, band)
    return [(ri, ci, (cx0, ry0, cx1 - cx0 + 1, ry1 - ry0 + 1))
            for ri, (ry0, ry1) in enumerate(rows)
            for ci, (cx0, cx1) in enumerate(cols)]

def cell_of(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    return sub(img, box)


def is_empty(cell: np.ndarray) -> bool:
    """An empty slot is flat dark tray background.

    Measured on real frames: empty slots have mean 9-13 and a 99th percentile of
    8-9 (a handful of stray bright pixels survive the drag animation), while every
    photo piece -- even a night shot -- has a 99th percentile well above 60.
    A plain mean/std test misread those stray pixels as pieces.

    The test runs on the middle half of the slot on purpose.  A piece's tabs reach
    ~19 px past its box, so a full-box crop of an *empty* slot still catches a
    neighbour's tab and reads as occupied -- which is how the solver came to believe
    the tray had no free slots while eleven of them were empty.
    """
    h, w = cell.shape[:2]
    if h > 8 and w > 8:
        cell = cell[h // 4:h - h // 4, w // 4:w - w // 4]
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    return float(g.mean()) < EMPTY_LEVEL and float(np.percentile(g, 99)) < EMPTY_P99


def is_blank(cell: np.ndarray) -> bool:
    """Reference-panel blank cell: the whole thumbnail is white, centre included."""
    return _centre_white(cell) >= 0.995


BG_LEVEL = 18.0          # the tray background is (9, 9, 9); the border line is ~10-14


def _bg_share(a: np.ndarray) -> float:
    return 0.0 if a.size == 0 else float((a < BG_LEVEL).mean())


def flat_sides(img: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, str]:
    """Per side: 'flat' (straight frame edge), 'tab' (protruding), 'notch' (carved in).

    The site cuts real jigsaw pieces (PieceCutter: tabSize = 0.22*min(w,h), margin =
    1.5*tabSize), so a piece on the board frame has a straight side there.  Measured
    from the tray: a tab shows content 3-13 px *outside* the body, a notch shows pure
    tray background 3-12 px *inside* it, and a flat side shows neither.  Decision is by
    share of background pixels, not brightness: the piece carries a photo, which may be
    darker than the tray itself.
    """
    x, y, w, h = box
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x1, x2 = x + w // 3, x + 2 * w // 3
    y1, y2 = y + h // 3, y + 2 * h // 3

    def kind(inner_bg: float, outer_bg: float) -> str:
        if inner_bg > 0.5:
            return "notch"
        if outer_bg < 0.5:
            return "tab"
        return "flat"

    return {
        "top": kind(_bg_share(g[y + 3:y + 12, x1:x2]),
                    _bg_share(g[max(0, y - 13):y - 3, x1:x2])),
        "bottom": kind(_bg_share(g[y + h - 12:y + h - 3, x1:x2]),
                       _bg_share(g[y + h + 3:y + h + 13, x1:x2])),
        "left": kind(_bg_share(g[y1:y2, x + 3:x + 12]),
                     _bg_share(g[y1:y2, max(0, x - 13):x - 3])),
        "right": kind(_bg_share(g[y1:y2, x + w - 12:x + w - 3]),
                      _bg_share(g[y1:y2, x + w + 3:x + w + 13])),
    }


def is_real_piece(img: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Reject slot boxes that are really a neighbour's tab bleeding over the gap."""
    x, y, w, h = box
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    core = g[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
    return _bg_share(core) <= 0.25


def _centre_white(cell: np.ndarray) -> float:
    """Share of near-white pixels in the middle half of the cell."""
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    core = g[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return 0.0 if core.size == 0 else float((core > 190).mean())


def is_white_piece(cell: np.ndarray) -> bool:
    """White decoy piece: the *centre* is white, not just the background.

    Measured on a real tray: the five decoys all score 1.00 on the middle half, while
    product photos shot on white (desktop computer, cup, suitcase) score 0.02-0.70
    because the object sits in the middle.  A plain "share of white pixels" test cannot
    tell those apart -- it flagged 10 pieces where only 5 exist.
    """
    return _centre_white(cell) >= 0.995



class Classifier:
    """YOLO11s-cls with a flip/scale ensemble for stability."""

    def __init__(self, weights: Path = WEIGHTS, imgsz: tuple[int, ...] = (160, 224)) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.imgsz = imgsz

    def probs(self, cell: np.ndarray) -> np.ndarray:
        out = []
        for size in self.imgsz:
            for flip in (False, True):
                view = cv2.flip(cell, 1) if flip else cell
                res = self.model.predict(view, imgsz=size, verbose=False)[0]
                out.append(res.probs.data.cpu().numpy())
        return np.mean(out, axis=0)

    def label(self, cell: np.ndarray) -> tuple[str, float]:
        if is_blank(cell):
            return BLANK, 1.0
        p = self.probs(cell)
        return str(self.names[int(p.argmax())]), float(p.max())


def find_reference(img: np.ndarray) -> Rect:
    """Right-hand reference panel: the largest bright blob right of the board.

    Detected rather than hard coded because the page shifts a few pixels between
    games; the measured box (1435, 517, 451, 339) is the fallback.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = cv2.morphologyEx((g > 150).astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((9, 9), np.uint8))
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(bright)
    best = None
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if x < 900 or w < 200 or area < 30000:
            continue
        if not 1.2 <= w / h <= 1.45:
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    return Rect(*best[:4]) if best else Rect(*REFERENCE)


def reference_grid(img: np.ndarray, n: int, clf: Classifier) -> list[str]:
    """Required class per board cell, row-major (index = row * n + col)."""
    ref = find_reference(img)
    return [clf.label(sub(img, box))[0] for _r, _c, box in ref.cells(n)]
PIECE_AREA = 9520          # 112 x 85, the area of one tray piece


def tray_pieces(img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Locate tray pieces by connected components.

    The gap-grid works while the tray is full, but once most slots are empty the
    gaps disappear and the grid cannot be recovered.  Components stay reliable in
    both cases; blobs that clearly hold two touching pieces are split.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for band in ("left", "right"):
        x0, x1, y0, y1 = tray_region(img, band)
        mask = cv2.morphologyEx((g[y0:y1, x0:x1] > 40).astype(np.uint8),
                                cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i][:5])
            if area < PIECE_AREA * 0.45:
                continue
            count = max(1, round(area / PIECE_AREA))
            step = (w if w >= h else h) / count
            for k in range(count):
                if w >= h:
                    bx, by, bw, bh = x0 + x + int(k * step), y0 + y, int(step), h
                else:
                    bx, by, bw, bh = x0 + x, y0 + y + int(k * step), w, int(step)
                out.append((band, (bx, by, bw, bh)))
    return out


def board_filled(img: np.ndarray, n: int) -> list[bool]:
    """A board cell is filled once it shows a photo instead of the dark placeholder.

    Measured: the placeholder sits at mean 23-27 (a few cells carry a faint ghost,
    std up to 30), while the darkest real piece still averages 73 -- so the board
    needs a brightness test, not the flatness test used for tray slots.
    """
    board = layout(img)[1]
    out = []
    for _r, _c, box in board.cells(n):
        g = cv2.cvtColor(sub(img, box), cv2.COLOR_BGR2GRAY)
        out.append(float(g.mean()) >= BOARD_FILLED_LEVEL)
    return out


# ---- progress counter: the only authoritative success signal -----------------
# "杩涘害: k/N" sits just above the reference panel; k increases exactly once per piece
# that is dropped on its own cell, so a change in this crop *is* the verdict.
PROGRESS_REL = (-235, -292, 470, 44)     # the whole "杩涘害: k/N" text row, relative to the reference panel


def progress_box(img: np.ndarray) -> tuple[int, int, int, int]:
    ref = find_reference(img)
    dx, dy, w, h = PROGRESS_REL
    x, y = ref.x + dx, ref.y + dy
    x = max(0, min(x, img.shape[1] - w))
    y = max(0, min(y, img.shape[0] - h))
    return x, y, w, h


def progress_crop(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(sub(img, progress_box(img)), cv2.COLOR_BGR2GRAY)


def ticking_columns(a: np.ndarray, b: np.ndarray, thresh: int = 30) -> set[int]:
    """Columns that change on their own (the running timer sits on the same text row)."""
    if a.shape != b.shape:
        return set()
    cols = (cv2.absdiff(a, b) > 40).any(axis=0)
    return set(np.nonzero(cols)[0].tolist())


def progress_changed(a: np.ndarray, b: np.ndarray, ignore: set[int], thresh: int = 8) -> bool:
    """True when something other than the ticking columns changed.

    The counter shares its row with a live timer, so raw frame comparison always reports a
    change.  Columns that move while the page is idle are masked out first, and what is
    left is the "杩涘害: k/N" text.
    """
    if a.shape != b.shape:
        return False
    d = cv2.absdiff(a, b) > 40
    if ignore:
        d[:, sorted(ignore)] = False
    return int(d.sum()) > thresh


def read_progress(img: np.ndarray) -> str:
    """Raw crop of the progress label; the caller compares it for changes."""
    return sub(img, PROGRESS).tobytes()




def has_piece(cell: np.ndarray, floor: float = 30.0) -> bool:
    """Does this cell hold a piece?  Middle half, brightness floor, per-context.

    The two contexts have different floors, and mixing them up is how two pieces ended up
    stacked in one slot:

      board cell   empty placeholder 22.8-23.3 (a few carry a faint ghost, std up to 30)
                   darkest placed piece 36.6                      -> floor 30
      tray slot    empty 8-9, pieces 34-255                          -> floor 20

    The middle half is what a piece always covers and a neighbour's tab never reaches.
    """
    h, w = cell.shape[:2]
    if h > 12 and w > 12:
        cell = cell[h // 4:h - h // 4, w // 4:w - w // 4]
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    return float(g.mean()) >= floor


def is_blank(cell: np.ndarray) -> bool:
    """Reference-panel blank cell: the whole thumbnail is white, centre included."""
    return _centre_white(cell) >= 0.995


BG_LEVEL = 18.0          # the tray background is (9, 9, 9); the border line is ~10-14


def _bg_share(a: np.ndarray) -> float:
    return 0.0 if a.size == 0 else float((a < BG_LEVEL).mean())


def flat_sides(img: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, str]:
    """Per side: 'flat' (straight frame edge), 'tab' (protruding), 'notch' (carved in).

    The site cuts real jigsaw pieces (PieceCutter: tabSize = 0.22*min(w,h), margin =
    1.5*tabSize), so a piece on the board frame has a straight side there.  Measured
    from the tray: a tab shows content 3-13 px *outside* the body, a notch shows pure
    tray background 3-12 px *inside* it, and a flat side shows neither.  Decision is by
    share of background pixels, not brightness: the piece carries a photo, which may be
    darker than the tray itself.
    """
    x, y, w, h = box
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x1, x2 = x + w // 3, x + 2 * w // 3
    y1, y2 = y + h // 3, y + 2 * h // 3

    def kind(inner_bg: float, outer_bg: float) -> str:
        if inner_bg > 0.5:
            return "notch"
        if outer_bg < 0.5:
            return "tab"
        return "flat"

    return {
        "top": kind(_bg_share(g[y + 3:y + 12, x1:x2]),
                    _bg_share(g[max(0, y - 13):y - 3, x1:x2])),
        "bottom": kind(_bg_share(g[y + h - 12:y + h - 3, x1:x2]),
                       _bg_share(g[y + h + 3:y + h + 13, x1:x2])),
        "left": kind(_bg_share(g[y1:y2, x + 3:x + 12]),
                     _bg_share(g[y1:y2, max(0, x - 13):x - 3])),
        "right": kind(_bg_share(g[y1:y2, x + w - 12:x + w - 3]),
                      _bg_share(g[y1:y2, x + w + 3:x + w + 13])),
    }


def is_real_piece(img: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Reject slot boxes that are really a neighbour's tab bleeding over the gap."""
    x, y, w, h = box
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    core = g[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
    return _bg_share(core) <= 0.25


def _centre_white(cell: np.ndarray) -> float:
    """Share of near-white pixels in the middle half of the cell."""
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    core = g[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return 0.0 if core.size == 0 else float((core > 190).mean())


def is_white_piece(cell: np.ndarray) -> bool:
    """White decoy piece: the *centre* is white, not just the background.

    Measured on a real tray: the five decoys all score 1.00 on the middle half, while
    product photos shot on white (desktop computer, cup, suitcase) score 0.02-0.70
    because the object sits in the middle.  A plain "share of white pixels" test cannot
    tell those apart -- it flagged 10 pieces where only 5 exist.
    """
    return _centre_white(cell) >= 0.995



class Classifier:
    """YOLO11s-cls with a flip/scale ensemble for stability."""

    def __init__(self, weights: Path = WEIGHTS, imgsz: tuple[int, ...] = (160, 224)) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.imgsz = imgsz

    def probs(self, cell: np.ndarray) -> np.ndarray:
        out = []
        for size in self.imgsz:
            for flip in (False, True):
                view = cv2.flip(cell, 1) if flip else cell
                res = self.model.predict(view, imgsz=size, verbose=False)[0]
                out.append(res.probs.data.cpu().numpy())
        return np.mean(out, axis=0)

    def label(self, cell: np.ndarray) -> tuple[str, float]:
        if is_blank(cell):
            return BLANK, 1.0
        p = self.probs(cell)
        return str(self.names[int(p.argmax())]), float(p.max())


def find_reference(img: np.ndarray) -> Rect:
    """Right-hand reference panel: the largest bright blob right of the board.

    Detected rather than hard coded because the page shifts a few pixels between
    games; the measured box (1435, 517, 451, 339) is the fallback.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = cv2.morphologyEx((g > 150).astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((9, 9), np.uint8))
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(bright)
    best = None
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if x < 900 or w < 200 or area < 30000:
            continue
        if not 1.2 <= w / h <= 1.45:
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    return Rect(*best[:4]) if best else Rect(*REFERENCE)


def reference_grid(img: np.ndarray, n: int, clf: Classifier) -> list[str]:
    """Required class per board cell, row-major (index = row * n + col)."""
    ref = find_reference(img)
    return [clf.label(sub(img, box))[0] for _r, _c, box in ref.cells(n)]
PIECE_AREA = 9520          # 112 x 85, the area of one tray piece


def tray_pieces(img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Locate tray pieces by connected components.

    The gap-grid works while the tray is full, but once most slots are empty the
    gaps disappear and the grid cannot be recovered.  Components stay reliable in
    both cases; blobs that clearly hold two touching pieces are split.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for band in ("left", "right"):
        x0, x1, y0, y1 = tray_region(img, band)
        mask = cv2.morphologyEx((g[y0:y1, x0:x1] > 40).astype(np.uint8),
                                cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i][:5])
            if area < PIECE_AREA * 0.45:
                continue
            count = max(1, round(area / PIECE_AREA))
            step = (w if w >= h else h) / count
            for k in range(count):
                if w >= h:
                    bx, by, bw, bh = x0 + x + int(k * step), y0 + y, int(step), h
                else:
                    bx, by, bw, bh = x0 + x, y0 + y + int(k * step), w, int(step)
                out.append((band, (bx, by, bw, bh)))
    return out


def board_filled(img: np.ndarray, n: int) -> list[bool]:
    """A board cell is filled once it shows a photo instead of the dark placeholder.

    Measured: the placeholder sits at mean 23-27 (a few cells carry a faint ghost,
    std up to 30), while the darkest real piece still averages 73 -- so the board
    needs a brightness test, not the flatness test used for tray slots.
    """
    board = layout(img)[1]
    out = []
    for _r, _c, box in board.cells(n):
        g = cv2.cvtColor(sub(img, box), cv2.COLOR_BGR2GRAY)
        out.append(float(g.mean()) >= BOARD_FILLED_LEVEL)
    return out


# ---- progress counter: the only authoritative success signal -----------------
# "杩涘害: k/N" sits just above the reference panel; k increases exactly once per piece
# that is dropped on its own cell, so a change in this crop *is* the verdict.
PROGRESS_REL = (-235, -292, 470, 44)     # the whole "杩涘害: k/N" text row, relative to the reference panel


def progress_box(img: np.ndarray) -> tuple[int, int, int, int]:
    ref = find_reference(img)
    dx, dy, w, h = PROGRESS_REL
    x, y = ref.x + dx, ref.y + dy
    x = max(0, min(x, img.shape[1] - w))
    y = max(0, min(y, img.shape[0] - h))
    return x, y, w, h


def progress_crop(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(sub(img, progress_box(img)), cv2.COLOR_BGR2GRAY)


def ticking_columns(a: np.ndarray, b: np.ndarray, thresh: int = 30) -> set[int]:
    """Columns that change on their own (the running timer sits on the same text row)."""
    if a.shape != b.shape:
        return set()
    cols = (cv2.absdiff(a, b) > 40).any(axis=0)
    return set(np.nonzero(cols)[0].tolist())


def progress_changed(a: np.ndarray, b: np.ndarray, ignore: set[int], thresh: int = 8) -> bool:
    """True when something other than the ticking columns changed.

    The counter shares its row with a live timer, so raw frame comparison always reports a
    change.  Columns that move while the page is idle are masked out first, and what is
    left is the "杩涘害: k/N" text.
    """
    if a.shape != b.shape:
        return False
    d = cv2.absdiff(a, b) > 40
    if ignore:
        d[:, sorted(ignore)] = False
    return int(d.sum()) > thresh


def read_progress(img: np.ndarray) -> str:
    """Raw crop of the progress label; the caller compares it for changes."""
    return sub(img, PROGRESS).tobytes()




def is_blank(cell: np.ndarray) -> bool:
    """Reference-panel blank cell: the whole thumbnail is white, centre included."""
    return _centre_white(cell) >= 0.995


BG_LEVEL = 18.0          # the tray background is (9, 9, 9); the border line is ~10-14


def _bg_share(a: np.ndarray) -> float:
    return 0.0 if a.size == 0 else float((a < BG_LEVEL).mean())


def flat_sides(img: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, str]:
    """Per side: 'flat' (straight frame edge), 'tab' (protruding), 'notch' (carved in).

    The site cuts real jigsaw pieces (PieceCutter: tabSize = 0.22*min(w,h), margin =
    1.5*tabSize), so a piece on the board frame has a straight side there.  Measured
    from the tray: a tab shows content 3-13 px *outside* the body, a notch shows pure
    tray background 3-12 px *inside* it, and a flat side shows neither.  Decision is by
    share of background pixels, not brightness: the piece carries a photo, which may be
    darker than the tray itself.
    """
    x, y, w, h = box
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x1, x2 = x + w // 3, x + 2 * w // 3
    y1, y2 = y + h // 3, y + 2 * h // 3

    def kind(inner_bg: float, outer_bg: float) -> str:
        if inner_bg > 0.5:
            return "notch"
        if outer_bg < 0.5:
            return "tab"
        return "flat"

    return {
        "top": kind(_bg_share(g[y + 3:y + 12, x1:x2]),
                    _bg_share(g[max(0, y - 13):y - 3, x1:x2])),
        "bottom": kind(_bg_share(g[y + h - 12:y + h - 3, x1:x2]),
                       _bg_share(g[y + h + 3:y + h + 13, x1:x2])),
        "left": kind(_bg_share(g[y1:y2, x + 3:x + 12]),
                     _bg_share(g[y1:y2, max(0, x - 13):x - 3])),
        "right": kind(_bg_share(g[y1:y2, x + w - 12:x + w - 3]),
                      _bg_share(g[y1:y2, x + w + 3:x + w + 13])),
    }


def is_real_piece(img: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Reject slot boxes that are really a neighbour's tab bleeding over the gap."""
    x, y, w, h = box
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    core = g[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
    return _bg_share(core) <= 0.25


def _centre_white(cell: np.ndarray) -> float:
    """Share of near-white pixels in the middle half of the cell."""
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    core = g[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return 0.0 if core.size == 0 else float((core > 190).mean())


def is_white_piece(cell: np.ndarray) -> bool:
    """White decoy piece: the *centre* is white, not just the background.

    Measured on a real tray: the five decoys all score 1.00 on the middle half, while
    product photos shot on white (desktop computer, cup, suitcase) score 0.02-0.70
    because the object sits in the middle.  A plain "share of white pixels" test cannot
    tell those apart -- it flagged 10 pieces where only 5 exist.
    """
    return _centre_white(cell) >= 0.995



class Classifier:
    """YOLO11s-cls with a flip/scale ensemble for stability."""

    def __init__(self, weights: Path = WEIGHTS, imgsz: tuple[int, ...] = (160, 224)) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.imgsz = imgsz

    def probs(self, cell: np.ndarray) -> np.ndarray:
        out = []
        for size in self.imgsz:
            for flip in (False, True):
                view = cv2.flip(cell, 1) if flip else cell
                res = self.model.predict(view, imgsz=size, verbose=False)[0]
                out.append(res.probs.data.cpu().numpy())
        return np.mean(out, axis=0)

    def label(self, cell: np.ndarray) -> tuple[str, float]:
        if is_blank(cell):
            return BLANK, 1.0
        p = self.probs(cell)
        return str(self.names[int(p.argmax())]), float(p.max())


def find_reference(img: np.ndarray) -> Rect:
    """Right-hand reference panel: the largest bright blob right of the board.

    Detected rather than hard coded because the page shifts a few pixels between
    games; the measured box (1435, 517, 451, 339) is the fallback.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = cv2.morphologyEx((g > 150).astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((9, 9), np.uint8))
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(bright)
    best = None
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if x < 900 or w < 200 or area < 30000:
            continue
        if not 1.2 <= w / h <= 1.45:
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    return Rect(*best[:4]) if best else Rect(*REFERENCE)


def reference_grid(img: np.ndarray, n: int, clf: Classifier) -> list[str]:
    """Required class per board cell, row-major (index = row * n + col)."""
    ref = find_reference(img)
    return [clf.label(sub(img, box))[0] for _r, _c, box in ref.cells(n)]
PIECE_AREA = 9520          # 112 x 85, the area of one tray piece


def tray_pieces(img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Locate tray pieces by connected components.

    The gap-grid works while the tray is full, but once most slots are empty the
    gaps disappear and the grid cannot be recovered.  Components stay reliable in
    both cases; blobs that clearly hold two touching pieces are split.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for band in ("left", "right"):
        x0, x1, y0, y1 = tray_region(img, band)
        mask = cv2.morphologyEx((g[y0:y1, x0:x1] > 40).astype(np.uint8),
                                cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i][:5])
            if area < PIECE_AREA * 0.45:
                continue
            count = max(1, round(area / PIECE_AREA))
            step = (w if w >= h else h) / count
            for k in range(count):
                if w >= h:
                    bx, by, bw, bh = x0 + x + int(k * step), y0 + y, int(step), h
                else:
                    bx, by, bw, bh = x0 + x, y0 + y + int(k * step), w, int(step)
                out.append((band, (bx, by, bw, bh)))
    return out


def board_filled(img: np.ndarray, n: int) -> list[bool]:
    """A board cell is filled once it shows a photo instead of the dark placeholder.

    Measured: the placeholder sits at mean 23-27 (a few cells carry a faint ghost,
    std up to 30), while the darkest real piece still averages 73 -- so the board
    needs a brightness test, not the flatness test used for tray slots.
    """
    board = layout(img)[1]
    out = []
    for _r, _c, box in board.cells(n):
        g = cv2.cvtColor(sub(img, box), cv2.COLOR_BGR2GRAY)
        out.append(float(g.mean()) >= BOARD_FILLED_LEVEL)
    return out


# ---- progress counter: the only authoritative success signal -----------------
# "杩涘害: k/N" sits just above the reference panel; k increases exactly once per piece
# that is dropped on its own cell, so a change in this crop *is* the verdict.
PROGRESS_REL = (-235, -292, 470, 44)     # the whole "杩涘害: k/N" text row, relative to the reference panel


def progress_box(img: np.ndarray) -> tuple[int, int, int, int]:
    ref = find_reference(img)
    dx, dy, w, h = PROGRESS_REL
    x, y = ref.x + dx, ref.y + dy
    x = max(0, min(x, img.shape[1] - w))
    y = max(0, min(y, img.shape[0] - h))
    return x, y, w, h


def progress_crop(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(sub(img, progress_box(img)), cv2.COLOR_BGR2GRAY)


def ticking_columns(a: np.ndarray, b: np.ndarray, thresh: int = 30) -> set[int]:
    """Columns that change on their own (the running timer sits on the same text row)."""
    if a.shape != b.shape:
        return set()
    cols = (cv2.absdiff(a, b) > 40).any(axis=0)
    return set(np.nonzero(cols)[0].tolist())


def progress_changed(a: np.ndarray, b: np.ndarray, ignore: set[int], thresh: int = 8) -> bool:
    """True when something other than the ticking columns changed.

    The counter shares its row with a live timer, so raw frame comparison always reports a
    change.  Columns that move while the page is idle are masked out first, and what is
    left is the "杩涘害: k/N" text.
    """
    if a.shape != b.shape:
        return False
    d = cv2.absdiff(a, b) > 40
    if ignore:
        d[:, sorted(ignore)] = False
    return int(d.sum()) > thresh


def read_progress(img: np.ndarray) -> str:
    """Raw crop of the progress label; the caller compares it for changes."""
    return sub(img, PROGRESS).tobytes()



