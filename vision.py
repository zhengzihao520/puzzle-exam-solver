"""Verified vision layer for puzzle.moyunliuyin.me (exam mode).

Everything here was measured on real screenshots, not guessed:

* board          (463, 460, 530, 418)  -- N x N cells, N comes from the reference panel
* reference      (1436, 509, 451, 338) -- N x N same-class *other* photos (no pixel match)
* progress       (1540, 225, 1790, 285) -- "进度: k/N" text, the only外部 success truth
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

BOARD = (463, 460, 530, 418)
REFERENCE = (1436, 509, 451, 338)
PROGRESS = (1540, 225, 1790, 285)
TRAY_REGIONS = {"left": (10, 460, 420, 1000), "right": (990, 1440, 420, 1000)}

GAP_LEVEL = 60.0        # darker than this on a band profile = gap between pieces
EMPTY_LEVEL = 45.0      # a tray cell darker than this holds no piece
BOARD_FILLED_LEVEL = 60.0  # board cell brighter than this shows a placed piece
BLANK_FRACTION = 0.55    # this share of near-white pixels marks the white decoy piece
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
BOARD_REL = (-972.0, -57.0, 530.0, 418.0)
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
    x0, x1, y0, y1 = tray_region(img, band)
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    return _keep_regular(_spans(_gap_runs(_dark_frac(g).mean(axis=0)), x0))


def tray_slots(img: np.ndarray, band: str,
               rows: list[tuple[int, int]] | None = None
               ) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """(row, col, (x, y, w, h)) for every slot of one band, 0-indexed."""
    rows = tray_rows(img) if rows is None else rows
    cols = tray_cols(img, band)
    return [(ri, ci, (cx0, ry0, cx1 - cx0 + 1, ry1 - ry0 + 1))
            for ri, (ry0, ry1) in enumerate(rows)
            for ci, (cx0, cx1) in enumerate(cols)]

def cell_of(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    return sub(img, box)


def is_empty(cell: np.ndarray) -> bool:
    """An empty slot is flat dark tray background, not merely a dark piece.

    Measured: empty slots are (9, 9, 9) everywhere (std 0.0), while dark *photos*
    reach std 8-30 (a just-emptied slot keeps a stray pixel: std 3.2), so the
    flatness cut sits at 6.
    """
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    return float(g.mean()) < EMPTY_LEVEL and float(g.std()) < 6.0


def is_blank(cell: np.ndarray) -> bool:
    """White decoy piece: most of the cell is near white (borders excluded by design)."""
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    return float((g > 200).mean()) >= BLANK_FRACTION


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
    ref = layout(img)[0]
    return [clf.label(sub(img, box))[0] for _r, _c, box in ref.cells(n)]


def tray_cells(img: np.ndarray, clf: Classifier, n: int | None = None
               ) -> list[tuple[str, str, int, int, str, tuple[int, int, int, int]]]:
    """Every occupied tray piece: (band, label, row, col, key, box)."""
    out = []
    for band in ("left", "right"):
        for ri, ci, box in tray_slots(img, band):
            cell = sub(img, box)
            if is_empty(cell):
                continue
            label, conf = clf.label(cell)
            out.append((band, label, ri, ci, f"{band[0]}{ri + 1}{ci + 1}", box))
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


def read_progress(img: np.ndarray) -> str:
    """Raw crop of the progress label; the caller compares it for changes."""
    return sub(img, PROGRESS).tobytes()
