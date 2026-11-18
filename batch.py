"""Small-batch labelling of real tray pieces, driven by measured cell geometry.

Usage:
  python batch.py show <band> <row>              render numbered crops of one row
  python batch.py save <band> <row> <l1,l2,...>  store crops with labels, then
                                                auto-advance to the next non-empty row
  python batch.py rebuild                        re-cut every stored crop from the
                                                measured cell rects
  python batch.py status                         per-class coverage + missing rows

band: left / right            row: 1..6  (1-indexed, top to bottom)

Geometry (measured from shots/band_<band>.png): each band is a 3 x 6 grid of cells
separated by dark gaps.  A piece fills its whole cell and the gaps are the only dark
thing on the band, so the cell rects are recovered from the column/row brightness
profiles: anything below GAP_LEVEL is a gap, the spans between gaps are cells.
A cell whose mean brightness is below EMPTY_LEVEL holds no piece.  Crops are taken
from the cell rect itself and resized to PW x PH, so a crop can never straddle a gap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"F:\exam_next")
SHOTS = ROOT / "shots"
BASE = ROOT / "real_pieces_fix"
REG = ROOT / "real_pieces_labels_batch.json"

PW, PH = 112, 85
GAP_LEVEL = 60.0
EMPTY_LEVEL = 45.0
BANDS = ("left", "right")
ROWS, COLS = 6, 3
MIN_RUN = 6

# Fallback geometry used when a band image is missing or the profile is unusable.
FALLBACK = {
    "left": ([(11, 122), (139, 251), (269, 380)],
             [(8, 94), (105, 190), (204, 287), (300, 385), (397, 480), (496, 579)]),
    "right": ([(21, 133), (149, 263), (280, 393)],
              [(9, 94), (106, 190), (205, 289), (302, 386), (397, 481), (494, 577)]),
}

CLASSES = [
    "dog", "cat", "elephant", "desktop", "bicycle", "car", "chair", "cup",
    "airplane", "backpack", "umbrella", "suitcase", "penguin", "tafei",
    "nailong", "blank",
]


def load_band(name: str) -> np.ndarray:
    im = cv2.imread(str(SHOTS / f"band_{name}.png"))
    if im is None:
        raise SystemExit(f"missing shots/band_{name}.png")
    return im


def _dark_runs(profile: np.ndarray) -> list[tuple[int, int]]:
    mask = profile < GAP_LEVEL
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


def _spans(dark: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Cell spans between consecutive dark runs (inclusive bounds)."""
    out = []
    for (a0, a1), (b0, _b1) in zip(dark, dark[1:]):
        if b0 - a1 >= 8:
            out.append((a1 + 1, b0 - 1))
    if len(dark) == 0:
        out.append((0, size - 1))
    return out


def grid(name: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Measured (col_rects, row_rects) of one band."""
    try:
        g = cv2.cvtColor(load_band(name), cv2.COLOR_BGR2GRAY).astype(np.float32)
    except SystemExit:
        return FALLBACK[name]
    cols = _spans(_dark_runs(g.mean(axis=0)), g.shape[1])
    rows = _spans(_dark_runs(g.mean(axis=1)), g.shape[0])
    if len(cols) != COLS or len(rows) != ROWS:
        print(f"  [grid] {name}: measured {len(cols)}x{len(rows)}, using fallback")
        return FALLBACK[name]
    return cols, rows


def cells(name: str) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], bool]]:
    """(row, col) -> (col_rect, row_rect, occupied); 1-indexed row/col."""
    im = load_band(name)
    cols, rows = grid(name)
    out = {}
    for ri, (y0, y1) in enumerate(rows):
        for ci, (x0, x1) in enumerate(cols):
            cell = im[y0:y1 + 1, x0:x1 + 1]
            out[(ri + 1, ci + 1)] = ((x0, x1), (y0, y1), float(cell.mean()) >= EMPTY_LEVEL)
    return out


def crop_cell(im: np.ndarray, col: tuple[int, int], row: tuple[int, int]) -> np.ndarray:
    (x0, x1), (y0, y1) = col, row
    return cv2.resize(im[y0:y1 + 1, x0:x1 + 1], (PW, PH), interpolation=cv2.INTER_AREA)


def centre(col: tuple[int, int], row: tuple[int, int]) -> tuple[int, int]:
    return ((col[0] + col[1]) // 2, (row[0] + row[1]) // 2)


def row_pieces(band: str, row: int) -> list[tuple[int, int, int, tuple[int, int], tuple[int, int]]]:
    """Occupied cells of one row: (row, col, cx, cy, col_rect, row_rect)."""
    out = []
    for (r, c), (col, rrow, ok) in sorted(cells(band).items()):
        if r == row and ok:
            cx, cy = centre(col, rrow)
            out.append((r, c, cx, cy, col, rrow))
    return out


def occupied(band: str, row: int) -> list[tuple[int, int]]:
    return [(p[2], p[3]) for p in row_pieces(band, row)]


def next_row(band: str, row: int) -> tuple[str, int] | None:
    r = row
    for b in BANDS[BANDS.index(band):]:
        while r <= ROWS:
            if occupied(b, r):
                return b, r
            r += 1
        r = 1
    return None


def show(band: str, row: int) -> Path:
    im = load_band(band)
    pts = row_pieces(band, row)
    if not pts:
        raise SystemExit(f"{band} row {row} is empty")
    pad, cap = 14, 46
    sheet = np.full((PH * 3 + cap + pad * 2, len(pts) * (PW * 3 + pad) + pad, 3), 25, np.uint8)
    for i, (_r, c, _cx, _cy, col, rrow) in enumerate(pts):
        big = cv2.resize(crop_cell(im, col, rrow), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        x, y = pad + i * (PW * 3 + pad), pad
        sheet[y:y + PH * 3, x:x + PW * 3] = big
        cv2.putText(sheet, f"#{i} (c{c})", (x + 6, y + PH * 3 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    out = SHOTS / f"batch_{band}{row}.png"
    cv2.imwrite(str(out), sheet)
    print(f"montage: {out}")
    print(f"{len(pts)} piece(s) in {band} row {row}: "
          f"{[(p[2], p[3], f'col{p[1]}') for p in pts]}")
    return out


def save(band: str, row: int, labels: list[str]) -> None:
    im = load_band(band)
    pts = row_pieces(band, row)
    if len(labels) != len(pts):
        raise SystemExit(f"{band} row {row} has {len(pts)} piece(s), got {len(labels)} labels")
    reg = json.loads(REG.read_text(encoding="utf-8")) if REG.exists() else {}
    tag = "L" if band == "left" else "R"
    for (_r, c, cx, cy, col, rrow), raw in zip(pts, labels):
        lab = raw.strip()
        if lab not in CLASSES:
            raise SystemExit(f"unknown class {lab!r}; expected one of {CLASSES}")
        d = BASE / lab
        d.mkdir(parents=True, exist_ok=True)
        fn = f"{lab}-{band}{row}-c{c}.jpg"
        cv2.imwrite(str(d / fn), crop_cell(im, col, rrow), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        reg[f"{tag}{row}-c{c}"] = {"label": lab, "centre": [cx, cy], "file": str(d / fn),
                                   "cell": [col[0], col[1], rrow[0], rrow[1]]}
        print(f"  {band} row{row} col{c} @{cx},{cy} -> {lab}")
    REG.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(labels)}; total {len(reg)}")


def rebuild() -> None:
    """Re-cut every registered crop from the measured cell rects (label kept)."""
    reg = json.loads(REG.read_text(encoding="utf-8"))
    for key, v in reg.items():
        band = "left" if key[0] == "L" else "right"
        r_txt, c_txt = key[1:].split("-c")
        im = load_band(band)
        cell = cells(band)[(int(r_txt), int(c_txt))]
        col, rrow, ok = cell
        if not ok:
            raise SystemExit(f"cell {key} reads as empty")
        d = BASE / v["label"]
        d.mkdir(parents=True, exist_ok=True)
        fn = f"{v['label']}-{band}{r_txt}-c{c_txt}.jpg"
        cv2.imwrite(str(d / fn), crop_cell(im, col, rrow), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        v["file"] = str(d / fn)
        v["centre"] = list(centre(col, rrow))
        v["cell"] = [col[0], col[1], rrow[0], rrow[1]]
    REG.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    stale = [p for p in BASE.rglob("*.jpg") if str(p) not in {v["file"] for v in reg.values()}]
    for p in stale:
        p.unlink()
    print(f"rebuilt {len(reg)} crops, removed {len(stale)} stale file(s)")


def status() -> None:
    reg = json.loads(REG.read_text(encoding="utf-8")) if REG.exists() else {}
    per = {k: 0 for k in CLASSES}
    for v in reg.values():
        per[v["label"]] = per.get(v["label"], 0) + 1
    print(f"labelled pieces: {len(reg)} / 30")
    print("  " + "  ".join(f"{k}:{n}" for k, n in per.items() if n))
    done = {(k[0], int(k[1:].split("-c")[0])) for k in reg}
    todo = [f"{b}{r}" for b in BANDS for r in range(1, ROWS + 1)
            if (("L" if b == "left" else "R"), r) not in done and occupied(b, r)]
    print("unlabelled rows:", ", ".join(todo) if todo else "none")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "status":
        status()
        return 0
    if cmd == "rebuild":
        rebuild()
        return 0
    if cmd == "grid":
        for b in BANDS:
            print(b, grid(b))
        return 0
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    band, row = sys.argv[2], int(sys.argv[3])
    if cmd == "save":
        labels = [s for s in sys.argv[4].split(",") if s.strip()]
        save(band, row, labels)
        nxt = next_row(band, row + 1)
        if nxt:
            print(f"next batch -> {nxt[0]} row {nxt[1]}")
            show(*nxt)
        else:
            status()
        return 0
    show(band, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
