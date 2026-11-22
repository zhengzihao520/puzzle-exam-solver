"""Produce the assignment table: which tray piece goes to which board cell. Read-only.

Both sides are read with the fused identifier (catalog match + YOLO, agreement = certain).
The reference panel says which class each cell needs; the tray says which pieces exist.  A
cell then gets a certain piece of its class, and any leftover ambiguity is listed as
alternatives rather than hidden.

  python assign.py --delay 12
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

import vision
from identify import Identifier, BLANK
from solve_live import check_foreground, _enable_dpi

N = 5
OUT = vision.ROOT / "logs" / "assignment.json"


def main() -> int:
    _enable_dpi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    global N
    N = args.grid
    if args.delay:
        print(f"waiting {args.delay:.0f}s ...", flush=True)
        time.sleep(args.delay)
    ok, why = check_foreground()
    print(f"foreground: {why}")
    if not ok:
        return 2

    img = vision.grab()
    cv2.imwrite(str(vision.ROOT / "shots" / "assign.png"), img)
    ref, board, _ = vision.layout(img)
    idf = Identifier()

    cells = [vision.sub(img, b) for _r, _c, b in ref.cells(N)]
    content = [float(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).std()) > 15 for c in cells]
    rows = vision.tray_rows(img)
    slots = [(f"{b[0]}{ri + 1}{ci + 1}", box) for b in ("left", "right")
             for ri, ci, box in vision.tray_slots(img, b, rows)]
    pieces = [(k, b) for k, b in slots if not vision.is_empty(vision.sub(img, b))]
    print(f"reference cells with content {sum(content)}/{N * N}; tray pieces {len(pieces)}")
    if sum(content) < 20 or len(pieces) < 20:
        print("PAGE NOT READY -- no assignment produced")
        return 3

    need = []
    print("\nreference panel:")
    for idx, (cell, filled) in enumerate(zip(cells, content)):
        if not filled:
            need.append(BLANK)
            print(f"  ({idx // N + 1},{idx % N + 1}) {BLANK}")
            continue
        r = idf.identify(cell)
        need.append(r.label)
        print(f"  ({idx // N + 1},{idx % N + 1}) {r}")

    have = {}
    print("\ntray:")
    for key, box in pieces:
        r = idf.identify(vision.sub(img, box))
        have[key] = r
        print(f"  {key}: {r}")

    pool: dict[str, list[str]] = defaultdict(list)
    uncertain: list[str] = []
    for key, r in have.items():
        if r.label == BLANK:
            continue
        pool[r.label].append(key)
        if not r.confident:
            uncertain.append(key)

    need_cnt = Counter(c for c in need if c != BLANK)
    print(f"\n{'class':10s} {'cells need':>10s} {'pieces have':>11s}")
    for cls in sorted(set(need_cnt) | set(pool), key=lambda k: -(need_cnt[k] + len(pool[k]))):
        flag = "" if len(pool[cls]) >= need_cnt[cls] else "   <-- SHORT"
        print(f"  {cls:10s} {need_cnt[cls]:10d} {len(pool[cls]):11d}{flag}")

    table: dict[str, dict] = {}
    for idx, cls in enumerate(need):
        if cls == BLANK:
            continue
        if pool.get(cls):
            table[f"{idx // N},{idx % N}"] = {"class": cls, "pieces": pool[cls][:]}
    short = [c for c in need_cnt if len(pool[c]) < need_cnt[c]]
    print(f"\nassigned {len(table)} cells; short classes: {short if short else 'none'}")
    print(f"uncertain pieces: {len(uncertain)}")
    if short:
        print("\nNO RELIABLE TABLE: some classes do not have enough pieces.")
        return 4

    OUT.write_text(json.dumps({"need": need, "table": table,
                               "pieces": {k: {"class": r.label, "confident": r.confident,
                                              "alts": r.alternatives}
                                          for k, r in have.items()}}, indent=2),
                   encoding="utf-8")
    print(f"\nwritten to {OUT}")
    print("\ncell -> candidate pieces (first one is the pick):")
    for cell, info in sorted(table.items(), key=lambda kv: (int(kv[0].split(',')[0]),
                                                           int(kv[0].split(',')[1]))):
        r, c = cell.split(",")
        print(f"  cell({int(r) + 1},{int(c) + 1}) {info['class']:9s} <- {', '.join(info['pieces'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
