"""Reconcile the reference panel and the tray against the catalog, read-only.

The two multisets must be equal: the tray holds exactly the photos the board needs (plus the
white decoys, one per blank cell).  If they are not equal, either the page is not a coherent
round yet or a crop is wrong, and placing anything would be guesswork -- so this refuses to
output an assignment until the multisets agree.

  python reconcile.py --delay 12
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import vision
from catalog import Catalog
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
    cv2.imwrite(str(vision.ROOT / "shots" / "reconcile.png"), img)
    ref, board, _ = vision.layout(img)
    cat = Catalog()
    print(f"reference {ref}  board {board}  catalog {len(cat.labels)} images")

    cells = [vision.sub(img, b) for _r, _c, b in ref.cells(N)]
    ref_content = [float(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).std()) > 15 for c in cells]
    print(f"reference cells with content: {sum(ref_content)}/{N * N}")

    rows = vision.tray_rows(img)
    slots = [(f"{b[0]}{ri + 1}{ci + 1}", box) for b in ("left", "right")
             for ri, ci, box in vision.tray_slots(img, b, rows)]
    pieces = [(k, b) for k, b in slots if not vision.is_empty(vision.sub(img, b))]
    print(f"tray pieces: {len(pieces)}")

    if sum(ref_content) < 20 or len(pieces) < 20:
        print("\nPAGE NOT READY -- no assignment produced")
        return 3

    need, need_score = [], []
    for c, filled in zip(cells, ref_content):
        if not filled:
            need.append("blank")
            need_score.append(1.0)
            continue
        m = cat.match(c)
        need.append(m.class_name)
        need_score.append(m.score)

    have, have_score, have_info = [], [], {}
    for key, box in pieces:
        cell = vision.sub(img, box)
        if vision.is_white_piece(cell):
            # decoys are catalog photos that are not on the board, but the white ones are
            # plain white and must be classified by brightness, not by the matcher
            have.append("blank")
            have_score.append(1.0)
            have_info[key] = {"class": "blank", "score": 1.0, "photo": "-"}
            continue
        m = cat.match_aligned(img, box)
        have.append(m.class_name)
        have_score.append(m.score)
        have_info[key] = {"class": m.class_name, "score": round(m.score, 3),
                          "photo": Path(m.path).name}

    print("\nreference (what each cell needs), rows top to bottom:")
    for r in range(N):
        print("   " + " ".join(f"{need[r * N + c][:8]:8s}" for c in range(N)))
    print(f"   score mean {np.mean(need_score):.2f} min {min(need_score):.2f}")

    print(f"\ntray pieces ({len(have)}), lowest score only:")
    for key, sc in sorted(have_info.items(), key=lambda kv: kv[1]["score"])[:8]:
        print(f"   {key}: {sc['class']:9s} {sc['score']:.3f}  {sc['photo']}")

    a, b = Counter(need), Counter(have)
    keys = sorted(set(a) | set(b), key=lambda k: -(a[k] + b[k]))
    short = []
    print(f"\n{'class':10s} {'needed':>7s} {'in tray':>8s}")
    for k in keys:
        deficit = a[k] - b[k]
        flag = "   <-- SHORT" if deficit > 0 else ("   (surplus = decoy)" if deficit < 0 else "")
        if deficit > 0:
            short.append((k, deficit))
        print(f"  {k:10s} {a[k]:7d} {b[k]:8d}{flag}")
    surplus = sum(max(0, b[k] - a[k]) for k in keys)
    print(f"\nshort classes: {short if short else 'none'};  surplus pieces (decoys): {surplus}")

    if short:
        print("\nNO ASSIGNMENT: some needed classes do not have enough pieces, so the "
              "reading of those pieces is still wrong.")
        return 4

    table = {}
    pool: dict[str, list[str]] = {}
    for key, cls in zip([k for k, _b in pieces], have):
        pool.setdefault(cls, []).append(key)
    for idx, cls in enumerate(need):
        if cls == "blank":
            continue
        if pool.get(cls):
            table[f"{idx // N},{idx % N}"] = pool[cls].pop(0)
    OUT.write_text(json.dumps({"need": need, "pieces": have_info, "table": table}, indent=2),
                   encoding="utf-8")
    print(f"\nassignment written to {OUT} ({len(table)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
