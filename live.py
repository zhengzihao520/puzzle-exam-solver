"""实机 v9：进度数字是唯一真值 + 全清盘 + 每格最多试 3 次。

三条硬规则（都是被实机证伪后改出来的）：
  1. 成功与否只看顶栏 "进度 N/25" 有没有变化——我此前三次用画面推断都错了；
  2. 开局若棋盘上有内容，一律拖回托盘（不再用抓取判断区分"吸附/松散"）；
  3. 每个目标格最多试 3 次，然后换格；同类的块轮着试，避免死磕一格。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import solve as S

PROG = (1540, 225, 1790, 285)          # 顶栏 "进度: N/25"
CACHE = Path(r"F:\exam_next\logs\board_cache.json")
NUDGES = [(0, 0), (5, 0), (-5, 0)]


def crop_at(img, x, y, w, h):
    x0, y0 = int(x - w / 2), int(y - h / 2)
    x1, y1 = x0 + int(w), y0 + int(h)
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None
    return img[y0:y1, x0:x1].copy()


def diff(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.abs(b.astype(np.int16) - a.astype(np.int16)).mean())


def prog(img):
    x0, y0, x1, y1 = PROG
    return img[y0:y1, x0:x1].copy()


def piece_like(img, x, y, cw, ch) -> bool:
    p = crop_at(img, x, y, cw * 0.7, ch * 0.7)
    if p is None:
        return False
    g = p.mean(axis=2)
    return float(g.std()) > 12.0 or float(g.mean()) > 62.0


def locate(img, grid):
    board = S.find_board(img)
    how = "flat"
    if board is None:
        board = S.find_board_border(img)
        how = "border"
    if board is None and CACHE.exists():
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        board, how = S.Rect(d["bx"], d["by"], d["bw"], d["bh"]), "cache"
    if board is None:
        return None, None, None, "none"
    ref = S.find_reference(img, board, grid)
    if ref is None and CACHE.exists():
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        if "rx" in d:
            ref, how = S.Rect(d["rx"], d["ry"], d["rw"], d["rh"]), how + "+refcache"
    if ref is not None:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps({"bx": board.x, "by": board.y, "bw": board.w, "bh": board.h,
                                     "rx": ref.x, "ry": ref.y, "rw": ref.w, "rh": ref.h}),
                         encoding="utf-8")
    return board, ref, S.find_stage(img, board, ref) if ref else None, how


def dense_points(stage, board, ref, grid):
    cw, ch = board.w / grid, board.h / grid
    pts = []
    for x0, x1 in ((stage.x + cw * 0.3, board.x - cw * 0.2),
                   (board.x + board.w + cw * 0.2, (ref.x if ref else stage.x + stage.w) - cw * 0.2)):
        if x1 <= x0:
            continue
        y = stage.y + ch * 0.3
        while y < stage.y + stage.h - ch * 0.2:
            x = x0
            while x < x1:
                pts.append((round(x, 1), round(y, 1)))
                x += cw * 0.4
            y += ch * 0.4
    return pts


def main() -> int:
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    S.enable_dpi_awareness()
    solver = S.Solver(grid=grid, part=1, dry_run=False)
    mouse, clf = solver.mouse, solver.clf

    img = S.grab()
    board, ref, stage, how = locate(img, grid)
    print(f"locate={how} board={board.box if board else None} ref={ref.box if ref else None}", flush=True)
    if not (board and ref and stage):
        print("!! 定位失败，中止")
        return 2
    cw, ch = board.w / grid, board.h / grid
    cells = S.cell_rects(board, grid)
    labels = [S.BLANK if S.is_blank(img[c.y:c.y + c.h, c.x:c.x + c.w])
              else clf.predict([img[c.y:c.y + c.h, c.x:c.x + c.w]])[0][0]
              for c in S.cell_rects(ref, grid)]
    print(f"cell=({cw:.0f}x{ch:.0f}) 参考格类别={labels}", flush=True)

    # --- 全清盘：棋盘上任何内容都拖回托盘（不用抓取判断） ---
    filled = S.filled_cells(img, board, grid)
    dirty = [i for i, v in enumerate(filled) if v]
    if dirty:
        print(f"清盘：棋盘上有内容的格子 {dirty}，全部拖回托盘", flush=True)
        for i in dirty:
            c = cells[i]
            mouse.drag(c.cx, c.cy, stage.x + cw * 0.6, stage.y + stage.h * 0.5)
            time.sleep(0.08)
        img = S.grab()

    # --- 库存 ---
    inv: dict[str, list[tuple[float, float, float]]] = {}
    for (x, y) in dense_points(stage, board, ref, grid):
        if not piece_like(img, x, y, cw, ch):
            continue
        c = crop_at(img, x, y, cw * 1.1, ch * 1.1)
        if c is None or S.is_blank(c):
            continue
        lab, sc = clf.predict([c])[0]
        inv.setdefault(lab, []).append((sc, x, y))
    for k in inv:
        inv[k].sort(reverse=True)
    print("inventory:", {k: len(v) for k, v in sorted(inv.items(), key=lambda t: -len(t[1]))}, flush=True)

    base = prog(S.grab())
    placed: list[int] = []
    attempts = 0
    while attempts < budget:
        img = S.grab()
        filled = S.filled_cells(img, board, grid)
        for i in placed:
            filled[i] = True
        anch = S.anchored_cells(filled, grid)
        todo = [i for i in range(grid * grid) if not filled[i] and i in anch and i not in placed]
        if not todo:
            print("没有可放的目标格")
            break
        # 选"库存里同类块最多"的格子，减少空跑
        todo.sort(key=lambda i: -len(inv.get(labels[i], [])))
        cell_idx = todo[0]
        need = labels[cell_idx]
        pieces = inv.get(need, [])
        print(f"\n=== cell{cell_idx} need={need} 库存 {len(pieces)}", flush=True)
        if not pieces:
            print("    库存无此类，跳过", flush=True)
            labels[cell_idx] = S.BLANK
            continue
        cell = cells[cell_idx]
        ok = False
        for score, px, py in pieces[:2]:
            for dx, dy in NUDGES:
                p0 = prog(S.grab())
                mouse.drag(px, py, cell.cx + dx, cell.cy + dy)
                time.sleep(0.18)
                p1 = prog(S.grab())
                attempts += 1
                if diff(p0, p1) > 6.0:                 # 进度 +1 = 游戏认可
                    print(f"    ** cell{cell_idx} <- ({px:.0f},{py:.0f}) 进度+1 ({score:.2f})", flush=True)
                    placed.append(cell_idx)
                    ok = True
                    break
                mouse.drag(cell.cx + dx, cell.cy + dy, px, py)   # 拖回原处
                if attempts >= budget:
                    break
            if ok or attempts >= budget:
                break
        if not ok:
            print(f"    cell{cell_idx} 试满未吸附，换格", flush=True)
            if attempts >= budget:
                break

    print(f"\n[done] 进度确认放置 {len(placed)} 块, drags {attempts}", flush=True)
    print("cells:", sorted(placed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
