"""实机 v6：三级定位兜底 + 加密采样 + 同一块试遍同类格（v4 已验证的核心）。

相对 v4 的改动：
  * 棋盘定位三级兜底：平坦区 → 边框 → 上次成功坐标（写 logs/board_cache.json），
    否则拼到一半棋盘被块盖住就再也接不上（v5 就是这样中止的）；
  * 托盘采样从 0.55 格加密到 0.4 格，减少"块被压住 / 漏采"导致的类别缺失；
  * 开跑前把棋盘上"非吸附"的松散块拖回托盘，避免 filled_cells 高估已填。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import solve as S

NUDGES = [(0, 0), (6, 0), (-6, 0), (0, 6), (0, -6)]
CACHE = Path(r"F:\exam_next\logs\board_cache.json")


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


def piece_like(img, x, y, cw, ch) -> bool:
    p = crop_at(img, x, y, cw * 0.7, ch * 0.7)
    if p is None:
        return False
    g = p.mean(axis=2)
    return float(g.std()) > 12.0 or float(g.mean()) > 62.0


def locate(img, grid):
    """三级兜底：平坦区 → 边框 → 缓存坐标。"""
    board = S.find_board(img)
    how = "flat"
    if board is None:
        board = S.find_board_border(img)
        how = "border"
    if board is None and CACHE.exists():
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        board = S.Rect(d["bx"], d["by"], d["bw"], d["bh"])
        how = "cache"
    if board is None:
        return None, None, None, "none"
    ref = S.find_reference(img, board, grid)
    if ref is None and CACHE.exists():
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        if "rx" in d:
            ref = S.Rect(d["rx"], d["ry"], d["rw"], d["rh"])
            how += "+ref-cache"
    if board is not None and ref is not None:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps({"bx": board.x, "by": board.y, "bw": board.w, "bh": board.h,
                                     "rx": ref.x, "ry": ref.y, "rw": ref.w, "rh": ref.h}),
                         encoding="utf-8")
    return board, ref, S.find_stage(img, board, ref), how


def snapped(mouse, img, cell, cw, ch) -> bool:
    before = crop_at(img, cell.cx, cell.cy, cw, ch)
    far_x = cell.cx + cw * 1.6 if cell.cx + cw * 2.4 < img.shape[1] else cell.cx - cw * 1.6
    far_before = crop_at(img, far_x, cell.cy, cw * 0.8, ch * 0.8)
    mouse.press(cell.cx, cell.cy)
    mouse.glide(far_x, cell.cy)
    time.sleep(0.1)
    shot = S.grab()
    mouse.glide(cell.cx, cell.cy)
    mouse.release()
    time.sleep(0.08)
    left = diff(before, crop_at(shot, cell.cx, cell.cy, cw, ch)) > 18
    appeared = diff(far_before, crop_at(shot, far_x, cell.cy, cw * 0.8, ch * 0.8)) > 18
    return (not left) and (not appeared)


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
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    S.enable_dpi_awareness()
    solver = S.Solver(grid=grid, part=1, dry_run=False)
    mouse, clf = solver.mouse, solver.clf

    img = S.grab()
    board, ref, stage, how = locate(img, grid)
    print(f"locate={how} board={board.box if board else None} ref={ref.box if ref else None}", flush=True)
    if board is None or ref is None:
        print("!! 定位失败，中止")
        return 2
    cw, ch = board.w / grid, board.h / grid
    cells = S.cell_rects(board, grid)
    print(f"cell=({cw:.0f}x{ch:.0f}) 容差≈{cw * 0.14:.0f}px", flush=True)

    labels = [S.BLANK if S.is_blank(img[c.y:c.y + c.h, c.x:c.x + c.w])
              else clf.predict([img[c.y:c.y + c.h, c.x:c.x + c.w]])[0][0]
              for c in S.cell_rects(ref, grid)]
    filled = S.filled_cells(img, board, grid)
    print(f"filled {sum(filled)}/25 -> {[i for i, v in enumerate(filled) if v]}", flush=True)

    # 清盘：棋盘上非吸附的松散块拖回托盘左侧
    if sum(filled) > 0:
        for i, is_fill in enumerate(filled):
            if not is_fill:
                continue
            cell = cells[i]
            if not snapped(mouse, S.grab(), cell, cw, ch):
                tx = stage.x + cw * 0.6
                ty = stage.y + stage.h * 0.5
                mouse.drag(cell.cx, cell.cy, tx, ty)
                filled[i] = False
        print(f"清盘后 filled {sum(filled)}/25", flush=True)

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

    placed: dict[int, tuple[float, float]] = {}
    attempts = 0
    while attempts < budget:
        img = S.grab()
        filled = S.filled_cells(img, board, grid)
        for i in placed:
            filled[i] = True
        # 双重确认：像素有内容 **且** 内容类别与参考格一致，才算"已填"。
        # 否则是松散块（拖过去没吸附的），先拖回托盘，别让它骗停主循环。
        for i, is_fill in enumerate(filled):
            if not is_fill or i in placed:
                continue
            c = cells[i]
            lab = clf.predict([img[c.y:c.y + c.h, c.x:c.x + c.w]])[0][0]
            if lab == labels[i]:
                continue
            if snapped(mouse, S.grab(), c, cw, ch):
                continue                      # 真吸附了，只是分类不同，保留
            tx = stage.x + cw * 0.6
            ty = stage.y + stage.h * 0.5
            mouse.drag(c.cx, c.cy, tx, ty)
            filled[i] = False
            print(f"    清掉 cell{i} 的松散块（内容={lab} 与参考={labels[i]} 不符）", flush=True)
        anch = S.anchored_cells(filled, grid)
        todo = [i for i in range(grid * grid) if not filled[i] and i in anch and labels[i] != S.BLANK]
        if not todo:
            print("没有可放的目标格")
            break
        need = labels[todo[0]]
        same_class = [i for i in todo if labels[i] == need]
        pieces = inv.get(need, [])
        print(f"\n=== 需要 {need}: 同类空格 {same_class}, 库存 {len(pieces)}", flush=True)
        if not pieces:
            for i in same_class:
                labels[i] = S.BLANK
            continue
        done = False
        for score, px, py in pieces[:3]:
            for cell_idx in same_class:
                cell = cells[cell_idx]
                for dx, dy in NUDGES:
                    mouse.drag(px, py, cell.cx + dx, cell.cy + dy)
                    time.sleep(0.1)
                    attempts += 1
                    if snapped(mouse, S.grab(), cell, cw, ch):
                        print(f"    ** cell{cell_idx} <- ({px:.0f},{py:.0f}) 成功 {score:.2f}", flush=True)
                        placed[cell_idx] = (cell.cx + dx, cell.cy + dy)
                        done = True
                        break
                    mouse.drag(cell.cx + dx, cell.cy + dy, px, py)
                    if attempts >= budget:
                        break
                if done or attempts >= budget:
                    break
            if done or attempts >= budget:
                break
        if not done and attempts >= budget:
            print("    达到预算，停止")
            break

    print(f"\n[done] placed {len(placed)}/25, drags {attempts}", flush=True)
    print("cells:", sorted(placed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
