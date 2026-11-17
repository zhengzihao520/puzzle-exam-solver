"""实机 v4：同一块试遍同类格 + 落点微调重试 + 双向校验。

v3 唯一的错误：失败后把"块"丢掉（pop），而不是把同一块换到另一个同类格。
引擎吸附要求「这块自己的正确格 + 有锚点 + 误差<0.14格(≈15px)」，同类常有多格，
所以必须：同一块 → 试遍同类空格 → 每格再 ±6px 微调 → 全失败才判为混淆块。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import solve as S

NUDGES = [(0, 0), (6, 0), (-6, 0), (0, 6), (0, -6)]


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


def snapped(mouse, img, cell, cw, ch) -> bool:
    """双向校验：按住落点拖到远处，看格心是否被让空、远处是否真出现块。"""
    before = crop_at(img, cell.cx, cell.cy, cw, ch)
    far_x = cell.cx + cw * 1.6 if cell.cx + cw * 2.4 < img.shape[1] else cell.cx - cw * 1.6
    far_before = crop_at(img, far_x, cell.cy, cw * 0.8, ch * 0.8)
    mouse.press(cell.cx, cell.cy)
    mouse.glide(far_x, cell.cy)
    time.sleep(0.1)
    shot = S.grab()
    mouse.glide(cell.cx, cell.cy)
    mouse.release()
    time.sleep(0.1)
    left = diff(before, crop_at(shot, cell.cx, cell.cy, cw, ch)) > 18
    appeared = diff(far_before, crop_at(shot, far_x, cell.cy, cw * 0.8, ch * 0.8)) > 18
    print(f"      校验 让空={left} 远处出现={appeared}", flush=True)
    return (not left) and (not appeared)


def main() -> int:
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    S.enable_dpi_awareness()
    solver = S.Solver(grid=grid, part=1, dry_run=False)
    mouse, clf = solver.mouse, solver.clf

    img = S.grab()
    board = S.find_board(img) or S.find_board_border(img)
    ref = S.find_reference(img, board, grid) if board else None
    if board is None or ref is None:
        print("!! 定位失败，中止")
        return 2
    stage = S.find_stage(img, board, ref)
    cw, ch = board.w / grid, board.h / grid
    cells = S.cell_rects(board, grid)
    print(f"board={board.box} ref={ref.box} cell=({cw:.0f}x{ch:.0f}) 容差≈{cw * 0.14:.0f}px", flush=True)

    labels = [S.BLANK if S.is_blank(img[c.y:c.y + c.h, c.x:c.x + c.w])
              else clf.predict([img[c.y:c.y + c.h, c.x:c.x + c.w]])[0][0]
              for c in S.cell_rects(ref, grid)]
    inv: dict[str, list[tuple[float, float, float]]] = {}
    for (x, y) in S.tray_candidates(stage, board, ref, grid):
        if not piece_like(img, x, y, cw, ch):
            continue
        c = crop_at(img, x, y, cw * 1.1, ch * 1.1)
        if c is None or S.is_blank(c):
            continue
        lab, sc = clf.predict([c])[0]
        inv.setdefault(lab, []).append((sc, x, y))
    for k in inv:
        inv[k].sort(reverse=True)

    placed: dict[int, tuple[float, float]] = {}
    attempts = 0
    while attempts < budget:
        img = S.grab()
        filled = S.filled_cells(img, board, grid)
        for i in placed:
            filled[i] = True
        anch = S.anchored_cells(filled, grid)
        todo = [i for i in range(grid * grid)
                if not filled[i] and i in anch and labels[i] != S.BLANK]
        if not todo:
            print("没有可放的目标格")
            break
        # 同类多格：把所有可锚定的同类格凑一起
        need = labels[todo[0]]
        same_class = [i for i in todo if labels[i] == need]
        pieces = inv.get(need, [])
        print(f"\n=== 需要 {need}: 同类空格 {same_class}, 库存 {len(pieces)} 块", flush=True)
        if not pieces:
            print(f"    库存无 {need}，跳过该类")
            for i in same_class:
                labels[i] = S.BLANK
            continue

        done = False
        for score, px, py in pieces[:3]:
            for cell_idx in same_class:
                cell = cells[cell_idx]
                for dx, dy in NUDGES:
                    mouse.drag(px, py, cell.cx + dx, cell.cy + dy)
                    time.sleep(0.12)
                    attempts += 1
                    if snapped(mouse, S.grab(), cell, cw, ch):
                        print(f"    ** cell{cell_idx} <- ({px:.0f},{py:.0f}) 吸附成功 score={score:.2f}", flush=True)
                        placed[cell_idx] = (cell.cx + dx, cell.cy + dy)
                        done = True
                        break
                    # 未吸附：拖回原处再试
                    mouse.drag(cell.cx + dx, cell.cy + dy, px, py)
                    if attempts >= budget:
                        break
                if done or attempts >= budget:
                    break
            if done or attempts >= budget:
                break
        if not done and attempts >= budget:
            print("    达到本次预算，停止")
            break

    print(f"\n[done] placed {len(placed)}/{grid * grid}, drags {attempts}", flush=True)
    for k, v in placed.items():
        print(f"  cell{k} @ ({v[0]:.0f},{v[1]:.0f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
