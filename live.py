"""实机跑一局（v2）：探测式拾取 + 类别匹配落位。

v1 实测发现的问题与修法：
  1. 探测点曾被“永久跳过”——但块被放回后位置会变，缓存失效 → 改为**每轮重置**；
  2. 探测失败时块被拖离原位、回位不精确，把托盘搅乱 → 改为**精确回位**（记录按下点）；
  3. 盲扫 240 个点太慢（每点 ~1.5s）→ 先用**画面特征预筛**（块是亮的、有纹理），
     只对像块的位置动鼠标，实测可砍掉大半无效探测；
  4. 目标格选择改为：优先挑“托盘里已探到过、次数最多”的类别，减少空跑。
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import solve as S


def crop_at(img, x, y, w, h):
    x0, y0 = int(x - w / 2), int(y - h / 2)
    x1, y1 = x0 + int(w), y0 + int(h)
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None
    return img[y0:y1, x0:x1].copy()


def looks_like_piece(img, x, y, cw, ch) -> bool:
    """块是亮的、带纹理；空托盘/页面是纯深色。用来预筛候选点，省掉大半无效探测。"""
    patch = crop_at(img, x, y, cw * 0.7, ch * 0.7)
    if patch is None:
        return False
    grey = patch.mean(axis=2)
    return float(grey.std()) > 12.0 or float(grey.mean()) > 62.0


def main() -> int:
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_drops = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    S.enable_dpi_awareness()
    solver = S.Solver(grid=grid, part=1, dry_run=False)
    mouse, clf = solver.mouse, solver.clf
    print("classifier:", "ready" if clf.ready else "MISSING", flush=True)

    img = S.grab()
    board = S.find_board(img) or S.find_board_border(img)
    if board is None:
        print("!! 找不到棋盘")
        return 2
    ref = S.find_reference(img, board, grid)
    if ref is None:
        print("!! 找不到参考面板")
        return 2
    stage = S.find_stage(img, board, ref)
    cw, ch = board.w / grid, board.h / grid
    cells = S.cell_rects(board, grid)
    print(f"board={board.box} ref={ref.box} stage={stage.box} cell=({cw:.0f},{ch:.0f})", flush=True)

    ref_crops = [img[c.y:c.y + c.h, c.x:c.x + c.w] for c in S.cell_rects(ref, grid)]
    labels = [S.BLANK if S.is_blank(c) else clf.predict([c])[0][0] for c in ref_crops]
    print("cell labels:", labels, flush=True)

    cands = S.tray_candidates(stage, board, ref, grid)
    print("candidates:", len(cands), flush=True)

    placed: set[int] = set()
    seen: Counter = Counter()          # 探测到的类别直方图，用来挑目标格
    drops = 0

    for attempt in range(max_drops):
        img = S.grab()
        filled = S.filled_cells(img, board, grid)
        for i in placed:
            filled[i] = True
        anch = S.anchored_cells(filled, grid)
        targets = [i for i in range(grid * grid)
                   if not filled[i] and i in anch and labels[i] != S.BLANK]
        if not targets:
            print("没有可放的图格目标了")
            break
        # 优先挑“托盘里探到过最多”的类别，减少空跑
        targets.sort(key=lambda i: (-seen[labels[i]], i))
        target = targets[0]
        need = labels[target]
        cell = cells[target]
        print(f"\n=== attempt {attempt + 1}: cell{target} class={need} @ {cell.box}", flush=True)

        hover = [(x, y) for (x, y) in cands if looks_like_piece(img, x, y, cw, ch)]
        print(f"    预筛后候选点 {len(hover)}/{len(cands)}", flush=True)

        ok = False
        for (px, py) in hover:
            sx = px + cw * 0.85 if px + cw * 1.8 < img.shape[1] else px - cw * 0.85
            mouse.press(px, py)
            time.sleep(0.04)
            mouse.glide(sx, py)                     # 中转点：验证块是否跟随
            time.sleep(0.06)
            shot = S.grab()
            before = crop_at(img, px, py, cw, ch)
            after = crop_at(shot, px, py, cw, ch)
            moved = (before is not None and after is not None and
                     float(np.abs(after.astype(np.int16) - before.astype(np.int16)).mean()) > 14)
            if not moved:
                mouse.glide(px, py)                 # 精确回到按下点
                mouse.release()
                continue
            piece = crop_at(shot, sx, py, cw * 1.15, ch * 1.15)
            if piece is None:
                mouse.glide(px, py)
                mouse.release()
                continue
            if S.is_blank(piece):
                label, score = S.BLANK, 1.0
            else:
                label, score = clf.predict([piece])[0]
            seen[label] += 1
            print(f"  ({px:.0f},{py:.0f}) -> {label}({score:.2f})", flush=True)
            if label != need:
                mouse.glide(px, py)                 # 类别不对：精确放回原处
                mouse.release()
                time.sleep(0.05)
                continue
            mouse.glide(cell.cx, cell.cy)
            mouse.release()
            time.sleep(0.15)
            drops += 1
            if solver.still_grabbable(cell, grid):
                print("  !! 未吸附，拖回原处", flush=True)
                mouse.drag(cell.cx, cell.cy, px, py)
                continue
            print(f"  ** cell{target} 吸附成功 ({label})", flush=True)
            placed.add(target)
            ok = True
            break
        if not ok:
            print(f"    这一轮没找到 {need} 块（托盘里的类别统计: {seen.most_common(6)}）", flush=True)
        if len(placed) >= grid * grid:
            break

    print(f"\n[done] placed {len(placed)}/{grid * grid}, drops {drops}", flush=True)
    print("tray class tally:", seen.most_common(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
