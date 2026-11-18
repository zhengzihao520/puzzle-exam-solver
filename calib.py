"""单块标定：拿一块，按顺序拖到 25 个格心，用"进度数字变化"找出它真正的家。

这是决定性实验：如果 25 个格心全部试完进度都不变，说明不是格心不准，而是拖拽/缩放
映射出了问题；只要有一次 +1，就同时验证了坐标映射与吸附机制，并可据此校准全盘。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import solve as S

PROG = (1540, 225, 1790, 285)


def diff(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.abs(b.astype(np.int16) - a.astype(np.int16)).mean())


def prog(img):
    x0, y0, x1, y1 = PROG
    return img[y0:y1, x0:x1].copy()


def main() -> int:
    S.enable_dpi_awareness()
    solver = S.Solver(grid=5, part=1, dry_run=False)
    mouse, clf = solver.mouse, solver.clf
    img = S.grab()
    board = S.find_board(img) or S.find_board_border(img)
    ref = S.find_reference(img, board, 5)
    stage = S.find_stage(img, board, ref)
    cells = S.cell_rects(board, 5)
    cw, ch = board.w / 5, board.h / 5
    print(f"board={board.box} ref={ref.box} cell=({cw:.0f}x{ch:.0f})", flush=True)

    # 取托盘里第一个"像块"的点（左上区域优先，那是初始布局起点）
    from live import piece_like  # noqa: E402  复用同一条判据
    pick = None
    y = stage.y + ch * 0.4
    while y < stage.y + stage.h and pick is None:
        x = stage.x + cw * 0.4
        while x < board.x - cw * 0.1:
            if piece_like(img, x, y, cw, ch):
                pick = (x, y)
                break
            x += cw * 0.4
        y += ch * 0.4
    if pick is None:
        print("!! 托盘里没找到块")
        return 2
    print(f"pick piece at ({pick[0]:.0f},{pick[1]:.0f})", flush=True)

    for idx, cell in enumerate(cells):
        for dx, dy in ((0, 0), (0, -4), (0, 4), (-4, 0), (4, 0)):
            p0 = prog(S.grab())
            mouse.drag(pick[0], pick[1], cell.cx + dx, cell.cy + dy)
            time.sleep(0.16)
            p1 = prog(S.grab())
            changed = diff(p0, p1) > 6.0
            print(f"  cell{idx:2d} @({cell.cx:.0f},{cell.cy:.0f})+({dx},{dy}) 进度变化={changed}", flush=True)
            if changed:
                print(f"\n*** 命中：cell{idx} 是真的，格心=({cell.cx:.0f},{cell.cy:.0f})", flush=True)
                return 0
            # 没吸附：把块拖回去
            mouse.drag(cell.cx + dx, cell.cy + dy, pick[0], pick[1])
            time.sleep(0.1)
    print("\n!!! 25 个格心全部试完，进度从未变化 → 映射/拖拽机制有问题", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
