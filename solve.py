"""自动完成 puzzle.moyunliuyin.me 考核拼图（OpenCV 定位 + YOLO11s 分类 + 鼠标拖拽）。

设计依据全部来自站点源码，见 docs/engine-notes.md：

* 参考面板每格是**同类别的另一张照片** → 像素匹配无效，必须分类；
* 三个 Part 都放 20% 混淆块（5×5 → 25 格 + 5 块 = 30 块），混淆块没有解，
  拖到哪里都吸不上（`checkSnap` 里 `SOLUTIONS.get(piece)` 为空直接 return false）；
* 吸附有“锚点”要求（`hasAnchor`）：只有外圈格子或**相邻格已有块落位**才允许吸附
  → 必须从外圈往内拼（BFS）；
* 吸附容差 `max(0.14×格宽, 10×pixelRatio)` → 必须精确瞄格心；
* 吸附只对“当前被拖拽的块”生效（`this.dragging !== piece` → false）→ 必须真实鼠标按下-移动-抬起。

流程：
  截屏 → 定位棋盘/托盘/参考网格 → 分类参考格与托盘块 → 类别配额分配
       → 外圈向内 BFS 放置 → 每块落位后用“能否再拖动”校验 → 吸不上的判为混淆块
       → 通关校验（placedCount 与通关弹窗）

用法：
  python solve.py --part 3 --grid 5            # 混合模式 5×5（考核主目标）
  python solve.py --grid 4 --dry-run           # 只出规划图，不动鼠标
  python solve.py --image shot.png --grid 5    # 离线：用截图跑规划
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "cls" / "weights" / "best.pt"
CLASSES = ["dog", "cat", "elephant", "desktop", "bicycle", "car", "chair", "cup",
           "airplane", "backpack", "umbrella", "suitcase", "penguin", "tafei", "nailong"]
BLANK = "__blank__"          # 纯白格/纯白块（Part 2 与 Part 3 的空白部分）

# ---------------------------------------------------------------- 屏幕与鼠标


def enable_dpi_awareness() -> None:
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:                       # noqa: BLE001
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:                   # noqa: BLE001
            pass


def grab() -> np.ndarray:
    import pyautogui
    img = np.array(pyautogui.screenshot())
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class Mouse:
    def __init__(self, speed: float = 0.22, settle: float = 0.05) -> None:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
        self.gui = pyautogui
        self.speed = speed
        self.settle = settle

    def drag(self, x0: float, y0: float, x1: float, y1: float) -> None:
        g = self.gui
        g.moveTo(int(x0), int(y0), duration=self.speed * 0.4)
        time.sleep(self.settle)
        g.mouseDown()
        time.sleep(self.settle)
        # 两段移动：先粗后细，模拟真实拖拽（引擎要求 dragging===piece）
        g.moveTo(int(x0 + (x1 - x0) * 0.8), int(y0 + (y1 - y0) * 0.8), duration=self.speed)
        g.moveTo(int(x1), int(y1), duration=self.speed * 0.5)
        time.sleep(self.settle)
        g.mouseUp()
        time.sleep(self.settle)

    def press(self, x: float, y: float) -> None:
        self.gui.moveTo(int(x), int(y), duration=self.speed * 0.3)
        time.sleep(0.02)
        self.gui.mouseDown()

    def glide(self, x: float, y: float) -> None:
        self.gui.moveTo(int(x), int(y), duration=self.speed * 0.6)

    def release(self) -> None:
        self.gui.mouseUp()


# ---------------------------------------------------------------- 视觉定位


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def _components(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_area:
            out.append((x, y, w, h, area))
    return out


def find_board(img: np.ndarray) -> Rect | None:
    """棋盘 = 页面里最大的**平坦**深色矩形（棋盘内部几乎无纹理，四周被控件/拼块包围）。

    实测（1920×1080 实机截图）：棋盘内部是均匀深色，用“局部标准差 < 4”的平坦区
    检测出来是 (453,477,525,387)，长宽比 1.36；页面背景虽然也平坦，但会触到图像边缘，
    因此排除“贴边”的连通域即可把它排除。
    """
    h, w = img.shape[:2]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(grey, -1, (31, 31))
    sq = cv2.boxFilter(grey * grey, -1, (31, 31))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    flat = (std < 4).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    best = None
    for (x, y, ww, hh, area) in _components(flat, int(0.03 * w * h)):
        if x <= 2 or y <= 2 or x + ww >= w - 2 or y + hh >= h - 2:
            continue                        # 触边 = 页面背景，不是棋盘
        if not (0.6 <= ww / hh <= 1.6):
            continue
        if best is None or area > best[4]:
            best = (x, y, ww, hh, area)
    return Rect(*best[:4]) if best else None


def find_board_border(img: np.ndarray, reference: Rect | None = None) -> Rect | None:
    """棋盘兜底检测：用**边框**而不是内部平坦性。

    居中开局时拼块铺在棋盘上，棋盘内部不再是平坦区（实测 `center_5x5.png` 平坦区法失败）。
    但棋盘四周始终有一圈比页面更亮的描边矩形，且**参考面板与棋盘等高**（三张实测图都成立：
    参考面板高 338，棋盘高 387/519），因此用 Canny + 近似矩形 + 高度对齐即可兜底。
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    h, w = grey.shape[:2]
    best = None
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        if peri < 200:
            continue
        ap = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, ww, hh = cv2.boundingRect(ap)
        area = ww * hh
        if not (0.04 * w * h <= area <= 0.45 * w * h):
            continue
        if not (0.8 <= ww / max(1, hh) <= 1.7):
            continue
        if reference is not None and abs(hh - reference.h) > 0.22 * reference.h:
            continue
        if best is None or area > best[0]:
            best = (area, x, y, ww, hh)
    return Rect(*best[1:]) if best else None
def find_reference(img: np.ndarray, board: Rect, grid: int) -> Rect | None:
    """右侧参考面板 = 棋盘右边最大的**明亮**连通域（缩略图是 4:3 照片，整体也是 4:3）。

    实测：面板缩略图网格为 (1436,509,451,338)，比例 1.334 = 4:3，与 5×5 的 4:3 缩略图一致。
    """
    h, w = img.shape[:2]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_axis = np.arange(w)[None, :]
    right = ((grey > 140) & (x_axis > board.x + board.w)).astype(np.uint8)
    right = cv2.morphologyEx(right, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    best = None
    for (x, y, ww, hh, area) in _components(right, int(0.02 * w * h)):
        if ww < board.w * 0.4 or hh < board.h * 0.4:
            continue
        if not (0.9 <= ww / hh <= 1.9):
            continue
        if x + ww > w - 2 and y <= 2:
            continue                        # 浏览器界面元素
        if best is None or area > best[4]:
            best = (x, y, ww, hh, area)
    return Rect(*best[:4]) if best else None


def find_stage(img: np.ndarray, board: Rect, reference: Rect | None) -> Rect:
    """拼图画布：棋盘左右两侧的托盘区域（实测左右各约 0.9 个棋盘宽）。"""
    h, w = img.shape[:2]
    left = max(0, board.x - int(board.w * 1.05))
    right = min(w, (reference.x if reference else board.x + board.w + int(board.w * 1.05)))
    top = max(0, board.y - int(board.h * 0.42))
    bottom = min(h, board.y + board.h + int(board.h * 0.42))
    return Rect(left, top, right - left, bottom - top)


def cell_rects(area: Rect, grid: int) -> list[Rect]:
    cw, ch = area.w / grid, area.h / grid
    return [Rect(int(round(area.x + c * cw)), int(round(area.y + r * ch)),
                 int(round(cw)), int(round(ch)))
            for r in range(grid) for c in range(grid)]


def foreground(img: np.ndarray, stage: Rect, board_bg: tuple[int, int, int]) -> np.ndarray:
    """托盘里的拼块 = 与舞台底色差异明显的像素。"""
    sub = img[stage.y:stage.y + stage.h, stage.x:stage.x + stage.w]
    bg = np.array(board_bg, np.int16)
    diff = np.abs(sub.astype(np.int16) - bg).max(axis=2)
    mask = (diff > 26).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def board_background(img: np.ndarray, board: Rect) -> tuple[int, int, int]:
    """棋盘外框像素的中位色（深蓝），用作托盘前景判定的参考色。"""
    x, y, w, h = board.box
    pad = max(3, int(min(w, h) * 0.01))
    strips = np.concatenate([
        img[max(0, y - pad):y, x:x + w].reshape(-1, 3),
        img[y + h:y + h + pad, x:x + w].reshape(-1, 3),
        img[y:y + h, max(0, x - pad):x].reshape(-1, 3),
        img[y:y + h, x + w:x + w + pad].reshape(-1, 3),
    ])
    if strips.size == 0:
        return (22, 20, 15)
    return tuple(int(v) for v in np.median(strips, axis=0))


# ---------------------------------------------------------------- 模型


class Classifier:
    """YOLO11s-cls；没训练好时退化为整图直方图（至少能跑通流程）。"""

    def __init__(self, weights: Path = WEIGHTS) -> None:
        self.model = None
        self.names = CLASSES
        if weights.exists():
            from ultralytics import YOLO
            self.model = YOLO(str(weights))
            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                self.names = [names[k] for k in sorted(names)]
            elif names:
                self.names = list(names)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        if self.model is None:
            return [("unknown", 0.0) for _ in crops]
        out = []
        for img in crops:
            if img is None or img.size == 0:
                out.append(("unknown", 0.0))
                continue
            res = self.model.predict(img, imgsz=224, verbose=False)[0]
            probs = res.probs
            idx = int(probs.top1)
            out.append((self.names[idx], float(probs.top1conf)))
        return out


def is_blank(crop: np.ndarray) -> bool:
    """纯白块判定：内框像素几乎都在均值 ±12 内（与旧站点同一套经验，已在新站图上验证）。"""
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    inner = crop[int(h * 0.18):int(h * 0.82), int(w * 0.18):int(w * 0.82)]
    if inner.size == 0:
        return False
    grey = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
    uniform = float((np.abs(grey - grey.mean()) < 12).mean())
    return uniform > 0.90


# ---------------------------------------------------------------- 求解


@dataclass
class Piece:
    index: int
    rect: Rect
    label: str = ""
    score: float = 0.0
    blank: bool = False


@dataclass
class Plan:
    grid: int
    board: Rect
    stage: Rect
    reference: Rect | None
    cells: list[Rect] = field(default_factory=list)
    cell_labels: list[str] = field(default_factory=list)
    pieces: list[Piece] = field(default_factory=list)
    assignment: dict[int, int] = field(default_factory=dict)   # piece idx -> cell idx
    decoys: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    centered: bool = False      # 居中开局：块铺在棋盘上，像素填充检测不能当进度用


def assign_by_class(pieces: list[Piece], cell_labels: list[str]) -> tuple[dict[int, int], list[int]]:
    """类别配额分配：某类别有 k 个格子，就取该类别置信度最高的 k 块，其余为混淆块。

    这正是站点 20% 混淆块的克星 —— 混淆块虽然也是 15 类之一，但它所属类别的
    配额已被真块占满，因此必然落在“配额之外”。
    """
    need = Counter(cell_labels)
    by_class: dict[str, list[Piece]] = {}
    for p in pieces:
        by_class.setdefault(p.label, []).append(p)

    assignment: dict[int, int] = {}
    used: set[int] = set()
    # 1) 空白格：用空白块填入（Part2/Part3）
    blank_cells = [i for i, lab in enumerate(cell_labels) if lab == BLANK]
    blank_pieces = sorted([p for p in pieces if p.blank], key=lambda p: -p.rect.w * p.rect.h)
    for cell, piece in zip(blank_cells, blank_pieces):
        assignment[piece.index] = cell
        used.add(piece.index)
    # 2) 图片格：按类别配额，置信度从高到低
    for label, count in need.items():
        if label == BLANK:
            continue
        cands = sorted([p for p in by_class.get(label, []) if p.index not in used],
                       key=lambda p: -p.score)
        cells = [i for i, lab in enumerate(cell_labels) if lab == label and i not in assignment.values()]
        for cell, piece in zip(cells, cands[:count]):
            assignment[piece.index] = cell
            used.add(piece.index)
    decoys = [p.index for p in pieces if p.index not in used]
    return assignment, decoys


def placement_order(cells: int, grid: int) -> list[int]:
    """外圈向内 BFS：`hasAnchor()` 要求格子在外圈或已有相邻块落位。"""
    def anchor(i: int) -> bool:
        r, c = divmod(i, grid)
        return r in (0, grid - 1) or c in (0, grid - 1)

    order: list[int] = []
    seen: set[int] = set()
    q = deque(sorted([i for i in range(grid * grid) if anchor(i)]))
    while q:
        i = q.popleft()
        if i in seen:
            continue
        seen.add(i)
        order.append(i)
        r, c = divmod(i, grid)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid and 0 <= nc < grid:
                j = nr * grid + nc
                if j not in seen and j not in q:
                    q.append(j)
    return order or list(range(grid * grid))


# ---------------------------------------------------------------- 执行


class Solver:
    def __init__(self, grid: int, part: int, dry_run: bool = False) -> None:
        self.grid = grid
        self.part = part
        self.dry_run = dry_run
        self.mouse = Mouse()
        self.clf = Classifier()
        self.logs: list[str] = []

    def log(self, text: str) -> None:
        self.logs.append(text)
        print(text, flush=True)

    # -- 规划 --
    def plan(self, img: np.ndarray) -> Plan:
        board = find_board(img)
        if board is None:
            board = find_board_border(img)          # 居中开局时棋盘内部被块盖住
        if board is None:
            raise RuntimeError("找不到棋盘：请确认游戏页面在最前且已开始某一 Part")
        reference = find_reference(img, board, self.grid)
        stage = find_stage(img, board, reference)
        plan = Plan(grid=self.grid, board=board, stage=stage, reference=reference)
        plan.cells = cell_rects(board, self.grid)
        filled0 = filled_cells(img, board, self.grid)
        plan.centered = sum(filled0) >= len(filled0) - 2
        if plan.centered:
            plan.notes.append("检测到居中开局：块铺在棋盘上，进度改由自记账+吸附校验判断")
        self.log(f"[plan] 棋盘 {board.box} 舞台 {stage.box} 参考面板 {plan.reference}")

        # 参考格类别
        labels: list[str] = []
        if plan.reference is not None:
            ref_cells = cell_rects(plan.reference, self.grid)
            ref_crops = [img[r.y:r.y + r.h, r.x:r.x + r.w].copy() for r in ref_cells]
            for crop, (label, score) in zip(ref_crops, self.clf.predict(ref_crops)):
                labels.append(BLANK if is_blank(crop) else label)
            self.log(f"[plan] 参考格类别 {labels}")
        else:
            labels = [f"cell{i}" for i in range(self.grid * self.grid)]
            plan.notes.append("没找到参考面板，无法确定每格类别（检查窗口是否最大化）")

        # 托盘块
        bg = board_background(img, board)
        fg = foreground(img, stage, bg)
        min_area = int((board.w / self.grid) * (board.h / self.grid) * 0.25)
        pieces: list[Piece] = []
        for (x, y, w, h, _area) in _components(fg, min_area):
            gx, gy = x + stage.x, y + stage.y
            rect = Rect(gx, gy, w, h)
            if board.x - 6 <= rect.cx <= board.x + board.w + 6 and \
               board.y - 6 <= rect.cy <= board.y + board.h + 6:
                continue                     # 已在棋盘上的块
            crop = img[gy:gy + h, gx:gx + w].copy()
            pieces.append(Piece(index=len(pieces), rect=rect, blank=is_blank(crop)))
        self.log(f"[plan] 托盘块 {len(pieces)} 个（期望 {self.grid * self.grid + round(self.grid * self.grid * 0.2)}）")
        for p in pieces:
            crop = img[p.rect.y:p.rect.y + p.rect.h, p.rect.x:p.rect.x + p.rect.w]
            if p.blank:
                p.label, p.score = BLANK, 1.0
        todo = [img[p.rect.y:p.rect.y + p.rect.h, p.rect.x:p.rect.x + p.rect.w] for p in pieces if not p.blank]
        for p, (label, score) in zip([p for p in pieces if not p.blank], self.clf.predict(todo)):
            p.label, p.score = label, score
        plan.pieces = pieces
        plan.cell_labels = labels
        plan.assignment, plan.decoys = assign_by_class(pieces, labels)
        self.log(f"[plan] 分配 {len(plan.assignment)} 块，判定混淆块 {len(plan.decoys)} 个")
        return plan

    # -- 校验：放置后能否再被拖动（吸附的块抓不起来） --
    def still_grabbable(self, cell: Rect, grid: int) -> bool:
        cellw = cell.w
        shot = grab()
        before = shot.copy()
        self.mouse.press(cell.cx, cell.cy)
        self.mouse.glide(cell.cx + cellw * 1.1, cell.cy + cell.h * 0.9)
        during = grab()
        self.mouse.glide(cell.cx, cell.cy)
        self.mouse.release()
        if during.shape != before.shape:
            return True
        x, y, w, h = cell.box
        inner = (slice(y + int(h * 0.25), y + int(h * 0.75)), slice(x + int(w * 0.25), x + int(w * 0.75)))
        moved = float(np.abs(during[inner].astype(np.int16) - before[inner].astype(np.int16)).mean())
        return moved > 12.0

    def run(self) -> dict:
        if not self.clf.ready:
            self.log("[warn] 没找到 runs/cls/weights/best.pt，分类将不可用（先跑 tools/train_yolo.py）")
        img = grab()
        plan = self.plan(img)
        if self.dry_run:
            vis = img.copy()
            for i, cell in enumerate(plan.cells):
                cv2.rectangle(vis, (cell.x, cell.y), (cell.x + cell.w, cell.y + cell.h), (0, 255, 0), 1)
                cv2.putText(vis, str(plan.cell_labels[i])[:6], (cell.x + 4, cell.y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            for p in plan.pieces:
                color = (0, 0, 255) if p.index in plan.decoys else (255, 160, 0)
                cv2.rectangle(vis, (p.rect.x, p.rect.y), (p.rect.x + p.rect.w, p.rect.y + p.rect.h), color, 2)
                cv2.putText(vis, f"{p.label[:8]}:{p.score:.2f}", (p.rect.x + 3, p.rect.y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            out = ROOT / "logs"
            out.mkdir(exist_ok=True)
            path = out / f"plan_{int(time.time())}.png"
            cv2.imwrite(str(path), vis)
            self.log(f"[dry-run] 规划图 -> {path}")
            return {"ok": True, "dry_run": True, "plan": path.name, "logs": self.logs}

        order = placement_order(len(plan.cells), self.grid)
        by_cell = {cell: piece for piece, cell in plan.assignment.items()}
        placed = 0
        for cell_idx in order:
            piece_idx = by_cell.get(cell_idx)
            if piece_idx is None:
                continue
            piece = plan.pieces[piece_idx]
            cell = plan.cells[cell_idx]
            self.mouse.drag(piece.rect.cx, piece.rect.cy, cell.cx, cell.cy)
            time.sleep(0.12)
            if self.still_grabbable(cell, self.grid):
                # 再补一次（0.14 格容差下，微调重放往往就中）
                self.log(f"[drop] #{piece.index}({piece.label}) -> cell{cell_idx} 未吸附，重试")
                piece_in_cell = Piece(piece.index, cell, piece.label, piece.score, piece.blank)
                self.mouse.drag(piece_in_cell.rect.cx, piece_in_cell.rect.cy, cell.cx, cell.cy)
                time.sleep(0.12)
            ok = not self.still_grabbable(cell, self.grid)
            placed += int(ok)
            self.log(f"[drop] #{piece.index}({piece.label}) -> cell{cell_idx} {'吸附成功' if ok else '仍未吸附'}")

        # 混淆块：拖到托盘角落停着（它们永远吸不上，别留在格子上）
        for idx in plan.decoys:
            p = plan.pieces[idx]
            self.log(f"[decoy] #{p.index}({p.label}) 判为混淆块，停在托盘")
        result = {"ok": placed >= len(plan.assignment), "placed": placed,
                  "assigned": len(plan.assignment), "decoys": len(plan.decoys),
                  "logs": self.logs}
        out = ROOT / "logs"
        out.mkdir(exist_ok=True)
        (out / f"run_{int(time.time())}.json").write_text(
            json.dumps({**result, "cell_labels": plan.cell_labels}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return result



# ---------------------------------------------------------------- 探测式拾取


def filled_cells(img: np.ndarray, board: Rect, grid: int,
                 bg: tuple[int, int, int] | None = None) -> list[bool]:
    """逐格判断“已填/空格”，代替识别顶栏的 `进度 N/25`。

    已填的格子显示的是照片：内框中位色与棋盘底色差很大、或者纹理很强；
    空格显示的是棋盘底色本身（几乎没有纹理）。这个结果同时给出 `hasAnchor()`
    需要的“哪些邻居已落位”。
    """
    if bg is None:
        bg = board_background(img, board)
    bg_arr = np.array(bg, np.int16)
    out: list[bool] = []
    for cell in cell_rects(board, grid):
        w, h = cell.w, cell.h
        patch = img[cell.y + int(h * 0.2):cell.y + int(h * 0.8),
                    cell.x + int(w * 0.2):cell.x + int(w * 0.8)]
        if patch.size == 0:
            out.append(False)
            continue
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        colour = np.median(patch.reshape(-1, 3), axis=0).astype(np.int16)
        diff = int(np.abs(colour - bg_arr).max())
        out.append(bool(grey.std() > 6.0 or diff > 34))
    return out


def anchored_cells(filled: list[bool], grid: int) -> set[int]:
    """当前哪些空格可以吸附：外圈格，或至少有一个相邻格已落位（`hasAnchor`）。"""
    ok: set[int] = set()
    for i, is_filled in enumerate(filled):
        if is_filled:
            continue
        r, c = divmod(i, grid)
        if r in (0, grid - 1) or c in (0, grid - 1):
            ok.add(i)
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid and 0 <= nc < grid and filled[nr * grid + nc]:
                ok.add(i)
                break
    return ok


def tray_candidates(stage: Rect, board: Rect, reference: Rect | None,
                    grid: int, centered: bool = False) -> list[tuple[float, float]]:
    """托盘候选抓取点：棋盘左右两条带里按 0.55 格步长铺点。

    拼块在托盘里紧贴甚至互相重叠（实测连通域会把 30 块并成一团），因此不试图“分割”，
    而是把这个点当作“可能抓到某块”的入口，由 probe 判断真假。
    """
    cw, ch = board.w / grid, board.h / grid
    step_x, step_y = max(24.0, cw * 0.55), max(20.0, ch * 0.55)
    if centered:
        # 居中开局：块铺在棋盘矩形内部（带抖动），候选抓取点必须落在棋盘里，
        # 否则 side-band 里一个块都抓不到（实测 center_5x5 侧带内 0 块）。
        pts_c: list[tuple[float, float]] = []
        y = board.y + ch * 0.3
        while y < board.y + board.h - ch * 0.2:
            x = board.x + cw * 0.3
            while x < board.x + board.w - cw * 0.2:
                pts_c.append((round(x, 1), round(y, 1)))
                x += step_x
            y += step_y
        return pts_c
    right_limit = reference.x if reference else stage.x + stage.w
    bands = [(stage.x + cw * 0.25, board.x - cw * 0.25),
             (board.x + board.w + cw * 0.25, right_limit - cw * 0.25)]
    pts: list[tuple[float, float]] = []
    for x0, x1 in bands:
        if x1 - x0 < cw * 0.3:
            continue
        y = stage.y + ch * 0.35
        while y < stage.y + stage.h - ch * 0.2:
            x = x0
            while x < x1:
                pts.append((round(x, 1), round(y, 1)))
                x += step_x
            y += step_y
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--part", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--image", default="", help="离线：用一张截图跑规划")
    ap.add_argument("--focus", default="", help="先把标题包含该字符串的窗口提到最前")
    args = ap.parse_args()

    enable_dpi_awareness()
    if args.focus:
        try:
            import pygetwindow as gw
            for w in gw.getWindowsWithTitle(args.focus):
                if w.title:
                    w.activate()
                    time.sleep(0.4)
                    break
        except Exception as exc:            # noqa: BLE001
            print("focus 失败:", exc)
    if args.image:
        solver = Solver(args.grid, args.part, dry_run=True)
        img = cv2.imread(args.image)
        if img is None:
            print("读不到截图:", args.image)
            return 2
        plan = solver.plan(img)
        print(json.dumps({k: (v if not isinstance(v, (dict, list)) else len(v))
                          for k, v in plan.__dict__.items() if k in ("grid", "notes")},
                         ensure_ascii=False))
        return 0
    solver = Solver(args.grid, args.part, dry_run=args.dry_run)
    res = solver.run()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
