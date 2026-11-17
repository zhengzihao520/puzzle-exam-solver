# 考核站点引擎关键规则（读源码所得，决定算法设计）

站点：<https://puzzle.moyunliuyin.me/?mode=exam>
存档文件：`recon/main.js`、`recon/catalog.js`、`recon/exame_mode.js`（`src/exam/ExamMode.js`）、`recon/puzzle_board.js`（`src/engine/PuzzleBoard.js`）

## 1. 参考图不是原图（必须做分类）

`src/data/catalog.js: alternatePhoto()`

```js
// A different photo of the same class: the reference grid uses this one, so a
// template match against the reference cannot solve the board - the class has to
// be recognised.
```

右侧参考面板每格用的是**同类别另一张照片**，拼块用的是原图。像素/NCC 模板匹配从原理上失效，必须做**类别识别**（YOLO）。

## 2. 混淆块（20%）

`main.js`
```js
const wantedDecoys = Math.max(1, Math.round(count * 0.2));   // 5x5 -> 5 块
```
- 5×5：25 格 + 5 块混淆 = **30 块**
- 混淆块从“未被使用的同源图库”里随机取，**类别可能与真块重复**
- Part 2 的混淆块是纯白片：*only tab geometry and the leftover slots give them away*

## 3. 吸附判定的真实规则（关键）

`src/engine/PuzzleBoard.js: checkSnap()`

```js
if (this.dragging !== piece) return false;                  // 必须是当前被拖拽的块
const spot = SOLUTIONS.get(piece); if (!spot) return false; // 混淆块没有解，永远吸不上
if (!this.hasAnchor(piece)) return false;                   // 必须“有锚点”才能吸附
const snapThreshold = Math.max(this.pieceWidth * 0.14, 10 * (this.pixelRatio || 1));
```

三条结论直接影响实现：

1. **混淆块永远不会吸附** → 不用靠类别去猜哪块是假的：依次尝试所有空格，全都吸不上的那块就是混淆块，park 到旁边即可。
2. **必须从边框向内拼**（`hasAnchor`）：格的 `row/col` 在最外圈，或**相邻格已有块落位**时才允许吸附。中间格孤零零地放上去不会粘住。→ 放置顺序必须是**从外圈向内的拓扑序（BFS）**。
3. **容差收紧到 0.14×格宽（且 ≥10 物理像素）**，比旧站 0.28 严格一倍 → 必须**精确瞄准格心**，随手一放不会吸。
4. 吸附成立即等价于“放对了”（`SOLUTIONS` 是每块唯一正确格），所以落位后可以用“还能不能把它拖走”作为校验。

## 4. 附加题：居中开局 / 重叠开局

`src/exam/ExamMode.js`（面板开关 `exam-overlap-toggle` / `exam-center-toggle`）：
```js
this.app.board.overlapStart = overlap || centred;
this.app.board.centerStart = centred;
```

`src/engine/PuzzleBoard.js: overlapPositions(count)`
- **居中开局**：在棋盘矩形内铺一层带抖动的格点（`ax0/ax1/ay0/ay1` 取棋盘 4%~96% 区域，抖动量 `±0.25*step`），块之间只有轻微重叠、无遮挡堆叠。
- **重叠开局**：块堆在左右两条侧带里的若干堆上，并有一次“修复遍历”确保没有块被完全遮住。

对求解器的影响：开局时块**互相覆盖**，所以要先按“最上层优先”逐块拾取；被盖住的块在下层，需要先把上层块移走。因吸附容差是 0.14 格，居中开局时块与格心的距离可能超出容差，仍需拖到格心。

## 5. 通关校验

`main.js`
```js
// Only a board that was genuinely assembled may finish the part.
} else if (this.board.placedCount !== this.board.totalPieces) { ... }
```
`placedCount === totalPieces` 才允许通关 → 混淆块也计入 totalPieces，说明**混淆块也要被处理**（要么被判定为假块按住不管，要么被移出托盘区域），实现上要留意这一点。
