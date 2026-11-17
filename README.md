# 拼图考核自动完成（OpenCV + YOLO11s + 鼠标自动化）

目标站点：<https://puzzle.moyunliuyin.me/?mode=exam>

要求：用 OpenCV 识别屏幕、用自训练 YOLO 识别类别、用代码驱动鼠标完成拼图。
必须能完成 **Part 3（混合模式）+ 5×5**；附加题：重叠开局、居中开局。

## 1. 这个站点和旧站的关键差异（决定了算法）

读站点源码（`src/main.js`、`src/data/catalog.js`）确认：

| 现象 | 源码依据 | 对算法的影响 |
|---|---|---|
| 参考面板每格是**同类别的另一张照片** | `catalog.js: alternatePhoto()`，注释明确写 “a template match against the reference cannot solve the board - the class has to be recognised” | **模板/NCC 像素匹配失效**，必须做**类别识别** |
| 三个 Part 都会放**混淆块** | `main.js: board.decoySources`，`wantedDecoys = round(count * 0.2)` | 5×5 = 25 格 + **5 块多余 = 30 块**；必须按**类别配额**剔除假块 |
| 取消了吸附判定 | 无 `checkSnap` 之类的落位校验 | 放错不会被引擎纠正 → **只能靠分类正确性保证结果** |
| 通关需真实拼合 | `// Only a board that was genuinely assembled may finish the part.` | 收官要有校验（通关弹窗/进度） |
| Part 2 混淆块是纯白片 | `renderWhitePuzzle()`：*only tab geometry and the leftover slots give them away* | 纯白格只能靠 OpenCV 凹凸边/轮廓几何 |

## 2. 算法总览

```
截屏 ──OpenCV──► 棋盘矩形 / 托盘区域 / 右侧参考网格
                    │
                    ├─► YOLO11s 分类每个参考格  → cell_class[25]
                    └─► YOLO11s 分类每个托盘块  → piece_class[30] + 置信度
                                        │
                    类别配额分配（同类别格子数 = 用几块，其余为混淆块）
                                        │
                    纯白格 / 纯白块 → OpenCV 凹凸边几何解算
                                        │
                     鼠标拖拽执行（无吸附反馈，按分配一次到位）
                                        │
                        收官校验：通关弹窗 / 进度 = n/n
```

## 3. 目录结构

```
recon/            站点源码存档（仅用于研究，不随仓库发布）
tools/            数据集下载与预处理脚本
dataset/          训练数据（分类集 / YOLO 检测集 / labelImg XML）
runs/             YOLO 训练痕迹（weights、results.csv、confusion_matrix…）
solve.py         主程序（单文件可运行）
README.md
```

## 4. 复现步骤

```powershell
# 1) 下载站点同源图库（15 类约 385 张）
python tools\fetch_catalog.py

# 2) 生成数据集（分类集 + 检测集 + labelImg XML）
python tools\build_dataset.py

# 3) 训练：12 类用 COCO 官方权重，企鹅/永雏塔菲/奶龙用自标数据微调
python tools\train_yolo.py

# 4) 运行
python solve.py --part 3 --grid 5
```

## 5. 题库与模型

| 预训练（COCO 官方权重） | 需自建标注微调 |
|---|---|
| dog, cat, elephant, bicycle, car, chair, cup, airplane, backpack, umbrella, suitcase, desktop(≈tv) | penguin, tafei(永雏塔菲), nailong(奶龙) |

## 6. 状态

- [x] 站点性质与规则 recon（混淆块、参考图同类异图、无吸附判定）
- [ ] 图库下载（网络受阻，进行中）
- [ ] 数据集与 YOLO11s 训练
- [ ] 求解器与鼠标执行
- [ ] 实机验证 Part3 + 5×5
- [ ] 演示视频（由组员录制）
