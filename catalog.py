"""Identify a piece by matching it against the site's own catalog.

The board's cells are whole photos from the catalog, stretched into a cell, so a tray piece
is one catalog image scaled down -- not a fragment.  Therefore the exact source photo can be
found by comparing the piece against all 748 catalog images, and its folder name *is* the
class.  That replaces the probabilistic classifier with a lookup that has essentially one
correct answer.

Catalog images are pre-resized to the cell aspect exactly the way the site draws them
(`drawImage(img, cell)` stretches, it does not crop), so a correct match is a near-exact
pixel match rather than a semantic guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "raw_assets"
CACHE = ROOT / "logs" / "catalog_features.npz"


@dataclass
class Match:
    class_name: str
    path: str
    score: float
    runner_up_class: str
    runner_up_score: float


class Catalog:
    def __init__(self, size: tuple[int, int] = (112, 85), central: tuple[int, int] = (72, 56),
                 cache: Path = CACHE) -> None:
        self.size = size
        self.central = central
        if cache.exists():
            data = np.load(cache)
            if tuple(data["size"]) == size and len(data["labels"]) > 0:
                self.feats = data["feats"]
                self.labels = [str(x) for x in data["labels"]]
                self.paths = [str(x) for x in data["paths"]]
                return
        feats, labels, paths = [], [], []
        for path in sorted(ASSETS.rglob("*.jpg")):
            img = cv2.imread(str(path))
            if img is None:
                continue
            feats.append(self._feature(img))
            parent = path.parent
            # raw_assets/<class>/web/*.jpg from the web-extra fetch: the class is one level up
            labels.append(parent.parent.name if parent.name == "web" else parent.name)
            paths.append(str(path))
        self.feats = np.stack(feats)
        self.labels = labels
        self.paths = paths
        cache.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache, feats=self.feats, labels=np.array(labels),
                            paths=np.array(paths), size=np.array(size))

    def _feature(self, img: np.ndarray) -> np.ndarray:
        """Zero-mean unit-norm patch of the cell centre.

        Only the middle is compared: a piece is clipped to its jigsaw outline, so its
        corners are background, and including them drags the correlation down (measured:
        whole-cell NCC 0.45 vs 0.84 for reference thumbnails that are not clipped).
        """
        g = cv2.cvtColor(cv2.resize(img, self.size), cv2.COLOR_BGR2GRAY).astype(np.float32)
        cw, ch = self.central
        x0 = (self.size[0] - cw) // 2
        y0 = (self.size[1] - ch) // 2
        g = g[y0:y0 + ch, x0:x0 + cw]
        g -= g.mean()
        n = float(np.linalg.norm(g))
        return (g / n if n else g).ravel()

    def match(self, cell: np.ndarray) -> Match:
        q = self._feature(cell)
        if not q.any():
            # a flat centre means a white decoy piece: nothing to correlate against
            return Match("blank", "", 0.0, "blank", 0.0)
        scores = self.feats @ q
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        best_class = self.labels[best]
        runner_class, runner_score = "?", 0.0
        for i in order[1:]:
            if self.labels[int(i)] != best_class:
                runner_class = self.labels[int(i)]
                runner_score = float(scores[int(i)])
                break
        return Match(best_class, self.paths[best], float(scores[best]),
                     runner_class, runner_score)


if __name__ == "__main__":
    import sys

    import vision

    cat = Catalog()
    print(f"catalog: {len(cat.labels)} images over {len(set(cat.labels))} classes")
    img = vision.grab() if len(sys.argv) == 1 else cv2.imread(sys.argv[1])
    ref, board, _ = vision.layout(img)
    rows = vision.tray_rows(img)
    print("\ntray pieces -> best catalog match")
    for band in ("left", "right"):
        for ri, ci, box in vision.tray_slots(img, band, rows):
            cell = vision.sub(img, box)
            if vision.is_empty(cell) or vision.is_white_piece(cell):
                continue
            m = cat.match(cell)
            print(f"  {band[0]}{ri + 1}{ci + 1}: {m.class_name:10s} score {m.score:.3f} "
                  f"(next class {m.runner_up_class} {m.runner_up_score:.3f})  "
                  f"{Path(m.path).name}")
