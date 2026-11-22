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
    def __init__(self, size: tuple[int, int] = (112, 85),
                 central: tuple[int, int] | None = (72, 56),
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
        if self.central is not None:
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

    def match_aligned(self, img: np.ndarray, box: tuple[int, int, int, int], radius: int = 8,
                      step: int = 2) -> Match:
        """Best catalog match over small offsets of the crop.

        Tray slots are a grid, but a piece sits inside its slot with a couple of pixels of
        slack and its jigsaw outline eats the corners, so the first crop is often a few
        pixels off.  Sliding the crop and keeping the best score fixes both at once and
        needs no edge detection: measured uplift 0.45 -> 0.83-0.955 on the good pieces.
        """
        x, y = box[0], box[1]
        w, h = self.size
        best = Match("?", "", -1.0, "?", -1.0)
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                X, Y = x + dx, y + dy
                if X < 2 or Y < 2 or X + w > img.shape[1] - 2 or Y + h > img.shape[0] - 2:
                    continue
                m = self.match(img[Y:Y + h, X:X + w])
                if m.score > best.score:
                    best = m
                    best.path = f"{m.path}|offset({dx},{dy})"
        return best

    def match_masked(self, img: np.ndarray, box: tuple[int, int, int, int],
                     radius: int = 8, step: int = 4) -> Match:
        """Catalog match using only the piece's own photo pixels.

        A slot box contains the piece plus background where a notch is carved and, at the
        corners, whatever is behind the jigsaw outline.  Feeding those pixels in drags the
        correlation down (0.3-0.7 measured).  Here the piece mask is taken from brightness
        (tray background is 9, every photo is brighter), holes are filled so a dark photo is
        not punched through, and the correlation is computed over that mask only.
        """
        w, h = self.size
        if self.central is not None:
            # masked correlation needs the whole cell in the feature vector
            self.central = None
            self.__init__(size=self.size, central=None, cache=CACHE.with_name("catalog_full.npz"))
        feats = self.feats
        best = Match("?", "", -1.0, "?", -1.0)
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                X, Y = box[0] + dx, box[1] + dy
                if X < 2 or Y < 2 or X + w > img.shape[1] - 2 or Y + h > img.shape[0] - 2:
                    continue
                crop = cv2.resize(img[Y:Y + h, X:X + w], self.size)
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                m = (g > 20).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                n_lab, lab, st, _ = cv2.connectedComponentsWithStats(m)
                if n_lab < 2:
                    continue
                cy, cx = h // 2, w // 2
                seed = lab[cy, cx]
                if seed == 0:                      # centre fell on a notch: take the biggest
                    seed = 1 + int(np.argmax(st[1:, 4]))
                m = (lab == seed).astype(np.uint8)
                ff = m.copy()
                cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
                holes = (ff == 0).astype(np.uint8)
                m = np.clip(m + holes, 0, 1)
                if m.sum() < 3000:
                    continue
                mask = m.ravel().astype(np.float32)
                k = float(mask.sum())
                q = g.ravel()
                qm = float((q * mask).sum() / k)
                qz = q - qm
                qn = float(np.sqrt(((qz ** 2) * mask).sum()))
                if qn <= 1e-6:
                    continue
                csum = feats @ mask
                csq = (feats ** 2) @ mask
                cmean = csum / k
                cvar = csq - k * cmean ** 2
                cvar = np.clip(cvar, 1e-6, None)
                num = feats @ (qz * mask)
                scores = num / (qn * np.sqrt(cvar))
                i = int(np.argmax(scores))
                if float(scores[i]) > best.score:
                    best = Match(self.labels[i], f"{self.paths[i]}|offset({dx},{dy})",
                                 float(scores[i]), "?", 0.0)
        if best.score > 0:
            q = self._feature(img[box[1]:box[1] + h, box[0]:box[0] + w])
            allsc = self.feats @ q
            for j in np.argsort(allsc)[::-1]:
                if self.labels[int(j)] != best.class_name:
                    best.runner_up_class = self.labels[int(j)]
                    best.runner_up_score = float(allsc[int(j)])
                    break
        return best


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
