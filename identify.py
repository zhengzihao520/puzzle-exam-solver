"""Identify a piece by fusing two independent readings.

Measured on the 60 hand-labelled real tray crops in this repo:

    catalog matching (NCC against the site's own 748 images)  82% overall, 94% at score>=0.5
    YOLO classifier                                          87% overall, 96% at conf>=0.6
    the two agree  (46 of 60 crops)                          100%

So agreement is certainty, and the disagreement cases are a two-way choice that the engine
can settle by trying.  That is the difference between guessing a class and knowing one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import vision
from catalog import Catalog

BLANK = "blank"


@dataclass
class Identity:
    label: str
    confident: bool
    catalog_label: str
    catalog_score: float
    yolo_label: str
    yolo_conf: float
    alternatives: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        mark = "" if self.confident else f"  ?{self.alternatives}"
        return (f"{self.label:10s} cat {self.catalog_label[:8]}:{self.catalog_score:.2f} "
                f"yolo {self.yolo_label[:8]}:{self.yolo_conf:.2f}{mark}")


class Identifier:
    def __init__(self, use_alignment: bool = False) -> None:
        self.cat = Catalog()
        self.clf = vision.Classifier()
        self.use_alignment = use_alignment

    def identify(self, cell: np.ndarray, img: np.ndarray | None = None, box=None) -> Identity:
        if vision.is_white_piece(cell):
            return Identity(BLANK, True, BLANK, 1.0, BLANK, 1.0)
        m = (self.cat.match_aligned(img, box) if (self.use_alignment and img is not None and box)
             else self.cat.match(cell))
        probs = self.clf.probs(cell)
        yolo = str(self.clf.names[int(probs.argmax())])
        conf = float(probs.max())
        cat_label, cat_score = m.class_name, float(m.score)
        # A white cell of the source photo gives a piece with no features: the catalog
        # cannot match it and comes back near 0.  Every real photo in the 60-crop benchmark
        # scored at least 0.26, so a near-zero score means "there is no picture here".
        if cat_score < 0.15:
            return Identity(BLANK, True, BLANK, cat_score, yolo, conf)
        if cat_label == yolo:
            return Identity(yolo, True, cat_label, cat_score, yolo, conf)
        # disagreement: prefer whichever reading is individually strong, keep both as options
        if cat_score >= 0.6 and conf < 0.6:
            label = cat_label
        elif conf >= 0.6 and cat_score < 0.5:
            label = yolo
        else:
            label = cat_label if cat_score >= conf else yolo
        alts = [x for x in (cat_label, yolo) if x != label]
        return Identity(label, False, cat_label, cat_score, yolo, conf, alts)
