"""Collect extra training photos for the three non-COCO classes from Bing Images.

Why: the exam site ships only penguin 39 / tafei 13 / nailong 16 photos.  Stock
YOLO has no such classes, so they must be fine-tuned - and 13 images is far too
few.  This script pulls more pictures of the same subjects from the web.

Bing's normal result page renders its list with JavaScript, but the ``async``
endpoint returns the same cards as plain HTML with ``murl`` entries, which is what
this uses.  Every download is validated with OpenCV, resized to a sane bound and
de-duplicated, so a broken/HTML error page never lands in the dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

OUT = r"F:\exam_next\raw_assets"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# several phrasings per class: face/whole-body/product shots all help a classifier
QUERIES = {
    "penguin": ["企鹅", "企鹅 南极", "penguin antarctica", "帝企鹅"],
    "tafei": ["永雏塔菲", "永雏塔菲 立绘", "Tafei vtuber", "塔菲 表情包"],
    "nailong": ["奶龙", "奶龙 表情包", "奶龙 卡通", "奶龙 玩偶"],
}
TARGET = 120          # extra images per class
PAGES = 4             # async pages per query (35 cards each)


def bing_urls(query: str, page: int) -> list[str]:
    q = urllib.parse.quote(query)
    url = (f"https://www.bing.com/images/async?q={q}&first={1 + page * 35}"
           f"&count=35&mmasync=1&qft=")
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Referer": "https://www.bing.com/images/search"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", "replace")
            found = [urllib.parse.unquote(m) for m in
                     re.findall(r'murl&quot;:&quot;(.+?)&quot;', html)]
            return found
        except Exception as exc:                      # noqa: BLE001
            print(f"    search retry {attempt + 1}: {exc}", flush=True)
            time.sleep(1.5)
    return []


def download(url: str, path: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read(4_000_000)
    except Exception:                                  # noqa: BLE001
        return False
    if len(data) < 12_000:
        return False
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.shape[0] < 200 or img.shape[1] < 200:
        return False
    h, w = img.shape[:2]
    scale = 900.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return True


def main() -> None:
    summary = {}
    for key, queries in QUERIES.items():
        folder = os.path.join(OUT, key, "web")
        os.makedirs(folder, exist_ok=True)
        have = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
        seen = set()
        got = have
        print(f"[{key}] start, already {have}", flush=True)
        for query in queries:
            if got >= TARGET:
                break
            for page in range(PAGES):
                if got >= TARGET:
                    break
                urls = bing_urls(query, page)
                for u in urls:
                    if got >= TARGET:
                        break
                    h = hashlib.sha1(u.encode()).hexdigest()[:12]
                    if h in seen:
                        continue
                    seen.add(h)
                    path = os.path.join(folder, f"{key}-web-{h}.jpg")
                    if os.path.exists(path):
                        continue
                    if download(u, path):
                        got += 1
                        if got % 10 == 0:
                            print(f"  {key}: {got}", flush=True)
                time.sleep(0.6)
        summary[key] = got
        print(f"[{key}] total {got}", flush=True)
    path = os.path.join(OUT, "_web_extra.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
