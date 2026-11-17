"""Download the site's own catalog photos -> dataset source for training.

URL pattern (from src/data/catalog.js):
    https://puzzle.moyunliuyin.me/assets/catalog/<key>/<key>-<n>.jpg?v=20261115n

Counts are taken from the same file, so the dataset matches exactly what the exam
site will show (12 COCO classes + penguin / tafei / nailong that need labelling).
"""
import json
import os
import time
import urllib.request

BASE = "https://puzzle.moyunliuyin.me/assets/catalog"
VER = "?v=20261115n"
OUT = r"F:\exam_next\raw_assets"

CATEGORIES = {
    "dog": 40, "cat": 40, "elephant": 40, "desktop": 40,
    "bicycle": 20, "car": 20, "chair": 20, "cup": 20,
    "airplane": 20, "backpack": 20, "umbrella": 20, "suitcase": 20,
    "penguin": 39, "tafei": 13, "nailong": 16,
}
# which classes stock COCO weights can already detect, and the closest COCO name
COCO_MAP = {k: (k if k != "desktop" else "tv") for k in CATEGORIES}
CUSTOM = [k for k in CATEGORIES if k in ("penguin", "tafei", "nailong")]


def fetch(url: str, path: str, tries: int = 4) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 8000:
        return True
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 3000:
                raise IOError(f"too small: {len(data)}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception as exc:      # noqa: BLE001
            print(f"  retry {i + 1}/{tries} {os.path.basename(path)}: {exc}", flush=True)
            time.sleep(1.5 * (i + 1))
    return False


def main() -> None:
    report = {}
    total = sum(CATEGORIES.values())
    done = 0
    for key, count in CATEGORIES.items():
        ok = 0
        for n in range(1, count + 1):
            url = f"{BASE}/{key}/{key}-{n}.jpg{VER}"
            path = os.path.join(OUT, key, f"{key}-{n}.jpg")
            if fetch(url, path):
                ok += 1
            done += 1
            if done % 25 == 0:
                print(f"progress {done}/{total}", flush=True)
        report[key] = {"expected": count, "got": ok, "coco": COCO_MAP[key],
                       "needs_finetune": key in CUSTOM}
        print(f"{key:10s} {ok}/{count}", flush=True)
    with open(os.path.join(OUT, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("DONE", sum(v["got"] for v in report.values()), "/", total, flush=True)


if __name__ == "__main__":
    main()
