"""Teken de labels van een foto -> runs/labels/<naam>.jpg. Handig om nieuwe labels te controleren.

uv run python scripts/show.py board_IMG_6241 ps_iphone-16-pro_IMG_6001
"""

import argparse

import cv2
from common import IMAGES, ROOT, draw, load_classes, load_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="fotonamen zonder extensie")
    args = ap.parse_args()
    out = ROOT / "runs/labels"
    out.mkdir(parents=True, exist_ok=True)
    for stem in args.stems:
        img = cv2.imread(str(IMAGES / f"{stem}.jpg"))
        cv2.imwrite(str(out / f"{stem}.jpg"), draw(img, load_label(stem)["objects"], load_classes()))
        print(out / f"{stem}.jpg")


if __name__ == "__main__":
    main()
