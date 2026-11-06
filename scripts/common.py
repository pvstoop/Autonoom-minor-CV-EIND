"""Gedeelde hulpfuncties: klassen, labels, verdeling en tekenen."""

import json
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/images"
LABELS = ROOT / "data/labels"
SPLIT = ROOT / "data/split.yaml"
IGNORE = "_negeer"  # gebied dat niet meetelt: geen training, en geen score (zie README)
COLORS = [(75, 25, 230), (75, 180, 60), (200, 130, 0), (48, 130, 245),
          (180, 30, 145), (240, 240, 70), (230, 50, 240), (128, 128, 0)]  # fmt: skip


def load_classes() -> list[str]:
    return (ROOT / "classes.txt").read_text().split()


def load_split(path: Path = SPLIT) -> dict[str, list[str]]:
    return yaml.safe_load(Path(path).read_text())


def load_label(stem: str) -> dict:
    return json.loads((LABELS / f"{stem}.json").read_text())


def to_rect(o: dict) -> tuple:
    """Object -> OpenCV-rechthoek ((cx, cy), (w, h), hoek)."""
    return tuple(o["center"]), tuple(o["size"]), o["angle"]


def to_corners(o: dict):
    return cv2.boxPoints(to_rect(o))


def draw(img, objects: list[dict], classes: list[str]):
    """Teken boxen, klassenamen en koppunten; `_negeer` in zwart."""
    out = img.copy()
    for o in objects:
        known = o["class"] in classes
        color = COLORS[classes.index(o["class"]) % len(COLORS)] if known else (0, 0, 0)
        cv2.drawContours(out, [to_corners(o).astype("int32")], 0, color, 3)
        if "head" in o:
            cv2.circle(out, tuple(int(v) for v in o["head"]), 9, color, -1)
        org = (int(o["center"][0]) - 30, int(o["center"][1]) + 8)
        cv2.putText(out, o["class"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5)
        cv2.putText(out, o["class"], org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out
