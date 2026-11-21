"""Gedeelde hulpfuncties: klassen, labels, verdeling en tekenen."""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/images"
LABELS = ROOT / "data/labels"
SPLIT = ROOT / "data/split.yaml"
IGNORE = "_negeer"  # gebied dat niet meetelt: geen training, en geen score (zie README)
COLORS = [(75, 25, 230), (75, 180, 60), (200, 130, 0), (48, 130, 245),
          (180, 30, 145), (240, 240, 70), (230, 50, 240), (128, 128, 0)]  # fmt: skip


def load_classes() -> list[str]:
    """De klassenamen uit classes.txt, in vaste volgorde."""
    return (ROOT / "classes.txt").read_text().split()


def load_split(path: Path = SPLIT) -> dict[str, list[str]]:
    """Welke foto in train, val of test zit."""
    return yaml.safe_load(Path(path).read_text())


def load_label(stem: str) -> dict:
    """Label van een foto (naam zonder extensie)."""
    return json.loads((LABELS / f"{stem}.json").read_text())


def to_rect(o: dict) -> tuple:
    """Object -> OpenCV-rechthoek ((cx, cy), (w, h), hoek)."""
    return tuple(o["center"]), tuple(o["size"]), o["angle"]


def to_corners(o: dict):
    """Object -> de 4 hoekpunten van de geroteerde box."""
    return cv2.boxPoints(to_rect(o))


def iou(a: dict, b: dict) -> float:
    """Overlap (IoU) van twee geroteerde boxen."""
    area_a, area_b = a["size"][0] * a["size"][1], b["size"][0] * b["size"][1]
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(to_corners(a), to_corners(b))
    return max(float(inter), 0.0) / (area_a + area_b - max(float(inter), 0.0) + 1e-9)


def color_of(cls: str, classes: list[str]) -> tuple:
    """Vaste kleur per klasse; `_negeer` (geen klasse) wordt zwart."""
    if cls not in classes:
        return (0, 0, 0)
    return COLORS[classes.index(cls) % len(COLORS)]


def draw_head(img, o: dict, color: tuple) -> None:
    """Stip op de kop en een pijl van de kop naar de punt."""
    x, y = o["head"]
    angle = np.radians(o["angle"])
    length = max(o["size"])
    tip = (int(x + length * np.cos(angle)), int(y + length * np.sin(angle)))
    head = (int(x), int(y))
    cv2.circle(img, head, 9, color, -1)
    cv2.arrowedLine(img, head, tip, color, 3, tipLength=0.1)


def draw_name(img, o: dict, color: tuple) -> None:
    """Klassenaam (met de maat als die er is) in het midden, met een witte rand eromheen."""
    x, y = o["center"]
    org = (int(x) - 30, int(y) + 8)
    text = f"{o['class']} {o['maat']}" if "maat" in o else o["class"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, 0.8, (255, 255, 255), 5)
    cv2.putText(img, text, org, font, 0.8, color, 2)


def draw(img, objects: list[dict], classes: list[str]):
    """Teken per object de box, de kop (als die er is) en de klassenaam met maat."""
    out = img.copy()
    for o in objects:
        color = color_of(o["class"], classes)
        cv2.drawContours(out, [to_corners(o).astype("int32")], 0, color, 3)
        if "head" in o:
            draw_head(out, o, color)
        draw_name(out, o, color)
    return out
