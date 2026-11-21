"""Zet data/labels om naar het formaat van YOLO: data/yolo.

    uv run python scripts/prepare_data.py

Draai dit opnieuw als je labels of data/split.yaml aanpast.

Naast de 8 onderdelen krijgt YOLO een 9e klasse `kop`: een kleine box op de kop van elke bout
en koppeling. Zo leert YOLO waar de kop zit, en weet evaluate.py welke kant van een bout de
kop is (de geroteerde box zelf zegt dat niet).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from common import IGNORE, IMAGES, ROOT, SPLIT, load_classes, load_label, load_split, to_corners

HEAD = "kop"


def head_box(obj: dict) -> dict | None:
    """Kleine box op de kop van een onderdeel, of None als het geen kop heeft.

    In de labels is `head` het uiteinde aan de kopkant en wijst `angle` van de kop naar de punt.
    De kop-box begint daar, is even breed als het onderdeel en hooguit zo lang als breed.
    """
    if "head" not in obj:
        return None
    length, width = obj["size"]
    direction = np.radians(obj["angle"])
    axis = np.array([np.cos(direction), np.sin(direction)])
    size = min(width, 0.3 * length)
    center = np.array(obj["head"]) + axis * size / 2
    return {"class": HEAD, "center": center.tolist(), "size": [size, width], "angle": obj["angle"]}


def label_lines(objects: list[dict], classes: list[str], image_size: np.ndarray) -> list[str]:
    """Labels van 1 foto in YOLO-OBB-formaat: klasse en 4 hoekpunten, als deel van de foto."""
    lines = []
    for obj in objects:
        if obj["class"] == IGNORE:
            continue
        boxes = [obj, head_box(obj)]
        for box in boxes:
            if box is None:
                continue
            corners = np.clip(to_corners(box) / image_size, 0, 1).flatten()
            numbers = " ".join(f"{v:.6f}" for v in corners)
            lines.append(f"{classes.index(box['class'])} {numbers}")
    return lines


def link(src: Path, dst: Path) -> None:
    """Maak een snelkoppeling naar de foto (scheelt kopieren)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def export_yolo(split: dict, out: Path, classes: list[str]) -> None:
    """Schrijf data/yolo: foto's, labels en data.yaml."""
    shutil.rmtree(out, ignore_errors=True)
    for name, stems in split.items():
        for stem in stems:
            ann = load_label(stem)
            link(IMAGES / ann["image"], out / "images" / name / ann["image"])
            size = np.array([ann["width"], ann["height"]])
            label = out / "labels" / name / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(label_lines(ann["objects"], classes, size)) + "\n")
    config = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(classes)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def main() -> None:
    """Zet de labels om naar data/yolo."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=SPLIT)
    args = ap.parse_args()
    split = load_split(args.split)
    export_yolo(split, ROOT / "data/yolo", load_classes() + [HEAD])
    print({name: len(stems) for name, stems in split.items()}, "->", ROOT / "data/yolo")


if __name__ == "__main__":
    main()
