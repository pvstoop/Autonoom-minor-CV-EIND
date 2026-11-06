"""Zet data/labels om naar de formaten die de modellen willen: data/yolo en data/coco.

Draai dit opnieuw als je labels of data/split.yaml aanpast.

    uv run python scripts/prepare_data.py
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from common import IGNORE, IMAGES, ROOT, SPLIT, load_classes, load_label, load_split, to_corners


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def export_yolo(split: dict, out: Path, classes: list[str]) -> None:
    shutil.rmtree(out, ignore_errors=True)
    for name, stems in split.items():
        for stem in stems:
            ann = load_label(stem)
            link(IMAGES / ann["image"], out / "images" / name / ann["image"])
            size = np.array([ann["width"], ann["height"]])
            lines = []
            for o in ann["objects"]:
                if o["class"] != IGNORE:
                    pts = np.clip(to_corners(o) / size, 0, 1).flatten()
                    lines.append(" ".join([str(classes.index(o["class"]))] + [f"{v:.6f}" for v in pts]))
            label = out / "labels" / name / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + "\n")
    config = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
              "test": "images/test", "names": dict(enumerate(classes))}  # fmt: skip
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def keypoints(o: dict, width: int, height: int) -> list[float]:
    """Koppunt, dan de 4 hoeken (voor de OBB-kop); punten buiten beeld krijgen zichtbaarheid 0."""
    points = [o.get("head")] + to_corners(o).tolist()
    out: list[float] = []
    for p in points:
        visible = p is not None and 0 <= p[0] < width and 0 <= p[1] < height
        out += [float(p[0]), float(p[1]), 2] if visible else [0.0, 0.0, 0]
    return out


def export_coco(split: dict, out: Path, classes: list[str]) -> None:
    shutil.rmtree(out, ignore_errors=True)
    categories = [{"id": i, "name": c, "supercategory": "onderdeel"} for i, c in enumerate(classes)]
    for name, stems in split.items():
        folder = out / ("valid" if name == "val" else name)
        coco = {"images": [], "annotations": [], "categories": categories}
        for image_id, stem in enumerate(stems):
            ann = load_label(stem)
            width, height = ann["width"], ann["height"]
            link(IMAGES / ann["image"], folder / ann["image"])
            coco["images"].append(
                {"id": image_id, "file_name": ann["image"], "width": width, "height": height}
            )
            for o in ann["objects"]:
                if o["class"] == IGNORE:
                    continue
                pts = np.clip(to_corners(o), [0, 0], [width - 1, height - 1])
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                kps = keypoints(o, width, height)
                coco["annotations"].append(
                    {
                        "id": len(coco["annotations"]),
                        "image_id": image_id,
                        "category_id": classes.index(o["class"]),
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "area": float((x1 - x0) * (y1 - y0)),
                        "segmentation": [[float(v) for v in pts.flatten()]],
                        "iscrowd": 0,
                        "keypoints": kps,
                        "num_keypoints": sum(1 for v in kps[2::3] if v > 0),
                    }
                )
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "_annotations.coco.json").write_text(json.dumps(coco))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    classes, split = load_classes(), load_split(args.split)
    export_yolo(split, args.out / "yolo", classes)
    export_coco(split, args.out / "coco", classes)
    print({name: len(stems) for name, stems in split.items()}, "->", args.out / "yolo", args.out / "coco")


if __name__ == "__main__":
    main()
