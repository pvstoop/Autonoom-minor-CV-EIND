"""Test het getrainde model op de testset of een map met eigen foto's."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import joblib
import numpy as np
from shapely.geometry import Polygon
import yaml

from classical_pipeline import (
    BOLT_CLASSES,
    CLASSES,
    detect_candidates,
    extract_features,
    load_annotations,
    rect_to_polygon,
    rotated_iou,
)


def ap50_for_class(preds: list[dict], gts_by_img: dict[str, list[Polygon]]) -> float:
    """Berekent de AP@50 voor een klasse bij IoU >= 0.50."""
    total_gt = sum(len(polys) for polys in gts_by_img.values())
    if total_gt == 0 or not preds:
        return 0.0

    ordered = sorted(preds, key=lambda item: item["conf"], reverse=True)
    matched: dict[str, set[int]] = {img_id: set() for img_id in gts_by_img}
    tp = np.zeros(len(ordered), dtype=np.float64)
    fp = np.zeros(len(ordered), dtype=np.float64)

    for i, pred in enumerate(ordered):
        img_id = pred["image"]
        best_iou, best_gt = 0.0, -1
        for g_idx, g_poly in enumerate(gts_by_img.get(img_id, [])):
            if g_idx in matched[img_id]:
                continue
            iou = rotated_iou(pred["poly"], g_poly)
            if iou > best_iou:
                best_iou, best_gt = iou, g_idx

        if best_iou >= 0.50 and best_gt >= 0:
            matched[img_id].add(best_gt)
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    rec = np.cumsum(tp) / total_gt
    prec = np.cumsum(tp) / np.maximum(1.0, np.cumsum(tp) + np.cumsum(fp))

    ap = 0.0
    for r_thr in np.linspace(0.0, 1.0, 101):
        vals = prec[rec >= r_thr]
        ap += float(vals.max()) if vals.size > 0 else 0.0
    return ap / 101.0


def print_metrics(
    title: str,
    stems: set[str],
    all_preds: list[dict],
    all_gts: list[dict],
    min_conf: float,
) -> None:
    """Print mAP@50, precision, recall en F1 voor een groep foto's."""
    sub_preds = [p for p in all_preds if p["image"] in stems]
    sub_gts = [g for g in all_gts if g["image"] in stems]

    aps: list[float] = []
    tp_total, fp_total, gt_total = 0, 0, len(sub_gts)

    for cls_name in CLASSES:
        cls_preds = [p for p in sub_preds if p["class"] == cls_name]
        gts_by_img: dict[str, list[Polygon]] = {s: [] for s in stems}
        for gt in sub_gts:
            if gt["class"] == cls_name:
                gts_by_img[gt["image"]].append(gt["poly"])

        aps.append(ap50_for_class(cls_preds, gts_by_img))

        filtered = [p for p in cls_preds if p["conf"] >= min_conf]
        matched: dict[str, set[int]] = {s: set() for s in stems}
        for pred in sorted(filtered, key=lambda x: x["conf"], reverse=True):
            best_iou, best_idx = 0.0, -1
            for idx, g_poly in enumerate(gts_by_img[pred["image"]]):
                if idx in matched[pred["image"]]:
                    continue
                iou = rotated_iou(pred["poly"], g_poly)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= 0.50 and best_idx >= 0:
                matched[pred["image"]].add(best_idx)
                tp_total += 1
            else:
                fp_total += 1

    map50 = float(np.mean(aps)) * 100.0
    prec = (tp_total / (tp_total + fp_total) * 100.0) if (tp_total + fp_total) else 0.0
    rec = (tp_total / gt_total * 100.0) if gt_total else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    print(f"\n=== {title} ({len(stems)} foto's, {gt_total} GT) ===")
    print(f"  mAP@50    : {map50:5.1f}%")
    print(f"  Precision : {prec:5.1f}%  (TP={tp_total}, FP={fp_total})")
    print(f"  Recall    : {rec:5.1f}%  ({tp_total}/{gt_total})")
    print(f"  F1-score  : {f1:5.1f}%")


def draw_result(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Tekent de gevonden boxen, labels en koprichting op een kopie van de foto."""
    canvas = image_bgr.copy()
    for det in detections:
        rect = ((det["cx"], det["cy"]), (det["w"], det["h"]), det["angle"])
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(canvas, [box], 0, (0, 220, 0), 3)
        cv2.putText(
            canvas,
            f"{det['class']} {det['conf']:.2f}",
            (int(det["cx"]) - 40, int(det["cy"]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        if det["head_angle"] is not None:
            rad = np.radians(det["head_angle"])
            length = max(det["w"], det["h"]) * 0.45
            end_pt = (
                int(round(det["cx"] + np.cos(rad) * length)),
                int(round(det["cy"] + np.sin(rad) * length)),
            )
            cv2.arrowedLine(
                canvas,
                (int(round(det["cx"])), int(round(det["cy"]))),
                end_pt,
                (0, 0, 255),
                3,
                tipLength=0.25,
            )
    return canvas


def parse_args() -> argparse.Namespace:
    """Leest de argumenten vanaf de command line."""
    parser = argparse.ArgumentParser(description="Evalueer klassieke CV-pipeline.")
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--min-conf", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Draait de evaluatie, maakt een CSV aan en slaat plaatjes met boxen op."""
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    labels_dir = root / "data" / "labels"
    results_dir = root / "results"
    vis_dir = root / "runs" / "classical_eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(root / "models" / "classical_classifier.joblib")
    clf = bundle["model"]
    min_conf = (
        float(args.min_conf)
        if args.min_conf is not None
        else float(bundle.get("min_conf", 0.25))
    )
    clf_classes = list(clf.classes_)

    if args.images_dir is not None:
        image_paths = sorted(
            p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        has_gt = False
    else:
        splits = yaml.safe_load((root / "data" / "split.yaml").read_text(encoding="utf-8"))
        images_dir = root / "data" / "images"
        image_paths = [
            matches[0]
            for name in splits.get("test", [])
            if (matches := list(images_dir.glob(f"{Path(name).stem}.*")))
        ]
        has_gt = True

    all_preds: list[dict] = []
    all_gts: list[dict] = []
    ps_stems: set[str] = set()
    all_stems: set[str] = set()

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        stem = img_path.stem
        is_board = img_path.name.startswith("board_")
        all_stems.add(stem)
        if not is_board:
            ps_stems.add(stem)

        if has_gt:
            for obj in load_annotations(labels_dir / f"{stem}.json"):
                cls_name = obj.get("class")
                if cls_name and cls_name != "_negeer":
                    all_gts.append(
                        {
                            "image": stem,
                            "class": cls_name,
                            "poly": rect_to_polygon(
                                tuple(obj["center"]), tuple(obj["size"]), float(obj["angle"])
                            ),
                        }
                    )

        shown_dets: list[dict] = []
        for center, size, angle in detect_candidates(img, is_board=is_board):
            feat, head_angle = extract_features(img, center, size, angle)
            if feat is None or not np.all(np.isfinite(feat)):
                continue

            probs = clf.predict_proba(feat.reshape(1, -1))[0]
            best_i = int(np.argmax(probs))
            pred_cls = str(clf_classes[best_i])
            conf = float(probs[best_i])
            if pred_cls == "_achtergrond" or conf < 0.12:
                continue

            det = {
                "image": stem,
                "filename": img_path.name,
                "class": pred_cls,
                "conf": conf,
                "cx": float(center[0]),
                "cy": float(center[1]),
                "w": float(size[0]),
                "h": float(size[1]),
                "angle": float(angle),
                "head_angle": head_angle if pred_cls in BOLT_CLASSES else None,
                "poly": rect_to_polygon(center, size, angle),
            }
            all_preds.append(det)
            if conf >= min_conf:
                shown_dets.append(det)

        cv2.imwrite(str(vis_dir / f"pred_{img_path.name}"), draw_result(img, shown_dets))

    if has_gt:
        print_metrics("Testset: ps_", ps_stems, all_preds, all_gts, min_conf)
        print_metrics("Testset: ps_ + board_", all_stems, all_preds, all_gts, min_conf)

    csv_path = args.output_csv or (results_dir / "classical_predictions.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "class",
                "confidence",
                "center_x",
                "center_y",
                "width",
                "height",
                "angle",
                "head_angle",
            ]
        )
        for p in all_preds:
            if p["conf"] >= min_conf:
                writer.writerow(
                    [
                        p["filename"],
                        p["class"],
                        f"{p['conf']:.4f}",
                        f"{p['cx']:.1f}",
                        f"{p['cy']:.1f}",
                        f"{p['w']:.1f}",
                        f"{p['h']:.1f}",
                        f"{p['angle']:.1f}",
                        f"{p['head_angle']:.1f}" if p["head_angle"] is not None else "",
                    ]
                )

    print(f"\nCSV opgeslagen in : {csv_path}")
    print(f"Plaatjes in     : {vis_dir}")


if __name__ == "__main__":
    main()