"""Test het getrainde model op de testset of een eigen map en bewaart CSV's in results/klassiek/."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

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
    """Berekent Average Precision bij IoU >= 0.50 voor één klasse."""
    total_gt = sum(len(polys) for polys in gts_by_img.values())
    if total_gt == 0 or not preds:
        return 0.0

    ordered = sorted(preds, key=lambda p: p["conf"], reverse=True)
    matched: dict[str, set[int]] = {img_id: set() for img_id in gts_by_img}
    tp, fp = np.zeros(len(ordered)), np.zeros(len(ordered))

    for i, pred in enumerate(ordered):
        best_iou, best_gt = 0.0, -1
        for g_idx, g_poly in enumerate(gts_by_img.get(pred["image"], [])):
            if g_idx not in matched[pred["image"]]:
                iou = rotated_iou(pred["poly"], g_poly)
                if iou > best_iou:
                    best_iou, best_gt = iou, g_idx
        if best_iou >= 0.50 and best_gt >= 0:
            matched[pred["image"]].add(best_gt)
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    rec = np.cumsum(tp) / total_gt
    prec = np.cumsum(tp) / np.maximum(1.0, np.cumsum(tp) + np.cumsum(fp))
    return float(np.mean([prec[rec >= r].max() if np.any(rec >= r) else 0.0 for r in np.linspace(0, 1, 101)]))


def eval_subset(title: str, stems: set[str], preds: list[dict], gts: list[dict], min_conf: float) -> dict:
    """Berekent en print mAP@50, Precision, Recall, F1 en boutkop-score voor een set foto's."""
    sub_preds = [p for p in preds if p["image"] in stems]
    sub_gts = [g for g in gts if g["image"] in stems]
    aps, tp_tot, fp_tot, kop_ok, kop_tot = [], 0, 0, 0, 0

    for cls_name in CLASSES:
        cls_preds = [p for p in sub_preds if p["class"] == cls_name]
        gts_by_img: dict[str, list[Polygon]] = {s: [] for s in stems}
        angles_by_img: dict[str, list[float]] = {s: [] for s in stems}
        for gt in sub_gts:
            if gt["class"] == cls_name:
                gts_by_img[gt["image"]].append(gt["poly"])
                angles_by_img[gt["image"]].append(gt["angle"])
        aps.append(ap50_for_class(cls_preds, gts_by_img))

        matched: dict[str, set[int]] = {s: set() for s in stems}
        for pred in sorted([p for p in cls_preds if p["conf"] >= min_conf], key=lambda x: x["conf"], reverse=True):
            best_iou, best_idx = 0.0, -1
            for idx, g_poly in enumerate(gts_by_img[pred["image"]]):
                if idx not in matched[pred["image"]]:
                    iou = rotated_iou(pred["poly"], g_poly)
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx
            if best_iou >= 0.50 and best_idx >= 0:
                matched[pred["image"]].add(best_idx)
                tp_tot += 1
                if cls_name in BOLT_CLASSES and pred["head_angle"] is not None:
                    diff = abs(((pred["head_angle"] - angles_by_img[pred["image"]][best_idx] + 180.0) % 360.0) - 180.0)
                    kop_tot += 1
                    kop_ok += int(diff < 90.0)
            else:
                fp_tot += 1

    gt_tot = len(sub_gts)
    prec = (tp_tot / (tp_tot + fp_tot) * 100.0) if (tp_tot + fp_tot) else 0.0
    rec = (tp_tot / gt_tot * 100.0) if gt_tot else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    kop_pct = (kop_ok / kop_tot * 100.0) if kop_tot else 0.0

    print(f"\n=== {title} ({len(stems)} foto's, {gt_tot} GT) ===")
    print(f"  mAP@50    : {np.mean(aps) * 100.0:5.1f}%")
    print(f"  Precision : {prec:5.1f}%  (TP={tp_tot}, FP={fp_tot})")
    print(f"  Recall    : {rec:5.1f}%  ({tp_tot}/{gt_tot})")
    print(f"  F1-score  : {f1:5.1f}%")
    print(f"  Boutkop   : {kop_pct:5.1f}%  ({kop_ok}/{kop_tot})")
    return {
        "mAP50": round(float(np.mean(aps)) * 100.0, 1),
        "F1": round(f1, 1),
        "precision": round(prec, 1),
        "recall": round(rec, 1),
        "boutkop_goed": f"{kop_ok}/{kop_tot}",
        "boutkop_pct": round(kop_pct, 1),
    }


def draw_result(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Tekent groene boxen, labels en een rode pijl naar de boutkop."""
    canvas = image_bgr.copy()
    for det in detections:
        box = np.int32(cv2.boxPoints(((det["cx"], det["cy"]), (det["w"], det["h"]), det["angle"])))
        cv2.drawContours(canvas, [box], 0, (0, 220, 0), 3)
        cv2.putText(
            canvas, f"{det['class']} {det['conf']:.2f}",
            (int(det["cx"]) - 40, int(det["cy"]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
        )
        if det["head_angle"] is not None:
            rad = np.radians(det["head_angle"])
            length = max(det["w"], det["h"]) * 0.45
            end_pt = (int(round(det["cx"] + np.cos(rad) * length)), int(round(det["cy"] + np.sin(rad) * length)))
            cv2.arrowedLine(canvas, (int(round(det["cx"])), int(round(det["cy"]))), end_pt, (0, 0, 255), 3, tipLength=0.25)
    return canvas


def main() -> None:
    """Draait de evaluatie op de testset of een eigen map en bewaart CSV en plaatjes."""
    root = Path(__file__).resolve().parent.parent
    images_dir, labels_dir = root / "data" / "images", root / "data" / "labels"
    klassiek_dir, vis_dir = root / "results" / "klassiek", root / "runs" / "classical_eval"
    klassiek_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(root / "models" / "classical_classifier.joblib")
    clf, min_conf = bundle["model"], float(bundle.get("min_conf", 0.35))
    clf_classes = list(clf.classes_)

    custom_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if custom_dir is not None:
        image_paths = sorted(p for p in custom_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        has_gt = False
    else:
        splits = yaml.safe_load((root / "data" / "split.yaml").read_text(encoding="utf-8"))
        image_paths = [matches[0] for name in splits["test"] if (matches := list(images_dir.glob(f"{Path(name).stem}.*")))]
        has_gt = True

    all_preds, all_gts, ps_stems, all_stems = [], [], set(), set()
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        stem, is_board = img_path.stem, img_path.name.startswith("board_")
        all_stems.add(stem)
        if not is_board:
            ps_stems.add(stem)

        if has_gt:
            for obj in load_annotations(labels_dir / f"{stem}.json"):
                if obj.get("class") and obj["class"] != "_negeer":
                    all_gts.append({
                        "image": stem, "class": obj["class"], "angle": float(obj.get("angle", 0.0)),
                        "poly": rect_to_polygon(tuple(obj["center"]), tuple(obj["size"]), float(obj["angle"])),
                    })

        shown = []
        for center, size, angle in detect_candidates(img, is_board=is_board):
            feat, head_angle = extract_features(img, center, size, angle)
            if feat is None or not np.all(np.isfinite(feat)):
                continue
            probs = clf.predict_proba(feat.reshape(1, -1))[0]
            best_i = int(np.argmax(probs))
            pred_cls, conf = str(clf_classes[best_i]), float(probs[best_i])
            if pred_cls == "_achtergrond":
                continue
            det = {
                "image": stem, "filename": img_path.name, "class": pred_cls, "conf": conf,
                "cx": center[0], "cy": center[1], "w": size[0], "h": size[1], "angle": angle,
                "head_angle": head_angle if pred_cls in BOLT_CLASSES else None,
                "poly": rect_to_polygon(center, size, angle),
            }
            all_preds.append(det)
            if conf >= min_conf:
                shown.append(det)
        cv2.imwrite(str(vis_dir / f"pred_{img_path.name}"), draw_result(img, shown))

    if has_gt:
        m_ps = eval_subset("Testset: ps_ (egale ondergrond)", ps_stems, all_preds, all_gts, min_conf)
        m_board = eval_subset("Testset: board_ (dambord)", all_stems - ps_stems, all_preds, all_gts, min_conf)
        m_tot = eval_subset("Testset: ps_ + board_ (totaal)", all_stems, all_preds, all_gts, min_conf)

        with (klassiek_dir / "varianten_vergelijking.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "beschrijving", "mAP50", "F1", "precision", "recall", "boutkop_goed", "boutkop_pct", "mAP50_egaal", "F1_egaal", "mAP50_dambord", "F1_dambord"])
            writer.writerow(["basis_rf", "Random Forest (11 kenmerken, close 5x5, containment)", m_tot["mAP50"], m_tot["F1"], m_tot["precision"], m_tot["recall"], m_tot["boutkop_goed"], m_tot["boutkop_pct"], m_ps["mAP50"], m_ps["F1"], m_board["mAP50"], m_board["F1"]])

        with (klassiek_dir / "kernresultaten.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["methode", "mAP50", "F1", "F1_egaal", "F1_dambord"])
            writer.writerow(["Random Forest (11 kenmerken, close 5x5, containment)", m_tot["mAP50"], m_tot["F1"], m_ps["F1"], m_board["F1"]])

    with (klassiek_dir / "voorspellingen.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["foto", "klasse", "zekerheid", "center_x", "center_y", "breedte", "hoogte", "hoek", "boutkop_hoek"])
        for p in all_preds:
            if p["conf"] >= min_conf:
                writer.writerow([p["filename"], p["class"], f"{p['conf']:.4f}", f"{p['cx']:.1f}", f"{p['cy']:.1f}", f"{p['w']:.1f}", f"{p['h']:.1f}", f"{p['angle']:.1f}", f"{p['head_angle']:.1f}" if p["head_angle"] is not None else ""])

    print(f"Klaar! Resultaten staan in: {klassiek_dir} en {vis_dir}")


if __name__ == "__main__":
    main()