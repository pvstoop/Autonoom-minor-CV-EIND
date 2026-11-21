"""Traint een Random Forest en SVM op de kenmerken en bewaart de beste."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import yaml

from classical_pipeline import (
    CLASSES,
    detect_candidates,
    extract_features,
    load_annotations,
    load_classes,
    rect_to_polygon,
    rotated_iou,
)


def find_image(images_dir: Path, name: str) -> Path | None:
    """Zoekt het plaatje op in data/images."""
    direct = images_dir / name
    if direct.exists():
        return direct
    matches = list(images_dir.glob(f"{Path(name).stem}.*"))
    return matches[0] if matches else None


def show_dataset_counts(
    splits: dict[str, list[str]], labels_dir: Path, class_names: list[str]
) -> None:
    """Laat zien hoeveel foto's en labels er per split in de dataset zitten."""
    for split_name, names in splits.items():
        counts: Counter[str] = Counter()
        ignored = 0
        for name in names:
            for obj in load_annotations(labels_dir / f"{Path(name).stem}.json"):
                cls = obj.get("class", "")
                if cls == "_negeer":
                    ignored += 1
                elif cls:
                    counts[cls] += 1

        ps_count = sum(1 for n in names if n.startswith("ps_"))
        board_count = sum(1 for n in names if n.startswith("board_"))
        print(
            f"\nSplit '{split_name}': {len(names)} foto's "
            f"({ps_count} ps_, {board_count} board_)"
        )
        for cls in class_names:
            print(f"  {cls:<18}: {counts[cls]:>3}")
        print(f"  {'_negeer':<18}: {ignored:>3}")


def collect_samples(
    image_names: list[str],
    images_dir: Path,
    labels_dir: Path,
    max_negatives: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Verzamelt features en labels uit echte labels, kandidaten en achtergrond."""
    x_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    negatives: list[np.ndarray] = []

    for name in image_names:
        img_path = find_image(images_dir, name)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        annots = [
            obj
            for obj in load_annotations(labels_dir / f"{img_path.stem}.json")
            if obj.get("class") and obj.get("class") != "_negeer"
        ]
        gt_items = [
            (
                str(obj["class"]),
                rect_to_polygon(tuple(obj["center"]), tuple(obj["size"]), float(obj["angle"])),
            )
            for obj in annots
        ]

        for obj in annots:
            feat, _ = extract_features(
                img, tuple(obj["center"]), tuple(obj["size"]), float(obj["angle"])
            )
            if feat is not None and np.all(np.isfinite(feat)):
                x_rows.append(feat)
                y_rows.append(str(obj["class"]))

        is_board = img_path.name.startswith("board_")
        for center, size, angle in detect_candidates(img, is_board=is_board):
            cand_poly = rect_to_polygon(center, size, angle)
            best_iou = max(
                (rotated_iou(cand_poly, gt_poly) for _, gt_poly in gt_items),
                default=0.0,
            )
            best_cls = max(
                gt_items,
                key=lambda item: rotated_iou(cand_poly, item[1]),
                default=("", None),
            )[0]

            feat, _ = extract_features(img, center, size, angle)
            if feat is None or not np.all(np.isfinite(feat)):
                continue

            if best_iou >= 0.50 and best_cls:
                x_rows.append(feat)
                y_rows.append(best_cls)
            elif best_iou < 0.12:
                negatives.append(feat)

    if not x_rows:
        raise RuntimeError("Geen features gevonden, kloppen de paden en labels?")

    if negatives:
        rng = np.random.default_rng(42)
        if len(negatives) > max_negatives:
            idx = rng.choice(len(negatives), size=max_negatives, replace=False)
            negatives = [negatives[i] for i in idx]
        for neg in negatives:
            x_rows.append(neg)
            y_rows.append("_achtergrond")

    return np.vstack(x_rows), np.array(y_rows)


def pick_best_threshold(
    clf: object, val_names: list[str], images_dir: Path, labels_dir: Path
) -> float:
    """Kijkt op de val-set (ps_) welke confidence-grens de hoogste F1 geeft."""
    clf_classes = list(clf.classes_)
    preds: list[tuple[str, str, float, object]] = []
    gts: list[tuple[str, str, object]] = []

    for name in val_names:
        if name.startswith("board_"):
            continue
        img_path = find_image(images_dir, name)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for obj in load_annotations(labels_dir / f"{img_path.stem}.json"):
            cls_name = obj.get("class")
            if cls_name and cls_name != "_negeer":
                gts.append(
                    (
                        img_path.stem,
                        str(cls_name),
                        rect_to_polygon(
                            tuple(obj["center"]), tuple(obj["size"]), float(obj["angle"])
                        ),
                    )
                )

        for center, size, angle in detect_candidates(img, is_board=False):
            feat, _ = extract_features(img, center, size, angle)
            if feat is None or not np.all(np.isfinite(feat)):
                continue
            probs = clf.predict_proba(feat.reshape(1, -1))[0]
            best_i = int(np.argmax(probs))
            pred_cls = str(clf_classes[best_i])
            if pred_cls != "_achtergrond":
                preds.append(
                    (
                        img_path.stem,
                        pred_cls,
                        float(probs[best_i]),
                        rect_to_polygon(center, size, angle),
                    )
                )

    if not gts or not preds:
        return 0.25

    best_thr, best_f1 = 0.25, -1.0
    for thr in np.arange(0.20, 0.36, 0.05):
        tp, fp = 0, 0
        matched: set[int] = set()
        for stem, p_cls, conf, p_poly in sorted(preds, key=lambda x: x[2], reverse=True):
            if conf < thr:
                continue
            best_iou, best_gt = 0.0, -1
            for g_idx, (g_stem, g_cls, g_poly) in enumerate(gts):
                if g_idx in matched or g_stem != stem or g_cls != p_cls:
                    continue
                iou = rotated_iou(p_poly, g_poly)
                if iou > best_iou:
                    best_iou, best_gt = iou, g_idx
            if best_iou >= 0.50 and best_gt >= 0:
                matched.add(best_gt)
                tp += 1
            else:
                fp += 1

        prec = tp / max(1, tp + fp)
        rec = tp / len(gts)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    return round(best_thr, 2)


def main() -> None:
    """Traint de modellen, kiest de beste op macro-F1 en slaat hem op."""
    root = Path(__file__).resolve().parent.parent
    images_dir = root / "data" / "images"
    labels_dir = root / "data" / "labels"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(root / "classes.txt")
    splits = yaml.safe_load((root / "data" / "split.yaml").read_text(encoding="utf-8"))
    show_dataset_counts(splits, labels_dir, class_names)

    print("\nFeatures ophalen voor train en val...")
    x_train, y_train = collect_samples(
        splits.get("train", []), images_dir, labels_dir, max_negatives=250
    )
    x_val, y_val = collect_samples(
        splits.get("val", []), images_dir, labels_dir, max_negatives=40
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(
                C=5.0,
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
        ),
    }

    best_name, best_f1, best_model, best_preds = "", -1.0, None, None
    for name, clf in models.items():
        clf.fit(x_train, y_train)
        preds = clf.predict(x_val)
        acc = accuracy_score(y_val, preds) * 100.0
        macro_f1 = (
            f1_score(y_val, preds, labels=CLASSES, average="macro", zero_division=0) * 100.0
        )
        print(f"\n=== {name} ===")
        print(f"  Accuracy = {acc:5.1f}% | Macro-F1 = {macro_f1:5.1f}%")
        if macro_f1 > best_f1:
            best_f1, best_name, best_model, best_preds = macro_f1, name, clf, preds

    print(f"\n=== Beste model: {best_name} ===")
    print(classification_report(y_val, best_preds, digits=3, zero_division=0))

    min_conf = pick_best_threshold(best_model, splits.get("val", []), images_dir, labels_dir)
    out_path = models_dir / "classical_classifier.joblib"
    joblib.dump({"model": best_model, "min_conf": min_conf}, out_path)
    print(f"Grens: {min_conf:.2f} | Model opgeslagen in: {out_path}")


if __name__ == "__main__":
    main()