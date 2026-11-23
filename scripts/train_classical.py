"""Traint een Random Forest en SVM op de kenmerken uit classical_pipeline.py."""

from __future__ import annotations

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
    rect_to_polygon,
    rotated_iou,
)


def find_image(images_dir: Path, name: str) -> Path | None:
    matches = list(images_dir.glob(f"{Path(name).stem}.*"))
    return matches[0] if matches else None


def collect_samples(
    image_names: list[str], images_dir: Path, labels_dir: Path, max_bg: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Haalt de 11 kenmerken op voor gelabelde onderdelen en achtergrondvoorbeelden."""
    x_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    bg_rows: list[np.ndarray] = []

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
        gt_polys = [
            rect_to_polygon(tuple(obj["center"]), tuple(obj["size"]), float(obj["angle"]))
            for obj in annots
        ]

        # Echte labels meenemen als positieve voorbeelden
        for obj in annots:
            feat, _ = extract_features(
                img, tuple(obj["center"]), tuple(obj["size"]), float(obj["angle"])
            )
            if feat is not None and np.all(np.isfinite(feat)):
                x_rows.append(feat)
                y_rows.append(str(obj["class"]))

        # Losse kandidaten zonder match met ground truth als achtergrond gebruiken
        is_board = img_path.name.startswith("board_")
        for center, size, angle in detect_candidates(img, is_board=is_board):
            cand_poly = rect_to_polygon(center, size, angle)
            max_iou = max((rotated_iou(cand_poly, gt) for gt in gt_polys), default=0.0)
            if max_iou < 0.10:
                feat, _ = extract_features(img, center, size, angle)
                if feat is not None and np.all(np.isfinite(feat)):
                    bg_rows.append(feat)

    # Achtergrond beperken zodat het niet de overhand krijgt
    rng = np.random.default_rng(42)
    if len(bg_rows) > max_bg:
        idx = rng.choice(len(bg_rows), size=max_bg, replace=False)
        bg_rows = [bg_rows[i] for i in idx]
    for feat in bg_rows:
        x_rows.append(feat)
        y_rows.append("_achtergrond")

    return np.vstack(x_rows), np.array(y_rows)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    images_dir = root / "data" / "images"
    labels_dir = root / "data" / "labels"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    splits = yaml.safe_load((root / "data" / "split.yaml").read_text(encoding="utf-8"))
    x_train, y_train = collect_samples(splits["train"], images_dir, labels_dir, max_bg=200)
    x_val, y_val = collect_samples(splits["val"], images_dir, labels_dir, max_bg=40)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=16, class_weight="balanced", random_state=42
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(C=5.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        ),
    }

    best_name, best_f1, best_model, best_preds = "", -1.0, None, None
    for name, clf in models.items():
        clf.fit(x_train, y_train)
        preds = clf.predict(x_val)
        acc = accuracy_score(y_val, preds) * 100.0
        macro_f1 = f1_score(y_val, preds, labels=CLASSES, average="macro", zero_division=0) * 100.0
        print(f"=== {name} === Accuracy: {acc:.1f}% | Macro-F1: {macro_f1:.1f}%")
        if macro_f1 > best_f1:
            best_f1, best_name, best_model, best_preds = macro_f1, name, clf, preds

    print(f"\nBeste model: {best_name}")
    print(classification_report(y_val, best_preds, digits=3, zero_division=0))

    out_path = models_dir / "classical_classifier.joblib"
    joblib.dump({"model": best_model, "min_conf": 0.35}, out_path)
    print(f"Opgeslagen in: {out_path}")


if __name__ == "__main__":
    main()