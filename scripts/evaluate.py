"""Test een getraind model op de testfoto's: scores in beeld, tekeningen en voorspellingen in runs/pred/.

    uv run python scripts/evaluate.py runs/yolo/eerste/weights/best.pt
    uv run python scripts/evaluate.py runs/rfdetr/eerste/checkpoint_final.pth

Scores: mAP50 (rangschikking over alle drempels) en F1 (bij de beste drempel, juiste klasse vereist).
Een box telt als goed bij IoU >= 0.5 met een geroteerde labelbox.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from common import IGNORE, IMAGES, ROOT, SPLIT, draw, load_classes, load_label, load_split, to_corners
from scipy.optimize import linear_sum_assignment

SAVE_CONF = 0.01  # lage drempel: mAP heeft ook de onzekere voorspellingen nodig
DRAW_CONF = 0.25


def predict_yolo(weights: str, paths: list[Path], classes: list[str], imgsz: int) -> dict:
    from ultralytics import YOLO

    model, out = YOLO(weights), {}
    for p in paths:
        r = model.predict(str(p), imgsz=imgsz, conf=SAVE_CONF, verbose=False)[0]
        out[p.stem] = [
            detection(classes[int(c)], s, cx, cy, w, h, np.degrees(rad))
            for (cx, cy, w, h, rad), c, s in zip(
                r.obb.xywhr.tolist(), r.obb.cls.tolist(), r.obb.conf.tolist(), strict=True
            )
        ]
    return out


def predict_rfdetr(weights: str, paths: list[Path], classes: list[str], size: str, resolution: int) -> dict:
    import rfdetr
    from PIL import Image
    from train_rfdetr import SIZES

    model = getattr(rfdetr, SIZES[size])(pretrain_weights=weights, resolution=resolution, obb_head=True)
    out = {}
    for p in paths:
        sv = model.predict(Image.open(p).convert("RGB"), threshold=SAVE_CONF)
        out[p.stem] = [
            detection(classes[int(c)], s, *box)
            for c, s, box in zip(sv.class_id, sv.confidence, sv.data["obb"], strict=True)
            if c < len(classes)  # RF-DETR heeft een extra achtergrondklasse
        ]
    return out


def detection(cls: str, conf, cx, cy, w, h, angle) -> dict:
    return {"class": cls, "shape": "obb", "center": [float(cx), float(cy)], "size": [float(w), float(h)],
            "angle": float(angle), "confidence": float(conf)}  # fmt: skip


def iou(a: dict, b: dict) -> float:
    area_a, area_b = a["size"][0] * a["size"][1], b["size"][0] * b["size"][1]
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(to_corners(a), to_corners(b))
    return max(float(inter), 0.0) / (area_a + area_b - max(float(inter), 0.0) + 1e-9)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    rec = np.concatenate([[0.0], recall, [1.0]])
    pre = np.concatenate([[0.0], precision, [0.0]])
    pre = np.maximum.accumulate(pre[::-1])[::-1]
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * pre[idx + 1]))


def class_ap(preds: dict, gts: dict, cls: str) -> float | None:
    n_gt = sum(g["class"] == cls for objs in gts.values() for g in objs)
    if n_gt == 0:
        return None
    cand = sorted(
        ((d["confidence"], s, d) for s in gts for d in preds[s] if d["class"] == cls), key=lambda t: -t[0]
    )
    if not cand:
        return 0.0
    used = {s: set() for s in gts}
    hits = np.zeros(len(cand))
    for k, (_, stem, d) in enumerate(cand):
        best, match = 0.5, None
        for j, g in enumerate(gts[stem]):
            if g["class"] == cls and j not in used[stem]:
                v = iou(d, g)
                if v >= best:
                    best, match = v, j
        if match is not None:
            used[stem].add(match)
            hits[k] = 1
    tp = np.cumsum(hits)
    return average_precision(tp / n_gt, tp / np.arange(1, len(cand) + 1))


def best_f1(preds: dict, gts: dict) -> tuple[float, float]:
    best = (0.0, 0.05)
    n_gt = sum(len(objs) for objs in gts.values())
    for conf in np.arange(0.05, 0.96, 0.05):
        tp = n_pred = 0
        for stem, objs in gts.items():
            kept = [d for d in preds[stem] if d["confidence"] >= conf]
            n_pred += len(kept)
            if not kept or not objs:
                continue
            m = np.array([[iou(d, g) for g in objs] for d in kept])
            rows, cols = linear_sum_assignment(np.where(m >= 0.5, -m, 1.0))
            tp += sum(m[r, c] >= 0.5 and kept[r]["class"] == objs[c]["class"]
                      for r, c in zip(rows, cols, strict=True))  # fmt: skip
        f1 = 2 * tp / max(n_pred + n_gt, 1)
        best = max(best, (f1, float(conf)))
    return best


def report(name: str, stems: list[str], preds: dict, gts: dict, classes: list[str]) -> None:
    p, g = {s: preds[s] for s in stems}, {s: gts[s] for s in stems}
    aps = {c: class_ap(p, g, c) for c in classes}
    scored = {c: v for c, v in aps.items() if v is not None}
    f1, conf = best_f1(p, g)
    n = sum(len(v) for v in g.values())
    print(f"\n{name}: {len(stems)} foto's, {n} objecten")
    print(f"  mAP50 {100 * np.mean(list(scored.values())):.1f}   F1 {100 * f1:.1f} (drempel {conf:.2f})")
    print("  " + "  ".join(f"{c} {100 * v:.0f}" for c, v in scored.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help=".pt = YOLO, .pth = RF-DETR")
    ap.add_argument("--name", help="uitvoermap onder runs/pred (standaard afgeleid uit het modelpad)")
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--imgsz", type=int, default=1024, help="YOLO")
    ap.add_argument("--size", default="small", help="RF-DETR: nano, small of medium")
    ap.add_argument("--resolution", type=int, default=768, help="RF-DETR")
    args = ap.parse_args()

    classes = load_classes()
    stems = load_split(args.split)["test"]
    paths = [IMAGES / f"{s}.jpg" for s in stems]
    if args.weights.endswith(".pt"):
        preds = predict_yolo(args.weights, paths, classes, args.imgsz)
    else:
        preds = predict_rfdetr(args.weights, paths, classes, args.size, args.resolution)

    gts = {}
    for stem in stems:
        objects = load_label(stem)["objects"]
        gts[stem] = [o for o in objects if o["class"] != IGNORE]
        ignored = [o for o in objects if o["class"] == IGNORE]
        # voorspellingen op een `_negeer`-gebied tellen niet als fout
        preds[stem] = [d for d in preds[stem] if all(iou(d, o) < 0.3 for o in ignored)]

    parts = Path(args.weights).parts
    name = args.name or ("_".join(parts[-4:-2]) if "weights" in parts else "_".join(parts[-3:-1]))
    out = ROOT / "runs/pred" / name
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (out / f"{stem}.json").write_text(json.dumps({"objects": preds[stem]}, indent=1))
        shown = [d for d in preds[stem] if d["confidence"] >= DRAW_CONF]
        cv2.imwrite(str(out / f"{stem}.jpg"), draw(cv2.imread(str(IMAGES / f"{stem}.jpg")), shown, classes))

    report("Alle testfoto's", stems, preds, gts, classes)
    for label, prefix in (("Eigen foto's", "ps_"), ("Dambordfoto's", "board_")):
        subset = [s for s in stems if s.startswith(prefix)]
        if subset:
            report(label, subset, preds, gts, classes)
    print(f"\nTekeningen -> {out}")


if __name__ == "__main__":
    main()
