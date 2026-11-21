"""Test een getraind YOLO-model op de testfoto's: scores in beeld, tekeningen in runs/pred/.

    uv run python scripts/evaluate.py runs/yolo/eerste/weights/best.pt
    uv run python scripts/evaluate.py runs/yolo/eerste/weights/best.pt --val   # val-foto's

Kies instellingen op de val-foto's en gebruik de testfoto's alleen voor het eindcijfer.

Scores:
    mAP50    rangschikking over alle zekerheidsdrempels, per klasse gemiddeld
    F1       bij de beste drempel, juiste klasse vereist
    hoekfout mediaan in graden, alleen voor langwerpige onderdelen
    kop goed hoe vaak de kop aan de goede kant zit (als het model de klasse `kop` kent)

Een voorspelling hoort bij een label als de geroteerde boxen minstens 50% overlappen (IoU >= 0.5).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from common import IGNORE, IMAGES, ROOT, SPLIT, draw, iou, load_classes, load_label, load_split
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

SAVE_CONF = 0.01  # lage drempel: mAP heeft ook de onzekere voorspellingen nodig
SURE = 0.5  # vanaf deze zekerheid tellen voorspellingen voor de tekening, hoek en kop
HEAD = "kop"
WITH_HEAD = ("inbusbout", "zeskantbout", "slotbout", "knopschroef", "insteekkoppeling")
LONG = 1.5  # langer dan zoveel keer de breedte = langwerpig, dan heeft de hoek betekenis


def predict(weights: str, paths: list[Path]) -> dict:
    """Voorspel met YOLO-OBB. Geeft per foto een lijst objecten in het formaat van de labels."""
    model = YOLO(weights)
    imgsz = model.ckpt["train_args"]["imgsz"]
    out = {}
    for path in paths:
        result = model.predict(str(path), imgsz=imgsz, conf=SAVE_CONF, verbose=False)[0]
        boxes = result.obb.xywhr.tolist()
        classes = result.obb.cls.tolist()
        scores = result.obb.conf.tolist()
        out[path.stem] = [
            detection(model.names[int(c)], s, cx, cy, w, h, np.degrees(rad))
            for (cx, cy, w, h, rad), c, s in zip(boxes, classes, scores, strict=True)
        ]
    return out


def detection(cls: str, conf, cx, cy, w, h, angle) -> dict:
    """Een voorspelling in hetzelfde formaat als de labels."""
    return {"class": cls, "center": [float(cx), float(cy)], "size": [float(w), float(h)],
            "angle": float(angle), "confidence": float(conf)}  # fmt: skip


# ---------- kop ----------


def ends(obj: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """De twee uiteinden van de lange as van een box, en de richting van die as in graden."""
    width, height = obj["size"]
    angle = obj["angle"] if width >= height else obj["angle"] + 90
    axis = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    half = max(width, height) / 2
    center = np.array(obj["center"])
    return center - axis * half, center + axis * half, angle


def add_heads(objects: list[dict], heads: list[dict]) -> None:
    """Zet bij elk onderdeel met een kop in de buurt aan welke kant die kop zit.

    Per onderdeel kijken we welk uiteinde het dichtst bij een gevonden kop ligt. Ligt er binnen
    1 breedte geen kop, dan blijft het onderdeel zonder kop. Daarna geldt hetzelfde als in de
    labels: size = [lengte, breedte], angle wijst van de kop naar de punt, head is het uiteinde.
    """
    centers = [np.array(h["center"]) for h in heads if h["confidence"] >= SURE]
    if not centers:
        return
    for obj in objects:
        if obj["class"] not in WITH_HEAD:  # moeren, ringen en koppelbussen hebben geen kop
            continue
        start, end, angle = ends(obj)
        width = min(obj["size"])
        to_start = min(np.linalg.norm(start - c) for c in centers)
        to_end = min(np.linalg.norm(end - c) for c in centers)
        if min(to_start, to_end) > width:
            continue
        head_at_start = to_start <= to_end
        obj["size"] = [max(obj["size"]), width]
        obj["angle"] = float(angle % 360 if head_at_start else (angle + 180) % 360)
        obj["head"] = (start if head_at_start else end).tolist()


# ---------- scores ----------


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Oppervlak onder de precision-recall-curve."""
    rec = np.concatenate([[0.0], recall, [1.0]])
    pre = np.concatenate([[0.0], precision, [0.0]])
    pre = np.maximum.accumulate(pre[::-1])[::-1]
    steps = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[steps + 1] - rec[steps]) * pre[steps + 1]))


def class_ap(preds: dict, gts: dict, cls: str) -> float | None:
    """AP50 van een klasse. None als die klasse niet in de labels zit."""
    n_gt = sum(g["class"] == cls for objs in gts.values() for g in objs)
    if n_gt == 0:
        return None
    candidates = [(d["confidence"], s, d) for s in gts for d in preds[s] if d["class"] == cls]
    candidates.sort(key=lambda t: -t[0])  # zekerste eerst
    if not candidates:
        return 0.0
    used = {s: set() for s in gts}
    hits = np.zeros(len(candidates))
    for k, (_, stem, d) in enumerate(candidates):
        best, match = 0.5, None
        for j, g in enumerate(gts[stem]):
            if g["class"] == cls and j not in used[stem]:
                overlap = iou(d, g)
                if overlap >= best:
                    best, match = overlap, j
        if match is not None:
            used[stem].add(match)
            hits[k] = 1
    found = np.cumsum(hits)
    return average_precision(found / n_gt, found / np.arange(1, len(candidates) + 1))


def best_f1(preds: dict, gts: dict) -> tuple[float, float]:
    """Beste F1 over alle drempels. Geeft (F1, drempel)."""
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
            for r, c in zip(rows, cols, strict=True):
                tp += m[r, c] >= 0.5 and kept[r]["class"] == objs[c]["class"]
        f1 = 2 * tp / max(n_pred + n_gt, 1)
        best = max(best, (f1, float(conf)))
    return best


def matched(preds: dict, gts: dict) -> list[tuple[dict, dict]]:
    """Per label de best overlappende zekere voorspelling, als die minstens 50% overlapt."""
    pairs = []
    for stem, objs in gts.items():
        sure = [d for d in preds[stem] if d["confidence"] >= SURE]
        for g in objs:
            if not sure:
                continue
            best = max(sure, key=lambda d: iou(d, g))
            if iou(best, g) >= 0.5:
                pairs.append((best, g))
    return pairs


def angle_error(pred: dict, label: dict) -> float:
    """Verschil tussen de lange assen in graden (0-90)."""
    diff = abs(ends(pred)[2] - ends(label)[2]) % 180
    return min(diff, 180 - diff)


def report(name: str, preds: dict, gts: dict, classes: list[str]) -> None:
    """Print mAP50, F1, hoekfout en kop voor deze foto's."""
    aps = {c: class_ap(preds, gts, c) for c in classes}
    scored = {c: v for c, v in aps.items() if v is not None}
    f1, conf = best_f1(preds, gts)
    pairs = matched(preds, gts)
    long_pairs = [(p, g) for p, g in pairs if max(g["size"]) >= LONG * min(g["size"])]
    with_head = [(p, g) for p, g in pairs if "head" in p and "head" in g]

    n = sum(len(v) for v in gts.values())
    print(f"\n{name}: {len(gts)} foto's, {n} objecten")
    mean_ap = 100 * np.mean(list(scored.values()))
    print(f"  mAP50 {mean_ap:.1f}   F1 {100 * f1:.1f} (drempel {conf:.2f})")
    print("  " + "  ".join(f"{c} {100 * v:.0f}" for c, v in scored.items()))
    if long_pairs:
        errors = [angle_error(p, g) for p, g in long_pairs]
        print(f"  hoekfout: mediaan {np.median(errors):.1f} graden ({len(errors)} onderdelen)")
    if with_head:
        good = [abs((p["angle"] - g["angle"] + 180) % 360 - 180) < 90 for p, g in with_head]
        print(f"  kop goed: {sum(good)} van {len(good)} ({100 * np.mean(good):.1f}%)")


def main() -> None:
    """Voorspel de test- of val-foto's, bewaar tekeningen en voorspellingen, en print de scores."""
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="een .pt-bestand van YOLO")
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument(
        "--val", action="store_true", help="test op de val-foto's in plaats van de testfoto's"
    )
    args = ap.parse_args()
    part = "val" if args.val else "test"

    classes = load_classes()
    stems = load_split(args.split)[part]
    preds = predict(args.weights, [IMAGES / f"{s}.jpg" for s in stems])

    weights = Path(args.weights)
    # runs/yolo/<naam>/weights/best.pt -> <naam>, models/yolo11m_obb.pt -> yolo11m_obb
    name = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    out = ROOT / "runs/pred" / (f"{name}_val" if args.val else name)
    out.mkdir(parents=True, exist_ok=True)
    gts = {}
    for stem in stems:
        labels = load_label(stem)["objects"]
        ignored = [o for o in labels if o["class"] == IGNORE]
        gts[stem] = [o for o in labels if o["class"] != IGNORE]
        heads = [d for d in preds[stem] if d["class"] == HEAD]
        # voorspellingen op een `_negeer`-gebied tellen niet als fout
        parts = [d for d in preds[stem] if d["class"] != HEAD]
        preds[stem] = [d for d in parts if all(iou(d, o) < 0.3 for o in ignored)]
        add_heads(preds[stem], heads)

        (out / f"{stem}.json").write_text(json.dumps({"objects": preds[stem]}, indent=1))
        shown = [d for d in preds[stem] if d["confidence"] >= SURE]
        drawing = draw(cv2.imread(str(IMAGES / f"{stem}.jpg")), shown, classes)
        cv2.imwrite(str(out / f"{stem}.jpg"), drawing)

    for title, prefix in (("Egale ondergrond", "ps_"), ("Dambord", "board_")):
        subset = [s for s in stems if s.startswith(prefix)]
        if subset:
            report(title, {s: preds[s] for s in subset}, {s: gts[s] for s in subset}, classes)
    print(f"\nTekeningen -> {out}")


if __name__ == "__main__":
    main()
