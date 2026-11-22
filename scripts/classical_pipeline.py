"""Klassieke CV-pipeline met OpenCV voor het herkennen van onderdelen en hun orientatie."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

CLASSES: list[str] = [
    "inbusbout",
    "insteekkoppeling",
    "knopschroef",
    "koppelbus",
    "moer",
    "ring",
    "slotbout",
    "zeskantbout",
]
BOLT_CLASSES: set[str] = {"inbusbout", "knopschroef", "slotbout", "zeskantbout"}

MIN_AREA = 250.0
MAX_AREA = 120000.0


def load_classes(classes_path: Path) -> list[str]:
    """Haalt de klassen op uit classes.txt."""
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_annotations(label_path: Path) -> list[dict]:
    """Leest labels uit een json-bestand."""
    if not label_path.exists():
        return []
    data = json.loads(label_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("annotations", "shapes", "objects", "labels"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    return []


def rect_to_polygon(center: tuple[float, float], size: tuple[float, float], angle: float) -> Polygon:
    """Maakt van een gedraaide box een polygoon voor de IoU-berekening."""
    poly = Polygon(cv2.boxPoints((center, size, angle)))
    return poly if poly.is_valid else poly.buffer(0)


def rotated_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    """Berekent hoeveel twee gedraaide boxen overlappen (0.0 tot 1.0)."""
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return float(inter / union) if inter > 0.0 and union > 0.0 else 0.0


def make_mask(image_bgr: np.ndarray, is_board: bool, close_size: int = 5) -> np.ndarray:
    """Maakt een binair masker: wit (255) = mogelijk onderdeel, zwart (0) = achtergrond."""
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    if not is_board:
        # Egale ondergrond: adaptive threshold past zich aan schaduw en licht aan.
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=81, C=8
        )
    else:
        # Dambord: pak gekleurde onderdelen (HSV saturatie) en zilveren middentonen.
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        colored = (hsv[:, :, 1] > 45).astype(np.uint8) * 255
        silver = ((gray >= 75) & (gray <= 185)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(colored, silver)

        # Negeer de buitenste 12% van de foto zodat de doosrand en tape wegvallen.
        h, w = mask.shape
        margin_y, margin_x = int(h * 0.12), int(w * 0.12)
        roi = np.zeros_like(mask)
        roi[margin_y : h - margin_y, margin_x : w - margin_x] = 255
        mask = cv2.bitwise_and(mask, roi)

    # Morfologie: eerst kleine glansgaatjes dichten (close), daarna ruis weghalen (open).
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def detect_candidates(
    image_bgr: np.ndarray,
    is_board: bool = False,
    use_containment: bool = True,
    close_size: int = 5,
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Zoekt contouren in het masker en geeft gedraaide bounding boxen terug."""
    mask = make_mask(image_bgr, is_board, close_size=close_size)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
        short_side, long_side = min(w, h), max(w, h)
        if short_side < 10.0 or (long_side / short_side) > 30.0:
            continue

        boxes.append(((float(cx), float(cy)), (float(w), float(h)), float(angle)))

    if not use_containment:
        return boxes

    # Verwijder dubbele/overlappende boxen op hetzelfde onderdeel (containment + NMS).
    boxes.sort(key=lambda b: b[1][0] * b[1][1], reverse=True)
    kept: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    kept_polys: list[Polygon] = []
    for box in boxes:
        poly = rect_to_polygon(box[0], box[1], box[2])
        is_inside = any(cv2.pointPolygonTest(cv2.boxPoints(big).reshape(-1, 1, 2), box[0], False) >= 0 for big in kept)
        overlaps = any(rotated_iou(poly, big_p) > 0.25 for big_p in kept_polys)
        if is_inside or overlaps:
            continue
        kept.append(box)
        kept_polys.append(poly)
    return kept


def crop_aligned_patch(
    image_bgr: np.ndarray, center: tuple[float, float], size: tuple[float, float], angle: float
) -> tuple[np.ndarray | None, float, float, float]:
    """Draait de uitsnede recht zodat het onderdeel horizontaal ligt."""
    w, h = float(size[0]), float(size[1])
    if w <= 2.0 or h <= 2.0:
        return None, 0.0, 0.0, 0.0

    long_side, short_side = max(w, h), min(w, h)
    rot_angle = angle if w >= h else angle + 90.0

    h_img, w_img = image_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rot_angle, 1.0)
    rotated = cv2.warpAffine(image_bgr, matrix, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

    crop_w = max(12, int(round(long_side * 1.15)))
    crop_h = max(12, int(round(short_side * 1.15)))
    x0 = max(0, int(round(center[0] - crop_w / 2.0)))
    y0 = max(0, int(round(center[1] - crop_h / 2.0)))
    patch = rotated[y0 : min(h_img, y0 + crop_h), x0 : min(w_img, x0 + crop_w)]

    axis_angle = ((rot_angle + 180.0) % 360.0) - 180.0
    return patch, axis_angle, long_side, short_side


def extract_features(
    image_bgr: np.ndarray, center: tuple[float, float], size: tuple[float, float], angle: float
) -> tuple[np.ndarray | None, float]:
    """Berekent 11 simpele vorm- en kleurkenmerken plus de richting van de kop."""
    patch, axis_angle, long_side, short_side = crop_aligned_patch(image_bgr, center, size, angle)
    if patch is None or patch.size == 0:
        return None, 0.0

    # Otsu-threshold op grijswaarden om het onderdeel in de uitsnede te scheiden.
    gray = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, obj_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Zoek buitencontour en eventueel gat (RETR_CCOMP geeft buitenrand + binnengaten).
    contours, hierarchy = cv2.findContours(obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    solidity, circularity, num_corners, has_hole, hole_ratio = 1.0, 0.5, 0.0, 0.0, 0.0

    if contours and hierarchy is not None:
        hier = hierarchy[0]
        outer = [i for i in range(len(contours)) if hier[i][3] == -1]
        best = max(outer, key=lambda i: cv2.contourArea(contours[i]))
        area = float(cv2.contourArea(contours[best]))
        perim = float(cv2.arcLength(contours[best], True))
        hull = float(cv2.contourArea(cv2.convexHull(contours[best])))

        solidity = (area / hull) if hull > 0 else 1.0
        circularity = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0

        # Tel het aantal hoekpunten van de buitencontour (~6 bij moer, meer bij ring).
        approx = cv2.approxPolyDP(contours[best], 0.03 * perim, True)
        num_corners = float(len(approx))

        child = hier[best][2]
        while child != -1:
            hole_area = float(cv2.contourArea(contours[child]))
            if hole_area >= 12.0 and area > 0:
                has_hole = 1.0
                hole_ratio = max(hole_ratio, hole_area / area)
            child = hier[child][0]

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(gray))
    saturation = float(np.mean(hsv[:, :, 1]))

    # Vergelijk linker- en rechterkwart van de uitsnede: de kop van een bout is breder.
    h, w = obj_mask.shape
    q = max(1, w // 4)
    left_fill = float(np.count_nonzero(obj_mask[:, :q])) / max(1.0, float(h * q))
    right_fill = float(np.count_nonzero(obj_mask[:, -q:])) / max(1.0, float(h * q))
    head_ratio = max(left_fill, right_fill) / max(0.05, min(left_fill, right_fill))

    head_angle = (axis_angle + 180.0) if left_fill > right_fill else axis_angle
    head_angle = ((head_angle + 180.0) % 360.0) - 180.0

    features = np.array(
        [
            long_side,
            short_side,
            long_side / max(1.0, short_side),
            solidity,
            circularity,
            num_corners,
            has_hole,
            hole_ratio,
            brightness,
            saturation,
            head_ratio,
        ],
        dtype=np.float32,
    )
    return features, head_angle