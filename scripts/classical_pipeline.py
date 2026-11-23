"""Klassieke CV-stappen en kenmerken voor bouten, moeren en ringen."""

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
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_annotations(label_path: Path) -> list[dict]:
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
    poly = Polygon(cv2.boxPoints((center, size, angle)))
    return poly if poly.is_valid else poly.buffer(0)


def rotated_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return float(inter / union) if inter > 0.0 and union > 0.0 else 0.0


def make_mask(image_bgr: np.ndarray, is_board: bool, close_size: int = 5) -> np.ndarray:
    # Beetje blur tegen ruis
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    if not is_board:
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=81, C=8
        )
    else:
        # Dambord: kleur + grijze metalen delen pakken
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        colored = (hsv[:, :, 1] > 45).astype(np.uint8) * 255
        silver = ((gray >= 75) & (gray <= 185)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(colored, silver)

        # Rand van het dambord eraf snijden
        h, w = mask.shape
        my, mx = int(h * 0.12), int(w * 0.12)
        roi = np.zeros_like(mask)
        roi[my : h - my, mx : w - mx] = 255
        mask = cv2.bitwise_and(mask, roi)

    # Gaatjes dichten en kleine ruis weg zonder dun draad te slopen
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def detect_candidates(
    image_bgr: np.ndarray,
    is_board: bool = False,
    use_containment: bool = True,
    close_size: int = 5,
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
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

    # Grote boxen eerst en overlappende/dubbele boxen opruimen
    boxes.sort(key=lambda b: b[1][0] * b[1][1], reverse=True)
    kept: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    kept_polys: list[Polygon] = []
    for box in boxes:
        poly = rect_to_polygon(box[0], box[1], box[2])
        if any(rotated_iou(poly, p) > 0.15 for p in kept_polys):
            continue
        kept.append(box)
        kept_polys.append(poly)
    return kept


def get_aligned_patch(
    image_bgr: np.ndarray, center: tuple[float, float], size: tuple[float, float], angle: float
) -> tuple[np.ndarray | None, float, float, float]:
    w, h = float(size[0]), float(size[1])
    if w <= 2.0 or h <= 2.0:
        return None, 0.0, 0.0, 0.0

    long_side, short_side = max(w, h), min(w, h)
    rot_angle = angle if w >= h else angle + 90.0

    # Rechtzetten langs de lange as zodat links/rechts klopt met kop en draadeind
    h_img, w_img = image_bgr.shape[:2]
    mat = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rot_angle, 1.0)
    rot_img = cv2.warpAffine(image_bgr, mat, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

    half_w = max(8, int(round(long_side * 0.55)))
    half_h = max(8, int(round(short_side * 0.60)))
    cx_i, cy_i = int(round(center[0])), int(round(center[1]))
    patch = rot_img[max(0, cy_i - half_h) : min(h_img, cy_i + half_h), max(0, cx_i - half_w) : min(w_img, cx_i + half_w)]

    axis_angle = ((rot_angle + 180.0) % 360.0) - 180.0
    return patch, axis_angle, long_side, short_side


def extract_features(
    image_bgr: np.ndarray, center: tuple[float, float], size: tuple[float, float], angle: float
) -> tuple[np.ndarray | None, float]:
    patch, axis_angle, long_side, short_side = get_aligned_patch(image_bgr, center, size, angle)
    if patch is None or patch.size == 0:
        return None, 0.0

    gray = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, obj_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Buitenrand en eventuele gaten bepalen
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
        approx = cv2.approxPolyDP(contours[best], 0.02 * perim, True)
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

    # Dikste kant links/rechts gebruiken als schatting voor de boutkop
    h, w = obj_mask.shape
    q = max(1, w // 4)
    left_fill = float(np.count_nonzero(obj_mask[:, :q])) / max(1.0, float(h * q))
    right_fill = float(np.count_nonzero(obj_mask[:, -q:])) / max(1.0, float(h * q))
    head_ratio = max(left_fill, right_fill) / max(0.05, min(left_fill, right_fill))

    raw_angle = (axis_angle + 180.0) if left_fill > right_fill else axis_angle
    head_angle = ((raw_angle + 180.0) % 360.0) - 180.0

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