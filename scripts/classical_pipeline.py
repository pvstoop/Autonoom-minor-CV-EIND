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

MIN_AREA = 100.0
MAX_AREA = 45000.0
MAX_IMAGE_SIDE = 1600


def load_classes(classes_path: Path) -> list[str]:
    """Haalt de klassen op uit classes.txt."""
    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_annotations(label_path: Path) -> list[dict]:
    """Leest labels uit een json-bestand (lege lijst als hij er niet is)."""
    if not label_path.exists():
        return []
    data = json.loads(label_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("annotations", "shapes", "objects", "labels"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def rect_to_polygon(
    center: tuple[float, float], size: tuple[float, float], angle: float
) -> Polygon:
    """Maakt van een gedraaide box een Shapely-polygoon voor de IoU."""
    poly = Polygon(cv2.boxPoints((center, size, angle)))
    return poly if poly.is_valid else poly.buffer(0)


def rotated_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    """Berekent de IoU tussen twee gedraaide boxen."""
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    if inter <= 0.0:
        return 0.0
    union = poly_a.union(poly_b).area
    return float(inter / union) if union > 0.0 else 0.0


def resize_for_detection(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Schaalt grote foto's even omlaag zodat het zoeken naar kandidaten sneller gaat."""
    h, w = image_bgr.shape[:2]
    largest = max(h, w)
    if largest <= MAX_IMAGE_SIDE:
        return image_bgr, 1.0
    scale = MAX_IMAGE_SIDE / largest
    resized = cv2.resize(
        image_bgr,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def make_plain_surface_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Maakt een masker voor foto's op een egale ondergrond (ps_)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Als de foto erg donker is, eerst wat extra contrast erin met CLAHE.
    if float(np.mean(lab[:, :, 0])) < 125.0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0].astype(np.uint8)).astype(np.float32)

    blurred = cv2.GaussianBlur(lab, (5, 5), 0)
    # Met een grote blur schat ik grof de achtergrond en trek die eraf.
    background = cv2.GaussianBlur(blurred, (0, 0), sigmaX=55)
    diff = np.linalg.norm(blurred - background, axis=2)

    threshold = max(6.5, float(np.median(diff)) * 2.5)
    mask = (diff > threshold).astype(np.uint8) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    return mask


def make_checkerboard_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Maakt een masker voor foto's op het dambord (board_)."""
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    board_squares = ((gray_full > 190) | (gray_full < 70)).astype(np.uint8) * 255

    # Eerst grof het dambordgebied pakken zodat we daarbinnen blijven.
    board_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    board_area = cv2.morphologyEx(board_squares, cv2.MORPH_CLOSE, board_kernel)
    board_area = cv2.morphologyEx(board_area, cv2.MORPH_OPEN, board_kernel)

    roi = np.zeros_like(gray_full)
    board_cnts, _ = cv2.findContours(board_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if board_cnts:
        largest_board = max(board_cnts, key=cv2.contourArea)
        cv2.drawContours(roi, [largest_board], -1, 255, thickness=cv2.FILLED)
        roi = cv2.erode(roi, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    else:
        roi[:, :] = 255

    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    sat_mask = (hsv[:, :, 1] > 40).astype(np.uint8) * 255
    mid_mask = ((gray >= 72) & (gray <= 185)).astype(np.uint8) * 255
    mid_mask = cv2.erode(mid_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    mask = cv2.bitwise_and(cv2.bitwise_or(sat_mask, mid_mask), roi)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return mask


def split_touching_contour(cnt: np.ndarray, mask_shape: tuple[int, int]) -> list[np.ndarray]:
    """Probeert twee onderdelen die tegen elkaar aan liggen los te trekken."""
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = (area / hull_area) if hull_area > 0 else 1.0

    # Alleen proberen als de contour groot genoeg is en er ingedeukt uitziet.
    if area <= MIN_AREA * 3.0 or solidity >= 0.78:
        return [cnt]

    comp = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(comp, [cnt], -1, 255, thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded = cv2.erode(comp, kernel, iterations=1)
    cores, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cores = [c for c in cores if cv2.contourArea(c) >= 40.0]

    if len(valid_cores) < 2:
        return [cnt]

    parts: list[np.ndarray] = []
    for core in valid_cores:
        core_mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(core_mask, [core], -1, 255, thickness=cv2.FILLED)
        grown = cv2.bitwise_and(cv2.dilate(core_mask, kernel, iterations=1), comp)
        grown_cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for part in grown_cnts:
            if MIN_AREA <= cv2.contourArea(part) <= MAX_AREA:
                parts.append(part)

    return parts if parts else [cnt]


def detect_candidates(
    image_bgr: np.ndarray, is_board: bool
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Zoekt mogelijke onderdelen en geeft gedraaide bounding boxen terug."""
    work_img, scale = resize_for_detection(image_bgr)
    mask = make_checkerboard_mask(work_img) if is_board else make_plain_surface_mask(work_img)

    raw_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours: list[np.ndarray] = []
    for cnt in raw_cnts:
        area = cv2.contourArea(cnt)
        if MIN_AREA <= area <= MAX_AREA:
            contours.extend(split_touching_contour(cnt, mask.shape))

    inv_scale = 1.0 / scale
    boxes: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for cnt in contours:
        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
        if min(w, h) >= 7.0:
            boxes.append(
                (
                    (float(cx * inv_scale), float(cy * inv_scale)),
                    (float(w * inv_scale), float(h * inv_scale)),
                    float(angle),
                )
            )
    return boxes


def crop_aligned_patch(
    image_bgr: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
    pad_scale: float = 1.2,
) -> tuple[np.ndarray | None, float, float, float]:
    """Draait de uitsnede recht zodat het onderdeel horizontaal ligt."""
    w, h = float(size[0]), float(size[1])
    if w <= 2.0 or h <= 2.0:
        return None, 0.0, 0.0, 0.0

    long_side = max(w, h)
    short_side = min(w, h)
    rot_angle = angle if w >= h else angle + 90.0

    h_img, w_img = image_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rot_angle, 1.0)
    rotated = cv2.warpAffine(
        image_bgr,
        matrix,
        (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    crop_w = max(12, int(round(long_side * pad_scale)))
    crop_h = max(12, int(round(short_side * pad_scale)))
    x0 = max(0, int(round(center[0] - crop_w / 2.0)))
    y0 = max(0, int(round(center[1] - crop_h / 2.0)))
    x1 = min(w_img, x0 + crop_w)
    y1 = min(h_img, y0 + crop_h)

    patch = rotated[y0:y1, x0:x1]
    axis_angle = ((rot_angle + 180.0) % 360.0) - 180.0
    return patch, axis_angle, long_side, short_side


def segment_patch(patch_bgr: np.ndarray) -> np.ndarray:
    """Haalt het onderdeel los van de rand van de uitsnede."""
    h, w = patch_bgr.shape[:2]
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Randpixels nemen als schatting voor wat er rondom het onderdeel zit.
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]], axis=0)
    bg_color = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg_color, axis=2).astype(np.uint8)

    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    fg_ratio = float(np.count_nonzero(mask)) / max(1.0, float(w * h))
    # Als Otsu er helemaal naast zit (bijna niks of alles wit), terugvallen op een ellips.
    if fg_ratio < 0.08 or fg_ratio > 0.96:
        fallback = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            fallback,
            (w // 2, h // 2),
            (max(2, int(w * 0.36)), max(2, int(h * 0.36))),
            0,
            0,
            360,
            255,
            -1,
        )
        return fallback
    return mask


def extract_features(
    image_bgr: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
) -> tuple[np.ndarray | None, float]:
    """Berekent 10 kenmerken per onderdeel plus een schatting van de koprichting."""
    patch, axis_angle, long_side, short_side = crop_aligned_patch(image_bgr, center, size, angle)
    if patch is None or patch.size == 0:
        return None, 0.0

    h, w = patch.shape[:2]
    aspect_ratio = long_side / max(1.0, short_side)
    obj_mask = segment_patch(patch)

    # RETR_CCOMP geeft buitenste contouren en eventuele gaten (zoals bij moeren/ringen).
    contours, hierarchy = cv2.findContours(obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    has_hole = 0.0
    hole_area_ratio = 0.0

    if contours and hierarchy is not None:
        hier = hierarchy[0]
        outer_ids = [i for i in range(len(contours)) if hier[i][3] == -1]
        best_id = max(outer_ids, key=lambda i: cv2.contourArea(contours[i]))
        cnt = contours[best_id]
        area = float(cv2.contourArea(cnt))
        perim = float(cv2.arcLength(cnt, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))

        solidity = (area / hull_area) if hull_area > 0 else 1.0
        circularity = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0

        child_id = hier[best_id][2]
        max_hole = 0.0
        while child_id != -1:
            hole_area = float(cv2.contourArea(contours[child_id]))
            if hole_area >= 12.0 and hole_area > max_hole:
                max_hole = hole_area
            child_id = hier[child_id][0]

        if max_hole >= 12.0 and area > 0:
            has_hole = 1.0
            hole_area_ratio = max_hole / area
    else:
        solidity = 1.0
        circularity = 0.5

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    mean_lab, _ = cv2.meanStdDev(lab, mask=obj_mask)
    mean_hsv, _ = cv2.meanStdDev(hsv, mask=obj_mask)

    quarter = max(1, w // 4)
    left_fill = float(np.count_nonzero(obj_mask[:, :quarter])) / max(1.0, float(h * quarter))
    right_fill = float(np.count_nonzero(obj_mask[:, -quarter:])) / max(1.0, float(h * quarter))
    head_to_tail_ratio = max(left_fill, right_fill) / max(0.05, min(left_fill, right_fill))

    head_angle = (axis_angle + 180.0) if left_fill > right_fill else axis_angle
    head_angle = ((head_angle + 180.0) % 360.0) - 180.0

    features = np.array(
        [
            long_side,
            short_side,
            aspect_ratio,
            solidity,
            circularity,
            has_hole,
            hole_area_ratio,
            float(mean_lab[0, 0]),
            float(mean_hsv[1, 0]),
            head_to_tail_ratio,
        ],
        dtype=np.float32,
    )
    return features, head_angle