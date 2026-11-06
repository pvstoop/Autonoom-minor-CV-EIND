# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Oriented-box (OBB) geometry for the partsorter OBB head.

An oriented rectangle with center ``c`` and half-side vectors ``a`` (along one side) and ``b`` (along the other) is
represented by its shape matrix ``M = a aᵀ + b bᵀ`` (2x2, symmetric positive definite). This representation:

* is invariant to the corner order, to sign flips of ``a``/``b`` and to swapping them, so there is no angle
  wrap-around and no width/height ambiguity (the problem that makes plain angle regression hard);
* transforms exactly under any linear map ``S`` as ``S M Sᵀ``. RF-DETR squashes every image to a square, which turns
  rotated rectangles into parallelograms; ``M`` of the parallelogram is exactly ``S M Sᵀ`` of the rectangle, so the
  head can learn in normalized network coordinates and decoding back to pixels is exact.

``N(c, M / 3)`` is the Gaussian with the covariance of a uniform rectangle, which is what YOLO-OBB's ProbIoU uses.
The Bhattacharyya distance between two Gaussians is affine invariant, so the loss is the same in any frame.

The head predicts ``M`` through its Cholesky factor relative to the (detached) axis-aligned box of the same query::

    L = diag(w/2, h/2) @ [[exp(s1), 0], [t, exp(s2)]],    M = L Lᵀ

With zero-initialized outputs this is exactly the axis-aligned box, a sensible starting point.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_LOG_CLAMP = (-7.0, 3.0)


def raw_to_matrix(raw: Tensor, boxes: Tensor) -> Tensor:
    """Head output ``(..., 3)`` + axis-aligned boxes ``(..., 4)`` cxcywh (normalized) -> ``M`` ``(..., 2, 2)``."""
    s1, t, s2 = raw.float().unbind(-1)
    hw, hh = boxes[..., 2].float() / 2, boxes[..., 3].float() / 2
    l11 = hw * s1.clamp(*_LOG_CLAMP).exp()
    l21 = hh * t.clamp(-50.0, 50.0)
    l22 = hh * s2.clamp(*_LOG_CLAMP).exp()
    m11 = l11 * l11
    m12 = l11 * l21
    m22 = l21 * l21 + l22 * l22
    return torch.stack([torch.stack([m11, m12], -1), torch.stack([m12, m22], -1)], -2)


def corners_to_matrix(corners: Tensor) -> tuple[Tensor, Tensor]:
    """Consecutive corners ``(..., 4, 2)`` of a rectangle (or its affine image) -> center ``(..., 2)``, ``M``."""
    corners = corners.float()
    center = corners.mean(-2)
    a = (corners[..., 1, :] - corners[..., 0, :]) / 2
    b = (corners[..., 3, :] - corners[..., 0, :]) / 2
    m = a.unsqueeze(-1) * a.unsqueeze(-2) + b.unsqueeze(-1) * b.unsqueeze(-2)
    return center, m


def probiou_loss(c1: Tensor, m1: Tensor, c2: Tensor, m2: Tensor, eps: float = 1e-30) -> Tensor:
    """Hellinger distance ``sqrt(1 - exp(-Bd))`` between ``N(c1, m1/3)`` and ``N(c2, m2/3)`` (0 = identical).

    Computed in float64: in normalized coordinates the determinants of thin parts are ~1e-10.
    """
    dtype = c1.dtype if c1.dtype in (torch.float32, torch.float64) else torch.float32
    c1, c2 = c1.double(), c2.double()
    s1, s2 = m1.double() / 3, m2.double() / 3
    s = (s1 + s2) / 2
    a, b, d = s[..., 0, 0], s[..., 0, 1], s[..., 1, 1]
    det = (a * d - b * b).clamp_min(eps)
    dx, dy = (c1 - c2).unbind(-1)
    maha = (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
    det1 = (s1[..., 0, 0] * s1[..., 1, 1] - s1[..., 0, 1] ** 2).clamp_min(eps)
    det2 = (s2[..., 0, 0] * s2[..., 1, 1] - s2[..., 0, 1] ** 2).clamp_min(eps)
    bd = maha / 8 + 0.5 * torch.log(det / (det1 * det2).sqrt())
    bd = bd.clamp(0.0, 100.0)
    return (1.0 - torch.exp(-bd) + 1e-12).sqrt().to(dtype)


def matrix_to_obb(center: Tensor, m: Tensor) -> Tensor:
    """Center ``(N, 2)`` + shape matrix ``(N, 2, 2)``, both in pixels -> ``(N, 5)`` = cx, cy, w, h, angle (degrees).

    ``w`` lies along ``angle`` (OpenCV ``minAreaRect`` convention: angle of the first side), ``w >= h``.
    """
    # closed-form eigen-decomposition of a symmetric 2x2 matrix (torch.linalg.eigh costs ms per call on CUDA)
    m = m.double()
    a, b, d = m[:, 0, 0], m[:, 0, 1], m[:, 1, 1]
    mean, r = (a + d) / 2, torch.sqrt(((a - d) / 2) ** 2 + b**2)
    w = 2 * (mean + r).clamp_min(0).sqrt()
    h = 2 * (mean - r).clamp_min(0).sqrt()
    angle = 0.5 * torch.atan2(2 * b, a - d) * (180.0 / math.pi)
    return torch.stack([center[:, 0].double(), center[:, 1].double(), w, h, angle], -1).float()
