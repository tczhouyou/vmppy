"""Minimal quaternion math for vmppy.

Quaternion convention in this package:
- quaternions are numpy arrays shaped (..., 4)
- order is (x, y, z, w)  (matches how tvmp code expects them)
"""

from __future__ import annotations

import numpy as np


def _normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, eps, None)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    out = q.copy()
    out[..., :3] *= -1
    return out


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product. Inputs/outputs are (x,y,z,w)."""
    x1, y1, z1, w1 = np.moveaxis(np.asarray(q1), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(q2), -1, 0)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=-1)


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions."""
    q0 = _normalize(q0)
    q1 = _normalize(q1)

    dot = np.sum(q0 * q1, axis=-1)
    # take shortest path
    if np.any(dot < 0):
        q1 = np.where(dot[..., None] < 0, -q1, q1)
        dot = np.abs(dot)

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        # nearly linear
        out = q0 + t * (q1 - q0)
        return _normalize(out)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1
