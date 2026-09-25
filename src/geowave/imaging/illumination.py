from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class IlluminationMap:
    x_m: np.ndarray
    z_m: np.ndarray
    hit_count: np.ndarray
    angle_sum: np.ndarray

def straight_ray_illumination(dataset: TraceSet, x_m: np.ndarray, z_m: np.ndarray, max_depth_m: float | None=None) -> IlluminationMap:
    x = np.asarray(x_m, float)
    z = np.asarray(z_m, float)
    hits = np.zeros((len(z), len(x)), dtype=float)
    angles = np.zeros_like(hits)
    zmax = max_depth_m if max_depth_m is not None else float(z.max()) if len(z) else 0.0
    for tr in dataset.traces:
        midpoint = tr.header.midpoint[0]
        half = tr.header.offset_m / 2
        for iz, depth in enumerate(z):
            if depth > zmax:
                continue
            spread = max(depth, 1.0)
            left = midpoint - half * depth / (depth + spread)
            right = midpoint + half * depth / (depth + spread)
            ix0 = int(np.argmin(np.abs(x - left)))
            ix1 = int(np.argmin(np.abs(x - right)))
            lo, hi = sorted((ix0, ix1))
            hits[iz, lo:hi + 1] += 1
            angle = np.degrees(np.arctan2(half, max(depth, 1.0)))
            angles[iz, lo:hi + 1] += angle
    return IlluminationMap(x, z, hits, angles)

def mean_angle(illumination: IlluminationMap) -> np.ndarray:
    return np.divide(illumination.angle_sum, illumination.hit_count, out=np.zeros_like(illumination.angle_sum), where=illumination.hit_count > 0)
