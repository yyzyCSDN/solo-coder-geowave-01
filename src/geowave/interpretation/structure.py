from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StructuralStats:
    mean_dip_deg: float
    p95_dip_deg: float
    curvature_rms: float
    closure_count: int

def structural_stats(surface: np.ndarray, dx_m: float, dy_m: float) -> StructuralStats:
    z = np.asarray(surface, float)
    gy, gx = np.gradient(z, dy_m, dx_m)
    dip = np.degrees(np.arctan(np.hypot(gx, gy)))
    gxx = np.gradient(gx, dx_m, axis=1)
    gyy = np.gradient(gy, dy_m, axis=0)
    curvature = gxx + gyy
    closure = local_closures(z)
    return StructuralStats(float(np.nanmean(dip)), float(np.nanpercentile(dip, 95)), float(np.sqrt(np.nanmean(curvature * curvature))), len(closure))

def local_closures(surface: np.ndarray, radius: int=1) -> list[tuple[int, int, float]]:
    z = np.asarray(surface, float)
    out = []
    for i in range(radius, z.shape[0] - radius):
        for j in range(radius, z.shape[1] - radius):
            patch = z[i - radius:i + radius + 1, j - radius:j + radius + 1]
            center = z[i, j]
            if np.isfinite(center) and center <= np.nanmin(patch):
                relief = float(np.nanmean(patch) - center)
                if relief > 0:
                    out.append((i, j, relief))
    return out
