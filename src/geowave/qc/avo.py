from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.inversion.avo import AVOFit

@dataclass(frozen=True)
class AVOQC:
    fit_count: int
    median_r2: float
    unstable_fraction: float
    extreme_gradient_fraction: float

def avo_qc(fits: list[AVOFit], r2_floor: float=0.7, gradient_limit: float=1.0) -> AVOQC:
    if not fits:
        return AVOQC(0, 0.0, 0.0, 0.0)
    r2 = np.asarray([f.r2 for f in fits], float)
    unstable = np.mean(r2 < r2_floor)
    extreme = np.mean([abs(f.gradient) > gradient_limit for f in fits])
    return AVOQC(len(fits), float(np.median(r2)), float(unstable), float(extreme))
