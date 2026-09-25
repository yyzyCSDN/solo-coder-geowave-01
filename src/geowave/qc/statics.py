from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.signal.statics import StaticShift

@dataclass(frozen=True)
class StaticQC:
    count: int
    median_ms: float
    p95_abs_ms: float
    low_confidence_fraction: float

def static_qc(shifts: list[StaticShift], confidence_floor: float=0.3) -> StaticQC:
    if not shifts:
        return StaticQC(0, 0.0, 0.0, 0.0)
    values = np.asarray([s.shift_s * 1000 for s in shifts], float)
    low = np.mean([s.score < confidence_floor for s in shifts])
    return StaticQC(len(shifts), float(np.median(values)), float(np.percentile(np.abs(values), 95)), float(low))

def suspicious_static_ids(shifts: list[StaticShift], limit_ms: float=40.0, confidence_floor: float=0.2) -> list[str]:
    return [s.trace_id for s in shifts if abs(s.shift_s * 1000) > limit_ms or s.score < confidence_floor]
