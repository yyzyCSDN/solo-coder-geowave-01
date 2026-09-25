from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.horizon import Horizon

@dataclass(frozen=True)
class HorizonStats:
    count: int
    mean_time_s: float
    std_time_s: float
    min_quality: float
    median_quality: float
    roughness: float

def horizon_stats(horizon: Horizon) -> HorizonStats:
    if not horizon.picks:
        return HorizonStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    t = np.asarray([p.time_s for p in horizon.picks], float)
    q = np.asarray([p.quality for p in horizon.picks], float)
    order = np.argsort([p.x for p in horizon.picks])
    ordered = t[order]
    rough = float(np.median(np.abs(np.diff(ordered)))) if len(ordered) > 1 else 0.0
    return HorizonStats(len(t), float(np.mean(t)), float(np.std(t)), float(np.min(q)), float(np.median(q)), rough)

def compare_horizons(a: Horizon, b: Horizon, tolerance_m: float=25.0) -> dict[str, float]:
    diffs = []
    for p in a.picks:
        q = b.nearest(p.x, p.y)
        if q is None:
            continue
        distance = ((q.x - p.x) ** 2 + (q.y - p.y) ** 2) ** 0.5
        if distance <= tolerance_m:
            diffs.append(q.time_s - p.time_s)
    if not diffs:
        return {'matched': 0, 'mean_shift_s': 0.0, 'rms_shift_s': 0.0}
    d = np.asarray(diffs, float)
    return {'matched': len(diffs), 'mean_shift_s': float(np.mean(d)), 'rms_shift_s': float(np.sqrt(np.mean(d * d)))}
