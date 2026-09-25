from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.velocity import VelocityFunction

@dataclass(frozen=True)
class PickQuality:
    pick_count: int
    low_confidence_count: int
    reversal_count: int
    max_acceleration_mps_per_s: float

def velocity_pick_quality(func: VelocityFunction, confidence_floor: float=0.4) -> PickQuality:
    picks = func.picks
    low = sum((p.confidence < confidence_floor for p in picks))
    reversals = 0
    acceleration = []
    for a, b in zip(picks, picks[1:]):
        if b.velocity_mps < a.velocity_mps:
            reversals += 1
        dt = b.time_s - a.time_s
        if dt > 0:
            acceleration.append(abs(b.velocity_mps - a.velocity_mps) / dt)
    return PickQuality(len(picks), low, reversals, max(acceleration, default=0.0))

def confidence_weighted_rms(func: VelocityFunction) -> float:
    if not func.picks:
        return 0.0
    v = np.asarray([p.velocity_mps for p in func.picks], float)
    w = np.asarray([p.confidence for p in func.picks], float)
    return float(np.sqrt(np.sum(w * v * v) / max(np.sum(w), 1e-12)))
