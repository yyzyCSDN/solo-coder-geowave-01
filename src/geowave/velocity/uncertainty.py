from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.velocity import VelocityFunction, VelocityPick

@dataclass(frozen=True)
class VelocityEnvelope:
    times_s: np.ndarray
    low_mps: np.ndarray
    median_mps: np.ndarray
    high_mps: np.ndarray

def bootstrap_velocity(func: VelocityFunction, sigma_fraction: float=0.03, samples: int=500, seed: int=0) -> VelocityEnvelope:
    rng = np.random.default_rng(seed)
    times = np.asarray([p.time_s for p in func.picks], float)
    base = np.asarray([p.velocity_mps for p in func.picks], float)
    confidence = np.asarray([max(p.confidence, 0.05) for p in func.picks], float)
    spread = sigma_fraction * base / np.sqrt(confidence)
    draws = rng.normal(base, spread, size=(samples, len(base)))
    draws = np.maximum(draws, 300.0)
    low, med, high = np.percentile(draws, [5, 50, 95], axis=0)
    return VelocityEnvelope(times, low, med, high)

def perturb_function(func: VelocityFunction, envelope: VelocityEnvelope, quantile: str) -> VelocityFunction:
    values = {'low': envelope.low_mps, 'median': envelope.median_mps, 'high': envelope.high_mps}.get(quantile)
    if values is None:
        raise ValueError('quantile must be low, median or high')
    picks = [VelocityPick(float(t), float(v), p.confidence) for t, v, p in zip(envelope.times_s, values, func.picks)]
    return VelocityFunction(func.x, func.y, picks)
