from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PickConsistency:
    count: int
    median_jump_samples: float
    p95_jump_samples: float
    outlier_indices: tuple[int, ...]

def pick_consistency(samples: list[int], jump_limit: int=8) -> PickConsistency:
    if len(samples) < 2:
        return PickConsistency(len(samples), 0.0, 0.0, ())
    jumps = np.abs(np.diff(samples))
    outliers = tuple((np.where(jumps > jump_limit)[0] + 1).astype(int).tolist())
    return PickConsistency(len(samples), float(np.median(jumps)), float(np.percentile(jumps, 95)), outliers)

def smooth_picks(samples: list[int], radius: int=2) -> list[int]:
    x = np.asarray(samples, float)
    out = []
    for i in range(len(x)):
        a = max(0, i - radius)
        b = min(len(x), i + radius + 1)
        out.append(int(round(np.median(x[a:b]))))
    return out
