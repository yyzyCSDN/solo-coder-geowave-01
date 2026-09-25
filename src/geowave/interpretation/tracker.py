from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TrackPoint:
    trace_index: int
    sample_index: int
    score: float

def track_peak(section: np.ndarray, seed_trace: int, seed_sample: int, search_radius: int=8, polarity: int=1) -> list[TrackPoint]:
    a = np.asarray(section, float)
    if a.ndim != 2:
        raise ValueError('section must be traces x samples')
    if not 0 <= seed_trace < a.shape[0] or not 0 <= seed_sample < a.shape[1]:
        raise ValueError('seed outside section')
    result = {seed_trace: TrackPoint(seed_trace, seed_sample, abs(float(a[seed_trace, seed_sample])))}
    for direction in (-1, 1):
        current = seed_sample
        i = seed_trace + direction
        while 0 <= i < a.shape[0]:
            lo = max(0, current - search_radius)
            hi = min(a.shape[1], current + search_radius + 1)
            window = polarity * a[i, lo:hi]
            if not window.size:
                break
            local = int(np.argmax(window))
            current = lo + local
            score = float(max(window[local], 0.0))
            result[i] = TrackPoint(i, current, score)
            i += direction
    return [result[i] for i in sorted(result)]

def track_continuity(points: list[TrackPoint]) -> float:
    if len(points) < 2:
        return 1.0
    jumps = np.diff([p.sample_index for p in points])
    return float(1.0 / (1.0 + np.median(np.abs(jumps))))
