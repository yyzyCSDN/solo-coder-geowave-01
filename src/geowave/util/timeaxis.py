from __future__ import annotations
import numpy as np

def sample_index(time_s: float, start_s: float, dt_s: float, mode: str='nearest') -> int:
    if dt_s <= 0:
        raise ValueError('dt must be positive')
    value = (time_s - start_s) / dt_s
    if mode == 'nearest':
        return int(round(value))
    if mode == 'floor':
        return int(np.floor(value))
    if mode == 'ceil':
        return int(np.ceil(value))
    raise ValueError('unknown rounding mode')

def time_axis(count: int, start_s: float, dt_s: float) -> np.ndarray:
    if count < 0 or dt_s <= 0:
        raise ValueError('invalid time axis')
    return start_s + np.arange(count) * dt_s

def common_time_window(starts: list[float], counts: list[int], dts: list[float]) -> tuple[float, float] | None:
    if not starts:
        return None
    if len(set((round(x, 12) for x in dts))) != 1:
        raise ValueError('mixed sample intervals')
    ends = [s + max(0, n - 1) * dt for s, n, dt in zip(starts, counts, dts)]
    lo = max(starts)
    hi = min(ends)
    return (lo, hi) if hi >= lo else None
