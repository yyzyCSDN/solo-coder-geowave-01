from __future__ import annotations
import math
from typing import Iterable
import numpy as np

def rms(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0

def robust_median(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    return float(np.median(a)) if a.size else 0.0

def mad(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return 0.0
    m = np.median(a)
    return float(np.median(np.abs(a - m)))

def percentile(values: Iterable[float], q: float) -> float:
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return 0.0
    return float(np.percentile(a, q))

def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def lerp(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return (y0 + y1) / 2.0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pairwise_dist(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(x1 - x0, y1 - y0)
