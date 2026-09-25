from __future__ import annotations
import numpy as np

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float=0.0) -> np.ndarray:
    a = np.asarray(numerator, float)
    b = np.asarray(denominator, float)
    if a.shape != b.shape:
        raise ValueError('shape mismatch')
    out = np.full_like(a, fill, dtype=float)
    np.divide(a, b, out=out, where=np.abs(b) > 1e-12)
    return out

def finite_percentiles(values: np.ndarray, percentiles: list[float]) -> dict[float, float]:
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {float(p): 0.0 for p in percentiles}
    result = np.percentile(a, percentiles)
    return {float(p): float(v) for p, v in zip(percentiles, result)}

def monotonic_segments(values: np.ndarray) -> list[tuple[int, int, int]]:
    x = np.asarray(values, float)
    if x.size < 2:
        return [(0, len(x), 0)] if x.size else []
    sign = np.sign(np.diff(x)).astype(int)
    out = []
    start = 0
    current = int(sign[0])
    for i, value in enumerate(sign[1:], 1):
        value = int(value)
        if value != current:
            out.append((start, i + 1, current))
            start = i
            current = value
    out.append((start, len(x), current))
    return out
