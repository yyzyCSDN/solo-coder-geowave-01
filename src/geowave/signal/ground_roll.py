from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt
from geowave.gather.cmp import Gather
from geowave.model.traces import Trace

def estimate_ground_roll_slope(gather: Gather, max_time_s: float=1.0) -> float:
    if len(gather.traces) < 3:
        return 0.0
    picks = []
    for tr in gather.traces:
        n = min(tr.samples.size, int(max_time_s / tr.header.dt_s))
        if n < 2:
            continue
        idx = int(np.argmax(np.abs(tr.samples[:n])))
        picks.append((tr.header.offset_m, tr.header.start_s + idx * tr.header.dt_s))
    if len(picks) < 3:
        return 0.0
    x = np.asarray([p[0] for p in picks], float)
    t = np.asarray([p[1] for p in picks], float)
    A = np.column_stack([x, np.ones_like(x)])
    slope, _ = np.linalg.lstsq(A, t, rcond=None)[0]
    return float(slope)

def attenuate_low_frequency(trace: Trace, cutoff_hz: float=8.0, order: int=4) -> Trace:
    fs = 1.0 / trace.header.dt_s
    if not 0 < cutoff_hz < fs / 2:
        raise ValueError('invalid cutoff')
    sos = butter(order, cutoff_hz / (fs / 2), btype='high', output='sos')
    if trace.samples.size < 32:
        return trace.copy_with(trace.samples, ground_roll='short-trace')
    y = sosfiltfilt(sos, trace.samples)
    return trace.copy_with(y, ground_roll=f'highpass:{cutoff_hz:g}')

def fan_mute_weight(offset_m: float, time_s: np.ndarray, min_velocity: float, max_velocity: float, taper_s: float=0.05) -> np.ndarray:
    if min_velocity <= 0 or max_velocity <= min_velocity:
        raise ValueError('invalid velocity fan')
    t = np.asarray(time_s, float)
    near = abs(offset_m) / max_velocity
    far = abs(offset_m) / min_velocity
    w = np.ones_like(t)
    inside = (t >= near) & (t <= far)
    w[inside] = 0.0
    if taper_s > 0:
        left = (t >= near - taper_s) & (t < near)
        right = (t > far) & (t <= far + taper_s)
        w[left] = 0.5 + 0.5 * np.cos(np.pi * (t[left] - (near - taper_s)) / taper_s)
        w[right] = 0.5 - 0.5 * np.cos(np.pi * (t[right] - far) / taper_s)
    return w
