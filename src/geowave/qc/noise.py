from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace

@dataclass(frozen=True)
class WindowSNR:
    signal_rms: float
    noise_rms: float
    snr_db: float

def snr(trace: Trace, signal_window: tuple[float, float], noise_window: tuple[float, float]) -> WindowSNR:

    def grab(w):
        i0 = max(0, int((w[0] - trace.header.start_s) / trace.header.dt_s))
        i1 = min(trace.samples.size, int((w[1] - trace.header.start_s) / trace.header.dt_s))
        return trace.samples[i0:i1]
    s = grab(signal_window)
    n = grab(noise_window)
    sr = float(np.sqrt(np.mean(s * s))) if s.size else 0
    nr = float(np.sqrt(np.mean(n * n))) if n.size else 0
    db = 20 * np.log10(sr / nr) if nr > 0 and sr > 0 else float('-inf')
    return WindowSNR(sr, nr, float(db))

def spike_indices(trace: Trace, z: float=8.0) -> list[int]:
    x = trace.samples
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-12
    score = 0.6745 * np.abs(x - med) / mad
    return np.where(score > z)[0].astype(int).tolist()
