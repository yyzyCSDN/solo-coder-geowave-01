from __future__ import annotations
import numpy as np
from geowave.model.traces import Trace

def real_cepstrum(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    x = trace.samples
    if not x.size:
        return (np.array([]), np.array([]))
    z = np.fft.rfft(x)
    log_mag = np.log(np.maximum(np.abs(z), 1e-12))
    c = np.fft.irfft(log_mag, n=x.size)
    q = np.arange(x.size) * trace.header.dt_s
    return (q, c)

def periodicity_seconds(trace: Trace, min_period_s: float, max_period_s: float) -> float | None:
    q, c = real_cepstrum(trace)
    if not q.size:
        return None
    mask = (q >= min_period_s) & (q <= max_period_s)
    if not np.any(mask):
        return None
    idx = np.where(mask)[0][int(np.argmax(c[mask]))]
    return float(q[idx])

def minimum_phase_reconstruction(trace: Trace) -> np.ndarray:
    x = trace.samples
    if not x.size:
        return x.copy()
    n = x.size
    log_mag = np.log(np.maximum(np.abs(np.fft.fft(x)), 1e-12))
    cep = np.fft.ifft(log_mag).real
    causal = np.zeros_like(cep)
    causal[0] = cep[0]
    causal[1:(n + 1) // 2] = 2 * cep[1:(n + 1) // 2]
    if n % 2 == 0:
        causal[n // 2] = cep[n // 2]
    return np.fft.ifft(np.exp(np.fft.fft(causal))).real
