from __future__ import annotations
import numpy as np
from scipy.linalg import toeplitz
from geowave.model.traces import Trace

def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float)
    r = np.correlate(x, x, mode='full')
    mid = x.size - 1
    return r[mid:mid + max_lag + 1] / max(1, x.size)

def spiking_decon(trace: Trace, operator_length: int=32, prewhiten: float=0.01) -> Trace:
    x = trace.samples
    if x.size < operator_length + 2:
        return trace.copy_with(x, decon='skipped')
    r = autocorrelation(x, operator_length)
    r[0] *= 1 + prewhiten
    R = toeplitz(r[:-1])
    d = np.zeros(operator_length)
    d[0] = 1.0
    op = np.linalg.solve(R, d)
    y = np.convolve(x, op, mode='same')
    return trace.copy_with(y, decon=f'spike:{operator_length}')

def predictive_decon(trace: Trace, prediction_lag: int, operator_length: int=32, prewhiten: float=0.01) -> Trace:
    x = trace.samples
    r = autocorrelation(x, operator_length + prediction_lag)
    base = r[:operator_length].copy()
    base[0] *= 1 + prewhiten
    rhs = r[prediction_lag:prediction_lag + operator_length]
    op = np.linalg.solve(toeplitz(base), rhs)
    pred = np.convolve(x, op, mode='same')
    return trace.copy_with(x - pred, decon=f'predictive:{prediction_lag}:{operator_length}')
