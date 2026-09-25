from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from geowave.model.traces import Trace

def bandpass(trace: Trace, low_hz: float, high_hz: float, order: int=4) -> Trace:
    fs = 1.0 / trace.header.dt_s
    ny = fs / 2
    if not 0 < low_hz < high_hz < ny:
        raise ValueError('invalid bandpass')
    sos = butter(order, [low_hz / ny, high_hz / ny], btype='bandpass', output='sos')
    if trace.samples.size < max(24, order * 6):
        return trace.copy_with(trace.samples, filter='bandpass-short')
    return trace.copy_with(sosfiltfilt(sos, trace.samples), filter=f'bandpass:{low_hz:g}-{high_hz:g}')

def lowpass(trace: Trace, cut_hz: float, order: int=4) -> Trace:
    fs = 1 / trace.header.dt_s
    ny = fs / 2
    if not 0 < cut_hz < ny:
        raise ValueError('invalid cutoff')
    sos = butter(order, cut_hz / ny, btype='low', output='sos')
    return trace.copy_with(sosfiltfilt(sos, trace.samples), filter=f'lowpass:{cut_hz:g}')

def notch(trace: Trace, freq_hz: float, q: float=30.0) -> Trace:
    fs = 1 / trace.header.dt_s
    if not 0 < freq_hz < fs / 2:
        raise ValueError('invalid notch')
    b, a = iirnotch(freq_hz, q, fs=fs)
    return trace.copy_with(filtfilt(b, a, trace.samples), filter=f'notch:{freq_hz:g}')
