from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class SurfaceOperator:
    key: str
    log_spectrum: np.ndarray

def average_log_spectra(dataset: TraceSet, nfft: int | None=None) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not dataset.traces:
        return (np.array([]), {}, {})
    nfft = nfft or max((t.samples.size for t in dataset.traces))
    shot: dict[str, list[np.ndarray]] = {}
    receiver: dict[str, list[np.ndarray]] = {}
    all_specs = []
    for tr in dataset.traces:
        z = np.fft.rfft(tr.samples, n=nfft)
        log = np.log(np.maximum(np.abs(z), 1e-10))
        all_specs.append(log)
        shot.setdefault(tr.header.shot_id, []).append(log)
        receiver.setdefault(tr.header.receiver_id, []).append(log)
    global_spec = np.mean(all_specs, axis=0)
    shot_mean = {k: np.mean(v, axis=0) for k, v in shot.items()}
    rec_mean = {k: np.mean(v, axis=0) for k, v in receiver.items()}
    return (global_spec, shot_mean, rec_mean)

def estimate_surface_operators(dataset: TraceSet, nfft: int | None=None) -> tuple[dict[str, SurfaceOperator], dict[str, SurfaceOperator]]:
    global_spec, shot, rec = average_log_spectra(dataset, nfft)
    shot_ops = {k: SurfaceOperator(k, global_spec - value) for k, value in shot.items()}
    rec_ops = {k: SurfaceOperator(k, global_spec - value) for k, value in rec.items()}
    return (shot_ops, rec_ops)

def apply_spectral_operator(samples: np.ndarray, log_correction: np.ndarray) -> np.ndarray:
    n = len(samples)
    z = np.fft.rfft(samples)
    correction = np.exp(log_correction[:len(z)])
    return np.fft.irfft(z * correction, n=n)
