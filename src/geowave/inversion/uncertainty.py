from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EnsembleSummary:
    mean: np.ndarray
    std: np.ndarray
    p05: np.ndarray
    p50: np.ndarray
    p95: np.ndarray

def summarize_ensemble(models: np.ndarray) -> EnsembleSummary:
    a = np.asarray(models, float)
    if a.ndim < 2 or a.shape[0] < 2:
        raise ValueError('ensemble must contain two or more models')
    p05, p50, p95 = np.percentile(a, [5, 50, 95], axis=0)
    return EnsembleSummary(np.mean(a, axis=0), np.std(a, axis=0), p05, p50, p95)

def normalized_uncertainty(summary: EnsembleSummary, floor: float=1e-09) -> np.ndarray:
    return summary.std / np.maximum(np.abs(summary.mean), floor)

def probability_above(models: np.ndarray, threshold: float) -> np.ndarray:
    a = np.asarray(models, float)
    return np.mean(a > threshold, axis=0)

def rank_uncertain_cells(summary: EnsembleSummary, limit: int=20) -> list[tuple[tuple[int, ...], float]]:
    score = normalized_uncertainty(summary)
    flat = np.argsort(score.ravel())[::-1][:limit]
    out = []
    for index in flat:
        coords = tuple((int(x) for x in np.unravel_index(index, score.shape)))
        out.append((coords, float(score[coords])))
    return out
