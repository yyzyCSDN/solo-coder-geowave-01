from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import sobel

@dataclass(frozen=True)
class MigrationQC:
    focus_score: float
    edge_energy: float
    mean_amplitude: float
    polarity_balance: float

def migration_qc(image: np.ndarray) -> MigrationQC:
    a = np.asarray(image, float)
    if a.ndim != 2 or not a.size:
        return MigrationQC(0.0, 0.0, 0.0, 0.0)
    gx = sobel(a, axis=1)
    gz = sobel(a, axis=0)
    edge = np.hypot(gx, gz)
    focus = float(np.percentile(np.abs(a), 99) / (np.mean(np.abs(a)) + 1e-12))
    pos = float(np.sum(a[a > 0]))
    neg = float(-np.sum(a[a < 0]))
    balance = min(pos, neg) / (max(pos, neg) + 1e-12)
    return MigrationQC(focus, float(np.mean(edge)), float(np.mean(np.abs(a))), balance)

def compare_migrations(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = np.asarray(reference, float)
    b = np.asarray(candidate, float)
    if a.shape != b.shape:
        raise ValueError('image shape mismatch')
    diff = b - a
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if np.std(a) and np.std(b) else 0.0
    return {'correlation': corr, 'rmse': float(np.sqrt(np.mean(diff * diff))), 'max_abs_change': float(np.max(np.abs(diff)))}
