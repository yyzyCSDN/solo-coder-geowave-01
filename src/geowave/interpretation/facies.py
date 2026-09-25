from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FaciesModel:
    centroids: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

def fit_facies(features: np.ndarray, classes: int, iterations: int=50, seed: int=0) -> FaciesModel:
    x = np.asarray(features, float)
    if x.ndim != 2 or len(x) < classes:
        raise ValueError('invalid feature matrix')
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0) + 1e-12
    z = (x - mean) / scale
    rng = np.random.default_rng(seed)
    centers = z[rng.choice(len(z), classes, replace=False)].copy()
    for _ in range(iterations):
        d = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        new = centers.copy()
        for k in range(classes):
            members = z[labels == k]
            if len(members):
                new[k] = np.mean(members, axis=0)
        if np.allclose(new, centers):
            break
        centers = new
    return FaciesModel(centers, mean, scale)

def classify_facies(model: FaciesModel, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (np.asarray(features, float) - model.feature_mean) / model.feature_scale
    d = np.linalg.norm(x[:, None, :] - model.centroids[None, :, :], axis=2)
    labels = np.argmin(d, axis=1)
    confidence = 1.0 / (1.0 + np.min(d, axis=1))
    return (labels, confidence)
