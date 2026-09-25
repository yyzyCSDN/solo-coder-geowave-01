from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

@dataclass(frozen=True)
class TravelTimeObservation:
    source_x: float
    receiver_x: float
    depth_m: float
    observed_time_s: float

@dataclass(frozen=True)
class TomographyResult:
    slowness: np.ndarray
    predicted_times: np.ndarray
    residuals: np.ndarray

def straight_ray_tomography(observations: list[TravelTimeObservation], cell_edges_m: np.ndarray, initial_velocity_mps: float=2000, damping: float=0.1) -> TomographyResult:
    edges = np.asarray(cell_edges_m, float)
    ncell = len(edges) - 1
    rows = []
    cols = []
    vals = []
    times = []
    for r, o in enumerate(observations):
        x0, x1 = (o.source_x, o.receiver_x)
        z = o.depth_m
        length = np.hypot(x1 - x0, z)
        samples = max(50, ncell * 5)
        xs = np.linspace(x0, x1, samples)
        ds = length / (samples - 1)
        for x in xs[:-1]:
            c = np.searchsorted(edges, x, side='right') - 1
            if 0 <= c < ncell:
                rows.append(r)
                cols.append(c)
                vals.append(ds)
        times.append(o.observed_time_s)
    G = csr_matrix((vals, (rows, cols)), shape=(len(observations), ncell))
    prior = np.full(ncell, 1 / initial_velocity_mps)
    rhs = np.asarray(times) - G @ prior
    sol = lsqr(G, rhs, damp=damping)[0] + prior
    pred = G @ sol
    res = np.asarray(times) - pred
    return TomographyResult(sol, np.asarray(pred), res)
