from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

@dataclass(frozen=True)
class ArrivalPick:
    shot: int
    receiver: int
    time_s: float
    offset_m: float

@dataclass(frozen=True)
class StaticSolution:
    shot_terms_s: np.ndarray
    receiver_terms_s: np.ndarray
    slowness_s_per_m: float
    residual_rms_s: float

def solve_linear_refraction_statics(picks: list[ArrivalPick], shot_count: int, receiver_count: int, damping: float=0.0001) -> StaticSolution:
    rows = []
    cols = []
    vals = []
    b = []
    velocity_col = shot_count + receiver_count
    for row, p in enumerate(picks):
        if not 0 <= p.shot < shot_count or not 0 <= p.receiver < receiver_count:
            raise ValueError('pick index outside declared geometry')
        rows.extend([row, row, row])
        cols.extend([p.shot, shot_count + p.receiver, velocity_col])
        vals.extend([1.0, 1.0, abs(p.offset_m)])
        b.append(p.time_s)
    G = csr_matrix((vals, (rows, cols)), shape=(len(picks), velocity_col + 1))
    sol = lsqr(G, np.asarray(b), damp=damping)[0]
    predicted = G @ sol
    residual = np.asarray(b) - predicted
    shot = sol[:shot_count] - np.mean(sol[:shot_count])
    receiver = sol[shot_count:velocity_col] - np.mean(sol[shot_count:velocity_col])
    return StaticSolution(shot, receiver, float(sol[-1]), float(np.sqrt(np.mean(residual * residual))))
