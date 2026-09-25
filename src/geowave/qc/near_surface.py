from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.velocity.refraction_statics import StaticSolution

@dataclass(frozen=True)
class NearSurfaceQC:
    shot_static_p95_ms: float
    receiver_static_p95_ms: float
    max_static_ms: float
    refraction_velocity_mps: float
    residual_rms_ms: float

def near_surface_qc(solution: StaticSolution) -> NearSurfaceQC:
    shot = np.abs(solution.shot_terms_s * 1000)
    receiver = np.abs(solution.receiver_terms_s * 1000)
    combined = np.concatenate([shot, receiver]) if shot.size or receiver.size else np.array([0.0])
    velocity = 1.0 / solution.slowness_s_per_m if solution.slowness_s_per_m > 0 else 0.0
    return NearSurfaceQC(float(np.percentile(shot, 95)) if shot.size else 0.0, float(np.percentile(receiver, 95)) if receiver.size else 0.0, float(np.max(combined)), float(velocity), float(solution.residual_rms_s * 1000))

def static_outliers(solution: StaticSolution, threshold_ms: float) -> dict[str, list[int]]:
    shot = np.where(np.abs(solution.shot_terms_s * 1000) > threshold_ms)[0].astype(int).tolist()
    receiver = np.where(np.abs(solution.receiver_terms_s * 1000) > threshold_ms)[0].astype(int).tolist()
    return {'shots': shot, 'receivers': receiver}
