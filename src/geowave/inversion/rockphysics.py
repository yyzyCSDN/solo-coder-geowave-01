from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Mineral:
    bulk_gpa: float
    shear_gpa: float
    density_gcc: float

def voigt_reuss_hill(minerals: list[Mineral], fractions: np.ndarray) -> Mineral:
    f = np.asarray(fractions, float)
    if len(minerals) != len(f) or np.any(f < 0) or abs(float(f.sum()) - 1.0) > 1e-06:
        raise ValueError('fractions must match minerals and sum to one')
    K = np.asarray([m.bulk_gpa for m in minerals], float)
    G = np.asarray([m.shear_gpa for m in minerals], float)
    rho = np.asarray([m.density_gcc for m in minerals], float)
    kv = float(np.sum(f * K))
    kr = float(1.0 / np.sum(f / K))
    gv = float(np.sum(f * G))
    gr = float(1.0 / np.sum(f / G))
    return Mineral((kv + kr) / 2, (gv + gr) / 2, float(np.sum(f * rho)))

def elastic_velocities(mineral: Mineral, porosity: float, fluid_bulk_gpa: float, fluid_density_gcc: float) -> tuple[float, float, float]:
    phi = float(np.clip(porosity, 0, 0.6))
    frame_k = mineral.bulk_gpa * (1 - phi) ** 2
    frame_g = mineral.shear_gpa * (1 - phi) ** 2.5
    sat_k = frame_k + phi * fluid_bulk_gpa
    rho = (1 - phi) * mineral.density_gcc + phi * fluid_density_gcc
    vp = np.sqrt((sat_k + 4 * frame_g / 3) * 1000000000.0 / (rho * 1000))
    vs = np.sqrt(frame_g * 1000000000.0 / (rho * 1000))
    return (float(vp), float(vs), float(rho))
