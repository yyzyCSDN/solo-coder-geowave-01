from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ElasticProperties:
    vp_mps: np.ndarray
    vs_mps: np.ndarray
    density_kgm3: np.ndarray

def lame_parameters(props: ElasticProperties) -> tuple[np.ndarray, np.ndarray]:
    vp = np.asarray(props.vp_mps, float)
    vs = np.asarray(props.vs_mps, float)
    rho = np.asarray(props.density_kgm3, float)
    mu = rho * vs * vs
    lam = rho * vp * vp - 2 * mu
    return (lam, mu)

def poisson_ratio(props: ElasticProperties) -> np.ndarray:
    vp2 = np.asarray(props.vp_mps, float) ** 2
    vs2 = np.asarray(props.vs_mps, float) ** 2
    return 0.5 * (vp2 - 2 * vs2) / np.maximum(vp2 - vs2, 1e-12)

def acoustic_and_shear_impedance(props: ElasticProperties) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(props.density_kgm3, float)
    return (rho * props.vp_mps, rho * props.vs_mps)

def elastic_reflectivity(props: ElasticProperties) -> dict[str, np.ndarray]:
    ai, si = acoustic_and_shear_impedance(props)
    out = {}
    for name, values in (('p', ai), ('s', si), ('density', np.asarray(props.density_kgm3, float))):
        r = np.zeros_like(values)
        r[1:] = (values[1:] - values[:-1]) / (values[1:] + values[:-1] + 1e-12)
        out[name] = r
    return out
