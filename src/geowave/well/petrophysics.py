from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PetrophysicalCurves:
    porosity: np.ndarray
    vshale: np.ndarray
    water_saturation: np.ndarray

def density_porosity(bulk_density_gcc: np.ndarray, matrix_density_gcc: float=2.65, fluid_density_gcc: float=1.0) -> np.ndarray:
    rho = np.asarray(bulk_density_gcc, float)
    return np.clip((matrix_density_gcc - rho) / (matrix_density_gcc - fluid_density_gcc), 0, 1)

def gamma_vshale(gamma: np.ndarray, clean: float, shale: float) -> np.ndarray:
    g = np.asarray(gamma, float)
    if shale <= clean:
        raise ValueError('shale endpoint must exceed clean endpoint')
    index = np.clip((g - clean) / (shale - clean), 0, 1)
    return np.clip(0.083 * (2 ** (3.7 * index) - 1), 0, 1)

def archie_water_saturation(resistivity: np.ndarray, porosity: np.ndarray, rw: float, a: float=1.0, m: float=2.0, n: float=2.0) -> np.ndarray:
    rt = np.maximum(np.asarray(resistivity, float), 1e-12)
    phi = np.maximum(np.asarray(porosity, float), 1e-06)
    sw = (a * rw / (rt * phi ** m)) ** (1 / n)
    return np.clip(sw, 0, 1)

def evaluate_petrophysics(density: np.ndarray, gamma: np.ndarray, resistivity: np.ndarray, rw: float, gamma_clean: float, gamma_shale: float) -> PetrophysicalCurves:
    phi = density_porosity(density)
    vsh = gamma_vshale(gamma, gamma_clean, gamma_shale)
    sw = archie_water_saturation(resistivity, phi, rw)
    return PetrophysicalCurves(phi, vsh, sw)
