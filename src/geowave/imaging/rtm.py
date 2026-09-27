"""Two-dimensional constant-density acoustic reverse-time migration.

The propagator uses a pressure/particle-velocity staggered grid with CPML on
all four edges. Forward wavefields are stored as verified checkpoints and are
replayed during backward propagation. Every checkpoint carries a grid
fingerprint, input digest and per-field checksums, so a corrupt or stale file
raises ``CheckpointMismatchError`` instead of producing a plausible image.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geowave.model.traces import TraceSet
from geowave.model.wavelet import Wavelet


class RTMError(RuntimeError):
    """Base failure for an acoustic RTM run."""


class UnstableGridError(RTMError):
    """The finite-difference grid is numerically inadmissible."""


class NumericalInstabilityError(RTMError):
    """A propagated field became non-finite."""


class CheckpointMismatchError(RTMError):
    """A checkpoint is corrupt, stale, or belongs to another run."""


class BoundaryReflectionError(RTMError):
    def __init__(
        self,
        message: str,
        forward_diagnostics: "BoundaryDiagnostics",
        reverse_diagnostics: "BoundaryDiagnostics",
    ):
        super().__init__(message)
        self.forward_diagnostics = forward_diagnostics
        self.reverse_diagnostics = reverse_diagnostics
        self.diagnostics = (forward_diagnostics, reverse_diagnostics)


@dataclass(frozen=True)
class CPMLConfig:
    thickness_x: int = 10
    thickness_z: int = 10
    reflection_target: float = 1.0e-4
    kappa_max: float = 1.0
    alpha_max: float = 0.0
    polynomial_order: int = 2

    def __post_init__(self) -> None:
        if self.thickness_x < 5 or self.thickness_z < 5:
            raise ValueError("CPML thickness must be at least five cells")
        if not 0.0 < self.reflection_target < 1.0:
            raise ValueError("reflection_target must lie in (0, 1)")
        if self.kappa_max < 1.0:
            raise ValueError("kappa_max must be >= 1")
        if self.alpha_max < 0.0:
            raise ValueError("alpha_max must be non-negative")
        if self.polynomial_order < 1:
            raise ValueError("polynomial_order must be positive")


@dataclass(frozen=True)
class AcousticGrid2D:
    velocity: np.ndarray
    dx: float
    dz: float
    dt: float
    nt: int
    cpml: CPMLConfig
    density: float = 1000.0
    spatial_order: int = 4
    cfl_safety: float = 0.98

    def __post_init__(self) -> None:
        v = np.asarray(self.velocity, dtype=np.float64)
        min_x = 2 * self.cpml.thickness_x + 5
        min_z = 2 * self.cpml.thickness_z + 5
        if v.ndim != 2 or v.shape[0] < min_z or v.shape[1] < min_x:
            raise ValueError("velocity must be 2-D and leave at least five physical cells")
        if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
            raise ValueError("velocity must be finite and positive everywhere")
        if self.dx <= 0.0 or self.dz <= 0.0 or self.dt <= 0.0:
            raise ValueError("dx, dz and dt must be positive")
        if not isinstance(self.nt, int) or self.nt < 2:
            raise ValueError("nt must be an integer >= 2")
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.spatial_order not in (2, 4):
            raise ValueError("spatial_order must be 2 or 4")
        if not 0.0 < self.cfl_safety <= 1.0:
            raise ValueError("cfl_safety must lie in (0, 1]")

        courant = float(np.max(v) * self.dt * np.sqrt(self.dx**-2 + self.dz**-2))
        limit = (np.sqrt(2.0) / 2.0 if self.spatial_order == 2 else 6.0 / 7.0) * self.cfl_safety
        if courant >= limit:
            raise UnstableGridError(
                f"CFL number {courant:.4g} exceeds stable limit {limit:.4g}; "
                "refuse to migrate on this grid"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(np.asarray(self.velocity).shape)  # type: ignore[return-value]

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.shape[1], dtype=np.float64) * self.dx

    @property
    def z(self) -> np.ndarray:
        return np.arange(self.shape[0], dtype=np.float64) * self.dz


@dataclass(frozen=True)
class PointLayout2D:
    x: np.ndarray
    z: np.ndarray
    indices: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return int(self.x.size)


@dataclass
class WavefieldState:
    p: np.ndarray
    vx: np.ndarray
    vz: np.ndarray
    psi_p_x: np.ndarray
    psi_p_z: np.ndarray
    psi_v_x: np.ndarray
    psi_v_z: np.ndarray

    @classmethod
    def zero(cls, shape: tuple[int, int]) -> "WavefieldState":
        nz, nx = shape
        return cls(
            np.zeros((nz, nx), np.float64),
            np.zeros((nz, nx - 1), np.float64),
            np.zeros((nz - 1, nx), np.float64),
            np.zeros((nz, nx - 1), np.float64),
            np.zeros((nz - 1, nx), np.float64),
            np.zeros((nz - 1, nx), np.float64),
            np.zeros((nz, nx - 1), np.float64),
        )

    def copy(self) -> "WavefieldState":
        return WavefieldState(*(a.copy() for a in self.astuple()))

    def astuple(self) -> tuple[np.ndarray, ...]:
        return self.p, self.vx, self.vz, self.psi_p_x, self.psi_p_z, self.psi_v_x, self.psi_v_z


@dataclass(frozen=True)
class ForwardResult:
    records: np.ndarray
    source_illumination: np.ndarray
    boundary: "BoundaryDiagnostics | None"


@dataclass(frozen=True)
class ReverseResult:
    image: np.ndarray
    receiver_illumination: np.ndarray
    boundary: "BoundaryDiagnostics | None"


@dataclass(frozen=True)
class EdgeFlux:
    name: str
    outward_energy: float
    inward_energy: float
    reflection_ratio: float


@dataclass(frozen=True)
class BoundaryDiagnostics:
    edge_flux: tuple[EdgeFlux, ...]
    outward_flux: np.ndarray
    max_reflection: float
    accepted: bool

    @property
    def edges(self) -> dict[str, EdgeFlux]:
        return {e.name: e for e in self.edge_flux}


@dataclass(frozen=True)
class RTMResult:
    x: np.ndarray
    z: np.ndarray
    image: np.ndarray
    illumination_compensated_image: np.ndarray
    source_illumination: np.ndarray
    receiver_illumination: np.ndarray
    illumination: np.ndarray
    forward_boundary: BoundaryDiagnostics | None
    reverse_boundary: BoundaryDiagnostics | None
    boundary_accepted: bool


EDGE_NAMES = ("left", "right", "top", "bottom")
_STATE_FIELDS = ("p", "vx", "vz", "psi_p_x", "psi_p_z", "psi_v_x", "psi_v_z")


def array_digest(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array, dtype=np.float64)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def digest_arrays(arrays) -> str:
    h = hashlib.sha256()
    for array in arrays:
        h.update(array_digest(np.asarray(array, dtype=np.float64)).encode())
    return h.hexdigest()


def json_digest(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_grid_dispersion(grid: AcousticGrid2D, max_frequency_hz: float) -> None:
    if max_frequency_hz <= 0.0:
        raise ValueError("max_frequency_hz must be positive")
    minimum_points = 5.0 if grid.spatial_order == 4 else 4.0
    points = float(np.min(grid.velocity) / (max_frequency_hz * max(grid.dx, grid.dz)))
    if points < minimum_points:
        raise UnstableGridError(
            f"only {points:.2f} points per wavelength; require {minimum_points:g}. "
            "Refuse to produce a potentially aliased RTM image"
        )


class AcousticSolver2D:
    def __init__(self, grid: AcousticGrid2D):
        self.grid = grid
        nz, nx = grid.shape
        self.pz, self.px = grid.cpml.thickness_z, grid.cpml.thickness_x
        self.full_shape = (nz + 2 * self.pz, nx + 2 * self.px)
        self.physical_slice = (slice(self.pz, self.pz + nz), slice(self.px, self.px + nx))
        physical_v = np.asarray(grid.velocity, np.float64)
        self.velocity = np.pad(physical_v, ((self.pz, self.pz), (self.px, self.px)), mode="edge")
        self.bulk_modulus = grid.density * self.velocity**2
        self.state = WavefieldState.zero(self.full_shape)

        fz, fx = self.full_shape
        self.bx_p = np.ones((fz, fx - 1))
        self.cx_p = np.zeros_like(self.bx_p)
        self.kx_p = np.ones_like(self.bx_p)
        self.bz_p = np.ones((fz - 1, fx))
        self.cz_p = np.zeros_like(self.bz_p)
        self.kz_p = np.ones_like(self.bz_p)
        self.bx_v = np.ones((fz - 1, fx))
        self.cx_v = np.zeros_like(self.bx_v)
        self.kx_v = np.ones_like(self.bx_v)
        self.bz_v = np.ones((fz - 1, fx))
        self.cz_v = np.zeros_like(self.bz_v)
        self.kz_v = np.ones_like(self.bz_v)
        self._build_cpml()
        self._fingerprint = self._compute_fingerprint()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        payload = {
            "velocity": array_digest(np.asarray(self.grid.velocity, np.float64)),
            "shape": list(self.grid.shape),
            "dx": self.grid.dx,
            "dz": self.grid.dz,
            "dt": self.grid.dt,
            "nt": self.grid.nt,
            "density": self.grid.density,
            "order": self.grid.spatial_order,
            "cfl_safety": self.grid.cfl_safety,
            "cpml": self.grid.cpml.__dict__,
        }
        return json_digest(payload)

    def _cpml_node_arrays(self, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = self.full_shape[axis]
        pml = self.pz if axis == 0 else self.px
        delta = self.grid.dz if axis == 0 else self.grid.dx
        distance = np.zeros(length)
        interior_end = length - pml
        distance[:pml] = (pml - np.arange(pml)) * delta
        distance[interior_end:] = (np.arange(interior_end, length) - (interior_end - 1)) * delta
        pml_thickness = pml * delta
        profile = np.divide(distance, pml_thickness, out=np.zeros_like(distance), where=distance > 0.0) ** self.grid.cpml.polynomial_order
        local_v = np.max(self.velocity, axis=1 if axis == 0 else 0)
        d0 = 1.5 * local_v / delta * np.log(1.0 / self.grid.cpml.reflection_target)
        damping = d0 * profile
        kappa = 1.0 + (self.grid.cpml.kappa_max - 1.0) * profile
        alpha = self.grid.cpml.alpha_max * (1.0 - np.minimum(distance / pml_thickness, 1.0))
        decay = damping / kappa + alpha
        b = np.exp(-decay * self.grid.dt)
        c = np.zeros_like(b)
        active = decay > 0.0
        c[active] = damping[active] / kappa[active] * (b[active] - 1.0) / decay[active]
        return b, c, kappa

    def _build_cpml(self) -> None:
        bz, cz, kz = self._cpml_node_arrays(0)
        bx, cx, kx = self._cpml_node_arrays(1)
        self.bx_v[:] = bx[None, :]
        self.cx_v[:] = cx[None, :]
        self.kx_v[:] = kx[None, :]
        self.bz_v[:] = bz[:, None]
        self.cz_v[:] = cz[:, None]
        self.kz_v[:] = kz[:, None]
        self.bx_p[:] = 0.5 * (bx[None, :-1] + bx[None, 1:])
        self.cx_p[:] = 0.5 * (cx[None, :-1] + cx[None, 1:])
        self.kx_p[:] = 0.5 * (kx[None, :-1] + kx[None, 1:])
        self.bz_p[:] = 0.5 * (bz[:-1, None] + bz[1:, None])
        self.cz_p[:] = 0.5 * (cz[:-1, None] + cz[1:, None])
        self.kz_p[:] = 0.5 * (kz[:-1, None] + kz[1:, None])

    def reset(self) -> None:
        self.state = WavefieldState.zero(self.full_shape)

    def set_state(self, state: WavefieldState) -> None:
        expected = [a.shape for a in WavefieldState.zero(self.full_shape).astuple()]
        actual = [a.shape for a in state.astuple()]
        if actual != expected:
            raise CheckpointMismatchError("wavefield state shape does not match solver grid")
        self.state = state.copy()

    def point_layout(self, x: float | np.ndarray, z: float | np.ndarray) -> PointLayout2D:
        xv = np.atleast_1d(np.asarray(x, dtype=np.float64))
        zv = np.atleast_1d(np.asarray(z, dtype=np.float64))
        if xv.shape != zv.shape or xv.ndim != 1:
            raise ValueError("point x and z must be equal-length 1-D arrays")
        nz, nx = self.grid.shape
        max_x = (nx - 1) * self.grid.dx
        max_z = (nz - 1) * self.grid.dz
        tol = 1.0e-10 * max(self.grid.dx, self.grid.dz)
        if np.any((xv < -tol) | (xv > max_x + tol) | (zv < -tol) | (zv > max_z + tol)):
            raise ValueError("source or receiver lies outside the physical velocity grid")
        gx = np.clip(xv / self.grid.dx, 0.0, max(nx - 2, 0))
        gz = np.clip(zv / self.grid.dz, 0.0, max(nz - 2, 0))
        ix0 = np.minimum(np.floor(gx).astype(int), nx - 2)
        iz0 = np.minimum(np.floor(gz).astype(int), nz - 2)
        wx, wz = gx - ix0, gz - iz0
        ix0 += self.px
        iz0 += self.pz
        indices = np.column_stack((iz0, ix0, iz0 + 1, ix0 + 1))
        weights = np.column_stack(((1 - wx) * (1 - wz), wx * (1 - wz), (1 - wx) * wz, wx * wz))
        return PointLayout2D(xv.copy(), zv.copy(), indices, weights)

    def inject_pressure(self, values: np.ndarray, layout: PointLayout2D) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(layout),):
            raise ValueError("injection values must have one value per point")
        p = self.state.p
        for j, (r0, c0, r1, c1) in enumerate(layout.indices):
            w00, w10, w01, w11 = layout.weights[j]
            np.add.at(p, (r0, c0), values[j] * w00)
            np.add.at(p, (r0, c1), values[j] * w10)
            np.add.at(p, (r1, c0), values[j] * w01)
            np.add.at(p, (r1, c1), values[j] * w11)

    def sample_pressure(self, layout: PointLayout2D) -> np.ndarray:
        p = self.state.p
        out = np.zeros(len(layout))
        for j, (r0, c0, r1, c1) in enumerate(layout.indices):
            w00, w10, w01, w11 = layout.weights[j]
            out[j] = p[r0, c0] * w00 + p[r0, c1] * w10 + p[r1, c0] * w01 + p[r1, c1] * w11
        return out

    @staticmethod
    def _require_finite(a: np.ndarray, what: str) -> None:
        if not np.all(np.isfinite(a)):
            raise NumericalInstabilityError(f"non-finite {what}; refuse to continue wave propagation")

    def _x_gradient_faces(self, p: np.ndarray) -> np.ndarray:
        face = np.empty((p.shape[0], p.shape[1] - 1))
        dx = self.grid.dx
        face[:, 0] = (p[:, 1] - p[:, 0]) / dx
        face[:, -1] = (p[:, -1] - p[:, -2]) / dx
        if self.grid.spatial_order == 4 and p.shape[1] >= 5:
            face[:, 1] = (p[:, 2] - p[:, 1]) / dx
            face[:, -2] = (p[:, -1] - p[:, -2]) / dx
            face[:, 2:-2] = (
                9 / 8 * (p[:, 2:-1] - p[:, 1:-2]) / dx
                - 1 / 24 * (p[:, 3:] - p[:, :-3]) / (4 * dx)
            )
        else:
            face[:, 1:-1] = (p[:, 2:] - p[:, 1:-1]) / dx
        return face

    def _z_gradient_faces(self, p: np.ndarray) -> np.ndarray:
        face = np.empty((p.shape[0] - 1, p.shape[1]))
        dz = self.grid.dz
        face[0, :] = (p[1, :] - p[0, :]) / dz
        face[-1, :] = (p[-1, :] - p[-2, :]) / dz
        if self.grid.spatial_order == 4 and p.shape[0] >= 5:
            face[1, :] = (p[2, :] - p[1, :]) / dz
            face[-2, :] = (p[-1, :] - p[-2, :]) / dz
            face[2:-2, :] = (
                9 / 8 * (p[2:-1, :] - p[1:-2, :]) / dz
                - 1 / 24 * (p[3:, :] - p[:-3, :]) / (4 * dz)
            )
        else:
            face[1:-1, :] = (p[2:, :] - p[1:-1, :]) / dz
        return face

    def _x_divergence_nodes(self, vx: np.ndarray) -> np.ndarray:
        out = np.empty((vx.shape[0], vx.shape[1] + 1))
        dx = self.grid.dx
        out[:, 0] = vx[:, 0] / dx
        out[:, -1] = -vx[:, -1] / dx
        if self.grid.spatial_order == 4 and out.shape[1] >= 5:
            out[:, 1] = (vx[:, 1] - vx[:, 0]) / dx
            out[:, -2] = (vx[:, -1] - vx[:, -2]) / dx
            out[:, 2:-2] = (
                9 / 8 * (vx[:, 2:-1] - vx[:, 1:-2]) / dx
                - 1 / 24 * (vx[:, 3:] - vx[:, :-3]) / (4 * dx)
            )
        else:
            out[:, 1:-1] = (vx[:, 1:] - vx[:, :-1]) / dx
        return out

    def _z_divergence_nodes(self, vz: np.ndarray) -> np.ndarray:
        out = np.empty((vz.shape[0] + 1, vz.shape[1]))
        dz = self.grid.dz
        out[0, :] = vz[0, :] / dz
        out[-1, :] = -vz[-1, :] / dz
        if self.grid.spatial_order == 4 and out.shape[0] >= 5:
            out[1, :] = (vz[1, :] - vz[0, :]) / dz
            out[-2, :] = (vz[-1, :] - vz[-2, :]) / dz
            out[2:-2, :] = (
                9 / 8 * (vz[2:-1, :] - vz[1:-2, :]) / dz
                - 1 / 24 * (vz[3:, :] - vz[:-3, :]) / (4 * dz)
            )
        else:
            out[1:-1, :] = (vz[1:, :] - vz[:-1, :]) / dz
        return out

    def velocity_forward(self) -> None:
        s = self.state
        gx = self._x_gradient_faces(s.p)
        gz = self._z_gradient_faces(s.p)
        s.psi_p_x = self.bx_p * s.psi_p_x + self.cx_p * gx
        s.psi_p_z = self.bz_p * s.psi_p_z + self.cz_p * gz
        s.vx += self.grid.dt / self.grid.density * (gx / self.kx_p + s.psi_p_x)
        s.vz += self.grid.dt / self.grid.density * (gz / self.kz_p + s.psi_p_z)
        self._require_finite(s.vx, "vx")
        self._require_finite(s.vz, "vz")

    def pressure_forward(self, source=None, source_layout: PointLayout2D | None = None) -> None:
        s = self.state
        dxv = self._x_divergence_nodes(s.vx)
        dzv = self._z_divergence_nodes(s.vz)
        s.psi_v_x = self.bx_v * s.psi_v_x + self.cx_v * dxv
        s.psi_v_z = self.bz_v * s.psi_v_z + self.cz_v * dzv
        s.p += self.grid.dt * self.bulk_modulus * (dxv / self.kx_v + s.psi_v_x + dzv / self.kz_v + s.psi_v_z)
        if source is not None:
            if source_layout is None:
                raise ValueError("source_layout is required for pressure injection")
            self.inject_pressure(source, source_layout)
        self._require_finite(s.p, "pressure")

    def pressure_backward(self, source=None, source_layout: PointLayout2D | None = None) -> None:
        s = self.state
        dxv = self._x_divergence_nodes(s.vx)
        dzv = self._z_divergence_nodes(s.vz)
        s.psi_v_x = self.bx_v * s.psi_v_x + self.cx_v * dxv
        s.psi_v_z = self.bz_v * s.psi_v_z + self.cz_v * dzv
        s.p -= self.grid.dt * self.bulk_modulus * (dxv / self.kx_v + s.psi_v_x + dzv / self.kz_v + s.psi_v_z)
        if source is not None:
            if source_layout is None:
                raise ValueError("source_layout is required for adjoint injection")
            self.inject_pressure(source, source_layout)
        self._require_finite(s.p, "reverse pressure")

    def velocity_backward(self) -> None:
        s = self.state
        gx = self._x_gradient_faces(s.p)
        gz = self._z_gradient_faces(s.p)
        s.psi_p_x = self.bx_p * s.psi_p_x + self.cx_p * gx
        s.psi_p_z = self.bz_p * s.psi_p_z + self.cz_p * gz
        s.vx -= self.grid.dt / self.grid.density * (gx / self.kx_p + s.psi_p_x)
        s.vz -= self.grid.dt / self.grid.density * (gz / self.kz_p + s.psi_p_z)
        self._require_finite(s.vx, "reverse vx")
        self._require_finite(s.vz, "reverse vz")

    def physical_pressure(self) -> np.ndarray:
        return self.state.p[self.physical_slice]

    def boundary_flux(self) -> np.ndarray:
        s = self.state
        fz, _ = self.full_shape
        q = np.zeros(4)
        rows = slice(self.pz, fz - self.pz)
        cols = slice(self.px, self.full_shape[1] - self.px)
        # Outward normal flux -p v_n is positive when energy leaves the model.
        q[0] = -self.grid.dz * np.sum(0.5 * (s.p[rows, self.px - 1] + s.p[rows, self.px]) * s.vx[rows, self.px - 1])
        q[1] = self.grid.dz * np.sum(0.5 * (s.p[rows, -self.px - 1] + s.p[rows, -self.px]) * s.vx[rows, -self.px - 1])
        q[2] = -self.grid.dx * np.sum(0.5 * (s.p[self.pz - 1, cols] + s.p[self.pz, cols]) * s.vz[self.pz - 1, cols])
        q[3] = self.grid.dx * np.sum(0.5 * (s.p[-self.pz - 1, cols] + s.p[-self.pz, cols]) * s.vz[-self.pz - 1, cols])
        return q


def boundary_diagnostics(flux: np.ndarray, dt: float, threshold: float = 0.1) -> BoundaryDiagnostics:
    q = np.asarray(flux, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("boundary flux must have shape (nt, 4)")
    outward = dt * np.sum(np.maximum(q, 0.0), axis=0)
    inward = dt * np.sum(np.maximum(-q, 0.0), axis=0)
    edges = []
    ratios = np.zeros(4)
    for i, name in enumerate(EDGE_NAMES):
        ratios[i] = inward[i] / outward[i] if outward[i] > 1e-30 else 0.0
        edges.append(EdgeFlux(name, float(outward[i]), float(inward[i]), float(ratios[i])))
    return BoundaryDiagnostics(tuple(edges), q, float(np.max(ratios)), bool(np.max(ratios) <= threshold))


def save_checkpoint(path: Path, manifest: dict, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["fields"] = {name: array_digest(np.asarray(a, np.float64)) for name, a in arrays.items()}
    tmp_npz = path.with_name(f"tmp_{path.name}.{os.getpid()}.npz")
    tmp_json = path.with_name(f"tmp_{path.name}.{os.getpid()}.json")
    final_json = path.with_suffix(".json")
    np.savez(tmp_npz, **arrays)
    tmp_json.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    tmp_npz.replace(path)
    tmp_json.replace(final_json)


def load_checkpoint(path: Path, kind: str, config_digest: str, input_digest: str, grid: AcousticGrid2D, time: int | None = None) -> dict[str, np.ndarray]:
    json_path = path.with_suffix(".json")
    if not path.is_file() or not json_path.is_file():
        raise CheckpointMismatchError(f"missing checkpoint pair: {path.name}")
    try:
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointMismatchError(f"unreadable checkpoint manifest: {json_path.name}") from exc
    required = {"kind", "time", "nt", "shape", "config", "input", "fields"}
    if not required.issubset(manifest):
        raise CheckpointMismatchError("checkpoint manifest is missing required fields")
    if manifest["kind"] != kind or manifest["config"] != config_digest or manifest["input"] != input_digest:
        raise CheckpointMismatchError("checkpoint does not match grid, propagation kind, or input digest")
    if manifest["nt"] != grid.nt or tuple(manifest["shape"]) != grid.shape:
        raise CheckpointMismatchError("checkpoint dimensions do not match current RTM")
    if time is not None and manifest["time"] != time:
        raise CheckpointMismatchError("checkpoint time stamp does not match requested time")
    try:
        with np.load(path) as npz:
            arrays = {name: np.asarray(npz[name], dtype=np.float64) for name in manifest["fields"]}
    except (OSError, KeyError, ValueError) as exc:
        raise CheckpointMismatchError(f"unreadable checkpoint array: {path.name}") from exc
    for name, expected in manifest["fields"].items():
        if name not in arrays or array_digest(arrays[name]) != expected:
            raise CheckpointMismatchError(f"checkpoint field failed checksum: {name}")
    return arrays


def checkpoint_times(nt: int, interval: int) -> list[int]:
    if interval < 1:
        raise ValueError("checkpoint interval must be >= 1")
    times = list(range(0, nt, interval))
    times.append(nt)
    return times


def prepare_checkpoint_dir(path: Path, resume: bool, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif not path.is_dir():
            raise CheckpointMismatchError(f"checkpoint path is not a directory: {path}")
        elif not resume:
            raise CheckpointMismatchError("checkpoint directory exists; use resume=True or overwrite=True")
    elif resume:
        raise CheckpointMismatchError(f"resume requested but checkpoint directory is missing: {path}")
    path.mkdir(parents=True, exist_ok=True)


def state_arrays(state: WavefieldState) -> dict[str, np.ndarray]:
    return dict(zip(_STATE_FIELDS, state.astuple()))


def state_from_arrays(arrays: dict[str, np.ndarray]) -> WavefieldState:
    try:
        return WavefieldState(*(arrays[name].copy() for name in _STATE_FIELDS))
    except KeyError as exc:
        raise CheckpointMismatchError("checkpoint is missing wavefield state") from exc


def forward_input_digest(source: np.ndarray, source_layout: PointLayout2D, receiver_layout: PointLayout2D) -> str:
    return digest_arrays((
        source,
        source_layout.x,
        source_layout.z,
        source_layout.weights,
        receiver_layout.x,
        receiver_layout.z,
        receiver_layout.weights,
    ))


def propagate_forward(
    grid: AcousticGrid2D,
    source_values: np.ndarray,
    source_layout: PointLayout2D,
    receiver_layout: PointLayout2D,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 32,
    resume: bool = False,
    overwrite: bool = False,
    monitor_boundaries: bool = False,
    accumulate_source_illumination: bool = False,
) -> ForwardResult:
    solver = AcousticSolver2D(grid)
    source = np.asarray(source_values, dtype=np.float64)
    if source.ndim == 1:
        source = source[:, None]
    if source.shape != (grid.nt, len(source_layout)):
        raise ValueError("source_values must have shape (nt, number_of_sources)")
    if len(receiver_layout) == 0:
        raise ValueError("at least one receiver is required")
    input_digest = forward_input_digest(source, source_layout, receiver_layout)

    cp = Path(checkpoint_dir) if checkpoint_dir is not None else None
    save_times = checkpoint_times(grid.nt, checkpoint_interval) if cp is not None else []
    if cp is not None:
        prepare_checkpoint_dir(cp, resume, overwrite)

    records = np.zeros((grid.nt + 1, len(receiver_layout)))
    illumination = np.zeros(grid.shape)
    flux = np.zeros((grid.nt, 4))
    start = 0
    if cp is not None:
        existing = [int(p.name.removeprefix("forward_").removesuffix(".npz")) for p in cp.glob("forward_*.npz")]
        if resume:
            if not existing:
                raise CheckpointMismatchError("resume requested but no forward checkpoint exists")
            start = max(existing)
        elif existing:
            raise CheckpointMismatchError("forward checkpoints exist but resume was not requested")
        if start:
            arrays = load_checkpoint(cp / f"forward_{start}.npz", "forward", solver.fingerprint, input_digest, grid, start)
            solver.set_state(state_from_arrays(arrays))
            records[: start + 1] = arrays["records"]
            illumination[:] = arrays["source_illumination"]

    if cp is not None and start == 0 and not existing:
        initial = state_arrays(solver.state)
        initial["records"] = records[:1].copy()
        initial["source_illumination"] = illumination.copy()
        save_checkpoint(
            cp / "forward_0.npz",
            {"kind": "forward", "time": 0, "nt": grid.nt, "shape": list(grid.shape), "config": solver.fingerprint, "input": input_digest},
            initial,
        )

    for k in range(start + 1, grid.nt + 1):
        solver.velocity_forward()
        if monitor_boundaries:
            flux[k - 1] = solver.boundary_flux()
        solver.pressure_forward(source[k - 1], source_layout)
        records[k] = solver.sample_pressure(receiver_layout)
        if accumulate_source_illumination:
            p = solver.physical_pressure()
            illumination += p * p
        if cp is not None and k in save_times:
            arrays = state_arrays(solver.state)
            arrays["records"] = records[: k + 1].copy()
            arrays["source_illumination"] = illumination.copy()
            save_checkpoint(
                cp / f"forward_{k}.npz",
                {"kind": "forward", "time": k, "nt": grid.nt, "shape": list(grid.shape), "config": solver.fingerprint, "input": input_digest},
                arrays,
            )
    return ForwardResult(records, illumination, boundary_diagnostics(flux, grid.dt) if monitor_boundaries else None)


def propagate_reverse(
    grid: AcousticGrid2D,
    data: np.ndarray,
    receiver_layout: PointLayout2D,
    source: np.ndarray,
    source_layout: PointLayout2D,
    forward_checkpoint_dir: str | Path,
    reverse_checkpoint_dir: str | Path,
    checkpoint_interval: int = 32,
    resume: bool = False,
    overwrite: bool = False,
    monitor_boundaries: bool = False,
) -> ReverseResult:
    solver = AcousticSolver2D(grid)
    data = np.asarray(data, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if source.ndim == 1:
        source = source[:, None]
    if source.shape != (grid.nt, len(source_layout)):
        raise ValueError("source_values must have shape (nt, number_of_sources)")
    if data.shape != (grid.nt, len(receiver_layout)):
        raise ValueError("reverse data must have shape (nt, number_of_receivers)")
    fdir = Path(forward_checkpoint_dir)
    rdir = Path(reverse_checkpoint_dir)
    if not fdir.is_dir():
        raise CheckpointMismatchError("forward checkpoint directory is missing")
    prepare_checkpoint_dir(rdir, resume, overwrite)

    forward_digest = forward_input_digest(source, source_layout, receiver_layout)
    reverse_digest = digest_arrays((data, receiver_layout.x, receiver_layout.z, receiver_layout.weights))
    times = checkpoint_times(grid.nt, checkpoint_interval)
    for t in times:
        # This verifies checksum, shape, grid fingerprint and forward input digest.
        load_checkpoint(fdir / f"forward_{t}.npz", "forward", solver.fingerprint, forward_digest, grid, t)

    image = np.zeros(grid.shape)
    receiver_illumination = np.zeros(grid.shape)
    flux = np.zeros((grid.nt, 4))
    reverse_files = {int(p.name.removeprefix("backward_").removesuffix(".npz")) for p in rdir.glob("backward_*.npz")}
    if resume:
        if not reverse_files:
            raise CheckpointMismatchError("resume requested but no reverse checkpoint exists")
        current = max(reverse_files)
        arrays = load_checkpoint(rdir / f"backward_{current}.npz", "backward", solver.fingerprint, reverse_digest, grid, current)
        solver.set_state(state_from_arrays(arrays))
        image[:] = arrays["image"]
        receiver_illumination[:] = arrays["receiver_illumination"]
        flux[:] = arrays["flux"]
    else:
        if reverse_files:
            raise CheckpointMismatchError("reverse checkpoints exist but resume was not requested")
        current = grid.nt
        solver.reset()
        arrays = state_arrays(solver.state)
        arrays.update({"image": image.copy(), "receiver_illumination": receiver_illumination.copy(), "flux": flux.copy()})
        save_checkpoint(
            rdir / f"backward_{grid.nt}.npz",
            {"kind": "backward", "time": grid.nt, "nt": grid.nt, "shape": list(grid.shape), "config": solver.fingerprint, "input": reverse_digest},
            arrays,
        )

    previous = [t for t in times if t < current]
    while previous:
        segment_start = previous[-1]
        f_arrays = load_checkpoint(fdir / f"forward_{segment_start}.npz", "forward", solver.fingerprint, forward_digest, grid, segment_start)
        solver.set_state(state_from_arrays(f_arrays))
        forward_pressure = [solver.physical_pressure().copy()]
        for k in range(segment_start + 1, current + 1):
            solver.velocity_forward()
            solver.pressure_forward(source[k - 1], source_layout)
            forward_pressure.append(solver.physical_pressure().copy())

        back = load_checkpoint(rdir / f"backward_{current}.npz", "backward", solver.fingerprint, reverse_digest, grid, current)
        solver.set_state(state_from_arrays(back))
        for n in range(current - 1, segment_start - 1, -1):
            solver.pressure_backward(data[n], receiver_layout)
            pb = solver.physical_pressure()
            image += pb * forward_pressure[n - segment_start]
            receiver_illumination += pb * pb
            if monitor_boundaries:
                flux[n - 1] = solver.boundary_flux()
            solver.velocity_backward()

        current = segment_start
        arrays = state_arrays(solver.state)
        arrays.update({"image": image.copy(), "receiver_illumination": receiver_illumination.copy(), "flux": flux.copy()})
        save_checkpoint(
            rdir / f"backward_{current}.npz",
            {"kind": "backward", "time": current, "nt": grid.nt, "shape": list(grid.shape), "config": solver.fingerprint, "input": reverse_digest},
            arrays,
        )
        previous = [t for t in times if t < current]

    return ReverseResult(image, receiver_illumination, boundary_diagnostics(flux, grid.dt) if monitor_boundaries else None)


def wavelet_source(wavelet: Wavelet, nt: int, dt: float, delay_samples: int | None = None) -> np.ndarray:
    if abs(wavelet.dt_s - dt) > 1e-12:
        raise ValueError("wavelet sample interval does not match propagation dt")
    w = np.asarray(wavelet.samples, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("wavelet must contain samples")
    center = w.size // 2
    delay = center if delay_samples is None else int(delay_samples)
    if delay < center or delay + center >= nt:
        raise ValueError("wavelet does not fit within propagation time with requested delay")
    out = np.zeros(nt)
    out[delay - center : delay - center + w.size] = w
    return out


def shot_digest(traces: list, source_x: float, source_z: float, receiver_z: float, wavelet: Wavelet) -> str:
    h = hashlib.sha256()
    h.update(str((source_x, source_z, receiver_z)).encode())
    h.update(array_digest(wavelet.samples).encode())
    for tr in sorted(traces, key=lambda t: t.header.receiver_x):
        h.update(str((tr.header.source_x, tr.header.source_y, tr.header.receiver_x, tr.header.receiver_y, tr.header.dt_s, tr.header.start_s)).encode())
        h.update(array_digest(tr.samples).encode())
    return h.hexdigest()


def acoustic_rtm_2d(
    dataset: TraceSet,
    velocity: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    wavelet: Wavelet,
    checkpoint_dir: str | Path,
    checkpoint_interval: int = 32,
    resume: bool = False,
    overwrite: bool = False,
    source_z: float = 0.0,
    receiver_z: float = 0.0,
    source_delay_samples: int | None = None,
    spatial_order: int = 4,
    cpml: CPMLConfig | None = None,
    max_frequency_hz: float | None = None,
    boundary_reflection_limit: float = 0.1,
    reject_boundary_reflection: bool = True,
    illumination_floor: float = 1e-3,
) -> RTMResult:
    """Migrate 2-D common-shot data with source-normalized and raw RTM outputs.

    The provided ``checkpoint_dir`` is mandatory because backward extrapolation
    reconstructs forward states from a verifiable checkpoint chain. The result
    includes source illumination, receiver illumination, their combined
    illumination map and independent forward/backward boundary-flux diagnostics.
    By default, the image is withheld when diagnostics exceed
    ``boundary_reflection_limit``; pass ``reject_boundary_reflection=False`` to
    inspect a failed diagnostic run.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    cpml = cpml or CPMLConfig()
    shots = dataset.by_shot()
    if not shots:
        raise ValueError("dataset contains no shots")
    shot_ids = sorted(shots)
    first_traces = shots[shot_ids[0]]
    nt = first_traces[0].samples.size

    for tr in dataset.traces:
        if tr.samples.ndim != 1 or tr.samples.size != nt:
            raise ValueError("all RTM traces must have the same non-empty sample count")
        if abs(tr.header.dt_s - dt) > 1e-12:
            raise ValueError("trace dt does not match propagation dt")
        if abs(tr.header.start_s) > 1e-12:
            raise ValueError("RTM traces must start at t=0")
        if abs(tr.header.source_y) > 1e-10 or abs(tr.header.receiver_y) > 1e-10:
            raise ValueError("acoustic_rtm_2d is 2-D; source_y and receiver_y must be zero")

    source_values = wavelet_source(wavelet, nt, dt, source_delay_samples)
    grid = AcousticGrid2D(velocity, dx, dz, dt, nt, cpml, spatial_order=spatial_order)
    if max_frequency_hz is not None:
        validate_grid_dispersion(grid, max_frequency_hz)
    template = AcousticSolver2D(grid)

    root = Path(checkpoint_dir)
    prepare_checkpoint_dir(root, resume, overwrite)

    image_sum = np.zeros(grid.shape)
    source_illum_sum = np.zeros(grid.shape)
    receiver_illum_sum = np.zeros(grid.shape)
    forward_flux_sum = np.zeros((nt, 4))
    reverse_flux_sum = np.zeros((nt, 4))

    for shot_index, shot_id in enumerate(shot_ids):
        traces = sorted(shots[shot_id], key=lambda tr: tr.header.receiver_x)
        source_xs = {tr.header.source_x for tr in traces}
        if len(source_xs) != 1:
            raise ValueError(f"shot {shot_id} has inconsistent source x")
        sx = float(source_xs.pop())
        rx = np.asarray([tr.header.receiver_x for tr in traces], dtype=np.float64)
        if np.unique(rx).size != rx.size:
            raise ValueError(f"shot {shot_id} contains duplicate receiver positions")
        data = np.column_stack([tr.samples for tr in traces]).astype(np.float64)
        source_layout = template.point_layout(np.asarray([sx]), np.asarray([source_z]))
        receiver_layout = template.point_layout(rx, np.full_like(rx, receiver_z))
        current_shot_digest = shot_digest(traces, sx, source_z, receiver_z, wavelet)

        shot_dir = root / f"shot_{shot_index:04d}"
        result_path = shot_dir / "shot_result.npz"
        if result_path.exists():
            if not resume:
                raise CheckpointMismatchError(f"completed shot checkpoint exists: {shot_dir}")
            result = load_checkpoint(result_path, "shot", template.fingerprint, current_shot_digest, grid, nt)
            shot_image = result["image"]
            shot_source = result["source_illumination"]
            shot_receiver = result["receiver_illumination"]
            shot_forward_flux = result["forward_flux"]
            shot_reverse_flux = result["reverse_flux"]
        else:
            shot_dir.mkdir(parents=True, exist_ok=resume)
            fdir = shot_dir / "forward"
            rdir = shot_dir / "reverse"
            f_resume = fdir.is_dir() and bool(list(fdir.glob("forward_*.npz")))
            r_resume = rdir.is_dir() and bool(list(rdir.glob("backward_*.npz")))
            forward = propagate_forward(
                grid,
                source_values,
                source_layout,
                receiver_layout,
                fdir,
                checkpoint_interval,
                resume=f_resume,
                overwrite=False,
                monitor_boundaries=True,
                accumulate_source_illumination=True,
            )
            reverse = propagate_reverse(
                grid,
                data,
                receiver_layout,
                source_values,
                source_layout,
                fdir,
                rdir,
                checkpoint_interval,
                resume=r_resume,
                overwrite=False,
                monitor_boundaries=True,
            )
            shot_image = reverse.image
            shot_source = forward.source_illumination
            shot_receiver = reverse.receiver_illumination
            shot_forward_flux = forward.boundary.outward_flux if forward.boundary else np.zeros((nt, 4))
            shot_reverse_flux = reverse.boundary.outward_flux if reverse.boundary else np.zeros((nt, 4))
            save_checkpoint(
                result_path,
                {"kind": "shot", "time": nt, "nt": nt, "shape": list(grid.shape), "config": template.fingerprint, "input": current_shot_digest},
                {
                    "image": shot_image,
                    "source_illumination": shot_source,
                    "receiver_illumination": shot_receiver,
                    "forward_flux": shot_forward_flux,
                    "reverse_flux": shot_reverse_flux,
                },
            )

        image_sum += shot_image
        source_illum_sum += shot_source
        receiver_illum_sum += shot_receiver
        forward_flux_sum += shot_forward_flux
        reverse_flux_sum += shot_reverse_flux

    forward_diag = boundary_diagnostics(forward_flux_sum, dt, boundary_reflection_limit)
    reverse_diag = boundary_diagnostics(reverse_flux_sum, dt, boundary_reflection_limit)
    accepted = forward_diag.accepted and reverse_diag.accepted
    if reject_boundary_reflection and not accepted:
        raise BoundaryReflectionError(
            f"boundary reflection exceeds limit {boundary_reflection_limit:g}; RTM image withheld",
            forward_diag,
            reverse_diag,
        )

    floor = illumination_floor * float(np.max(source_illum_sum)) if source_illum_sum.size else 0.0
    compensated = np.zeros_like(image_sum)
    valid = source_illum_sum > floor
    compensated[valid] = image_sum[valid] / source_illum_sum[valid]
    joint = np.sqrt(source_illum_sum * receiver_illum_sum)
    return RTMResult(
        grid.x,
        grid.z,
        image_sum,
        compensated,
        source_illum_sum,
        receiver_illum_sum,
        joint,
        forward_diag,
        reverse_diag,
        accepted,
    )
