"""2-D acoustic reverse-time migration with checkpointed forward replay.

Forward and backward propagation share the same (2,4) finite-difference
operator with a Cerjan sponge absorbing boundary. Receiver data is
injected as the adjoint source during backpropagation. The forward
wavefield is checkpointed to disk and replayed segment by segment during
the imaging loop; every replayed segment is verified bit-exactly against
the next stored checkpoint.

Refusal policy: an unstable grid, a non-finite wavefield, or a checkpoint
that does not match the current configuration raises an RtmRefusal
subclass instead of returning a plausible-looking but wrong image.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geowave.model.wavelet import Wavelet

__all__ = [
    'RtmGrid', 'RtmRefusal', 'GridStabilityError', 'NumericalInstabilityError',
    'CheckpointMismatchError', 'CheckpointIntegrityError', 'SimulatedInterruption',
    'BoundaryDiagnostics', 'ForwardResult', 'BackpropResult', 'Shot2D', 'ShotImage',
    'RtmResult', 'CFL_LIMIT_2_4', 'stability_report', 'require_stable',
    'sponge_taper', 'forward', 'backprop_image', 'synthesize_shot', 'rtm_shot',
    'rtm_migrate', 'ForwardReplay',
]

CFL_LIMIT_2_4 = math.sqrt(3.0) / 2.0  # von Neumann limit, 2nd-order time / 4th-order space
_C0, _C1, _C2 = -2.5, 4.0 / 3.0, -1.0 / 12.0
_EDGES = ('left', 'right', 'top', 'bottom')


class RtmRefusal(ValueError):
    """RTM refused to run or to return output, rather than returning a wrong image."""


class GridStabilityError(RtmRefusal):
    """The grid/dt/velocity combination violates the CFL stability limit."""


class NumericalInstabilityError(RtmRefusal):
    """The wavefield became non-finite during time stepping."""


class CheckpointMismatchError(RtmRefusal):
    """Checkpoints on disk do not match the current grid/model/geometry, or replay diverged."""


class CheckpointIntegrityError(RtmRefusal):
    """Checkpoint files are missing, unreadable, or failed sha256 verification."""


class SimulatedInterruption(Exception):
    """Test hook: raised by forward() when stop_after_iteration is reached."""


@dataclass(frozen=True)
class RtmGrid:
    nx: int
    nz: int
    dx_m: float
    dz_m: float
    dt_s: float
    nt: int
    sponge_cells: int = 30

    def __post_init__(self):
        if min(self.nx, self.nz) < 8 or self.nt < 1:
            raise ValueError('grid too small')
        if min(self.dx_m, self.dz_m, self.dt_s) <= 0:
            raise ValueError('positive dx, dz, dt required')
        if self.sponge_cells < 0:
            raise ValueError('sponge_cells must be >= 0')
        if min(self.nx, self.nz) < 2 * self.sponge_cells + 6:
            raise ValueError('grid too small for the requested sponge layer')

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def interior_margin(self) -> int:
        return self.sponge_cells + 2

    def x_m(self) -> np.ndarray:
        return np.arange(self.nx) * self.dx_m

    def z_m(self) -> np.ndarray:
        return np.arange(self.nz) * self.dz_m


def stability_report(grid: RtmGrid, velocity_mps: np.ndarray, f_max_hz: float | None = None) -> dict:
    v = np.asarray(velocity_mps, float)
    if v.shape != grid.shape:
        raise ValueError('velocity grid shape')
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError('velocity must be finite and positive')
    vmax, vmin = float(v.max()), float(v.min())
    cfl = vmax * grid.dt_s * math.hypot(1.0 / grid.dx_m, 1.0 / grid.dz_m)
    report = {
        'cfl': cfl,
        'limit': CFL_LIMIT_2_4,
        'v_max_mps': vmax,
        'v_min_mps': vmin,
        'max_dt_s': CFL_LIMIT_2_4 / (vmax * math.hypot(1.0 / grid.dx_m, 1.0 / grid.dz_m)),
    }
    if f_max_hz is not None and f_max_hz > 0:
        report['points_per_wavelength'] = vmin / (f_max_hz * max(grid.dx_m, grid.dz_m))
    return report


def require_stable(grid: RtmGrid, velocity_mps: np.ndarray, f_max_hz: float | None = None, safety: float = 0.95) -> dict:
    report = stability_report(grid, velocity_mps, f_max_hz)
    allowed = CFL_LIMIT_2_4 * safety
    if report['cfl'] > allowed:
        raise GridStabilityError(
            f"CFL number {report['cfl']:.3f} exceeds {safety:.2f} x stability limit "
            f"{CFL_LIMIT_2_4:.3f}; reduce dt below {report['max_dt_s'] * safety:.4e} s "
            f"(current dt {grid.dt_s:.4e} s, v_max {report['v_max_mps']:.0f} m/s). Refusing to run.")
    ppw = report.get('points_per_wavelength')
    if ppw is not None and ppw < 5.0:
        warnings.warn(f'only {ppw:.1f} grid points per shortest wavelength; expect numerical dispersion')
    return report


def sponge_taper(grid: RtmGrid, coeff: float = 0.0153) -> np.ndarray:
    # classic Cerjan sponge: gentle per-cell factor exp(-(coeff*j)^2), j = cells into
    # the sponge; a steeper taper reflects at its own inner edge. Use >= ~30 cells.
    nb = grid.sponge_cells
    damp = np.ones(grid.shape)
    if nb > 0:
        profile = np.exp(-((coeff * np.arange(nb, 0, -1)) ** 2))
        damp[:, :nb] = np.minimum(damp[:, :nb], profile[None, :])
        damp[:, -nb:] = np.minimum(damp[:, -nb:], profile[::-1][None, :])
        damp[:nb, :] = np.minimum(damp[:nb, :], profile[:, None])
        damp[-nb:, :] = np.minimum(damp[-nb:, :], profile[::-1][:, None])
    damp[:2, :] = 0.0
    damp[-2:, :] = 0.0
    damp[:, :2] = 0.0
    damp[:, -2:] = 0.0
    return damp


@dataclass(frozen=True)
class BoundaryDiagnostics:
    """Energy flux through the inner edge of the sponge layer, per boundary."""
    outgoing_j: dict[str, float]
    incoming_j: dict[str, float]

    def ratio(self, edge: str) -> float:
        return self.incoming_j[edge] / max(self.outgoing_j[edge], 1e-300)

    @property
    def worst(self) -> tuple[str, float]:
        return max(((e, self.ratio(e)) for e in self.outgoing_j), key=lambda t: t[1])

    def combined(self, other: 'BoundaryDiagnostics') -> 'BoundaryDiagnostics':
        keys = set(self.outgoing_j) | set(other.outgoing_j)
        return BoundaryDiagnostics(
            {k: self.outgoing_j.get(k, 0.0) + other.outgoing_j.get(k, 0.0) for k in keys},
            {k: self.incoming_j.get(k, 0.0) + other.incoming_j.get(k, 0.0) for k in keys})

    def to_dict(self) -> dict:
        return {'outgoing_j': dict(self.outgoing_j), 'incoming_j': dict(self.incoming_j)}

    @staticmethod
    def from_dict(d: dict) -> 'BoundaryDiagnostics':
        return BoundaryDiagnostics(
            {k: float(x) for k, x in d['outgoing_j'].items()},
            {k: float(x) for k, x in d['incoming_j'].items()})


def _flux_lines(grid: RtmGrid) -> dict:
    nb = max(grid.sponge_cells, 3)
    nx, nz = grid.nx, grid.nz
    return {
        'left': ((slice(None), nb), (slice(None), nb + 1), grid.dz_m, grid.dx_m),
        'right': ((slice(None), nx - 1 - nb), (slice(None), nx - 2 - nb), grid.dz_m, grid.dx_m),
        'top': ((nb, slice(None)), (nb + 1, slice(None)), grid.dx_m, grid.dz_m),
        'bottom': ((nz - 1 - nb, slice(None)), (nz - 2 - nb, slice(None)), grid.dx_m, grid.dz_m),
    }


class BoundaryFlux:
    """Accumulates outgoing/incoming energy flux on lines just inside the sponge."""

    def __init__(self, grid: RtmGrid):
        self._lines = _flux_lines(grid)
        self._vn = {e: np.zeros(np.zeros(grid.shape)[line].shape) for e, (line, _, _, _) in self._lines.items()}
        self._out = {e: 0.0 for e in self._lines}
        self._in = {e: 0.0 for e in self._lines}

    def update(self, u: np.ndarray, dt: float):
        for name, (line, inward, ds, h) in self._lines.items():
            p = u[line]
            vn = self._vn[name]
            vn -= dt * (p - u[inward]) / h  # rho=1: dv_n/dt = -dp/dn
            f = p * vn
            pos = f > 0
            self._out[name] += dt * ds * float(f[pos].sum())
            self._in[name] -= dt * ds * float(f[~pos].sum())

    def state_arrays(self) -> dict:
        state = {f'vn_{e}': self._vn[e] for e in _EDGES}
        state['flux_out'] = np.array([self._out[e] for e in _EDGES])
        state['flux_in'] = np.array([self._in[e] for e in _EDGES])
        return state

    def set_state(self, arrays: dict):
        for e in _EDGES:
            self._vn[e] = np.asarray(arrays[f'vn_{e}'], float).copy()
        for i, e in enumerate(_EDGES):
            self._out[e] = float(arrays['flux_out'][i])
            self._in[e] = float(arrays['flux_in'][i])

    def result(self) -> BoundaryDiagnostics:
        return BoundaryDiagnostics(dict(self._out), dict(self._in))


def _step(u_cur: np.ndarray, u_prev: np.ndarray, c2dt2: np.ndarray, damp: np.ndarray, idx2: float, idz2: float) -> np.ndarray:
    # Cerjan sponge must damp both the new and the current field: damping only the
    # new field gives |lambda| = sqrt(damp) per step, damping both gives damp.
    # u_cur is modified in place; checkpoint/replay rely on this exact op order.
    lap = np.zeros_like(u_cur)
    c = u_cur
    lap[2:-2, 2:-2] = (
        (_C2 * (c[4:, 2:-2] + c[:-4, 2:-2]) + _C1 * (c[3:-1, 2:-2] + c[1:-3, 2:-2]) + _C0 * c[2:-2, 2:-2]) * idz2
        + (_C2 * (c[2:-2, 4:] + c[2:-2, :-4]) + _C1 * (c[2:-2, 3:-1] + c[2:-2, 1:-3]) + _C0 * c[2:-2, 2:-2]) * idx2
    )
    u_next = 2.0 * c - u_prev + c2dt2 * lap
    u_next *= damp
    u_cur *= damp
    return u_next


def _wavelet_samples(wavelet, grid: RtmGrid) -> np.ndarray:
    if isinstance(wavelet, Wavelet):
        if abs(wavelet.dt_s - grid.dt_s) > 1e-12:
            raise ValueError(f'wavelet dt {wavelet.dt_s} does not match grid dt {grid.dt_s}')
        samples = np.asarray(wavelet.samples, float)
    else:
        samples = np.asarray(wavelet, float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError('wavelet samples must be a non-empty 1-D array')
    if samples.size < grid.nt:
        samples = np.pad(samples, (0, grid.nt - samples.size))
    return np.ascontiguousarray(samples[:grid.nt])


def _geometry_indices(grid: RtmGrid, src_xz, rec_xz):
    rec = np.asarray(rec_xz, float)
    if rec.ndim != 2 or rec.shape[1] != 2 or rec.shape[0] < 1:
        raise ValueError('receiver geometry must be (n_rec, 2) x/z in metres')
    sx_i = int(round(float(src_xz[0]) / grid.dx_m))
    sz_i = int(round(float(src_xz[1]) / grid.dz_m))
    rx_i = np.rint(rec[:, 0] / grid.dx_m).astype(np.int64)
    rz_i = np.rint(rec[:, 1] / grid.dz_m).astype(np.int64)
    m = grid.interior_margin
    if not (m <= sx_i < grid.nx - m and m <= sz_i < grid.nz - m):
        raise ValueError(f'source at grid index ({sx_i}, {sz_i}) is inside the sponge layer or off the grid')
    bad = (rx_i < m) | (rx_i >= grid.nx - m) | (rz_i < m) | (rz_i >= grid.nz - m)
    if bad.any():
        raise ValueError(f'{int(bad.sum())} receiver(s) inside the sponge layer or off the grid')
    return sx_i, sz_i, rx_i, rz_i


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _write_json_atomic(path: Path, obj: dict):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def _run_meta(grid: RtmGrid, velocity: np.ndarray, wavelet_samples: np.ndarray, src_xz, rec_xz, every: int) -> dict:
    return {
        'kind': 'geowave.rtm.checkpoints/v1',
        'grid': {
            'nx': grid.nx, 'nz': grid.nz, 'dx_m': grid.dx_m, 'dz_m': grid.dz_m,
            'dt_s': grid.dt_s, 'nt': grid.nt, 'sponge_cells': grid.sponge_cells,
        },
        'every': int(every),
        'velocity_sha256': _sha256_array(velocity),
        'wavelet_sha256': _sha256_array(wavelet_samples),
        'geometry_sha256': _sha256_array(np.concatenate([
            np.asarray(src_xz, float).ravel(), np.asarray(rec_xz, float).ravel()])),
    }


def _verify_meta(recorded: dict, expected: dict):
    diffs = []
    for key in ('kind', 'every', 'velocity_sha256', 'wavelet_sha256', 'geometry_sha256'):
        if recorded.get(key) != expected.get(key):
            diffs.append(f'{key}: checkpoint={recorded.get(key)!r} != current={expected.get(key)!r}')
    rg, eg = recorded.get('grid', {}), expected.get('grid', {})
    for key in sorted(set(rg) | set(eg)):
        if rg.get(key) != eg.get(key):
            diffs.append(f'grid.{key}: checkpoint={rg.get(key)!r} != current={eg.get(key)!r}')
    if diffs:
        raise CheckpointMismatchError('checkpoint metadata mismatch, refusing to reuse: ' + '; '.join(diffs))


class CheckpointStore:
    """Self-describing wavefield checkpoints: state (u_cur, u_prev), recorded data and
    boundary-flux accumulators in one atomic file per checkpoint step."""

    def __init__(self, directory, every: int):
        self.directory = Path(directory)
        self.every = int(every)
        self.manifest_path = self.directory / 'manifest.json'

    def manifest_exists(self) -> bool:
        return self.manifest_path.exists()

    def read_manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise CheckpointIntegrityError(f'unreadable checkpoint manifest: {e}') from e

    def initialize(self, meta: dict):
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.glob('ckpt_*.npz')):
            raise CheckpointIntegrityError(f'{self.directory} contains checkpoints but no valid manifest')
        _write_json_atomic(self.manifest_path, {'meta': meta, 'checkpoints': []})

    def write_checkpoint(self, step: int, u_cur, u_prev, data, flux: BoundaryFlux):
        name = f'ckpt_{step:06d}.npz'
        tmp = self.directory / (name + '.tmp')
        with open(tmp, 'wb') as f:
            np.savez(f, u_cur=u_cur, u_prev=u_prev, data=data, **flux.state_arrays())
        os.replace(tmp, self.directory / name)
        manifest = self.read_manifest()
        manifest['checkpoints'] = [c for c in manifest['checkpoints'] if c['step'] != step]
        manifest['checkpoints'].append({
            'step': int(step), 'file': name, 'sha256': _sha256_file(self.directory / name)})
        manifest['checkpoints'].sort(key=lambda c: c['step'])
        _write_json_atomic(self.manifest_path, manifest)

    def verify_integrity(self) -> dict:
        manifest = self.read_manifest()
        for entry in manifest.get('checkpoints', []):
            p = self.directory / entry['file']
            if not p.exists():
                raise CheckpointIntegrityError(f"missing checkpoint file {entry['file']}")
            if _sha256_file(p) != entry['sha256']:
                raise CheckpointIntegrityError(
                    f"checkpoint file {entry['file']} failed sha256 verification (corrupted or tampered)")
        final = manifest.get('final')
        if final is not None:
            p = self.directory / final['data_file']
            if not p.exists() or _sha256_file(p) != final['sha256']:
                raise CheckpointIntegrityError('final data file failed sha256 verification')
        return manifest

    def is_done(self) -> bool:
        return (self.directory / 'done.flag').exists()

    def steps(self) -> list[int]:
        return [c['step'] for c in self.read_manifest().get('checkpoints', [])]

    def load_state(self, step: int) -> dict:
        manifest = self.verify_integrity()
        entry = next((c for c in manifest['checkpoints'] if c['step'] == step), None)
        if entry is None:
            raise CheckpointIntegrityError(f'no checkpoint at step {step}')
        with np.load(self.directory / entry['file']) as z:
            return {k: z[k] for k in z.files}

    def finalize(self, data: np.ndarray, diagnostics: BoundaryDiagnostics):
        data_file = 'data_final.npy'
        tmp = self.directory / (data_file + '.tmp')
        with open(tmp, 'wb') as f:
            np.save(f, data)
        os.replace(tmp, self.directory / data_file)
        _write_json_atomic(self.directory / 'diagnostics.json', diagnostics.to_dict())
        manifest = self.read_manifest()
        manifest['final'] = {'data_file': data_file, 'sha256': _sha256_file(self.directory / data_file)}
        _write_json_atomic(self.manifest_path, manifest)
        (self.directory / 'done.flag').touch()

    def load_final(self) -> tuple[np.ndarray, BoundaryDiagnostics]:
        self.verify_integrity()
        manifest = self.read_manifest()
        data = np.load(self.directory / manifest['final']['data_file'])
        diag_path = self.directory / 'diagnostics.json'
        diagnostics = BoundaryDiagnostics.from_dict(json.loads(diag_path.read_text()))
        return data, diagnostics


@dataclass(frozen=True)
class ForwardResult:
    data: np.ndarray  # (n_rec, nt) receiver data
    boundary: BoundaryDiagnostics
    checkpoint_dir: Path | None
    resumed_from_iteration: int
    stability: dict


def forward(grid: RtmGrid, velocity_mps: np.ndarray, wavelet, src_xz, rec_xz,
            ckpt_dir=None, ckpt_every: int = 100, safety: float = 0.95,
            check: bool = True, stop_after_iteration: int | None = None) -> ForwardResult:
    """Acoustic forward modelling with sponge boundary, point-source injection and
    optional checkpointing. With ckpt_dir set, a previously interrupted run resumes
    from its latest checkpoint; a completed run is returned from disk unchanged."""
    v = np.ascontiguousarray(np.asarray(velocity_mps, dtype=np.float64))
    w = _wavelet_samples(wavelet, grid)
    sx_i, sz_i, rx_i, rz_i = _geometry_indices(grid, src_xz, rec_xz)
    stab = require_stable(grid, v, safety=safety) if check else stability_report(grid, v)
    c2dt2 = (v * grid.dt_s) ** 2
    damp = sponge_taper(grid)
    idx2, idz2 = 1.0 / grid.dx_m ** 2, 1.0 / grid.dz_m ** 2
    flux = BoundaryFlux(grid)
    data = np.zeros((rx_i.size, grid.nt))
    u_cur = np.zeros(grid.shape)
    u_prev = np.zeros(grid.shape)
    store = None
    start_it = 0
    if ckpt_dir is not None:
        store = CheckpointStore(ckpt_dir, ckpt_every)
        meta = _run_meta(grid, v, w, src_xz, rec_xz, ckpt_every)
        if store.manifest_exists():
            _verify_meta(store.read_manifest().get('meta', {}), meta)
            store.verify_integrity()
            if store.is_done():
                final_data, final_diag = store.load_final()
                return ForwardResult(final_data, final_diag, store.directory, grid.nt, stab)
            steps = store.steps()
            if steps:
                start_it = max(steps)
                state = store.load_state(start_it)
                u_cur, u_prev, data = state['u_cur'], state['u_prev'], state['data']
                flux.set_state(state)
        else:
            store.initialize(meta)
    for it in range(start_it, grid.nt):
        if stop_after_iteration is not None and it >= stop_after_iteration:
            raise SimulatedInterruption(f'simulated crash at iteration {it}')
        u_next = _step(u_cur, u_prev, c2dt2, damp, idx2, idz2)
        u_next[sz_i, sx_i] += c2dt2[sz_i, sx_i] * w[it]
        data[:, it] = u_next[rz_i, rx_i]
        flux.update(u_next, grid.dt_s)
        u_prev, u_cur = u_cur, u_next
        if (it + 1) % 32 == 0 and not np.isfinite(u_cur).all():
            raise NumericalInstabilityError(
                f"non-finite wavefield at iteration {it + 1}; run aborted, no output produced "
                f"(CFL {stab['cfl']:.3f} vs limit {CFL_LIMIT_2_4:.3f})")
        if store is not None and (it + 1) % store.every == 0:
            store.write_checkpoint(it + 1, u_cur, u_prev, data, flux)
    if not np.isfinite(u_cur).all():
        raise NumericalInstabilityError('non-finite wavefield at final iteration; run aborted')
    if store is not None:
        if not store.steps() or max(store.steps()) < grid.nt:
            store.write_checkpoint(grid.nt, u_cur, u_prev, data, flux)
        store.finalize(data, flux.result())
    return ForwardResult(data, flux.result(), store.directory if store else None, start_it, stab)


class ForwardReplay:
    """Recomputes the forward wavefield segment by segment from checkpoints and
    verifies each segment bit-exactly against the next stored checkpoint."""

    def __init__(self, ckpt_dir, grid: RtmGrid, velocity_mps: np.ndarray, wavelet, src_xz, rec_xz, safety: float = 0.95):
        self.store = CheckpointStore(ckpt_dir, every=0)
        if not self.store.manifest_exists():
            raise CheckpointMismatchError(f'no checkpoint manifest in {ckpt_dir}')
        manifest = self.store.read_manifest()
        v = np.ascontiguousarray(np.asarray(velocity_mps, dtype=np.float64))
        self._w = _wavelet_samples(wavelet, grid)
        meta = _run_meta(grid, v, self._w, src_xz, rec_xz, manifest['meta']['every'])
        _verify_meta(manifest.get('meta', {}), meta)
        self.store.verify_integrity()
        if not self.store.is_done():
            raise CheckpointMismatchError(
                'forward run did not complete (no done.flag); refusing to image from partial checkpoints')
        require_stable(grid, v, safety=safety)
        self._grid = grid
        self._c2dt2 = (v * grid.dt_s) ** 2
        self._damp = sponge_taper(grid)
        self._sx_i, self._sz_i, _, _ = _geometry_indices(grid, src_xz, rec_xz)
        steps = [0] + self.store.steps()
        self._segments = list(zip(steps[:-1], steps[1:]))

    @property
    def segments(self) -> list[tuple[int, int]]:
        return list(self._segments)

    def replay_segment(self, start: int) -> list[np.ndarray]:
        segment = next((s for s in self._segments if s[0] == start), None)
        if segment is None:
            raise CheckpointMismatchError(f'no checkpoint segment starts at iteration {start}')
        start, stop = segment
        if start == 0:
            u_cur = np.zeros(self._grid.shape)
            u_prev = np.zeros(self._grid.shape)
        else:
            state = self.store.load_state(start)
            u_cur, u_prev = state['u_cur'], state['u_prev']
        idx2, idz2 = 1.0 / self._grid.dx_m ** 2, 1.0 / self._grid.dz_m ** 2
        fields = []
        for it in range(start, stop):
            u_next = _step(u_cur, u_prev, self._c2dt2, self._damp, idx2, idz2)
            u_next[self._sz_i, self._sx_i] += self._c2dt2[self._sz_i, self._sx_i] * self._w[it]
            fields.append(u_next)
            u_prev, u_cur = u_cur, u_next
        expected = self.store.load_state(stop)
        if not (np.array_equal(u_cur, expected['u_cur']) and np.array_equal(u_prev, expected['u_prev'])):
            raise CheckpointMismatchError(
                f'replay from checkpoint {start} diverged from checkpoint {stop}; '
                'checkpoints were not produced by this exact configuration')
        return fields


@dataclass(frozen=True)
class BackpropResult:
    image: np.ndarray
    illumination_src: np.ndarray
    illumination_rcv: np.ndarray
    boundary: BoundaryDiagnostics
    segments_verified: int


def backprop_image(grid: RtmGrid, velocity_mps: np.ndarray, wavelet, src_xz, rec_xz,
                   data: np.ndarray, ckpt_dir, safety: float = 0.95) -> BackpropResult:
    """Backpropagates receiver data and applies the cross-correlation imaging
    condition against the checkpoint-replayed source wavefield."""
    v = np.ascontiguousarray(np.asarray(velocity_mps, dtype=np.float64))
    _, _, rx_i, rz_i = _geometry_indices(grid, src_xz, rec_xz)
    replay = ForwardReplay(ckpt_dir, grid, v, wavelet, src_xz, rec_xz, safety)
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if data.shape != (rx_i.size, grid.nt):
        raise ValueError(f'data shape {data.shape} does not match (n_rec, nt) = {(rx_i.size, grid.nt)}')
    c2dt2 = (v * grid.dt_s) ** 2
    damp = sponge_taper(grid)
    idx2, idz2 = 1.0 / grid.dx_m ** 2, 1.0 / grid.dz_m ** 2
    flux = BoundaryFlux(grid)
    image = np.zeros(grid.shape)
    illum_src = np.zeros(grid.shape)
    illum_rcv = np.zeros(grid.shape)
    ub_cur = np.zeros(grid.shape)
    ub_prev = np.zeros(grid.shape)
    verified = 0
    for start, stop in reversed(replay.segments):
        u_fwd = replay.replay_segment(start)
        verified += 1
        for it in range(stop - 1, start - 1, -1):
            ub_next = _step(ub_cur, ub_prev, c2dt2, damp, idx2, idz2)
            np.add.at(ub_next, (rz_i, rx_i), c2dt2[rz_i, rx_i] * data[:, it])
            flux.update(ub_next, grid.dt_s)
            uf = u_fwd[it - start]
            image += uf * ub_next
            illum_src += uf * uf
            illum_rcv += ub_next * ub_next
            ub_prev, ub_cur = ub_cur, ub_next
        if not np.isfinite(ub_cur).all():
            raise NumericalInstabilityError(f'non-finite backpropagated field near iteration {stop}; no image produced')
    return BackpropResult(image, illum_src, illum_rcv, flux.result(), verified)


@dataclass(frozen=True)
class Shot2D:
    src_x_m: float
    src_z_m: float
    rec_x_m: np.ndarray
    rec_z_m: np.ndarray

    def src(self) -> tuple[float, float]:
        return (float(self.src_x_m), float(self.src_z_m))

    def rec(self) -> np.ndarray:
        return np.column_stack([np.asarray(self.rec_x_m, float), np.asarray(self.rec_z_m, float)])


def synthesize_shot(grid: RtmGrid, velocity_mps: np.ndarray, wavelet, shot: Shot2D, safety: float = 0.95) -> ForwardResult:
    return forward(grid, velocity_mps, wavelet, shot.src(), shot.rec(), ckpt_dir=None, safety=safety)


@dataclass(frozen=True)
class ShotImage:
    image: np.ndarray
    illumination_src: np.ndarray
    illumination_rcv: np.ndarray
    boundary_forward: BoundaryDiagnostics
    boundary_backward: BoundaryDiagnostics
    segments_verified: int


def rtm_shot(grid: RtmGrid, velocity_mps: np.ndarray, wavelet, shot: Shot2D,
             data: np.ndarray, ckpt_dir, ckpt_every: int = 100, safety: float = 0.95) -> ShotImage:
    fwd = forward(grid, velocity_mps, wavelet, shot.src(), shot.rec(),
                  ckpt_dir=ckpt_dir, ckpt_every=ckpt_every, safety=safety)
    bwd = backprop_image(grid, velocity_mps, wavelet, shot.src(), shot.rec(), data,
                         ckpt_dir=ckpt_dir, safety=safety)
    return ShotImage(bwd.image, bwd.illumination_src, bwd.illumination_rcv,
                     fwd.boundary, bwd.boundary, bwd.segments_verified)


@dataclass(frozen=True)
class RtmResult:
    x_m: np.ndarray
    z_m: np.ndarray
    image: np.ndarray
    illumination_src: np.ndarray
    illumination_rcv: np.ndarray
    boundary_forward: BoundaryDiagnostics
    boundary_backward: BoundaryDiagnostics
    shots: int
    segments_verified: int

    def normalized_image(self, eps_fraction: float = 1e-2) -> np.ndarray:
        illum = self.illumination_src
        return self.image / (illum + eps_fraction * max(float(illum.max()), 1e-300))


def rtm_migrate(grid: RtmGrid, velocity_mps: np.ndarray, wavelet, shots: list[Shot2D],
                data_list: list[np.ndarray], ckpt_root, ckpt_every: int = 100,
                safety: float = 0.95) -> RtmResult:
    if len(shots) != len(data_list):
        raise ValueError('shots and data_list length mismatch')
    if not shots:
        raise ValueError('at least one shot required')
    image = np.zeros(grid.shape)
    illum_src = np.zeros(grid.shape)
    illum_rcv = np.zeros(grid.shape)
    boundary_fwd = BoundaryDiagnostics({e: 0.0 for e in _EDGES}, {e: 0.0 for e in _EDGES})
    boundary_bwd = BoundaryDiagnostics({e: 0.0 for e in _EDGES}, {e: 0.0 for e in _EDGES})
    verified = 0
    for i, (shot, data) in enumerate(zip(shots, data_list)):
        r = rtm_shot(grid, velocity_mps, wavelet, shot, data,
                     Path(ckpt_root) / f'shot_{i:03d}', ckpt_every, safety)
        image += r.image
        illum_src += r.illumination_src
        illum_rcv += r.illumination_rcv
        boundary_fwd = boundary_fwd.combined(r.boundary_forward)
        boundary_bwd = boundary_bwd.combined(r.boundary_backward)
        verified += r.segments_verified
    return RtmResult(grid.x_m(), grid.z_m(), image, illum_src, illum_rcv,
                     boundary_fwd, boundary_bwd, len(shots), verified)
