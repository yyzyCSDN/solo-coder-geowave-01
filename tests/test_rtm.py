import numpy as np
import pytest

from geowave.imaging.rtm import (
    RtmGrid, forward, backprop_image, synthesize_shot, rtm_migrate, Shot2D,
    ForwardReplay, BoundaryDiagnostics, GridStabilityError, NumericalInstabilityError,
    CheckpointMismatchError, CheckpointIntegrityError, SimulatedInterruption,
    sponge_taper, require_stable,
)
from geowave.model.wavelet import ricker


def make_grid(nt=400, dt=1e-3, nx=90, nz=70, sponge=15):
    return RtmGrid(nx=nx, nz=nz, dx_m=10.0, dz_m=10.0, dt_s=dt, nt=nt, sponge_cells=sponge)


def make_geometry(grid, n_rec=12):
    m = grid.interior_margin
    src = (grid.nx / 2 * grid.dx_m, (m + 3) * grid.dz_m)
    rec_x = np.linspace((m + 2) * grid.dx_m, (grid.nx - m - 3) * grid.dx_m, n_rec)
    rec = np.column_stack([rec_x, np.full(n_rec, (m + 3) * grid.dz_m)])
    return src, rec


def test_cfl_violation_refused():
    grid = make_grid(dt=4e-3)  # CFL = 2000*0.004*sqrt(2)/10 = 1.13 > 0.866
    v = np.full(grid.shape, 2000.0)
    src, rec = make_geometry(grid)
    with pytest.raises(GridStabilityError, match='CFL'):
        forward(grid, v, ricker(20.0, grid.dt_s), src, rec)


def test_report_and_ppw_warning():
    grid = make_grid()
    v = np.full(grid.shape, 2000.0)
    report = require_stable(grid, v, f_max_hz=200.0)
    assert report['cfl'] == pytest.approx(2000 * 1e-3 * np.hypot(0.1, 0.1))
    with pytest.warns(UserWarning, match='dispersion'):
        require_stable(grid, v, f_max_hz=200.0)


def test_nan_guard_refuses_when_precheck_disabled():
    grid = make_grid(dt=4e-3, nt=600)
    v = np.full(grid.shape, 2000.0)
    src, rec = make_geometry(grid)
    with pytest.raises(NumericalInstabilityError):
        forward(grid, v, ricker(20.0, grid.dt_s), src, rec, check=False)


def test_forward_deterministic_and_finite():
    grid = make_grid()
    v = np.full(grid.shape, 2000.0)
    src, rec = make_geometry(grid)
    w = ricker(20.0, grid.dt_s)
    a = forward(grid, v, w, src, rec)
    b = forward(grid, v, w, src, rec)
    assert np.array_equal(a.data, b.data)
    assert np.isfinite(a.data).all() and np.abs(a.data).max() > 0
    assert set(a.boundary.outgoing_j) == {'left', 'right', 'top', 'bottom'}
    assert all(e > 0 for e in a.boundary.outgoing_j.values())


def test_sponge_strongly_reduces_boundary_reflection():
    src, rec = None, None
    ratios = {}
    for sponge in (15, 0):
        grid = make_grid(sponge=sponge)
        v = np.full(grid.shape, 2000.0)
        s, r = make_geometry(grid)
        res = forward(grid, v, ricker(20.0, grid.dt_s), s, r)
        ratios[sponge] = res.boundary.worst[1]
    assert ratios[15] < 0.05
    assert ratios[0] > 0.3
    assert ratios[0] > 10 * ratios[15]


def test_checkpoint_replay_bit_exact(tmp_path):
    grid = make_grid()
    v = np.full(grid.shape, 2000.0)
    src, rec = make_geometry(grid)
    w = ricker(20.0, grid.dt_s)
    res = forward(grid, v, w, src, rec, ckpt_dir=tmp_path / 'ck', ckpt_every=50)
    replay = ForwardReplay(tmp_path / 'ck', grid, v, w, src, rec)
    assert len(replay.segments) == 8
    fields = [f for seg in replay.segments for f in replay.replay_segment(seg[0])]
    assert len(fields) == grid.nt
    # replayed final state must equal the stored final checkpoint bit-exactly
    final = replay.store.load_state(grid.nt)
    assert np.array_equal(fields[-1], final['u_cur'])
    # checkpointed run records identical data to a checkpoint-free run
    ref = forward(grid, v, w, src, rec)
    assert np.array_equal(res.data, ref.data)


def test_resume_after_crash_bit_exact(tmp_path):
    grid = make_grid()
    v = np.full(grid.shape, 2000.0)
    src, rec = make_geometry(grid)
    w = ricker(20.0, grid.dt_s)
    ck = tmp_path / 'ck'
    with pytest.raises(SimulatedInterruption):
        forward(grid, v, w, src, rec, ckpt_dir=ck, ckpt_every=50, stop_after_iteration=230)
    res = forward(grid, v, w, src, rec, ckpt_dir=ck, ckpt_every=50)
    assert res.resumed_from_iteration == 200
    ref = forward(grid, v, w, src, rec)
    assert np.array_equal(res.data, ref.data)
    again = forward(grid, v, w, src, rec, ckpt_dir=ck, ckpt_every=50)
    assert again.resumed_from_iteration == grid.nt  # completed run served from disk
    assert np.array_equal(again.data, ref.data)


def test_tampered_checkpoint_refused(tmp_path):
    grid = make_grid()
    v = np.full(grid.shape, 2000.0)
    src, rec = make_geometry(grid)
    w = ricker(20.0, grid.dt_s)
    ck = tmp_path / 'ck'
    forward(grid, v, w, src, rec, ckpt_dir=ck, ckpt_every=50)
    target = sorted(ck.glob('ckpt_*.npz'))[2]
    blob = bytearray(target.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    target.write_bytes(bytes(blob))
    with pytest.raises(CheckpointIntegrityError, match='sha256'):
        forward(grid, v, w, src, rec, ckpt_dir=ck, ckpt_every=50)


def test_mismatched_configuration_refused(tmp_path):
    grid = make_grid()
    v = np.full(grid.shape, 2000.0)
    src, rec = make_geometry(grid)
    w = ricker(20.0, grid.dt_s)
    ck = tmp_path / 'ck'
    res = forward(grid, v, w, src, rec, ckpt_dir=ck, ckpt_every=50)
    with pytest.raises(CheckpointMismatchError, match='velocity_sha256'):
        backprop_image(grid, v * 1.01, w, src, rec, res.data, ck)
    with pytest.raises(CheckpointMismatchError, match='wavelet_sha256'):
        backprop_image(grid, v, ricker(25.0, grid.dt_s), src, rec, res.data, ck)
    other_grid = make_grid(nt=401)
    with pytest.raises(CheckpointMismatchError, match='grid.nt'):
        backprop_image(other_grid, v, w, src, rec, res.data, ck)


def test_incomplete_forward_refused_for_imaging(tmp_path):
    grid = make_grid()
    v = np.full(grid.shape, 2000.0)
    src, rec = make_geometry(grid)
    w = ricker(20.0, grid.dt_s)
    ck = tmp_path / 'ck'
    with pytest.raises(SimulatedInterruption):
        forward(grid, v, w, src, rec, ckpt_dir=ck, ckpt_every=50, stop_after_iteration=120)
    data = np.zeros((rec.shape[0], grid.nt))
    with pytest.raises(CheckpointMismatchError, match='did not complete'):
        backprop_image(grid, v, w, src, rec, data, ck)


def test_geometry_outside_interior_refused():
    grid = make_grid()
    v = np.full(grid.shape, 2000.0)
    _, rec = make_geometry(grid)
    with pytest.raises(ValueError, match='sponge'):
        forward(grid, v, ricker(20.0, grid.dt_s), (5.0, 5.0), rec)


def test_rtm_images_horizontal_reflector(tmp_path):
    grid = make_grid(nt=500)
    v_true = np.full(grid.shape, 2000.0)
    v_true[45:, :] = 2600.0  # reflector at z = 450 m
    v_mig = np.full(grid.shape, 2000.0)
    w = ricker(20.0, grid.dt_s)
    m = grid.interior_margin
    rec_x = np.linspace((m + 3) * 10, (grid.nx - m - 4) * 10, 25)
    shots = [Shot2D(sx, (m + 3) * 10.0, rec_x, np.full(rec_x.size, (m + 3) * 10.0))
             for sx in (300.0, 450.0, 600.0)]
    data = []
    for s in shots:
        d = synthesize_shot(grid, v_true, w, s).data
        offset = np.abs(rec_x - s.src_x_m)
        mute = (offset / 2000.0 + 0.05) / grid.dt_s  # tapered mute of the direct arrival
        taper = 20
        for ir, im in enumerate(mute.astype(int)):
            d[ir, :im] = 0.0
            d[ir, im:im + taper] *= 0.5 * (1 - np.cos(np.pi * np.arange(taper) / taper))
        data.append(d)
    result = rtm_migrate(grid, v_mig, w, shots, data, tmp_path / 'ck', ckpt_every=100)
    assert result.image.shape == grid.shape
    assert result.segments_verified == 3 * 5
    cols = slice(m + 5, grid.nx - m - 5)
    row_energy = np.abs(result.image[:, cols]).sum(axis=1)
    window = row_energy[40:51]  # reflector window; shallow low-wavenumber artefacts excluded
    peak_iz = int(np.argmax(window)) + 40
    assert abs(peak_iz - 45) <= 4
    assert row_energy[40:51].max() > 3 * row_energy[52:].max()  # reflector stands out above floor
    assert result.illumination_src[20, cols].sum() > 0
    assert result.boundary_forward.worst[1] < 0.05
