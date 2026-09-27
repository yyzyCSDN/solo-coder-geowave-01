import shutil

import numpy as np
import pytest

from geowave.imaging.rtm import (
    CPMLConfig,
    CheckpointMismatchError,
    UnstableGridError,
    acoustic_rtm_2d,
)
from geowave.model.traces import Trace, TraceHeader, TraceSet
from geowave.model.wavelet import ricker


def _small_model():
    nz, nx = 45, 80
    z, x = np.mgrid[0:nz, 0:nx]
    velocity = np.full((nz, nx), 2000.0)
    velocity[z * 8.0 + (x * 8.0 - 320.0) * 0.35 > 210.0] = 2500.0
    return velocity


def _dataset(nt=90, dt=0.002):
    t = np.arange(nt) * dt
    traces = []
    for sx in (180.0, 340.0):
        for ix, rx in enumerate(np.arange(80.0, 570.0, 80.0)):
            travel = 0.10 + abs(rx - sx) / 2500.0
            envelope = np.exp(-0.5 * ((t - travel) / 0.012) ** 2)
            samples = np.sin(2 * np.pi * 25.0 * (t - travel)) * envelope
            header = TraceHeader(f"t-{sx:g}-{ix}", f"s-{sx:g}", f"r-{ix}", sx, 0.0, rx, 0.0, dt)
            traces.append(Trace(header, samples))
    return TraceSet(traces, "rtm-test")


def test_acoustic_rtm_outputs_image_illumination_and_boundary_diagnostics(tmp_path):
    velocity = _small_model()
    wave = ricker(25.0, 0.002, 0.08)
    result = acoustic_rtm_2d(
        _dataset(),
        velocity,
        dx=8.0,
        dz=8.0,
        dt=0.002,
        wavelet=wave,
        cpml=CPMLConfig(8, 8),
        checkpoint_dir=tmp_path / "rtm",
        checkpoint_interval=20,
        max_frequency_hz=50.0,
        boundary_reflection_limit=1.0,
    )
    assert result.image.shape == velocity.shape
    assert np.all(np.isfinite(result.image))
    assert np.count_nonzero(result.image) > 100
    assert np.max(result.source_illumination) > 0.0
    assert np.max(result.receiver_illumination) > 0.0
    assert np.max(result.illumination) > 0.0
    assert np.all(np.isfinite(result.illumination_compensated_image))
    assert set(result.forward_boundary.edges) == {"left", "right", "top", "bottom"}
    assert set(result.reverse_boundary.edges) == {"left", "right", "top", "bottom"}
    assert result.boundary_accepted is True


def test_rtm_resume_matches_fresh_run_and_rejects_corrupt_checkpoint(tmp_path):
    velocity = _small_model()
    wave = ricker(25.0, 0.002, 0.08)
    cp = tmp_path / "rtm"
    result = acoustic_rtm_2d(
        _dataset(),
        velocity,
        dx=8.0,
        dz=8.0,
        dt=0.002,
        wavelet=wave,
        cpml=CPMLConfig(8, 8),
        checkpoint_dir=cp,
        checkpoint_interval=15,
        boundary_reflection_limit=1.0,
    )

    # Simulate an interrupted shot: retain only a verified early forward checkpoint.
    shot = cp / "shot_0001"
    shutil.rmtree(shot / "reverse")
    for path in (shot / "forward").glob("forward_*.npz"):
        time_index = int(path.name.removeprefix("forward_").removesuffix(".npz"))
        if time_index not in (0, 15):
            path.unlink()
            path.with_suffix(".json").unlink()
    (shot / "shot_result.npz").unlink()
    shot.joinpath("shot_result.json").unlink()

    shutil.rmtree(tmp_path / "resume", ignore_errors=True)
    shutil.copytree(cp, tmp_path / "resume")
    resumed = acoustic_rtm_2d(
        _dataset(),
        velocity,
        dx=8.0,
        dz=8.0,
        dt=0.002,
        wavelet=wave,
        cpml=CPMLConfig(8, 8),
        checkpoint_dir=tmp_path / "resume",
        checkpoint_interval=15,
        resume=True,
        boundary_reflection_limit=1.0,
    )
    assert np.allclose(resumed.image, result.image, rtol=0.0, atol=0.0)
    assert np.allclose(resumed.source_illumination, result.source_illumination, rtol=0.0, atol=0.0)

    json_path = shot / "forward" / "forward_15.json"
    manifest = json_path.read_text()
    json_path.write_text(manifest.replace('"kind": "forward"', '"kind": "tampered"'))
    with pytest.raises(CheckpointMismatchError, match="does not match"):
        acoustic_rtm_2d(
            _dataset(),
            velocity,
            dx=8.0,
            dz=8.0,
            dt=0.002,
            wavelet=wave,
            cpml=CPMLConfig(8, 8),
            checkpoint_dir=cp,
            checkpoint_interval=15,
            resume=True,
            boundary_reflection_limit=1.0,
        )


def test_unstable_or_aliased_grid_is_rejected_before_migration():
    velocity = _small_model()
    wave = ricker(25.0, 0.002, 0.08)
    with pytest.raises(UnstableGridError, match="CFL"):
        acoustic_rtm_2d(
            _dataset(),
            velocity,
            dx=8.0,
            dz=8.0,
            dt=0.0026,
            wavelet=wave,
            cpml=CPMLConfig(8, 8),
            checkpoint_dir="/tmp/never-created",
        )

    with pytest.raises(UnstableGridError, match="points per wavelength"):
        acoustic_rtm_2d(
            _dataset(),
            velocity,
            dx=10.0,
            dz=10.0,
            dt=0.002,
            wavelet=wave,
            cpml=CPMLConfig(8, 8),
            checkpoint_dir="/tmp/never-created",
            max_frequency_hz=50.0,
        )
