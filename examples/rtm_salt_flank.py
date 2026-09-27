"""Salt-flank 2-D acoustic RTM demo.

Builds a salt body with a steep (~65 deg) flank, synthesizes shot records on the
true model, and migrates with a smoothed model. Produces the RTM image (raw and
illumination-normalized), source/receiver illumination maps, and per-boundary
reflection diagnostics for the absorbing layer, plus a showcase of the refusal
behaviour on unstable grids and mismatched/corrupted checkpoints.

Run:  python3 examples/rtm_salt_flank.py
Output: examples/output_rtm/  (PNG + NPY + checkpoint directories)
"""
from __future__ import annotations

import shutil
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from geowave.imaging.rtm import (
    RtmGrid, Shot2D, forward, backprop_image, synthesize_shot, rtm_migrate,
    require_stable, GridStabilityError, CheckpointMismatchError, CheckpointIntegrityError,
)
from geowave.model.wavelet import ricker

OUT = Path(__file__).parent / 'output_rtm'


def save_png(path, rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    raw = b''.join(b'\x00' + rgb[i].tobytes() for i in range(h))

    def chunk(tag, payload):
        c = tag + payload
        return struct.pack('>I', len(payload)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)

    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 6))
           + chunk(b'IEND', b''))
    Path(path).write_bytes(png)


def cmap_diverging(a, clip_pct=98.0):
    s = np.percentile(np.abs(a), clip_pct) or 1.0
    x = np.clip(a / s, -1.0, 1.0)
    r = (1 - np.clip(-x, 0, 1)) * 255
    g = (1 - np.abs(x)) * 255
    b = (1 - np.clip(x, 0, 1)) * 255
    return np.stack([r, g, b], -1)


def cmap_sequential(a, log=False):
    a = np.asarray(a, float)
    if log:
        a = np.log10(a / max(a.max(), 1e-300) + 1e-4) + 4.0
    x = np.clip((a - a.min()) / (a.max() - a.min() + 1e-300), 0, 1)
    g = x * 255
    return np.stack([g, g, g], -1)


def cmap_velocity(a):
    x = (a - a.min()) / (a.max() - a.min() + 1e-300)
    r = np.clip(1.6 * x - 0.2, 0, 1) * 255
    g = (1 - np.abs(2 * x - 1)) * 140
    b = np.clip(1.2 - 1.6 * x, 0, 1) * 255
    return np.stack([r, g, b], -1)


def box_smooth(a, w=5, passes=2):
    k = np.ones(w) / w
    out = np.asarray(a, float)
    for _ in range(passes):
        out = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, out)
        out = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, out)
    return out


def build_models(grid):
    zz, xx = np.meshgrid(grid.z_m(), grid.x_m(), indexing='ij')
    v_bg = 1800.0 + 0.55 * zz
    x_flank = 1550.0 - 0.45 * (zz - 400.0)  # steep flank: ~65 deg from horizontal
    salt = (zz >= 400.0) & (xx >= x_flank)
    v_salt = 4400.0
    v_true = np.where(salt, v_salt, v_bg)
    refl = 1.0 + 0.05 * (zz >= 650.0) - 0.07 * (zz >= 950.0) + 0.06 * (zz >= 1200.0)
    v_true = np.where(salt, v_true, v_true * refl)
    v_mig = box_smooth(np.where(salt, v_salt, v_bg), w=5, passes=2)
    return v_true, v_mig


def print_boundary(title, diag):
    print(f'  {title}')
    print(f'  {"edge":<8}{"outgoing J/m":>14}{"incoming J/m":>14}{"ratio":>10}')
    for e in ('left', 'right', 'top', 'bottom'):
        print(f'  {e:<8}{diag.outgoing_j[e]:>14.4e}{diag.incoming_j[e]:>14.4e}{diag.ratio(e):>10.5f}')
    edge, ratio = diag.worst
    print(f'  worst edge: {edge} (reflection ratio {ratio:.5f})')


def refusal_showcase(grid, v_mig, wavelet, shot, data, ckpt_dir, tmp_dir):
    print('\n--- refusal showcase: these must fail loudly, no image produced ---')
    bad_grid = RtmGrid(nx=grid.nx, nz=grid.nz, dx_m=grid.dx_m, dz_m=grid.dz_m,
                       dt_s=3e-3, nt=grid.nt, sponge_cells=grid.sponge_cells)
    try:
        forward(bad_grid, v_mig, np.asarray(wavelet.samples), shot.src(), shot.rec())
        print('BUG: unstable grid was not refused')
    except GridStabilityError as e:
        print(f'REFUSED [grid stability]   : {e}')

    try:
        backprop_image(grid, v_mig * 1.02, wavelet, shot.src(), shot.rec(), data, ckpt_dir)
        print('BUG: mismatched velocity was not refused')
    except CheckpointMismatchError as e:
        print(f'REFUSED [checkpoint meta]  : {e}')

    corrupted = tmp_dir / 'corrupted'
    shutil.copytree(ckpt_dir, corrupted)
    victim = sorted(corrupted.glob('ckpt_*.npz'))[1]
    blob = bytearray(victim.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    victim.write_bytes(bytes(blob))
    try:
        backprop_image(grid, v_mig, wavelet, shot.src(), shot.rec(), data, corrupted)
        print('BUG: corrupted checkpoint was not refused')
    except CheckpointIntegrityError as e:
        print(f'REFUSED [checkpoint sha256]: {e}')

    partial = tmp_dir / 'partial'
    shutil.copytree(ckpt_dir, partial)
    (partial / 'done.flag').unlink()
    try:
        backprop_image(grid, v_mig, wavelet, shot.src(), shot.rec(), data, partial)
        print('BUG: incomplete forward was not refused')
    except CheckpointMismatchError as e:
        print(f'REFUSED [incomplete run]   : {e}')


def main():
    t_start = time.time()
    grid = RtmGrid(nx=200, nz=160, dx_m=10.0, dz_m=10.0, dt_s=1.2e-3, nt=1400, sponge_cells=30)
    v_true, v_mig = build_models(grid)
    f0 = 13.0
    report = require_stable(grid, v_mig, f_max_hz=2.5 * f0)
    print('=== grid / stability ===')
    print(f'  grid {grid.nx}x{grid.nz}, dx=dz={grid.dx_m} m, dt={grid.dt_s * 1e3:.2f} ms, nt={grid.nt}, sponge={grid.sponge_cells} cells')
    print(f"  CFL {report['cfl']:.3f} <= 0.95 x limit {report['limit']:.3f}; "
          f"max dt {report['max_dt_s'] * 1e3:.2f} ms; {report['points_per_wavelength']:.1f} pts/wavelength")

    wavelet = ricker(f0, grid.dt_s, 0.25)
    rec_x = np.linspace(330.0, 1660.0, 134)
    rec_z = np.full(rec_x.size, 330.0)
    shots = [Shot2D(float(sx), 330.0, rec_x, rec_z) for sx in np.linspace(420.0, 1650.0, 7)]

    print(f'\n=== synthesizing observed data on true model ({len(shots)} shots) ===')
    data_list = []
    for i, s in enumerate(shots):
        t0 = time.time()
        obs = synthesize_shot(grid, v_true, wavelet, s)
        offset = np.abs(rec_x - s.src_x_m)
        d = obs.data.copy()
        for ir, im in enumerate(((offset / 2000.0 + 0.06) / grid.dt_s).astype(int)):  # direct-arrival mute
            d[ir, :im] = 0.0
        data_list.append(d)
        print(f'  shot {i + 1}/{len(shots)} x={s.src_x_m:.0f} m  ({time.time() - t0:.1f} s)')
    print_boundary('observed-data forward, shot 1 absorbing-boundary diagnostics', obs.boundary)

    print(f'\n=== RTM on smoothed model ({len(shots)} shots, checkpointed) ===')
    t0 = time.time()
    result = rtm_migrate(grid, v_mig, wavelet, shots, data_list, OUT / 'checkpoints', ckpt_every=200)
    print(f'  migrated {result.shots} shots in {time.time() - t0:.1f} s; '
          f'{result.segments_verified} checkpoint segments replayed and verified bit-exactly')
    print_boundary('RTM forward (migration model), all shots combined', result.boundary_forward)
    print_boundary('RTM backpropagation, all shots combined', result.boundary_backward)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'image_raw.npy', result.image)
    np.save(OUT / 'image_illumination_normalized.npy', result.normalized_image())
    np.save(OUT / 'illumination_src.npy', result.illumination_src)
    np.save(OUT / 'illumination_rcv.npy', result.illumination_rcv)
    np.save(OUT / 'velocity_true.npy', v_true)
    np.save(OUT / 'velocity_migration.npy', v_mig)
    np.save(OUT / 'observed_shot0.npy', data_list[0])
    save_png(OUT / 'velocity_true.png', cmap_velocity(v_true))
    save_png(OUT / 'velocity_migration.png', cmap_velocity(v_mig))
    save_png(OUT / 'image_raw.png', cmap_diverging(result.image))
    save_png(OUT / 'image_illumination_normalized.png', cmap_diverging(result.normalized_image()))
    save_png(OUT / 'illumination_src.png', cmap_sequential(result.illumination_src, log=True))
    save_png(OUT / 'observed_shot0.png', cmap_diverging(data_list[0]))
    print(f'\n  outputs written to {OUT}/ ({time.time() - t_start:.0f} s total)')

    refusal_showcase(grid, v_mig, wavelet, shots[0], data_list[0],
                     OUT / 'checkpoints' / 'shot_000', OUT / 'tmp_refusal')
    shutil.rmtree(OUT / 'tmp_refusal', ignore_errors=True)


if __name__ == '__main__':
    main()
