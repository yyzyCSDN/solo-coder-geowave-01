from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class GridVelocity:
    x: np.ndarray
    z: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.z = np.asarray(self.z, float)
        self.velocity = np.asarray(self.velocity, float)
        if self.velocity.shape != (self.z.size, self.x.size):
            raise ValueError('velocity grid shape')

    def at(self, x: float, z: float) -> float:
        ix = int(np.clip(np.searchsorted(self.x, x), 1, len(self.x) - 1))
        iz = int(np.clip(np.searchsorted(self.z, z), 1, len(self.z) - 1))
        x0, x1 = (self.x[ix - 1], self.x[ix])
        z0, z1 = (self.z[iz - 1], self.z[iz])
        q11 = self.velocity[iz - 1, ix - 1]
        q12 = self.velocity[iz, ix - 1]
        q21 = self.velocity[iz - 1, ix]
        q22 = self.velocity[iz, ix]
        tx = (x - x0) / (x1 - x0 or 1)
        tz = (z - z0) / (z1 - z0 or 1)
        return float((1 - tx) * (1 - tz) * q11 + (1 - tx) * tz * q12 + tx * (1 - tz) * q21 + tx * tz * q22)

def path_time(grid: GridVelocity, path: np.ndarray) -> float:
    p = np.asarray(path, float)
    total = 0.0
    for a, b in zip(p[:-1], p[1:]):
        mid = (a + b) / 2
        ds = float(np.linalg.norm(b - a))
        total += ds / grid.at(float(mid[0]), float(mid[1]))
    return total

def bend_path(grid: GridVelocity, start: tuple[float, float], end: tuple[float, float], nodes: int=31, iterations: int=50, step: float=0.2) -> np.ndarray:
    path = np.column_stack([np.linspace(start[0], end[0], nodes), np.linspace(start[1], end[1], nodes)])
    scale = min(np.ptp(grid.x), np.ptp(grid.z)) / max(nodes, 1)
    for _ in range(iterations):
        changed = 0.0
        for i in range(1, nodes - 1):
            p = path[i].copy()
            normal = np.array([path[i + 1, 1] - path[i - 1, 1], -(path[i + 1, 0] - path[i - 1, 0])])
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal /= norm
            eps = max(scale * 0.25, 0.001)
            path[i] = p + normal * eps
            t1 = path_time(grid, path)
            path[i] = p - normal * eps
            t2 = path_time(grid, path)
            grad = (t1 - t2) / (2 * eps)
            move = -step * grad * scale * normal
            path[i] = p + move
            changed = max(changed, float(np.linalg.norm(move)))
        if changed < 1e-05:
            break
    return path
