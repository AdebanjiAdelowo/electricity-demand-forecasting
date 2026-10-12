"""Gaussian PCA window model: a small probabilistic generative baseline."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .network import solve


def reference(n: int, seed: int, hours: int = 24) -> tuple[np.ndarray, dict]:
    """Independent simulated days, conditioned on whole-day network feasibility.

    Rejection is part of the explicitly defined reference distribution. It is
    not used to hide generator failures. All draws and acceptance are recorded.
    """
    rng = np.random.default_rng(seed)
    windows = []
    attempted = 0
    h = np.arange(hours)
    load_shape = 0.85 + 0.22 * np.cos(2 * np.pi * (h - 19) / 24)
    solar_shape = np.maximum(0, np.sin(np.pi * (h - 6) / 12))
    while len(windows) < n and attempted < 100 * n:
        attempted += 1
        common = rng.lognormal(0, 0.30)
        local = rng.lognormal(0, 0.12, 4)
        eps = np.zeros((hours, 4))
        for t in range(1, hours):
            eps[t] = 0.8 * eps[t - 1] + rng.normal(0, 0.015, 4)
        loads = np.maximum(0, common * local * np.array([0.22, 0.19, 0.16, 0.15]) * load_shape[:, None] + eps)
        solar = rng.uniform(0.1, 0.65) * solar_shape * np.clip(1 + rng.normal(0, 0.1, hours), 0.5, 1.5)
        x = np.column_stack([loads, solar])
        if solve(x)["feasible"].all():
            windows.append(x)
    if len(windows) != n:
        raise RuntimeError("Insufficient feasible reference trajectories")
    return np.stack(windows), {"attempted": attempted, "accepted": n, "seed": seed}


class GaussianWindows:
    """Standardized PCA with Gaussian factors and isotropic residual noise."""
    def fit(self, windows: np.ndarray, rank: int = 12):
        if windows.ndim != 3 or not np.isfinite(windows).all() or len(windows) < 3:
            raise ValueError("Need finite training windows")
        self.shape = windows.shape[1:]
        a = windows.reshape(len(windows), -1)
        self.mean = a.mean(axis=0)
        self.scale = np.maximum(a.std(axis=0), 0.02)
        z = (a - self.mean) / self.scale
        self.pca = PCA(n_components=min(rank, len(a) - 1, a.shape[1]), svd_solver="full").fit(z)
        self.noise = max(float(self.pca.noise_variance_), 1e-6)
        return self

    def sample(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(n, self.pca.n_components_)) * np.sqrt(self.pca.explained_variance_)
        z = latent @ self.pca.components_ + self.pca.mean_
        # PPCA residual in the orthogonal complement, avoiding double variance on retained factors.
        eps = rng.normal(size=z.shape)
        eps -= (eps @ self.pca.components_.T) @ self.pca.components_
        z += np.sqrt(self.noise) * eps
        return ((z * self.scale) + self.mean).reshape((n, *self.shape))
