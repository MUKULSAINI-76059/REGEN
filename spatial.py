import numpy as np
from scipy.ndimage import gaussian_filter


def fix_rain_place(values, sigma=1.0, seed=42):
    """Mimic a U-Net: add small noise to the 10x10 rain field, then smooth it."""
    g = np.asarray(values, float).reshape(10, 10)
    noisy = g + np.random.default_rng(seed).normal(0, 0.05 * g.std() + 1e-6, g.shape)
    return np.clip(gaussian_filter(noisy, sigma, mode="nearest"), 0, None).ravel()
