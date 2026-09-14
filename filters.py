"""Despeckling filter bank.

Median / Lee / SRAD / NLM / Wavelet are moved here unchanged from the
original gen_us.py (identical math, identical parameters) so they can be
reused by both the generic-phantom pipeline and the multi-tissue study.

Kuan, Frost and Bilateral are added as classical despeckling filters
documented in the literature review (see plan): Kuan is the
multiplicative-noise-correct counterpart to Lee's linear MMSE model, Frost
is an adaptive exponentially-weighted window filter, and Bilateral is the
edge-preserving joint domain/range filter that speckle-reducing bilateral
filter variants build on.
"""
import numpy as np
import cv2
from scipy.ndimage import median_filter, uniform_filter


# ---------------- existing filters (unchanged from gen_us.py) ----------------

def median(img, size=9):
    return median_filter(img, size)


def lee(img, size=11, cu=0.523):
    x = img.astype(np.float64)
    mu = uniform_filter(x, size)
    mu2 = uniform_filter(x * x, size)
    var = np.maximum(mu2 - mu * mu, 0)
    ci2 = var / (mu * mu + 1e-8)
    W = np.clip(1.0 - (cu * cu) / (ci2 + 1e-8), 0, 1)
    return mu + W * (x - mu)


def srad(img, n=400, dt=0.12):
    x = img.astype(np.float64) + 1.0
    for t in range(n):
        q0 = 1.0 / np.sqrt(max(1e-6, 1.0 + t * 0.012)) * 0.9
        N = np.roll(x, -1, 0); S = np.roll(x, 1, 0); E = np.roll(x, -1, 1); W = np.roll(x, 1, 1)
        dN, dS, dE, dW = N - x, S - x, E - x, W - x
        g2 = (dN ** 2 + dS ** 2 + dE ** 2 + dW ** 2) / (x * x)
        lap = (dN + dS + dE + dW) / x
        num = 0.5 * g2 - (1 / 16.0) * lap ** 2
        den = (1.0 + 0.25 * lap) ** 2 + 1e-8
        q2 = np.maximum(num / den, 0)
        c = 1.0 / (1.0 + (q2 - q0 * q0) / (q0 * q0 * (1 + q0 * q0) + 1e-8))
        c = np.clip(c, 0, 1)
        cN, cS = c, np.roll(c, 1, 0); cE, cW = c, np.roll(c, 1, 1)
        x = x + (dt / 4.0) * (cN * dN + cS * dS + cE * dE + cW * dW)
    return x - 1.0


def wavelet_despeckle(img):
    from skimage.restoration import denoise_wavelet
    x = np.log1p(img.astype(np.float64) / 255.0)
    y = denoise_wavelet(x, sigma=0.10, wavelet='db4', mode='soft',
                         wavelet_levels=4, rescale_sigma=True)
    return np.expm1(y) * 255.0


def nlm(img):
    from skimage.restoration import denoise_nl_means
    x = img.astype(np.float64) / 255.0
    y = denoise_nl_means(x, h=0.16, sigma=0.10, fast_mode=True,
                          patch_size=5, patch_distance=11)
    return y * 255.0


# ---------------- new filters ----------------

def kuan(img, size=11, cu=0.523):
    """Kuan filter: adaptive MMSE weighting derived for multiplicative
    (speckle) noise, W = (1 - Cu^2/Ci^2) / (1 + Cu^2), vs. Lee's simpler
    linear-model weighting W = 1 - Cu^2/Ci^2."""
    x = img.astype(np.float64)
    mu = uniform_filter(x, size)
    mu2 = uniform_filter(x * x, size)
    var = np.maximum(mu2 - mu * mu, 0)
    ci2 = var / (mu * mu + 1e-8)
    cu2 = cu * cu
    W = np.clip((1.0 - cu2 / (ci2 + 1e-8)) / (1.0 + cu2), 0, 1)
    return mu + W * (x - mu)


def frost(img, size=7, damping=2.0):
    """Frost filter: pixels in a window are combined with an exponentially
    decaying kernel whose decay rate alpha is driven by the local
    coefficient of variation (more decay = more smoothing in homogeneous
    areas, less smoothing where local variance/edges are high).

    Implemented as a sum of shifted-array taps (same vectorization style as
    the srad() diffusion above) so the whole window is applied in NumPy
    without a per-pixel Python loop.
    """
    x = img.astype(np.float64)
    mu = uniform_filter(x, size)
    mu2 = uniform_filter(x * x, size)
    var = np.maximum(mu2 - mu * mu, 0)
    ci2 = var / (mu * mu + 1e-8)
    alpha = damping * ci2

    half = size // 2
    num = np.zeros_like(x)
    den = np.zeros_like(x)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            dist = np.hypot(dy, dx)
            shifted = np.roll(np.roll(x, -dy, axis=0), -dx, axis=1)
            w = np.exp(-alpha * dist)
            num += w * shifted
            den += w
    return num / (den + 1e-8)


def bilateral(img, d=9, sigma_color=None, sigma_space=9):
    """Classical bilateral filter, the edge-preserving joint domain/range
    smoother that speckle-reducing bilateral filter variants build on.
    sigma_color defaults to a fraction of the image's own intensity spread
    so it self-calibrates to the speckle level of each phantom."""
    x = img.astype(np.float32)
    if sigma_color is None:
        sigma_color = 0.18 * (x.max() - x.min() + 1e-6)
    return cv2.bilateralFilter(x, d, sigma_color, sigma_space)


# ---------------- enhancement (unchanged from gen_us.py) ----------------

def clahe_unsharp(img, clip_limit=1.3, tile=(8, 8), sigma=2.0, amount=1.28, subtract=0.28):
    cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(img)
    blur = cv2.GaussianBlur(cl, (0, 0), sigma)
    return cv2.addWeighted(cl, amount, blur, -subtract, 0)


def norm8(a, mask=None):
    a = np.clip(a, 0, 255).astype(np.uint8)
    if mask is not None:
        a = np.where(mask, a, 0).astype(np.uint8)
    return a


# name -> (function, kwargs) for the full 8-filter bank used by the tissue study
FILTER_BANK = {
    'Median': (median, {}),
    'Lee': (lee, {}),
    'Kuan': (kuan, {}),
    'Frost': (frost, {}),
    'Bilateral': (bilateral, {}),
    'SRAD': (srad, {}),
    'NLM': (nlm, {}),
    'Wavelet': (wavelet_despeckle, {}),
}
