"""Shared quality-metric functions.

Extracted from the original metrics.py so the same CNR/SNR/EPI definitions
can be reused for the generic phantom and for the multi-tissue study without
duplication. PSNR/SSIM are used directly from skimage at the call site.
"""
import numpy as np
import cv2


def disk(H, W, cy, cx, r):
    yy, xx = np.mgrid[0:H, 0:W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def cnr(x, lesion_roi, bg_roi):
    a, b = x[lesion_roi], x[bg_roi]
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-9)


def snr_speckle(x, bg_roi):
    b = x[bg_roi]
    return b.mean() / (b.std() + 1e-9)


def epi(x, ref, mask):
    lx = cv2.Laplacian(x.astype(np.float32), cv2.CV_32F)
    lr = cv2.Laplacian(ref.astype(np.float32), cv2.CV_32F)
    lx = lx[mask] - lx[mask].mean()
    lr = lr[mask] - lr[mask].mean()
    return float(np.sum(lx * lr) / np.sqrt(np.sum(lx * lx) * np.sum(lr * lr) + 1e-12))
