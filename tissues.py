"""Multi-tissue synthetic B-mode phantom presets.

Each preset is a fully synthetic reflectivity map T (same speckle-synthesis
and scan-conversion machinery as the generic phantom in gen_us.py, via
sim_core.py) built to loosely reproduce a distinguishing B-mode
characteristic of that organ reported in the literature, rather than one
generic phantom reused for everything:

- Liver:   fairly homogeneous mid-level echogenicity, coarser speckle
           (larger PSF correlation length), one embedded hypoechoic
           fibrotic-lesion-like region.
- Kidney:  layered cortex (lower echogenicity) vs. medulla/sinus (higher
           echogenicity) -- the cortex/sinus echogenicity contrast is the
           basis of real renal ultrasound reads -- plus an embedded
           anechoic cyst.
- Breast:  heterogeneous fibroglandular background (higher speckle
           variance than liver/kidney) with an embedded hypoechoic mass
           (fibroadenoma-like).
- Thyroid: homogeneous fine-textured background (small PSF correlation
           length, shallow depth/low attenuation) with an embedded nodule
           showing a peripheral halo -- a commonly cited benign-nodule
           B-mode sign.

These are still synthetic testbench phantoms with known ground truth (per
the project's existing scope), not real patient data, but each targets a
distinct, documented echo-texture pattern instead of one generic phantom
reused for every organ.
"""
import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

import sim_core as sc

OUT = "img/tissues"
NL, NS = 220, 520  # same resolution as the generic phantom


def _base_grid():
    return sc.make_grid(NS, NL)


def _liver(rng):
    r, th = _base_grid()
    T = np.full((NS, NL), 1.0)
    T += 0.25 * np.sin(r * 14.0) * 0.5           # coarse, gentle layering
    T[r[:, 0] < 0.05, :] = 2.1                    # near-field skin/capsule
    lesion = (0.50, 0.05, 0.16, 0.24)
    T[sc.ellipse(r, th, *lesion)] = 0.10            # hypoechoic fibrotic lesion
    bg = (0.50, -0.55, 0.08, 0.10)
    params = dict(psf_ax=3.2, psf_lat=4.2, attenuation=1.3, clutter=0.05, dynamic_range=42.0)
    return T, r, th, lesion, bg, params


def _kidney(rng):
    r, th = _base_grid()
    T = np.full((NS, NL), 1.0)
    T[r[:, 0] < 0.05, :] = 2.0                     # skin
    kidney_capsule = sc.ellipse(r, th, 0.55, 0.0, 0.34, 0.32)
    T[kidney_capsule] = 0.55                       # cortex (lower echogenicity)
    T[sc.ellipse(r, th, 0.55, 0.0, 0.15, 0.14)] = 3.4   # medulla/sinus (hyperechoic)
    lesion = (0.38, -0.20, 0.09, 0.11)
    T[sc.ellipse(r, th, *lesion)] = 0.03            # renal cyst (anechoic)
    bg = (0.68, 0.20, 0.06, 0.08)                   # plain cortex, away from cyst/sinus
    params = dict(psf_ax=2.4, psf_lat=3.0, attenuation=1.0, clutter=0.06, dynamic_range=38.0)
    return T, r, th, lesion, bg, params


def _breast(rng):
    r, th = _base_grid()
    T = np.full((NS, NL), 1.0)
    # heterogeneous fibroglandular background: smooth low-frequency texture
    low_freq = gaussian_filter(rng.normal(size=(NS, NL)), (40, 30))
    low_freq = 1.0 + 0.6 * (low_freq / (np.abs(low_freq).max() + 1e-9))
    T *= low_freq
    T[r[:, 0] < 0.04, :] = 2.3                     # skin
    lesion = (0.45, 0.10, 0.13, 0.18)
    T[sc.ellipse(r, th, *lesion)] = 0.14            # hypoechoic mass (fibroadenoma-like)
    bg = (0.45, -0.55, 0.09, 0.11)
    params = dict(psf_ax=2.2, psf_lat=2.8, attenuation=0.9, clutter=0.07, dynamic_range=36.0)
    return T, r, th, lesion, bg, params


def _thyroid(rng):
    r, th = _base_grid()
    T = np.full((NS, NL), 1.0)
    T[r[:, 0] < 0.03, :] = 2.0                     # skin (shallow structure)
    T[sc.ellipse(r, th, 0.35, 0.0, 0.16, 0.18)] = 1.9   # peripheral halo (benign sign)
    lesion = (0.35, 0.0, 0.10, 0.12)
    T[sc.ellipse(r, th, *lesion)] = 0.20            # hypoechoic nodule core
    bg = (0.35, 0.55, 0.08, 0.10)
    params = dict(psf_ax=1.8, psf_lat=2.2, attenuation=0.8, clutter=0.05, dynamic_range=34.0)
    return T, r, th, lesion, bg, params


# name -> (builder, seed)
TISSUE_PRESETS = {
    'liver':   (_liver, 101),
    'kidney':  (_kidney, 102),
    'breast':  (_breast, 103),
    'thyroid': (_thyroid, 104),
}


def generate_tissue(name, save=True):
    """Build one tissue preset end-to-end: reflectivity map -> speckle
    synthesis -> log compression -> scan conversion -> ground truth ->
    pixel-space lesion/background ROIs. Returns a dict of arrays/ROIs and
    (if save=True) writes them under img/tissues/<name>/."""
    if name not in TISSUE_PRESETS:
        raise ValueError(f"unknown tissue preset '{name}', choose from {list(TISSUE_PRESETS)}")
    builder, seed = TISSUE_PRESETS[name]
    rng = np.random.default_rng(seed)
    T, r, th, lesion_polar, bg_polar, params = builder(rng)

    dr = params.pop('dynamic_range')
    env = sc.synthesize_envelope(T, rng, **{k: params[k] for k in ('psf_ax', 'psf_lat', 'attenuation', 'clutter')})
    polar = sc.log_compress(env, dr)
    raw, mask = sc.scan_convert(polar)
    gt, _ = sc.ground_truth_image(T, rng, psf_ax=params['psf_ax'], psf_lat=params['psf_lat'],
                                   attenuation=params['attenuation'], dynamic_range=dr)

    lesion_roi = sc.region_pixel_roi(r, th, *lesion_polar)
    bg_roi = sc.region_pixel_roi(r, th, *bg_polar)

    out = dict(name=name, raw=raw, gt=gt, mask=mask, lesion_roi=lesion_roi, bg_roi=bg_roi)

    if save:
        d = f"{OUT}/{name}"
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(f"{d}/raw.png", raw); np.save(f"{d}/raw.npy", raw)
        cv2.imwrite(f"{d}/gt.png", gt); np.save(f"{d}/gt.npy", gt)
        np.save(f"{d}/mask.npy", mask)
        np.save(f"{d}/roi.npy", np.array([lesion_roi, bg_roi]))

    return out


if __name__ == "__main__":
    for name in TISSUE_PRESETS:
        r = generate_tissue(name)
        print(f"{name:8s} lesion_roi={r['lesion_roi']}  bg_roi={r['bg_roi']}")
