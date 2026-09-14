"""Shared synthetic B-mode simulation primitives.

Extracted from the original gen_us.py so the same speckle-synthesis and
scan-conversion machinery can be reused across the generic phantom
(gen_us.py) and the tissue-type presets (tissues.py) without duplication.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def make_grid(NS, NL):
    """Normalized polar-space depth (r) and lateral (th) coordinate grids."""
    th = np.linspace(-1, 1, NL)[None, :]
    r = np.linspace(0, 1, NS)[:, None]
    return r, th


def ellipse(r, th, cy, cx, ry, rx):
    return ((r - cy) / ry) ** 2 + ((th - cx) / rx) ** 2 <= 1.0


def synthesize_envelope(T, rng, psf_ax=2.6, psf_lat=3.4, attenuation=1.15,
                         clutter=0.06, clutter_decay=6.0):
    """Complex Gaussian speckle field * tissue reflectivity, blurred by an
    anisotropic PSF, then depth-attenuated with a touch of near-field clutter.
    Returns the envelope-detected (magnitude) image."""
    NS, NL = T.shape
    r, _ = make_grid(NS, NL)
    field = (rng.normal(size=(NS, NL)) + 1j * rng.normal(size=(NS, NL))) * np.sqrt(T)
    rf = (gaussian_filter(field.real, (psf_ax, psf_lat))
          + 1j * gaussian_filter(field.imag, (psf_ax, psf_lat)))
    env = np.abs(rf)
    env *= np.exp(-attenuation * r)
    env += clutter * np.abs(rng.normal(size=(NS, NL))) * np.exp(-clutter_decay * r)
    return env


def log_compress(env, dynamic_range=40.0):
    """Envelope -> 8-bit log-compressed image, normalized to its own max."""
    env = env / env.max()
    logimg = 20 * np.log10(env + 1e-4)
    b = np.clip((logimg + dynamic_range) / dynamic_range, 0, 1)
    return (b * 255).astype(np.uint8)


def scan_convert(p, H=560, W=880, ang=76.0, apex=0.28):
    """Polar (depth x angle) image -> Cartesian sector/fan image + validity mask."""
    NSr, NLr = p.shape
    half = np.radians(ang / 2.0)
    xmax = np.sin(half); ymin = apex * np.cos(half); ymax = 1.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    X = (xx / (W - 1)) * 2 * xmax - xmax
    Y = ymin + (yy / (H - 1)) * (ymax - ymin)
    rad = np.sqrt(X * X + Y * Y)
    ta = np.arctan2(X, Y)
    rr = (rad - apex) / (1.0 - apex) * (NSr - 1)
    cc = (ta / half + 1.0) / 2.0 * (NLr - 1)
    m = (rr >= 0) & (rr <= NSr - 1) & (np.abs(ta) <= half)
    cc = np.clip(cc, 0, NLr - 1)
    out = np.zeros((H, W))
    out[m] = map_coordinates(p.astype(np.float64), [rr[m], cc[m]], order=1)
    return out.astype(np.uint8), m


def region_pixel_roi(r, th, cy, cx, ry, rx, H=560, W=880, ang=76.0, apex=0.28):
    """Map a polar-space ellipse (as used to carve out a tissue structure in
    T) to a Cartesian-space (row, col, radius) ROI, by scan-converting the
    ellipse's indicator mask and taking the centroid of the result. Avoids
    hand-computing the polar->fan trigonometry for every tissue preset."""
    m = (ellipse(r, th, cy, cx, ry, rx).astype(np.uint8)) * 255
    cart, _ = scan_convert(m, H=H, W=W, ang=ang, apex=apex)
    ys, xs = np.where(cart > 127)
    if len(ys) == 0:
        raise ValueError("region maps entirely outside the visible sector")
    py, px = int(ys.mean()), int(xs.mean())
    radius = max(6, int(np.sqrt(len(ys) / np.pi) * 0.6))  # shrink to stay inside the region
    return py, px, radius


def ground_truth_image(T, rng, psf_ax=2.6, psf_lat=3.4, attenuation=1.15,
                        dynamic_range=40.0):
    """Noise-free (no speckle) reference: same PSF blur and attenuation, but
    built from sqrt(T) directly instead of a random speckle field."""
    NS, NL = T.shape
    r, _ = make_grid(NS, NL)
    amp = np.sqrt(gaussian_filter(T, (psf_ax, psf_lat))) * np.exp(-attenuation * r)
    polar = log_compress(amp, dynamic_range)
    return scan_convert(polar)
