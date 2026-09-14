import numpy as np, cv2, os

import sim_core as sc
import filters as flt

rng = np.random.default_rng(7)
OUT = "img"; os.makedirs(OUT, exist_ok=True)

NL, NS = 220, 520          # scan lines (angle), samples (depth)

# ---------- tissue reflectivity map in polar space ----------
r, th = sc.make_grid(NS, NL)
T = np.full((NS, NL), 1.0)

# layered tissue bands
T += 0.35 * np.sin(r * 22.0) * 0.5
T[r[:, 0] < 0.05, :] = 2.2                   # bright near-field skin layer

T[sc.ellipse(r, th, 0.42, -0.34, 0.13, 0.22)] = 0.04   # anechoic cyst
T[sc.ellipse(r, th, 0.60,  0.36, 0.12, 0.19)] = 4.5    # hyperechoic lesion
T[sc.ellipse(r, th, 0.28,  0.26, 0.05, 0.07)] = 5.5    # small target (hard to see)
T[sc.ellipse(r, th, 0.80, -0.12, 0.07, 0.11)] = 0.12   # hypoechoic nodule

# point scatterers for resolution
for (cy, cx) in [(0.16, -0.60), (0.16, 0.0), (0.16, 0.60)]:
    T[sc.ellipse(r, th, cy, cx, 0.006, 0.010)] = 5.0

# ---------- speckle: complex gaussian field * PSF, log compression ----------
PSF_AX, PSF_LAT, ATTEN, DR = 2.6, 3.4, 1.15, 40.0
env = sc.synthesize_envelope(T, rng, psf_ax=PSF_AX, psf_lat=PSF_LAT, attenuation=ATTEN)
polar = sc.log_compress(env, DR)

# ---------- scan conversion: polar -> fan (Cartesian) ----------
raw, mask = sc.scan_convert(polar)
cv2.imwrite(f"{OUT}/us_raw.png", raw)
np.save(f"{OUT}/us_raw.npy", raw); np.save(f"{OUT}/mask.npy", mask)

# ground-truth (noise-free) version for reference figures
gt, _ = sc.ground_truth_image(T, rng, psf_ax=PSF_AX, psf_lat=PSF_LAT, attenuation=ATTEN, dynamic_range=DR)
cv2.imwrite(f"{OUT}/us_gt.png", gt)
np.save(f"{OUT}/us_gt.npy", gt)

# =================== FILTERS ===================
results = {}
results['Median']  = flt.norm8(flt.median(raw, 9), mask)
results['Lee']      = flt.norm8(flt.lee(raw, 11), mask)
results['SRAD']     = flt.norm8(flt.srad(raw), mask)
results['NLM']      = flt.norm8(flt.nlm(raw), mask)
results['Wavelet']  = flt.norm8(flt.wavelet_despeckle(raw), mask)

for k, v in results.items():
    cv2.imwrite(f"{OUT}/f_{k.lower()}.png", v)
    np.save(f"{OUT}/f_{k.lower()}.npy", v)

# CLAHE + unsharp on the best (SRAD)
sharp = flt.clahe_unsharp(results['SRAD'])
final = flt.norm8(sharp, mask)
cv2.imwrite(f"{OUT}/f_enhanced.png", final)
np.save(f"{OUT}/f_enhanced.npy", final)
print("done", list(results))
