"""Multi-tissue despeckling study.

Runs the full 8-filter bank (Median, Lee, Kuan, Frost, Bilateral, SRAD, NLM,
Wavelet) + SRAD+CLAHE enhancement across 5 phantom instances -- the existing
generic phantom plus the 4 tissue-type presets in tissues.py -- computes
CNR/SNR/PSNR/SSIM/EPI for each, benchmarks real per-filter runtime, then:

  - runs a paired t-test (SciPy) across the 5 phantom instances comparing
    each filter to the raw input and to SRAD, on CNR and SSIM
  - sweeps SRAD's iteration count and Median's kernel size on the generic
    phantom (which has known ground truth) to show how each filter's
    quality/parameter tradeoff behaves

All outputs are written under study/*.json and img/tissues/<name>/f_*.png,
purely additive -- nothing under img/*.png (the original generic-phantom
outputs) is touched.
"""
import os
import json
import time
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy import stats

import filters as flt
import metrics_core as mc
import tissues as ts

STUDY = "study"
os.makedirs(STUDY, exist_ok=True)


def run_filters_on(raw, mask):
    """Run the full filter bank + SRAD-CLAHE enhancement, returning
    (outputs, per-filter wall-clock seconds)."""
    results, timings = {}, {}
    for name, (fn, kwargs) in flt.FILTER_BANK.items():
        t0 = time.perf_counter()
        out = flt.norm8(fn(raw, **kwargs), mask)
        timings[name] = time.perf_counter() - t0
        results[name] = out

    t0 = time.perf_counter()
    enh = flt.norm8(flt.clahe_unsharp(results['SRAD']), mask)
    timings['SRAD + CLAHE'] = time.perf_counter() - t0
    results['SRAD + CLAHE'] = enh
    return results, timings


def compute_metrics(raw, gt, mask, LES, BG, results):
    raw_f, gt_f = raw.astype(float), gt.astype(float)

    def one(x):
        return {'CNR': mc.cnr(x, LES, BG), 'SNR': mc.snr_speckle(x, BG),
                'PSNR': psnr(gt_f, x, data_range=255),
                'SSIM': ssim(gt_f, x, data_range=255),
                'EPI': mc.epi(x, gt_f, mask)}

    out = {'Original (raw)': one(raw_f)}
    out['Original (raw)']['EPI'] = 1.0  # convention: raw is the EPI reference point
    for name, x in results.items():
        out[name] = one(x.astype(float))
    return out


def significance_study(tissue_metrics):
    filter_names = list(flt.FILTER_BANK.keys()) + ['SRAD + CLAHE']
    tissue_names = list(tissue_metrics.keys())

    def series(filter_name, metric):
        return np.array([tissue_metrics[t][filter_name][metric] for t in tissue_names])

    raw_series = {m: series('Original (raw)', m) for m in ('CNR', 'SSIM')}
    srad_series = {m: series('SRAD', m) for m in ('CNR', 'SSIM')}

    results = {'vs_original': {}, 'vs_srad': {}}
    for f in filter_names:
        fs = {m: series(f, m) for m in ('CNR', 'SSIM')}
        results['vs_original'][f] = {
            m: dict(zip(('t', 'p'), map(float, stats.ttest_rel(fs[m], raw_series[m]))))
            for m in ('CNR', 'SSIM')
        }
        if f != 'SRAD':
            results['vs_srad'][f] = {
                m: dict(zip(('t', 'p'), map(float, stats.ttest_rel(srad_series[m], fs[m]))))
                for m in ('CNR', 'SSIM')
            }
    results['note'] = (
        "Paired t-test across n=5 synthetic phantom instances (generic + liver + kidney + "
        "breast + thyroid). Indicative only given the small, fully synthetic sample size -- "
        "not a clinical-scale validation."
    )
    return results


def sensitivity_study(raw, gt, mask, LES, BG):
    gt_f = gt.astype(float)

    def score(x):
        xf = x.astype(float)
        return {'CNR': mc.cnr(xf, LES, BG), 'EPI': mc.epi(xf, gt_f, mask),
                'SSIM': ssim(gt_f, xf, data_range=255)}

    srad_sweep = []
    for n in (50, 100, 200, 400, 600, 800):
        x = flt.norm8(flt.srad(raw, n=n), mask)
        srad_sweep.append({'n': n, **score(x)})

    median_sweep = []
    for k in (3, 5, 7, 9, 11, 13, 15):
        x = flt.norm8(flt.median(raw, k), mask)
        median_sweep.append({'k': k, **score(x)})

    return {'srad_iterations': srad_sweep, 'median_kernel': median_sweep}


def save_tissue_filters(name, results):
    d = f"img/tissues/{name}"
    os.makedirs(d, exist_ok=True)
    for fname, arr in results.items():
        key = fname.lower().replace(' + ', '_').replace(' ', '_')
        cv2.imwrite(f"{d}/f_{key}.png", arr)
        np.save(f"{d}/f_{key}.npy", arr)


def main():
    tissue_metrics = {}
    runtime = {}

    # ---- generic phantom (reuse gen_us.py's existing output, unmodified) ----
    raw = np.load('img/us_raw.npy'); gt = np.load('img/us_gt.npy'); mask = np.load('img/mask.npy')
    roi = np.load('img/roi.npy')  # boolean [LES, BG], written by metrics.py
    LES, BG = roi[0], roi[1]

    results, timings = run_filters_on(raw, mask)
    tissue_metrics['generic'] = compute_metrics(raw, gt, mask, LES, BG, results)
    runtime['generic'] = timings
    save_tissue_filters('generic', results)

    # ---- 4 tissue-type presets ----
    for name in ts.TISSUE_PRESETS:
        t = ts.generate_tissue(name)
        H, W = t['raw'].shape
        ly, lx, lr = t['lesion_roi']; by, bx, br = t['bg_roi']
        LESt, BGt = mc.disk(H, W, ly, lx, lr), mc.disk(H, W, by, bx, br)

        res, tim = run_filters_on(t['raw'], t['mask'])
        tissue_metrics[name] = compute_metrics(t['raw'], t['gt'], t['mask'], LESt, BGt, res)
        runtime[name] = tim
        save_tissue_filters(name, res)
        print(f"{name}: done")

    with open(f"{STUDY}/tissue_metrics.json", "w") as f:
        json.dump(tissue_metrics, f, indent=1)
    with open(f"{STUDY}/runtime_benchmark.json", "w") as f:
        json.dump(runtime, f, indent=1)

    sig = significance_study(tissue_metrics)
    with open(f"{STUDY}/significance.json", "w") as f:
        json.dump(sig, f, indent=1)

    sens = sensitivity_study(raw, gt, mask, LES, BG)
    with open(f"{STUDY}/sensitivity.json", "w") as f:
        json.dump(sens, f, indent=1)

    print("tissue study done:", list(tissue_metrics.keys()))


if __name__ == "__main__":
    main()
