import numpy as np, json
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

import metrics_core as mc

raw = np.load('img/us_raw.npy').astype(float); gt = np.load('img/us_gt.npy').astype(float)
mask = np.load('img/mask.npy')
H, W = raw.shape
# ROI: lesion (dark cyst) and background
LES = mc.disk(H, W, int(0.36 * H), int(0.40 * W), 28)
BG = mc.disk(H, W, int(0.36 * H), int(0.66 * W), 28)

names = ['Median', 'Lee', 'SRAD', 'NLM', 'Wavelet']
rows = []
base = {'CNR': mc.cnr(raw, LES, BG), 'SNR': mc.snr_speckle(raw, BG), 'PSNR': psnr(gt, raw, data_range=255),
        'SSIM': ssim(gt, raw, data_range=255), 'EPI': 1.0}
rows.append(['Original (raw)', base])
for n in names:
    x = np.load(f'img/f_{n.lower()}.npy').astype(float)
    rows.append([n, {'CNR': mc.cnr(x, LES, BG), 'SNR': mc.snr_speckle(x, BG), 'PSNR': psnr(gt, x, data_range=255),
                      'SSIM': ssim(gt, x, data_range=255), 'EPI': mc.epi(x, gt, mask)}])
x = np.load('img/f_enhanced.npy').astype(float)
rows.append(['SRAD + CLAHE', {'CNR': mc.cnr(x, LES, BG), 'SNR': mc.snr_speckle(x, BG), 'PSNR': psnr(gt, x, data_range=255),
                               'SSIM': ssim(gt, x, data_range=255), 'EPI': mc.epi(x, gt, mask)}])
print(f"{'Method':16s} {'CNR':>6s} {'SNRs':>6s} {'PSNR':>6s} {'SSIM':>6s} {'EPI':>6s}")
for n, m in rows:
    print(f"{n:16s} {m['CNR']:6.2f} {m['SNR']:6.2f} {m['PSNR']:6.2f} {m['SSIM']:6.3f} {m['EPI']:6.3f}")
json.dump({n: m for n, m in rows}, open('metrics.json', 'w'), indent=1)
np.save('img/roi.npy', np.stack([LES, BG]))
