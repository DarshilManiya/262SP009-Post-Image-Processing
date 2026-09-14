"""Figures for the multi-tissue despeckling study (additive -- does not
touch gen_figs.py or anything under the plain fig/ directory)."""
import json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = "fig/tissues"; os.makedirs(F, exist_ok=True)
BLUE = "#2479C6"; ORANGE = "#ED7D31"; GREEN = "#70AD47"; GRAY = "#404040"; LGRAY = "#D9D9D9"
plt.rcParams["font.family"] = "DejaVu Sans"

TISSUES = ["generic", "liver", "kidney", "breast", "thyroid"]
TISSUE_LABELS = {"generic": "Generic", "liver": "Liver", "kidney": "Kidney",
                  "breast": "Breast", "thyroid": "Thyroid"}
FILTERS = ["Median", "Lee", "Kuan", "Frost", "Bilateral", "SRAD", "NLM", "Wavelet"]


def save(fig, name, dpi=180):
    fig.savefig(f"{F}/{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig); print("wrote", name)


def load_raw(name):
    path = "img/us_raw.png" if name == "generic" else f"img/tissues/{name}/raw.png"
    return cv2.imread(path, 0)


def load_filter(name, fname):
    key = fname.lower().replace(' + ', '_').replace(' ', '_')
    return cv2.imread(f"img/tissues/{name}/f_{key}.png", 0)


# ---------------- 1. raw tissue grid ----------------
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
for a, name in zip(axes, TISSUES):
    a.imshow(load_raw(name), cmap="gray"); a.axis("off")
    a.set_title(TISSUE_LABELS[name], fontsize=13, color=GRAY, fontweight="bold")
fig.suptitle("Synthetic phantoms across tissue types", fontsize=14, color=GRAY, fontweight="bold", y=1.05)
fig.tight_layout(); save(fig, "tissue_raw_grid")

# ---------------- 2. per-tissue filter comparison grids ----------------
for name in TISSUES:
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
    raw = load_raw(name)
    show = [(raw, "Original (raw)")] + [(load_filter(name, f), f) for f in FILTERS]
    for a, (im, t) in zip(axes.ravel(), show):
        a.imshow(im, cmap="gray"); a.axis("off")
        c = ORANGE if t == "SRAD" else GRAY
        a.set_title(t, fontsize=12, color=c, fontweight="bold", pad=4)
    fig.suptitle(f"{TISSUE_LABELS[name]} phantom — filter bank comparison",
                 fontsize=14, color=GRAY, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, f"filters_{name}")

# ---------------- 3. aggregated bar charts across tissue types ----------------
tm = json.load(open("study/tissue_metrics.json"))
for metric, ylabel in [("CNR", "Contrast-to-noise ratio"), ("SSIM", "Structural similarity")]:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(TISSUES)); width = 0.10
    all_names = ["Original (raw)"] + FILTERS + ["SRAD + CLAHE"]
    colors = [LGRAY] + [BLUE] * len(FILTERS) + [GREEN]
    for i, fname in enumerate(all_names):
        vals = [tm[t][fname][metric] for t in TISSUES]
        c = ORANGE if fname == "SRAD" else colors[i]
        ax.bar(x + (i - len(all_names) / 2) * width, vals, width, label=fname, color=c, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([TISSUE_LABELS[t] for t in TISSUES], fontsize=11, color=GRAY)
    ax.set_ylabel(ylabel, fontsize=11, color=GRAY)
    ax.set_title(f"{metric} across tissue types and filters", fontsize=13, color=GRAY, fontweight="bold")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save(fig, f"bar_{metric.lower()}_by_tissue")

# ---------------- 4. sensitivity curves ----------------
sens = json.load(open("study/sensitivity.json"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
srad = sens["srad_iterations"]
ns = [r["n"] for r in srad]
for metric, c in [("CNR", BLUE), ("SSIM", ORANGE), ("EPI", GREEN)]:
    axes[0].plot(ns, [r[metric] for r in srad], marker="o", color=c, label=metric, lw=2)
axes[0].set_xlabel("SRAD iterations", fontsize=11, color=GRAY)
axes[0].set_title("SRAD: quality vs. iteration count", fontsize=12.5, color=GRAY, fontweight="bold")
axes[0].legend(fontsize=10); axes[0].spines[["top", "right"]].set_visible(False)

med = sens["median_kernel"]
ks = [r["k"] for r in med]
for metric, c in [("CNR", BLUE), ("SSIM", ORANGE), ("EPI", GREEN)]:
    axes[1].plot(ks, [r[metric] for r in med], marker="o", color=c, label=metric, lw=2)
axes[1].set_xlabel("Median kernel size", fontsize=11, color=GRAY)
axes[1].set_title("Median: quality vs. kernel size", fontsize=12.5, color=GRAY, fontweight="bold")
axes[1].legend(fontsize=10); axes[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout(); save(fig, "sensitivity_curves")

# ---------------- 5. significance table ----------------
sig = json.load(open("study/significance.json"))
fig, ax = plt.subplots(figsize=(9, 4.2)); ax.axis("off")
rows = [f for f in FILTERS + ["SRAD + CLAHE"] if f != "SRAD"]
cell = [[f, f"{sig['vs_srad'][f]['CNR']['p']:.4f}", f"{sig['vs_srad'][f]['SSIM']['p']:.4f}"] for f in rows]
tbl = ax.table(cellText=cell, colLabels=["Filter vs. SRAD", "CNR p-value", "SSIM p-value"],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.8)
for j in range(3):
    tbl[(0, j)].set_facecolor(BLUE); tbl[(0, j)].set_text_props(color="white", fontweight="bold")
for i, f in enumerate(rows, start=1):
    sig_cnr = sig['vs_srad'][f]['CNR']['p'] < 0.05
    if sig_cnr:
        tbl[(i, 1)].set_facecolor("#E4F0DC")
ax.set_title("Paired t-test: SRAD vs. each filter (n=5 phantom instances)\n"
             "Shaded = statistically significant CNR difference (p < 0.05)",
             fontsize=12, color=GRAY, fontweight="bold", pad=14)
save(fig, "significance_table")

print("ALL TISSUE FIGURES DONE")
