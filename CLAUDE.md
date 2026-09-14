# Project Handoff: Post Image Processing in Ultrasound Simulation

This file gives Claude (running in this project folder via Claude Code) the
full context of work already done in a prior claude.ai conversation, so it can
continue seamlessly without the student re-explaining everything.

## Who this is for

**Student:** Darshil Maniya, Roll No. 262SP009
**Program:** M.Tech, Signal Processing and Machine Learning
**Department:** Electronics and Communication Engineering, NITK Surathkal
**Subject:** Seminar (Course Code: EC787)

## Project Context

NITK-UsoundSim is a 16-member student project building a complete software
ultrasound scanner, split into 8 two-person modules (Transducer, Transmit
Beamforming, Acoustic Propagation, Tissue Interaction, Receiver Beamforming,
B-Mode Image Formation, Image Reconstruction, Post Image Processing). Darshil
owns the final module: **Post Image Processing** — taking the raw speckled
B-mode image from the upstream team and returning a display-ready, despeckled
image.

At the time of this work, the upstream B-Mode Image Formation module had not
delivered production output, so a **synthetic B-mode testbench** was built
(`gen_us.py`) to simulate a realistic sector-format ultrasound image with
known ground truth, to develop and validate the pipeline against.

## What has been built so far

### 1. Core simulation & filtering (`gen_us.py`)
Generates a synthetic B-mode ultrasound phantom: complex Gaussian speckle
field, anisotropic PSF, depth attenuation, log compression, polar-to-fan scan
conversion. Contains an anechoic cyst, hyperechoic lesion, hypoechoic nodule,
and point targets. Implements and runs five despeckling filters: Median, Lee,
SRAD, Non-Local Means, Wavelet soft-thresholding. Outputs saved to `img/`.

### 2. Quality metrics (`metrics.py`)
Computes CNR, speckle SNR, PSNR, SSIM, EPI for each filter output against the
ground-truth reference. Results saved to `metrics.json`.

**Key measured results (from the testbench, NOT yet re-validated against real
upstream module output):**

| Method | CNR | Speckle SNR | EPI | SSIM | Speed |
|---|---|---|---|---|---|
| Original (raw) | 1.78 | 4.28 | 1.00 | 0.484 | — |
| Median | 2.45 | 7.07 | 0.759 | 0.625 | Very fast |
| Lee | 2.58 | 7.55 | 0.943 | 0.667 | Fast |
| Wavelet | 2.00 | 5.08 | 0.570 | 0.509 | Fast |
| NLM | 3.03 | 9.77 | 0.806 | 0.728 | Slow |
| **SRAD** | 2.98 | 9.29 | **0.955** | **0.744** | Moderate |

**SRAD was selected as the recommended default filter** — best balance of
noise suppression and edge preservation. This conclusion is used consistently
across all deliverables below.

### 3. Figure generation (`gen_figs.py`)
Produces all diagrams and result figures (25 PNGs in `fig/`) used in the
seminar PPT and the LaTeX report: pipeline diagrams, before/after filter
comparisons, the sliding-window explainer, metrics bar chart, etc.

### 4. Seminar PPT (`build_deck.py` → `262SP009_Darshil_Maniya_Seminar.pptx`)
23-slide deck in the NITK director's official template format (blue header
bar, NITK logo, Calibri font), aimed at non-domain-expert seminar faculty.
Plain-language explanations, real before/after images throughout.

### 5. Implementation summary PDF (`make_summary_pdf.py` →
`262SP009_Implementation_Summary.pdf`)
A 4-page standalone PDF summarizing the implementation plan for quick sharing
(e.g., with the project guide or teammates).

### 6. Full LaTeX project report (`report/main.tex`, `report/references.bib`,
`report/figures/`)
A complete submission-ready academic report, single-column, A4, Times font.

**Structure:** Title page → Declaration → Certificate → Abstract →
Acknowledgement → Table of Contents (own page) → List of Figures (own page)
→ List of Tables + Abbreviations (shared page) → 4 chapters (Introduction and
Background; Methodology and Implementation; Results and Discussion;
Conclusion and Future Work) → Bibliography (18 IEEE-format references).

**Current length: 26 pages** (target range: 20–25 preferred, 15–30 hard
limit — student wants it trimmed closer to 25 if possible without losing
technical content).

**Formatting decisions already locked in — do not re-litigate unless asked:**
- Title page branch: "Signal Processing and Machine Learning"
- Subject: Seminar, Course Code EC787
- Single-column, NOT IEEE two-column
- TOC / List of Figures / List of Tables+Abbreviations each get dedicated
  page treatment as specified above
- Margins: left 1.2in, right/top/bottom 0.95in
- SRAD is the recommended filter throughout — keep all chapters consistent
  with this conclusion if edited

**Known open item:** report is 26 pages; student asked to get it to 20–25.
Chapters were already merged once (6→4) and several redundant figures/tables
already removed to get from 44→26 pages. Further cuts should prioritize
trimming prose over removing remaining figures/equations, since the report
must stay "detailed and technically accurate."

## File Map

```
gen_us.py              — synthetic B-mode testbench + 5 filters (run first)
metrics.py              — quality metrics computation (run after gen_us.py)
gen_figs.py              — figure/diagram generation (run after metrics.py)
img/                    — raw simulation + filter output arrays/PNGs
fig/                    — generated diagrams and comparison figures
metrics.json            — computed metric values

build_deck.py            — builds the seminar PPTX from a NITK template
make_summary_pdf.py       — builds the 4-page implementation summary PDF

report/main.tex          — the full LaTeX report source
report/references.bib     — bibliography
report/figures/          — figures embedded in the report (copied from fig/, img/)
```

## Working conventions from the prior session

- All despeckling is **classical image processing only** — no machine
  learning / deep learning anywhere in the implementation (this is a
  stated scope boundary in the report; don't introduce ML methods without
  the student explicitly asking).
- Filter parameters are documented and fixed (see report Table 4.2 /
  methodology chapter): Median k=9, Lee k=11, SRAD 400 iterations dt=0.12,
  NLM patch 5x5 search radius 11 h=0.16, Wavelet Daubechies-4 4 levels.
- LaTeX build: `latexmk -pdf -interaction=nonstopmode main.tex` from inside
  `report/`. Check `main.log` for `Overfull \hbox` and `undefined` reference
  warnings after every build — the prior session treated both as required
  to fix before considering a build "done."
- Page-count discipline matters a lot to this student — always report the
  page count after any report edit.
- Python deps: numpy, scipy, opencv-python (cv2), scikit-image, PyWavelets,
  matplotlib, python-pptx, reportlab.

## Suggested next steps (not yet done)

- Re-validate the pipeline and metrics once real B-Mode Image Formation
  module output becomes available (currently synthetic-testbench-only).
- Try to trim report from 26 → 25 pages via prose compression rather than
  cutting more figures.
- Nothing else was explicitly requested as "next" at handoff time — ask the
  student what they want to work on.
