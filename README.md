# Retinal Vessel Segmentation — LAD-OS

Implementation of the **LAD-OS** (Locally Adaptive Derivative filter on Orientation Scores) retinal vessel segmentation algorithm from:

> Zhang, J., Dashtbozorg, B., Bekkers, E., Pluim, J.P.W., Duits, R., & ter Haar Romeny, B.M. (2016).
> *Robust Retinal Vessel Segmentation via Locally Adaptive Derivative Frames in Orientation Scores.*
> IEEE Transactions on Medical Imaging, 35(12), 2631–2644.

The method is **fully unsupervised**: retinal images are lifted to a 3D orientation score space using cake wavelets, vessel cross-sections are enhanced using adaptive derivative filters in that space, then projected back and thresholded.

---

## Repository Structure

```
retinal-vessel-segmentation/
├── src/
│   ├── preprocessing.py       # SPEC_01 — green channel, Foracchia normalisation, geodesic opening, black top-hat
│   ├── cake_wavelets.py       # SPEC_02 — cake wavelet bank (Fourier domain, B-spline × M_N radial window)
│   ├── orientation_score.py   # SPEC_03 — forward OS transform (Wψ) and inverse (Wψ*)
│   ├── lid_filter.py          # SPEC_04 — LID frame derivatives, LID-OS vessel enhancement
│   ├── lad_filter.py          # SPEC_05 — LI Hessian, H_μ eigendecomposition, LAD-OS enhancement
│   └── segmentation.py        # SPEC_06 — thresholding, confusion matrix, Se/Sp/Acc/MCC/AUC, ROC
│
└── notebooks/
    └── 01_preprocessing_visualization.ipynb   # end-to-end walkthrough on DRIVE
```

---

## Algorithm Pipeline

```
Input image f (RGB fundus)
    │
    ▼
[1] PREPROCESSING  (SPEC_01)
    ├─ Extract green channel
    ├─ Luminosity/contrast normalisation  (Foracchia et al. 2005, tile-based bicubic)
    ├─ Geodesic opening  (erosion + reconstruction by dilation, removes large bright structures)
    └─ Black top-hat  closing(I) − I  →  vessels appear bright
    │
    ▼
[2] ORIENTATION SCORE TRANSFORM  (SPEC_02 + SPEC_03)
    ├─ Build No=16 cake wavelets in Fourier domain
    │    ψ̂_i(ω) = B-spline angular window × M_N radial window  (double-sided, half-circle convention)
    │    Admissibility: Σ_i |ψ̂_i|² ≈ 1 in passband
    └─ Lift:  U_f(x, θ_i) = conj(ψ̂_i) · F̂[f]  (per orientation, via FFT)
    │
    ▼
[3] VESSEL ENHANCEMENT  (SPEC_04 + SPEC_05, multi-scale over S = {0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5} px)
    │
    ├── LID-OS:  Φ_η = −μ⁻² ∂²_η (G_{σs} ★ U_f)
    │    rotating frame: ∂_ξ = cosθ ∂_x + sinθ ∂_y,  ∂_η = −sinθ ∂_x + cosθ ∂_y
    │
    └── LAD-OS  (primary method):
         ├─ Build full 3×3 left-invariant Hessian H_Uf
         ├─ Scale-normalise: H_μ  (diagonal rescaling by μ = σo/σs)
         ├─ c* = smallest eigenvector of H_μ  →  local tangent direction
         ├─ κ = curvature,  d_H = deviation from horizontality
         ├─ e_b = [sin(d_H), cos(d_H), 0]  →  perpendicular-in-image direction
         └─ Φ_b = −μ⁻² · e_b^T H e_b
    │
    ▼
[4] RECONSTRUCTION
    Υ(f)(x) = max_{θ_i} { Σ_{σs ∈ S} Φ_norm(U_f)(x, θ_i) }
    │
    ▼
[5] SEGMENTATION  (SPEC_06)
    binary_map = Υ(f) > T_h
    T_h chosen to maximise MCC (pooled across training images)
```

---

## Results on DRIVE (5-fold CV, training set)

5-fold cross-validation on the 20 DRIVE training images. Each fold trains the threshold on 16 images and evaluates on the remaining 4.

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Se     | 0.596 | 0.062 | 0.402 | 0.668 |
| Sp     | 0.967 | 0.010 | 0.943 | 0.983 |
| Acc    | 0.920 | 0.013 | 0.883 | 0.935 |
| MCC    | 0.609 | 0.049 | 0.446 | 0.685 |
| AUC    | 0.894 | 0.029 | 0.798 | 0.933 |

**Paper targets (Table III, LAD-OS on DRIVE):** Se=0.774, Sp=0.973, Acc=0.948, AUC=0.964, MCC=0.757

> The gap reflects that results are computed on training images only (no held-out test labels in the Kaggle DRIVE release). CV-optimal thresholds (~0.18) differ from the paper's global T_h=0.554, likely due to normalisation differences.

---

## DRIVE Parameters (SPEC_07)

| Parameter | Value |
|-----------|-------|
| Pixel size ρ | 27 μm/px |
| Luminosity window W_l | 500 μm → 19 px |
| Top-hat kernel W_t | 150 μm → 6 px |
| Orientations N_o | 16 |
| Angular scale σ_o | π/40 |
| Spatial scales S | [0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5] px |
| Threshold T_h | 0.554 (paper) |

---

## Data

The DRIVE dataset is downloaded automatically via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download(
    'andrewmvd/drive-digital-retinal-images-for-vessel-extraction'
)
```

---

## Dependencies

```
numpy
scipy
scikit-image
matplotlib
imageio
kagglehub
```

---

## Notebook

[notebooks/01_preprocessing_visualization.ipynb](notebooks/01_preprocessing_visualization.ipynb) walks through the full pipeline step by step:

1. Load DRIVE dataset
2. Step-by-step preprocessing (green channel → Foracchia → geodesic opening → top-hat)
3. Intensity histograms at each stage
4. Cake wavelet bank (Fourier magnitude, spatial wavelets, admissibility)
5. Orientation score transform (16-layer grid, dominant orientation map, reconstruction)
6. LID-OS vessel enhancement
7. LAD-OS vessel enhancement (LID vs LAD comparison)
8. Segmentation at paper threshold + MCC-optimal threshold (single image)
9. 5-fold cross-validation: threshold distribution, prediction grid, summary metrics
