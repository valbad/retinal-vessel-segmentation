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
│   ├── preprocessing.py       # green channel, Foracchia normalisation, geodesic opening, black top-hat
│   ├── cake_wavelets.py       # cake wavelet bank (Fourier domain, B-spline × M_N radial window)
│   ├── orientation_score.py   # forward OS transform (Wψ) and inverse (Wψ*)
│   ├── lid_filter.py          # LID frame derivatives, LID-OS vessel enhancement
│   ├── lad_filter.py          # LI Hessian, H_μ eigendecomposition, LAD-OS enhancement
│   └── segmentation.py        # thresholding, confusion matrix, Se/Sp/Acc/MCC/AUC, ROC
│
└── notebooks/
    └── 01_preprocessing_visualization.ipynb   # end-to-end walkthrough on DRIVE
```

---

## Algorithm Pipeline

### 1. Preprocessing

- Extract green channel from RGB fundus image $f$
- Luminosity/contrast normalisation (Foracchia et al. 2005, tile-based bicubic)
- Geodesic opening (erosion + reconstruction by dilation, removes large bright structures)
- Black top-hat: $I_{\text{th}} = \operatorname{closing}(I) - I$ → vessels appear bright

### 2. Orientation Score Transform

Build $N_o = 16$ cake wavelets in the Fourier domain:

$$\hat{\psi}_i(\boldsymbol{\omega}) = B_{\text{spline}}(\text{angular}) \times M_N(\text{radial}), \qquad \sum_i |\hat{\psi}_i|^2 \approx 1 \text{ (admissibility)}$$

Lift to orientation score space:

$$U_f(\mathbf{x},\, \theta_i) = \mathcal{F}^{-1}\!\left[\,\overline{\hat{\psi}_i}\cdot\hat{f}\,\right](\mathbf{x})$$

### 3. Vessel Enhancement &emsp; <sub>multi-scale over $S = \{0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5\}$ px</sub>

**LID-OS** — rotating frame derivatives:

$$\partial_\xi = \cos\theta\,\partial_x + \sin\theta\,\partial_y, \qquad \partial_\eta = -\sin\theta\,\partial_x + \cos\theta\,\partial_y$$

$$\Phi_\eta = -\mu^{-2}\,\partial^2_\eta\!\left(G_{\sigma_s} \star U_f\right)$$

**LAD-OS** (primary method):

1. Build full $3\times 3$ left-invariant Hessian $H_{U_f}$
2. Scale-normalise: $H_\mu$ with diagonal rescaling $\mu = \sigma_o / \sigma_s$
3. $\mathbf{c}^* =$ smallest eigenvector of $H_\mu$ → local vessel tangent direction
4. $\kappa =$ curvature, $\;d_H =$ deviation from horizontality
5. $\mathbf{e}_b = \bigl[\sin(d_H),\;\cos(d_H),\;0\bigr]^\top$ — perpendicular-in-image direction
6. $\Phi_b = -\mu^{-2}\,\mathbf{e}_b^\top H\,\mathbf{e}_b$

### 4. Reconstruction

$$\Upsilon(f)(\mathbf{x}) = \max_{\theta_i} \left\{ \sum_{\sigma_s \in S} \Phi_{\text{norm}}(U_f)(\mathbf{x},\,\theta_i) \right\}$$

### 5. Segmentation

$$\text{binary\_map} = \mathbf{1}\!\left[\,\Upsilon(f) > T_h\,\right]$$

$T_h$ is chosen to maximise MCC pooled across training images.

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

> The gap reflects that results are computed on training images only (no held-out test labels in the Kaggle DRIVE release). CV-optimal thresholds (~0.18) differ from the paper's global $T_h$=0.554, likely due to normalisation differences.

---

## DRIVE Parameters

| Parameter | Value |
|-----------|-------|
| Pixel size $\rho$ | 27 μm/px |
| Luminosity window $W_l$ | 500 μm → 19 px |
| Top-hat kernel $W_t$ | 150 μm → 6 px |
| Orientations $N_o$ | 16 |
| Angular scale $\sigma_o$ | $\pi/40$ |
| Spatial scales $S$ | [0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5] px |
| Threshold $T_h$ | 0.554 (paper) |

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
