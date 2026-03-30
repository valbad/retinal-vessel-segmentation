# Retinal Vessel Segmentation — LAD-OS Implementation

## Project Overview

This repository implements the **LAD-OS** (Locally Adaptive Derivative filter on Orientation Scores) retinal vessel segmentation algorithm described in:

> Zhang, J., Dashtbozorg, B., Bekkers, E., Pluim, J.P.W., Duits, R., & ter Haar Romeny, B.M. (2016).
> *Robust Retinal Vessel Segmentation via Locally Adaptive Derivative Frames in Orientation Scores.*
> IEEE Transactions on Medical Imaging, 35(12), 2631–2644.

The method is **fully unsupervised** and works by lifting 2D retinal images into a 3D orientation score space using cake wavelets, enhancing vessel cross-sections using adaptive derivative filters in that space, then projecting back and thresholding.

---

## Repository Structure

```
retinal_vessel_seg/
├── README.md
├── requirements.txt
├── setup.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Luminosity normalization, top-hat, geodesic opening
│   ├── cake_wavelets.py       # Cake wavelet construction and orientation score transform
│   ├── orientation_score.py   # OS transform: forward (Wψ) and inverse (Wψ*)
│   ├── lid_filter.py          # Left-Invariant Derivative (LID) filter
│   ├── lad_filter.py          # Locally Adaptive Derivative (LAD) filter
│   ├── segmentation.py        # Thresholding, MCC computation, binary map
│   └── utils.py               # FOV mask, dataset loaders, metric computation
│
├── data/
│   └── README.md              # Instructions for downloading DRIVE, STARE, etc.
│
├── configs/
│   ├── drive.yaml             # Dataset-specific parameters
│   ├── stare.yaml
│   ├── chase_db1.yaml
│   ├── hrf.yaml
│   ├── iostar.yaml
│   └── rc_slo.yaml
│
├── scripts/
│   ├── run_segmentation.py    # CLI entry point
│   └── evaluate.py            # Compute Se, Sp, Acc, AUC, MCC vs ground truth
│
├── notebooks/
│   └── demo.ipynb             # End-to-end demo on one image
│
└── tests/
    ├── test_preprocessing.py
    ├── test_cake_wavelets.py
    ├── test_filters.py
    └── test_segmentation.py
```

---

## Implementation Specifications

Detailed implementation specs are split across the following documents. **Read them in order:**

| File | Contents |
|------|----------|
| [`SPEC_01_preprocessing.md`](SPEC_01_preprocessing.md) | Luminosity normalization, top-hat, geodesic opening |
| [`SPEC_02_cake_wavelets.md`](SPEC_02_cake_wavelets.md) | Cake wavelet construction in the Fourier domain |
| [`SPEC_03_orientation_score.md`](SPEC_03_orientation_score.md) | Forward and inverse orientation score transforms |
| [`SPEC_04_lid_filter.md`](SPEC_04_lid_filter.md) | LID frame construction and LID-OS filter |
| [`SPEC_05_lad_filter.md`](SPEC_05_lad_filter.md) | Exponential curve fit, LAD frame, LAD-OS filter |
| [`SPEC_06_segmentation.md`](SPEC_06_segmentation.md) | Thresholding, MCC, binary map, evaluation metrics |
| [`SPEC_07_datasets_and_params.md`](SPEC_07_datasets_and_params.md) | Dataset details, physical pixel sizes, parameter tables |
| [`SPEC_08_pipeline_and_tests.md`](SPEC_08_pipeline_and_tests.md) | Full pipeline assembly, CLI, tests, expected results |

---

## Algorithm Summary

The full pipeline for one image is:

```
Input image f (RGB fundus or SLO)
    │
    ▼
[1] PREPROCESSING
    ├─ Extract green channel (RGB datasets only)
    ├─ Luminosity normalization (Foracchia et al. 2005)
    ├─ Geodesic opening (structuring element: disk, radius = Wt/2ρ px)
    └─ Top-hat transform (removes optic disk brightness + central reflex)
    │
    ▼
[2] ORIENTATION SCORE TRANSFORM  (cake wavelets, No orientations)
    f(x) ──Wψ──▶ U_f(x, θ_i)   for i = 1 … No
    │
    ▼
[3] VESSEL ENHANCEMENT  (multi-scale, scales S)
    For each scale σs ∈ S and each orientation θ_i:
    │
    ├── LID-OS:  Φ_η(U_f) = −μ⁻² ∂²_η G_{σs,σo} * U_f
    │
    └── LAD-OS:
        ├─ Compute left-invariant Hessian H_Uf (via LID derivatives)
        ├─ Eigendecompose H_μ to get optimal tangent vector c*
        ├─ Compute κ (curvature) and d_H (deviation from horizontality)
        ├─ Construct LAD frame {∂_a, ∂_b, ∂_c} from LID frame
        └─ Φ_b(U_f) = −μ⁻² ∂²_b G_{σs,σo} * U_f
    │
    ▼
[4] IMAGE RECONSTRUCTION
    Υ(f)(x) = max over θ_i { Σ_{σs∈S} Φ_{norm}(U_f)(x, θ_i) }
    │
    ▼
[5] HARD SEGMENTATION
    binary_map = Υ(f) > T_h
    (T_h chosen to maximize MCC on each dataset)
```

---

## Key Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| f | Input 2D image |
| U_f | Orientation score: 3D function on SE(2) = R² ⋊ S¹ |
| Wψ | Wavelet transform lifting f → U_f |
| ψ | Cake wavelet kernel |
| No | Number of orientation samples |
| θ_i = iπ/No | Discrete orientation angles |
| {∂_ξ, ∂_η, ∂_θ} | Left-invariant rotating derivative (LID) frame |
| {∂_a, ∂_b, ∂_c} | Locally adaptive derivative (LAD) frame |
| σs | Spatial Gaussian scale (pixels) |
| σo | Angular Gaussian scale (radians) |
| μ = σo/σs | Scale normalization factor (units: 1/length) |
| S | Set of spatial scales |
| κ | Local curvature of vessel |
| d_H | Deviation from horizontality |
| Th | Segmentation threshold |
| MCC | Matthews Correlation Coefficient |
| ρ | Physical pixel size (μm/px) |

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
scikit-image >= 0.21
matplotlib >= 3.7
pyyaml >= 6.0
tqdm >= 4.65
```

Optional (for notebook):
```
jupyter
```

---

## Quick Start (once implemented)

```bash
# Install
pip install -e .

# Run LAD-OS on DRIVE test set
python scripts/run_segmentation.py \
    --dataset drive \
    --data_dir /path/to/DRIVE \
    --method lad \
    --output_dir results/drive/

# Evaluate against ground truth
python scripts/evaluate.py \
    --pred_dir results/drive/ \
    --gt_dir /path/to/DRIVE/test/1st_manual \
    --mask_dir /path/to/DRIVE/test/mask
```

---

## Expected Performance (from paper, Table III & IV)

| Dataset | Method | Se | Sp | Acc | AUC | MCC |
|---------|--------|----|----|-----|-----|-----|
| DRIVE | LAD-OS | 0.7743 | 0.9725 | 0.9476 | 0.9636 | 0.7571 |
| STARE | LAD-OS | 0.7791† | 0.9758 | 0.9554 | 0.9748* | 0.7626* |
| CHASE_DB1 | LAD-OS | 0.7626 | 0.9661 | 0.9452 | 0.9606 | — |
| HRF | LAD-OS | 0.7978 | 0.9717 | 0.9556 | 0.9608 | 0.7410 |
| IOSTAR | LAD-OS | 0.7545 | 0.9740 | 0.9514 | 0.9615 | 0.7318 |
| RC-SLO | LAD-OS | 0.7787 | 0.9710 | 0.9512 | 0.9626 | 0.7327 |

*Best among unsupervised methods. †Best overall including supervised methods.
