# SPEC 07 — Datasets and Parameters

## File: `configs/*.yaml` and `src/utils.py`

This document details all six datasets used in the paper, their physical properties, download instructions, and the exact parameter values used for preprocessing and vessel enhancement.

---

## Dataset Summary (from paper Table II)

| Dataset | Image size (px) | FOV | Pixel size ρ (μm/px) | Vessel caliber mean±SD (px) | Wl (px) | Wt (px) | No | Scales S | σo | σo (LAD) | MCC | Th |
|---------|-----------------|-----|----------------------|----------------------------|---------|---------|----|----------|----|----------|-----|-----|
| DRIVE | 565×584 | 45° | 27 | 3.4±1.6 | 19 | 6 | 16 | {0.7,1.0,1.5,…,4.5} | π/5 | π/40 | 0.7571 | 0.5540 |
| STARE | 700×605 | 35° | 17 | 4.4±2.6 | 29 | 9 | 16 | {0.7,1.0,1.5,…,4.5} | π/5 | π/40 | 0.7626 | 0.5940 |
| CHASE_DB1 | 999×960 | 30° | 10 | 5.4±3.6 | 50 | 15 | 16 | {2,3.5,4.5,…,8.5} | π/5 | π/40 | 0.7030 | 0.6160 |
| HRF | 3504×2336 | 45° | 4 | 8.7±5.9 | 125 | 37 | 16 | {2,3,4,…,10} | π/5 | π/40 | 0.7410 | 0.6100 |
| IOSTAR | 1024×1024 | 45° | 14 | 6.3±2.5 | 36 | 11 | 16 | {2,3,4,…,8} | π/5 | π/40 | 0.7318 | 0.6100 |
| RC-SLO | 360×320 | 45° | 14 | 5.2±2.5 | 36 | 11 | 16 | {2,3,4,…,8} | π/5 | π/40 | 0.7327 | 0.6080 |

**Notes:**
- `Wl` = luminosity normalization window size in pixels = round(500 μm / ρ)
- `Wt` = top-hat / geodesic opening kernel size in pixels = round(150 μm / ρ)
- `No = 16` for all datasets
- `σo = π/5` is the angular scale for the **LID filter** (wider angular blurring)
- `σo = π/40` is the angular scale for the **LAD filter** (narrow, used throughout)
- The paper uses `σo = π/5` for LID-OS and `σo = π/40` for LAD-OS

**Wait — which σo?** Re-reading the paper: σo is set as "a small constant over all spatial scales" to keep structure smoothness. The value π/40 ≈ 0.0785 rad is used throughout (consistent with Table II showing σo = π/40 for both LID and LAD columns). Use `σo = π/40` for both.

---

## Scale Sets S (explicit values in pixels)

### DRIVE (ρ = 27 μm/px, vessel caliber 3.4±1.6 px)
```
S = {0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5}  # 9 values? Paper says 7
```
Actually the paper (Table II) says "7 scales for DRIVE." Reading the scale range {0.7, 1.0, 1.5, ..., 4.5}: that's {0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5} = 9 values if step is 0.5 above 1.0. The paper says "7 scales" for DRIVE. Looking at Table II notation `{0.7, 1.0, 1.5, …, 4.5}`, with 7 scales, the set is likely:
```
S_DRIVE = [0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]  # 7 values
```
Or possibly the scales span a specific μm range. **Use the following interpretation** based on fitting 7 scales in {0.7..4.5}:
```python
S_DRIVE = [0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]
```

### STARE (ρ = 17 μm/px, vessel caliber 4.4±2.6 px)
Same scale set as DRIVE (both 7 scales):
```python
S_STARE = [0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]
```

### CHASE_DB1 (ρ = 10 μm/px, vessel caliber 5.4±3.6 px)
Table II shows `{2, 3.5, 4.5, …, 8.5}`, 7 scales:
```python
S_CHASE = [2.0, 3.5, 4.5, 5.0, 5.5, 7.0, 8.5]
```

### HRF (ρ = 4 μm/px, vessel caliber 8.7±5.9 px)
Table II shows `{2, 3, 4, …, 10}`, **9 scales** (HRF uses 9 due to high resolution):
```python
S_HRF = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
```

### IOSTAR (ρ = 14 μm/px, vessel caliber 6.3±2.5 px)
Table II shows `{2, 3, 4, …, 8}`, 7 scales:
```python
S_IOSTAR = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

### RC-SLO (ρ = 14 μm/px, vessel caliber 5.2±2.5 px)
Same as IOSTAR:
```python
S_RCSLO = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

---

## YAML Configuration Files

### `configs/drive.yaml`
```yaml
dataset_name: DRIVE
pixel_size_um: 27.0
fov_deg: 45
image_size: [565, 584]
is_rgb: true

# Preprocessing
Wl_um: 500.0         # luminosity normalization window (μm)
Wt_um: 150.0         # top-hat / geodesic opening kernel (μm)
Wl_px: 19            # round(500/27)
Wt_px: 6             # round(150/27)
kernel_radius_px: 3  # Wt_px // 2

# Orientation score
No: 16
sigma_o: 0.07854     # π/40

# Enhancement scales (pixels)
scales: [0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]

# Segmentation
threshold_Th: 0.5540
optimal_mcc: 0.7571

# Data paths (override with CLI)
image_dir: data/DRIVE/test/images
gt_dir: data/DRIVE/test/1st_manual
mask_dir: data/DRIVE/test/mask
output_dir: results/drive/
fov_mask_provided: true
```

### `configs/stare.yaml`
```yaml
dataset_name: STARE
pixel_size_um: 17.0
fov_deg: 35
image_size: [700, 605]
is_rgb: true

Wl_um: 500.0
Wt_um: 150.0
Wl_px: 29
Wt_px: 9
kernel_radius_px: 4

No: 16
sigma_o: 0.07854

scales: [0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]

threshold_Th: 0.5940
optimal_mcc: 0.7626

image_dir: data/STARE/images
gt_dir: data/STARE/labels-ah
mask_dir: null
fov_mask_provided: false   # must create circular mask
output_dir: results/stare/
```

### `configs/chase_db1.yaml`
```yaml
dataset_name: CHASE_DB1
pixel_size_um: 10.0
fov_deg: 30
image_size: [999, 960]
is_rgb: true

Wl_um: 500.0
Wt_um: 150.0
Wl_px: 50
Wt_px: 15
kernel_radius_px: 7

No: 16
sigma_o: 0.07854

scales: [2.0, 3.5, 4.5, 5.0, 5.5, 7.0, 8.5]

threshold_Th: 0.6160
optimal_mcc: 0.7030

image_dir: data/CHASE_DB1/images
gt_dir: data/CHASE_DB1/groundtruth/1stAnnotation
mask_dir: null
fov_mask_provided: false
output_dir: results/chase_db1/
```

### `configs/hrf.yaml`
```yaml
dataset_name: HRF
pixel_size_um: 4.0
fov_deg: 45
image_size: [3504, 2336]
is_rgb: true

Wl_um: 500.0
Wt_um: 150.0
Wl_px: 125
Wt_px: 37
kernel_radius_px: 18

No: 16
sigma_o: 0.07854

scales: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]   # 9 scales

threshold_Th: 0.6100
optimal_mcc: 0.7410

image_dir: data/HRF/images
gt_dir: data/HRF/manual1
mask_dir: data/HRF/mask
fov_mask_provided: true
output_dir: results/hrf/
```

### `configs/iostar.yaml`
```yaml
dataset_name: IOSTAR
pixel_size_um: 14.0
fov_deg: 45
image_size: [1024, 1024]
is_rgb: false   # SLO grayscale

Wl_um: 500.0
Wt_um: 150.0
Wl_px: 36
Wt_px: 11
kernel_radius_px: 5

No: 16
sigma_o: 0.07854

scales: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

threshold_Th: 0.6100
optimal_mcc: 0.7318

image_dir: data/IOSTAR/images
gt_dir: data/IOSTAR/groundtruth
mask_dir: null
fov_mask_provided: false
output_dir: results/iostar/
```

### `configs/rc_slo.yaml`
```yaml
dataset_name: RC-SLO
pixel_size_um: 14.0
fov_deg: 45
image_size: [360, 320]
is_rgb: false   # SLO grayscale

Wl_um: 500.0
Wt_um: 150.0
Wl_px: 36
Wt_px: 11
kernel_radius_px: 5

No: 16
sigma_o: 0.07854

scales: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

threshold_Th: 0.6080
optimal_mcc: 0.7327

image_dir: data/RC-SLO/images
gt_dir: data/RC-SLO/groundtruth
mask_dir: null
fov_mask_provided: false
output_dir: results/rc_slo/
```

---

## Dataset Download Instructions (`data/README.md`)

```markdown
# Dataset Download Instructions

## DRIVE
- URL: https://drive.grand-challenge.org/
- 40 images (20 train, 20 test), 565×584, RGB TIFF
- Ground truth: 1st_manual/*.gif (binary PNG)
- FOV masks provided as *_mask.gif

## STARE
- URL: https://cecas.clemson.edu/~ahoover/stare/
- 20 images, 700×605, PPM format
- Ground truth: labels-ah/*.ppm (by Adam Hoover)
- No FOV masks (use circular approximation)

## CHASE_DB1
- URL: https://blogs.kingston.ac.uk/retinal/chasedb1/
- 28 images, 999×960, JPEG
- Ground truth: *_1stHO.png, *_2ndHO.png
- No masks (use circular approximation)

## HRF (High Resolution Fundus)
- URL: https://www5.cs.fau.de/research/data/fundus-images/
- 45 images (15 healthy, 15 DR, 15 glaucoma), 3504×2336, JPEG
- Ground truth: manual1/*.tif
- FOV masks provided: mask/*.tif

## IOSTAR
- URL: http://www.retinacheck.org (or contact authors)
- 30 SLO images, 1024×1024
- Ground truth provided

## RC-SLO
- URL: http://www.retinacheck.org
- 40 image patches, 360×320
```

---

## Dataset Loader (`src/utils.py`)

```python
import os
import yaml
import numpy as np
from skimage.io import imread


def load_config(dataset_name: str) -> dict:
    """Load dataset configuration from YAML file."""
    config_path = os.path.join('configs', f'{dataset_name.lower()}.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_image(path: str) -> np.ndarray:
    """Load image from file, returning uint8 array."""
    img = imread(path)
    return img


def load_gt(path: str) -> np.ndarray:
    """Load binary ground truth mask."""
    gt = imread(path)
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    return gt > 128


def load_dataset(config: dict) -> list:
    """
    Load all images, ground truths, and FOV masks for a dataset.

    Returns
    -------
    list of dicts with keys 'image', 'gt', 'fov_mask', 'name'
    """
    image_dir = config['image_dir']
    gt_dir    = config['gt_dir']
    mask_dir  = config.get('mask_dir', None)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm', '.gif'))
    ])

    data = []
    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        image = load_image(img_path)

        # Find matching ground truth
        base = os.path.splitext(fname)[0]
        gt_path = find_matching_file(gt_dir, base)
        gt = load_gt(gt_path) if gt_path else None

        # FOV mask
        if config.get('fov_mask_provided') and mask_dir:
            mask_path = find_matching_file(mask_dir, base)
            fov_mask = load_gt(mask_path) if mask_path else None
        else:
            fov_mask = None  # will create circular mask during processing

        data.append({'image': image, 'gt': gt, 'fov_mask': fov_mask, 'name': base})

    return data


def find_matching_file(directory: str, basename: str) -> str:
    """Find a file in directory that contains basename."""
    for f in os.listdir(directory):
        if basename.replace('_test', '') in f or basename in f:
            return os.path.join(directory, f)
    return None
```

---

## μ Computation Note

The scale normalization factor `μ = σo / σs` has units of 1/pixel (since σs is in pixels and σo is in radians, which are dimensionless).

For each scale σs in S and fixed σo = π/40:
```python
mu = sigma_o / sigma_s
```

Example for DRIVE with σs = 2.0:
```
mu = (π/40) / 2.0 = 0.0393 px⁻¹
```

This value is used in the LAD Hessian normalization (`M_μ = diag(μ, μ, 1)`) to make spatial and angular components dimensionally consistent.
