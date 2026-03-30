# SPEC 08 — Pipeline Assembly, CLI, Tests, and Expected Results

## Files: `scripts/run_segmentation.py`, `scripts/evaluate.py`, `setup.py`

This document describes how to assemble the full pipeline, the CLI interface, testing strategy, and how to verify the implementation is correct at each stage.

---

## 1. Full Pipeline Assembly

### `scripts/run_segmentation.py`

```python
#!/usr/bin/env python3
"""
Retinal vessel segmentation using LAD-OS (or LID-OS).

Usage:
    python scripts/run_segmentation.py \
        --dataset drive \
        --data_dir /path/to/DRIVE \
        --method lad \
        --output_dir results/drive/ \
        [--threshold 0.554] \
        [--optimize_threshold]
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imsave

from src.preprocessing import preprocess
from src.cake_wavelets import build_cake_wavelets
from src.lid_filter import lid_os_enhance
from src.lad_filter import lad_os_enhance
from src.segmentation import segment, find_optimal_threshold_mcc, evaluate_segmentation
from src.utils import load_config, load_dataset, create_circular_fov_mask


def run_pipeline(config: dict, method: str = 'lad', threshold: float = None,
                 optimize_threshold: bool = False, output_dir: str = None):
    """
    Run the full segmentation pipeline on a dataset.

    Parameters
    ----------
    config : dict
        Dataset configuration (from YAML).
    method : str
        'lad' or 'lid'.
    threshold : float, optional
        Override threshold. If None, use config['threshold_Th'].
    optimize_threshold : bool
        If True, optimize threshold via MCC before segmenting.
    output_dir : str, optional
        Override output directory.
    """
    output_dir = output_dir or config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading {config['dataset_name']} dataset...")
    data = load_dataset(config)
    print(f"  {len(data)} images found.")

    # Build cake wavelets (once, reused for all images)
    print("Building cake wavelets...")
    example_img = data[0]['image']
    # Get image shape (handle RGB)
    if config['is_rgb']:
        img_shape = example_img.shape[:2]
    else:
        img_shape = example_img.shape[:2]

    wavelets = build_cake_wavelets(
        image_shape=img_shape,
        No=config['No'],
    )
    print(f"  Wavelets built: shape {wavelets.shape}")

    # Process all images
    enhanced_images = []
    gt_images = []
    fov_masks = []
    names = []

    print("Running enhancement pipeline...")
    for item in tqdm(data):
        image = item['image']
        gt = item['gt']
        fov_mask = item['fov_mask']
        name = item['name']

        # Determine FOV mask
        if fov_mask is None:
            if config['is_rgb']:
                h, w = image.shape[:2]
            else:
                h, w = image.shape[:2]
            fov_mask = create_circular_fov_mask((h, w))

        # Preprocessing
        preprocessed = preprocess(
            image,
            pixel_size_um=config['pixel_size_um'],
            Wl_um=config['Wl_um'],
            Wt_um=config['Wt_um'],
            is_rgb=config['is_rgb'],
        )

        # Vessel enhancement
        if method == 'lad':
            enhanced = lad_os_enhance(
                image=preprocessed,
                wavelets=wavelets,
                scales=config['scales'],
                sigma_o=config['sigma_o'],
                No=config['No'],
            )
        elif method == 'lid':
            enhanced = lid_os_enhance(
                image=preprocessed,
                wavelets=wavelets,
                scales=config['scales'],
                sigma_o=config['sigma_o'],
                No=config['No'],
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        enhanced_images.append(enhanced)
        gt_images.append(gt)
        fov_masks.append(fov_mask)
        names.append(name)

    # Threshold selection
    if optimize_threshold:
        print("Optimizing threshold via MCC...")
        # Normalize enhanced images first
        normed = [
            (e - e.min()) / (e.max() - e.min() + 1e-8)
            for e in enhanced_images
        ]
        th, best_mcc = find_optimal_threshold_mcc(normed, gt_images, fov_masks)
        print(f"  Optimal threshold: {th:.4f} (MCC = {best_mcc:.4f})")
    else:
        th = threshold if threshold is not None else config['threshold_Th']
        print(f"  Using threshold: {th:.4f}")

    # Segment and evaluate
    print("Segmenting and evaluating...")
    all_metrics = []
    for enhanced, gt, fov_mask, name in zip(enhanced_images, gt_images, fov_masks, names):
        # Normalize
        e_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)

        # Segment
        pred = segment(e_norm, threshold=th, fov_mask=fov_mask)

        # Evaluate
        if gt is not None:
            metrics = evaluate_segmentation(pred, gt, e_norm, fov_mask)
            all_metrics.append(metrics)
            print(f"  {name}: Se={metrics['Se']:.4f}, Sp={metrics['Sp']:.4f}, "
                  f"Acc={metrics['Acc']:.4f}, AUC={metrics['AUC']:.4f}, MCC={metrics['MCC']:.4f}")

        # Save output
        pred_img = (pred.astype(np.uint8) * 255)
        imsave(os.path.join(output_dir, f'{name}_pred.png'), pred_img)

        # Also save enhanced image (for ROC analysis)
        e_uint8 = (e_norm * 255).astype(np.uint8)
        imsave(os.path.join(output_dir, f'{name}_enhanced.png'), e_uint8)

    # Print mean metrics
    if all_metrics:
        print("\n=== Mean Metrics ===")
        for key in ['Se', 'Sp', 'Acc', 'AUC', 'MCC']:
            vals = [m[key] for m in all_metrics]
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Retinal vessel segmentation LAD-OS')
    parser.add_argument('--dataset', required=True,
                        choices=['drive', 'stare', 'chase_db1', 'hrf', 'iostar', 'rc_slo'])
    parser.add_argument('--data_dir', required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--method', default='lad', choices=['lad', 'lid'],
                        help='Enhancement method: lad (default) or lid')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for segmentation results')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Manual threshold (overrides config)')
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Optimize threshold via MCC on the dataset')
    args = parser.parse_args()

    config = load_config(args.dataset)
    # Override data paths with CLI argument
    config['image_dir'] = os.path.join(args.data_dir, 'test', 'images')  # adjust per dataset
    if args.output_dir:
        config['output_dir'] = args.output_dir

    run_pipeline(config, method=args.method, threshold=args.threshold,
                 optimize_threshold=args.optimize_threshold)


if __name__ == '__main__':
    main()
```

---

## 2. setup.py

```python
from setuptools import setup, find_packages

setup(
    name='retinal_vessel_seg',
    version='0.1.0',
    description='LAD-OS Retinal Vessel Segmentation (Zhang et al. 2016)',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'scikit-image>=0.21',
        'matplotlib>=3.7',
        'pyyaml>=6.0',
        'tqdm>=4.65',
    ],
    python_requires='>=3.9',
)
```

---

## 3. requirements.txt

```
numpy>=1.24
scipy>=1.10
scikit-image>=0.21
matplotlib>=3.7
pyyaml>=6.0
tqdm>=4.65
```

---

## 4. Testing Strategy

### Run tests with pytest:
```bash
pip install pytest
pytest tests/ -v
```

### Test hierarchy (fast to slow):

| Test file | Speed | Description |
|-----------|-------|-------------|
| `tests/test_preprocessing.py` | < 5s | Basic shape/dtype checks |
| `tests/test_cake_wavelets.py` | < 30s | Wavelet construction and properties |
| `tests/test_orientation_score.py` | < 60s | Forward/inverse transform accuracy |
| `tests/test_filters.py` | < 5 min | LID and LAD filter responses on synthetic images |
| `tests/test_segmentation.py` | < 5s | Metric computation correctness |
| `tests/test_integration.py` | < 30 min | End-to-end on 1–2 real images from DRIVE |

---

## 5. Integration Test (`tests/test_integration.py`)

```python
"""
Integration test: run full pipeline on one DRIVE image.
Requires DRIVE dataset to be available at DRIVE_DIR environment variable.
"""

import os
import pytest
import numpy as np

DRIVE_DIR = os.environ.get('DRIVE_DIR', None)


@pytest.mark.skipif(DRIVE_DIR is None, reason="DRIVE_DIR not set")
def test_full_pipeline_drive():
    from src.preprocessing import preprocess
    from src.cake_wavelets import build_cake_wavelets
    from src.lad_filter import lad_os_enhance
    from src.segmentation import segment, evaluate_segmentation
    from src.utils import load_config, load_image, load_gt, create_circular_fov_mask
    from skimage.io import imread

    config = load_config('drive')

    # Load one test image (21_test.tif)
    img_path = os.path.join(DRIVE_DIR, 'test', 'images', '01_test.tif')
    gt_path = os.path.join(DRIVE_DIR, 'test', '1st_manual', '01_manual1.gif')
    mask_path = os.path.join(DRIVE_DIR, 'test', 'mask', '01_test_mask.gif')

    if not os.path.exists(img_path):
        pytest.skip("DRIVE test image not found")

    image = imread(img_path)
    gt = load_gt(gt_path)
    fov_mask = load_gt(mask_path)

    # Preprocessing
    preprocessed = preprocess(image, pixel_size_um=27.0, is_rgb=True)
    assert preprocessed.shape == (584, 565)  # H, W of DRIVE
    assert preprocessed.dtype == np.float32

    # Wavelet construction
    wavelets = build_cake_wavelets(preprocessed.shape, No=16)
    assert wavelets.shape == (16, 584, 565)

    # LAD-OS enhancement (use fewer scales for speed)
    enhanced = lad_os_enhance(
        preprocessed, wavelets,
        scales=[1.5, 2.5, 3.5],  # subset of full scales for speed
        sigma_o=np.pi/40,
        No=16,
    )
    assert enhanced.shape == preprocessed.shape
    assert np.all(enhanced >= 0)
    assert enhanced.max() > 0  # some vessels detected

    # Segmentation
    e_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
    pred = segment(e_norm, threshold=0.554, fov_mask=fov_mask)
    assert pred.dtype == bool

    # Evaluation
    metrics = evaluate_segmentation(pred, gt, e_norm, fov_mask)
    print(f"\nDRIVE integration test metrics: {metrics}")

    # Loose checks — exact values depend on all scales being used
    assert metrics['Se'] > 0.5, f"Sensitivity too low: {metrics['Se']}"
    assert metrics['Sp'] > 0.9, f"Specificity too low: {metrics['Sp']}"
    assert metrics['Acc'] > 0.9, f"Accuracy too low: {metrics['Acc']}"
    assert metrics['AUC'] > 0.85, f"AUC too low: {metrics['AUC']}"
```

---

## 6. Debugging Guide

### If filter responses are all zero:
- Check preprocessing: is the preprocessed image non-trivial? `assert preprocessed.max() > 0.01`
- Check orientation score: `assert np.max(np.abs(U_f)) > 0`
- Check scale range: scales too small (< 0.5 px) or too large (> image_size/4) will give poor results

### If vessels are not enhanced (low AUC):
- Verify cake wavelet orientation selectivity (see test in SPEC_02)
- Check the sign of the LID/LAD filter — vessels should give **positive** response
- Confirm the second derivative is computed in the **perpendicular** direction (η, not ξ)

### If segmentation has many false positives:
- Optic disk not removed? Check geodesic opening kernel size (Wt_px)
- Central reflex not removed? Check top-hat kernel size
- Threshold too low — try increasing Th

### If segmentation misses thin vessels:
- Scale set too coarse — add smaller scales (e.g., 0.5 px)
- Preprocessing over-smoothing — check Wl_px value

### Common numpy pitfalls:
- `gaussian_filter(..., order=(0,2))` computes `∂²/∂x²` — note scipy uses `(row_order, col_order)` = `(y_order, x_order)`
- So `order=(2,0)` = `∂²/∂y²` and `order=(0,2)` = `∂²/∂x²`
- When computing `∂_ξ = cos θ ∂_x + sin θ ∂_y`: `∂_x` = `order=(0,1)`, `∂_y` = `order=(1,0)`

### For the Hessian eigensystem:
- `np.linalg.eigh` returns eigenvalues in **ascending** order → use `eigenvectors[:, 0]` for the minimum eigenvector
- The Hessian is 3×3 and not symmetric — you must symmetrize it via `H_mu = M⁻¹ Hᵀ M⁻² H M⁻¹`
- If `mu` is very small (large scales), `M_μ` can become ill-conditioned — use `np.clip(mu, 0.001, 100)`

---

## 7. Reproducibility Checklist

Before claiming reproduction of the paper results:

- [ ] Using green channel for RGB datasets
- [ ] Luminosity normalization window = 500 μm (not px)
- [ ] Top-hat kernel = 150 μm (not px)
- [ ] No = 16 orientations
- [ ] σo = π/40 ≈ 0.0785 rad
- [ ] Scale normalization: μ = σo/σs applied to second-order derivatives (μ⁻²)
- [ ] Multi-scale: **sum** over scales, then **max** over orientations
- [ ] Threshold selected by MCC optimization on each dataset
- [ ] Metrics computed only within FOV mask
- [ ] Using 1st human observer annotations as ground truth (DRIVE, STARE)

---

## 8. Expected Output Format

Each image produces:
- `{name}_pred.png`: binary segmentation (0 or 255)
- `{name}_enhanced.png`: continuous enhanced image (0–255 scaled)

Results summary is printed to stdout in the format:
```
=== DRIVE Test Set (LAD-OS) ===
Mean Se:  0.7743 ± 0.0321
Mean Sp:  0.9725 ± 0.0045
Mean Acc: 0.9476 ± 0.0023
Mean AUC: 0.9636 ± 0.0089
Mean MCC: 0.7571 ± 0.0198
```

---

## 9. Implementation Order Recommendation

Build and test in this order to catch errors early:

1. **`src/utils.py`** — dataset loading (can test with just reading files)
2. **`src/preprocessing.py`** — test visually by inspecting preprocessed images
3. **`src/cake_wavelets.py`** — test partition of unity and reconstruction
4. **`src/orientation_score.py`** — test orientation selectivity on line images
5. **`src/lid_filter.py`** — test on synthetic vessel images (Gaussian ridges)
6. **`src/segmentation.py`** — test metrics on known confusion matrices
7. **`src/lad_filter.py`** — most complex; test last after all dependencies verified
8. **`scripts/run_segmentation.py`** — integration test on real data

This order ensures each new component has a tested foundation beneath it.

---

## 10. Notes on Efficiency

For the paper's results with DRIVE (565×584, 16 orientations, 7 scales):
- Each LAD filter call involves 9 derivative computations + 1 eigendecomposition per pixel
- Total: 7 scales × 16 orientations × 9 derivatives ≈ 1000 Gaussian filter calls
- Eigendecompositions: 565×584 ≈ 330k per orientation per scale

**Suggested optimizations (optional):**
1. Cache `U_f_stack` (orientation score) — compute once, reuse across scales
2. Precompute all Gaussian derivative kernels using FFT multiplication
3. Use `numba` JIT for the pixel-wise eigensystem loop
4. For HRF (3504×2336): consider tiled processing to manage memory

The baseline implementation in plain numpy is correct and sufficient for reproducing the paper; optimization is secondary.
