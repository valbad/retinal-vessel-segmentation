# SPEC 06 — Segmentation and Evaluation Metrics

## File: `src/segmentation.py`

After the enhancement step, we have a float image `Υ(f)` where high values indicate likely vessel pixels. This module converts that enhanced image to a binary segmentation and evaluates performance.

---

## 1. Hard Segmentation (Thresholding)

```python
def segment(
    enhanced: np.ndarray,
    threshold: float,
    fov_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Convert enhanced image to binary vessel map by thresholding.

    Parameters
    ----------
    enhanced : np.ndarray (H, W) float32
        Output of lid_os_enhance or lad_os_enhance. Values in [0, ∞).
    threshold : float
        Segmentation threshold T_h. Pixels with enhanced > T_h are vessels.
        Should be determined by MCC optimization on each dataset.
    fov_mask : np.ndarray (H, W) bool, optional
        Field-of-view mask. Only pixels inside FOV are evaluated.
        If None, all pixels are used.

    Returns
    -------
    binary_map : np.ndarray (H, W) bool
        True = vessel, False = background.
    """
    # Rescale enhanced image to [0, 1]
    e_min = enhanced.min()
    e_max = enhanced.max()
    if e_max > e_min:
        enhanced_norm = (enhanced - e_min) / (e_max - e_min)
    else:
        enhanced_norm = np.zeros_like(enhanced)

    binary_map = enhanced_norm > threshold

    # Apply FOV mask: pixels outside FOV are always background
    if fov_mask is not None:
        binary_map = binary_map & fov_mask

    return binary_map.astype(bool)
```

---

## 2. Performance Metrics

All metrics are computed **only within the FOV mask** (on pixels within the circular retinal field of view).

### Confusion Matrix

```python
def confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    fov_mask: np.ndarray = None,
) -> dict:
    """
    Compute TP, FP, TN, FN within FOV.

    Parameters
    ----------
    pred : np.ndarray (H, W) bool
        Predicted binary vessel map.
    gt : np.ndarray (H, W) bool
        Ground truth binary vessel map (first observer).
    fov_mask : np.ndarray (H, W) bool, optional
        If provided, restrict evaluation to FOV pixels.

    Returns
    -------
    dict with keys 'TP', 'FP', 'TN', 'FN', 'N'
    """
    if fov_mask is not None:
        pred = pred[fov_mask]
        gt   = gt[fov_mask]
    else:
        pred = pred.ravel()
        gt   = gt.ravel()

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    TP = int(np.sum(pred & gt))
    FP = int(np.sum(pred & ~gt))
    TN = int(np.sum(~pred & ~gt))
    FN = int(np.sum(~pred & gt))
    N  = TP + FP + TN + FN

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'N': N}
```

### Sensitivity, Specificity, Accuracy

```python
def compute_se_sp_acc(cm: dict) -> dict:
    """
    Compute sensitivity, specificity, and accuracy.

    Se  = TP / (TP + FN)       True Positive Rate
    Sp  = TN / (TN + FP)       True Negative Rate
    Acc = (TP + TN) / N        Overall Accuracy
    """
    TP, FP, TN, FN, N = cm['TP'], cm['FP'], cm['TN'], cm['FN'], cm['N']
    Se  = TP / (TP + FN + 1e-8)
    Sp  = TN / (TN + FP + 1e-8)
    Acc = (TP + TN) / (N + 1e-8)
    return {'Se': Se, 'Sp': Sp, 'Acc': Acc}
```

### Matthews Correlation Coefficient (MCC)

MCC is the primary metric for threshold optimization in this paper. It handles class imbalance well (vessels are only ~9–14% of pixels).

```
MCC = (TP/N − S·P) / √(P · S · (1−S) · (1−P))
```

where:
- `S = (TP + FN) / N` = fraction of positives (vessel prevalence)
- `P = (TP + FP) / N` = fraction of predicted positives

```python
def compute_mcc(cm: dict) -> float:
    """
    Compute Matthews Correlation Coefficient.

    MCC = (TP/N − S*P) / sqrt(P * S * (1−S) * (1−P))

    Returns value in [−1, 1]. Higher is better.
    Returns 0 if denominator is zero (degenerate case).
    """
    TP, FP, TN, FN, N = cm['TP'], cm['FP'], cm['TN'], cm['FN'], cm['N']
    S = (TP + FN) / N   # sensitivity denominator / prevalence
    P = (TP + FP) / N   # precision denominator

    numerator   = TP / N - S * P
    denominator = np.sqrt(P * S * (1 - S) * (1 - P) + 1e-12)

    return float(numerator / denominator)
```

### ROC Curve and AUC

The ROC curve is computed by sweeping the threshold `T_h` from 0 to 1 and computing (Se, 1−Sp) at each threshold.

```python
def compute_roc_auc(
    enhanced: np.ndarray,
    gt: np.ndarray,
    fov_mask: np.ndarray = None,
    n_thresholds: int = 200,
) -> dict:
    """
    Compute ROC curve and AUC.

    Parameters
    ----------
    enhanced : np.ndarray (H, W) float32
        Enhanced (probability) image, normalized to [0, 1].
    gt : np.ndarray (H, W) bool
        Ground truth.
    fov_mask : np.ndarray (H, W) bool, optional
    n_thresholds : int
        Number of threshold values to sweep.

    Returns
    -------
    dict with keys 'fpr' (list), 'tpr' (list), 'auc' (float), 'thresholds' (list)
    """
    # Normalize enhanced to [0, 1]
    e = enhanced.copy()
    e = (e - e.min()) / (e.max() - e.min() + 1e-8)

    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    fpr_list = []

    for th in thresholds:
        pred = e > th
        cm = confusion_matrix(pred, gt, fov_mask)
        metrics = compute_se_sp_acc(cm)
        tpr_list.append(metrics['Se'])
        fpr_list.append(1 - metrics['Sp'])

    # AUC via trapezoidal rule (sort by FPR)
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sort_idx = np.argsort(fpr_arr)
    auc = float(np.trapz(tpr_arr[sort_idx], fpr_arr[sort_idx]))

    return {
        'fpr': fpr_arr[sort_idx].tolist(),
        'tpr': tpr_arr[sort_idx].tolist(),
        'auc': auc,
        'thresholds': thresholds[sort_idx].tolist(),
    }
```

---

## 3. Threshold Optimization (MCC-based)

The paper selects the threshold `T_h` that maximizes MCC on each dataset. This is done **globally** (one threshold per dataset, not per image).

```python
def find_optimal_threshold_mcc(
    enhanced_images: list,
    gt_images: list,
    fov_masks: list,
    n_thresholds: int = 200,
) -> float:
    """
    Find the single threshold T_h that maximizes MCC across all images
    in a dataset.

    This implements the paper's approach: one global threshold per dataset,
    optimized using the MCC metric.

    Parameters
    ----------
    enhanced_images : list of np.ndarray (H, W) float32
        Enhanced images (normalized to [0,1]).
    gt_images : list of np.ndarray (H, W) bool
        Ground truth binary maps.
    fov_masks : list of np.ndarray (H, W) bool or None
    n_thresholds : int

    Returns
    -------
    optimal_threshold : float
    optimal_mcc : float
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    mcc_values = []

    for th in thresholds:
        # Accumulate global confusion matrix across all images
        global_cm = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'N': 0}
        for enhanced, gt, fov in zip(enhanced_images, gt_images, fov_masks):
            pred = enhanced > th
            cm = confusion_matrix(pred, gt, fov)
            for key in global_cm:
                global_cm[key] += cm[key]
        mcc = compute_mcc(global_cm)
        mcc_values.append(mcc)

    best_idx = np.argmax(mcc_values)
    return float(thresholds[best_idx]), float(mcc_values[best_idx])
```

---

## 4. FOV Mask Handling

The Field of View (FOV) mask defines which pixels are within the retinal image boundary (circular region). Pixels outside are not evaluated.

```python
def load_fov_mask(mask_path: str) -> np.ndarray:
    """
    Load FOV mask from image file.

    Mask images are typically provided as binary PNGs (white = inside FOV).

    Returns
    -------
    fov_mask : np.ndarray (H, W) bool
    """
    from skimage.io import imread
    mask = imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 128


def create_circular_fov_mask(image_shape: tuple, margin: int = 5) -> np.ndarray:
    """
    Create an approximate circular FOV mask based on image shape.
    Used when no mask file is provided (STARE, CHASE_DB1, IOSTAR, RC-SLO).

    Parameters
    ----------
    image_shape : tuple (H, W)
    margin : int
        Pixels to erode from the detected boundary.

    Returns
    -------
    fov_mask : np.ndarray (H, W) bool
    """
    from skimage.morphology import disk, binary_erosion

    H, W = image_shape
    cy, cx = H // 2, W // 2
    r = min(H, W) // 2 - margin

    Y, X = np.ogrid[:H, :W]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    return mask.astype(bool)
```

---

## 5. Comprehensive Evaluation Function

```python
def evaluate_segmentation(
    pred: np.ndarray,
    gt: np.ndarray,
    enhanced: np.ndarray,
    fov_mask: np.ndarray = None,
) -> dict:
    """
    Compute all metrics: Se, Sp, Acc, MCC, AUC.

    Parameters
    ----------
    pred : np.ndarray (H, W) bool
        Binary prediction at optimal threshold.
    gt : np.ndarray (H, W) bool
        Ground truth.
    enhanced : np.ndarray (H, W) float32
        Continuous enhanced image (for AUC computation).
    fov_mask : np.ndarray (H, W) bool, optional

    Returns
    -------
    dict with keys 'Se', 'Sp', 'Acc', 'MCC', 'AUC'
    """
    cm = confusion_matrix(pred, gt, fov_mask)
    se_sp_acc = compute_se_sp_acc(cm)
    mcc = compute_mcc(cm)
    roc = compute_roc_auc(enhanced, gt, fov_mask)

    return {
        'Se': se_sp_acc['Se'],
        'Sp': se_sp_acc['Sp'],
        'Acc': se_sp_acc['Acc'],
        'MCC': mcc,
        'AUC': roc['auc'],
        'TP': cm['TP'],
        'FP': cm['FP'],
        'TN': cm['TN'],
        'FN': cm['FN'],
    }
```

---

## 6. Per-Dataset Thresholds (from paper, Table II)

| Dataset | T_h (MCC optimal) |
|---------|--------------------|
| DRIVE | 0.5540 |
| STARE | 0.5940 |
| CHASE_DB1 | 0.6160 |
| HRF | 0.6100 |
| IOSTAR | 0.6100 |
| RC-SLO | 0.6080 |

These are the thresholds used in the paper. For reproduction, **use these directly** rather than re-optimizing, to compare fairly.

For your own datasets, use `find_optimal_threshold_mcc` on a held-out set.

---

## Unit Tests (`tests/test_segmentation.py`)

```python
def test_confusion_matrix_basic():
    pred = np.array([[True, False], [True, True]])
    gt   = np.array([[True, True],  [False, True]])
    cm = confusion_matrix(pred, gt)
    assert cm['TP'] == 2  # (0,0) and (1,1)
    assert cm['FP'] == 1  # (1,0)
    assert cm['FN'] == 1  # (0,1)
    assert cm['TN'] == 0
    assert cm['N'] == 4

def test_sensitivity_perfect():
    cm = {'TP': 10, 'FP': 0, 'TN': 90, 'FN': 0, 'N': 100}
    m = compute_se_sp_acc(cm)
    assert abs(m['Se'] - 1.0) < 1e-6
    assert abs(m['Sp'] - 1.0) < 1e-6
    assert abs(m['Acc'] - 1.0) < 1e-6

def test_mcc_range():
    cm = {'TP': 5, 'FP': 3, 'TN': 85, 'FN': 7, 'N': 100}
    mcc = compute_mcc(cm)
    assert -1.0 <= mcc <= 1.0

def test_mcc_perfect():
    cm = {'TP': 10, 'FP': 0, 'TN': 90, 'FN': 0, 'N': 100}
    mcc = compute_mcc(cm)
    assert abs(mcc - 1.0) < 1e-4

def test_auc_random():
    """AUC of random predictor should be close to 0.5."""
    np.random.seed(42)
    H, W = 100, 100
    enhanced = np.random.rand(H, W).astype(np.float32)
    gt = (np.random.rand(H, W) > 0.9).astype(bool)
    roc = compute_roc_auc(enhanced, gt, n_thresholds=50)
    assert 0.3 < roc['auc'] < 0.7  # roughly 0.5 for random

def test_threshold_optimization():
    H, W = 100, 100
    enhanced = np.random.rand(H, W).astype(np.float32)
    gt = enhanced > 0.7  # ground truth matches high-valued pixels
    # Add noise
    enhanced = enhanced + 0.1 * np.random.rand(H, W)
    enhanced = np.clip(enhanced, 0, 1).astype(np.float32)

    th, mcc = find_optimal_threshold_mcc(
        [enhanced], [gt], [None], n_thresholds=50
    )
    assert 0.0 <= th <= 1.0
    assert -1.0 <= mcc <= 1.0
```
