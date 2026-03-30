"""
SPEC_06 — Segmentation and Evaluation Metrics
==============================================
Converts the enhanced float image Υ(f) to a binary vessel map and
computes performance metrics against ground-truth annotations.

All metrics are evaluated *within the FOV mask*.

Metrics
-------
Se   = TP / (TP + FN)            Sensitivity (True Positive Rate)
Sp   = TN / (TN + FP)            Specificity (True Negative Rate)
Acc  = (TP + TN) / N             Accuracy
MCC  = (TP/N − S·P) / √(P·S·(1−S)·(1−P))   Matthews Correlation Coefficient
AUC                               Area Under ROC Curve

Thresholds T_h (from paper Table II, MCC-optimal per dataset):
    DRIVE 0.554 | STARE 0.594 | CHASE_DB1 0.616 | HRF 0.610 | IOSTAR 0.610 | RC-SLO 0.608
"""
from __future__ import annotations

from typing import Optional, List

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Hard segmentation
# ---------------------------------------------------------------------------

def segment(
    enhanced: Array,
    threshold: float,
    fov_mask: Optional[Array] = None,
) -> Array:
    """
    Threshold the enhanced image to produce a binary vessel map.

    The enhanced image is first normalised to [0, 1]; pixels above
    `threshold` are labelled as vessels.  Pixels outside `fov_mask`
    are always labelled background.

    Parameters
    ----------
    enhanced : (H, W) float32 — output of lid_os_enhance / lad_os_enhance
    threshold : float in [0, 1]
    fov_mask : (H, W) bool, optional

    Returns
    -------
    binary_map : (H, W) bool
    """
    e_min, e_max = float(enhanced.min()), float(enhanced.max())
    if e_max > e_min:
        enhanced_norm = (enhanced - e_min) / (e_max - e_min)
    else:
        enhanced_norm = np.zeros_like(enhanced)

    binary_map = enhanced_norm > threshold

    if fov_mask is not None:
        binary_map = binary_map & fov_mask.astype(bool)

    return binary_map.astype(bool)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    pred: Array,
    gt: Array,
    fov_mask: Optional[Array] = None,
) -> dict:
    """
    Compute TP, FP, TN, FN within the FOV.

    Parameters
    ----------
    pred : (H, W) bool
    gt   : (H, W) bool
    fov_mask : (H, W) bool, optional

    Returns
    -------
    dict with keys 'TP', 'FP', 'TN', 'FN', 'N'
    """
    p = pred.astype(bool)
    g = gt.astype(bool)

    if fov_mask is not None:
        m = fov_mask.astype(bool)
        p, g = p[m], g[m]
    else:
        p, g = p.ravel(), g.ravel()

    TP = int(np.sum( p &  g))
    FP = int(np.sum( p & ~g))
    TN = int(np.sum(~p & ~g))
    FN = int(np.sum(~p &  g))

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'N': TP + FP + TN + FN}


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_se_sp_acc(cm: dict) -> dict:
    """
    Sensitivity, Specificity, Accuracy from a confusion matrix dict.

    Se  = TP / (TP + FN)
    Sp  = TN / (TN + FP)
    Acc = (TP + TN) / N
    """
    TP, FP, TN, FN, N = cm['TP'], cm['FP'], cm['TN'], cm['FN'], cm['N']
    eps = 1e-8
    return {
        'Se':  TP / (TP + FN + eps),
        'Sp':  TN / (TN + FP + eps),
        'Acc': (TP + TN) / (N + eps),
    }


def compute_mcc(cm: dict) -> float:
    """
    Matthews Correlation Coefficient.

    MCC = (TP/N − S·P) / √(P·S·(1−S)·(1−P))

    where S = (TP+FN)/N (prevalence) and P = (TP+FP)/N (predicted prevalence).
    Returns 0.0 for degenerate cases (all-positive or all-negative predictions).
    """
    TP, FP, TN, FN, N = cm['TP'], cm['FP'], cm['TN'], cm['FN'], cm['N']
    if N == 0:
        return 0.0

    S = (TP + FN) / N
    P = (TP + FP) / N

    numerator   = TP / N - S * P
    denominator = np.sqrt(P * S * (1.0 - S) * (1.0 - P) + 1e-12)

    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# ROC curve and AUC
# ---------------------------------------------------------------------------

def compute_roc_auc(
    enhanced: Array,
    gt: Array,
    fov_mask: Optional[Array] = None,
    n_thresholds: int = 200,
) -> dict:
    """
    Compute the ROC curve and AUC by sweeping thresholds over [0, 1].

    Parameters
    ----------
    enhanced : (H, W) float32 — continuous response (will be normalised to [0,1])
    gt : (H, W) bool
    fov_mask : (H, W) bool, optional
    n_thresholds : int

    Returns
    -------
    dict with keys 'fpr', 'tpr', 'auc', 'thresholds'
    """
    e = enhanced.astype(np.float32)
    e_min, e_max = float(e.min()), float(e.max())
    if e_max > e_min:
        e = (e - e_min) / (e_max - e_min)
    else:
        e = np.zeros_like(e)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    tpr_list, fpr_list = [], []

    for th in thresholds:
        pred = e > th
        cm   = confusion_matrix(pred, gt, fov_mask)
        m    = compute_se_sp_acc(cm)
        tpr_list.append(m['Se'])
        fpr_list.append(1.0 - m['Sp'])

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    order   = np.argsort(fpr_arr)
    auc     = float(np.trapz(tpr_arr[order], fpr_arr[order]))

    return {
        'fpr':        fpr_arr[order].tolist(),
        'tpr':        tpr_arr[order].tolist(),
        'auc':        auc,
        'thresholds': thresholds[order].tolist(),
    }


# ---------------------------------------------------------------------------
# Threshold optimisation (MCC-based, global across a dataset)
# ---------------------------------------------------------------------------

def find_optimal_threshold_mcc(
    enhanced_images: List[Array],
    gt_images: List[Array],
    fov_masks: List[Optional[Array]],
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """
    Find the single threshold T_h that maximises MCC across all images.

    The enhanced images are normalised to [0, 1] before sweeping.

    Returns
    -------
    optimal_threshold : float
    optimal_mcc : float
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    mcc_values = []

    for th in thresholds:
        global_cm = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'N': 0}
        for enhanced, gt, fov in zip(enhanced_images, gt_images, fov_masks):
            pred = segment(enhanced, float(th), fov)
            cm   = confusion_matrix(pred, gt, fov)
            for key in global_cm:
                global_cm[key] += cm[key]
        mcc_values.append(compute_mcc(global_cm))

    best_idx = int(np.argmax(mcc_values))
    return float(thresholds[best_idx]), float(mcc_values[best_idx])


# ---------------------------------------------------------------------------
# FOV mask utilities
# ---------------------------------------------------------------------------

def load_fov_mask(mask_path: str) -> Array:
    """
    Load a FOV mask from an image file.  White pixels (> 128) are inside FOV.

    Returns (H, W) bool.
    """
    from skimage.io import imread
    mask = imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 128).astype(bool)


def create_circular_fov_mask(image_shape: tuple, margin: int = 5) -> Array:
    """
    Create an approximate circular FOV mask for datasets without provided masks
    (STARE, CHASE_DB1, IOSTAR, RC-SLO).

    Parameters
    ----------
    image_shape : (H, W)
    margin : int — pixels to trim from the circle edge

    Returns (H, W) bool.
    """
    H, W = image_shape
    cy, cx = H // 2, W // 2
    r = min(H, W) // 2 - margin

    Y, X = np.ogrid[:H, :W]
    return ((X - cx)**2 + (Y - cy)**2 <= r**2).astype(bool)


# ---------------------------------------------------------------------------
# All-in-one evaluation
# ---------------------------------------------------------------------------

def evaluate_segmentation(
    pred: Array,
    gt: Array,
    enhanced: Array,
    fov_mask: Optional[Array] = None,
) -> dict:
    """
    Compute Se, Sp, Acc, MCC, AUC in one call.

    Parameters
    ----------
    pred : (H, W) bool — binary prediction at chosen threshold
    gt   : (H, W) bool — ground truth
    enhanced : (H, W) float32 — continuous response (for AUC)
    fov_mask : (H, W) bool, optional

    Returns
    -------
    dict with keys 'Se', 'Sp', 'Acc', 'MCC', 'AUC', 'TP', 'FP', 'TN', 'FN'
    """
    cm      = confusion_matrix(pred, gt, fov_mask)
    metrics = compute_se_sp_acc(cm)
    mcc     = compute_mcc(cm)
    roc     = compute_roc_auc(enhanced, gt, fov_mask)

    return {
        'Se':  metrics['Se'],
        'Sp':  metrics['Sp'],
        'Acc': metrics['Acc'],
        'MCC': mcc,
        'AUC': roc['auc'],
        'TP':  cm['TP'],
        'FP':  cm['FP'],
        'TN':  cm['TN'],
        'FN':  cm['FN'],
    }


# ---------------------------------------------------------------------------
# Per-dataset thresholds from the paper (Table II)
# ---------------------------------------------------------------------------

DATASET_THRESHOLDS = {
    'drive':     0.5540,
    'stare':     0.5940,
    'chase_db1': 0.6160,
    'hrf':       0.6100,
    'iostar':    0.6100,
    'rc_slo':    0.6080,
}
