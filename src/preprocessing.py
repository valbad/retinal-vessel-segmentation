"""
SPEC_01 — Preprocessing
=======================
Steps applied before orientation score construction:

  1. Green channel extraction  (RGB fundus datasets)
  2. Luminosity & contrast normalisation  (Foracchia et al., 2005)
  3. Geodesic opening  (suppresses large bright structures: optic disk, artefacts)
  4. Top-hat transform  (enhances dark vessel contrast)

Output: single-channel float32 image in [0, 1].
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
from scipy.ndimage import zoom as _zoom
from skimage.morphology import disk, erosion, closing, reconstruction
from skimage.transform import resize as _resize

import imageio.v2 as imageio

Array = np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float01(image: Array) -> Array:
    """Convert any image array to float32 in [0, 1]."""
    if np.issubdtype(image.dtype, np.floating):
        x = image.astype(np.float32, copy=False)
        return np.clip(x / 255.0 if x.max() > 1.5 else x, 0.0, 1.0)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    x = image.astype(np.float32)
    mx = float(x.max())
    return np.clip(x / mx if mx > 0 else x, 0.0, 1.0)


def _percentile_normalize(
    img: Array,
    mask: Optional[Array] = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
    eps: float = 1e-8,
) -> Array:
    """Robust min-max normalisation using percentiles computed inside mask."""
    x = img.astype(np.float32, copy=False)
    vals = x[mask.astype(bool)] if mask is not None else x.ravel()
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi - lo < eps:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1 — Channel extraction
# ---------------------------------------------------------------------------

def extract_green_channel(image: Array) -> Array:
    """
    Extract the green channel from an RGB fundus image.

    Parameters
    ----------
    image : np.ndarray
        uint8 or float RGB array of shape (H, W, 3).
        For grayscale input (H, W) the array is returned as float32 in [0, 1].

    Returns
    -------
    np.ndarray : float32 in [0, 1], shape (H, W).
    """
    if image.ndim == 3 and image.shape[2] >= 3:
        g = image[..., 1]
        if image.dtype == np.uint8:
            return g.astype(np.float32) / 255.0
        return _to_float01(g)
    # Grayscale
    return _to_float01(image if image.ndim == 2 else image[..., 0])


# ---------------------------------------------------------------------------
# Step 2 — Luminosity & contrast normalisation  (Foracchia et al., 2005)
# ---------------------------------------------------------------------------

def normalize_luminosity_contrast(
    image: Array,
    window_size_px: int,
    threshold: float = 1.0,
    fov_mask: Optional[Array] = None,
    eps: float = 1e-6,
) -> Array:
    """
    Estimate and remove spatially varying luminosity (L̂) and contrast (Ĉ) drifts.

    Model:  I(x,y) = C(x,y)·I°(x,y) + L(x,y)
    Estimate:  Î°(x,y) = (I(x,y) − L̂(x,y)) / Ĉ(x,y)

    Algorithm
    ---------
    a. Partition image into square tiles of side `window_size_px`.
    b. Compute per-tile mean μ and std σ.
    c. Bicubic-interpolate μ and σ grids to full resolution.
    d. Mark background pixels: |I − μ_full| / σ_full < threshold.
    e. Re-estimate L̂ and Ĉ from background pixels only, bicubic-interpolate.
    f. Apply: Î° = (I − L̂) / Ĉ, then rescale to [0, 1].

    Parameters
    ----------
    image : float32, (H, W), in [0, 1]
    window_size_px : tile side length in pixels  (DRIVE: 19, HRF: 125)
    threshold : Mahalanobis threshold for background classification (paper: 1.0)
    fov_mask : optional bool FOV mask; statistics restricted to FOV pixels
    eps : small floor to avoid division by zero

    Returns
    -------
    float32 in [0, 1].
    """
    img = image.astype(np.float32, copy=False)
    H, W = img.shape
    fov = fov_mask.astype(bool) if fov_mask is not None else np.ones((H, W), dtype=bool)

    s = max(1, int(window_size_px))
    ny = int(np.ceil(H / s))
    nx = int(np.ceil(W / s))

    # ------------------------------------------------------------------
    # Steps a + b: per-tile mean and std
    # ------------------------------------------------------------------
    mu_g = np.full((ny, nx), np.nan, dtype=np.float32)
    sg_g = np.full((ny, nx), np.nan, dtype=np.float32)

    for iy in range(ny):
        y0, y1 = iy * s, min((iy + 1) * s, H)
        for ix in range(nx):
            x0, x1 = ix * s, min((ix + 1) * s, W)
            vals = img[y0:y1, x0:x1][fov[y0:y1, x0:x1]]
            if vals.size < 5:
                continue
            mu_g[iy, ix] = float(vals.mean())
            sg_g[iy, ix] = float(vals.std(ddof=0))

    # Fallback: global stats inside FOV
    fov_vals = img[fov]
    g_mu = float(fov_vals.mean()) if fov_vals.size else 0.5
    g_sg = float(fov_vals.std()) if fov_vals.size else 0.1
    g_sg = max(g_sg, eps)

    mu_g = np.where(np.isfinite(mu_g), mu_g, g_mu)
    sg_g = np.where(np.isfinite(sg_g) & (sg_g > eps), sg_g, g_sg)

    # ------------------------------------------------------------------
    # Step c: bicubic interpolation of grid → full resolution
    # ------------------------------------------------------------------
    mu_full = _resize(mu_g, (H, W), order=3, mode="reflect",
                      anti_aliasing=False, preserve_range=True).astype(np.float32)
    sg_full = _resize(sg_g, (H, W), order=3, mode="reflect",
                      anti_aliasing=False, preserve_range=True).astype(np.float32)

    # ------------------------------------------------------------------
    # Step d: background mask
    # ------------------------------------------------------------------
    d_M = np.abs(img - mu_full) / np.maximum(sg_full, eps)
    bg_mask = (d_M < threshold) & fov

    # ------------------------------------------------------------------
    # Step e: re-estimate L̂ and Ĉ from background pixels
    # ------------------------------------------------------------------
    L_g = np.full((ny, nx), np.nan, dtype=np.float32)
    C_g = np.full((ny, nx), np.nan, dtype=np.float32)

    for iy in range(ny):
        y0, y1 = iy * s, min((iy + 1) * s, H)
        for ix in range(nx):
            x0, x1 = ix * s, min((ix + 1) * s, W)
            bg_vals = img[y0:y1, x0:x1][bg_mask[y0:y1, x0:x1]]
            if bg_vals.size < 5:
                continue
            L_val = float(bg_vals.mean())
            C_val = float(bg_vals.std(ddof=0))
            if L_val > 0.75:          # likely optic disk / bright lesion — skip
                continue
            L_g[iy, ix] = L_val
            C_g[iy, ix] = max(C_val, eps)

    L_g = np.where(np.isfinite(L_g), L_g, g_mu)
    C_g = np.where(np.isfinite(C_g) & (C_g > eps), C_g, g_sg)

    L_hat = _resize(L_g, (H, W), order=3, mode="reflect",
                    anti_aliasing=False, preserve_range=True).astype(np.float32)
    C_hat = _resize(C_g, (H, W), order=3, mode="reflect",
                    anti_aliasing=False, preserve_range=True).astype(np.float32)

    # Floor C_hat inside FOV to prevent amplifying noise in flat regions
    if fov.any():
        c_floor = max(float(np.percentile(C_hat[fov], 5.0)), eps)
        C_hat = np.maximum(C_hat, c_floor)

    # ------------------------------------------------------------------
    # Step f: normalise and rescale to [0, 1]
    # ------------------------------------------------------------------
    norm = (img - L_hat) / np.maximum(C_hat, eps)

    # Rescale within FOV
    if fov.any():
        lo, hi = float(norm[fov].min()), float(norm[fov].max())
    else:
        lo, hi = float(norm.min()), float(norm.max())

    if hi - lo < eps:
        return np.zeros_like(norm)

    norm = (norm - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3 — Geodesic opening
# ---------------------------------------------------------------------------

def geodesic_opening(image: Array, kernel_size_px: int) -> Array:
    """
    Suppress large bright structures (optic disk, artefacts) via geodesic opening.

    Implementation: reconstruct the eroded image (marker) under the original
    image as mask using morphological dilation.

        marker = erosion(image, disk(r))
        opened = reconstruction(marker, image, method='dilation')

    This removes features that cannot be reached by geodesic dilation from the
    eroded marker — i.e., large bright blobs disconnected from thin vessels.

    Parameters
    ----------
    image : float32, (H, W)
    kernel_size_px : disk radius for the structuring element

    Returns
    -------
    float32, (H, W).
    """
    selem = disk(kernel_size_px)
    marker = erosion(image, selem)
    opened = reconstruction(marker, image, method="dilation")
    return opened.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4 — Top-hat transform
# ---------------------------------------------------------------------------

def top_hat_transform(image: Array, kernel_size_px: int) -> Array:
    """
    Enhance dark vessel structures via the black top-hat transform.

        black_tophat(I) = closing(I) - I

    Vessels (dark on a bright background) yield positive responses; the smooth
    background is suppressed.

    Note: SPEC_01 refers to the combined effect as "top-hat"; the black variant
    is appropriate because vessels are *dark* in the green channel.

    Parameters
    ----------
    image : float32, (H, W)  — output of geodesic_opening
    kernel_size_px : disk radius (same Wt/2ρ as geodesic opening)

    Returns
    -------
    float32, (H, W), clipped to [0, 1].
    """
    selem = disk(kernel_size_px)
    closed = closing(image, selem)
    result = closed - image               # black top-hat: highlights dark features
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(
    image: Array,
    pixel_size_um: float,
    Wl_um: float = 500.0,
    Wt_um: float = 150.0,
    is_rgb: bool = True,
    fov_mask: Optional[Array] = None,
) -> Array:
    """
    Full preprocessing pipeline (SPEC_01).

    Parameters
    ----------
    image : np.ndarray
        Raw image — uint8 RGB (H, W, 3) or uint8/float grayscale (H, W).
    pixel_size_um : float
        Physical pixel size in μm/px.  DRIVE: 27.0
    Wl_um : float
        Luminosity normalisation window size in μm.  Default 500 μm.
    Wt_um : float
        Top-hat / geodesic opening kernel size in μm.  Default 150 μm.
    is_rgb : bool
        True for colour fundus images; False for SLO/grayscale.
    fov_mask : np.ndarray or None
        Boolean FOV mask.  If provided, statistics are restricted to the FOV.

    Returns
    -------
    float32, (H, W), in [0, 1].
    """
    # 1. Extract channel
    img = extract_green_channel(image) if is_rgb else _to_float01(image)

    # 2. Convert physical sizes to pixels
    Wl_px = max(1, round(Wl_um / pixel_size_um))
    Wt_px = max(1, round(Wt_um / pixel_size_um))
    kernel_radius_px = max(1, Wt_px // 2)

    # 3. Luminosity & contrast normalisation
    img = normalize_luminosity_contrast(img, window_size_px=Wl_px, fov_mask=fov_mask)

    # 4. Geodesic opening (removes large bright structures)
    img = geodesic_opening(img, kernel_size_px=kernel_radius_px)

    # 5. Black top-hat (enhances dark vessels)
    img = top_hat_transform(img, kernel_size_px=kernel_radius_px)

    # 6. Robust percentile normalisation within FOV
    img = _percentile_normalize(img, mask=fov_mask)

    return img


# ---------------------------------------------------------------------------
# DRIVE dataset I/O helpers
# ---------------------------------------------------------------------------

def _load_image(path: str) -> Array:
    """Load any image file as a raw numpy array."""
    return imageio.imread(path)


def _load_mask(path: str) -> Array:
    """Load a DRIVE .gif mask or manual annotation as bool (H, W)."""
    m = imageio.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(bool)


def load_drive_sample(
    drive_paths: Dict[str, Dict[str, str]],
    sample_id: str,
    pixel_size_um: float = 27.0,
    Wl_um: float = 500.0,
    Wt_um: float = 150.0,
) -> Dict[str, Any]:
    """
    Load and preprocess one DRIVE sample.

    Parameters
    ----------
    drive_paths : dict mapping sample_id → {"image": ..., "mask": ..., "manual": ...}
    sample_id : key into drive_paths
    pixel_size_um : DRIVE physical pixel size (default 27 μm/px)
    Wl_um, Wt_um : preprocessing window sizes in μm

    Returns
    -------
    dict with keys:
      id, rgb, green, fov_mask, manual, green_preproc
    """
    if sample_id not in drive_paths:
        raise KeyError(f"sample_id '{sample_id}' not found in drive_paths")

    paths = drive_paths[sample_id]
    raw = _load_image(paths["image"])
    rgb = _to_float01(raw)
    green = extract_green_channel(raw)
    fov_mask = _load_mask(paths["mask"])
    manual = _load_mask(paths["manual"])

    green_preproc = preprocess(
        raw,
        pixel_size_um=pixel_size_um,
        Wl_um=Wl_um,
        Wt_um=Wt_um,
        is_rgb=(raw.ndim == 3),
        fov_mask=fov_mask,
    )

    return {
        "id": sample_id,
        "rgb": rgb,
        "green": green,
        "fov_mask": fov_mask,
        "manual": manual,
        "green_preproc": green_preproc,
    }


def load_all_drive_samples(
    drive_paths: Dict[str, Dict[str, str]],
    pixel_size_um: float = 27.0,
    Wl_um: float = 500.0,
    Wt_um: float = 150.0,
    # Legacy keyword accepted for compatibility with older notebooks
    preprocess_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all DRIVE samples from a paths dict.

    Returns dict mapping sample_id → sample dict (see load_drive_sample).
    """
    if preprocess_params is not None:
        # Extract pixel sizes from legacy params format if present
        Wl_px_override = preprocess_params.get("Wl")
        Wt_px_override = preprocess_params.get("Wt")
        if Wl_px_override is not None:
            Wl_um = Wl_px_override * pixel_size_um
        if Wt_px_override is not None:
            Wt_um = Wt_px_override * pixel_size_um

    return {
        sid: load_drive_sample(
            drive_paths, sid,
            pixel_size_um=pixel_size_um,
            Wl_um=Wl_um,
            Wt_um=Wt_um,
        )
        for sid in sorted(drive_paths.keys())
    }
