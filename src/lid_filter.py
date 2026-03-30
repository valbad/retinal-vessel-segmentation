"""
SPEC_04 — Left-Invariant Derivative (LID) Filter
==================================================
Vessel enhancement via second-order Gaussian derivatives in the
left-invariant frame {∂_ξ, ∂_η, ∂_θ} of the orientation score.

LID filter (scale-normalised):
    Φ^{σs,σo}_{η,norm}(U_f) = −μ⁻² · ∂²_η (G_{σs,σo} ★ U_f)

where μ = σo / σs.  Vessels appear as ridges with ∂²_η < 0,
so the negated response is positive at vessel centres.

Multi-scale reconstruction:
    Υ(f)(x) = max_i { Σ_{σs ∈ S} Φ^{σs,σo}_{η,norm}(U_f)(x, θ_i) }
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from orientation_score import compute_orientation_score_layer

Array = np.ndarray


# ---------------------------------------------------------------------------
# Angular smoothing (shared with LAD filter)
# ---------------------------------------------------------------------------

def angular_smooth_stack(
    U_f_stack: Array,
    sigma_o: float,
    No: int,
) -> Array:
    """
    Apply 1D Gaussian smoothing along the orientation axis (axis 0).

    Converts σo (radians) to index units using Δθ = π/No.
    For the paper's σo = π/40 and No=16, σ_idx ≈ 0.4 — barely any mixing.
    Uses circular ('wrap') boundary to respect the periodic orientation domain.

    Parameters
    ----------
    U_f_stack : (No, H, W) float32
    sigma_o : float  — angular std in radians
    No : int

    Returns
    -------
    (No, H, W) float32
    """
    delta_theta = np.pi / No
    sigma_idx = sigma_o / delta_theta

    if sigma_idx < 0.1:   # essentially a delta — skip smoothing
        return U_f_stack

    return gaussian_filter1d(
        U_f_stack.astype(np.float32),
        sigma=sigma_idx,
        axis=0,
        mode='wrap',
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial derivative helpers  (also imported by lad_filter)
# ---------------------------------------------------------------------------

def compute_xi_eta_derivatives(
    U_f_layer: Array,
    theta: float,
    sigma_s: float,
) -> dict:
    """
    Compute first- and second-order left-invariant frame derivatives at angle θ.

    Uses scipy Gaussian derivative filters to smooth-and-differentiate in one
    step, then rotates to the LID frame:
        ∂_ξ = cos θ ∂_x + sin θ ∂_y     (along vessel)
        ∂_η = −sin θ ∂_x + cos θ ∂_y    (perpendicular)

    Parameters
    ----------
    U_f_layer : (H, W) float32 — one orientation score layer
    theta : float — orientation angle in radians
    sigma_s : float — spatial Gaussian scale in pixels

    Returns
    -------
    dict with keys:
        d_xi, d_eta                         (H, W) first-order LID derivatives
        d2_xi_xi, d2_eta_eta, d2_xi_eta     (H, W) second-order LID derivatives
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # scipy convention: order=(row_order, col_order) = (∂_y order, ∂_x order)
    dx  = gaussian_filter(U_f_layer, sigma=sigma_s, order=(0, 1))  # ∂_x
    dy  = gaussian_filter(U_f_layer, sigma=sigma_s, order=(1, 0))  # ∂_y

    dxx = gaussian_filter(U_f_layer, sigma=sigma_s, order=(0, 2))  # ∂²_x
    dyy = gaussian_filter(U_f_layer, sigma=sigma_s, order=(2, 0))  # ∂²_y
    dxy = gaussian_filter(U_f_layer, sigma=sigma_s, order=(1, 1))  # ∂²_xy

    # Rotate to LID frame
    d_xi  =  cos_t * dx + sin_t * dy
    d_eta = -sin_t * dx + cos_t * dy

    s2, c2, sc = sin_t**2, cos_t**2, sin_t * cos_t
    d2_xi_xi   =  c2 * dxx + 2 * sc * dxy + s2 * dyy
    d2_eta_eta =  s2 * dxx - 2 * sc * dxy + c2 * dyy
    d2_xi_eta  = -sc * dxx + (c2 - s2) * dxy + sc * dyy

    return {
        'd_xi':       d_xi.astype(np.float32),
        'd_eta':      d_eta.astype(np.float32),
        'd2_xi_xi':   d2_xi_xi.astype(np.float32),
        'd2_eta_eta': d2_eta_eta.astype(np.float32),
        'd2_xi_eta':  d2_xi_eta.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# LID filter components
# ---------------------------------------------------------------------------

def second_derivative_perpendicular(
    U_f_layer: Array,
    theta: float,
    sigma_s: float,
) -> Array:
    """
    Compute ∂²_η (G_σs ★ U_f_layer) via rotated Gaussian second derivatives.

    ∂²_η = sin²θ ∂²_xx + cos²θ ∂²_yy − 2 sinθ cosθ ∂²_xy

    Returns (H, W) float32.
    """
    return compute_xi_eta_derivatives(U_f_layer, theta, sigma_s)['d2_eta_eta']


def lid_filter_response(
    U_f_layer: Array,
    theta: float,
    sigma_s: float,
    sigma_o: float,
) -> Array:
    """
    Scale-normalised LID filter response for one orientation layer.

    Φ^{σs,σo}_{η,norm}(U_f)(x, θ) = −μ⁻² ∂²_η (G_σs ★ U_f)(x)

    where μ = σo / σs,  μ⁻² = σs² / σo².

    Negative responses (background, edges) are clamped to zero.

    Returns (H, W) float32 ≥ 0.
    """
    mu_sq     = (sigma_o / sigma_s) ** 2
    mu_inv_sq = 1.0 / mu_sq              # σs² / σo²

    d2_eta   = second_derivative_perpendicular(U_f_layer, theta, sigma_s)
    response = -mu_inv_sq * d2_eta

    return np.maximum(response, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Full LID-OS enhancement pipeline
# ---------------------------------------------------------------------------

def lid_os_enhance(
    image: Array,
    wavelets: Array,
    scales: list,
    sigma_o: float,
    No: int,
) -> Array:
    """
    Full LID-OS vessel enhancement.

    Υ(f)(x) = max_{i=0..No-1} { Σ_{σs ∈ S} Φ^{σs,σo}_{η,norm}(U_f)(x, θ_i) }

    Parameters
    ----------
    image : (H, W) float32
    wavelets : (No, H, W) complex64
    scales : list of float — spatial scales σs in pixels
    sigma_o : float — angular scale (paper: π/40 for all datasets)
    No : int — number of orientations (paper: 16)

    Returns
    -------
    enhanced : (H, W) float32
    """
    H, W = image.shape
    enhanced = np.zeros((H, W), dtype=np.float32)

    # Pre-compute image FFT once
    F_image = np.fft.fft2(image.astype(np.float64))

    for i in range(No):
        theta_i = i * np.pi / No

        # Orientation score layer i (real part only)
        U_f_i = compute_orientation_score_layer(F_image, wavelets[i])

        # Sum filter responses over all scales
        orientation_response = np.zeros((H, W), dtype=np.float32)
        for sigma_s in scales:
            orientation_response += lid_filter_response(
                U_f_i, theta_i, sigma_s, sigma_o
            )

        # Max-projection over orientations
        enhanced = np.maximum(enhanced, orientation_response)

    return enhanced
