"""
SPEC_05 — Locally Adaptive Derivative (LAD) Filter
====================================================
The main contribution of Zhang et al. (2016).

Starting from the LID frame {∂_ξ, ∂_η, ∂_θ}, a locally adaptive frame
{∂_a, ∂_b, ∂_c} is constructed at each pixel via two rotations driven by
the local vessel geometry (curvature κ, deviation from horizontality d_H).

The LAD filter then applies:
    Φ^{σs,σo}_{b,norm}(U_f) = −μ⁻² ∂²_b (G_{σs,σo} ★ U_f)

Pipeline per orientation i, scale σs
--------------------------------------
1. Assemble 3×3 left-invariant Hessian H_Uf (spatial + angular finite diffs)
2. Form symmetrised μ-normalised Hessian H_μ = M_μ⁻¹ H^T M_μ⁻² H M_μ⁻¹
3. Eigendecompose H_μ → smallest eigenvector c̃*, then c* = M_μ⁻¹ c̃*
4. Derive curvature κ and deviation d_H from c*
5. Compute e_b (perpendicular vessel direction in LID frame)
6. ∂²_b = e_b^T H_Uf e_b
7. Response = −μ⁻² ∂²_b,  clamped to ≥ 0
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from orientation_score import build_orientation_score_stack
from lid_filter import angular_smooth_stack, compute_xi_eta_derivatives

Array = np.ndarray


# ---------------------------------------------------------------------------
# Hessian helpers
# ---------------------------------------------------------------------------

def _theta_derivatives(
    U_f_stack: Array,
    i: int,
    No: int,
    delta_theta: float,
    sigma_s: float,
) -> tuple[Array, Array]:
    """
    Central-difference ∂_θ and ∂²_θ at orientation index i (circular wrap).

    Returns
    -------
    d_theta : (H, W) float32
    d2_theta : (H, W) float32
    """
    i_prev = (i - 1) % No
    i_next = (i + 1) % No

    prev = gaussian_filter(U_f_stack[i_prev], sigma=sigma_s)
    curr = gaussian_filter(U_f_stack[i],      sigma=sigma_s)
    nxt  = gaussian_filter(U_f_stack[i_next], sigma=sigma_s)

    d_theta  = ((nxt - prev) / (2.0 * delta_theta)).astype(np.float32)
    d2_theta = ((nxt - 2.0 * curr + prev) / delta_theta**2).astype(np.float32)

    return d_theta, d2_theta


def _cross_theta_terms(
    U_f_stack: Array,
    i: int,
    No: int,
    delta_theta: float,
    sigma_s: float,
) -> tuple[Array, Array]:
    """
    Central-difference ∂_θ∂_ξ and ∂_θ∂_η at orientation index i.

    Computed by central-differencing the ∂_ξ / ∂_η derivatives evaluated
    at the adjacent orientation layers with their own angles.

    Returns
    -------
    d_theta_d_xi : (H, W) float32
    d_theta_d_eta : (H, W) float32
    """
    i_prev = (i - 1) % No
    i_next = (i + 1) % No
    theta_prev = i_prev * np.pi / No
    theta_next = i_next * np.pi / No

    deriv_prev = compute_xi_eta_derivatives(U_f_stack[i_prev], theta_prev, sigma_s)
    deriv_next = compute_xi_eta_derivatives(U_f_stack[i_next], theta_next, sigma_s)

    d_theta_d_xi  = ((deriv_next['d_xi']  - deriv_prev['d_xi'])  / (2.0 * delta_theta)).astype(np.float32)
    d_theta_d_eta = ((deriv_next['d_eta'] - deriv_prev['d_eta']) / (2.0 * delta_theta)).astype(np.float32)

    return d_theta_d_xi, d_theta_d_eta


def assemble_hessian(
    U_f_stack: Array,
    i: int,
    No: int,
    theta_i: float,
    sigma_s: float,
    sigma_o: float,
    mu: float,
) -> dict:
    """
    Assemble the 3×3 left-invariant Hessian H_Uf at orientation layer i.

    The Hessian is expressed in the LID frame {∂_ξ, ∂_η, ∂_θ}:

        H_Uf = | d²_ξξ    d²_ξη    d_θ∂_ξ |
                | d²_ξη    d²_ηη    d_θ∂_η |
                | d_ξ∂_θ   d_η∂_θ   d²_θθ  |

    Note: the (θ,ξ) and (ξ,θ) off-diagonal blocks are set equal here
    (symmetric approximation); the true asymmetry from [∂_θ, ∂_ξ] = ∂_η
    is second-order and negligible for smooth fields.

    Returns
    -------
    H : dict with integer-tuple keys (i,j) → np.ndarray (H_img, W_img) float32
    """
    delta_theta = np.pi / No

    spatial = compute_xi_eta_derivatives(U_f_stack[i], theta_i, sigma_s)
    d_theta, d2_theta = _theta_derivatives(U_f_stack, i, No, delta_theta, sigma_s)
    d_theta_d_xi, d_theta_d_eta = _cross_theta_terms(U_f_stack, i, No, delta_theta, sigma_s)

    H = {
        (0, 0): spatial['d2_xi_xi'],
        (0, 1): spatial['d2_xi_eta'],
        (0, 2): d_theta_d_xi,
        (1, 0): spatial['d2_xi_eta'],   # symmetric in spatial part
        (1, 1): spatial['d2_eta_eta'],
        (1, 2): d_theta_d_eta,
        (2, 0): d_theta_d_xi,
        (2, 1): d_theta_d_eta,
        (2, 2): d2_theta,
    }
    return H


# ---------------------------------------------------------------------------
# μ-normalised symmetrised Hessian
# ---------------------------------------------------------------------------

def compute_H_mu(H: dict, mu: float) -> Array:
    """
    Form the symmetrised μ-normalised Hessian:

        H_μ = M_μ⁻¹ H^T M_μ⁻² H M_μ⁻¹

    where M_μ = diag(μ, μ, 1), hence M_μ⁻¹ = diag(1/μ, 1/μ, 1).

    With diagonal scaling matrices this simplifies to:

        B     = H ⊙ d1[j]           (scale columns of H by M_μ⁻¹)
        C     = B ⊙ d2[i]           (scale rows  of B by M_μ⁻²)
        D     = H^T · C             (matrix multiply per pixel)
        H_μ   = D ⊙ d1[i]          (scale rows  of D by M_μ⁻¹)

    The result is symmetric by construction.

    Parameters
    ----------
    H : dict (i,j) → (H_img, W_img) float32
    mu : float

    Returns
    -------
    H_mu : (H_img, W_img, 3, 3) float32
    """
    H_img, W_img = H[(0, 0)].shape

    H_arr = np.stack(
        [H[(i, j)] for i in range(3) for j in range(3)],
        axis=-1,
    ).reshape(H_img, W_img, 3, 3).astype(np.float32)   # (H, W, 3, 3)

    d1 = np.array([1.0 / mu, 1.0 / mu, 1.0], dtype=np.float32)   # M_μ⁻¹ diagonal
    d2 = d1 ** 2                                                    # M_μ⁻² diagonal

    # B = H @ M_μ⁻¹  →  B[...,i,j] = H[...,i,j] * d1[j]
    B = H_arr * d1[None, None, None, :]

    # C = M_μ⁻² @ B  →  C[...,i,j] = B[...,i,j] * d2[i]
    C = B * d2[None, None, :, None]

    # D = H^T @ C  →  D[...,i,j] = Σ_k H[...,k,i] * C[...,k,j]
    D = np.einsum('...ki,...kj->...ij', H_arr, C)

    # H_mu = M_μ⁻¹ @ D  →  H_mu[...,i,j] = D[...,i,j] * d1[i]
    H_mu = D * d1[None, None, :, None]

    return H_mu.astype(np.float32)


# ---------------------------------------------------------------------------
# Eigenvector analysis → optimal tangent c*
# ---------------------------------------------------------------------------

def compute_optimal_tangent(H_mu: Array) -> Array:
    """
    Find the eigenvector of H_μ with the smallest eigenvalue at each pixel.

    H_μ is symmetric ⇒ use np.linalg.eigh (stable, eigenvalues ascending).

    Parameters
    ----------
    H_mu : (H, W, 3, 3) float32

    Returns
    -------
    c_tilde_star : (H, W, 3) float32
        Eigenvector for the smallest eigenvalue at each pixel.
        Caller must apply M_μ⁻¹ to obtain c* = M_μ⁻¹ c̃*.
    """
    H, W = H_mu.shape[:2]
    H_flat = H_mu.reshape(-1, 3, 3).astype(np.float64)   # (H*W, 3, 3)

    # eigh returns eigenvalues ascending → smallest at index 0
    _, eigvecs = np.linalg.eigh(H_flat)   # eigvecs[:, :, i] = i-th eigenvector

    c_tilde = eigvecs[:, :, 0]            # (H*W, 3) — smallest-eigenvalue eigenvec
    return c_tilde.reshape(H, W, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# κ and d_H
# ---------------------------------------------------------------------------

def compute_kappa_dH(c_star: Array, mu: float) -> tuple[Array, Array]:
    """
    Compute local curvature κ and deviation from horizontality d_H.

    From c* = (c_ξ, c_η, c_θ) in the LID frame:

        κ   = (c_θ · sign(c_ξ)) / √(c_ξ² + c_η²)
        d_H = arctan2(c_η, c_ξ)

    Parameters
    ----------
    c_star : (H, W, 3) float32 — c* = M_μ⁻¹ c̃*
    mu : float  — unused here but kept for API clarity

    Returns
    -------
    kappa : (H, W) float32
    d_H   : (H, W) float32  in (−π/2, π/2]
    """
    eps = 1e-8
    c_xi  = c_star[:, :, 0]
    c_eta = c_star[:, :, 1]
    c_th  = c_star[:, :, 2]

    kappa = (c_th * np.sign(c_xi + eps)) / (np.sqrt(c_xi**2 + c_eta**2) + eps)
    d_H   = np.arctan2(c_eta, c_xi + eps)

    return kappa.astype(np.float32), d_H.astype(np.float32)


# ---------------------------------------------------------------------------
# LAD frame direction e_b
# ---------------------------------------------------------------------------

def compute_e_b(kappa: Array, d_H: Array, mu: float) -> Array:
    """
    Compute the unit vector e_b of the LAD frame at each pixel.

    The LAD frame is obtained via two rotations of the LID frame:
        (∂_a, ∂_b, ∂_c)^T = Q^T_{κ,μ} · R̃^T_{d_H} · (∂_ξ, ∂_η, μ∂_θ)^T

    The b-direction satisfies:
        e_b = Q^T R̃^T [0, 1, 0]^T

    Since Q^T applied to [0,1,0]^T leaves it unchanged (b lies in the ξ-η plane),
        e_b = R̃^T_{d_H} [0, 1, 0]^T = [sin(d_H), cos(d_H), 0]^T

    Sanity check: d_H = 0 (horizontal vessel) → e_b = [0, 1, 0] = ∂_η direction. ✓

    Parameters
    ----------
    kappa : (H, W) float32  — not used in e_b formula but kept for completeness
    d_H   : (H, W) float32
    mu    : float

    Returns
    -------
    e_b : (H, W, 3) float32
    """
    zeros = np.zeros_like(kappa, dtype=np.float32)
    e_b = np.stack([np.sin(d_H), np.cos(d_H), zeros], axis=-1)
    return e_b.astype(np.float32)


# ---------------------------------------------------------------------------
# ∂²_b via Hessian projection
# ---------------------------------------------------------------------------

def compute_d2_b(H_uf_arr: Array, e_b: Array) -> Array:
    """
    Compute ∂²_b U_f = e_b^T H_Uf e_b at each pixel.

    Parameters
    ----------
    H_uf_arr : (H, W, 3, 3) float32
    e_b      : (H, W, 3) float32

    Returns
    -------
    d2_b : (H, W) float32
    """
    return np.einsum('...i,...ij,...j->...', e_b, H_uf_arr, e_b).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-layer LAD filter response
# ---------------------------------------------------------------------------

def lad_filter_response(
    U_f_stack: Array,
    i: int,
    theta_i: float,
    sigma_s: float,
    sigma_o: float,
    mu: float,
    No: int,
) -> Array:
    """
    LAD filter response for orientation layer i at spatial scale σs.

    Φ^{σs,σo}_{b,norm} = −μ⁻² ∂²_b (G_{σs,σo} ★ U_f)

    Returns (H, W) float32 ≥ 0.
    """
    # Step 1 — left-invariant Hessian
    H_uf = assemble_hessian(U_f_stack, i, No, theta_i, sigma_s, sigma_o, mu)

    # Step 2 — symmetrised μ-normalised Hessian
    H_mu = compute_H_mu(H_uf, mu)

    # Step 3 — smallest eigenvector c̃*, then c* = M_μ⁻¹ c̃*
    c_tilde = compute_optimal_tangent(H_mu)
    inv_w   = np.array([1.0 / mu, 1.0 / mu, 1.0], dtype=np.float32)
    c_star  = c_tilde * inv_w[None, None, :]

    # Step 4 — κ and d_H
    kappa, d_H = compute_kappa_dH(c_star, mu)

    # Step 5 — e_b direction in LID frame
    e_b = compute_e_b(kappa, d_H, mu)

    # Step 6 — assemble H_Uf as 4-D array for projection
    H_img, W_img = kappa.shape
    H_arr = np.stack(
        [H_uf[(ii, jj)] for ii in range(3) for jj in range(3)],
        axis=-1,
    ).reshape(H_img, W_img, 3, 3).astype(np.float32)

    # Step 7 — ∂²_b = e_b^T H_Uf e_b
    d2_b = compute_d2_b(H_arr, e_b)

    # Step 8 — scale-normalised response
    response = -(1.0 / mu**2) * d2_b
    return np.maximum(response, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Full LAD-OS enhancement pipeline
# ---------------------------------------------------------------------------

def lad_os_enhance(
    image: Array,
    wavelets: Array,
    scales: list,
    sigma_o: float,
    No: int,
) -> Array:
    """
    Full LAD-OS vessel enhancement.

    Υ(f)(x) = max_{i=0..No-1} { Σ_{σs ∈ S} Φ^{σs,σo}_{b,norm}(U_f)(x, θ_i) }

    Parameters
    ----------
    image : (H, W) float32
    wavelets : (No, H, W) complex64
    scales : list of float
    sigma_o : float
    No : int

    Returns
    -------
    enhanced : (H, W) float32
    """
    H, W = image.shape

    # Compute full real-valued orientation score stack (needed for ∂_θ)
    U_f_stack = build_orientation_score_stack(image, wavelets)

    # Angular smoothing of the full stack
    U_f_stack = angular_smooth_stack(U_f_stack, sigma_o, No)

    enhanced = np.zeros((H, W), dtype=np.float32)

    for i in range(No):
        theta_i = i * np.pi / No
        orientation_response = np.zeros((H, W), dtype=np.float32)

        for sigma_s in scales:
            mu = sigma_o / sigma_s
            orientation_response += lad_filter_response(
                U_f_stack, i, theta_i, sigma_s, sigma_o, mu, No
            )

        enhanced = np.maximum(enhanced, orientation_response)

    return enhanced
