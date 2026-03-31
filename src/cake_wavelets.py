"""
SPEC_02 — Cake Wavelet Construction
=====================================
Cake wavelets are anisotropic kernels used to lift a 2D image into its
orientation score.  They are built entirely in the Fourier domain, which
gives direct control over angular and radial selectivity.

Mathematical definition (Bekkers et al., 2014, Section 2.3):

    psi_tilde_cake(omega) = B_k( (phi - pi/2 - theta) / s_theta ) * M_N(rho)

where:
  omega = (rho cos phi, rho sin phi)  are polar Fourier coordinates
  s_theta = 2pi / No           is the angular resolution per slice
  B_k                     is the k-th order B-spline  (paper: k=2, quadratic)
  M_N(rho)                  is the radial window (smooth low-pass)

Default parameters for all datasets in the paper:
  No = 16,  k = 2,  N = 60,  gamma = 0.9
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.special import gammaincc

Array = np.ndarray


# Parameters dataclass

@dataclass(frozen=True)
class CakeWaveletParams:
    """
    Parameters for the cake wavelet bank.

    Attributes
    ----------
    No : int
        Number of orientation slices.  Paper uses 16 for all datasets.
    k : int
        B-spline order.  Paper uses 2 (quadratic).
    N : int
        Taylor order for the radial window M_N.  Paper uses 60.
    gamma : float
        Fraction of Nyquist at which M_N has its inflection.  Paper: 0.9.
    spatial_window_sigma : float or None
        If set, multiply each spatial-domain wavelet by a Gaussian of this
        sigma (pixels) to avoid long tails (Eq. 12 of the paper).
        Recommended: min(H, W) / 6  or ~80–100 px for DRIVE.
    """
    No: int = 16
    k: int = 2
    N: int = 60
    gamma: float = 0.9
    spatial_window_sigma: Optional[float] = None


# Radial window  M_N
def make_radial_window(
    rho: Array,
    N: int = 60,
    gamma: float = 0.9,
    rho_nyq: float = np.pi,
) -> Array:
    """
    Compute the radial window M_N as defined in SPEC_02.

    M_N(rho^2/t) = exp(-rho^2/t) * Sum_{k=0}^{N} (rho^2/t)^k / k!

    This equals the *regularised upper* incomplete gamma Q(N+1, u) = Gamma(N+1, u)/Gamma(N+1)
    (scipy.special.gammaincc), which gives M_N(0)=1 and M_N(inf)=0 as expected.
    Note: lower incomplete gamma (gammainc) is the complement and gives the wrong
    result — 0 in the passband and 1 at high frequencies.

    Parameters
    ----------
    rho : array of radial frequencies (rad/px); Nyquist = pi
    N : Taylor order  (paper: 60)
    gamma : inflection fraction of Nyquist  (paper: 0.9)
    rho_nyq : Nyquist frequency in the same units as rho  (default pi)

    Returns
    -------
    float32 array of the same shape as rho, values in [0, 1].
    """
    t = 2.0 * (gamma * rho_nyq) ** 2 / (1.0 + 2.0 * N)
    u = (rho.astype(np.float64) ** 2) / (t + 1e-300)
    # gammaincc(a, x) = regularised upper incomplete gamma Q(a, x) = 1 - P(a, x)
    # Q(a, 0) = 1  (full passband at DC),  Q(a, inf) = 0  (zero at very high freq)
    M = gammaincc(N + 1.0, u)
    return np.clip(M, 0.0, 1.0).astype(np.float32)


# Angular B-spline window
def _bspline2(x: Array) -> Array:
    """
    Quadratic B-spline B_2 evaluated at positions x.

    B_2(x) =
        3/4 - x^2              if |x| < 1/2
        (3/2 - |x|)^2 / 2      if 1/2 <= |x| < 3/2
        0                      if |x| >= 3/2
    """
    ax = np.abs(x).astype(np.float32)
    result = np.zeros_like(ax)
    m1 = ax < 0.5
    m2 = (ax >= 0.5) & (ax < 1.5)
    result[m1] = 0.75 - ax[m1] ** 2
    result[m2] = 0.5 * (1.5 - ax[m2]) ** 2
    return result


def make_bspline_angular(phi: Array, theta: float, No: int, k: int = 2) -> Array:
    """
    Double-sided angular B-spline window for orientation theta.

    SPEC_02 uses the half-circle convention (theta_i in [0, pi)), so each wavelet
    must respond to *both* angle theta and the antipodal angle theta + pi.  This is
    what "double-sided" means: vessels look the same in both directions.

    The window is therefore the sum of two B-spline lobes:
        w(phi) = B_k( wrap(phi - theta) / s_theta )  +  B_k( wrap(phi - theta - pi) / s_theta )

    with s_theta = 2pi / No.  The two lobes sit on opposite sides of the origin,
    giving uniform coverage of the full Fourier circle when all No orientations
    are summed — i.e., a flat admissibility function M_psi.

    Parameters
    ----------
    phi : (H, W) array of Fourier angles in (-pi, pi]
    theta : centre orientation (rad), half-circle convention
    No : total number of orientations
    k : B-spline order (must be 2)

    Returns
    -------
    float32 (H, W) in [0, inf).
    """
    if k != 2:
        raise NotImplementedError("Only quadratic B-spline (k=2) is implemented.")
    s_theta = 2.0 * np.pi / float(No)

    def _lobe(offset: float) -> Array:
        dphi = ((phi - offset) + np.pi) % (2.0 * np.pi) - np.pi
        return _bspline2(dphi / s_theta)

    w = _lobe(theta) + _lobe(theta + np.pi)
    return np.clip(w, 0.0, None)


# Main builder
def build_cake_wavelets(
    image_shape: Tuple[int, int],
    No: int = 16,
    N_radial: int = 60,
    gamma: float = 0.9,
    spatial_window_sigma: Optional[float] = None,
    order: int = 2,
) -> Array:
    """
    Build the full cake wavelet bank in the Fourier domain.

    For each orientation theta_i = i*pi/No (half-circle convention):
        psi_tilde_{theta_i}(omega) = B_k( wrap(phi - theta_i) / s_theta ) * M_N(rho)
    Optionally multiplied by a spatial Gaussian window.

    Parameters
    ----------
    image_shape : (H, W)
    No : number of orientations
    N_radial : radial window Taylor order
    gamma : radial window inflection fraction of Nyquist
    spatial_window_sigma : Gaussian window sigma in pixels (None = no windowing)
    order : B-spline order (must be 2)

    Returns
    -------
    wavelets : complex64 array of shape (No, H, W).
        The real part of each spatial-domain wavelet is the even (ridge-detecting)
        filter; the imaginary part is the odd (edge-detecting) filter.
    """
    H, W = image_shape

    # Step 1: Fourier coordinate grids
    fy = np.fft.fftfreq(H) * 2.0 * np.pi   # (H,)
    fx = np.fft.fftfreq(W) * 2.0 * np.pi   # (W,)
    FX, FY = np.meshgrid(fx, fy)            # (H, W)

    rho = np.sqrt(FX ** 2 + FY ** 2).astype(np.float32)
    phi = np.arctan2(FY, FX).astype(np.float32)

    # Step 2: Radial window
    M = make_radial_window(rho, N=N_radial, gamma=gamma)

    # Step 3: Per-orientation angular slice
    wavelets = np.zeros((No, H, W), dtype=np.complex64)

    for i in range(No):
        # Half-circle convention: theta_i = i*pi/No in [0, pi)
        theta_i = i * np.pi / float(No)

        B = make_bspline_angular(phi, theta=theta_i, No=No, k=order)
        Psi_hat = (B * M).astype(np.float32)   # real Fourier spectrum

        # Inverse FFT → spatial domain wavelet
        # Psi_hat uses standard fftfreq layout (DC at [0, 0])
        psi_spatial = np.fft.ifft2(Psi_hat)    # complex (H, W)

        # Optional: multiply by spatial Gaussian window (Eq. 12)
        if spatial_window_sigma is not None:
            sigma = float(spatial_window_sigma)
            # Pixel coordinates centred at the DC corner (FFT convention)
            yc = np.fft.fftfreq(H) * H          # (H,)
            xc = np.fft.fftfreq(W) * W          # (W,)
            XC, YC = np.meshgrid(xc, yc)
            G = np.exp(-(XC ** 2 + YC ** 2) / (2.0 * sigma ** 2 + 1e-12))
            psi_spatial = psi_spatial * G

        wavelets[i] = psi_spatial.astype(np.complex64)

    return wavelets


# Convenience wrapper matching the CakeWaveletParams dataclass API
def cake_wavelet_bank(
    image_shape: Tuple[int, int],
    params: Optional[CakeWaveletParams] = None,
) -> Array:
    """
    Build cake wavelets from a CakeWaveletParams object.

    Returns complex64 (No, H, W) — spatial-domain wavelets.
    """
    if params is None:
        params = CakeWaveletParams()
    return build_cake_wavelets(
        image_shape=image_shape,
        No=params.No,
        N_radial=params.N,
        gamma=params.gamma,
        spatial_window_sigma=params.spatial_window_sigma,
        order=params.k,
    )


def cake_wavelet_bank_fft(
    image_shape: Tuple[int, int],
    params: Optional[CakeWaveletParams] = None,
) -> Array:
    """
    Build cake wavelets and return their FFT spectra (No, H, W) complex64.

    This is the format needed by the orientation score transform (SPEC_03).
    """
    psi_spatial = cake_wavelet_bank(image_shape, params)
    psi_fft = np.stack(
        [np.fft.fft2(psi_spatial[i]) for i in range(psi_spatial.shape[0])],
        axis=0,
    ).astype(np.complex64)
    return psi_fft


# Diagnostics

def wavelet_admissibility(wavelets: Array) -> Array:
    """
    Compute the discrete admissibility function M_psi(omega).

    M_psi(omega) = Sum_i |psi_hat_i(omega)|^2

    Should be approximately constant (~= 1 after normalisation) in the passband.
    A flat M_psi ensures an approximate stable inverse transform.

    Parameters
    ----------
    wavelets : (No, H, W) complex array — *spatial* domain wavelets.

    Returns
    -------
    float32 (H, W).
    """
    M = np.zeros(wavelets.shape[1:], dtype=np.float64)
    for i in range(wavelets.shape[0]):
        M += np.abs(np.fft.fft2(wavelets[i])) ** 2
    return M.astype(np.float32)
