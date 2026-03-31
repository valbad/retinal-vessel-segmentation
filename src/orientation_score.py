"""
SPEC_03 — Orientation Score Transform
======================================
Lifts a 2D image f(x) to a 3D orientation score U_f(x, theta) on SE(2) = R^2 rtimes S^1.

Forward transform (Wpsi):
    U_f(x, theta_i) = (psi_bar_theta * f)(x)
    In Fourier:  F[U_f(*, theta_i)](omega) = conj(F[psi_i](omega)) * F[f](omega)

Inverse transform (approximate reconstruction, valid when M_psi ~= 1):
    f_approx(x) = (1/No) Sum_i Re[U_f(x, theta_i)]

Convention
----------
U_f.shape = (No, H, W),  U_f[i, y, x] = value at pixel (x,y) and orientation theta_i = i*pi/No.
Only the real part of U_f is used downstream (real-valued cake wavelets).
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def orientation_score_transform(
    image: Array,
    wavelets: Array,
) -> Array:
    """
    Forward orientation score transform.

    Parameters
    ----------
    image : (H, W) float32
        Preprocessed single-channel image.
    wavelets : (No, H, W) complex64
        Spatial-domain cake wavelets as returned by build_cake_wavelets().

    Returns
    -------
    U_f : (No, H, W) complex64
        Orientation score.  U_f[i] is the layer at orientation theta_i = i*pi/No.
    """
    H, W = image.shape
    No = wavelets.shape[0]

    # FFT of image — compute once, reuse across orientations
    F_image = np.fft.fft2(image.astype(np.float64))   # (H, W) complex128

    U_f = np.zeros((No, H, W), dtype=np.complex64)

    for i in range(No):
        # F[psi_i] — transform wavelet to Fourier domain
        F_wavelet = np.fft.fft2(wavelets[i].astype(np.complex128))

        # Correlation (convolution with conjugate):  conj(F[psi_i]) * F[f]
        F_U_i = np.conj(F_wavelet) * F_image

        # Back to spatial domain
        U_f[i] = np.fft.ifft2(F_U_i).astype(np.complex64)

    return U_f


def inverse_orientation_score(
    U_f: Array,
    wavelets: Array,   # accepted for API symmetry; not used in simple inversion
) -> Array:
    """
    Approximate reconstruction of the original image from its orientation score.

    Valid when the cake wavelets satisfy M_psi ~= 1 (partition-of-unity property).

        f_approx(x) = (1/No) Sum_i Re[U_f(x, theta_i)]

    Parameters
    ----------
    U_f : (No, H, W) complex64
    wavelets : (No, H, W) complex64  — unused, kept for API symmetry

    Returns
    -------
    f_rec : (H, W) float32
    """
    return np.mean(np.real(U_f), axis=0).astype(np.float32)


def compute_orientation_score_layer(
    F_image: Array,
    wavelet_i: Array,
) -> Array:
    """
    Compute a single orientation score layer from a precomputed image FFT.

    This is the inner loop body, exposed for use by the filter modules.

    Parameters
    ----------
    F_image : (H, W) complex array — FFT of the image (precomputed)
    wavelet_i : (H, W) complex64 — spatial-domain wavelet at one orientation

    Returns
    -------
    U_f_i : (H, W) float32 — real part of the orientation score layer
    """
    F_wavelet = np.fft.fft2(wavelet_i.astype(np.complex128))
    U_f_i = np.fft.ifft2(np.conj(F_wavelet) * F_image)
    return np.real(U_f_i).astype(np.float32)


def build_orientation_score_stack(
    image: Array,
    wavelets: Array,
) -> Array:
    """
    Compute the full real-valued orientation score stack.

    Returns the real parts of all No orientation layers as float32.
    Used by lad_os_enhance which needs access to adjacent layers for
    angular finite-difference derivatives.

    Parameters
    ----------
    image : (H, W) float32
    wavelets : (No, H, W) complex64

    Returns
    -------
    U_f_stack : (No, H, W) float32
    """
    No, H, W = wavelets.shape
    F_image = np.fft.fft2(image.astype(np.float64))
    U_f_stack = np.zeros((No, H, W), dtype=np.float32)
    for i in range(No):
        U_f_stack[i] = compute_orientation_score_layer(F_image, wavelets[i])
    return U_f_stack
