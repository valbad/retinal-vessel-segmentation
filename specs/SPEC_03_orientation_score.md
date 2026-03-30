# SPEC 03 — Orientation Score Transform

## File: `src/orientation_score.py`

This module implements:
1. **Forward transform** `Wψ`: lifts a 2D image `f(x)` to a 3D orientation score `U_f(x, θ)`
2. **Inverse transform** `Wψ*`: reconstructs a 2D image from an orientation score

The orientation score is the central object of the method: it is a function on the Lie group SE(2) = R² ⋊ S¹ (positions × orientations).

---

## Mathematical Definition

### Forward Transform

```
U_f(x, θ) = (Wψ f)(x, θ) = ∫_{R²} ψ̄(R_{-θ}(y − x)) f(y) dy
```

This is a **convolution** of the image `f` with the rotated conjugate wavelet `ψ̄_θ`:
```
U_f(x, θ) = (ψ̄_θ ★ f)(x)
```
where `ψ_θ(x) = ψ(R_{-θ} x)` is the wavelet rotated by angle `θ`, and `★` denotes convolution.

**In Fourier space**, convolution becomes multiplication:
```
F[U_f(·, θ)](ω) = F[ψ̄_θ](ω) · F[f](ω) = F[ψ_θ](−ω)* · F[f](ω)
```

For real-valued images and real-valued cake wavelets (we use only the real part):
```
F[U_f(·, θ)](ω) = ψ̃_θ(ω) · F̂(ω)
```
where `ψ̃_θ(ω) = ψ̃_cake(R_{-θ} ω)` and `F̂ = F[f]`.

### Inverse Transform (Approximate Reconstruction)

For cake wavelets with the partition-of-unity property `M_ψ ≈ 1`:
```
f_approx(x) ≈ (1/2π) ∫₀^{2π} U_f(x, θ) dθ
```

In the discrete case:
```
f_approx(x) = (1/No) Σ_{i=0}^{No-1} Re[U_f(x, θ_i)]
```

This approximate reconstruction is used for verification. In the actual segmentation pipeline, we do **not** reconstruct `f` from the score — instead we compute filtered versions of the score and project them back.

---

## Implementation

```python
import numpy as np
from typing import List, Tuple


def orientation_score_transform(
    image: np.ndarray,
    wavelets: np.ndarray,
) -> np.ndarray:
    """
    Compute the invertible orientation score U_f from image f.

    Parameters
    ----------
    image : np.ndarray of shape (H, W), float32
        Preprocessed single-channel image.
    wavelets : np.ndarray of shape (No, H, W), complex64
        Cake wavelets as returned by build_cake_wavelets().
        wavelets[i] is the spatial-domain wavelet at orientation θ_i.

    Returns
    -------
    U_f : np.ndarray of shape (No, H, W), complex64
        Orientation score: U_f[i] = U_f(x, θ_i).
    """
    H, W = image.shape
    No = wavelets.shape[0]

    # Pre-compute FFT of image
    F_image = np.fft.fft2(image)  # shape (H, W), complex128

    U_f = np.zeros((No, H, W), dtype=np.complex64)

    for i in range(No):
        # FFT of wavelet (already in spatial domain from build_cake_wavelets)
        F_wavelet = np.fft.fft2(wavelets[i])  # complex64

        # Convolution in Fourier domain: multiply F[ψ̄_θ] * F[f]
        # ψ̄ is the complex conjugate of ψ
        # F[ψ̄_θ ★ f] = conj(F[ψ_θ]) * F[f]
        # (Note: for real wavelets, conj(F[ψ]) = F[ψ] but let's be general)
        F_U_i = np.conj(F_wavelet) * F_image

        # Inverse FFT to get U_f(·, θ_i)
        U_f[i] = np.fft.ifft2(F_U_i).astype(np.complex64)

    return U_f


def inverse_orientation_score(
    U_f: np.ndarray,
    wavelets: np.ndarray,
) -> np.ndarray:
    """
    Approximate reconstruction of image from orientation score.
    Uses the simple averaging formula valid when M_ψ ≈ 1.

    Parameters
    ----------
    U_f : np.ndarray of shape (No, H, W), complex64
    wavelets : np.ndarray of shape (No, H, W), complex64
        Not actually used in approximate reconstruction; included for API symmetry.

    Returns
    -------
    f_reconstructed : np.ndarray of shape (H, W), float32
    """
    No = U_f.shape[0]
    # Average real parts over orientations
    f_rec = np.mean(np.real(U_f), axis=0).astype(np.float32)
    return f_rec
```

---

## Efficient Computation via FFT

The key efficiency insight: **all No orientation layers can be computed from a single FFT of the image.** Only the wavelet FFTs vary across orientations.

For large images (HRF: 3504×2336), computing all No=16 orientation layers at once requires:
- 1 FFT of the image: O(HW log HW)
- No FFTs of wavelets: O(No × HW log HW) — can be precomputed once
- No pointwise multiplications: O(No × HW)
- No inverse FFTs: O(No × HW log HW)

**Memory layout:** `U_f[i, y, x]` = orientation score at position (x,y) and orientation `θ_i`.

**For very large images** (HRF), process the orientation score one layer at a time to avoid excessive memory use. The enhancement filters can be computed and accumulated (max-projection) without storing all layers simultaneously.

---

## Efficient One-Layer-at-a-Time Pipeline

For the segmentation pipeline, we never need all orientation layers simultaneously. The multi-scale maximum projection can be accumulated on-the-fly:

```python
def compute_enhanced_image(
    image: np.ndarray,
    wavelets: np.ndarray,
    filter_fn,  # callable: (U_f_layer, theta_i, sigma_s, sigma_o, mu) -> response
    scales: List[float],
    sigma_o: float,
    mu_scale: float,
) -> np.ndarray:
    """
    Compute the multi-scale, multi-orientation enhanced image Υ(f).
    
    Υ(f)(x) = max_{θ_i} { Σ_{σs ∈ S} Φ_{norm}(U_f)(x, θ_i) }
    
    Memory efficient: computes and accumulates one orientation layer at a time.
    
    Parameters
    ----------
    image : np.ndarray (H, W) float32
    wavelets : np.ndarray (No, H, W) complex64
    filter_fn : callable
        Function that takes one orientation layer U_f[i] and returns
        the multi-scale filtered response for that orientation.
    scales : list of float
        Spatial scales σs in pixels.
    sigma_o : float
        Angular scale in radians.
    mu_scale : float
        mu = sigma_o / sigma_s (computed per scale).

    Returns
    -------
    enhanced : np.ndarray (H, W) float32
        Υ(f): maximum over orientations of summed-scale responses.
    """
    H, W = image.shape
    No = wavelets.shape[0]

    # Pre-compute image FFT once
    F_image = np.fft.fft2(image.astype(np.float64))

    # Accumulate max over orientations
    enhanced = np.zeros((H, W), dtype=np.float32)

    for i in range(No):
        # Compute orientation score layer i
        F_wavelet = np.fft.fft2(wavelets[i].astype(np.complex128))
        U_f_i = np.fft.ifft2(np.conj(F_wavelet) * F_image).real.astype(np.float32)

        # Apply multi-scale filter for this orientation layer
        # Sum over scales
        layer_response = np.zeros((H, W), dtype=np.float32)
        for sigma_s in scales:
            mu = sigma_o / sigma_s
            response = filter_fn(U_f_i, i, sigma_s, sigma_o, mu)
            layer_response += response

        # Max projection over orientations
        enhanced = np.maximum(enhanced, layer_response)

    return enhanced
```

---

## The Role of the Imaginary Part

Cake wavelets are **quadrature filters**: the real part detects symmetric (even) structures like ridges/lines, while the imaginary part detects antisymmetric (odd) structures like edges.

For **vessel detection**, we use only the **real part** of the orientation score:
- `Re[U_f(x, θ)]` responds to ridge-like (vessel cross-section) structures
- Vessels appear as ridges in the green channel image (bright on dark background after preprocessing)

In the LAD-OS paper, only real-valued cake wavelets are used (confirmed in Section II.C: "we choose the real-valued cake wavelets").

**In practice:** When computing derivative filters on the orientation score, we differentiate `Re[U_f(x, θ)]` only.

---

## Left-Invariance and SE(2) Geometry

The orientation score lives on SE(2) — the roto-translation group. This has important consequences:

1. **Left-invariance:** All operations (derivatives, Gaussian filtering) on the score must use the **left-invariant frame** `{∂_ξ, ∂_η, ∂_θ}`, not the Cartesian frame `{∂_x, ∂_y, ∂_θ}`.

2. **The left-invariant frame at orientation θ:**
   ```
   ∂_ξ = cos θ ∂_x + sin θ ∂_y    (along vessel direction)
   ∂_η = −sin θ ∂_x + cos θ ∂_y  (perpendicular to vessel)
   ∂_θ = ∂_θ                       (in orientation direction)
   ```

3. **Practical consequence:** At orientation layer `θ_i`, the derivative perpendicular to the vessel (`∂_η`) is computed by rotating the Cartesian derivatives by `−θ_i`. This is the key operation in the LID filter (see SPEC_04).

4. **Euclidean invariance:** Because the score uses the left-invariant frame, processing the orientation score with left-invariant operators corresponds to Euclidean-invariant processing of the original image. Rotating/translating the input image before processing gives the same result as rotating/translating the output after processing.

---

## Data Layout Convention

Throughout the codebase, use the following convention:

```python
# Orientation score
U_f.shape == (No, H, W)
U_f[i, y, x]  # value at position (x, y) and orientation θ_i = i*π/No

# Wavelets
wavelets.shape == (No, H, W)
wavelets[i]    # spatial-domain wavelet at orientation θ_i

# Image coordinates
# (0,0) = top-left, y increases downward, x increases rightward
# This matches standard numpy/image conventions
```

---

## Verification Tests (`tests/test_orientation_score.py`)

```python
def test_orientation_score_shape():
    No = 16
    H, W = 128, 128
    f = np.random.rand(H, W).astype(np.float32)
    wavelets = build_cake_wavelets((H, W), No)
    U_f = orientation_score_transform(f, wavelets)
    assert U_f.shape == (No, H, W)
    assert U_f.dtype == np.complex64

def test_reconstruction_accuracy():
    No = 16
    H, W = 128, 128
    f = np.zeros((H, W), dtype=np.float32)
    # Draw a line
    f[60:70, :] = 1.0
    wavelets = build_cake_wavelets((H, W), No)
    U_f = orientation_score_transform(f, wavelets)
    f_rec = inverse_orientation_score(U_f, wavelets)
    rel_err = np.linalg.norm(f_rec - f) / (np.linalg.norm(f) + 1e-8)
    assert rel_err < 0.1  # should be < 10% for cake wavelets

def test_orientation_selectivity():
    """Horizontal line should give strongest response in horizontal orientation layer."""
    No = 16
    H, W = 128, 128
    f = np.zeros((H, W), dtype=np.float32)
    f[64, 20:108] = 1.0  # horizontal line at y=64
    wavelets = build_cake_wavelets((H, W), No)
    U_f = orientation_score_transform(f, wavelets)
    # Horizontal = θ = π/2 = layer No//2
    responses = [np.max(np.abs(U_f[i, 64, :])) for i in range(No)]
    peak_layer = np.argmax(responses)
    # Should be near No//2 (θ = π/2 for horizontal)
    assert abs(peak_layer - No // 2) <= 2  # within 2 layers

def test_left_invariance():
    """Rotating image by 90° should shift orientation score layers by No//4."""
    No = 16
    H = W = 64
    f = np.zeros((H, W), dtype=np.float32)
    f[32, 10:54] = 1.0  # horizontal line
    wavelets = build_cake_wavelets((H, W), No)
    
    U_f_orig = orientation_score_transform(f, wavelets)
    
    # Rotate image 90°
    f_rot = np.rot90(f)
    U_f_rot = orientation_score_transform(f_rot, wavelets)
    
    # Peak orientation should shift by No//4 (90°)
    peak_orig = np.argmax([np.max(np.abs(U_f_orig[i])) for i in range(No)])
    peak_rot = np.argmax([np.max(np.abs(U_f_rot[i])) for i in range(No)])
    shift = (peak_rot - peak_orig) % No
    assert abs(shift - No // 4) <= 1
```
