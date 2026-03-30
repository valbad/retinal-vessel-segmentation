# SPEC 02 — Cake Wavelet Construction

## File: `src/cake_wavelets.py`

Cake wavelets are the anisotropic kernels used to lift a 2D image into its orientation score. They are constructed entirely in the **Fourier domain**, which gives direct control over angular and radial selectivity. The real part of cake wavelets responds to symmetric (ridge/line) structures — which is what we use for vessel filtering.

### Reference
Bekkers, E., Duits, R., Berendschot, T., & ter Haar Romeny, B. (2014). *A Multi-Orientation Analysis Approach to Retinal Vessel Tracking.* JMIV, 49(3), 583–610. (Section 2.3)

---

## Mathematical Definition

A cake wavelet in the Fourier domain is:

```
ψ̃_cake(ω) = B_k( (φ mod 2π − π/2) / s_θ ) · M_N(ρ)
```

where:
- `ω = (ρ cos φ, ρ sin φ)` are polar Fourier coordinates
- `ρ = |ω|` is the radial frequency (magnitude)
- `φ = atan2(ω_y, ω_x)` is the angular coordinate
- `s_θ = 2π / No` is the angular resolution per orientation slice
- `No` is the total number of orientations
- `B_k` is the k-th order B-spline (paper uses k=2, quadratic)
- `M_N(ρ)` is the radial window function

The spatial domain wavelet is obtained as:
```
ψ_cake(x) = F⁻¹[ ψ̃_cake(ω) · G_{σs}(ω) ]
```
where `G_{σs}` is a Gaussian window in the spatial domain (applied in Fourier domain as a multiplication, standard deviation `σs` large, typically `σs = image_size / 6` to avoid long tails).

For orientation `θ_i`, the rotated wavelet in the Fourier domain is:
```
ψ̃_{θ_i}(ω) = ψ̃_cake(R_{-θ_i} ω)
```
which corresponds to rotating the angular argument `φ → φ − θ_i`.

---

## Radial Window M_N

The radial window ensures a stable inverse transform by making `M_ψ(ω) ≈ 1` for all relevant frequencies:

```
M_N(ρ²/t) = exp(−ρ²/t) · Σ_{k=0}^{N} (ρ²/t)^k / k!
```

where `t = 2(γΩ)² / (1 + 2N)` with:
- `Ω` = Nyquist frequency = `π` (in normalized units where image coords in [−π, π])
- `γ ∈ (0, 1)` controls where the inflection point is (use `γ = 0.9`)
- `N` controls the sharpness of the cutoff (use `N = 60` as in the paper, gives very flat response up to Nyquist)

**Practical note:** `M_N` is essentially 1 for `ρ < γΩ` and decays smoothly to 0 for `ρ > γΩ`. It replaces a hard low-pass with a smooth cutoff that avoids Gibbs artifacts.

---

## B-Spline B_k

Quadratic B-spline (k=2) is built by convolution:
```
B_0(x) = 1   if |x| < 1/2, else 0
B_1 = B_0 * B_0
B_2 = B_1 * B_0   (= quadratic B-spline)
```

The B-spline is used to smoothly partition the angular dimension. The sum of all No angular slices of B-splines sums to 1 over the full angular range, ensuring the partition of unity property (no angular frequency is double-counted or missed).

---

## Implementation

```python
import numpy as np
from scipy.signal import bspline

def make_radial_window(
    rho: np.ndarray,
    N: int = 60,
    gamma: float = 0.9,
) -> np.ndarray:
    """
    Compute M_N radial window function.

    Parameters
    ----------
    rho : np.ndarray
        Radial frequency values (normalized, Nyquist = π).
    N : int
        Taylor series order. Default 60 (very flat response).
    gamma : float
        Fraction of Nyquist at which inflection occurs. Default 0.9.

    Returns
    -------
    np.ndarray : M_N values, same shape as rho. Values in [0, 1].
    """
    Omega = np.pi  # Nyquist in normalized frequency
    t = 2 * (gamma * Omega)**2 / (1 + 2 * N)
    u = rho**2 / t

    # Compute M_N = exp(-u) * sum_{k=0}^{N} u^k / k!
    # Use log-space for numerical stability for large u
    result = np.zeros_like(rho, dtype=np.float64)
    log_exp_neg_u = -u  # log of exp(-u)

    # Taylor sum: accumulate exp(-u) * u^k / k!
    term = np.exp(-u)  # k=0 term
    result = term.copy()
    for k in range(1, N + 1):
        term = term * u / k
        result = result + term

    return result.astype(np.float32)


def make_bspline_angular(
    phi: np.ndarray,
    order: int = 2,
) -> np.ndarray:
    """
    Evaluate B-spline of given order at angular positions phi.
    The B-spline is defined on [-0.5, 0.5] * s_θ support.
    
    In practice, use scipy.signal.bspline or implement directly.
    """
    # B_k using scipy's bspline (order k means polynomial degree k)
    # bspline(x, n, ...) where n is the order
    return bspline(phi, order)


def build_cake_wavelets(
    image_shape: tuple,
    No: int,
    N_radial: int = 60,
    gamma: float = 0.9,
    gaussian_sigma_frac: float = 0.5,
    order: int = 2,
) -> np.ndarray:
    """
    Build the full set of No cake wavelet kernels in the spatial domain.

    Parameters
    ----------
    image_shape : tuple (H, W)
        Shape of the image to be transformed.
    No : int
        Number of orientation layers.
    N_radial : int
        Radial window Taylor order. Default 60.
    gamma : float
        Radial window inflection point fraction of Nyquist. Default 0.9.
    gaussian_sigma_frac : float
        Spatial Gaussian window sigma as fraction of min(H, W). Default 0.5.
        (Controls tails: larger = tighter spatial support = broader Fourier)
    order : int
        B-spline order. Default 2 (quadratic).

    Returns
    -------
    wavelets : np.ndarray of shape (No, H, W), complex64
        Real part of each wavelet is the even (ridge-detecting) filter.
        Imaginary part is the odd (edge-detecting) filter.
    """
```

**Detailed construction steps:**

#### Step 1: Build Fourier coordinate grids

```python
H, W = image_shape
# Frequency grids centered at 0, normalized so Nyquist = π
# Use fftshift convention
fy = np.fft.fftfreq(H) * 2 * np.pi  # shape (H,)
fx = np.fft.fftfreq(W) * 2 * np.pi  # shape (W,)
FX, FY = np.meshgrid(fx, fy)         # shape (H, W)

# Polar coordinates in frequency domain
rho = np.sqrt(FX**2 + FY**2)         # radial frequency
phi = np.arctan2(FY, FX)             # angle in [-π, π]
```

#### Step 2: Compute radial window M_N(rho)

```python
M = make_radial_window(rho, N=N_radial, gamma=gamma)
```

#### Step 3: For each orientation θ_i, build the angular slice

```python
s_theta = 2 * np.pi / No  # angular width of each slice

wavelets = np.zeros((No, H, W), dtype=np.complex64)

for i in range(No):
    theta_i = i * np.pi / No  # angles in [0, π) for double-sided wavelets

    # Rotate angular coordinate: shift phi by -theta_i
    phi_rot = (phi - theta_i + np.pi) % (2 * np.pi) - np.pi  # in (-π, π]

    # Normalize to B-spline input range
    # B-spline is evaluated at phi_rot / s_theta
    phi_normalized = phi_rot / s_theta

    # Evaluate quadratic B-spline
    # B_2 has support on [-3/2, 3/2] in normalized units
    # Use a direct implementation for accuracy
    B = bspline2(phi_normalized)  # see helper below

    # Fourier domain wavelet slice
    Psi_hat = B * M  # shape (H, W), real

    # Apply spatial Gaussian window (in Fourier domain it's a convolution
    # with a Gaussian, but it's easier to apply in spatial domain after ifft)
    # So: psi_spatial = ifft2(Psi_hat), then multiply by spatial Gaussian
    psi_spatial = np.fft.ifft2(np.fft.ifftshift(Psi_hat))

    # Spatial Gaussian window
    sigma_s = gaussian_sigma_frac * min(H, W)
    y_coords = np.fft.fftfreq(H) * H  # pixel coords centered
    x_coords = np.fft.fftfreq(W) * W
    YC, XC = np.meshgrid(y_coords, x_coords, indexing='ij')
    G_spatial = np.exp(-(XC**2 + YC**2) / (2 * sigma_s**2))

    psi_windowed = psi_spatial * G_spatial

    wavelets[i] = psi_windowed.astype(np.complex64)
```

#### Helper: Quadratic B-spline (B_2)

```python
def bspline2(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the quadratic B-spline B_2 at positions x.
    Support: x in [-3/2, 3/2].
    
    B_2(x) = 
        3/4 - x²                   for |x| < 1/2
        (3/2 - |x|)² / 2           for 1/2 ≤ |x| < 3/2
        0                           for |x| ≥ 3/2
    """
    ax = np.abs(x)
    result = np.zeros_like(x, dtype=np.float32)
    mask1 = ax < 0.5
    mask2 = (ax >= 0.5) & (ax < 1.5)
    result[mask1] = 0.75 - ax[mask1]**2
    result[mask2] = 0.5 * (1.5 - ax[mask2])**2
    return result
```

---

## Orientation Convention

- Angles `θ_i = i * π / No` for `i = 0, 1, ..., No-1`
- Range: `[0, π)` — half-circle convention (double-sided wavelets)
- `θ = 0` corresponds to a vertically oriented vessel (along x-axis in image coords)
- The B-spline slice at orientation `θ_i` responds maximally to structures aligned at angle `θ_i`

**Invertibility condition:**  
Because `Σ_i B_2(φ/s_θ - i) ≈ 1` (B-splines partition of unity) and `M_N ≈ 1` in the passband, the sum of all cake wavelet responses reconstructs the original image (approximately):
```
f_approx(x) ≈ (1/2π) ∫₀^π U_f(x, θ) dθ
```
In discrete form: `f_approx(x) = (1/No) Σ_i Re[U_f(x, θ_i)]`

---

## Output and Storage

For large images (e.g. HRF: 3504×2336), storing all No=16 complex orientation score layers simultaneously requires:
```
16 layers × 3504 × 2336 × 8 bytes (complex64) ≈ 830 MB
```

For the enhancement pipeline, we process **one scale at a time** and **accumulate the max-projection**. This avoids storing the full orientation score:

```python
# Memory-efficient approach: process scale-by-scale
enhanced = np.zeros(image_shape, dtype=np.float32)
for sigma_s in scales:
    for i, theta_i in enumerate(thetas):
        response = compute_filter_response(U_f_layer_i, sigma_s, sigma_o)
        enhanced = np.maximum(enhanced, response)
```

---

## Validation

To verify your wavelet construction:

1. **Partition of unity:** `Σ_i |ψ̃_{θ_i}(ω)|² ≈ 1` for most frequencies ω.
2. **Reconstruction test:** Apply `Wψ` to a simple test image (a line or cross), then reconstruct via `Wψ*`. Verify `||f_reconstructed − f||/||f|| < 0.01`.
3. **Orientation selectivity:** A vertical line image should produce high response only in the `θ ≈ 0` orientation layer and near-zero response in perpendicular layers.

```python
def test_partition_of_unity(image_shape=(128, 128), No=16):
    wavelets = build_cake_wavelets(image_shape, No)
    # Sum of |Fourier transforms|² should ≈ 1
    sum_power = np.zeros(image_shape)
    for i in range(No):
        W_hat = np.abs(np.fft.fft2(wavelets[i]))**2
        sum_power += W_hat
    # Should be ≈ No in the passband (each wavelet contributes ~1/No, summed No times)
    assert np.mean(sum_power[10:-10, 10:-10]) > 0.5  # rough check

def test_reconstruction(image_shape=(128, 128), No=16):
    wavelets = build_cake_wavelets(image_shape, No)
    f = np.zeros(image_shape, dtype=np.float32)
    f[50:60, 60] = 1.0  # vertical line segment
    
    # Forward transform
    U_f = orientation_score_transform(f, wavelets)
    # Inverse
    f_rec = inverse_orientation_score(U_f, wavelets)
    
    rel_error = np.linalg.norm(f_rec - f) / (np.linalg.norm(f) + 1e-8)
    assert rel_error < 0.05  # less than 5% reconstruction error
```

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| No | 16 | Number of orientations (used for all datasets in paper) |
| B-spline order k | 2 | Quadratic |
| N (radial) | 60 | Taylor order for M_N |
| γ | 0.9 | Inflection at 90% of Nyquist |
| Spatial Gaussian σs | large (≥ image_size/6) | Avoids spatial truncation |

All datasets in the paper use `No = 16`. The paper notes that LAD-OS achieves competitive AUC even with `No = 4` (due to local adaptivity), while LID-OS needs larger No.
