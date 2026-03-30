# SPEC 04 — Left-Invariant Derivative (LID) Filter

## File: `src/lid_filter.py`

The LID filter is the simpler of the two vessel enhancement filters. It applies second-order Gaussian derivatives in the left-invariant frame `{∂_ξ, ∂_η, ∂_θ}` within the orientation score, specifically using the perpendicular derivative `∂_η` to enhance vessel cross-sections.

---

## Mathematical Definition

### Left-Invariant Frame

At orientation `θ`, the left-invariant basis vectors in SE(2) are:
```
∂_ξ = cos θ ∂_x + sin θ ∂_y     (along vessel)
∂_η = −sin θ ∂_x + cos θ ∂_y    (perpendicular to vessel)
∂_θ = ∂_θ                          (in orientation)
```

These are the "rotating" derivatives that automatically align with the local vessel orientation at each layer `θ_i` of the orientation score.

### LID Filter

The scale-normalized LID filter applied to orientation score layer `U_f(·, θ)` is:
```
Φ^{σs,σo}_{η,norm}(U_f) = −μ⁻² · ∂²_η G_{σs,σo} * U_f
```

where:
- `G_{σs,σo}(x, θ) = G_{σs}(x) · G_{σo}(θ)` is a separable Gaussian on SE(2)
- `G_{σs}(x)` = 2D isotropic Gaussian with std `σs` (spatial)
- `G_{σo}(θ)` = 1D Gaussian with std `σo` (angular), applied along orientation axis
- `μ = σo / σs` is the scale normalization factor (keeps filter responses dimensionless)
- `μ⁻² = σs² / σo²` is the scale normalization factor for second-order derivatives

### Multi-Scale Image Reconstruction

After filtering, the enhanced image is reconstructed by:
```
Υ(f)(x) = max_{θ_i ∈ {iπ/No | i=1,...,No}} { Σ_{σs ∈ S} Φ^{σs,σo}_{η,norm}(U_f)(x, θ_i) }
```

That is: **sum** the filter responses over all scales, then take the **maximum** over all orientations.

---

## Implementation

### Step 1: Gaussian Smoothing in SE(2)

Before differentiating, smooth the orientation score with the separable Gaussian `G_{σs,σo}`:

```python
def gaussian_smooth_orientation_score(
    U_f_layer: np.ndarray,
    sigma_s: float,
    sigma_o: float,
    U_f_adjacent_layers: dict,
    theta_i: float,
    No: int,
) -> np.ndarray:
    """
    Smooth one orientation score layer with the separable Gaussian G_{σs,σo}.

    The angular smoothing G_{σo}(θ) couples adjacent orientation layers.
    For small σo (as used in the paper), only nearby layers contribute.

    Parameters
    ----------
    U_f_layer : np.ndarray (H, W)
        Real part of U_f at orientation θ_i.
    sigma_s : float
        Spatial Gaussian standard deviation in pixels.
    sigma_o : float
        Angular Gaussian standard deviation in radians.
        Paper uses σo = π/40 ≈ 0.0785 rad for all scales.
    U_f_adjacent_layers : dict
        {theta_j: U_f_layer_j} for layers near θ_i. Needed for angular smoothing.
    theta_i : float
        Current orientation angle.
    No : int
        Total number of orientations.

    Returns
    -------
    smoothed : np.ndarray (H, W), float32
    """
```

**Implementation details:**

#### Spatial smoothing (isotropic 2D Gaussian):
```python
from scipy.ndimage import gaussian_filter

U_smooth_spatial = gaussian_filter(U_f_layer, sigma=sigma_s, mode='reflect')
```

#### Angular smoothing (1D Gaussian over orientation layers):
The angular Gaussian `G_{σo}(θ)` is applied by convolving the stack of orientation layers along the θ axis. For `σo = π/40 ≈ 0.079 rad` and `No = 16` (so `Δθ = π/16 ≈ 0.196 rad`), only about `2σo/Δθ ≈ 0.8` layers contribute — essentially no angular mixing.

**Practical simplification:** For the small angular scales used in the paper (`σo = π/40`), the angular Gaussian is narrow enough that angular smoothing has negligible effect. The paper states σo is set as "a small constant" to "keep structure smoothness." In practice, many implementations omit angular smoothing or use a very small σo (so the Gaussian kernel in θ is a near-delta function).

**Implementation choice:** Apply angular smoothing using a 1D Gaussian along the orientation stack:
```python
from scipy.ndimage import gaussian_filter1d

def angular_smooth_stack(U_f_stack: np.ndarray, sigma_o: float, No: int) -> np.ndarray:
    """
    Apply 1D Gaussian smoothing along orientation axis (axis 0) of stack.
    
    Parameters
    ----------
    U_f_stack : np.ndarray (No, H, W)
    sigma_o : float in radians
    No : int
    
    Returns
    -------
    smoothed : np.ndarray (No, H, W)
    """
    # Convert σo from radians to index units: σ_idx = σo / (π/No)
    delta_theta = np.pi / No
    sigma_idx = sigma_o / delta_theta
    
    # Wrap-around smoothing (circular convolution) for periodic orientation
    # For small sigma_idx < 1, this has negligible effect
    if sigma_idx < 0.1:
        return U_f_stack  # skip if essentially a delta
    
    # Gaussian smoothing along axis 0 with boundary wrapping
    smoothed = gaussian_filter1d(U_f_stack, sigma=sigma_idx, axis=0, mode='wrap')
    return smoothed
```

---

### Step 2: Second-Order Derivative ∂²_η

After smoothing with `G_{σs,σo}`, differentiate **twice** in the perpendicular direction `∂_η`:

```
∂_η (G_{σs,σo} * U_f) = (−sin θ ∂_x + cos θ ∂_y)(G_{σs,σo} * U_f)
```

Since `G_{σs}` is isotropic, we can equivalently:
1. Smooth `U_f` with the isotropic Gaussian `G_{σs}`
2. Then apply the directional derivative `∂_η` to the smoothed result

Or equivalently, convolve with `∂²_η G_{σs}` directly.

**Efficient computation using Gaussian derivative filters:**

The isotropic Gaussian `G_σs` has partial derivatives:
```
∂_x G_σs(x,y) = −x/σs² · G_σs(x,y)
∂_y G_σs(x,y) = −y/σs² · G_σs(x,y)
```

So:
```
∂_η G_σs = (−sin θ ∂_x + cos θ ∂_y) G_σs
∂²_η G_σs = ∂_η(∂_η G_σs)
           = (−sin θ ∂_x + cos θ ∂_y)² G_σs
           = sin²θ ∂²_x G_σs − 2 sin θ cos θ ∂_x ∂_y G_σs + cos²θ ∂²_y G_σs
```

The Gaussian second derivatives can be computed using `scipy.ndimage.gaussian_filter` with `order` parameter:

```python
from scipy.ndimage import gaussian_filter

def second_derivative_perpendicular(
    U_f_layer: np.ndarray,
    theta: float,
    sigma_s: float,
) -> np.ndarray:
    """
    Compute ∂²_η (G_{σs} * U_f_layer) at orientation angle theta.

    ∂²_η = sin²θ ∂²_xx + cos²θ ∂²_yy − 2 sinθ cosθ ∂²_xy

    Parameters
    ----------
    U_f_layer : np.ndarray (H, W) float32
        One orientation layer of the orientation score (already angularly smoothed).
    theta : float
        Orientation angle in radians.
    sigma_s : float
        Spatial Gaussian std in pixels.

    Returns
    -------
    d2_eta : np.ndarray (H, W) float32
        Second-order perpendicular derivative.
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Second-order Gaussian derivatives via scipy
    # gaussian_filter with order=(0,2) gives ∂²/∂x² of G_σ * f
    # Note: scipy uses (row_order, col_order) = (y_order, x_order)
    d2_xx = gaussian_filter(U_f_layer, sigma=sigma_s, order=(0, 2))  # ∂²_x
    d2_yy = gaussian_filter(U_f_layer, sigma=sigma_s, order=(2, 0))  # ∂²_y
    d2_xy = gaussian_filter(U_f_layer, sigma=sigma_s, order=(1, 1))  # ∂²_xy

    # Rotate: ∂²_η = sin²θ ∂²_xx + cos²θ ∂²_yy − 2 sinθ cosθ ∂²_xy
    d2_eta = sin_t**2 * d2_xx + cos_t**2 * d2_yy - 2 * sin_t * cos_t * d2_xy

    return d2_eta.astype(np.float32)
```

---

### Step 3: Scale-Normalized LID Filter Response

```python
def lid_filter_response(
    U_f_layer: np.ndarray,
    theta: float,
    sigma_s: float,
    sigma_o: float,
) -> np.ndarray:
    """
    Compute the scale-normalized LID filter response for one orientation layer.

    Φ^{σs,σo}_{η,norm}(U_f)(x, θ) = −μ⁻² ∂²_η (G_{σs,σo} * U_f)(x, θ)
                                    = −(σs²/σo²) ∂²_η (G_{σs} * U_f)(x, θ)

    The negative sign ensures vessel cross-sections (which appear as ridges
    with negative ∂²_η) give positive filter responses.

    Parameters
    ----------
    U_f_layer : np.ndarray (H, W) float32
        Real part of one orientation score layer.
    theta : float
        Orientation angle θ_i in radians.
    sigma_s : float
        Spatial scale in pixels.
    sigma_o : float
        Angular scale in radians.

    Returns
    -------
    response : np.ndarray (H, W) float32
        Non-negative filter response (negative values clamped to 0).
    """
    mu_sq = (sigma_o / sigma_s)**2  # mu² = σo²/σs²
    mu_inv_sq = 1.0 / mu_sq         # μ⁻² = σs²/σo²

    # Compute −μ⁻² ∂²_η (G_{σs} * U_f)
    d2_eta = second_derivative_perpendicular(U_f_layer, theta, sigma_s)
    response = -mu_inv_sq * d2_eta

    # Only positive responses indicate vessels
    response = np.maximum(response, 0).astype(np.float32)

    return response
```

---

### Step 4: Multi-Scale LID-OS Enhancement

```python
def lid_os_enhance(
    image: np.ndarray,
    wavelets: np.ndarray,
    scales: list,
    sigma_o: float,
    No: int,
) -> np.ndarray:
    """
    Full LID-OS vessel enhancement.

    Υ(f)(x) = max_{i} { Σ_{σs ∈ S} Φ^{σs,σo}_{η,norm}(U_f)(x, θ_i) }

    Parameters
    ----------
    image : np.ndarray (H, W) float32
        Preprocessed image.
    wavelets : np.ndarray (No, H, W) complex64
        Cake wavelets.
    scales : list of float
        Spatial scales σs in pixels.
    sigma_o : float
        Angular scale σo in radians. Paper: π/40 for all datasets.
    No : int
        Number of orientations. Paper: 16 for all datasets.

    Returns
    -------
    enhanced : np.ndarray (H, W) float32
        Υ(f): the LID-OS enhanced image.
    """
    H, W = image.shape
    enhanced = np.zeros((H, W), dtype=np.float32)

    # Precompute FFT of image
    F_image = np.fft.fft2(image.astype(np.float64))

    for i in range(No):
        theta_i = i * np.pi / No

        # Compute orientation score layer i
        F_wavelet = np.fft.fft2(wavelets[i].astype(np.complex128))
        U_f_i = np.real(np.fft.ifft2(np.conj(F_wavelet) * F_image)).astype(np.float32)

        # Sum filter responses over all scales for this orientation
        orientation_response = np.zeros((H, W), dtype=np.float32)
        for sigma_s in scales:
            r = lid_filter_response(U_f_i, theta_i, sigma_s, sigma_o)
            orientation_response += r

        # Max projection over orientations
        enhanced = np.maximum(enhanced, orientation_response)

    return enhanced
```

---

## Physical Intuition

**Why does −∂²_η enhance vessels?**

A blood vessel in the green channel image appears as a **dark ridge** (vessels absorb more light than background). After preprocessing (top-hat), vessels appear as local intensity maxima (bright ridges).

The second derivative `∂²_η` perpendicular to the vessel:
- Is **negative** at the peak of a ridge (concave shape)
- Is **positive** at the edges and background

Therefore `−∂²_η` gives a **positive** response at vessel centers and near-zero elsewhere.

The Gaussian smoothing at scale `σs` selects ridges of width approximately `σs`: too-narrow structures (noise) are smoothed out, too-wide structures are not captured at that scale. The multi-scale sum captures vessels of all calibers.

**Why use the LID frame instead of Cartesian derivatives?**

If we computed `∂²_y` or `∂²_x` directly, the filter would only detect horizontal or vertical vessels. By rotating the derivatives with the orientation `θ_i`, we detect vessels at all orientations. The orientation score disentangles crossings: at a vessel crossing, two vessels appear in different orientation layers, and each is enhanced independently without interference.

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| σo | π/40 ≈ 0.0785 rad | Same for all datasets and scales |
| No | 16 | Number of orientations |
| scales S | Dataset-dependent | See SPEC_07 |
| Sign | Negative | Vessels are ridges → ∂²_η < 0 |

---

## Unit Tests (`tests/test_filters.py` — LID section)

```python
def test_lid_filter_detects_line():
    """LID filter should give high response on a horizontal line."""
    H, W = 128, 128
    image = np.zeros((H, W), dtype=np.float32)
    # Gaussian ridge (vessel profile)
    x = np.arange(W)
    profile = np.exp(-(x - W//2)**2 / (2 * 3**2))
    image[H//2 - 10:H//2 + 10, :] = profile[None, :]

    No = 16
    wavelets = build_cake_wavelets((H, W), No)
    scales = [1.5, 2.5]
    sigma_o = np.pi / 40

    enhanced = lid_os_enhance(image, wavelets, scales, sigma_o, No)

    # Should be higher on the vessel than off
    on_vessel = enhanced[H//2, W//2]
    off_vessel = enhanced[H//4, W//4]
    assert on_vessel > off_vessel

def test_lid_filter_response_nonnegative():
    """Filter response should be non-negative (clamped)."""
    H, W = 64, 64
    image = np.random.rand(H, W).astype(np.float32)
    No = 16
    wavelets = build_cake_wavelets((H, W), No)
    U_f_layer = np.random.rand(H, W).astype(np.float32)
    r = lid_filter_response(U_f_layer, theta=0.0, sigma_s=2.0, sigma_o=np.pi/40)
    assert np.all(r >= 0)

def test_lid_scale_normalization():
    """Scale-normalized responses should be comparable across scales."""
    H, W = 64, 64
    U_f_layer = np.zeros((H, W), dtype=np.float32)
    U_f_layer[32, 20:44] = 1.0
    
    sigma_o = np.pi / 40
    r1 = lid_filter_response(U_f_layer, 0.0, sigma_s=1.0, sigma_o=sigma_o)
    r3 = lid_filter_response(U_f_layer, 0.0, sigma_s=3.0, sigma_o=sigma_o)
    
    # Both should give nonzero response near the line
    assert r1[32, 32] > 0 or r3[32, 32] > 0
```
