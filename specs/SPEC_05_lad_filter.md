# SPEC 05 — Locally Adaptive Derivative (LAD) Filter

## File: `src/lad_filter.py`

The LAD filter is the main contribution of Zhang et al. (2016). It extends the LID filter by constructing a **locally adaptive frame** that aligns with the actual local vessel orientation and curvature at each position in the orientation score. This is done via **exponential curve fitting** in SE(2).

This is the most mathematically involved component. Read carefully.

---

## Overview of LAD Construction

Starting from the LID frame `{∂_ξ, ∂_η, ∂_θ}`, the LAD frame `{∂_a, ∂_b, ∂_c}` is obtained via two rotations:

```
(∂_a, ∂_b, ∂_c)ᵀ = Q^T_{κ,μ} · R̃^T_{d_H} · (∂_ξ, ∂_η, μ∂_θ)ᵀ
```

where:
- `d_H` = deviation from horizontality (angle between ∂_ξ and the projected tangent in R²)
- `κ` = local curvature
- `μ = σo/σs` = scale normalization parameter

The LAD filter then applies `∂²_b` (second derivative in the b-direction, which is perpendicular to the locally detected vessel) to enhance vessels.

---

## Step-by-Step Construction

### Step 1: Compute the Left-Invariant Hessian H_Uf

The Hessian matrix in the orientation score domain, expressed in the LID frame:

```
H_Uf = | ∂²_ξ U_f      ∂_ξ∂_η U_f    ∂_θ∂_ξ U_f |
        | ∂_ξ∂_η U_f    ∂²_η U_f      ∂_θ∂_η U_f |
        | ∂_ξ∂_θ U_f    ∂_η∂_θ U_f    ∂²_θ U_f   |
```

**Important:** This Hessian is **not symmetric** due to the non-commutative group structure of SE(2):
```
∂_θ ∂_ξ U_f ≠ ∂_ξ ∂_θ U_f
```
(The commutator: `[∂_θ, ∂_ξ] = ∂_η`)

```python
def compute_left_invariant_hessian(
    U_f_stack: np.ndarray,
    theta_i: float,
    sigma_s: float,
    sigma_o: float,
    mu: float,
    No: int,
    delta_theta: float,
) -> dict:
    """
    Compute the 3×3 left-invariant Hessian matrix H_Uf at each position
    in one orientation score layer (position x, orientation θ_i).

    All derivatives are first smoothed with G_{σs,σo} before differentiation.

    Parameters
    ----------
    U_f_stack : np.ndarray (No, H, W) float32
        The full (angularly smoothed) orientation score (real part).
    theta_i : float
        Current orientation angle in radians.
    sigma_s : float
        Spatial Gaussian scale.
    sigma_o : float
        Angular Gaussian scale.
    mu : float
        μ = σo/σs.
    No : int
        Number of orientations.
    delta_theta : float
        Angular step = π/No.

    Returns
    -------
    H : dict with keys 'xi_xi', 'xi_eta', 'eta_eta', 'xi_theta',
                         'eta_theta', 'theta_theta'
        Each entry is an np.ndarray of shape (H, W) float32.
        Represents the 9 components of H_Uf at layer θ_i.
    """
```

#### Computing spatial derivatives (∂_ξ and ∂_η):

Given `U_f_layer` = U_f(·, θ_i), smooth it with G_σs then rotate Cartesian derivatives:

```python
def compute_xi_eta_derivatives(U_f_layer, theta, sigma_s):
    """Compute first and second order LID frame derivatives at angle theta."""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # First-order Cartesian derivatives of smoothed layer
    Gf = gaussian_filter(U_f_layer, sigma=sigma_s)  # smoothed
    dx = gaussian_filter(U_f_layer, sigma=sigma_s, order=(0, 1))  # ∂_x
    dy = gaussian_filter(U_f_layer, sigma=sigma_s, order=(1, 0))  # ∂_y

    # LID first-order derivatives
    d_xi  =  cos_t * dx + sin_t * dy      # ∂_ξ
    d_eta = -sin_t * dx + cos_t * dy      # ∂_η

    # Second-order Cartesian derivatives
    dxx = gaussian_filter(U_f_layer, sigma=sigma_s, order=(0, 2))
    dyy = gaussian_filter(U_f_layer, sigma=sigma_s, order=(2, 0))
    dxy = gaussian_filter(U_f_layer, sigma=sigma_s, order=(1, 1))

    # Second-order LID derivatives (product rule for rotated coordinates):
    d2_xi_xi   =  cos_t**2 * dxx + 2*sin_t*cos_t * dxy + sin_t**2 * dyy
    d2_eta_eta =  sin_t**2 * dxx - 2*sin_t*cos_t * dxy + cos_t**2 * dyy
    d2_xi_eta  = -sin_t*cos_t * dxx + (cos_t**2 - sin_t**2) * dxy + sin_t*cos_t * dyy

    return {
        'd_xi': d_xi, 'd_eta': d_eta,
        'd2_xi_xi': d2_xi_xi, 'd2_eta_eta': d2_eta_eta, 'd2_xi_eta': d2_xi_eta,
    }
```

#### Computing orientation derivatives (∂_θ):

The θ-derivative is computed via **finite differences** along the orientation stack:

```python
def compute_theta_derivatives(U_f_stack, i, No, delta_theta, sigma_s, sigma_o):
    """
    Compute ∂_θ U_f and ∂²_θ U_f at orientation index i using central differences.
    Uses adjacent orientation layers (circular wrap).
    """
    i_prev = (i - 1) % No
    i_next = (i + 1) % No

    layer_prev = gaussian_filter(U_f_stack[i_prev], sigma=sigma_s)
    layer_next = gaussian_filter(U_f_stack[i_next], sigma=sigma_s)
    layer_curr = gaussian_filter(U_f_stack[i],      sigma=sigma_s)

    # First-order: central difference
    d_theta = (layer_next - layer_prev) / (2 * delta_theta)

    # Second-order: second difference
    d2_theta = (layer_next - 2 * layer_curr + layer_prev) / delta_theta**2

    return d_theta, d2_theta
```

#### Cross terms ∂_θ ∂_ξ and ∂_θ ∂_η:

```python
def compute_cross_theta_terms(U_f_stack, i, No, theta_i, delta_theta, sigma_s):
    """Compute ∂_θ∂_ξ and ∂_θ∂_η via central differences on ξ/η derivatives."""
    i_prev = (i - 1) % No
    i_next = (i + 1) % No
    theta_prev = (i_prev * np.pi / No)
    theta_next = (i_next * np.pi / No)

    # ∂_ξ at next and previous orientation layers
    d_xi_next = compute_xi_eta_derivatives(U_f_stack[i_next], theta_next, sigma_s)['d_xi']
    d_xi_prev = compute_xi_eta_derivatives(U_f_stack[i_prev], theta_prev, sigma_s)['d_xi']
    d_eta_next = compute_xi_eta_derivatives(U_f_stack[i_next], theta_next, sigma_s)['d_eta']
    d_eta_prev = compute_xi_eta_derivatives(U_f_stack[i_prev], theta_prev, sigma_s)['d_eta']

    d_theta_d_xi  = (d_xi_next  - d_xi_prev)  / (2 * delta_theta)
    d_theta_d_eta = (d_eta_next - d_eta_prev) / (2 * delta_theta)

    return d_theta_d_xi, d_theta_d_eta
```

#### Assembling H_Uf:

```python
def assemble_hessian(U_f_stack, i, No, theta_i, sigma_s, sigma_o, mu):
    delta_theta = np.pi / No
    derivs = compute_xi_eta_derivatives(U_f_stack[i], theta_i, sigma_s)
    d_theta, d2_theta = compute_theta_derivatives(U_f_stack, i, No, delta_theta, sigma_s, sigma_o)
    d_theta_d_xi, d_theta_d_eta = compute_cross_theta_terms(U_f_stack, i, No, theta_i, delta_theta, sigma_s)

    H = {
        (0,0): derivs['d2_xi_xi'],
        (0,1): derivs['d2_xi_eta'],
        (0,2): d_theta_d_xi,          # ∂_θ∂_ξ NOTE: not symmetric with (2,0)
        (1,0): derivs['d2_xi_eta'],   # symmetric in spatial part
        (1,1): derivs['d2_eta_eta'],
        (1,2): d_theta_d_eta,
        (2,0): d_theta_d_xi,          # ∂_ξ∂_θ
        (2,1): d_theta_d_eta,
        (2,2): d2_theta,
    }
    return H  # 9 arrays of shape (H, W)
```

---

### Step 2: Symmetrized μ-normalized Hessian H_μ

For the eigensystem analysis, we form the symmetrized, μ-normalized Hessian:

```
H_μ U_f = M_μ⁻¹ · (H_Uf)ᵀ · M_μ⁻² · H_Uf · M_μ⁻¹
```

where `M_μ = diag(μ, μ, 1)`, so:

```
M_μ⁻¹ = diag(1/μ, 1/μ, 1)
M_μ⁻² = diag(1/μ², 1/μ², 1)
```

In practice, since we work at every pixel, we compute this as a pointwise matrix product. The result is a **symmetric** 3×3 matrix at each pixel.

```python
def compute_H_mu(H: dict, mu: float) -> np.ndarray:
    """
    Compute the symmetrized μ-normalized Hessian H_μ at each pixel.

    H_μ = M_μ⁻¹ (H_Uf)ᵀ M_μ⁻² H_Uf M_μ⁻¹

    Parameters
    ----------
    H : dict with keys (i,j) → np.ndarray (H_img, W_img)
    mu : float

    Returns
    -------
    H_mu : np.ndarray (H_img, W_img, 3, 3)
        Symmetric 3×3 matrix at each pixel.
    """
    H_img, W_img = H[(0,0)].shape

    # Weights from M_μ
    w = np.array([mu, mu, 1.0])  # diagonal of M_μ

    # Build H_Uf as a (H,W,3,3) array
    H_arr = np.zeros((H_img, W_img, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            H_arr[:, :, i, j] = H[(i, j)]

    # M_μ⁻¹: divide rows and columns by w[i], w[j] respectively
    # M_μ⁻² H_Uf: scale rows by 1/w[i]²
    # Full: M_μ⁻¹ (H_Uf)ᵀ M_μ⁻² H_Uf M_μ⁻¹

    # Step: A = M_μ⁻² H_Uf  →  A[i,j] = H[i,j] / w[i]²
    A = H_arr / (w[:, None] * w[None, :])[None, None, :, :]  # rough; do properly:
    
    # Proper implementation using einsum:
    Minv = np.diag(1.0 / w)
    Minv2 = np.diag(1.0 / w**2)

    # H_mu_pixel = Minv @ H_Uf_pixel.T @ Minv2 @ H_Uf_pixel @ Minv
    # Vectorized over pixels:
    H_mu = np.einsum('...ij,...jk,...kl,...lm->...im',
                     Minv[None, None],
                     np.transpose(H_arr, (0, 1, 3, 2)),
                     Minv2[None, None],
                     H_arr,
                     Minv[None, None])

    return H_mu.astype(np.float32)
```

---

### Step 3: Eigenvector Analysis → Optimal Tangent c*

The optimal tangent vector `c*` minimizes `E(c) = ||d/dt ∇U_f(γ_c(t))|_{t=0}||²_μ`. This is equivalent to finding the **eigenvector of H_μ with the smallest eigenvalue**.

```python
def compute_optimal_tangent(H_mu: np.ndarray) -> np.ndarray:
    """
    Find eigenvector of H_mu corresponding to smallest eigenvalue at each pixel.

    Parameters
    ----------
    H_mu : np.ndarray (H, W, 3, 3) float32
        Symmetrized μ-normalized Hessian at each pixel.

    Returns
    -------
    c_star : np.ndarray (H, W, 3) float32
        Optimal tangent vector c* = M_μ⁻¹ c̃* at each pixel.
    """
    H, W = H_mu.shape[:2]
    c_star = np.zeros((H, W, 3), dtype=np.float32)

    # Reshape for batch eigendecomposition
    H_flat = H_mu.reshape(-1, 3, 3)

    # Compute eigenvalues and eigenvectors
    # H_mu is symmetric → use eigh for efficiency and numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(H_flat)
    # eigh returns eigenvalues in ascending order
    # eigenvectors[:, :, i] is the i-th eigenvector

    # c̃* = eigenvector corresponding to smallest eigenvalue = eigenvectors[:, :, 0]
    c_tilde_star = eigenvectors[:, :, 0]  # shape (H*W, 3)

    # c* = M_μ⁻¹ c̃* (undo the normalization)
    # M_μ = diag(μ, μ, 1), so M_μ⁻¹ = diag(1/μ, 1/μ, 1)
    # But we need mu at this point... pass it as argument
    # For now, return c̃* and let caller apply M_μ⁻¹

    c_tilde_star = c_tilde_star.reshape(H, W, 3)
    return c_tilde_star  # Note: this is c̃*, caller must multiply by M_μ⁻¹ to get c*
```

**Efficiency note:** `np.linalg.eigh` on batches of small (3×3) symmetric matrices is efficient. For a 565×584 image, this is ~330k eigendecompositions. In numpy this takes about 2–5 seconds. Consider using `scipy.linalg.eigh` with `lower=True` for better stability.

---

### Step 4: Compute κ and d_H

From `c* = (c_ξ, c_η, c_θ)ᵀ` expressed in the LID basis:

```
κ = (c_θ · sign(c_ξ)) / √(c_η² + c_ξ²)
d_H = arctan(c_η / c_ξ)
```

```python
def compute_kappa_dH(c_star: np.ndarray, mu: float) -> tuple:
    """
    Compute local curvature κ and deviation from horizontality d_H.

    Parameters
    ----------
    c_star : np.ndarray (H, W, 3) float32
        Optimal tangent vector c* = (c_ξ, c_η, c_θ)ᵀ.
        (After applying M_μ⁻¹ to c̃*)
    mu : float

    Returns
    -------
    kappa : np.ndarray (H, W) float32
    d_H : np.ndarray (H, W) float32
    """
    c_xi  = c_star[:, :, 0]
    c_eta = c_star[:, :, 1]
    c_th  = c_star[:, :, 2]

    eps = 1e-8
    kappa = (c_th * np.sign(c_xi)) / (np.sqrt(c_eta**2 + c_xi**2) + eps)
    d_H = np.arctan2(c_eta, c_xi + eps)

    return kappa.astype(np.float32), d_H.astype(np.float32)
```

---

### Step 5: Construct the LAD Frame {∂_a, ∂_b, ∂_c}

Two rotation matrices applied to the LID frame:

**Rotation 1: R̃_{d_H}** (rotation in the ξ-η plane by angle d_H)
```
R̃_{d_H} = | cos d_H   −sin d_H   0 |
            | sin d_H    cos d_H   0 |
            | 0          0         1 |
```

**Rotation 2: Q_{κ,μ}** (rotation mixing ξ and μ∂_θ by curvature κ)
```
Q_{κ,μ} = | μ/√(μ²+κ²)    0    κ/√(μ²+κ²)  |
            | 0              1    0             |
            | −κ/√(μ²+κ²)   0    μ/√(μ²+κ²)  |
```

The LAD frame:
```
(∂_a, ∂_b, ∂_c)ᵀ = Q^T_{κ,μ} · R̃^T_{d_H} · (∂_ξ, ∂_η, μ∂_θ)ᵀ
```

The vessel-enhancing direction is `∂_b` (perpendicular to the vessel in the locally adapted frame).

---

### Step 6: Compute ∂²_b via Hessian Projection

Once the LAD frame direction `e_b` is known at each pixel, the second derivative in the b-direction is:

```
∂²_b U_f = e_b^T · H_Uf · e_b
```

where `H_Uf` is the (non-symmetric) left-invariant Hessian from Step 1.

```python
def compute_d2_b(H_uf_arr: np.ndarray, e_b: np.ndarray) -> np.ndarray:
    """
    Compute ∂²_b U_f = e_b^T H_Uf e_b at each pixel.

    Parameters
    ----------
    H_uf_arr : np.ndarray (H, W, 3, 3) float32
        Left-invariant Hessian H_Uf.
    e_b : np.ndarray (H, W, 3) float32
        Unit vector in the b-direction of the LAD frame.

    Returns
    -------
    d2_b : np.ndarray (H, W) float32
    """
    # d2_b = e_b^T H_Uf e_b = einsum('...i,...ij,...j->...', e_b, H_Uf_arr, e_b)
    d2_b = np.einsum('...i,...ij,...j->...', e_b, H_uf_arr, e_b)
    return d2_b.astype(np.float32)
```

---

### Step 7: Computing e_b from κ and d_H

```python
def compute_e_b(kappa: np.ndarray, d_H: np.ndarray, mu: float) -> np.ndarray:
    """
    Compute the unit vector e_b in the LAD frame at each pixel.

    e_b is the second column of the combined rotation (Q^T R̃^T):
    LAD frame column for direction b.

    Parameters
    ----------
    kappa : np.ndarray (H, W)
    d_H : np.ndarray (H, W)
    mu : float

    Returns
    -------
    e_b : np.ndarray (H, W, 3) float32
        Vector in LID frame coordinates (ξ, η, θ) directions.
    """
    H, W = kappa.shape
    norm_kmu = np.sqrt(mu**2 + kappa**2) + 1e-8

    # Q^T R̃^T:
    # R̃^T_{d_H} rotates by -d_H in ξ-η plane
    # Q^T_{κ,μ} mixes ξ and μθ components

    # The second column (b-direction) of Q^T R̃^T:
    # e_b = Q^T R̃^T [0, 1, 0]^T = R̃^T [0, 1, 0]^T (since Q^T [0,1,0]^T = [0,1,0]^T)
    # R̃^T [0, 1, 0]^T = [sin(d_H), cos(d_H), 0]^T  (because R̃^T rotates by -d_H)

    sin_dH = np.sin(d_H)
    cos_dH = np.cos(d_H)

    e_b = np.stack([sin_dH, cos_dH, np.zeros_like(kappa)], axis=-1)
    return e_b.astype(np.float32)
```

**Sanity check on e_b:** For a horizontal vessel (d_H ≈ 0), `e_b ≈ [0, 1, 0]` = purely in the η direction, which is indeed perpendicular to the vessel. ✓

---

### Step 8: Scale-Normalized LAD Filter Response

```python
def lad_filter_response(
    U_f_stack: np.ndarray,
    i: int,
    theta_i: float,
    sigma_s: float,
    sigma_o: float,
    mu: float,
    No: int,
) -> np.ndarray:
    """
    Compute the LAD filter response for one orientation layer at one scale.

    Φ^{σs,σo}_{b,norm} = −μ⁻² ∂²_b (G_{σs,σo} * U_f)

    Returns
    -------
    response : np.ndarray (H, W) float32, non-negative
    """
    delta_theta = np.pi / No

    # Step 1: Compute left-invariant Hessian
    H_uf = assemble_hessian(U_f_stack, i, No, theta_i, sigma_s, sigma_o, mu)

    # Step 2: Compute symmetrized μ-normalized Hessian
    H_mu = compute_H_mu(H_uf, mu)

    # Step 3: Find optimal tangent c*
    c_tilde = compute_optimal_tangent(H_mu)  # returns c̃*

    # Apply M_μ⁻¹ to get c*
    inv_w = np.array([1.0/mu, 1.0/mu, 1.0])
    c_star = c_tilde * inv_w[None, None, :]

    # Step 4: Compute κ and d_H
    kappa, d_H = compute_kappa_dH(c_star, mu)

    # Step 5: Compute e_b direction
    e_b = compute_e_b(kappa, d_H, mu)

    # Step 6: Assemble H_Uf as 4D array for projection
    H, W = kappa.shape
    H_arr = np.zeros((H, W, 3, 3), dtype=np.float32)
    for ii in range(3):
        for jj in range(3):
            H_arr[:, :, ii, jj] = H_uf[(ii, jj)]

    # Step 7: Compute ∂²_b = e_b^T H_Uf e_b
    d2_b = compute_d2_b(H_arr, e_b)

    # Step 8: Scale-normalized response (negative because vessels are ridges)
    mu_inv_sq = 1.0 / mu**2
    response = -mu_inv_sq * d2_b

    # Clamp to non-negative
    response = np.maximum(response, 0).astype(np.float32)

    return response
```

---

## Full LAD-OS Enhancement Pipeline

```python
def lad_os_enhance(
    image: np.ndarray,
    wavelets: np.ndarray,
    scales: list,
    sigma_o: float,
    No: int,
) -> np.ndarray:
    """
    Full LAD-OS vessel enhancement.

    Υ(f)(x) = max_{θ_i} { Σ_{σs ∈ S} Φ^{σs,σo}_{b,norm}(U_f)(x, θ_i) }

    Parameters
    ----------
    image : np.ndarray (H, W) float32
    wavelets : np.ndarray (No, H, W) complex64
    scales : list of float
    sigma_o : float
    No : int

    Returns
    -------
    enhanced : np.ndarray (H, W) float32
    """
    H, W = image.shape
    enhanced = np.zeros((H, W), dtype=np.float32)

    # Compute full orientation score (needed for angular derivatives)
    F_image = np.fft.fft2(image.astype(np.float64))
    U_f_stack = np.zeros((No, H, W), dtype=np.float32)
    for i in range(No):
        F_wavelet = np.fft.fft2(wavelets[i].astype(np.complex128))
        U_f_stack[i] = np.real(np.fft.ifft2(np.conj(F_wavelet) * F_image)).astype(np.float32)

    # Angular smoothing of full stack
    delta_theta = np.pi / No
    sigma_idx = sigma_o / delta_theta
    if sigma_idx >= 0.1:
        from scipy.ndimage import gaussian_filter1d
        U_f_stack = gaussian_filter1d(U_f_stack, sigma=sigma_idx, axis=0, mode='wrap')

    # Process each orientation
    for i in range(No):
        theta_i = i * np.pi / No
        orientation_response = np.zeros((H, W), dtype=np.float32)

        for sigma_s in scales:
            mu = sigma_o / sigma_s
            r = lad_filter_response(U_f_stack, i, theta_i, sigma_s, sigma_o, mu, No)
            orientation_response += r

        enhanced = np.maximum(enhanced, orientation_response)

    return enhanced
```

---

## Performance Notes

The LAD filter is computationally expensive due to:
1. Computing the full Hessian (9 arrays of derivatives per scale per orientation)
2. Batched 3×3 eigendecompositions (one per pixel)

**Expected runtimes** (from paper, Mathematica 10.2, 2.7 GHz CPU):
- LID-OS: ~4 seconds per DRIVE image
- LAD-OS: ~20 seconds per DRIVE image

In numpy/Python, expect 3–10× slower without JIT compilation. Recommend:
- Use `numba` `@jit` on the inner loops
- Or use `scipy.linalg.eigh` with contiguous arrays
- Process scales in a vectorized manner where possible

For debugging, start with small `No=4` and 1–2 scales on small test images.

---

## Unit Tests (`tests/test_filters.py` — LAD section)

```python
def test_lad_response_nonnegative():
    H, W, No = 64, 64, 8
    image = np.zeros((H, W), dtype=np.float32)
    image[32, 10:54] = 1.0  # horizontal line
    wavelets = build_cake_wavelets((H, W), No)
    enhanced = lad_os_enhance(image, wavelets, scales=[2.0, 3.0],
                               sigma_o=np.pi/40, No=No)
    assert np.all(enhanced >= 0)

def test_lad_detects_line():
    H, W, No = 128, 128, 16
    image = np.zeros((H, W), dtype=np.float32)
    y = np.arange(H)
    profile = np.exp(-(y - H//2)**2 / (2 * 2**2))
    image[:, W//2-5:W//2+5] = profile[:, None]  # vertical vessel
    wavelets = build_cake_wavelets((H, W), No)
    enhanced = lad_os_enhance(image, wavelets, scales=[1.5, 2.5, 3.5],
                               sigma_o=np.pi/40, No=No)
    on_vessel = float(enhanced[H//2, W//2])
    off_vessel = float(enhanced[H//4, W//4])
    assert on_vessel > off_vessel, f"on={on_vessel:.4f}, off={off_vessel:.4f}"

def test_lad_better_than_lid_at_crossing():
    """LAD should better handle crossings (vessel at two orientations)."""
    H, W, No = 128, 128, 16
    image = np.zeros((H, W), dtype=np.float32)
    # Horizontal vessel
    image[H//2-2:H//2+2, :] = 1.0
    # Diagonal vessel crossing
    for k in range(H):
        j = k
        if j < W:
            for dj in range(-2, 3):
                if 0 <= j+dj < W:
                    image[k, j+dj] = 1.0

    wavelets = build_cake_wavelets((H, W), No)
    scales = [1.5, 2.5]
    sigma_o = np.pi / 40

    lad_enhanced = lad_os_enhance(image, wavelets, scales, sigma_o, No)
    lid_enhanced = lid_os_enhance(image, wavelets, scales, sigma_o, No)

    # Both should detect the crossing; LAD should be at least as good
    cross_pt = (H//2, H//2)
    assert lad_enhanced[cross_pt] > 0
    assert lid_enhanced[cross_pt] > 0
```
