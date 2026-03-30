# SPEC 01 — Preprocessing

## File: `src/preprocessing.py`

This module implements the three preprocessing steps applied before orientation score construction:
1. **Channel extraction** (green channel for RGB datasets)
2. **Luminosity & contrast normalization** (Foracchia et al., 2005)
3. **Geodesic opening** (removes large bright structures like optic disk)
4. **Morphological top-hat transform** (removes residual central reflex, enhances vessels)

The output is a single-channel float image in [0, 1] with uniform illumination and enhanced vessel-to-background contrast.

---

## 1. Channel Extraction

```python
def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    For RGB fundus images: extract green channel (index 1).
    For grayscale SLO images: return as-is.
    
    Returns float32 array in [0, 1].
    """
```

**Why green channel:** The green channel of RGB fundus images provides the best contrast between vessels and background. This is consistent across DRIVE, STARE, CHASE_DB1, and HRF datasets. The IOSTAR and RC-SLO datasets are SLO images (already grayscale).

**Implementation:**
- If input has 3 channels: `img = image[:, :, 1].astype(np.float32) / 255.0`
- If input is grayscale: normalize to [0, 1] as float32

---

## 2. Luminosity and Contrast Normalization

### Reference
Foracchia, M., Grisan, E., & Ruggeri, A. (2005). *Luminosity and contrast normalization in retinal images.* Medical Image Analysis, 9(3), 179–190.

### Mathematical Model

The observed image I is modeled as:
```
I(x,y) = C(x,y) · I°(x,y) + L(x,y)
```
where:
- `I°(x,y)` = original uncorrupted image
- `C(x,y)` = spatially varying contrast drift (positive scalar field, low-frequency)
- `L(x,y)` = spatially varying luminosity drift (scalar field, low-frequency)

The normalized image estimate is:
```
Î°(x,y) = (I(x,y) − L̂(x,y)) / Ĉ(x,y)
```

### Algorithm

```python
def normalize_luminosity_contrast(
    image: np.ndarray,
    window_size_px: int,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Estimate and remove spatially varying luminosity and contrast drifts
    from background pixels.

    Parameters
    ----------
    image : np.ndarray
        Float32 single-channel image in [0, 1].
    window_size_px : int
        Side length of square patches used for local statistics.
        Paper uses Wl = 500 μm → convert to pixels as round(500 / ρ).
        See SPEC_07 for per-dataset values.
    threshold : float
        Mahalanobis distance threshold for background pixel classification.
        Paper uses t = 1.0 (retains ~68% of pixels in each patch as background).

    Returns
    -------
    np.ndarray : Normalized float32 image, clipped to [0, 1].
    """
```

**Step-by-step implementation:**

#### Step 2a: Partition image into square tiles
```
tile_size = window_size_px
Divide image into a grid of non-overlapping tiles Si of side `tile_size`.
Handle border tiles by padding or cropping.
```

#### Step 2b: Compute local statistics per tile
For each tile Si:
```python
mu_i = np.mean(Si_pixels)
sigma_i = np.std(Si_pixels)
```
This gives sub-sampled images `mu_grid` and `sigma_grid` (shape: [n_tiles_y, n_tiles_x]).

#### Step 2c: Bicubic interpolation to full image size
```python
from scipy.ndimage import zoom
# Compute zoom factors
zy = image.shape[0] / mu_grid.shape[0]
zx = image.shape[1] / mu_grid.shape[1]
mu_full = zoom(mu_grid, (zy, zx), order=3)      # bicubic
sigma_full = zoom(sigma_grid, (zy, zx), order=3) # bicubic
```

#### Step 2d: Identify background pixels per tile
A pixel (x,y) belongs to background set B if its Mahalanobis distance from the local mean is below threshold:
```
d_M(x,y) = |I(x,y) − μ_N(x,y)| / σ_N(x,y) < t
```
where μ_N and σ_N are the interpolated full-resolution versions.

Create binary mask `background_mask = (d_M < threshold)`.

#### Step 2e: Re-estimate L̂ and Ĉ from background pixels only
For each tile Si, restrict to background pixels in that tile:
```python
bg_pixels_in_tile = image[tile_mask & background_mask]
L_hat_i = np.mean(bg_pixels_in_tile)   # luminosity estimate
C_hat_i = np.std(bg_pixels_in_tile)    # contrast estimate
```
Then bicubic-interpolate `L_hat_i` and `C_hat_i` grids to full resolution.

**Special cases:**
- If a tile has fewer than 5 background pixels, use nearest valid tile's values.
- Exclude tiles where `L_hat_i > 0.75` (these are likely optic disk / large bright lesions).

#### Step 2f: Apply normalization
```python
normalized = (image - L_hat_full) / np.maximum(C_hat_full, 1e-6)
# Rescale to [0, 1]
normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
normalized = normalized.astype(np.float32)
```

**Important:** The normalization produces a float image. Downstream steps work on float.

---

## 3. Geodesic Opening

Geodesic opening suppresses large bright structures (optic disk, bright artifacts) while preserving thin vessels.

```python
def geodesic_opening(
    image: np.ndarray,
    kernel_size_px: int,
) -> np.ndarray:
    """
    Apply morphological geodesic opening.

    This is equivalent to: reconstruction by dilation of the eroded image
    under the original image as mask (marker/mask geodesic reconstruction).

    Parameters
    ----------
    image : np.ndarray
        Float32 image in [0, 1].
    kernel_size_px : int
        Radius of the disk-shaped structuring element in pixels.
        Paper uses Wt = 150 μm → round(150 / ρ / 2) pixels radius.
        See SPEC_07.

    Returns
    -------
    np.ndarray : Image after geodesic opening, float32 in [0, 1].
    """
```

**Implementation using scikit-image:**
```python
from skimage.morphology import disk, erosion, reconstruction

def geodesic_opening(image, kernel_size_px):
    selem = disk(kernel_size_px)
    # Marker: eroded image
    marker = erosion(image, selem)
    # Reconstruct marker under mask=image (geodesic dilation)
    opened = reconstruction(marker, image, method='dilation')
    return opened.astype(np.float32)
```

**Why geodesic opening instead of standard opening:**  
Standard opening removes bright structures but may also alter vessel intensities near those structures. Geodesic opening (reconstruction-based) only removes features that cannot be reached during reconstruction, thus preserving connected vessel structures near the optic disk more faithfully.

---

## 4. Top-Hat Transform

The top-hat transform extracts small bright objects (vessels appear dark → we use **black top-hat** to enhance dark structures, or equivalently apply **white top-hat** to the inverted image).

```python
def top_hat_transform(
    image: np.ndarray,
    kernel_size_px: int,
) -> np.ndarray:
    """
    Apply morphological top-hat transform to enhance vessel contrast
    and reduce optic disk brightness and central reflex artifacts.

    Uses white top-hat on the image followed by subtraction:
        result = image - opening(image)
    Or equivalently:
        result = white_tophat(image)

    Parameters
    ----------
    image : np.ndarray
        Float32 image after geodesic opening.
    kernel_size_px : int
        Radius of disk structuring element. Same Wt/2ρ as geodesic opening.

    Returns
    -------
    np.ndarray : Top-hat enhanced image, float32 clipped to [0, 1].
    """
```

**Implementation:**
```python
from skimage.morphology import disk, white_tophat

def top_hat_transform(image, kernel_size_px):
    selem = disk(kernel_size_px)
    result = white_tophat(image, selem)
    result = np.clip(result, 0, 1).astype(np.float32)
    return result
```

**Combined effect of geodesic opening + top-hat:**  
The geodesic opening removes large bright blobs (optic disk). The subsequent top-hat on the opening-corrected image then amplifies the residual vessel signal by removing the smooth background, leaving primarily vessels and pathological bright spots. The combined result has reduced DC offset and improved local vessel-to-background contrast.

---

## 5. Full Preprocessing Pipeline

```python
def preprocess(
    image: np.ndarray,
    pixel_size_um: float,
    Wl_um: float = 500.0,
    Wt_um: float = 150.0,
    is_rgb: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    image : np.ndarray
        Raw image (uint8 RGB or uint8/float grayscale).
    pixel_size_um : float
        Physical pixel size in micrometers (μm/px). Dataset-specific.
    Wl_um : float
        Luminosity normalization window size in μm. Default 500 μm.
    Wt_um : float
        Top-hat / geodesic opening kernel size in μm. Default 150 μm.
    is_rgb : bool
        True for color fundus images; False for SLO grayscale.

    Returns
    -------
    np.ndarray : Preprocessed float32 image in [0, 1], single channel.
    """
    # 1. Extract channel
    img = extract_green_channel(image) if is_rgb else to_float(image)

    # 2. Compute pixel sizes for kernels
    Wl_px = round(Wl_um / pixel_size_um)
    Wt_px = round(Wt_um / pixel_size_um)
    kernel_radius_px = max(1, Wt_px // 2)

    # 3. Luminosity normalization
    img = normalize_luminosity_contrast(img, window_size_px=Wl_px)

    # 4. Geodesic opening
    img = geodesic_opening(img, kernel_size_px=kernel_radius_px)

    # 5. Top-hat transform
    img = top_hat_transform(img, kernel_size_px=kernel_radius_px)

    return img
```

---

## Notes for Implementation

- All intermediate images should be `np.float32`.
- Avoid integer arithmetic at any intermediate stage.
- The `zoom` function from `scipy.ndimage` is suitable for bicubic interpolation; use `order=3`.
- For the DRIVE dataset: `pixel_size_um ≈ 27`, so `Wl_px ≈ 19`, `Wt_px ≈ 6`.
- For the HRF dataset: `pixel_size_um ≈ 4`, so `Wl_px ≈ 125`, `Wt_px ≈ 37`.
- Detailed per-dataset values are in `SPEC_07_datasets_and_params.md`.

---

## Unit Tests (`tests/test_preprocessing.py`)

```python
def test_green_channel_extraction():
    # Create synthetic RGB image
    rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = extract_green_channel(rgb)
    assert result.shape == (100, 100)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() <= 1.0

def test_luminosity_normalization_range():
    img = np.random.rand(200, 200).astype(np.float32)
    result = normalize_luminosity_contrast(img, window_size_px=50)
    assert result.shape == img.shape
    assert result.dtype == np.float32
    # Output should be in [0, 1] after normalization
    assert result.min() >= 0.0 and result.max() <= 1.0

def test_top_hat_non_negative():
    img = np.random.rand(100, 100).astype(np.float32)
    result = top_hat_transform(img, kernel_size_px=5)
    assert np.all(result >= 0)

def test_preprocess_output_shape():
    img = np.random.randint(0, 255, (565, 584, 3), dtype=np.uint8)
    result = preprocess(img, pixel_size_um=27.0, is_rgb=True)
    assert result.shape == (565, 584)
    assert result.dtype == np.float32
```
