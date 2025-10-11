# Pokémon Color Palette Swapping

An implementation of optimal color palette swapping for images, designed for transferring color schemes between Pokémon or any other images.

![Example: Charizard recolored using different other Pokemons](batch-results-five-colors/charizard_all_transformations.png)
[Blog post](https://fanyangxyz.github.io/2025/10/04/recoloring/).


## Overview

This project implements an automatic color palette extraction and swapping system that can:
- Extract dominant color palettes from images using multiple methods:
  - **K-means clustering** (default): Fast and robust, hard assignment for crisp edges
  - **Blind Color Separation**: Gradient descent with L0 sparsity constraints (slower, soft assignment for smooth blending)
- Apply any extracted palette to any image
- Find the optimal palette color matching using perceptual distance and image space comparison
- Visualize palette extraction and matching results

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- torch
- torchvision
- pillow
- matplotlib
- scikit-learn
- scipy

## Usage

**Required:**
- `--source`: Path to source image
- `--target`: Path to target image to extract palette from

**Optional:**
- `--output`: Path to save recolored image (default: `recolored_pokemon.png`)
- `--num-colors`: Number of colors in palette (default: 5)
  - Smaller = faster but less detail
  - Recommended: 3-8 colors (Note: 8+ colors will be slow due to K! permutations)
- `--extraction-method`: Palette extraction method (default: `kmeans`)
  - `kmeans`: Fast k-means clustering with hard assignment (crisp edges)
  - `blind_separation`: Gradient descent with soft assignment (smooth blending, experimental)
- `--hue-steps`: Hue transformation steps (default: 8)
- `--sat-steps`: Saturation transformation steps (default: 3)
- `--val-steps`: Value transformation steps (default: 3)
- `--device`: Device for computation (`cuda` or `cpu`, default: auto-detect)
- `--workers`: Number of parallel workers (default: 4)
- `--visualize`: Generate visualization images
- `--extract-only`: Only extract palettes, skip matching
- `--show-all-permutations`: Visualize top 10 permutation results (uses heap to save memory)
- `--no-parallel`: Disable parallel processing

## Algorithm

### 1. Palette Extraction

Two methods are available:

**K-means (default):**
```
1. Sample pixels from the image
2. Run k-means to find K cluster centers (palette colors)
3. Assign each pixel to nearest cluster (hard assignment)
4. Result: Crisp color boundaries, posterized look
```

**Blind Color Separation (experimental):**
```
1. Initialize palette using k-means
2. Optimize using gradient descent to minimize:
   - Reconstruction loss (L2)
   - Sparsity loss (L0 approximation with progressive β)
   - Diversity loss (ensure colors are different)
   - Color constraint (palette colors from original image)
3. Result: Soft weights, smooth blending (may not converge well)
```

### 2. Color Transformation Space

For each image, we generate a dense set of color variations by shifting HSV parameters:

```
T(I) = {I transformed by all combinations of hue, sat, val shifts}
Total transforms = hue_steps × sat_steps × val_steps
```

### 3. Feature Extraction

**CPU Mode (Lightweight Features - Default):**
- 3D color histograms (16×16×16 bins)
- Multi-scale spatial color statistics
- Edge gradient histograms
- ~100x faster than VGG on CPU

**GPU Mode (VGG Features):**
- Pretrained VGG16 features from multiple layers
- Higher quality perceptual distance
- Requires CUDA-capable GPU

### 4. Image Space Distance

The distance between two images is the Hausdorff distance between their transformation spaces in feature space:

```
D(I₁, I₂) = H(φ(T(I₁)), φ(T(I₂)))
```

where `φ` is the feature extractor and `H` is Hausdorff distance.

### 5. Optimal Permutation

Find the palette permutation π that minimizes:

```
π* = argmin_π D(I_source, apply_palette(I_source, palette_target[π]))
```

## References

- [【毕业论文】求解最优的任意宝可梦颜色交换算法](https://zhuanlan.zhihu.com/p/695729586)

## License

MIT License
