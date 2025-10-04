# Pokémon Color Palette Swapping

An implementation of optimal color palette swapping for images, designed for transferring color schemes between Pokémon or any other images.

## Overview

This project implements an automatic color palette extraction and swapping system that can:
- Extract dominant color palettes from images using multiple methods:
  - **K-means clustering** (default): Fast and robust
  - **Blind Color Separation**: Gradient descent with L0 sparsity constraints (slower but potentially more accurate)
- Apply any extracted palette to any image
- Find the optimal palette color matching using perceptual distance and image space comparison
- Visualize palette extraction and matching results

## Algorithm

The algorithm consists of several key components:

1. **Palette Extraction**: Two methods available:
   - **K-means** (default): Uses k-means clustering to extract K dominant colors with hard assignment for crisp color boundaries
   - **Blind Color Separation**: Gradient descent optimization with L0 approximation for sparsity, inspired by "L0 Gradient Minimization". Includes constraints to select colors from the original image

2. **Dense Color Transformation Space**: Generates variations of images using HSV color shifts (Hue, Saturation, Value) to create a robust color transformation space

3. **Perceptual Features**:
   - **CPU mode** (default): Uses lightweight features (color histograms, spatial statistics, edge features) for fast processing
   - **GPU mode**: Uses pretrained VGG16 network for deep feature-based distance

4. **Hausdorff Distance**: Measures distance between image transformation spaces to find color-invariant geometric similarity

5. **Optimal Matching**: Tests all possible palette permutations and finds the one that minimizes the Hausdorff distance between transformation spaces

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

### Basic usage:

```bash
python pokemon_recolor.py --source pikachu.png --target charmander.png --output recolored.png
```

### With visualization:

```bash
python pokemon_recolor.py \
    --source pikachu.png \
    --target charmander.png \
    --output recolored.png \
    --visualize
```

### Extract palettes only (for debugging):

```bash
python pokemon_recolor.py \
    --source image.png \
    --target image2.png \
    --extract-only \
    --visualize
```

### Show all permutation results:

```bash
python pokemon_recolor.py \
    --source image.png \
    --target image2.png \
    --show-all-permutations \
    --visualize
```

### Use Blind Color Separation for palette extraction:

```bash
python pokemon_recolor.py \
    --source image.png \
    --target image2.png \
    --extraction-method blind_separation \
    --output recolored.png
```

### Arguments

**Required:**
- `--source`: Path to source image
- `--target`: Path to target image to extract palette from

**Optional:**
- `--output`: Path to save recolored image (default: `recolored_pokemon.png`)
- `--num-colors`: Number of colors in palette (default: 5)
  - Smaller = faster but less detail
  - Recommended: 3-8 colors
- `--extraction-method`: Palette extraction method (default: `kmeans`)
  - `kmeans`: Fast k-means clustering (recommended)
  - `blind_separation`: Gradient descent with L0 sparsity (4-5x slower, potentially more accurate)
- `--hue-steps`: Hue transformation steps (default: 8)
- `--sat-steps`: Saturation transformation steps (default: 3)
- `--val-steps`: Value transformation steps (default: 3)
- `--device`: Device for computation (`cuda` or `cpu`, default: auto-detect)
- `--workers`: Number of parallel workers (default: 4)
- `--visualize`: Generate visualization images
- `--extract-only`: Only extract palettes, skip matching
- `--show-all-permutations`: Visualize all permutation results
- `--no-parallel`: Disable parallel processing

## How It Works

### 1. Palette Extraction

Two methods are available:

**K-means (default):**
```
1. Sample pixels from the image
2. Run k-means to find K cluster centers (palette colors)
3. Assign each pixel to nearest cluster (hard assignment)
4. Each pixel is represented as: I(x,y) = c_i where i = nearest cluster
```

**Blind Color Separation:**
```
1. Initialize palette with random colors from image
2. Optimize using gradient descent to minimize:
   - Reconstruction loss (L2)
   - Sparsity loss (L0 approximation with progressive β)
   - Diversity loss (ensure colors are different)
   - Color constraint (palette colors from original image)
3. Weights are normalized with softmax for smooth blending
```

### 2. Color Transformation Space

For each image, we generate a dense set of color variations by shifting HSV parameters:

```
T(I) = {I transformed by all combinations of hue, sat, val shifts}
Total transforms = hue_steps × sat_steps × val_steps
```

### 3. Feature Extraction

**CPU Mode (Lightweight Features):**
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

## Module Structure

- `palette_extraction.py`: K-means based palette extraction
- `color_transforms.py`: Dense color transformation space (HSV shifts)
- `perceptual_loss.py`: VGG and lightweight feature extraction
- `hausdorff_distance.py`: Hausdorff distance for image spaces
- `palette_matching.py`: Optimal palette matching algorithm
- `pokemon_recolor.py`: Main script for color swapping

## Performance Notes

### Computational Complexity

For K colors:
- **Permutations to test**: K! (e.g., 5 colors = 120 permutations)
- **Transforms per image**: hue_steps × sat_steps × val_steps (default: 72)
- **Total feature extractions**: K! × transforms (e.g., 120 × 72 = 8,640)

### Speed Optimization

- **CPU mode**: Uses parallelized lightweight features
- **GPU mode**: Batched VGG feature extraction
- **Palette extraction**: Parallel extraction of source and target
- **Workers**: Use `--workers` to set parallelism (default: 4)

### Recommended Settings

**Fast (< 1 min):**
```bash
--num-colors 3 --hue-steps 4 --sat-steps 2 --val-steps 2
# 6 permutations × 16 transforms = 96 feature extractions
```

**Balanced (1-5 min):**
```bash
--num-colors 5 --hue-steps 5 --sat-steps 3 --val-steps 3
# 120 permutations × 45 transforms = 5,400 feature extractions
```

**High Quality (5-15 min):**
```bash
--num-colors 6 --hue-steps 8 --sat-steps 3 --val-steps 3
# 720 permutations × 72 transforms = 51,840 feature extractions
```

## Visualization Output

When `--visualize` is used, the following files are generated:

**Palette extraction mode (`--extract-only`):**
- `output_palettes.png`: Side-by-side palette comparison
- `output_kmeans_source.png`: Original vs quantized source + palette
- `output_kmeans_target.png`: Original vs quantized target + palette

**Full pipeline mode:**
- `output_palettes.png`: Palette comparison with optimal matching
- `output_comparison.png`: Original vs recolored result
- `output_all_permutations.png`: Grid of all permutation results (with `--show-all-permutations`)

## Examples

### Transfer color scheme:
```bash
python pokemon_recolor.py \
    --source images/eevee.png \
    --target images/squirtle.png \
    --output eevee_water.png \
    --visualize
```

### Use more colors for detail:
```bash
python pokemon_recolor.py \
    --source pokemon1.png \
    --target pokemon2.png \
    --num-colors 8 \
    --output result.png
```

### Fast preview:
```bash
python pokemon_recolor.py \
    --source image1.png \
    --target image2.png \
    --num-colors 3 \
    --hue-steps 4 \
    --sat-steps 2 \
    --val-steps 2 \
    --output preview.png
```

## Technical Details

### K-means Palette Extraction

- Samples 10% of pixels for speed (configurable via `sample_fraction`)
- Uses k-means++ initialization for better convergence
- 10 random initializations to find best clustering
- Hard assignment ensures crisp color boundaries
- Logs color distribution statistics

### Lightweight Features (CPU)

Designed for fast CPU processing:
- **Color histogram**: 16³ = 4,096 bins
- **Spatial features**: Multi-scale (σ=0,2,4) mean and std
- **Edge features**: Gradient magnitude histogram (16 bins)
- **Total feature dimension**: ~4,130 per image

### Parallelization

- Palette extraction: 2 workers (source + target in parallel)
- Feature extraction: CPU count workers (default)
- Transform generation: Configurable via `--workers`

## License

MIT License
