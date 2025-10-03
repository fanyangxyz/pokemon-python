# Pokémon Color Palette Swapping

An implementation of the optimal Pokémon color palette swapping algorithm, based on research for automatically transferring color schemes between any Pokémon images.

## Overview

This project implements an automatic color palette extraction and swapping system that can:
- Extract color palettes from Pokémon images using Blind Color Separation
- Apply any extracted palette to any Pokémon image
- Find the optimal palette color matching using perceptual loss and image space distance

## Algorithm

The algorithm consists of several key components:

1. **Palette Extraction**: Uses a modified Blind Color Separation approach with gradient descent optimization to extract K dominant colors from an image

2. **Dense Color Transformation Space**: Generates variations of images using HSV color shifts (Hue, Saturation, Value) to create a robust color transformation space

3. **Perceptual Distance**: Uses pretrained VGG16 network to compute deep feature-based distance between images, capturing perceptual similarity better than RGB distance

4. **Hausdorff Distance**: Measures distance between image transformation spaces to find color-invariant geometric similarity

5. **Optimal Matching**: Tests all possible palette permutations and finds the one that minimizes the Hausdorff distance between transformation spaces

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python pokemon_recolor.py --source pikachu.png --target charmander.png --output recolored.png
```

With visualization:

```bash
python pokemon_recolor.py --source pikachu.png --target charmander.png --output recolored.png --visualize
```

### Arguments

- `--source`: Path to source Pokémon image (required)
- `--target`: Path to target Pokémon image to extract palette from (required)
- `--output`: Path to save recolored image (default: recolored_pokemon.png)
- `--num-colors`: Number of colors in palette (default: 5)
- `--hue-steps`: Hue transformation steps (default: 8)
- `--sat-steps`: Saturation transformation steps (default: 3)
- `--val-steps`: Value transformation steps (default: 3)
- `--device`: Device for computation (cuda or cpu, default: auto)
- `--visualize`: Generate visualization images

## How It Works

### 1. Palette Extraction

Images are represented as a weighted combination of palette colors:

```
I(x,y) = Σ w_i(x,y) * c_i
```

where `c_i` are palette colors and `w_i` are spatially-varying weights.

### 2. Color Transformation Space

For each image, we generate a dense set of color variations by shifting HSV parameters:

```
T(I) = {I transformed by all combinations of hue, sat, val shifts}
```

### 3. Image Space Distance

The distance between two images is defined as the Hausdorff distance between their transformation spaces in VGG feature space:

```
D(I₁, I₂) = H(φ(T(I₁)), φ(T(I₂)))
```

where `φ` is the VGG feature extractor and `H` is Hausdorff distance.

### 4. Optimal Permutation

Find the palette permutation π that minimizes:

```
π* = argmin_π D(I_source, apply_palette(I_source, palette_target[π]))
```

## Module Structure

- `palette_extraction.py`: Palette extraction using Blind Color Separation
- `color_transforms.py`: Dense color transformation space (HSV shifts)
- `perceptual_loss.py`: VGG-based perceptual distance computation
- `hausdorff_distance.py`: Hausdorff distance for image spaces
- `palette_matching.py`: Optimal palette matching algorithm
- `pokemon_recolor.py`: Main script for color swapping

## Performance Notes

- For 5 colors, there are 120 permutations to test
- Each permutation requires computing transformation space (default: 8×3×3 = 72 transforms)
- VGG feature extraction is GPU-accelerated when available
- Approximate Hausdorff distance is used by default for speed

## Examples

Transfer Charmander's color scheme to Pikachu:
```bash
python pokemon_recolor.py --source pikachu.png --target charmander.png --output pikachu_fire.png --visualize
```

Use more colors for finer detail:
```bash
python pokemon_recolor.py --source pokemon1.png --target pokemon2.png --num-colors 7 --output result.png
```

## Citation

This implementation is based on the research paper on optimal Pokémon color swapping algorithms. The core idea uses:
- Blind Color Separation for palette extraction
- L0 gradient minimization for sparsity
- Perceptual loss with pretrained VGG
- Hausdorff distance for image space comparison

## License

MIT License
