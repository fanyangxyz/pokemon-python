"""
Optimal palette matching algorithm.
Finds the best permutation of palette colors to minimize image space distance.
"""

import numpy as np
from itertools import permutations
from typing import Tuple, List
from color_transforms import FastColorTransformSpace
from perceptual_loss import DeepImageDistance
from hausdorff_distance import ImageSpaceDistance, ApproximateHausdorff


class PaletteMatcher:
    """
    Find optimal palette color matching between source and target images.
    """

    def __init__(
        self,
        hue_steps: int = 8,
        sat_steps: int = 3,
        val_steps: int = 3,
        device: str = None,
        use_approximate: bool = True
    ):
        """
        Initialize palette matcher.

        Args:
            hue_steps: Number of hue transformation steps
            sat_steps: Number of saturation transformation steps
            val_steps: Number of value transformation steps
            device: Device for VGG computation
            use_approximate: Use approximate Hausdorff for speed
        """
        self.transform_space = FastColorTransformSpace(hue_steps, sat_steps, val_steps)
        self.deep_distance = DeepImageDistance(device=device)

        if use_approximate:
            self.hausdorff = ApproximateHausdorff(sample_size=50)
        else:
            self.hausdorff = ImageSpaceDistance(use_modified=True)

    def find_optimal_matching(
        self,
        source_image: np.ndarray,
        source_weights: np.ndarray,
        source_palette: np.ndarray,
        target_palette: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Find optimal palette color matching.

        Args:
            source_image: Original source image (H, W, 3) in [0, 1]
            source_weights: Palette weights (H, W, K) for source
            source_palette: Source palette colors (K, 3)
            target_palette: Target palette colors (K, 3)

        Returns:
            - best_permutation: Optimal permutation indices
            - best_distance: Minimum distance achieved
            - best_recolored_image: Best recolored image
        """
        num_colors = len(source_palette)

        # Generate all possible permutations
        all_perms = list(permutations(range(num_colors)))

        print(f"Testing {len(all_perms)} permutations...")

        # Precompute source image transformation space features
        print("Computing source image transformation space...")
        source_transforms = self.transform_space.apply_all_transforms(source_image)
        source_features = self.deep_distance.batch_get_features(source_transforms)

        best_distance = float('inf')
        best_permutation = None
        best_recolored_image = None

        for i, perm in enumerate(all_perms):
            if i % 10 == 0:
                print(f"Testing permutation {i + 1}/{len(all_perms)}...")

            # Apply permuted palette
            permuted_palette = target_palette[list(perm)]

            # Reconstruct image with new palette
            H, W, K = source_weights.shape
            weights_flat = source_weights.reshape(-1, K)
            recolored_image = np.matmul(weights_flat, permuted_palette).reshape(H, W, 3)
            recolored_image = np.clip(recolored_image, 0, 1)

            # Generate transformation space for recolored image
            recolored_transforms = self.transform_space.apply_all_transforms(recolored_image)
            recolored_features = self.deep_distance.batch_get_features(recolored_transforms)

            # Compute Hausdorff distance
            distance = self.hausdorff.compute_distance(source_features, recolored_features)

            if distance < best_distance:
                best_distance = distance
                best_permutation = perm
                best_recolored_image = recolored_image
                print(f"  New best distance: {best_distance:.6f}")

        print(f"Optimal permutation found: {best_permutation}")
        print(f"Best distance: {best_distance:.6f}")

        return np.array(best_permutation), best_distance, best_recolored_image

    def quick_match(
        self,
        source_palette: np.ndarray,
        target_palette: np.ndarray
    ) -> np.ndarray:
        """
        Quick heuristic matching based on color similarity.
        Useful for initialization or when full search is too expensive.

        Args:
            source_palette: Source palette (K, 3)
            target_palette: Target palette (K, 3)

        Returns:
            Permutation indices
        """
        from scipy.spatial.distance import cdist

        # Compute pairwise color distances
        distances = cdist(source_palette, target_palette, metric='euclidean')

        # Greedy matching
        num_colors = len(source_palette)
        used = set()
        permutation = []

        for i in range(num_colors):
            # Find closest unused target color
            min_dist = float('inf')
            best_j = -1

            for j in range(num_colors):
                if j not in used and distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    best_j = j

            permutation.append(best_j)
            used.add(best_j)

        return np.array(permutation)


class OptimalPaletteSwap:
    """
    Complete pipeline for optimal palette swapping.
    """

    def __init__(
        self,
        num_colors: int = 5,
        hue_steps: int = 8,
        sat_steps: int = 3,
        val_steps: int = 3,
        device: str = None
    ):
        """
        Initialize optimal palette swap.

        Args:
            num_colors: Number of colors in palette
            hue_steps: HSV hue steps for transformation space
            sat_steps: HSV saturation steps
            val_steps: HSV value steps
            device: Device for neural network
        """
        from palette_extraction import PaletteExtractor

        self.palette_extractor = PaletteExtractor(num_colors=num_colors)
        self.palette_matcher = PaletteMatcher(
            hue_steps=hue_steps,
            sat_steps=sat_steps,
            val_steps=val_steps,
            device=device
        )

    def swap_palette(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        extract_source: bool = True,
        extract_target: bool = True,
        source_palette: np.ndarray = None,
        source_weights: np.ndarray = None,
        target_palette: np.ndarray = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Swap palette from target to source image.

        Args:
            source_image: Source image (H, W, 3) in [0, 1]
            target_image: Target image to extract palette from (H, W, 3) in [0, 1]
            extract_source: Whether to extract source palette
            extract_target: Whether to extract target palette
            source_palette: Pre-extracted source palette (optional)
            source_weights: Pre-extracted source weights (optional)
            target_palette: Pre-extracted target palette (optional)

        Returns:
            - result_image: Recolored image
            - info: Dictionary with palette info and metrics
        """
        # Extract source palette if needed
        if extract_source or source_palette is None or source_weights is None:
            print("Extracting source palette...")
            source_palette, source_weights = self.palette_extractor.extract_palette(source_image)

        # Extract target palette if needed
        if extract_target or target_palette is None:
            print("Extracting target palette...")
            target_palette, _ = self.palette_extractor.extract_palette(target_image)

        # Find optimal matching
        print("Finding optimal palette matching...")
        permutation, distance, result_image = self.palette_matcher.find_optimal_matching(
            source_image,
            source_weights,
            source_palette,
            target_palette
        )

        info = {
            'source_palette': source_palette,
            'target_palette': target_palette,
            'permutation': permutation,
            'distance': distance,
            'source_weights': source_weights
        }

        return result_image, info
