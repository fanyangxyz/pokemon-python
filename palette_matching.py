"""
Optimal palette matching algorithm.
Finds the best permutation of palette colors to minimize image space distance.
"""

import numpy as np
from itertools import permutations
from typing import Tuple, List
from color_transforms import FastColorTransformSpace
from perceptual_loss import DeepImageDistance, LightweightImageFeatures
from hausdorff_distance import ImageSpaceDistance, ApproximateHausdorff
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s:%(lineno)d - %(message)s')


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
        use_approximate: bool = True,
        use_lightweight: bool = None
    ):
        """
        Initialize palette matcher.

        Args:
            hue_steps: Number of hue transformation steps
            sat_steps: Number of saturation transformation steps
            val_steps: Number of value transformation steps
            device: Device for VGG computation
            use_approximate: Use approximate Hausdorff for speed
            use_lightweight: Use lightweight CPU-friendly features instead of VGG (auto-detects if None)
        """
        self.transform_space = FastColorTransformSpace(hue_steps, sat_steps, val_steps)

        # Auto-detect: use lightweight features on CPU, VGG on GPU
        if use_lightweight is None:
            if device is None:
                use_lightweight = not torch.cuda.is_available()
            else:
                use_lightweight = (device == 'cpu')

        self.use_lightweight = use_lightweight

        if self.use_lightweight:
            logging.info("Using lightweight CPU-friendly features (color histograms + statistics)")
            self.feature_extractor = LightweightImageFeatures()
        else:
            logging.info("Using VGG deep features")
            self.feature_extractor = DeepImageDistance(device=device)

        if use_approximate:
            self.hausdorff = ApproximateHausdorff(sample_size=50)
        else:
            self.hausdorff = ImageSpaceDistance(use_modified=True)

    def _generate_recolored_images(
        self,
        all_perms: List[tuple],
        source_weights: np.ndarray,
        target_palette: np.ndarray
    ) -> List[np.ndarray]:
        """Generate all recolored images for all permutations."""
        H, W, K = source_weights.shape
        weights_flat = source_weights.reshape(-1, K)

        recolored_images = []
        for perm in all_perms:
            permuted_palette = target_palette[list(perm)]
            recolored_image = np.matmul(weights_flat, permuted_palette).reshape(H, W, 3)
            recolored_image = np.clip(recolored_image, 0, 1)
            recolored_images.append(recolored_image)

        return recolored_images

    def _test_single_permutation(
        self,
        perm: tuple,
        source_weights: np.ndarray,
        target_palette: np.ndarray,
        source_features: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Test a single permutation and return distance and recolored image."""
        # Apply permuted palette
        permuted_palette = target_palette[list(perm)]

        # Reconstruct image with new palette
        H, W, K = source_weights.shape
        weights_flat = source_weights.reshape(-1, K)
        recolored_image = np.matmul(weights_flat, permuted_palette).reshape(H, W, 3)
        recolored_image = np.clip(recolored_image, 0, 1)

        # Generate transformation space for recolored image
        recolored_transforms = self.transform_space.apply_all_transforms(recolored_image)
        recolored_features = self.feature_extractor.batch_get_features(recolored_transforms)

        # Compute Hausdorff distance
        distance = self.hausdorff.compute_distance(source_features, recolored_features)

        return distance, recolored_image

    def find_optimal_matching(
        self,
        source_image: np.ndarray,
        source_weights: np.ndarray,
        source_palette: np.ndarray,
        target_palette: np.ndarray,
        use_parallel: bool = True,
        num_workers: int = 4,
        return_all_results: bool = False
    ) -> Tuple[np.ndarray, float, np.ndarray, dict]:
        """
        Find optimal palette color matching.

        Args:
            source_image: Original source image (H, W, 3) in [0, 1]
            source_weights: Palette weights (H, W, K) for source
            source_palette: Source palette colors (K, 3)
            target_palette: Target palette colors (K, 3)
            use_parallel: Use parallel processing for permutations
            num_workers: Number of parallel workers
            return_all_results: Return all permutation results for visualization

        Returns:
            - best_permutation: Optimal permutation indices
            - best_distance: Minimum distance achieved
            - best_recolored_image: Best recolored image
            - all_results: Dict with all permutation results (if return_all_results=True)
        """
        num_colors = len(source_palette)

        # Generate all possible permutations
        all_perms = list(permutations(range(num_colors)))

        logging.info(f"Testing {len(all_perms)} permutations...")

        # Precompute source image transformation space features
        logging.info("Computing source image transformation space...")
        source_transforms = self.transform_space.apply_all_transforms(source_image)
        logging.info(f"Extracting features from source transforms ({len(source_transforms)} transforms)...")
        source_features = self.feature_extractor.batch_get_features(source_transforms)

        best_distance = float('inf')
        best_permutation = None
        best_recolored_image = None

        # Store all results if requested
        all_results = {'permutations': [], 'distances': [], 'images': []} if return_all_results else None

        if use_parallel and len(all_perms) > 10:
            logging.info(f"Using optimized batch processing...")

            # Step 1: Generate all recolored images (fast, CPU only)
            logging.info("Generating all recolored images...")
            recolored_images = self._generate_recolored_images(all_perms, source_weights, target_palette)

            # Step 2: Generate all transformation spaces in parallel
            logging.info(f"Generating transformation spaces with {num_workers} workers...")

            def process_transforms(idx):
                return idx, self.transform_space.apply_all_transforms(recolored_images[idx])

            all_transforms = [None] * len(all_perms)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_transforms, i) for i in range(len(all_perms))]

                for future in futures:
                    idx, transforms = future.result()
                    all_transforms[idx] = transforms
                    if (idx + 1) % 20 == 0:
                        logging.info(f"Generated transforms for {idx + 1}/{len(all_perms)} permutations")

            # Step 3: Batch extract features for all transforms
            logging.info("Extracting features for all permutations (batched)...")

            # Flatten all transforms into one big list
            all_images_flat = []
            for transforms in all_transforms:
                all_images_flat.extend(transforms)

            # Batch process features
            if self.use_lightweight:
                # Lightweight features can process all at once
                logging.info(f"Processing {len(all_images_flat)} images with lightweight features...")
                all_features_flat = self.feature_extractor.batch_get_features(all_images_flat)
            else:
                # VGG needs batching
                vgg_batch_size = 32  # Process 32 images at a time through VGG
                all_features_flat = []

                for batch_start in range(0, len(all_images_flat), vgg_batch_size):
                    batch_end = min(batch_start + vgg_batch_size, len(all_images_flat))
                    if batch_start % (vgg_batch_size * 10) == 0:
                        logging.info(f"VGG processing {batch_start}/{len(all_images_flat)} images...")

                    batch_images = all_images_flat[batch_start:batch_end]
                    batch_features = self.feature_extractor.batch_get_features(batch_images)
                    all_features_flat.append(batch_features)

                all_features_flat = np.concatenate(all_features_flat, axis=0)

            # Reshape back to per-permutation structure
            num_transforms_per_perm = len(all_transforms[0])
            all_perm_features = []
            for i in range(len(all_perms)):
                start_idx = i * num_transforms_per_perm
                end_idx = start_idx + num_transforms_per_perm
                perm_features = all_features_flat[start_idx:end_idx]
                all_perm_features.append(perm_features)

            # Step 4: Compute distances
            logging.info("Computing Hausdorff distances...")

            for i in range(len(all_perms)):
                if i % 20 == 0:
                    logging.info(f"Computing distance for permutation {i + 1}/{len(all_perms)}...")

                distance = self.hausdorff.compute_distance(source_features, all_perm_features[i])

                # Store all results if requested
                if return_all_results:
                    all_results['permutations'].append(all_perms[i])
                    all_results['distances'].append(distance)
                    all_results['images'].append(recolored_images[i])

                if distance < best_distance:
                    best_distance = distance
                    best_permutation = all_perms[i]
                    best_recolored_image = recolored_images[i]
                    logging.info(f"  New best distance: {best_distance:.6f} at permutation {i + 1}")
        else:
            for i, perm in enumerate(all_perms):
                if i % 10 == 0:
                    logging.info(f"Testing permutation {i + 1}/{len(all_perms)}...")

                distance, recolored_image = self._test_single_permutation(
                    perm, source_weights, target_palette, source_features
                )

                # Store all results if requested
                if return_all_results:
                    all_results['permutations'].append(perm)
                    all_results['distances'].append(distance)
                    all_results['images'].append(recolored_image)

                if distance < best_distance:
                    best_distance = distance
                    best_permutation = perm
                    best_recolored_image = recolored_image
                    logging.info(f"  New best distance: {best_distance:.6f} at permutation {i + 1}")

        logging.info(f"Optimal permutation found: {best_permutation}")
        logging.info(f"Best distance: {best_distance:.6f}")

        if return_all_results:
            return np.array(best_permutation), best_distance, best_recolored_image, all_results
        else:
            return np.array(best_permutation), best_distance, best_recolored_image, None

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
        device: str = None,
        extraction_method: str = 'kmeans'
    ):
        """
        Initialize optimal palette swap.

        Args:
            num_colors: Number of colors in palette
            hue_steps: HSV hue steps for transformation space
            sat_steps: HSV saturation steps
            val_steps: HSV value steps
            device: Device for neural network
            extraction_method: Palette extraction method ('kmeans' or 'blind_separation')
        """
        from palette_extraction import PaletteExtractor

        self.palette_extractor = PaletteExtractor(num_colors=num_colors, method=extraction_method)
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
        target_palette: np.ndarray = None,
        use_parallel: bool = True,
        num_workers: int = 4
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
        # Extract palettes in parallel if both needed
        if (extract_source or source_palette is None or source_weights is None) and \
           (extract_target or target_palette is None):
            logging.info("Extracting source and target palettes in parallel...")

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_source = executor.submit(self.palette_extractor.extract_palette, source_image)
                future_target = executor.submit(self.palette_extractor.extract_palette, target_image)

                source_palette, source_weights = future_source.result()
                target_palette, _ = future_target.result()
        else:
            # Extract source palette if needed
            if extract_source or source_palette is None or source_weights is None:
                logging.info("Extracting source palette...")
                source_palette, source_weights = self.palette_extractor.extract_palette(source_image)

            # Extract target palette if needed
            if extract_target or target_palette is None:
                logging.info("Extracting target palette...")
                target_palette, _ = self.palette_extractor.extract_palette(target_image)

        # Find optimal matching
        logging.info("Finding optimal palette matching...")
        return_all = getattr(self, '_return_all_results', False)
        permutation, distance, result_image, all_results = self.palette_matcher.find_optimal_matching(
            source_image,
            source_weights,
            source_palette,
            target_palette,
            use_parallel=use_parallel,
            num_workers=num_workers,
            return_all_results=return_all
        )

        info = {
            'source_palette': source_palette,
            'target_palette': target_palette,
            'permutation': permutation,
            'distance': distance,
            'source_weights': source_weights,
            'all_results': all_results
        }

        return result_image, info
