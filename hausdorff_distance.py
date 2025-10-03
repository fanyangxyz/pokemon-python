"""
Hausdorff distance computation for image space distance.
Used to measure distance between sets of transformed images.
"""

import numpy as np
from typing import List
from scipy.spatial.distance import cdist


class HausdorffDistance:
    """
    Compute Hausdorff distance between two point sets.
    Used for measuring distance between image transformation spaces.
    """

    @staticmethod
    def directed_hausdorff(set_a: np.ndarray, set_b: np.ndarray) -> float:
        """
        Compute directed Hausdorff distance from set_a to set_b.

        h(A, B) = max_{a in A} min_{b in B} d(a, b)

        Args:
            set_a: First point set (N, D)
            set_b: Second point set (M, D)

        Returns:
            Directed Hausdorff distance
        """
        # Compute pairwise distances
        distances = cdist(set_a, set_b, metric='euclidean')

        # For each point in A, find minimum distance to B
        min_distances = np.min(distances, axis=1)

        # Take maximum of these minimum distances
        hausdorff_dist = np.max(min_distances)

        return hausdorff_dist

    @staticmethod
    def hausdorff_distance(set_a: np.ndarray, set_b: np.ndarray) -> float:
        """
        Compute symmetric Hausdorff distance.

        H(A, B) = max(h(A, B), h(B, A))

        Args:
            set_a: First point set (N, D)
            set_b: Second point set (M, D)

        Returns:
            Symmetric Hausdorff distance
        """
        h_ab = HausdorffDistance.directed_hausdorff(set_a, set_b)
        h_ba = HausdorffDistance.directed_hausdorff(set_b, set_a)

        return max(h_ab, h_ba)

    @staticmethod
    def modified_hausdorff_distance(set_a: np.ndarray, set_b: np.ndarray) -> float:
        """
        Compute modified Hausdorff distance (average instead of max).

        More robust to outliers.

        Args:
            set_a: First point set (N, D)
            set_b: Second point set (M, D)

        Returns:
            Modified Hausdorff distance
        """
        # A to B
        distances_ab = cdist(set_a, set_b, metric='euclidean')
        min_distances_ab = np.min(distances_ab, axis=1)
        avg_ab = np.mean(min_distances_ab)

        # B to A
        min_distances_ba = np.min(distances_ab, axis=0)
        avg_ba = np.mean(min_distances_ba)

        return max(avg_ab, avg_ba)


class ImageSpaceDistance:
    """
    Compute distance between image transformation spaces using Hausdorff distance.
    """

    def __init__(self, use_modified: bool = True):
        """
        Initialize image space distance calculator.

        Args:
            use_modified: If True, use modified Hausdorff distance (more robust)
        """
        self.use_modified = use_modified

    def compute_distance(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray
    ) -> float:
        """
        Compute distance between two feature sets.

        Args:
            features_a: Feature set from first image space (N, D)
            features_b: Feature set from second image space (M, D)

        Returns:
            Distance between the two spaces
        """
        if self.use_modified:
            return HausdorffDistance.modified_hausdorff_distance(features_a, features_b)
        else:
            return HausdorffDistance.hausdorff_distance(features_a, features_b)

    def compute_image_space_distance(
        self,
        image_set_a: List[np.ndarray],
        image_set_b: List[np.ndarray],
        feature_extractor
    ) -> float:
        """
        Compute distance between two sets of images using their features.

        Args:
            image_set_a: List of images from first transformation space
            image_set_b: List of images from second transformation space
            feature_extractor: Feature extractor with batch_get_features method

        Returns:
            Distance between the two image spaces
        """
        # Extract features
        features_a = feature_extractor.batch_get_features(image_set_a)
        features_b = feature_extractor.batch_get_features(image_set_b)

        # Compute Hausdorff distance
        return self.compute_distance(features_a, features_b)


class ApproximateHausdorff:
    """
    Approximate Hausdorff distance using sampling for faster computation.
    """

    def __init__(self, sample_size: int = 100):
        """
        Initialize approximate Hausdorff distance.

        Args:
            sample_size: Number of samples to use for approximation
        """
        self.sample_size = sample_size

    def compute_distance(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray
    ) -> float:
        """
        Compute approximate Hausdorff distance using sampling.

        Args:
            features_a: Feature set (N, D)
            features_b: Feature set (M, D)

        Returns:
            Approximate Hausdorff distance
        """
        # Sample if sets are too large
        if len(features_a) > self.sample_size:
            indices_a = np.random.choice(len(features_a), self.sample_size, replace=False)
            features_a = features_a[indices_a]

        if len(features_b) > self.sample_size:
            indices_b = np.random.choice(len(features_b), self.sample_size, replace=False)
            features_b = features_b[indices_b]

        # Compute modified Hausdorff on sampled sets
        return HausdorffDistance.modified_hausdorff_distance(features_a, features_b)
