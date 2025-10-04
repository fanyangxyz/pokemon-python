"""
Palette extraction module using k-means clustering.
Simple and robust approach for extracting dominant colors from images.
"""

import numpy as np
from PIL import Image
from typing import Tuple
import logging
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class PaletteExtractor:
    """Extract color palette from an image using k-means clustering."""

    def __init__(self, num_colors: int = 5, max_iterations: int = 300, sample_fraction: float = 0.1):
        """
        Initialize palette extractor.

        Args:
            num_colors: Number of colors in the palette
            max_iterations: Maximum k-means iterations
            sample_fraction: Fraction of pixels to sample for k-means (for speed)
        """
        self.num_colors = num_colors
        self.max_iterations = max_iterations
        self.sample_fraction = sample_fraction

    def extract_palette(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract palette and weights from an image using k-means.

        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]

        Returns:
            palette: (num_colors, 3) array of RGB colors
            weights: (H, W, num_colors) array of weights for each pixel
        """
        H, W, C = image.shape
        pixels = image.reshape(-1, 3)
        n_pixels = len(pixels)

        # Sample pixels for faster k-means
        sample_size = max(1000, int(n_pixels * self.sample_fraction))
        sample_size = min(sample_size, n_pixels)

        if sample_size < n_pixels:
            sample_indices = np.random.choice(n_pixels, sample_size, replace=False)
            sample_pixels = pixels[sample_indices]
        else:
            sample_pixels = pixels

        logging.info(f"Running k-means on {sample_size} pixels...")

        # Run k-means clustering
        kmeans = KMeans(
            n_clusters=self.num_colors,
            max_iter=self.max_iterations,
            n_init=10,
            random_state=42,
            verbose=0
        )
        kmeans.fit(sample_pixels)

        # Extract palette (cluster centers)
        palette = kmeans.cluster_centers_

        logging.info(f"K-means converged, extracted {self.num_colors} colors")

        # Assign all pixels to nearest cluster (hard assignment for crisp edges)
        # For each pixel, compute distance to all palette colors
        distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2, axis=2))

        # Hard assignment: each pixel gets weight 1.0 for nearest color, 0.0 for others
        nearest_cluster = np.argmin(distances, axis=1)
        weights = np.zeros((n_pixels, self.num_colors))
        weights[np.arange(n_pixels), nearest_cluster] = 1.0

        # Reshape weights back to image dimensions
        weights = weights.reshape(H, W, self.num_colors)

        # Log palette color statistics
        for i, color in enumerate(palette):
            color_rgb = (color * 255).astype(int)
            pixel_count = np.sum(np.argmax(weights, axis=2) == i)
            percentage = 100 * pixel_count / (H * W)
            logging.info(f"  Color {i}: RGB{tuple(color_rgb)} - {percentage:.1f}% of pixels")

        return palette, weights

    def visualize_kmeans_result(self, image: np.ndarray, palette: np.ndarray, weights: np.ndarray, save_path: str = None):
        """
        Visualize k-means clustering result showing original vs quantized image.

        Args:
            image: Original image (H, W, 3)
            palette: Extracted palette (K, 3)
            weights: Pixel assignments (H, W, K)
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt

        H, W, K = weights.shape

        # Reconstruct quantized image from palette
        weights_flat = weights.reshape(-1, K)
        quantized = np.matmul(weights_flat, palette).reshape(H, W, 3)
        quantized = np.clip(quantized, 0, 1)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Quantized image
        axes[1].imshow(quantized)
        axes[1].set_title(f'K-means Quantized ({K} colors)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Palette with statistics
        palette_height = 100
        palette_img = np.tile(palette[np.newaxis, :, :], (palette_height, 1, 1))
        axes[2].imshow(palette_img, aspect='auto')
        axes[2].set_title('Extracted Palette', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # Add color information below palette
        info_text = ""
        for i, color in enumerate(palette):
            color_rgb = (color * 255).astype(int)
            pixel_count = np.sum(np.argmax(weights, axis=2) == i)
            percentage = 100 * pixel_count / (H * W)
            info_text += f"Color {i}: RGB{tuple(color_rgb)} ({percentage:.1f}%)\n"

        fig.text(0.72, 0.15, info_text, fontsize=10, verticalalignment='top', family='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"K-means visualization saved to {save_path}")

        plt.close()

    def compute_weights(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """
        Compute hard assignment weights for an image given a palette.

        Args:
            image: Input image (H, W, 3)
            palette: Palette colors (K, 3)

        Returns:
            weights: (H, W, K) hard assignment weights
        """
        H, W, C = image.shape
        pixels = image.reshape(-1, 3)
        n_pixels = len(pixels)
        K = len(palette)

        # Compute distances to palette colors
        distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2, axis=2))

        # Hard assignment
        nearest_cluster = np.argmin(distances, axis=1)
        weights = np.zeros((n_pixels, K))
        weights[np.arange(n_pixels), nearest_cluster] = 1.0

        return weights.reshape(H, W, K)

    def apply_palette(self, weights: np.ndarray, new_palette: np.ndarray) -> np.ndarray:
        """
        Apply a new palette to existing weights.

        Args:
            weights: (H, W, num_colors) weight array
            new_palette: (num_colors, 3) new color palette

        Returns:
            New image as (H, W, 3) array
        """
        H, W, K = weights.shape
        weights_flat = weights.reshape(-1, K)  # (H*W, K)
        new_image = np.matmul(weights_flat, new_palette)  # (H*W, 3)
        return new_image.reshape(H, W, 3)


def load_image(path: str) -> np.ndarray:
    """Load image and convert to numpy array in [0, 1] range."""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def save_image(image: np.ndarray, path: str):
    """Save image from numpy array."""
    img = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
