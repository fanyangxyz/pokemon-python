"""
Palette extraction module with multiple methods:
1. K-means clustering - Simple and robust
2. Blind Color Separation - Gradient descent optimization with sparsity constraints
"""

import numpy as np
from PIL import Image
from typing import Tuple
import logging
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class PaletteExtractor:
    """Extract color palette from an image using k-means clustering or Blind Color Separation."""

    def __init__(self, num_colors: int = 5, max_iterations: int = 3000, sample_fraction: float = 0.1, method: str = 'kmeans'):
        """
        Initialize palette extractor.

        Args:
            num_colors: Number of colors in the palette
            max_iterations: Maximum iterations (for k-means or gradient descent)
            sample_fraction: Fraction of pixels to sample for k-means (for speed)
            method: Extraction method - 'kmeans' or 'blind_separation'
        """
        self.num_colors = num_colors
        self.max_iterations = max_iterations
        self.sample_fraction = sample_fraction
        self.method = method

        if method not in ['kmeans', 'blind_separation']:
            raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'blind_separation'")

    def extract_palette(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract palette and weights from an image.

        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]

        Returns:
            palette: (num_colors, 3) array of RGB colors
            weights: (H, W, num_colors) array of weights for each pixel
        """
        if self.method == 'kmeans':
            return self._extract_palette_kmeans(image)
        elif self.method == 'blind_separation':
            return self._extract_palette_blind_separation(image)

    def _extract_palette_kmeans(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract palette and weights using k-means clustering.

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

    def _extract_palette_blind_separation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract palette using Blind Color Separation with gradient descent.
        Based on L0 gradient minimization approach with sparsity constraints.

        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]

        Returns:
            palette: (num_colors, 3) array of RGB colors
            weights: (H, W, num_colors) array of weights for each pixel
        """
        H, W, C = image.shape
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logging.info(f"Running Blind Color Separation on {device}...")

        # Convert image to torch tensor
        image_tensor = torch.from_numpy(image).float().to(device)
        image_flat = image_tensor.reshape(-1, 3)  # (H*W, 3)

        # Initialize palette and weights using k-means
        from sklearn.cluster import KMeans
        pixels_np = image_flat.cpu().numpy()
        sample_size = min(10000, len(pixels_np))
        sample_indices = np.random.choice(len(pixels_np), sample_size, replace=False)
        kmeans = KMeans(n_clusters=self.num_colors, n_init=10, max_iter=100, random_state=42)
        kmeans.fit(pixels_np[sample_indices])

        # Initialize palette from k-means centers
        palette = nn.Parameter(torch.from_numpy(kmeans.cluster_centers_).float().to(device))

        # Initialize weights based on k-means assignments
        distances = torch.cdist(image_flat, palette)
        nearest = torch.argmin(distances, dim=1)
        weights_init = torch.zeros(H * W, self.num_colors, device=device)
        weights_init[torch.arange(H * W, device=device), nearest] = 1.0
        weights = nn.Parameter(weights_init)

        # Optimizer
        optimizer = optim.Adam([palette, weights], lr=0.01)

        # Progressive L0 approximation parameter (gradually increase)
        beta_start = 0.1
        beta_end = 10.0
        beta_schedule = np.logspace(np.log10(beta_start), np.log10(beta_end), self.max_iterations)

        logging.info(f"Starting optimization for {self.max_iterations} iterations...")

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # Ensure weights are non-negative and sum to 1
            weights_normalized = torch.softmax(weights, dim=1)

            # Reconstruct image
            reconstructed = torch.matmul(weights_normalized, palette)

            # Reconstruction loss (L2)
            recon_loss = torch.mean((reconstructed - image_flat) ** 2)

            # Sparsity loss (L0 approximation using smooth approximation)
            # Using sigmoid-based smooth L0: 1 / (1 + exp(-beta * (w - threshold)))
            beta = beta_schedule[iteration]
            threshold = 0.1
            sparsity_loss = torch.mean(
                torch.sigmoid(beta * (weights_normalized - threshold))
            )

            # Palette diversity loss (ensure colors are different)
            palette_distances = torch.cdist(palette, palette, p=2)
            # Penalize if colors are too similar (exclude diagonal)
            diversity_loss = -torch.mean(
                palette_distances + torch.eye(self.num_colors, device=device) * 10.0
            )

            # Color constraint: palette colors should be from the image
            # Find nearest image color for each palette color
            distances_to_image = torch.cdist(palette, image_flat, p=2)
            min_distances, _ = torch.min(distances_to_image, dim=1)
            color_constraint_loss = torch.mean(min_distances)

            # Total loss
            loss = recon_loss + 0.1 * sparsity_loss + 0.01 * diversity_loss + 0.05 * color_constraint_loss

            loss.backward()
            optimizer.step()

            # Clamp palette to [0, 1]
            with torch.no_grad():
                palette.data.clamp_(0, 1)

            if (iteration + 1) % 50 == 0:
                logging.info(
                    f"Iteration {iteration + 1}/{self.max_iterations}: "
                    f"Loss={loss.item():.6f}, "
                    f"Recon={recon_loss.item():.6f}, "
                    f"Sparsity={sparsity_loss.item():.6f}, "
                    f"Beta={beta:.2f}"
                )

        # Extract final palette and weights
        with torch.no_grad():
            final_palette = palette.detach().cpu().numpy()
            final_weights = torch.softmax(weights, dim=1).detach().cpu().numpy()
            final_weights = final_weights.reshape(H, W, self.num_colors)

        # Log palette statistics and weight distribution
        for i, color in enumerate(final_palette):
            color_rgb = (color * 255).astype(int)
            pixel_count = np.sum(np.argmax(final_weights, axis=2) == i)
            percentage = 100 * pixel_count / (H * W)

            # Check weight statistics for this color
            color_weights = final_weights[:, :, i]
            avg_weight = np.mean(color_weights)
            max_weight = np.max(color_weights)
            num_high_weight = np.sum(color_weights > 0.9)

            logging.info(
                f"  Color {i}: RGB{tuple(color_rgb)} - {percentage:.1f}% dominant pixels, "
                f"avg_weight={avg_weight:.3f}, max_weight={max_weight:.3f}, "
                f"pixels_with_weight>0.9={num_high_weight}"
            )

        return final_palette, final_weights

    def visualize_kmeans_result(self, image: np.ndarray, palette: np.ndarray, weights: np.ndarray, save_path: str = None):
        """
        Visualize palette extraction result showing original vs reconstructed image.
        For blind_separation, also shows weight layers.

        Args:
            image: Original image (H, W, 3)
            palette: Extracted palette (K, 3)
            weights: Pixel assignments (H, W, K)
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt

        H, W, K = weights.shape

        # Reconstruct image from palette
        weights_flat = weights.reshape(-1, K)
        reconstructed = np.matmul(weights_flat, palette).reshape(H, W, 3)
        reconstructed = np.clip(reconstructed, 0, 1)

        # For blind_separation, create extended visualization with weight layers
        if self.method == 'blind_separation':
            # Create grid: top row = original, reconstructed, palette
            # bottom rows = weight layers for each color
            fig = plt.figure(figsize=(18, 8 + K * 2))
            gs = fig.add_gridspec(2 + K, 3, height_ratios=[3, 3] + [2] * K)

            # Original image
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(image)
            ax0.set_title('Original Image', fontsize=14, fontweight='bold')
            ax0.axis('off')

            # Reconstructed image
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(reconstructed)
            ax1.set_title(f'Blind Separation ({K} colors)', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # Palette
            ax2 = fig.add_subplot(gs[0, 2])
            palette_height = 100
            palette_img = np.tile(palette[np.newaxis, :, :], (palette_height, 1, 1))
            ax2.imshow(palette_img, aspect='auto')
            ax2.set_title('Extracted Palette', fontsize=14, fontweight='bold')
            ax2.axis('off')

            # Weight layers for each color
            for i in range(K):
                # Weight map
                ax_weight = fig.add_subplot(gs[2 + i, 0])
                weight_map = weights[:, :, i]
                im = ax_weight.imshow(weight_map, cmap='viridis', vmin=0, vmax=1)
                color_rgb = (palette[i] * 255).astype(int)
                pixel_count = np.sum(np.argmax(weights, axis=2) == i)
                percentage = 100 * pixel_count / (H * W)
                avg_weight = np.mean(weight_map)
                max_weight = np.max(weight_map)
                ax_weight.set_title(f'Color {i} Weight Map\nRGB{tuple(color_rgb)} - {percentage:.1f}% dom\navg={avg_weight:.3f} max={max_weight:.3f}', fontsize=9)
                ax_weight.axis('off')
                plt.colorbar(im, ax=ax_weight, fraction=0.046)

                # Color layer (weight * color)
                ax_layer = fig.add_subplot(gs[2 + i, 1])
                color_layer = weight_map[:, :, np.newaxis] * palette[i][np.newaxis, np.newaxis, :]
                ax_layer.imshow(color_layer)
                ax_layer.set_title(f'Color {i} Layer', fontsize=9)
                ax_layer.axis('off')

                # Weight histogram
                ax_hist = fig.add_subplot(gs[2 + i, 2])
                ax_hist.hist(weight_map.flatten(), bins=50, range=(0, 1), color='blue', alpha=0.7)
                ax_hist.set_xlim(0, 1)
                ax_hist.set_title(f'Color {i} Weight Distribution', fontsize=9)
                ax_hist.set_xlabel('Weight')
                ax_hist.set_ylabel('Pixel Count')
                ax_hist.grid(True, alpha=0.3)

            plt.tight_layout()
        else:
            # Standard visualization for kmeans
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            # Reconstructed image
            method_name = f'{self.method.replace("_", " ").title()}'
            axes[1].imshow(reconstructed)
            axes[1].set_title(f'{method_name} ({K} colors)', fontsize=14, fontweight='bold')
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
            logging.info(f"Palette extraction visualization saved to {save_path}")

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
