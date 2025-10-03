"""
Palette extraction module using Blind Color Separation approach.
Based on the paper by Zhang et al., with gradient descent optimization.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class PaletteExtractor:
    """Extract color palette from an image using Blind Color Separation."""

    def __init__(self, num_colors: int = 5, max_iterations: int = 200, lr: float = 0.05):
        """
        Initialize palette extractor.

        Args:
            num_colors: Number of colors in the palette
            max_iterations: Maximum number of gradient descent iterations
            lr: Learning rate for optimization
        """
        self.num_colors = num_colors
        self.max_iterations = max_iterations
        self.lr = lr

    def extract_palette(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract palette and weights from an image.

        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]

        Returns:
            palette: (num_colors, 3) array of RGB colors
            weights: (H, W, num_colors) array of weights for each pixel
        """
        H, W, C = image.shape

        # Convert to torch tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).reshape(-1, 3)  # (H*W, 3)

        # Initialize palette with k-means++ style initialization
        palette = self._initialize_palette(image_tensor)
        palette = torch.nn.Parameter(palette)

        # Initialize weights uniformly
        weights = torch.nn.Parameter(
            torch.ones(H * W, self.num_colors, dtype=torch.float32) / self.num_colors
        )

        # Optimizer
        optimizer = torch.optim.Adam([palette, weights], lr=self.lr)

        # Progressive L0 approximation parameter
        lambda_param = 2.0

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # Normalize weights to sum to 1 (non-negative constraint)
            w = F.softmax(weights, dim=1)

            # Ensure palette colors are in [0, 1]
            p = torch.sigmoid(palette)

            # Reconstruct image: I = sum(w_i * c_i)
            reconstructed = torch.matmul(w, p)  # (H*W, 3)

            # Reconstruction loss (L2)
            recon_loss = torch.mean((image_tensor - reconstructed) ** 2)

            # L0 approximation for sparsity (progressive increase of lambda)
            # Approximating L0 with smooth function: 1 - exp(-lambda * w^2)
            if iteration < self.max_iterations * 0.8:
                lambda_current = lambda_param * (1 + iteration / (self.max_iterations * 0.8))
            else:
                lambda_current = lambda_param * 2

            l0_approx = torch.mean(1 - torch.exp(-lambda_current * w ** 2))

            # Palette constraint: encourage palette to pick colors from image
            # Find nearest image colors to palette
            image_colors_unique = torch.unique(image_tensor, dim=0)
            if len(image_colors_unique) > 0:
                dist_to_image = torch.cdist(p, image_colors_unique)
                min_dist = torch.min(dist_to_image, dim=1)[0]
                palette_loss = torch.mean(min_dist)
            else:
                palette_loss = 0

            # Total loss
            loss = recon_loss + 0.1 * l0_approx + 0.05 * palette_loss

            loss.backward()
            optimizer.step()

            if iteration % 50 == 0:
                logging.info(f"Palette extraction iteration {iteration}/{self.max_iterations}: Loss={loss.item():.6f}")

        # Final weights and palette
        final_weights = F.softmax(weights, dim=1).detach().numpy().reshape(H, W, self.num_colors)
        final_palette = torch.sigmoid(palette).detach().numpy()

        return final_palette, final_weights

    def _initialize_palette(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Initialize palette using k-means++ style initialization."""
        n_pixels = image_tensor.shape[0]

        # First color: random pixel
        centers = [image_tensor[np.random.randint(n_pixels)]]

        # Remaining colors: weighted by distance to nearest center
        for _ in range(1, self.num_colors):
            distances = torch.stack([
                torch.sum((image_tensor - c) ** 2, dim=1)
                for c in centers
            ])
            min_distances = torch.min(distances, dim=0)[0]

            # Sample proportional to distance squared
            probs = min_distances / torch.sum(min_distances)
            idx = torch.multinomial(probs, 1).item()
            centers.append(image_tensor[idx])

        palette = torch.stack(centers)
        # Use inverse sigmoid for initialization since we'll apply sigmoid later
        palette = torch.logit(torch.clamp(palette, 0.01, 0.99))

        return palette

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
