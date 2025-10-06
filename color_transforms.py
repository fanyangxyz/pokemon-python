"""
Dense color transformation space implementation.
Provides HSV-based color transformations for creating image variations.
"""

import numpy as np
from typing import List, Union
import colorsys
import torch


class ColorTransformSpace:
    """Generate dense color transformation space using HSV shifts."""

    def __init__(self, hue_steps: int = 12, sat_steps: int = 5, val_steps: int = 5):
        """
        Initialize color transform space.

        Args:
            hue_steps: Number of hue shift steps
            sat_steps: Number of saturation shift steps
            val_steps: Number of value/brightness shift steps
        """
        self.hue_steps = hue_steps
        self.sat_steps = sat_steps
        self.val_steps = val_steps

    def generate_transforms(self) -> List[dict]:
        """
        Generate all transformation parameters.

        Returns:
            List of dictionaries with 'hue', 'sat', 'val' shift parameters
        """
        transforms = []

        # Hue: shift from -180 to 180 degrees
        hue_shifts = np.linspace(-0.5, 0.5, self.hue_steps)

        # Saturation: multiply from 0.5 to 1.5
        sat_shifts = np.linspace(0.7, 1.3, self.sat_steps)

        # Value: multiply from 0.5 to 1.5
        val_shifts = np.linspace(0.7, 1.3, self.val_steps)

        for h_shift in hue_shifts:
            for s_shift in sat_shifts:
                for v_shift in val_shifts:
                    transforms.append({
                        'hue': h_shift,
                        'sat': s_shift,
                        'val': v_shift
                    })

        return transforms

    def apply_transform(self, image: np.ndarray, transform: dict) -> np.ndarray:
        """
        Apply color transformation to an image.

        Args:
            image: Input image (H, W, 3) in RGB, values in [0, 1]
            transform: Dictionary with 'hue', 'sat', 'val' parameters

        Returns:
            Transformed image (H, W, 3) in RGB
        """
        H, W, C = image.shape

        # Convert RGB to HSV
        hsv_image = self.rgb_to_hsv(image)

        # Apply transformations
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + transform['hue']) % 1.0  # Hue shift
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * transform['sat'], 0, 1)  # Sat scale
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * transform['val'], 0, 1)  # Val scale

        # Convert back to RGB
        rgb_image = self.hsv_to_rgb(hsv_image)

        return rgb_image

    def apply_all_transforms(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply all transformations to an image.

        Args:
            image: Input image (H, W, 3) in RGB

        Returns:
            List of transformed images
        """
        transforms = self.generate_transforms()
        return [self.apply_transform(image, t) for t in transforms]

    @staticmethod
    def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to HSV.

        Args:
            rgb: (H, W, 3) array in [0, 1]

        Returns:
            hsv: (H, W, 3) array with H in [0, 1], S in [0, 1], V in [0, 1]
        """
        H, W, C = rgb.shape
        hsv = np.zeros_like(rgb)

        for i in range(H):
            for j in range(W):
                r, g, b = rgb[i, j]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv[i, j] = [h, s, v]

        return hsv

    @staticmethod
    def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        """
        Convert HSV image to RGB.

        Args:
            hsv: (H, W, 3) array

        Returns:
            rgb: (H, W, 3) array in [0, 1]
        """
        H, W, C = hsv.shape
        rgb = np.zeros_like(hsv)

        for i in range(H):
            for j in range(W):
                h, s, v = hsv[i, j]
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                rgb[i, j] = [r, g, b]

        return rgb


class FastColorTransformSpace:
    """
    Optimized version using vectorized operations for faster transformation.
    """

    def __init__(self, hue_steps: int = 12, sat_steps: int = 5, val_steps: int = 5):
        self.hue_steps = hue_steps
        self.sat_steps = sat_steps
        self.val_steps = val_steps

    def generate_transforms(self) -> List[dict]:
        """Generate all transformation parameters."""
        transforms = []
        hue_shifts = np.linspace(-0.5, 0.5, self.hue_steps)
        sat_shifts = np.linspace(0.7, 1.3, self.sat_steps)
        val_shifts = np.linspace(0.7, 1.3, self.val_steps)

        for h_shift in hue_shifts:
            for s_shift in sat_shifts:
                for v_shift in val_shifts:
                    transforms.append({
                        'hue': h_shift,
                        'sat': s_shift,
                        'val': v_shift
                    })

        return transforms

    def apply_transform(self, image: np.ndarray, transform: dict) -> np.ndarray:
        """Apply color transformation using vectorized operations."""
        # Vectorized RGB to HSV
        hsv = self._rgb_to_hsv_vectorized(image)

        # Apply transformations
        hsv[:, :, 0] = (hsv[:, :, 0] + transform['hue']) % 1.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * transform['sat'], 0, 1)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * transform['val'], 0, 1)

        # Vectorized HSV to RGB
        rgb = self._hsv_to_rgb_vectorized(hsv)

        return rgb

    def apply_all_transforms(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply all transformations to an image."""
        transforms = self.generate_transforms()
        return [self.apply_transform(image, t) for t in transforms]

    @staticmethod
    def _rgb_to_hsv_vectorized(rgb: np.ndarray) -> np.ndarray:
        """Vectorized RGB to HSV conversion."""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc

        deltac = maxc - minc
        s = deltac / (maxc + 1e-8)
        s[maxc == 0] = 0

        # Hue calculation
        rc = (maxc - r) / (deltac + 1e-8)
        gc = (maxc - g) / (deltac + 1e-8)
        bc = (maxc - b) / (deltac + 1e-8)

        h = np.zeros_like(r)
        h[r == maxc] = bc[r == maxc] - gc[r == maxc]
        h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
        h[b == maxc] = 4.0 + gc[b == maxc] - rc[b == maxc]

        h = (h / 6.0) % 1.0
        h[deltac == 0] = 0

        return np.stack([h, s, v], axis=-1)

    @staticmethod
    def _hsv_to_rgb_vectorized(hsv: np.ndarray) -> np.ndarray:
        """Vectorized HSV to RGB conversion."""
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        rgb = np.zeros_like(hsv)

        # Create masks for each case
        mask0 = (i == 0)
        mask1 = (i == 1)
        mask2 = (i == 2)
        mask3 = (i == 3)
        mask4 = (i == 4)
        mask5 = (i == 5)

        rgb[mask0] = np.stack([v[mask0], t[mask0], p[mask0]], axis=-1)
        rgb[mask1] = np.stack([q[mask1], v[mask1], p[mask1]], axis=-1)
        rgb[mask2] = np.stack([p[mask2], v[mask2], t[mask2]], axis=-1)
        rgb[mask3] = np.stack([p[mask3], q[mask3], v[mask3]], axis=-1)
        rgb[mask4] = np.stack([t[mask4], p[mask4], v[mask4]], axis=-1)
        rgb[mask5] = np.stack([v[mask5], p[mask5], q[mask5]], axis=-1)

        return rgb


class GPUColorTransformSpace:
    """
    GPU-accelerated color transformation space using PyTorch.
    Much faster than CPU version, especially for large batches.
    """

    def __init__(self, hue_steps: int = 12, sat_steps: int = 5, val_steps: int = 5, device: str = None):
        self.hue_steps = hue_steps
        self.sat_steps = sat_steps
        self.val_steps = val_steps

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def generate_transforms(self) -> List[dict]:
        """Generate all transformation parameters."""
        transforms = []
        hue_shifts = np.linspace(-0.5, 0.5, self.hue_steps)
        sat_shifts = np.linspace(0.7, 1.3, self.sat_steps)
        val_shifts = np.linspace(0.7, 1.3, self.val_steps)

        for h_shift in hue_shifts:
            for s_shift in sat_shifts:
                for v_shift in val_shifts:
                    transforms.append({
                        'hue': h_shift,
                        'sat': s_shift,
                        'val': v_shift
                    })

        return transforms

    def apply_all_transforms(self, image: np.ndarray, batch_size: int = 45) -> List[np.ndarray]:
        """
        Apply all transformations using fully batched GPU operations.
        Most efficient version - processes multiple transforms simultaneously.

        Args:
            image: Input image (H, W, 3) in RGB, values in [0, 1]
            batch_size: Number of transforms to process in parallel (default 45 = full batch)

        Returns:
            List of transformed images as numpy arrays
        """
        # Generate all transform parameters
        transforms = self.generate_transforms()
        num_transforms = len(transforms)

        # Convert image to tensor
        img_tensor = torch.from_numpy(image).float().to(self.device)

        # Pre-convert to HSV once
        hsv = self._rgb_to_hsv_torch(img_tensor)  # (H, W, 3)

        results = []

        # Process in batches
        for batch_start in range(0, num_transforms, batch_size):
            batch_end = min(batch_start + batch_size, num_transforms)
            batch_transforms = transforms[batch_start:batch_end]
            current_batch_size = len(batch_transforms)

            # Replicate HSV image for batch processing (B, H, W, 3)
            hsv_batch = hsv.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)

            # Extract transform parameters as tensors
            h_shifts = torch.tensor([t['hue'] for t in batch_transforms], device=self.device).view(-1, 1, 1, 1)
            s_shifts = torch.tensor([t['sat'] for t in batch_transforms], device=self.device).view(-1, 1, 1, 1)
            v_shifts = torch.tensor([t['val'] for t in batch_transforms], device=self.device).view(-1, 1, 1, 1)

            # Apply all transforms in batch
            hsv_batch[:, :, :, 0:1] = (hsv_batch[:, :, :, 0:1] + h_shifts) % 1.0
            hsv_batch[:, :, :, 1:2] = torch.clamp(hsv_batch[:, :, :, 1:2] * s_shifts, 0, 1)
            hsv_batch[:, :, :, 2:3] = torch.clamp(hsv_batch[:, :, :, 2:3] * v_shifts, 0, 1)

            # Convert batch back to RGB
            rgb_batch = self._hsv_to_rgb_torch_batch(hsv_batch)

            # Convert to numpy and append
            rgb_numpy = rgb_batch.cpu().numpy()
            for i in range(current_batch_size):
                results.append(rgb_numpy[i])

        return results

    @staticmethod
    def _rgb_to_hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
        """
        Vectorized RGB to HSV conversion using PyTorch.

        Args:
            rgb: (H, W, 3) tensor in [0, 1]

        Returns:
            hsv: (H, W, 3) tensor
        """
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        maxc = torch.max(torch.max(r, g), b)
        minc = torch.min(torch.min(r, g), b)
        v = maxc

        deltac = maxc - minc
        s = deltac / (maxc + 1e-8)
        s = torch.where(maxc == 0, torch.zeros_like(s), s)

        # Hue calculation
        rc = (maxc - r) / (deltac + 1e-8)
        gc = (maxc - g) / (deltac + 1e-8)
        bc = (maxc - b) / (deltac + 1e-8)

        h = torch.zeros_like(r)
        h = torch.where(r == maxc, bc - gc, h)
        h = torch.where(g == maxc, 2.0 + rc - bc, h)
        h = torch.where(b == maxc, 4.0 + gc - rc, h)

        h = (h / 6.0) % 1.0
        h = torch.where(deltac == 0, torch.zeros_like(h), h)

        return torch.stack([h, s, v], dim=-1)

    @staticmethod
    def _hsv_to_rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
        """
        Vectorized HSV to RGB conversion using PyTorch.

        Args:
            hsv: (H, W, 3) tensor

        Returns:
            rgb: (H, W, 3) tensor in [0, 1]
        """
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        i = (h * 6.0).long()
        f = (h * 6.0) - i.float()
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        # Create masks for each case
        mask0 = (i == 0)
        mask1 = (i == 1)
        mask2 = (i == 2)
        mask3 = (i == 3)
        mask4 = (i == 4)
        mask5 = (i == 5)

        rgb = torch.zeros_like(hsv)
        rgb[mask0] = torch.stack([v[mask0], t[mask0], p[mask0]], dim=-1)
        rgb[mask1] = torch.stack([q[mask1], v[mask1], p[mask1]], dim=-1)
        rgb[mask2] = torch.stack([p[mask2], v[mask2], t[mask2]], dim=-1)
        rgb[mask3] = torch.stack([p[mask3], q[mask3], v[mask3]], dim=-1)
        rgb[mask4] = torch.stack([t[mask4], p[mask4], v[mask4]], dim=-1)
        rgb[mask5] = torch.stack([v[mask5], p[mask5], q[mask5]], dim=-1)

        return rgb

    @staticmethod
    def _hsv_to_rgb_torch_batch(hsv: torch.Tensor) -> torch.Tensor:
        """
        Batched HSV to RGB conversion using PyTorch.

        Args:
            hsv: (B, H, W, 3) tensor

        Returns:
            rgb: (B, H, W, 3) tensor in [0, 1]
        """
        h, s, v = hsv[:, :, :, 0], hsv[:, :, :, 1], hsv[:, :, :, 2]

        i = (h * 6.0).long()
        f = (h * 6.0) - i.float()
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        # Create masks for each case
        mask0 = (i == 0)
        mask1 = (i == 1)
        mask2 = (i == 2)
        mask3 = (i == 3)
        mask4 = (i == 4)
        mask5 = (i == 5)

        rgb = torch.zeros_like(hsv)
        rgb[mask0] = torch.stack([v[mask0], t[mask0], p[mask0]], dim=-1)
        rgb[mask1] = torch.stack([q[mask1], v[mask1], p[mask1]], dim=-1)
        rgb[mask2] = torch.stack([p[mask2], v[mask2], t[mask2]], dim=-1)
        rgb[mask3] = torch.stack([p[mask3], q[mask3], v[mask3]], dim=-1)
        rgb[mask4] = torch.stack([t[mask4], p[mask4], v[mask4]], dim=-1)
        rgb[mask5] = torch.stack([v[mask5], p[mask5], q[mask5]], dim=-1)

        return rgb
