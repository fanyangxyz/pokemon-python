"""
Perceptual loss using pretrained VGG network.
Implements deep image distance for better image similarity measurement.
Also includes lightweight CPU-friendly alternatives.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import List
from scipy.ndimage import gaussian_filter


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG16 network.
    Extracts features from multiple layers for better similarity measurement.
    """

    def __init__(self, layers: List[int] = None, device: str = None):
        """
        Initialize VGG perceptual loss.

        Args:
            layers: List of layer indices to extract features from.
                   Default: [3, 8, 15, 22] corresponding to relu1_2, relu2_2, relu3_3, relu4_3
            device: Device to run on ('cuda' or 'cpu')
        """
        super(VGGPerceptualLoss, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if layers is None:
            # Default layers: relu1_2, relu2_2, relu3_3, relu4_3
            self.layers = [3, 8, 15, 22]
        else:
            self.layers = layers

        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features.to(self.device).eval()

        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

        # Normalization for ImageNet pretrained models
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        )

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from specified VGG layers.

        Args:
            x: Input tensor (B, 3, H, W) in [0, 1] range

        Returns:
            List of feature tensors from specified layers
        """
        # Normalize input
        x = (x - self.mean) / self.std

        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)

        return features

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between two images.

        Args:
            x: First image (B, 3, H, W) in [0, 1]
            y: Second image (B, 3, H, W) in [0, 1]

        Returns:
            Perceptual loss value
        """
        features_x = self.extract_features(x)
        features_y = self.extract_features(y)

        loss = 0.0
        for fx, fy in zip(features_x, features_y):
            loss += torch.mean((fx - fy) ** 2)

        return loss / len(self.layers)

    def compute_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute perceptual distance between two numpy images.

        Args:
            img1: First image (H, W, 3) in [0, 1]
            img2: Second image (H, W, 3) in [0, 1]

        Returns:
            Perceptual distance
        """
        # Convert to tensor
        tensor1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        tensor2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            distance = self.forward(tensor1, tensor2).item()

        return distance

    def extract_features_numpy(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract VGG features from a numpy image.

        Args:
            image: Input image (H, W, 3) in [0, 1]

        Returns:
            List of feature arrays
        """
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            features = self.extract_features(tensor)

        # Convert to numpy and flatten
        feature_list = []
        for feat in features:
            # Flatten spatial dimensions
            feat_np = feat.cpu().numpy().reshape(feat.shape[1], -1).T  # (H*W, C)
            feature_list.append(feat_np)

        # Concatenate all features
        return feature_list


class DeepImageDistance:
    """
    Compute deep image distance using VGG features.
    """

    def __init__(self, device: str = None):
        """
        Initialize deep image distance calculator.

        Args:
            device: Device to run on
        """
        self.vgg_loss = VGGPerceptualLoss(device=device)

    def compute_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute deep distance between two images.

        Args:
            img1: First image (H, W, 3) in [0, 1]
            img2: Second image (H, W, 3) in [0, 1]

        Returns:
            Distance value
        """
        return self.vgg_loss.compute_distance(img1, img2)

    def get_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """
        Get flattened feature vector from image for distance computation.

        Args:
            image: Input image (H, W, 3) in [0, 1]

        Returns:
            Flattened feature vector
        """
        features = self.vgg_loss.extract_features_numpy(image)

        # Concatenate and average pool all features
        feature_vectors = []
        for feat in features:
            # Average over spatial dimension
            feat_avg = np.mean(feat, axis=0)  # (C,)
            feature_vectors.append(feat_avg)

        return np.concatenate(feature_vectors)

    def batch_get_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Get feature vectors for a batch of images using batched VGG forward pass.

        Args:
            images: List of images, each (H, W, 3) in [0, 1]

        Returns:
            Feature matrix (N, D) where N is number of images
        """
        if len(images) == 0:
            return np.array([])

        # Convert all images to tensor batch
        tensors = []
        for img in images:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            tensors.append(tensor)

        # Stack into batch (N, 3, H, W)
        batch_tensor = torch.stack(tensors).to(self.vgg_loss.device)

        with torch.no_grad():
            # Single batched VGG forward pass
            batch_features = self.vgg_loss.extract_features(batch_tensor)

        # Extract feature vectors for each image
        all_features = []
        for i in range(len(images)):
            feature_vectors = []
            for layer_feat in batch_features:
                # Get features for image i
                feat = layer_feat[i]  # (C, H, W)
                feat_np = feat.cpu().numpy().reshape(feat.shape[0], -1).T  # (H*W, C)
                # Average over spatial dimension
                feat_avg = np.mean(feat_np, axis=0)  # (C,)
                feature_vectors.append(feat_avg)

            all_features.append(np.concatenate(feature_vectors))

        return np.stack(all_features)


class LightweightImageFeatures:
    """
    CPU-friendly lightweight feature extractor using color histograms and statistics.
    Much faster than VGG on CPU, suitable for palette matching.
    """

    def __init__(self):
        """Initialize lightweight feature extractor."""
        self.hist_bins = 16  # Number of bins per channel for color histogram

    def extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features from image.

        Args:
            image: Input image (H, W, 3) in [0, 1]

        Returns:
            Histogram features
        """
        # Compute 3D color histogram
        hist, _ = np.histogramdd(
            image.reshape(-1, 3),
            bins=[self.hist_bins, self.hist_bins, self.hist_bins],
            range=[[0, 1], [0, 1], [0, 1]]
        )

        # Normalize
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-10)

        return hist

    def extract_spatial_color_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract spatial color features using multi-scale Gaussian pyramids.

        Args:
            image: Input image (H, W, 3) in [0, 1]

        Returns:
            Spatial color features
        """
        features = []

        # Multi-scale color statistics
        for sigma in [0, 2, 4]:
            if sigma > 0:
                smoothed = np.stack([
                    gaussian_filter(image[:, :, c], sigma=sigma)
                    for c in range(3)
                ], axis=2)
            else:
                smoothed = image

            # Compute statistics
            features.append(np.mean(smoothed, axis=(0, 1)))  # Mean color
            features.append(np.std(smoothed, axis=(0, 1)))   # Std color

        return np.concatenate(features)

    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract simple edge/gradient features.

        Args:
            image: Input image (H, W, 3) in [0, 1]

        Returns:
            Edge features
        """
        # Convert to grayscale
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

        # Compute gradients
        grad_y = np.diff(gray, axis=0, prepend=gray[0:1, :])
        grad_x = np.diff(gray, axis=1, prepend=gray[:, 0:1])

        # Edge magnitude histogram
        edge_mag = np.sqrt(grad_x**2 + grad_y**2)
        hist, _ = np.histogram(edge_mag, bins=16, range=(0, 1))
        hist = hist / (np.sum(hist) + 1e-10)

        return hist

    def get_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """
        Get complete feature vector from image.

        Args:
            image: Input image (H, W, 3) in [0, 1]

        Returns:
            Feature vector
        """
        # Color histogram
        color_hist = self.extract_color_histogram(image)

        # Spatial color features
        spatial_features = self.extract_spatial_color_features(image)

        # Edge features
        edge_features = self.extract_edge_features(image)

        # Concatenate all features
        return np.concatenate([color_hist, spatial_features, edge_features])

    def batch_get_features(self, images: List[np.ndarray], num_workers: int = None, show_progress: bool = True) -> np.ndarray:
        """
        Get feature vectors for a batch of images (parallelized).

        Args:
            images: List of images, each (H, W, 3) in [0, 1]
            num_workers: Number of parallel workers (default: CPU count)
            show_progress: Whether to show progress logging

        Returns:
            Feature matrix (N, D)
        """
        if len(images) == 0:
            return np.array([])

        # Use parallel processing for large batches
        if len(images) > 10:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import multiprocessing
            import logging

            if num_workers is None:
                num_workers = multiprocessing.cpu_count()

            features = [None] * len(images)
            completed = 0

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks with indices
                future_to_idx = {executor.submit(self.get_feature_vector, img): idx
                                for idx, img in enumerate(images)}

                # Process completed futures
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    features[idx] = future.result()
                    completed += 1

                    # Log progress every 100 images or at key milestones
                    if show_progress and (completed % 100 == 0 or completed == len(images) or
                                         completed == len(images) // 4 or completed == len(images) // 2):
                        pct = 100 * completed / len(images)
                        logging.info(f"  Feature extraction progress: {completed}/{len(images)} ({pct:.1f}%)")
        else:
            features = [self.get_feature_vector(img) for img in images]

        return np.stack(features)
