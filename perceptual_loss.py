"""
Perceptual loss using pretrained VGG network.
Implements deep image distance for better image similarity measurement.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import List


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
