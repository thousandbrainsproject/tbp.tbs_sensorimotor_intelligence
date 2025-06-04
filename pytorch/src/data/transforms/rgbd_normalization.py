from typing import TypeAlias

import torch
import torchvision.transforms as T

# Type aliases for improved readability
RGBNormParams: TypeAlias = tuple[float, float, float]


class RGBDNormalization:
    """Applies normalization to RGB-D data for compatibility with ViT models.

    This transform handles the preprocessing of RGB-D (RGB + Depth) data by:
    1. Normalizing RGB channels using ViT-specific parameters
    2. Normalizing depth channels to the same range as RGB

    The RGB normalization uses the same parameters as Hugging Face ViT models
    (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) to ensure compatibility with
    pre-trained models. For more details, please see:
    https://huggingface.co/google/vit-base-patch16-224-in21k#preprocessing

    This transform serves as the default preprocessing for both training and validation
    datasets when no custom transforms are provided. It ensures consistent normalization
    across all data splits, maintaining compatibility with pre-trained ViT models.

    Attributes:
        rgb_transform: Normalization transform for RGB channels.
        rgb_means: Mean values for RGB channel normalization.
        rgb_stds: Standard deviation values for RGB channel normalization.
    """

    def __init__(
        self,
        rgb_means: RGBNormParams = (0.5, 0.5, 0.5),
        rgb_stds: RGBNormParams = (0.5, 0.5, 0.5),
    ) -> None:
        """Initializes the RGB-D normalization transform.

        Args:
            rgb_means: Mean values for RGB channel normalization.
                Defaults to (0.5, 0.5, 0.5).
            rgb_stds: Standard deviation values for RGB channel normalization.
                Defaults to (0.5, 0.5, 0.5).

        Raises:
            ValueError: If rgb_means or rgb_stds don't have exactly 3 values.
        """
        if len(rgb_means) != 3 or len(rgb_stds) != 3:
            raise ValueError(
                "RGB means and standard deviations must have exactly 3 values"
            )

        self.rgb_means: RGBNormParams = rgb_means
        self.rgb_stds: RGBNormParams = rgb_stds
        self.rgb_transform: T.Normalize = T.Normalize(mean=rgb_means, std=rgb_stds)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization to the input RGB-D tensor.

        Args:
            x: Input RGB-D tensor with shape [C, H, W] where C >= 4.
                First 3 channels are RGB, remaining are depth.
                Expected value range is [0, 1] for both RGB and depth.

        Returns:
            Normalized RGB-D tensor with the same shape.
                Both RGB and depth channels are normalized to [-1, 1].

        Raises:
            TypeError: If input is not a torch.Tensor.
            ValueError: If input tensor has fewer than 4 channels or
                incorrect dimension order, or if values are outside [0, 1] range.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D tensor [C, H, W], got shape {x.shape}")

        if x.shape[0] < 4:
            raise ValueError(
                f"Expected at least 4 channels (RGB + depth), got {x.shape[0]}"
            )

        rgb: torch.Tensor = x[:3, :, :]
        depth: torch.Tensor = x[3:, :, :]

        # Ensure input values are in [0, 1] range
        if torch.any(rgb < 0) or torch.any(rgb > 1):
            raise ValueError("RGB values must be in range [0, 1]")
        if torch.any(depth < 0) or torch.any(depth > 1):
            raise ValueError("Depth values must be in range [0, 1]")

        rgb = self.rgb_transform(rgb)  # Normalize RGB channels to [-1, 1]
        depth = depth * 2 - 1  # Normalize depth channels to [-1, 1]

        return torch.cat([rgb, depth], dim=0)
