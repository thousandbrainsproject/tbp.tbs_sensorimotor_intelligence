# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, ViTModel

from .norm_linear_head import NormLinearHead

ProjectionInfo = Tuple[torch.Tensor, torch.Tensor, tuple, tuple, tuple]
class ViTRgbdObjectClassifierWithRotation(nn.Module):
    """Vision Transformer model adapted for RGB-D input with classification and quaternion
    prediction.

    This model extends a standard ViT to handle 4-channel RGB-D input by modifying the patch
    embedding projection. It includes two heads: one for classification and one for quaternion
    prediction.

    Args:
        model_name: Name of the base ViT model to use
        num_classes: Number of classification classes
        freeze_backbone: Whether to freeze the backbone parameters
        use_pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        freeze_backbone: bool = False,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()
        model_id = MODEL_DICT[model_name]
        config = AutoConfig.from_pretrained(model_id)
        config.num_channels = 4

        if use_pretrained:
            rgb_weights, rgb_bias, stride, kernel_size, padding = (
                extract_rgb_projection_information(model_name)
            )

            vit = ViTModel.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)

            new_proj = self._create_rgbd_projection(
                rgb_weights, rgb_bias, stride, kernel_size, padding
            )
            vit.embeddings.patch_embeddings.projection = new_proj
        else:
            vit = ViTModel(config)

        # Initialize heads
        self.classification_head = NormLinearHead(config.hidden_size, num_classes)
        self.quaternion_head = NormLinearHead(config.hidden_size, 4, normalize_output=True)

        self.vit = vit

        if freeze_backbone:
            self.freeze_backbone()

    def _create_rgbd_projection(
        self,
        rgb_weights: torch.Tensor,
        rgb_bias: torch.Tensor,
        stride: tuple,
        kernel_size: tuple,
        padding: tuple,
    ) -> nn.Conv2d:
        """Create a new projection layer for RGB-D input.

        Args:
            rgb_weights: Original RGB weights
            rgb_bias: Original RGB bias
            stride: Convolution stride
            kernel_size: Convolution kernel size
            padding: Convolution padding

        Returns:
            New Conv2d layer for RGB-D input
        """
        out_channels, _, _, _ = rgb_weights.shape

        new_proj = nn.Conv2d(
            in_channels=4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        with torch.no_grad():
            depth_weight = rgb_weights.mean(dim=1, keepdim=True)
            new_weight = torch.cat([rgb_weights, depth_weight], dim=1)

            new_proj.weight = nn.Parameter(new_weight)
            new_proj.bias = nn.Parameter(rgb_bias.clone())

        return new_proj

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 4, height, width)

        Returns:
            Tuple of (classification_logits, quaternion_predictions)
        """
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0]
        pred_class = self.classification_head(cls_token)
        pred_quaternion = self.quaternion_head(cls_token)
        return pred_class, pred_quaternion

    def freeze_backbone(self) -> None:
        """Freeze the backbone parameters while keeping heads trainable."""
        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.embeddings.patch_embeddings.projection.parameters():
            param.requires_grad = True

        for param in self.classification_head.parameters():
            param.requires_grad = True
        for param in self.quaternion_head.parameters():
            param.requires_grad = True


def extract_rgb_projection_information(
    model_name: str,
) -> ProjectionInfo:
    """Extract RGB projection information from a pretrained ViT model.

    Args:
        model_name: Name of the model to extract information from

    Returns:
        Tuple containing (weights, bias, stride, kernel_size, padding)
    """
    model_id = MODEL_DICT[model_name]
    vit = ViTModel.from_pretrained(model_id)

    with torch.no_grad():
        weights = vit.embeddings.patch_embeddings.projection.weight.data.clone()
        bias = vit.embeddings.patch_embeddings.projection.bias.data.clone()

    stride = vit.embeddings.patch_embeddings.projection.stride
    kernel_size = vit.embeddings.patch_embeddings.projection.kernel_size
    padding = vit.embeddings.patch_embeddings.projection.padding
    del vit
    return weights, bias, stride, kernel_size, padding


MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l16-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
}
