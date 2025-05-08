# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Count the FLOPs (Floating Point Operations) in the ViT model.

This script calculates the computational complexity of a ViT (Vision Transformer) model
in terms of FLOPs. It provides functions to compute both forward pass FLOPs and total
training FLOPs, including pretraining on ImageNet-21K and finetuning on the YCB dataset.

The script handles a custom RGBD ViT model that processes both RGB and depth information
(4 channels total) and outputs both classification and quaternion predictions. The model
architecture is based on ViT-B/16 with a 224x224 input resolution.

For clarity, we break down the FLOPs calculation:

1. Forward Pass FLOPs:
   - Calculated using the calflops library
   - Input shape is (1, 4, 224, 224) for a single RGBD image

2. Training FLOPs:
   - Forward pass FLOPs (as above)
   - Backward pass FLOPs (estimated as 2x forward pass)
   - Multiplied by number of images and epochs
   - Separate calculations for pretraining and finetuning

The script provides functions to calculate:
- Single forward pass FLOPs
- Total training FLOPs for any dataset
- Specific calculation for ImageNet-21K pretraining
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from calflops import calculate_flops
# TODO (Hojae): Update import path once ViT code is merged into this repo.
from benchmark_vit.src.models.custom_vit import RGBD_ViT

if TYPE_CHECKING:
    import torch

# Model Architecture Parameters
backbone_vit_model: str = "vit-b16-224-in21k" 
num_ycb_classes: int = 77
input_shape: Tuple[int, int, int, int] = (1, 4, 224, 224)  # (batch_size, channels, height, width)

# Model Configuration
freeze_backbone: bool = True
classification_head_type: str = "norm_linear"
quaternion_head_type: str = "norm_linear"
use_pretrained: bool = True

# Training Parameters for YCB Dataset
num_rotations_per_class: int = 14
num_images: int = num_ycb_classes * num_rotations_per_class
num_epochs_for_pretrained_vit: int = 25
num_epochs_for_vit: int = 75

# ImageNet-21K Pretraining Parameters
imagenet21k_num_images: int = 14_000_000
imagenet21k_num_epochs: int = 90  # See Table 3 of ViT paper; value for ViT-B/16

model = RGBD_ViT(
    model_name=backbone_vit_model,
    num_classes=num_ycb_classes,
    freeze_backbone=freeze_backbone, # Has no impact on FLOPs
    classification_head_type=classification_head_type,
    quaternion_head_type=quaternion_head_type,
    use_pretrained=use_pretrained, # Has no impact on FLOPs
)

def get_forward_flops(model: torch.nn.Module, input_shape: Tuple[int, int, int, int]) -> float:
    """Get the forward FLOPs of a model.
    
    Args:
        model: The model to get the forward FLOPs of.
        input_shape: The shape of the input to the model.

    Returns:
        The forward FLOPs of the model. If input_shape has a batch size of 1, then
        the FLOPs are for a forward pass of a single image.
        
    Raises:
        ValueError: If input_shape has a batch size other than 1.
        RuntimeError: If FLOPs calculation fails.
    """
    if input_shape[0] != 1:
        raise ValueError("Input shape must have a batch size of 1")
    
    try:
        flops, _, _ = calculate_flops(model=model, 
                                    input_shape=input_shape,
                                    output_as_string=False,
                                    output_precision=4)
        return flops
    except Exception as e:
        raise RuntimeError(f"Failed to calculate FLOPs: {str(e)}")

def calculate_training_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    num_images: int,
    num_epochs: int
) -> float:
    """Calculate total FLOPs for training a model (pretraining or finetuning).
    
    Args:
        model: The model to calculate FLOPs for.
        input_shape: The shape of the input to the model (batch_size, channels, height, width).
        num_images: Total number of images in the dataset.
        num_epochs: Number of epochs to train for.
        
    Returns:
        Total FLOPs for the entire training process.
        
    Raises:
        ValueError: If num_images or num_epochs is non-positive.
        RuntimeError: If FLOPs calculation fails.
    """
    if num_images <= 0:
        raise ValueError("Number of images must be positive")
    if num_epochs <= 0:
        raise ValueError("Number of epochs must be positive")
        
    try:
        # Calculate FLOPs for a single forward pass
        forward_flops = get_forward_flops(model, input_shape)
        
        # Backward pass is typically 2x forward pass
        backward_flops = 2 * forward_flops
        
        # Total FLOPs per image (forward + backward)
        flops_per_image = forward_flops + backward_flops
        
        # Total FLOPs for entire training
        total_flops = flops_per_image * num_images * num_epochs
        
        return total_flops
    except Exception as e:
        raise RuntimeError(f"Failed to calculate training FLOPs: {str(e)}")

def get_imagenet21k_pretraining_flops(model: torch.nn.Module) -> float:
    """Calculate total FLOPs for pretraining on ImageNet-21K.
    
    Args:
        model: The model to calculate FLOPs for.
        
    Returns:
        Total FLOPs for pretraining on ImageNet-21K dataset.
    """
    return calculate_training_flops(
        model=model,
        input_shape=input_shape,
        num_images=imagenet21k_num_images,
        num_epochs=imagenet21k_num_epochs
    )

if __name__ == "__main__":
    try:
        # Calculate and print forward pass FLOPs
        forward_flops = get_forward_flops(model, input_shape)
        print("\nViT Model FLOPs Analysis")
        print("=" * 50)
        print(f"Model Configuration:")
        print(f"- Backbone: {backbone_vit_model}")
        print(f"- Input Shape: {input_shape}")
        print(f"- Classification Head: {classification_head_type}")
        print(f"- Quaternion Head: {quaternion_head_type}")
        print(f"\nForward pass FLOPs: {forward_flops:,.0f}")
        
        # Calculate and print training FLOPs
        training_flops = calculate_training_flops(
            model, input_shape, num_images, num_epochs_for_pretrained_vit
        )
        print(f"\nTraining FLOPs (YCB dataset):")
        print(f"- Number of images: {num_images:,}")
        print(f"- Number of epochs: {num_epochs_for_pretrained_vit}")
        print(f"- Total FLOPs: {training_flops:,.0f}")
        
        # Calculate and print pretraining FLOPs
        pretraining_flops = get_imagenet21k_pretraining_flops(model)
        print(f"\nPretraining FLOPs (ImageNet-21K):")
        print(f"- Number of images: {imagenet21k_num_images:,}")
        print(f"- Number of epochs: {imagenet21k_num_epochs}")
        print(f"- Total FLOPs: {pretraining_flops:,.0f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
