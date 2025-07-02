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
   - Backward pass FLOPs (estimated as 2x forward pass, following Kaplan et al. 2020,
   "Scaling laws for neural language models.")
   - Multiplied by number of images and epochs
   - Separate calculations for pretraining and finetuning

The script provides functions to calculate:
- Single forward pass FLOPs
- Total training FLOPs for any dataset
- Specific calculation for ImageNet-21K pretraining
"""

from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING, Tuple

from calflops import calculate_flops

import sys
import os

# Add paths relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_path = os.path.join(script_dir, "..", "pytorch")
monty_path = os.path.join(script_dir, "..", "monty")

sys.path.append(pytorch_path)
sys.path.append(monty_path)
from configs.common import DMC_RESULTS_DIR
from src.models.components.rgbd_vit import ViTRgbdObjectClassifierWithRotation

if TYPE_CHECKING:
    import torch

# Model Architecture Parameters
num_ycb_classes: int = 77
input_shape: Tuple[int, int, int, int] = (1, 4, 224, 224)  # (batch_size, channels, height, width)

# Model Configuration
freeze_backbone: bool = True
use_pretrained: bool = True

# Training Parameters for YCB Dataset
num_rotations_per_class: int = 14
num_images: int = num_ycb_classes * num_rotations_per_class
num_epochs_for_pretrained_vit: int = 25
num_epochs_for_vit: int = 75

# ImageNet-21K Pretraining Parameters
imagenet21k_num_images: int = 14_000_000
imagenet21k_num_epochs: int = 90  # See Table 3 of ViT paper - value for ViT-B/16 (Dosovitskiy et al, 2020)

# Create different ViT model instances
vit_b16 = ViTRgbdObjectClassifierWithRotation(
    model_name="vit-b16-224-in21k",
    num_classes=num_ycb_classes,
    freeze_backbone=freeze_backbone,
    use_pretrained=use_pretrained,
)

vit_b32 = ViTRgbdObjectClassifierWithRotation(
    model_name="vit-b32-224-in21k",
    num_classes=num_ycb_classes,
    freeze_backbone=freeze_backbone,
    use_pretrained=use_pretrained,
)

vit_l16 = ViTRgbdObjectClassifierWithRotation(
    model_name="vit-l16-224-in21k",
    num_classes=num_ycb_classes,
    freeze_backbone=freeze_backbone,
    use_pretrained=use_pretrained,
)

vit_l32 = ViTRgbdObjectClassifierWithRotation(
    model_name="vit-l32-224-in21k",
    num_classes=num_ycb_classes,
    freeze_backbone=freeze_backbone,
    use_pretrained=use_pretrained,
)

vit_h14 = ViTRgbdObjectClassifierWithRotation(
    model_name="vit-h14-224-in21k",
    num_classes=num_ycb_classes,
    freeze_backbone=freeze_backbone,
    use_pretrained=use_pretrained,
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
        print("\nViT Model FLOPs Analysis")
        print("=" * 50)
        print("Model Configuration:")
        print(f"- Input Shape: {input_shape}")
        
        # Fig8a: Training FLOPs
        print("\nFig8a: Training FLOPs")
        print("-" * 30)
        
        # 1. Training from scratch
        vit_b16_scratch_training_flops = calculate_training_flops(
            vit_b16, input_shape, num_images, num_epochs_for_vit
        )
        print(f"1. Training from scratch: {vit_b16_scratch_training_flops:,.0f}")
        
        # 2. Tuning from pretrained
        vit_b16_pretrained_tuning_flops = calculate_training_flops(
            vit_b16, input_shape, num_images, num_epochs_for_pretrained_vit
        )
        print(f"2. Tuning from pretrained: {vit_b16_pretrained_tuning_flops:,.0f}")
        
        # 3. Pretraining on ImageNet21k
        pretraining_flops = get_imagenet21k_pretraining_flops(vit_b16)
        print(f"3. Pretraining on ImageNet21k: {pretraining_flops:,.0f}")

        # Fig8b: Inference FLOPs
        print("\nFig8b: Inference FLOPs")
        print("-" * 30)
        
        # Calculate inference FLOPs for all models
        vit_b32_inference_flops = get_forward_flops(vit_b32, input_shape)
        print(f"1. vit-b32: {vit_b32_inference_flops:,.0f}")

        vit_b16_inference_flops = get_forward_flops(vit_b16, input_shape)
        print(f"2. vit-b16: {vit_b16_inference_flops:,.0f}")

        vit_l32_inference_flops = get_forward_flops(vit_l32, input_shape)
        print(f"3. vit-l32: {vit_l32_inference_flops:,.0f}")

        vit_l16_inference_flops = get_forward_flops(vit_l16, input_shape)
        print(f"4. vit-l16: {vit_l16_inference_flops:,.0f}")

        vit_h14_inference_flops = get_forward_flops(vit_h14, input_shape)
        print(f"5. vit-h14: {vit_h14_inference_flops:,.0f}")
    
        fig8a_results = pd.DataFrame({
            'flops': [vit_b16_scratch_training_flops, vit_b16_pretrained_tuning_flops, pretraining_flops],
            'model': ['vit-b16_scratch_training_flops', 'vit-b16_pretrained_tuning_flops', 'pretraining_flops']
        })
        fig8a_results.to_csv(DMC_RESULTS_DIR / "fig8a_results.csv", index=False)

        fig8b_results = pd.DataFrame({
            'flops': [vit_b32_inference_flops, vit_b16_inference_flops, vit_l32_inference_flops, vit_l16_inference_flops, vit_h14_inference_flops],
            'model': ['vit-b32_inference_flops', 'vit-b16_inference_flops', 'vit-l32_inference_flops', 'vit-l16_inference_flops', 'vit-h14_inference_flops']
        })
        fig8b_results.to_csv(DMC_RESULTS_DIR / "fig8b_results.csv", index=False)

        print(f"(Saved results to {DMC_RESULTS_DIR / 'fig8a_results.csv'} and {DMC_RESULTS_DIR / 'fig8b_results.csv'})")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
