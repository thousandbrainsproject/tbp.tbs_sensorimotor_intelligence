# Copyright 2025 Thousand Brains Project

#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import torch
import torch.nn.functional as F


def quaternion_geodesic_loss(
    pred_quaternion: torch.Tensor,
    target_quaternion: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the geodesic distance between two quaternions.

    Args:
        pred_quaternion (torch.Tensor): Predicted quaternion of shape (N, 4)
        target_quaternion (torch.Tensor): Target quaternion of shape (N, 4)

    Returns:
        torch.Tensor: Geodesic loss between the quaternions
    """
    # Verify input shapes
    assert pred_quaternion.shape == target_quaternion.shape, (
        f"Shape mismatch: pred {pred_quaternion.shape} vs target {target_quaternion.shape}"
    )
    assert pred_quaternion.shape[-1] == 4, (
        f"Expected last dimension to be 4, got {pred_quaternion.shape[-1]}"
    )

    # Normalize the quaternions
    pred_norm = F.normalize(pred_quaternion, p=2, dim=-1)
    target_norm = F.normalize(target_quaternion, p=2, dim=-1)

    # Verify normalization
    pred_norms = torch.norm(pred_norm, p=2, dim=-1)
    target_norms = torch.norm(target_norm, p=2, dim=-1)
    assert torch.allclose(pred_norms, torch.ones_like(pred_norms), atol=1e-6), (
        "Pred quaternions not normalized"
    )
    assert torch.allclose(target_norms, torch.ones_like(target_norms), atol=1e-6), (
        "Target quaternions not normalized"
    )

    # Compute the dot product
    dot_product = torch.sum(pred_norm * target_norm, dim=-1)

    # Clamp the dot product to avoid numerical instability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the geodesic distance (angle)
    # Use absolute value to ensure we take the shortest path
    # This ensures the angle is always <= 180 degrees
    angle = 2 * torch.acos(torch.abs(dot_product))

    # Verify angle range
    assert torch.all(angle >= 0) and torch.all(angle <= torch.pi), (
        f"Angle out of range [0, π]: min={angle.min()}, max={angle.max()}"
    )

    # Return the mean loss for batch
    if reduction == "mean":
        return torch.mean(angle)
    elif reduction is None:
        return angle
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def masked_cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid_class_indices: torch.Tensor
) -> torch.Tensor:
    """Compute masked cross-entropy loss for continual learning.

    This function calculates the cross-entropy loss only for valid classes specified by
    valid_class_indices. It masks out invalid classes by setting their logits to a very
    negative value before applying softmax. This is a global mask, i.e. there is no
    granularity in the form of different masks for different members of the batch.

    Args:
        logits (torch.Tensor): Predicted logits of shape (N, C) where C is the total number of
            classes
        targets (torch.Tensor): Target class indices of shape (N)
        valid_class_indices (torch.Tensor): Tensor of valid class indices to include in loss
            calculation

    Returns:
        torch.Tensor: Masked cross-entropy loss
    """
    # Verify input shapes
    assert len(logits.shape) == 2, f"Expected logits to be 2D, got shape {logits.shape}"
    assert len(targets.shape) == 1, (
        f"Expected targets to be 1D, got shape {targets.shape}"
    )
    assert len(targets) == logits.shape[0], (
        f"Batch size mismatch: logits {logits.shape[0]} vs targets {len(targets)}"
    )

    # Create a mask for valid classes
    num_classes = logits.shape[1]
    mask = torch.zeros(num_classes, dtype=torch.bool, device=logits.device)
    mask[valid_class_indices] = True

    # Create masked logits by setting invalid class logits to a large negative value
    masked_logits = logits.clone()
    masked_logits[:, ~mask] = -1e10

    # Compute cross-entropy loss with the masked logits
    loss = F.cross_entropy(masked_logits, targets)

    return loss
