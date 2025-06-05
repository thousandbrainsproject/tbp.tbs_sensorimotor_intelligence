# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import List, Optional

import torch


def calculate_continual_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_subset: Optional[List[int]] = None,
) -> float:
    """Calculate accuracy between predictions and targets, optionally for a subset of classes.

    This function does not require specifying the number of classes upfront,
    making it suitable for continual learning scenarios where new classes are
    added over time.

    Args:
        predictions (torch.Tensor): Model predictions (class indices) of shape (N,)
        targets (torch.Tensor): Ground truth labels of shape (N,)
        class_subset (Optional[List[int]]): If provided, calculate accuracy only for
                                            samples belonging to these classes

    Returns:
        float: Accuracy score between 0.0 and 1.0

    Raises:
        ValueError: If inputs are empty or have mismatched shapes
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)

    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Input validation
    if predictions.numel() == 0 or targets.numel() == 0:
        raise ValueError("Input tensors cannot be empty")

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} != targets {targets.shape}"
        )

    if predictions.dim() != 1 or targets.dim() != 1:
        raise ValueError("Input tensors must be 1-dimensional")

    # Ensure both are on the same device
    device = predictions.device
    targets = targets.to(device)

    # If class_subset is provided, create a mask for those classes
    if class_subset is not None:
        # Convert class_subset to tensor if it's not already
        if not isinstance(class_subset, torch.Tensor):
            class_subset = torch.tensor(class_subset, device=device)

        # Create mask for samples belonging to the specified classes (vectorized)
        class_mask = torch.isin(targets, class_subset)

        # If no samples match the classes, return 0
        if not class_mask.any():
            return 0.0

        # Filter predictions and targets
        predictions = predictions[class_mask]
        targets = targets[class_mask]

    # Calculate accuracy
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()

    return accuracy
