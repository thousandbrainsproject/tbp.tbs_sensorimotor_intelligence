# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Constants
EPSILON = 1e-8
MAX_ROTATION_ERROR = 180.0


class InvalidQuaternionError(Exception):
    """Raised when a quaternion is invalid."""

    pass


def get_rotation_error_in_degrees(
    predicted_quaternion: torch.Tensor,
    target_quaternion: torch.Tensor,
    reduce: bool = False,
) -> torch.Tensor:
    """Get the rotation error between two quaternions.

    Assumes quaternions are in xyzw (scalar-last) format.

    Args:
        predicted_quaternion (torch.Tensor): Predicted quaternion (batch_size, 4) or (4,) in xyzw format.
        target_quaternion (torch.Tensor): Target quaternion (batch_size, 4) or (4,) in xyzw format.
        reduce (bool, optional): If True, returns the mean error. If False,
            returns individual errors. Defaults to False.

    Returns:
        torch.Tensor: Rotation error for each sample in the batch or mean error if reduce=True.

    Raises:
        InvalidQuaternionError: If quaternions have invalid shape or zero norm.
    """
    # Input validation
    if not isinstance(predicted_quaternion, torch.Tensor) or not isinstance(
        target_quaternion, torch.Tensor
    ):
        raise TypeError("Inputs must be torch.Tensor")

    if predicted_quaternion.shape[-1] != 4 or target_quaternion.shape[-1] != 4:
        raise InvalidQuaternionError("Quaternions must have 4 components")

    if predicted_quaternion.shape != target_quaternion.shape:
        raise InvalidQuaternionError(
            "Predicted and target quaternions must have the same shape"
        )

    # Convert to numpy
    predicted_quaternion = predicted_quaternion.detach().cpu().numpy()
    target_quaternion = target_quaternion.detach().cpu().numpy()

    # Handle single quaternion case
    if predicted_quaternion.ndim == 1:
        predicted_quaternion = predicted_quaternion.reshape(1, -1)
        target_quaternion = target_quaternion.reshape(1, -1)

    # Check for zero norm quaternions
    pred_norms = np.linalg.norm(predicted_quaternion, axis=1)
    target_norms = np.linalg.norm(target_quaternion, axis=1)

    # Initialize result array with maximum error for zero norm cases
    rotation_error_degrees = np.full(predicted_quaternion.shape[0], MAX_ROTATION_ERROR)

    # Process only valid quaternions (non-zero norm)
    valid_indices = (pred_norms > EPSILON) & (target_norms > EPSILON)

    if np.any(valid_indices):
        # Convert to Rotation object only for valid quaternions
        predicted_rotation = R.from_quat(predicted_quaternion[valid_indices])
        target_rotation = R.from_quat(target_quaternion[valid_indices])

        # Compute rotation error (geodesic distance)
        rotation_vector = predicted_rotation.inv() * target_rotation

        # Convert to rotation error in radians and degrees
        rotation_error_radians = rotation_vector.magnitude()
        rotation_error_degrees[valid_indices] = np.degrees(rotation_error_radians)

    # Convert back to torch tensor
    rotation_error_degrees = torch.from_numpy(rotation_error_degrees)

    # Return single value if input was a single quaternion
    if predicted_quaternion.shape[0] == 1:
        return rotation_error_degrees[0].reshape(1)

    # Return mean if reduce=True, otherwise return individual errors
    if reduce:
        return rotation_error_degrees.mean().reshape(1)

    return rotation_error_degrees
