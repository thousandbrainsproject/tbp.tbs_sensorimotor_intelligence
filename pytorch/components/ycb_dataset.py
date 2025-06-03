# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from src.data.transforms.rgbd_normalization import RGBDNormalization


class EpisodeData(TypedDict):
    """Type definition for episode data structure.

    Attributes:
        episode: Episode number.
        object: Name of the object.
        rotation: List of rotation angles in degrees [x, y, z].
    """

    episode: int
    object: str
    rotation: list[float]


class YCBDataset(Dataset):
    """YCB Dataset for ViewFinder images with rotation information.

    This dataset loads RGBD images and their corresponding object and rotation information
    from a directory structure. It supports optional transformations and can limit the
    number of rotations used for training.

    Attributes:
        data_dir: Path to the directory containing the data.
        image_dir: Path to the directory containing the image arrays.
        num_rotations_to_train: Optional limit on number of rotations to use.
        episodes: List of episode data containing object and rotation information.
        unique_object_names: List of unique object names in the dataset.
        label_encoder: Encoder for converting object names to numeric IDs.
        transform: Optional transform to apply to the images.

    Class Attributes:
        NUM_YCB_OBJECTS: Number of objects in the YCB dataset.
    """

    # YCB dataset contains 77 unique objects
    NUM_YCB_OBJECTS: int = 77

    def __init__(
        self,
        data_dir: str | Path,
        transform: RGBDNormalization | None = None,
        num_rotations_to_train: int | None = None,
    ) -> None:
        """Initialize the YCB Dataset.

        Args:
            data_dir: Path to the directory containing the data.
                It should have a subdirectory called 'arrays' and 'episodes.jsonl' file.
            transform: Transform to apply to the images.
                Defaults to None.
            num_rotations_to_train: If set, limits the dataset to use N rotations.
                Defaults to None (use all data).

        Raises:
            FileNotFoundError: If data_dir or required files don't exist.
            ValueError: If episodes.jsonl is empty or malformed.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        self.image_dir = self.data_dir / "arrays"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist")

        episodes_file = self.data_dir / "episodes.jsonl"
        if not episodes_file.exists():
            raise FileNotFoundError(f"Episodes file {episodes_file} does not exist")

        self.num_rotations_to_train = num_rotations_to_train

        with open(episodes_file) as f:
            all_episodes: list[EpisodeData] = [json.loads(line) for line in f]
            if not all_episodes:
                raise ValueError("Episodes file is empty")

            if num_rotations_to_train is not None:
                self.episodes = all_episodes[
                    : self.NUM_YCB_OBJECTS * num_rotations_to_train
                ]
            else:
                self.episodes = all_episodes

        self.unique_object_names = self._get_unique_object_names()
        if len(self.unique_object_names) != self.NUM_YCB_OBJECTS:
            raise ValueError(
                f"Expected {self.NUM_YCB_OBJECTS} unique objects, \
                but found {len(self.unique_object_names)}"
            )

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.unique_object_names)
        self.transform = transform

    @property
    def object_classes(self) -> list[str]:
        """Get list of unique object classes in the dataset.

        Returns:
            List of object class names.
        """
        return self.unique_object_names

    def get_label_encoder(self) -> LabelEncoder:
        """Get the label encoder used for object name to ID conversion.

        Returns:
            The fitted label encoder instance.
        """
        return self.label_encoder

    def get_object_ids(self) -> list[int]:
        """Get all object IDs in the dataset.

        Returns:
            List of object IDs for all items in the dataset.
        """
        return [self.extract_object_id(idx) for idx in range(len(self))]

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        if self.num_rotations_to_train is not None:
            return self.NUM_YCB_OBJECTS * self.num_rotations_to_train
        return len(list(self.image_dir.glob("*.npy")))

    def _get_unique_object_names(self) -> list[str]:
        """Get unique object names from episodes data.

        Returns:
            Sorted list of unique object names.
        """
        return sorted(list({episode["object"] for episode in self.episodes}))

    def extract_object_id(self, idx: int) -> int:
        """Extract object ID from episodes data.

        Args:
            idx: Index of the episode.

        Returns:
            Numeric ID of the object.
        """
        object_name = self.episodes[idx]["object"]
        return int(self.label_encoder.transform([object_name])[0])

    def extract_rotation(self, idx: int) -> torch.Tensor:
        """Extract rotation angles from episodes data.

        Args:
            idx: Index of the episode.

        Returns:
            Tensor containing rotation angles in degrees [x, y, z].
        """
        rotation = self.episodes[idx]["rotation"]
        return torch.tensor(rotation, dtype=torch.float32)

    def normalize_rotation_to_unit_quaternion(
        self, rotation: torch.Tensor
    ) -> torch.Tensor:
        """Convert rotation angles to unit quaternion representation.

        Args:
            rotation: Tensor containing rotation angles in degrees [x, y, z].

        Returns:
            Unit quaternion representation [x, y, z, w].
        """
        rotation_rad = np.radians(rotation.numpy())
        r = R.from_euler("xyz", rotation_rad, degrees=False)
        quaternion = r.as_quat()
        return torch.tensor(quaternion, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a data sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - rgbd_image: RGBD image tensor with shape (C, H, W)
                - object_id: Integer tensor representing the object class
                - unit_quaternion: Unit quaternion representing the object's rotation
                - object_name: String name of the object

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image data is corrupted or invalid.
        """
        image_path = self.image_dir / f"{idx}.npy"
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist")

        try:
            rgbd_image = np.load(image_path)
            rgbd_image = torch.tensor(rgbd_image, dtype=torch.float32)
            rgbd_image = rgbd_image.permute(2, 0, 1)
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"Failed to load image at {image_path}: {str(e)}")

        if self.transform is not None:
            rgbd_image = self.transform(rgbd_image)

        object_id = self.extract_object_id(idx)
        object_id_tensor = torch.tensor(object_id, dtype=torch.int64)
        euler_rotation = self.extract_rotation(idx)
        unit_quaternion = self.normalize_rotation_to_unit_quaternion(euler_rotation)
        object_name = self.label_encoder.inverse_transform([object_id])[0]

        return {
            "rgbd_image": rgbd_image,
            "object_id": object_id_tensor,
            "unit_quaternion": unit_quaternion,
            "object_name": object_name,
        }
