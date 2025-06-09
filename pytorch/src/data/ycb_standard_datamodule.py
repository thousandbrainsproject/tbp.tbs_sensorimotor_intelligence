# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Optional

import torch
from torch.utils.data import random_split, Subset

from src.data.base_ycb_datamodule import BaseYCBDataModule
from src.data.components.ycb_dataset import YCBDataset
from src.data.transforms.rgbd_normalization import RGBDNormalization


class YCBStandardDataModule(BaseYCBDataModule):
    """Standard YCB DataModule with train/val/test splits and customizable transforms."""

    def __init__(
        self,
        data_dir: str = "view_finder_base/view_finder_rgbd/arrays",
        test_dir: str = "view_finder_randrot/view_finder_rgbd/arrays",
        num_rotations_to_train: Optional[int] = None,
        train_val_split: tuple[float, float] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transform: Any | None = None,
        val_transform: Any | None = None,
    ) -> None:
        """Initialize the YCB Standard DataModule.

        Args:
            data_dir: Path to the training data directory containing RGBD arrays.
                Defaults to "view_finder_base/view_finder_rgbd/arrays".
            test_dir: Path to the test data directory containing RGBD arrays.
                Defaults to "view_finder_randrot/view_finder_rgbd/arrays".
            num_rotations_to_train: Number of rotations to use for training.
                If None, all rotations are used.
            train_val_split: Tuple of (train_ratio, val_ratio) for splitting the dataset.
                Must sum to 1.0. Defaults to (0.8, 0.2).
            batch_size: Number of samples per batch during training.
                Defaults to 64.
            num_workers: Number of subprocesses for data loading.
                0 means data will be loaded in the main process. Defaults to 0.
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
                before returning them. Defaults to False.
            train_transform: Transform to apply to training data. If None,
                RGBDBaseTransform is used. Defaults to None.
            val_transform: Transform to apply to validation and test data. If None,
                RGBDBaseTransform is used. Defaults to None.

        Raises:
            ValueError: If train_val_split values don't sum to 1.0.
        """
        super().__init__(
            data_dir=data_dir,
            test_dir=test_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if sum(train_val_split) != 1.0:
            raise ValueError("train_val_split values must sum to 1.0")

        self.num_rotations_to_train = num_rotations_to_train
        self.train_val_split = train_val_split

        # Set default transforms if not provided
        if train_transform is None:
            train_transform = RGBDNormalization()
        if val_transform is None:
            val_transform = RGBDNormalization()

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = val_transform

    def setup(self, stage: str | None = None) -> None:
        """Load data and set up the training, validation, and test datasets.

        This method handles:
        1. Batch size adjustment for distributed training
        2. Dataset creation and splitting
        3. Transform application

        Args:
            stage: The stage to setup. Either "fit", "validate", "test", or "predict".
                Defaults to None.

        Raises:
            RuntimeError: If batch size is not divisible by the number of devices
                in distributed training.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of "
                    f"devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Create a base dataset without transforms for splitting
            base_dataset = YCBDataset(
                data_dir=self.hparams.data_dir,
                transform=None,  # No transform initially
                num_rotations_to_train=self.hparams.num_rotations_to_train,
            )

            dataset_length = len(base_dataset)
            print(f"Dataset length: {dataset_length}")
            train_length = int(dataset_length * self.train_val_split[0])
            val_length = dataset_length - train_length

            # Split the dataset indices
            train_split, val_split = random_split(
                dataset=base_dataset,
                lengths=[train_length, val_length],
                generator=torch.Generator().manual_seed(42),
            )

            # Create separate datasets with their respective transforms
            train_dataset = YCBDataset(
                data_dir=self.hparams.data_dir,
                transform=self.train_transform,
                num_rotations_to_train=self.num_rotations_to_train,
            )
            # Select train indicdes
            self.data_train = Subset(train_dataset, train_split.indices)

            val_dataset = YCBDataset(
                data_dir=self.hparams.data_dir,
                transform=self.val_transform,
                num_rotations_to_train=self.num_rotations_to_train,
            )
            # Select val indicdes
            self.data_val = Subset(val_dataset, val_split.indices)

            # Create test dataset with test transform
            self.data_test = YCBDataset(
                data_dir=self.hparams.test_dir,
                transform=self.test_transform,
            )
