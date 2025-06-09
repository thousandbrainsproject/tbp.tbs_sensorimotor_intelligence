# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.transforms.rgbd_normalization import RGBDNormalization


class BaseYCBDataModule(LightningDataModule):
    """Base YCB DataModule for handling RGBD data.

    This module provides the base functionality for loading and processing YCB RGBD data.
    It handles data loading, batch size management for distributed training, and basic
    transforms. Specific implementations (like standard and continual learning variants)
    inherit from this base class.

    Attributes:
        data_train: Training dataset.
        data_val: Validation dataset.
        data_test: Test dataset.
        batch_size_per_device: Batch size adjusted for distributed training.
        transform: Base transform applied to RGBD data.
    """

    def __init__(
        self,
        data_dir: str = "view_finder_base/view_finder_rgbd/arrays",
        test_dir: str = "view_finder_randrot/view_finder_rgbd/arrays",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the base YCB DataModule.

        Args:
            data_dir: Path to the training data directory containing RGBD arrays.
                Defaults to "view_finder_base/view_finder_rgbd/arrays".
            test_dir: Path to the test data directory containing RGBD arrays.
                Defaults to "view_finder_randrot/view_finder_rgbd/arrays".
            batch_size: Number of samples per batch during training.
                Defaults to 64.
            num_workers: Number of subprocesses for data loading.
                Defaults to 0.
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
                before returning them. Defaults to False.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.transform = RGBDNormalization()

    def prepare_data(self) -> None:
        """Verify data directory exists.

        Raises:
            AssertionError: If the data directory specified in self.hparams.data_dir does not exist
        """
        assert os.path.exists(self.hparams.data_dir), (
            "Data directory does not exist. Please set up the data directory from README.md."
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            DataLoader: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            DataLoader: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data module for the specified stage. This method should be implemented by
        child classes.

        Args:
            stage: The stage to setup. Either "fit", "validate", "test", or "predict".
                Defaults to None.
        """
        raise NotImplementedError("setup() must be implemented by child classes")
