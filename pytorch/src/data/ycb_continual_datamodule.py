# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""YCB dataset module for continual learning scenarios with task-based class filtering."""

from collections.abc import Sequence

import numpy as np
from torch.utils.data import Dataset, Subset

from src.data.base_ycb_datamodule import BaseYCBDataModule
from src.data.components.ycb_dataset import YCBDataset
from src.utils.continual_learning_utils import compute_class_ranges


class YCBContinualDataModule(BaseYCBDataModule):
    """YCB DataModule for continual learning with task-based class filtering.

    This module extends the base YCB DataModule to support continual learning scenarios where the
    model learns tasks sequentially. Each task consists of a subset of classes, and the module
    manages class filtering and tracking of seen classes.
    """

    def __init__(
        self,
        data_dir: str = "view_finder_base/view_finder_rgbd/arrays",
        test_dir: str = "view_finder_randrot/view_finder_rgbd/arrays",
        task_id: int = 0,
        num_classes_for_task: int | Sequence[int] = 77,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the YCB Continual DataModule.

        Args:
            data_dir: Path to the training data directory containing RGBD arrays.
                Defaults to "view_finder_base/view_finder_rgbd/arrays".
            test_dir: Path to the test data directory containing RGBD arrays.
                Defaults to "view_finder_randrot/view_finder_rgbd/arrays".
            task_id: The ID of the current task. Used to determine which classes
                to include in the current task. Defaults to 0.
            num_classes_for_task: The number of classes per task or a sequence specifying
                the number of classes for each task. Defaults to 77.
            batch_size: Number of samples per batch during training.
                Defaults to 64.
            num_workers: Number of subprocesses for data loading.
                0 means data will be loaded in the main process. Defaults to 0.
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
                before returning them. Defaults to False.
        """
        super().__init__(
            data_dir=data_dir,
            test_dir=test_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.task_id = task_id
        _, _, self.current_task_classes, self.all_seen_classes = compute_class_ranges(
            num_classes_for_task, task_id
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up the data module for the specified stage.

        This method handles:
        1. Batch size adjustment for distributed training
        2. Dataset creation with task-specific class filtering
        3. Setting up train, validation, and test datasets

        Args:
            stage: The stage to setup. Either "fit", "validate", "test", or "predict".
                Defaults to None.

        Raises:
            RuntimeError: If batch size is not divisible by the number of devices
                in distributed training.
        """
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load full dataset
        self.full_dataset = YCBDataset(
            data_dir=self.hparams.data_dir,
            transform=self.transform,
            num_rotations_for_train=14,
        )

        # Filter only the training dataset by class, keeping data_test unfiltered
        # to enable visualization and analysis of model performance across all classes
        self.data_train = self.filter_dataset_by_class(
            self.full_dataset, self.current_task_classes
        )
        self.data_test = YCBDataset(
            data_dir=self.hparams.test_dir,
            transform=self.transform,
        )
        self.data_val = self.data_test  # Use test set for validation

    def filter_dataset_by_class(self, dataset: Dataset, class_list: Sequence[int]) -> Dataset:
        """Filter a dataset to only include samples from specified classes.

        Args:
            dataset: The dataset to filter.
            class_list: List or sequence of class IDs to include.

        Returns:
            A filtered subset of the original dataset containing only samples
            from the specified classes.
        """
        targets = np.array(dataset.get_object_ids())
        mask = np.isin(targets, class_list)
        indices = np.where(mask)[0]
        return Subset(dataset, indices)

    def get_current_task_class_names(self) -> list[str]:
        """Return the list of object names for the current task classes.

        Returns:
            A list of object names corresponding to the classes in the current task.

        Note:
            This method will call prepare_data and setup if the dataset hasn't been
            initialized yet.
        """
        if not hasattr(self, "full_dataset"):
            self.prepare_data()
            self.setup()

        label_encoder = self.full_dataset.get_label_encoder()
        return label_encoder.inverse_transform(self.current_task_classes).tolist()

    def get_all_seen_class_names(self) -> list[str]:
        """Return the list of object names for all classes seen so far.

        Returns:
            A list of object names corresponding to all classes seen across all
            tasks up to the current task.

        Note:
            This method will call prepare_data and setup if the dataset hasn't been
            initialized yet.
        """
        if not hasattr(self, "full_dataset"):
            self.prepare_data()
            self.setup()

        label_encoder = self.full_dataset.get_label_encoder()
        return label_encoder.inverse_transform(self.all_seen_classes).tolist()
