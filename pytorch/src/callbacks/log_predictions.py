# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module implements a callback for logging model predictions to Weights & Biases.

A Callback in PyTorch Lightning is a self-contained set of methods that are called at specific
points during the training loop, allowing for customization without modifying the core training
code.
"""

from typing import Any

import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import Dataset, Subset
from torchvision.utils import make_grid

from src.metrics.rotation_error import get_rotation_error_in_degrees


class LogPredictionsCallback(Callback):
    """A PyTorch Lightning callback that logs model predictions and performance metrics to W&B.

    This callback captures model predictions during validation and testing phases, logging:
    - Input RGB images
    - Object type predictions vs ground truth
    - Quaternion predictions vs ground truth
    - Rotation errors in both radians and degrees

    The logged data is organized in a W&B Table for easy visualization and analysis.
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()

    def _get_dataset(self, trainer: Trainer, stage: str) -> Dataset:
        """Get the appropriate dataset based on the stage.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            stage: The stage of training ('val' or 'test').

        Returns:
            The dataset corresponding to the specified stage.
        """
        dataset = (
            trainer.datamodule.data_val
            if stage == "val"
            else trainer.datamodule.data_test
        )
        return dataset

    def _get_label_encoder(self, dataset: Dataset) -> Any:
        """Extract the label encoder from the dataset.

        Args:
            dataset: The dataset to extract the label encoder from.

        Returns:
            The label encoder object.
        """
        if isinstance(dataset, Subset):
            return dataset.dataset.get_label_encoder()
        return dataset.get_label_encoder()

    def _log_predictions(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Log model predictions and metrics to Weights & Biases.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model).
            stage: The stage of training ('val' or 'test').
        """
        device = pl_module.device
        dataloader = (
            trainer.datamodule.val_dataloader()
            if stage == "val"
            else trainer.datamodule.test_dataloader()
        )

        dataset = self._get_dataset(trainer, stage)
        label_encoder = self._get_label_encoder(dataset)

        # Create a wandb Table
        columns = [
            "Epoch",
            "Object Image",
            "Object Name",
            "Predicted Object Type",
            "Ground Truth Quaternion",
            "Predicted Quaternion",
            "Rotation Error (Radians)",
            "Rotation Error (Degrees)",
        ]
        table = wandb.Table(columns=columns)

        for batch in dataloader:
            rgbd_images, object_ids, unit_quaternions = (
                batch["rgbd_image"],
                batch["object_ids"],
                batch["unit_quaternion"],
            )

            # Get model predictions
            with torch.no_grad():
                pred_class, pred_unit_quaternions = pl_module(rgbd_images.to(device))
                pred_class = pred_class.argmax(dim=1)

            # Get object names
            pred_object_names = [
                str(label_encoder.inverse_transform([pred_class[i].item()])[0])
                for i in range(len(pred_class))
            ]
            gt_object_names = [
                str(label_encoder.inverse_transform([object_ids[i].item()])[0])
                for i in range(len(object_ids))
            ]

            # Add rows to the table
            for i in range(len(rgbd_images)):
                image = rgbd_images[i].cpu()  # (4, 224, 224)
                rgb_image = image[:3, :, :]
                gt_object_name = gt_object_names[i]
                gt_quaternion = unit_quaternions[i]
                pred_object_name = pred_object_names[i]
                pred_quaternion = pred_unit_quaternions[i]
                rotation_error_degrees = get_rotation_error_in_degrees(
                    pred_quaternion, gt_quaternion
                )
                rotation_error_radians = rotation_error_degrees * np.pi / 180

                # Convert image tensor to wandb Image
                image_wandb = wandb.Image(
                    make_grid(rgb_image, normalize=True, scale_each=True)
                )

                # Add row to the table
                table.add_data(
                    trainer.current_epoch,
                    image_wandb,
                    gt_object_name,
                    pred_object_name,
                    str(gt_quaternion),
                    str(pred_quaternion),
                    str(rotation_error_radians),
                    str(rotation_error_degrees),
                )

        # Log the table to wandb
        trainer.logger.experiment.log({f"{stage}_predictions_table": table})

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation phase ends.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model).
        """
        # Log predictions only at the last epoch
        if trainer.current_epoch == (trainer.max_epochs - 1):
            self._log_predictions(trainer, pl_module, "val")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test phase ends.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model).
        """
        self._log_predictions(trainer, pl_module, "test")
