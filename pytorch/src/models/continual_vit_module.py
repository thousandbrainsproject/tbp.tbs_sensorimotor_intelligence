# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# ---
# MIT License
#
# Copyright (c) 2021 ashleve
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Continual Learning extension of the base Vision Transformer (ViT) Lightning Module. This module
extends the base ViT implementation to support continual learning scenarios where the model needs
to learn new tasks while maintaining performance on previously learned tasks.

Key Features:
- Task-specific class ranges management
- Metrics tracking for current and all seen classes
- Masked cross-entropy loss for continual learning
- Task ID management
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanMetric

from src.losses.loss import masked_cross_entropy_loss, quaternion_geodesic_loss
from src.metrics.continual_accuracy import calculate_continual_accuracy
from src.models.base_vit_module import (
    BaseViTLitModule,
    BatchDict,
    MetricsDict,
    ModelStepOutput,
    OptimizerConfig,
    PredictOutput,
)
from src.utils.continual_learning_utils import compute_class_ranges


class ContinualViTLitModule(BaseViTLitModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        compile: bool,
        rotation_weight: float,
        task_id: int,
        num_classes_for_task: Optional[int] = None,
    ) -> None:
        """Initialize a ContinualViTLitModule.

        Args:
            net: The model to finetune.
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            compile: Whether to compile the model for faster training.
            rotation_weight: Weight for the rotation loss component in the combined loss.
            task_id: Current task identifier for continual learning.
            num_classes_for_task: Number of classes in the current task. If None, uses default
                class ranges.
        """
        super().__init__(net, optimizer, scheduler, compile, rotation_weight)
        self.task_id = task_id

        # Update classification loss function from BaseViTLitModule
        self.classification_loss = masked_cross_entropy_loss
        self.quaternion_geodesic_loss = quaternion_geodesic_loss

        # Create additional metrics specific to continual learning
        self.train_metrics.update(self.create_continual_metrics("train"))
        self.val_metrics.update(self.create_continual_metrics("val"))
        self.test_metrics.update(self.create_continual_metrics("test"))

        _, _, self.current_task_classes, self.all_seen_classes = compute_class_ranges(
            num_classes_for_task, task_id
        )

    def create_continual_metrics(self, prefix: str) -> MetricsDict:
        """Create additional metrics specific to continual learning.

        Args:
            prefix: Dataset split identifier ('train', 'val', or 'test')

        Returns:
            Dictionary mapping metric names to metric objects for continual learning metrics.
        """
        metrics = {
            f"{prefix}/all_seen_class_acc": MeanMetric(),
            f"{prefix}/current_task_class_acc": MeanMetric(),
        }

        for name, metric in metrics.items():
            setattr(self, name.replace("/", "_"), metric)

        return metrics

    def model_step(self, batch: BatchDict) -> ModelStepOutput:
        """Perform a single model step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_ids: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)

        Returns:
            Tuple containing:
                - loss: Combined loss tensor of shape (1,)
                - classification_loss: Classification loss component of shape (1,)
                - quaternion_geodesic_loss: Rotation loss component of shape (1,)
                - pred_class: Class prediction logits of shape (batch_size, num_classes)
                - pred_quaternion: Quaternion prediction tensor of shape (batch_size, 4)
                - object_ids: Ground truth object class IDs of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
        """
        rgbd_image, object_ids, unit_quaternion = (
            batch["rgbd_image"],
            batch["object_ids"],
            batch["unit_quaternion"],
        )
        pred_class, pred_quaternion = self.forward(rgbd_image)

        # Always use all_seen_classes for valid classes
        # This is a form of forward masking that ensures the model does not try to predict
        # classes it has not seen yet. By restricting predictions to only seen classes,
        # we prevent the model from making predictions on future/unseen classes during training.
        valid_classes = self.all_seen_classes

        classification_loss = self.classification_loss(pred_class, object_ids, valid_classes)
        quaternion_geodesic_loss = self.quaternion_geodesic_loss(pred_quaternion, unit_quaternion)
        loss = classification_loss + self.hparams.rotation_weight * quaternion_geodesic_loss

        return (
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_ids,
            unit_quaternion,
        )

    def log_metrics(
        self,
        prefix: str,
        loss: torch.Tensor,
        classification_loss: torch.Tensor,
        quaternion_geodesic_loss: torch.Tensor,
        pred_class: torch.Tensor,
        pred_quaternion: torch.Tensor,
        object_ids: torch.Tensor,
        unit_quaternion: torch.Tensor,
    ) -> None:
        """Log metrics for a given prefix (train, val, test).

        Args:
            prefix: Dataset split identifier ('train', 'val', or 'test')
            loss: Combined loss tensor of shape (1,)
            classification_loss: Classification loss component of shape (1,)
            quaternion_geodesic_loss: Quaternion loss component of shape (1,)
            pred_class: Class prediction logits of shape (batch_size, num_classes)
            pred_quaternion: Quaternion prediction tensor of shape (batch_size, 4)
            object_ids: Ground truth object class IDs of shape (batch_size,)
            unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
        """
        # First call parent class log_metrics
        super().log_metrics(
            prefix,
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_ids,
            unit_quaternion,
        )

        metrics = (
            self.train_metrics
            if prefix == "train"
            else self.val_metrics
            if prefix == "val"
            else self.test_metrics
        )

        # Get predicted class indices
        pred_indices = torch.argmax(pred_class, dim=1)

        # Calculate accuracy for current task classes
        current_task_acc = calculate_continual_accuracy(
            pred_indices, object_ids, self.current_task_classes
        )
        metrics[f"{prefix}/current_task_class_acc"].update(current_task_acc)

        # Calculate accuracy for all seen classes
        all_seen_acc = calculate_continual_accuracy(
            pred_indices, object_ids, self.all_seen_classes
        )
        metrics[f"{prefix}/all_seen_class_acc"].update(all_seen_acc)

        # Log additional metrics
        self.log_dict(
            {
                f"{prefix}/current_task_class_acc": metrics[
                    f"{prefix}/current_task_class_acc"
                ].compute(),
                f"{prefix}/all_seen_class_acc": metrics[f"{prefix}/all_seen_class_acc"].compute(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch: BatchDict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_ids: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
            batch_idx: The index of the current batch.

        Returns:
            Combined loss tensor for backpropagation.
        """
        (
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_ids,
            unit_quaternion,
        ) = self.model_step(batch)
        self.log_metrics(
            "train",
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_ids,
            unit_quaternion,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Reset training metrics at the end of each training epoch.

        This method is called automatically by PyTorch Lightning after each training epoch to
        ensure metrics are properly reset for the next epoch.
        """
        for metric in self.train_metrics.values():
            metric.reset()

    def test_step(self, batch: BatchDict, batch_idx: int) -> None:
        """Perform a single test step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_ids: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
            batch_idx: The index of the current batch.
        """
        (
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_ids,
            unit_quaternion,
        ) = self.model_step(batch)
        self.log_metrics(
            "test",
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_ids,
            unit_quaternion,
        )

    def on_test_epoch_end(self) -> None:
        """Reset test metrics at the end of each test epoch.

        This method is called automatically by PyTorch Lightning after each test epoch to ensure
        metrics are properly reset for the next epoch.
        """
        for metric in self.test_metrics.values():
            metric.reset()

    def setup(self, stage: Optional[str]) -> None:
        """Set up the model for the specified stage.

        Args:
            stage: The current stage ('fit', 'validate', 'test', or 'predict').
                  Can be None during initialization.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure optimizers and learning rate schedulers.

        This method sets up the optimization algorithm and learning rate scheduler
        for training the model.

        Returns:
            Dict containing optimizer configuration and optional lr_scheduler configuration.
            The dictionary has the following structure:
            - With scheduler: {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
            - Without scheduler: {'optimizer': optimizer}
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            total_steps = 1400  # 200 epochs * 7 steps/epoch when we did hparam search
            warmup_steps = int(0.05 * total_steps)  # 5% of total steps
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return {"optimizer": optimizer}

    def predict_step(
        self,
        batch: BatchDict,
        batch_idx: int,
    ) -> PredictOutput:
        """Perform a single prediction step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_ids: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
            batch_idx: The index of the current batch.

        Returns:
            Dictionary containing:
                - class_probabilities: Softmax probabilities for object classes of shape
                    (batch_size, num_classes)
                - predicted_quaternion: Predicted rotation quaternions of shape (batch_size, 4)
                - object_ids: Ground truth object class IDs of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
        """
        # Call model_step without mode parameter
        _, _, _, pred_class, pred_quaternion, object_ids, unit_quaternion = self.model_step(batch)

        class_probabilities = torch.softmax(pred_class, dim=1)

        return {
            "class_probabilities": class_probabilities,
            "predicted_quaternion": pred_quaternion,
            "object_ids": object_ids,
            "unit_quaternion": unit_quaternion,
        }
