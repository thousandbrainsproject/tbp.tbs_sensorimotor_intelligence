# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Base Vision Transformer (ViT) Lightning Module implementation. This module provides the core
functionality for training and evaluating ViT models on classification and rotation prediction
tasks. It serves as the foundation for specialized implementations like the continual learning
variant.

Key Features:
- Classification and quaternion prediction
- Standard metrics tracking
- Basic loss computation
- Core training/validation/testing loops
"""

from typing import Any, Dict, Optional, Tuple, TypeAlias

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from lightning import LightningModule
from torchmetrics import Accuracy, MeanMetric

from src.losses.loss import quaternion_geodesic_loss
from src.metrics.rotation_error import get_rotation_error_in_degrees

# Type aliases
BatchDict: TypeAlias = Dict[str, torch.Tensor]
ModelOutput: TypeAlias = Tuple[
    torch.Tensor, torch.Tensor
]  # (pred_class, pred_quaternion)
ModelStepOutput: TypeAlias = Tuple[
    torch.Tensor,  # loss
    torch.Tensor,  # classification_loss
    torch.Tensor,  # quaternion_geodesic_loss
    torch.Tensor,  # pred_class
    torch.Tensor,  # pred_quaternion
    torch.Tensor,  # object_id
    torch.Tensor,  # unit_quaternion
]
MetricsDict: TypeAlias = Dict[str, torchmetrics.Metric]
OptimizerConfig: TypeAlias = Dict[str, Any]
PredictOutput: TypeAlias = Dict[str, torch.Tensor]


class BaseViTLitModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        rotation_weight: float,
        compile: bool,
    ) -> None:
        """Initialize a ViTLitModule.

        Args:
            net: The model to finetune.
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            rotation_weight: Weight for the rotation loss component in the combined loss.
            compile: Whether to compile the model for faster training.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.classification_loss = nn.CrossEntropyLoss()
        self.quaternion_geodesic_loss = quaternion_geodesic_loss

        # Create metric objects
        self.train_metrics = self.create_metrics("train")
        self.val_metrics = self.create_metrics("val")
        self.test_metrics = self.create_metrics("test")

    def create_metrics(self, prefix: str) -> MetricsDict:
        """Create metric objects for tracking model performance.

        Args:
            prefix: Dataset split identifier ('train', 'val', or 'test')

        Returns:
            metrics: Dictionary mapping metric names to metric objects
        """
        metrics = {
            f"{prefix}/loss": MeanMetric(),
            f"{prefix}/classification_loss": MeanMetric(),
            f"{prefix}/quaternion_geodesic_loss": MeanMetric(),
            f"{prefix}/class_acc": Accuracy(task="multiclass", num_classes=77),
            f"{prefix}/rotation_error": MeanMetric(),
        }

        # Register metrics as attributes
        for name, metric in metrics.items():
            setattr(self, name.replace("/", "_"), metric)

        return metrics

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass through the network.

        Args:
            x: Input tensor containing RGBD images

        Returns:
            Tuple containing:
                pred_class: Class prediction logits
                pred_quaternion: Quaternion prediction tensor
        """
        return self.net(x)

    def model_step(self, batch: BatchDict) -> ModelStepOutput:
        """Perform a single model step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_id: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
                - object_name: Ground truth object names of shape (batch_size,)

        Returns:
            Tuple containing:
                - loss: Combined loss tensor of shape (1,)
                - classification_loss: Classification loss component of shape (1,)
                - quaternion_geodesic_loss: Rotation loss component of shape (1,)
                - pred_class: Class prediction logits of shape (batch_size, num_classes)
                - pred_quaternion: Quaternion prediction tensor of shape (batch_size, 4)
                - object_id: Ground truth object class IDs of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
        """
        rgbd_image, object_id, unit_quaternion = (
            batch["rgbd_image"],
            batch["object_id"],
            batch["unit_quaternion"],
        )
        pred_class, pred_quaternion = self.forward(rgbd_image)
        classification_loss = self.classification_loss(pred_class, object_id)
        quaternion_geodesic_loss = self.quaternion_geodesic_loss(
            pred_quaternion, unit_quaternion
        )
        loss = (
            classification_loss
            + self.hparams.rotation_weight * quaternion_geodesic_loss
        )

        return (
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_id,
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
        object_id: torch.Tensor,
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
            object_id: Ground truth object class IDs of shape (batch_size,)
            unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
        """
        metrics = (
            self.train_metrics
            if prefix == "train"
            else self.val_metrics
            if prefix == "val"
            else self.test_metrics
        )

        rotation_errors = get_rotation_error_in_degrees(
            pred_quaternion, unit_quaternion
        )

        # Move metrics to the correct device
        for metric in metrics.values():
            metric.to(self.device)

        # Update metrics with current batch values
        metrics[f"{prefix}/loss"].update(loss)
        metrics[f"{prefix}/classification_loss"].update(classification_loss)
        metrics[f"{prefix}/quaternion_geodesic_loss"].update(quaternion_geodesic_loss)
        metrics[f"{prefix}/class_acc"].update(pred_class, object_id)
        metrics[f"{prefix}/rotation_error"].update(rotation_errors)

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch: BatchDict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_id: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
                - object_name: Ground truth object names of shape (batch_size,)
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
            object_id,
            unit_quaternion,
        ) = self.model_step(batch)
        self.log_metrics(
            "train",
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_id,
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

    def validation_step(self, batch: BatchDict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_id: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
                - object_name: Ground truth object names of shape (batch_size,)
            batch_idx: The index of the current batch.
        """
        (
            val_loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_id,
            unit_quaternion,
        ) = self.model_step(batch)
        self.log_metrics(
            "val",
            val_loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_id,
            unit_quaternion,
        )

    def on_validation_epoch_end(self) -> None:
        """Reset validation metrics at the end of each validation epoch.

        This method is called automatically by PyTorch Lightning after each validation epoch to
        ensure metrics are properly reset for the next epoch.
        """
        for metric in self.val_metrics.values():
            metric.reset()

    def test_step(self, batch: BatchDict, batch_idx: int) -> None:
        """Perform a single test step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_id: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
                - object_name: Ground truth object names of shape (batch_size,)
            batch_idx: The index of the current batch.
        """
        (
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_id,
            unit_quaternion,
        ) = self.model_step(batch)
        self.log_metrics(
            "test",
            loss,
            classification_loss,
            quaternion_geodesic_loss,
            pred_class,
            pred_quaternion,
            object_id,
            unit_quaternion,
        )

    def on_test_epoch_end(self) -> None:
        """Reset test metrics at the end of each test epoch.

        This method is called automatically by PyTorch Lightning after each test epoch to ensure
        metrics are properly reset for the next epoch.
        """
        for metric in self.test_metrics.values():
            metric.reset()

    def predict_step(self, batch: BatchDict, batch_idx: int) -> PredictOutput:
        """Perform a single prediction step on a batch of data.

        Args:
            batch: A dictionary containing:
                - rgbd_image: Input tensor of shape (batch_size, channels, height, width)
                - object_id: Ground truth class labels of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
                - object_name: Ground truth object names of shape (batch_size,)
            batch_idx: The index of the current batch.

        Returns:
            Dictionary containing:
                - class_probabilities: Softmax probabilities for object classes of shape
                    (batch_size, num_classes)
                - predicted_quaternion: Predicted rotation quaternions of shape (batch_size, 4)
                - object_id: Ground truth object class IDs of shape (batch_size,)
                - unit_quaternion: Ground truth rotation quaternions of shape (batch_size, 4)
        """
        rgbd_image, object_id, unit_quaternion = (
            batch["rgbd_image"],
            batch["object_id"],
            batch["unit_quaternion"],
        )
        pred_class, pred_quaternion = self.forward(rgbd_image)

        # Convert logits to probabilities using softmax
        class_probabilities = torch.softmax(pred_class, dim=1)

        return {
            "class_probabilities": class_probabilities,
            "predicted_quaternion": pred_quaternion,
            "object_id": object_id,
            "unit_quaternion": unit_quaternion,
        }

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
            # total_steps = self.trainer.estimated_stepping_batches
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
