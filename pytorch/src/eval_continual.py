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


import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.losses.loss import quaternion_geodesic_loss
from src.utils import RankedLogger, extras, task_wrapper
from src.utils.lightning_utils import (
    setup_datamodule,
    setup_model,
    setup_trainer,
)
from src.utils.eval_utils import (
    generate_predictions,
    analyze_predictions,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Type aliases for better readability
PredictionBatch = dict[str, torch.Tensor]
MetricsDict = dict[str, Any]


def get_checkpoint_path(cfg: DictConfig, model_id: int) -> Path:
    """Get the checkpoint path for a specific model ID.

    Args:
        cfg: Configuration containing base directory information.
        model_id: ID of the model to get checkpoint for.

    Returns:
        Path to the model checkpoint.
    """
    exp_name = f"fig7b_vit-b16-224-in21k_task{model_id}_classes1"
    return Path(cfg.base_dir) / exp_name / "checkpoints/last.ckpt"


def evaluate_model(cfg: DictConfig, model_id: int, ckpt_path: Path | str) -> None:
    """Evaluate a specific model checkpoint on the test dataset.

    This function handles the evaluation of a single model in the continual learning
    sequence, including data preparation, model loading, and prediction analysis.

    Args:
        cfg: Configuration composed by Hydra.
        model_id: The model ID being evaluated.
        ckpt_path: Path to the checkpoint file.

    Raises:
        AssertionError: If save directory is not provided.
    """
    _validate_config(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Update configuration for the specific task being evaluated
    cfg.data.task_id = model_id
    cfg.model.task_id = model_id
    log.info(f"Evaluating task_id={model_id} with checkpoint: {ckpt_path}")

    # Initialize and validate components
    datamodule, model = _setup_components(cfg)
    _validate_classes(datamodule, model)

    # Setup evaluation
    trainer = setup_trainer(cfg)
    predictions = generate_predictions(trainer, model, datamodule, ckpt_path)
    masker = _get_masked_predictions_fn(model_id)
    results_df = analyze_predictions(predictions, class_masker=masker)

    # Remove quaternion error column since it's not relevant for continual learning evaluation
    # Quaternion predictions are continuous values and don't need masking like class predictions
    results_df = results_df.drop(columns=['quaternion_error_degs'])

    # Save and summarize results
    results_df.to_csv(Path(cfg.save_dir) / f"predictions_model{model_id}.csv", index=False)


def _validate_config(cfg: DictConfig) -> None:
    """Validate the evaluation configuration.

    Args:
        cfg: Configuration to validate.

    Raises:
        AssertionError: If save directory is not provided.
    """
    assert cfg.save_dir is not None, "save_dir must be provided"


def _setup_components(
    cfg: DictConfig,
) -> tuple[LightningDataModule, LightningModule]:
    """Initialize and setup data module and model.

    Args:
        cfg: Configuration containing component specifications.

    Returns:
        Tuple of (datamodule, model).
    """
    datamodule = setup_datamodule(cfg)
    datamodule.setup()
    model = setup_model(cfg)
    return datamodule, model


def _validate_classes(datamodule: LightningDataModule, model: LightningModule) -> None:
    """Validate and log class information for datamodule and model.

    Args:
        datamodule: The instantiated datamodule.
        model: The instantiated model.
    """
    data_test_classes = datamodule.get_current_task_class_names()
    datamodule_classes = datamodule.current_task_classes
    model_trained_classes = model.current_task_classes

    log.info(f"Data test classes: {data_test_classes}")
    log.info(f"Data datamodule classes: {datamodule_classes}")
    log.info(f"Model trained classes: {model_trained_classes}")


def _get_masked_predictions_fn(model_id: int):
    """Create a function that masks logits to only consider classes seen up to model_id.

    Args:
        model_id: The ID of the current model in the sequence

    Returns:
        A function that takes logits tensor and returns masked predictions
    """

    def masker(class_probs: torch.Tensor) -> torch.Tensor:
        total_classes = class_probs.shape[1]
        if model_id + 1 < total_classes:
            mask = torch.zeros_like(class_probs, dtype=torch.bool)
            mask[:, : model_id + 1] = True
            # Sets probabilities of unseen classes to -inf to avoid them being selected
            # This ensures argmax will never pick unseen classes, even if the logits
            # go through softmax (since exp(-inf) = 0)
            masked_probs = torch.where(
                mask, class_probs, torch.tensor(-float("inf"), device=class_probs.device)
            )
            return masked_probs.argmax(dim=1)
        return class_probs.argmax(dim=1)

    return masker


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for sequential model evaluation.

    This function evaluates a sequence of models in order, representing the
    continual learning process. Each model is evaluated on its known classes.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)

    # Get max_model_id from config or command line override, default to 77 (all tasks)
    max_model_id = cfg.get("max_model_id", 77)
    log.info(f"Evaluating models 0 to {max_model_id-1}")
    
    for model_id in range(max_model_id):
        ckpt_path = get_checkpoint_path(cfg, model_id)
        if not ckpt_path.exists():
            log.warning(f"Checkpoint for model {model_id} not found at {ckpt_path}")
            continue

        log.info(f"Evaluating model {model_id}")
        evaluate_model(cfg, model_id, ckpt_path)


if __name__ == "__main__":
    main()
