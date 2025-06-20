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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.losses.loss import quaternion_geodesic_loss
from src.utils import (
    RankedLogger,
    extras,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.lightning_utils import (
    prepare_object_dict,
    setup_datamodule,
    setup_loggers,
    setup_model,
    setup_trainer,
)
from src.utils.eval_utils import (
    generate_predictions,
    analyze_predictions,
    save_and_summarize_results,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Type aliases for better readability
PredictionBatch = dict[str, torch.Tensor]
MetricsDict = dict[str, Any]


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[MetricsDict, dict[str, Any]]:
    """Evaluate a model checkpoint on a datamodule's test set.

    This function handles the complete evaluation pipeline including data preparation,
    model instantiation, testing, and prediction analysis.

    Args:
        cfg: A DictConfig configuration composed by Hydra containing all necessary
            parameters for evaluation.

    Returns:
        Tuple containing:
            - dict[str, Any]: Evaluation metrics
            - dict[str, Any]: Dictionary containing all instantiated objects

    Raises:
        AssertionError: If checkpoint path or save directory is not provided.
    """
    _validate_config(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Initialize components
    datamodule = setup_datamodule(cfg)
    model = setup_model(cfg)
    loggers = setup_loggers(cfg)
    trainer = setup_trainer(cfg, loggers=loggers)

    # Prepare object dictionary for logging
    object_dict = prepare_object_dict(cfg, datamodule, model, loggers=loggers, trainer=trainer)

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Run evaluation
    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # Generate and analyze predictions
    predictions = generate_predictions(trainer, model, datamodule, cfg.ckpt_path)
    results_df = analyze_predictions(predictions)

    # Save and summarize results
    save_and_summarize_results(results_df, cfg.save_dir)

    return trainer.callback_metrics, object_dict


def _validate_config(cfg: DictConfig) -> None:
    """Validate the evaluation configuration.

    Args:
        cfg: Configuration to validate.

    Raises:
        AssertionError: If required paths are not provided.
    """
    assert cfg.ckpt_path, "Checkpoint path must be provided for evaluation"
    assert cfg.save_dir, "Save directory must be specified"


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for model evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
