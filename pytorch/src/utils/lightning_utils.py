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

from collections.abc import Sequence
from typing import Any

import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import RankedLogger, instantiate_callbacks, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)

# Type aliases for better readability
ObjectDict = dict[str, Any]


def setup_datamodule(cfg: DictConfig) -> LightningDataModule:
    """Initialize the data module from configuration.

    Args:
        cfg: Configuration containing datamodule specifications.

    Returns:
        Instantiated LightningDataModule.
    """
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    return hydra.utils.instantiate(cfg.data)


def setup_model(cfg: DictConfig) -> LightningModule:
    """Initialize the model from configuration.

    Args:
        cfg: Configuration containing model specifications.

    Returns:
        Instantiated LightningModule.
    """
    log.info(f"Instantiating model <{cfg.model._target_}>")
    return hydra.utils.instantiate(cfg.model)


def setup_loggers(cfg: DictConfig) -> list[Logger]:
    """Initialize training loggers from configuration.

    Args:
        cfg: Configuration containing logger specifications.

    Returns:
        List of instantiated loggers.
    """
    log.info("Instantiating loggers...")
    return instantiate_loggers(cfg.get("logger"))


def setup_callbacks(cfg: DictConfig) -> list[Callback]:
    """Initialize training callbacks from configuration.

    Args:
        cfg: Configuration containing callback specifications.

    Returns:
        List of instantiated callbacks.
    """
    log.info("Instantiating callbacks...")
    return instantiate_callbacks(cfg.get("callbacks"))


def setup_trainer(
    cfg: DictConfig,
    callbacks: Sequence[Callback] | None = None,
    loggers: Sequence[Logger] | None = None,
) -> Trainer:
    """Initialize the PyTorch Lightning trainer.

    Args:
        cfg: Configuration containing trainer specifications.
        callbacks: Optional list of callbacks to attach to the trainer.
        loggers: Optional list of loggers to attach to the trainer.

    Returns:
        Instantiated Trainer object.
    """
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_kwargs = {}
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks
    if loggers is not None:
        trainer_kwargs["logger"] = loggers
    return hydra.utils.instantiate(cfg.trainer, **trainer_kwargs)


def prepare_object_dict(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    model: LightningModule,
    callbacks: Sequence[Callback] | None = None,
    loggers: Sequence[Logger] | None = None,
    trainer: Trainer | None = None,
) -> ObjectDict:
    """Prepare dictionary containing all instantiated objects.

    Args:
        cfg: Main configuration object.
        datamodule: Instantiated datamodule.
        model: Instantiated model.
        callbacks: Optional list of callbacks.
        loggers: Optional list of loggers.
        trainer: Optional trainer instance.

    Returns:
        Dictionary containing all instantiated objects.
    """
    object_dict: ObjectDict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
    }

    if callbacks is not None:
        object_dict["callbacks"] = callbacks
    if loggers is not None:
        object_dict["logger"] = loggers
    if trainer is not None:
        object_dict["trainer"] = trainer

    return object_dict
