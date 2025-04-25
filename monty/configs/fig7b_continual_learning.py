# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Figure 7b: Continual Learning.

Consists of one pretraining experiment with checkpointing after each epoch
and evaluation experiments after each epoch.

The dataloader is customized such that each epoch contains one object
across all rotations.

Experiment configs:
- pretrain_continual_learning_dist_agent_1lm_checkpoints
- continual_learning_dist_agent_1lm_task0
- continual_learning_dist_agent_1lm_task1
...
- continual_learning_dist_agent_1lm_task76

This means performance is evaluated with:
- 77 objects
- 5 random rotations
- NO sensor noise*
- NO hypothesis-testing*
- No voting
- Each evaluation observes all objects up to that point
"""

import logging
from copy import deepcopy

from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)

from .common import DMC_PRETRAIN_DIR, RANDOM_ROTATIONS_5
from .continual_learning_utils import (
    EnvironmentDataLoaderPerRotation,
    InformedEnvironmentDataLoader,
)
from .fig3_robust_sensorimotor_inference import dist_agent_1lm
from .fig7_rapid_learning import PretrainingExperimentWithCheckpointing
from .pretraining_experiments import (
    TRAIN_ROTATIONS,
    DMCPretrainLoggingConfig,
    pretrain_dist_agent_1lm,
)

# Create a logger for this module
logger = logging.getLogger(__name__)

"""
Pretraining Configs
--------------------------------------------------------------------------------
"""


class PretrainingContinualLearningExperimentWithCheckpointing(
    PretrainingExperimentWithCheckpointing
):
    """Extends parent class for continual learning pretraining with checkpointing.

    NOTE: Experiments using this class cannot be run in parallel.
    """

    def run_epoch(self) -> None:
        """Run a single epoch of continual learning pretraining.

        Raises:
            TypeError: If the dataloader is not EnvironmentDataLoaderPerRotation.
        """
        self.pre_epoch()
        if isinstance(self.dataloader, EnvironmentDataLoaderPerRotation):
            for _ in range(len(TRAIN_ROTATIONS)):
                logger.info(
                    "Current object: %(object)s at rotation: %(rotation)s",
                    extra={
                        "object": self.dataloader.current_object,
                        "rotation": self.dataloader.object_params["euler_rotation"],
                    },
                )
                self.run_episode()
        else:
            error_msg = "Dataloader should be EnvironmentDataLoaderPerRotation"
            logger.error(error_msg)
            raise TypeError(error_msg)
        self.post_epoch()


class EvalContinualLearningExperiment(MontyObjectRecognitionExperiment):
    """Continual learning evaluation experiment."""

    def run_epoch(self) -> None:
        """Run a single epoch of continual learning evaluation.

        Raises:
            TypeError: If the dataloader is not EnvironmentDataLoaderPerRotation.
        """
        self.pre_epoch()
        if isinstance(self.dataloader, EnvironmentDataLoaderPerRotation):
            for _ in range(len(RANDOM_ROTATIONS_5)):
                logger.info(
                    "Current object: %(object)s",
                    extra={"object": self.dataloader.current_object},
                )
                logger.info(
                    "Simulating object: %(object_name)s with params: %(params)s",
                    extra={
                        "object_name": self.dataloader.object_names[
                            self.dataloader.current_object
                        ],
                        "params": self.dataloader.object_params,
                    },
                )
                self.run_episode()
        else:
            error_msg = "Dataloader should be EnvironmentDataLoaderPerRotation"
            logger.error(error_msg)
            raise TypeError(error_msg)
        self.post_epoch()

    @property
    def logger_args(self) -> dict:
        """Get the arguments for the logger.

        Returns:
            dict: The arguments for the logger.
        """
        args = dict(
            total_train_steps=self.total_train_steps,
            train_episodes=self.train_episodes,
            train_epochs=self.train_epochs,
            total_eval_steps=self.total_eval_steps,
            eval_episodes=self.eval_episodes,
            eval_epochs=self.eval_epochs,
        )
        if isinstance(self.dataloader, EnvironmentDataLoaderPerRotation):
            args.update(target=self.dataloader.primary_target)
        return args


def make_continual_learning_eval_config(task_id: int) -> dict:
    """Make an eval config that loads a pretrained model checkpoint.

    The returned config specifies a 1-LM distant agent that evaluates on
      - All objects up to and including the `task_id`th object
      - 5 random rotations
      - no sensor noise
      - no hypothesis-testing
    and loads a model pretrained after `task_id` epochs.

    Args:
        task_id (int): The ID of the task to evaluate on.

    Returns:
        dict: Config for a partially trained model.
    """
    config = deepcopy(dist_agent_1lm)
    config["experiment_args"].n_eval_epochs = task_id + 1

    # Change model loading path
    model_path = str(
        DMC_PRETRAIN_DIR
        / "continual_learning_dist_agent_1lm_checkpoints"
        / "pretrained"
        / "checkpoints"
        / f"{task_id + 1}"
        / "model.pt"
    )

    config["experiment_class"] = EvalContinualLearningExperiment
    config["experiment_args"].model_name_or_path = model_path

    config["eval_dataloader_class"] = InformedEnvironmentDataLoader
    config["eval_dataloader_args"] = EnvironmentDataloaderPerObjectArgs(
        object_names=sorted(SHUFFLED_YCB_OBJECTS)[: task_id + 1],
        object_init_sampler=PredefinedObjectInitializer(
            change_every_episode=True, rotations=RANDOM_ROTATIONS_5
        ),
    )

    # Rename the experiment
    config[
        "logging_config"
    ].run_name = f"continual_learning_dist_agent_1lm_checkpoints_task{task_id}"
    config["logging_config"].python_log_level = "INFO"
    # Disable wandb logging to save WandB space and time
    config["logging_config"].wandb_handlers = []

    # Disable hypothesis-testing
    config[
        "monty_config"
    ].motor_system_config.motor_system_args.use_goal_state_driven_actions = False

    return config


pretrain_continual_learning_dist_agent_1lm_checkpoints = deepcopy(
    pretrain_dist_agent_1lm
)
pretrain_continual_learning_dist_agent_1lm_checkpoints.update(
    dict(
        experiment_class=PretrainingContinualLearningExperimentWithCheckpointing,
        experiment_args=ExperimentArgs(
            n_train_epochs=len(SHUFFLED_YCB_OBJECTS),
            do_eval=False,
        ),
        logging_config=DMCPretrainLoggingConfig(
            run_name="continual_learning_dist_agent_1lm_checkpoints"
        ),
        train_dataloader_class=InformedEnvironmentDataLoader,
        train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=SHUFFLED_YCB_OBJECTS,
            object_init_sampler=PredefinedObjectInitializer(
                change_every_episode=True, rotations=TRAIN_ROTATIONS
            ),
        ),
    )
)


CONFIGS = {
    "pretrain_continual_learning_dist_agent_1lm_checkpoints": (
        pretrain_continual_learning_dist_agent_1lm_checkpoints
    ),
}

# Add all per-task eval configs
for task_id in range(77):
    eval_config_name = f"continual_learning_dist_agent_1lm_task{task_id}"
    CONFIGS[eval_config_name] = make_continual_learning_eval_config(task_id)
