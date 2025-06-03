# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Figure 7b: Continual Learning.

Consists of one pretraining experiment, which saves a model checkpoint after each epoch,
and a series of evaluation experiments backed by the stored checkpoints.


The dataloader is customized such that each epoch contains one object
shown at all rotations.

Experiment configs:
- pretrain_continual_learning_dist_agent_1lm_checkpoints
- continual_learning_dist_agent_1lm_task0
- continual_learning_dist_agent_1lm_task1
...
- continual_learning_dist_agent_1lm_task76

This means performance is evaluated on:
- N objects seen in pretraining
- 5 random rotations
- NO sensor noise*
- NO hypothesis-testing*
- No voting
"""

import logging
from copy import deepcopy
from typing import Any

from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    ExperimentArgs,
    PredefinedObjectInitializer,
    EnvironmentDataloaderPerObjectArgs,
)
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)

from .common import DMC_PRETRAIN_DIR, RANDOM_ROTATIONS_5
from .fig3_robust_sensorimotor_inference import dist_agent_1lm
from .fig7a_rapid_learning import PretrainingExperimentWithCheckpointing
from .pretraining_experiments import (
    TRAIN_ROTATIONS,
    DMCPretrainLoggingConfig,
    pretrain_dist_agent_1lm,
)

"""
Continual Learning Dataloader
"""


class InformedEnvironmentDataLoaderPerRotation(ED.InformedEnvironmentDataLoader):
    """InformedEnvironmentDataLoader for continual learning.

    This dataloader overrides the following functions from
     the base InformedEnvironmentDataLoader.
    - pre_episode
    - post_episode
    - pre_epoch
    - post_epoch
    - cycle_rotation
    - update_primary_target_object

    Note that these functions are located in the parent class
    InformedEnvironmentDataLoader (at EnvironmentDataLoaderPerObject).
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        object_names = kwargs.get("object_names", [])
        self.source_object_list = sorted(list(dict.fromkeys(object_names)))

        self.object_init_sampler.post_episode()
        self.cycle_rotation()
        self.episodes += 1

    def pre_epoch(self) -> None:
        """Pre-epoch setup for continual learning dataloader."""
        self.update_primary_target_object()

    def post_epoch(self) -> None:
        """Post-epoch setup for continual learning dataloader.

        This method handles the transition between epochs by:
        - Incrementing the epoch counter
        - Moving to the next object in the sequence
        - Updating object parameters through the sampler
        - Resetting the agent state
        """
        self.epochs += 1
        self.current_object += 1
        self.object_init_sampler.post_epoch()
        self.object_params = self.object_init_sampler()
        self.reset_agent()

    def cycle_rotation(self) -> None:
        """Cycle the rotation of the object."""
        current_rotation = self.object_params["euler_rotation"]
        self.object_params = self.object_init_sampler()
        next_rotation = self.object_params["euler_rotation"]
        logging.info(
            f"Going from rotation: {current_rotation} to rotation: {next_rotation}",
        )
        self.update_primary_target_object()

    def update_primary_target_object(self) -> None:
        """Update the primary target object in the scene.

        Raises:
            ValueError: If the current object index is greater than the number of
            objects.

        Note:
            Analogous to EnvironmentDataLoaderPerObject.change_object_by_idx.
        """
        if self.current_object > self.n_objects:
            error_msg = f"current_object must be <= self.n_objects: {self.n_objects}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        self.dataset.env.remove_all_objects()

        # Specify config for the primary target object and then add it
        init_params = self.object_params.copy()
        init_params.pop("euler_rotation")
        if "quat_rotation" in init_params:
            init_params.pop("quat_rotation")
        init_params["semantic_id"] = self.semantic_label_to_id[
            self.object_names[self.current_object]
        ]

        _ = self.dataset.env.add_object(
            name=self.object_names[self.current_object], **init_params
        )

        self.primary_target = {
            "object": self.object_names[self.current_object],
            "semantic_id": self.semantic_label_to_id[
                self.object_names[self.current_object]
            ],
            **self.object_params,
        }
        logging.info(
            f"New primary target: {self.primary_target}",
        )


"""
Pretraining Configs
--------------------------------------------------------------------------------
"""

ALPHABETICALLY_SORTED_OBJECT_NAMES = sorted(SHUFFLED_YCB_OBJECTS)


class PretrainingContinualLearningExperimentWithCheckpointing(
    PretrainingExperimentWithCheckpointing
):
    """Extends parent class for continual learning pretraining with checkpointing."""

    def run_epoch(self) -> None:
        """Run a single epoch of continual learning pretraining.

        Raises:
            TypeError: If the dataloader is not InformedEnvironmentDataLoaderPerRotation
        """
        self.pre_epoch()
        if isinstance(self.dataloader, InformedEnvironmentDataLoaderPerRotation):
            for _ in range(len(TRAIN_ROTATIONS)):
                logging.info(
                    f"Current object: {self.dataloader.current_object} at rotation: \
                    {self.dataloader.object_params['euler_rotation']}",
                )
                self.run_episode()
        else:
            error_msg = (
                "Dataloader should be InformedEnvironmentDataLoaderForContinualLearning"
            )
            logging.error(error_msg)
            raise TypeError(error_msg)
        self.post_epoch()


class EvalContinualLearningExperiment(MontyObjectRecognitionExperiment):
    """Continual learning evaluation experiment."""

    def run_epoch(self) -> None:
        """Run a single epoch of continual learning evaluation.

        One object is shown for each epoch, and each episode within an epoch
        shows the object at a different rotation.

        Raises:
            TypeError: If the dataloader is not InformedEnvironmentDataLoaderPerRotation
        """
        self.pre_epoch()
        if isinstance(self.dataloader, InformedEnvironmentDataLoaderPerRotation):
            for _ in range(len(RANDOM_ROTATIONS_5)):
                logging.info(
                    f"Current object: {self.dataloader.current_object}",
                )
                logging.info(
                    f"Simulating object: \
                    {self.dataloader.object_names[self.dataloader.current_object]} \
                    with params: \
                    {self.dataloader.object_params}",
                )
                self.run_episode()
        else:
            error_msg = (
                "Dataloader should be InformedEnvironmentDataLoaderForContinualLearning"
            )
            logging.error(error_msg)
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
        if isinstance(self.dataloader, InformedEnvironmentDataLoaderPerRotation):
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
        task_id (int): The ID of the task to evaluate on, indexed from 0.

    Returns:
        dict: Config for a partially trained model.
    """
    config = deepcopy(dist_agent_1lm)
    num_objects_seen = task_id + 1
    config["experiment_args"].n_eval_epochs = num_objects_seen

    # Change model loading path
    model_path = str(
        DMC_PRETRAIN_DIR
        / "continual_learning_dist_agent_1lm_checkpoints"
        / "pretrained"
        / "checkpoints"
        / f"{num_objects_seen}"
        / "model.pt"
    )

    config["experiment_class"] = EvalContinualLearningExperiment
    config["experiment_args"].model_name_or_path = model_path

    config["eval_dataloader_class"] = InformedEnvironmentDataLoaderPerRotation
    config["eval_dataloader_args"] = EnvironmentDataloaderPerObjectArgs(
        object_names=ALPHABETICALLY_SORTED_OBJECT_NAMES[:num_objects_seen],
        object_init_sampler=PredefinedObjectInitializer(
            change_every_episode=True, rotations=RANDOM_ROTATIONS_5
        ),
    )

    # Rename the experiment
    config[
        "logging_config"
    ].run_name = f"continual_learning_dist_agent_1lm_task{task_id}"
    config["logging_config"].python_log_level = "INFO"

    # Disable wandb logging to save WandB space
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
            run_name="continual_learning_dist_agent_1lm_checkpoints",
            python_log_level="INFO",
        ),
        train_dataloader_class=InformedEnvironmentDataLoaderPerRotation,
        train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=ALPHABETICALLY_SORTED_OBJECT_NAMES,
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
for task_id in range(len(SHUFFLED_YCB_OBJECTS)):
    eval_config_name = f"continual_learning_dist_agent_1lm_task{task_id}"
    CONFIGS[eval_config_name] = make_continual_learning_eval_config(task_id)
