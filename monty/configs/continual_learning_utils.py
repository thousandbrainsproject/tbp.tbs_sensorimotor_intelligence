# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from tbp.monty.frameworks.environments import embodied_data as ED

if TYPE_CHECKING:
    import numpy as np
    from tbp.monty.frameworks.models.motor_policies import MotorSystem
    from tbp.monty.frameworks.environments.embodied_data import EnvironmentDataLoader

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentDataloaderPerRotationArgs:
    """Arguments for the EnvironmentDataloaderPerRotation class.

    Args:
        object_names: List of object names to be used in the dataloader.
        object_init_sampler: Callable that returns the initial parameters for
            the object.

    Note:
        This class is copied after EnvironmentDataLoaderPerObjectArgs in tbp.monty v0.1.0.
    """

    object_names: list[str]
    object_init_sampler: Callable


class EnvironmentDataLoaderPerRotation(ED.EnvironmentDataLoader):
    """Dataloader for continual learning with rotation-based variation.

    This dataloader cycles through different rotations of the same object within
    episodes, then switches to a new object after completing an epoch.

    Note:
        The implementation has been simplified by removing code related to distractor
        objectsfor readability and focus on the continual learning task,
        but otherwise nearly identical to EnvironmentDataLoaderPerObject class.
    """

    def __init__(
        self,
        object_names: list[str],
        object_init_sampler: Callable,
        dataset: ED.EnvironmentDataset,
        motor_system: MotorSystem,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize the EnvironmentDataLoaderPerRotation.

        Args:
            object_names: List of object names to be used in the dataloader.
            object_init_sampler: Callable that returns the initial parameters for
                the object.
            dataset: The dataset to be used.
            motor_system: The motor system to be used.
            rng: The random number generator to be used.

        Raises:
            TypeError: If object names is not a list.
        """
        super().__init__(dataset=dataset, motor_system=motor_system, rng=rng)

        if not isinstance(object_names, list):
            error_msg = "Object names should be a list"
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Sort to match the order of classes in ViT Continual Learning
        self.object_names = sorted(object_names)
        self.source_object_list = sorted(list(dict.fromkeys(object_names)))

        self.dataloader_utils = EnvironmentDataLoaderUtils(self)
        self.semantic_id_to_label, self.semantic_label_to_id = (
            self.dataloader_utils.create_semantic_mapping()
        )
        self.object_init_sampler = object_init_sampler
        self.object_init_sampler.rng = self.rng
        self.object_params = self.object_init_sampler()
        self.current_object = 0
        self.n_objects = len(self.object_names)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None

    def pre_episode(self) -> None:
        """Pre-episode setup for continual learning dataloader."""
        super().pre_episode()
        self.reset_agent()

    def post_episode(self) -> None:
        """Post-episode setup for continual learning dataloader."""
        super().post_episode()
        self.object_init_sampler.post_episode()
        self.cycle_rotation()
        self.episodes += 1

    def pre_epoch(self) -> None:
        """Pre-epoch setup for continual learning dataloader."""
        self.update_primary_target_object(self.current_object, self.object_params)

    def post_epoch(self) -> None:
        """Post-epoch setup for continual learning dataloader."""
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
        logger.info(
            "Going from rotation: %(current_rotation)s to rotation: %(next_rotation)s",
            extra={
                "current_rotation": current_rotation,
                "next_rotation": next_rotation,
            },
        )
        self.update_primary_target_object(self.current_object, self.object_params)

    def update_primary_target_object(self, idx: int, object_params: dict) -> None:
        """Update the primary target object in the scene based on the given index.

        Args:
            idx: Index of the new object and ints parameters in object_params
            object_params: Parameters for the new object

        Raises:
            ValueError: If the index is greater than the number of objects.

        Note:
            Analogous to EnvironmentDataLoaderPerObject.change_object_by_idx.
        """
        if idx > self.n_objects:
            error_msg = f"idx must be <= self.n_objects: {self.n_objects}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.dataset.env.remove_all_objects()

        # Specify config for the primary target object and then add it
        init_params = object_params.copy()
        init_params.pop("euler_rotation")
        if "quat_rotation" in init_params:
            init_params.pop("quat_rotation")
        init_params["semantic_id"] = self.semantic_label_to_id[self.object_names[idx]]

        _ = self.dataset.env.add_object(name=self.object_names[idx], **init_params)

        self.primary_target = {
            "object": self.object_names[idx],
            "semantic_id": self.semantic_label_to_id[self.object_names[idx]],
            **object_params,
        }
        logger.info(
            "New primary target: %(primary_target)",
            extra={"primary_target": self.primary_target},
        )

    def create_semantic_mapping(self) -> None:
        """Create a unique semantic ID for each object.

        Raises:
            ValueError: If the object names are not a subset of the source object list.
        """
        if not set(self.object_names).issubset(set(self.source_object_list)):
            error_msg = "Semantic mapping requires primary targets \
                sampled from source list"
            logger.error(error_msg)
            raise ValueError(error_msg)

        starting_integer = 1  # Start at 1 so that we can distinguish on-object semantic
        # IDs (>0) from being off object (semantic_id == 0 in Habitat by default)
        self.semantic_id_to_label = {
            i + starting_integer: label
            for i, label in enumerate(self.source_object_list)
        }
        self.semantic_label_to_id = {
            label: i + starting_integer
            for i, label in enumerate(self.source_object_list)
        }

    def reset_agent(self) -> dict:
        """Reset the agent to the initial state.

        Returns:
            The observation from the reset.
        """
        logger.debug("resetting agent------")
        self._observation, self.motor_system.state = self.dataset.reset()
        self._counter = 0

        self._action = None
        self._amount = None
        self.motor_system.state[self.motor_system.agent_id]["motor_only_step"] = False

        return self._observation


class InformedEnvironmentDataLoaderPerRotation(EnvironmentDataLoaderPerRotation):
    """Adapter class for InformedEnvironmentDataLoader.

    This class wraps an instance of the original InformedEnvironmentDataLoader and
    delegates all attribute access to it to avoid code duplication. The only difference
    to InformedEnvironmentDastaLoader in tbp.monty v.0.1.0 is that it inherits
    from EnvironmentDataLoaderPerRotation instead of EnvironmentDataLoaderPerObject.
    """

    def __init__(
        self,
        object_names: list[str],
        object_init_sampler: Callable,
        dataset: ED.EnvironmentDataset,
        motor_system: MotorSystem,
        rng: np.random.RandomState,
    ) -> None:
        """Initialize the InformedEnvironmentDataLoaderPerRotation.

        Args:
            object_names: List of object names to be used in the dataloader.
            object_init_sampler: Callable that returns the initial parameters for
                the object.
            dataset: The dataset to be used.
            motor_system: The motor system to be used.
            rng: The random number generator to be used.
        """
        super().__init__(object_names, object_init_sampler, dataset, motor_system, rng)
        self._original = ED.InformedEnvironmentDataLoader(dataset, motor_system, rng)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate attribute access to InformedEnvironmentDataLoader instance.

        Args:
            name (str): The attribute name to access.

        Returns:
            Any: The corresponding attribute from the wrapped instance.
        """
        return getattr(self._original, name)
