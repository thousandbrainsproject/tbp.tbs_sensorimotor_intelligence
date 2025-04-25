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
from typing import Callable, List

import numpy as np
import quaternion

from tbp.monty.frameworks.actions.actions import (
    MoveTangentially,
    SetAgentPose,
    SetSensorRotation,
)
from tbp.monty.frameworks.environments import embodied_data as ED

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
    """Dataloader specialized for continual learning that focuses on presenting a single object 
    with varying rotations across episodes.
    
    This dataloader cycles through different rotations of the same object within episodes,
    then switches to a new object after completing an epoch. This approach enables
    learning across different viewpoints of the same object before moving to another object,
    facilitating continual learning scenarios where view-invariant representations are desired.
    
    Key workflow:
    - Within an epoch: maintain the same object but cycle through different rotations for each episode
    - Between epochs: switch to the next object in the list with initial rotation
    
    Unlike EnvironmentDataLoaderPerObject, this class:
    - Prioritizes rotation variations over object variations
    - Handles rotation cycling within episodes via cycle_rotation()
    - Tracks episode and epoch counts for continual learning procedures
    
    Note:
        The implementation has been simplified by removing distractor object functionality
        for readability and focus on the continual learning task, but otherwise nearly identical 
        to EnvironmentDataLoaderPerObject class.
    """

    def __init__(
        self,
        object_names: List[str],
        object_init_sampler: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(object_names, list):
            self.object_names = sorted(object_names)  # Sort to match the order of ViT Continual Learning
            self.source_object_list = sorted(list(dict.fromkeys(object_names)))
        else:
            raise ValueError("Object names should be a list")
    
        self.create_semantic_mapping()
        self.object_init_sampler = object_init_sampler
        self.object_init_sampler.rng = self.rng
        self.object_params = self.object_init_sampler()
        self.current_object = 0
        self.n_objects = len(self.object_names)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None

    def pre_episode(self) -> None:
        """Pre-episode setup for continual learning dataloader.

        This method performs the following actions:
        1. Calls the superclass's pre_episode method to perform the default pre-episode setup.
        2. Resets the agent to the initial state.
        """
        super().pre_episode()
        self.reset_agent()

    def post_episode(self) -> None:
        """Post-episode setup for continual learning dataloader.

        This method performs the following actions:
        1. Calls the superclass's post_episode method to perform the default post-episode setup.
        2. Calls the object_init_sampler's post_episode method to perform any necessary post-episode setup.
        3. Cycles the rotation of the object.
        4. Increments the episode count.
        """
        super().post_episode()
        self.object_init_sampler.post_episode()
        self.cycle_rotation()
        self.episodes += 1
    
    def pre_epoch(self) -> None:
        """Pre-epoch setup for continual learning dataloader.

        This method performs the following actions:
        1. Calls the superclass's pre_epoch method to perform the default pre-epoch setup.
        2. Changes the object to the next object in the list.
        """
        self.change_object_by_idx(self.current_object, self.object_params)

    def post_epoch(self) -> None:
        """Post-epoch setup for continual learning dataloader.

        This method performs the following actions:
        1. Increments the epoch count.
        2. Cycles the rotation of the object.
        3. Resets the agent to the initial state.
        """
        self.epochs += 1
        self.current_object += 1
        self.object_init_sampler.post_epoch()
        self.object_params = self.object_init_sampler()
        self.reset_agent()

    def cycle_rotation(self) -> None:
        """Cycle the rotation of the object.

        This method performs the following actions:
        1. Gets the current rotation of the object.
        2. Cycles the rotation of the object.
        """
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
        self.change_object_by_idx(self.current_object, self.object_params)

    def create_semantic_mapping(self) -> None:
        """Create a unique semantic ID (positive integer) for each object.

        Used by Habitat for the semantic sensor.

        In addition, create a dictionary mapping back and forth between these IDs and
        the corresponding name of the object
        """
        assert set(self.object_names).issubset(
            set(self.source_object_list)
        ), "Semantic mapping requires primary targets sampled from source list"

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
    
    def change_object_by_idx(self, idx: int, object_params: dict) -> None:
        """Update the primary target object in the scene based on the given index.

        The given `idx` is the index of the object in the `self.object_names` list,
        which should correspond to the index of the object in the `self.object_params`
        list.

        Also add any distractor objects if required.

        Args:
            idx: Index of the new object and ints parameters in object_params
            object_params: Parameters for the new object

        Raises:
            ValueError: If the index is greater than the number of objects.
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

        _ = self.dataset.env.add_object(
            name=self.object_names[idx], **init_params
        )

        self.primary_target = {
            "object": self.object_names[idx],
            "semantic_id": self.semantic_label_to_id[self.object_names[idx]],
            **object_params,
        }
        logger.info("New primary target: %(primary_target)", 
                    extra={"primary_target": self.primary_target})

    def reset_agent(self) -> None: 
        """Reset the agent to the initial state.

        This method performs the following actions:
        1. Resets the dataset.
        2. Resets the counter.
        """
        logger.debug("resetting agent------")
        self._observation, self.motor_system.state = self.dataset.reset()
        self._counter = 0

        self._action = None
        self._amount = None
        self.motor_system.state[self.motor_system.agent_id]["motor_only_step"] = False

        return self._observation

class InformedEnvironmentDataLoaderPerRotation(EnvironmentDataLoaderPerRotation):
    """
    Adapter class for InformedEnvironmentDataLoader that inherits from
    EnvironmentDataLoaderPerRotation instead of EnvironmentDataLoaderPerObject.

    This class wraps an instance of the original InformedEnvironmentDataLoader and delegates
    all attribute access to it. It inherits from EnvironmentDataLoaderPerRotation
    in order to change the data-loading semantics for use cases where rotation-based
    logic is required.

    Args:
        *args: Positional arguments to be passed to both the parent and the original dataloader.
        **kwargs: Keyword arguments to be passed to both the parent and the original dataloader.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original = ED.InformedEnvironmentDataLoader(*args, **kwargs)

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped InformedEnvironmentDataLoader instance.

        Args:
            name (str): The attribute name to access.

        Returns:
            Any: The corresponding attribute from the wrapped instance.
        """
        return getattr(self._original, name)