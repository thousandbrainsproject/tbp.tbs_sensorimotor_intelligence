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
from typing import Any

from tbp.monty.frameworks.environments import embodied_data as ED


class InformedEnvironmentDataLoaderForContinualLearning(
    ED.InformedEnvironmentDataLoader
):
    """InformedEnvironmentDataLoader for continual learning."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        object_names = kwargs.get("object_names", [])
        self.object_names = sorted(object_names)
        self.source_object_list = sorted(list(dict.fromkeys(object_names)))

    def post_episode(self) -> None:
        """Post-episode setup for continual learning dataloader."""
        self.motor_system.post_episode()
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
        logging.info(
            f"Going from rotation: {current_rotation} to rotation: {next_rotation}",
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
            logging.error(error_msg)
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
        logging.info(
            f"New primary target: {self.primary_target}",
        )
