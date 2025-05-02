# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 3: Robust Sensorimotor Inference.

This module defines the following experiments:
 - `dist_agent_1lm`
 - `dist_agent_1lm_noise_all`
 - `dist_agent_1lm_randrot_14`
 - `dist_agent_1lm_randrot_14_noise_all`
 - `dist_agent_1lm_randrot_14_noise_all_color_clamped`

 Experiments use:
 - 77 objects
 - 14 rotations
 - Goal-state-driven/hypothesis-testing policy active
 - A single LM (no voting)

NOTE: random rotation variants use the random object initializer and 14 rotations.
`dist_agent_1lm_randrot_noise` which uses the 5 predefined "random" rotations
is defined in `fig5_rapid_inference_with_voting.py`.
"""

from copy import deepcopy

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    PatchAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.sensor_modules import FeatureChangeSM
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMountHabitatDatasetArgs,
)

from .common import (
    DMC_PRETRAIN_DIR,
    MAX_EVAL_STEPS,
    MAX_TOTAL_STEPS,
    MIN_EVAL_STEPS,
    DMCEvalLoggingConfig,
    get_eval_lm_config,
    get_eval_motor_config,
    get_eval_patch_config,
    get_view_finder_config,
    make_noise_variant,
)

# - 14 Rotation used during training (cube faces + corners)
TEST_ROTATIONS = get_cube_face_and_corner_views_rotations()


dist_agent_1lm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_1lm/pretrained"),
        n_eval_epochs=len(TEST_ROTATIONS),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="dist_agent_1lm"),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyArgs(min_eval_steps=MIN_EVAL_STEPS),
        sensor_module_configs=dict(
            sensor_module_0=get_eval_patch_config("dist"),
            sensor_module_1=get_view_finder_config(),
        ),
        learning_module_configs=dict(
            learning_module_0=get_eval_lm_config("dist"),
        ),
        motor_system_config=get_eval_motor_config("dist"),
    ),
    # Set up environment.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TEST_ROTATIONS),
    ),
)

# Noisy/random rotation variants
# ------------------------------------------------------------------------------

noise_all_params = {
    "location": 0.002,
    "features": {
        "pose_vectors": 2.0,
        "hsv": 0.1,
        "principal_curvatures_log": 0.1,
        "pose_fully_defined": 0.01,
    },
}

# - Noisy sensor variant
dist_agent_1lm_noise_all = make_noise_variant(
    dist_agent_1lm,
    noise_all_params,
    run_name="dist_agent_1lm_noise_all",
)

# - Random rotation variant (14 random rotations)
dist_agent_1lm_randrot_14 = deepcopy(dist_agent_1lm)
dist_agent_1lm_randrot_14["logging_config"].run_name = "dist_agent_1lm_randrot_14"
dist_agent_1lm_randrot_14[
    "eval_dataloader_args"
].object_init_sampler = RandomRotationObjectInitializer()

# - Random rotation variant (14 random rotations) and sensor noise
dist_agent_1lm_randrot_14_noise_all = make_noise_variant(
    dist_agent_1lm_randrot_14,
    noise_all_params,
    run_name="dist_agent_1lm_randrot_14_noise_all",
)

class ClampedColorSM(FeatureChangeSM):
    """Sensor module that clamps the hsv feature to blue."""

    def step(self, data):
        """Clamp hsv to solid blue if the observation is usable."""
        # Extract features.
        patch_observation = super().step(data)
        # Force hsv to solid blue.
        if patch_observation.use_state:
            if "hsv" in patch_observation.non_morphological_features:
                patch_observation.non_morphological_features["hsv"] = np.array(
                    [0.667, 1.0, 1.0]
                )
            else:
                raise ValueError("hsv feature not found")
        return patch_observation


# - Random rotation variant (14 random rotations) and sensor noise with
#   the 'hsv' feature clamped to solid blue.
dist_agent_1lm_randrot_14_noise_all_color_clamped = deepcopy(
    dist_agent_1lm_randrot_14_noise_all
)
dist_agent_1lm_randrot_14_noise_all_color_clamped[
    "logging_config"
].run_name = "dist_agent_1lm_randrot_14_noise_all_color_clamped"

dist_agent_1lm_randrot_14_noise_all_color_clamped["monty_config"].sensor_module_configs[
    "sensor_module_0"
]["sensor_module_class"] = ClampedColorSM


CONFIGS = {
    "dist_agent_1lm": dist_agent_1lm,
    "dist_agent_1lm_noise_all": dist_agent_1lm_noise_all,
    "dist_agent_1lm_randrot_14": dist_agent_1lm_randrot_14,
    "dist_agent_1lm_randrot_14_noise_all": dist_agent_1lm_randrot_14_noise_all,
    "dist_agent_1lm_randrot_14_noise_all_color_clamped": dist_agent_1lm_randrot_14_noise_all_color_clamped,
}
