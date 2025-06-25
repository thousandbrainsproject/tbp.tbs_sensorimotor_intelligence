# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 8: Flops Comparison.

This module defines the following inference experiments:
 - `dist_agent_1lm_randrot_nohyp` (No hypothesis testing)
 - `dist_agent_1lm_randrot` (Hypothesis testing)

And the following training experiment:
 - `pretrain_dist_agent_1lm_k_none`

Note that the training experiment is identical to `pretrain_dist_agent_1lm` except
that the argument k=0 in DisplacementGraphLM. This is to prevent FLOP
counts associated with building unncessary edges of a graph, as these are not used 
during inference.

Inference experiments use:
 - 77 objects
 - 5 random rotations
 - No sensor noise
 - No voting

The main output measure is accuracy and FLOPs as a function of x-percent threshold and 
whether hypothesis testing is used.
"""

import copy

from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM

from .fig5_rapid_inference_with_voting import (
    dist_agent_1lm_randrot_noise,  # With hypothesis testing
)
from .fig6_rapid_inference_with_model_based_policies import (
    dist_agent_1lm_randrot_noise_nohyp,
)  # No hypothesis testing
from .pretraining_experiments import pretrain_dist_agent_1lm


def update_x_percent_threshold_in_config(
    template: dict,
    x_percent_threshold: int,
    evidence_update_threshold: str = "80%",
) -> dict:
    """Update the x_percent threshold in the config.
    This function modifies the config in-place.

    Args:
        template (dict): The config to update.
        x_percent_threshold (int): The percentage of the threshold to update.
        evidence_update_threshold (str): The evidence update threshold to set.

    Returns:
        dict: The updated config.
    """
    config = copy.deepcopy(template)

    # Update the x_percent_threshold
    lm_config_dict = config["monty_config"].learning_module_configs
    lm_config_dict["learning_module_0"]["learning_module_args"][
        "x_percent_threshold"
    ] = x_percent_threshold

    # Set the evidence update to "80%"
    lm_config_dict["learning_module_0"]["learning_module_args"][
        "evidence_update_threshold"
    ] = evidence_update_threshold

    # Update the logging run name
    config[
        "logging_config"
    ].run_name = f"{config['logging_config'].run_name}_x_percent_{x_percent_threshold}"

    return config


##########################
# Base Inference Configs #
##########################
"""Creates two base configurations:

1. dist_agent_1lm_randrot_nohyp:
   - Based on dist_agent_1lm_randrot_noise_nohyp
   - Disables sensor noise
   - No hypothesis testing
   
2. dist_agent_1lm_randrot:
   - Based on dist_agent_1lm_randrot_noise
   - Disables sensor noise
   - Includes hypothesis testing

Both configurations serve as templates for the various x-percent threshold experiments
that follow. The main difference between them is the presence/absence of hypothesis
testing functionality.
"""

# Define dist_agent_1lm_randrot_nohyp
dist_agent_1lm_randrot_nohyp = copy.deepcopy(dist_agent_1lm_randrot_noise_nohyp)
for sm_dict in dist_agent_1lm_randrot_nohyp[
    "monty_config"
].sensor_module_configs.values():
    sm_args = sm_dict["sensor_module_args"]
    if sm_args["sensor_module_id"] == "view_finder":
        continue
    sm_args["noise_params"] = {}  # Set noise_param to empty dictionary to remove noise
dist_agent_1lm_randrot_nohyp["logging_config"].run_name = "dist_agent_1lm_randrot_nohyp"

# Define dist_agent_1lm_randrot
dist_agent_1lm_randrot = copy.deepcopy(dist_agent_1lm_randrot_noise)
for sm_dict in dist_agent_1lm_randrot["monty_config"].sensor_module_configs.values():
    sm_args = sm_dict["sensor_module_args"]
    if sm_args["sensor_module_id"] == "view_finder":
        continue
    sm_args["noise_params"] = {}  # Set noise_param to empty dictionary to remove noise
dist_agent_1lm_randrot["logging_config"].run_name = "dist_agent_1lm_randrot"

# No Hypothesis Testing Config
# Here we use the default x-percent threshold of 20%.
# The update_x_percent_threshold_in_config function can be used to modify this
# and evaluate FLOPs and accuracy performance as a function of x-percent threshold.
dist_agent_1lm_randrot_nohyp = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp, 20
)
# Hypothesis Testing Config
dist_agent_1lm_randrot = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot, 20
)

###################
# Training Config #
###################

pretrain_dist_agent_1lm_k_none = copy.deepcopy(pretrain_dist_agent_1lm)

# Replace DisplacementGraphLM with EvidenceGraphLM
pretrain_dist_agent_1lm_k_none["monty_config"].learning_module_configs["learning_module_0"].update({
    "learning_module_args": dict(
        k=None,
    )
})

# Update the logging config run name
pretrain_dist_agent_1lm_k_none["logging_config"].run_name = "pretrain_dist_agent_1lm_k_none"

CONFIGS = {
    "dist_agent_1lm_randrot_nohyp": dist_agent_1lm_randrot_nohyp,
    "dist_agent_1lm_randrot": dist_agent_1lm_randrot,
    "pretrain_dist_agent_1lm_k_none": pretrain_dist_agent_1lm_k_none,
}
