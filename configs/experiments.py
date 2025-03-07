# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Module for the paper's experiment configurations."""

from dataclasses import asdict

from configs.names import Experiments

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)

experiments = Experiments(
    # For each experiment name in Experiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
)
CONFIGS = asdict(experiments)
