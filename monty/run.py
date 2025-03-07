# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Entrypoint for running an experiment."""

import os

from tbp.monty.frameworks.config_utils.cmd_parser import create_cmd_parser
from tbp.monty.frameworks.run_env import setup_env

from configs.experiments import CONFIGS
from configs.names import NAMES

setup_env()

from tbp.monty.frameworks.run import main  # noqa: E402

if __name__ == "__main__":
    cmd_args = None
    cmd_parser = create_cmd_parser(experiments=NAMES)
    cmd_args = cmd_parser.parse_args()
    experiments = cmd_args.experiments

    if cmd_args.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    main(all_configs=CONFIGS, experiments=cmd_args.experiments)
