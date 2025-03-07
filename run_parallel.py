# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Entrypoint for running an experiment with episodes executing in parallel."""

import os

from tbp.monty.frameworks.config_utils.cmd_parser import create_cmd_parser_parallel
from tbp.monty.frameworks.run_env import setup_env

from configs.experiments import CONFIGS
from configs.names import NAMES

setup_env()

from tbp.monty.frameworks.run_parallel import main  # noqa: E402

if __name__ == "__main__":
    cmd_args = None
    cmd_parser = create_cmd_parser_parallel(experiments=NAMES)
    cmd_args = cmd_parser.parse_args()
    experiment = cmd_args.experiment
    num_parallel = cmd_args.num_parallel
    quiet_habitat_logs = cmd_args.quiet_habitat_logs
    print_cfg = cmd_args.print_cfg
    is_unittest = False

    if quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    main(
        all_configs=CONFIGS,
        exp=CONFIGS[experiment],
        experiment=experiment,
        num_parallel=num_parallel,
        quiet_habitat_logs=quiet_habitat_logs,
        print_cfg=print_cfg,
        is_unittest=is_unittest,
    )
