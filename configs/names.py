# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""The names of declared experiments.

This module keeps experiment names separate from the configuration for the
experiments. The reason for doing this is so that we can import the configurations
selectively to avoid importing uninstalled dependencies (e.g., not installing a
specific simulator). While this feature may not be needed for the current project,
the structure is retained here to make it look similar to tbp.monty setup.

The use of dataclasses assists in raising early errors when experiment names defined
here and the corresponding experiment configurations drift apart. For additional
discussion, see: https://github.com/thousandbrainsproject/tbp.monty/pull/153.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

NAMES: list[str] = []


@dataclass
class Experiments:
    """Class declaring all experiment names."""

    # Add your experiment names here
    # e.g.: my_experiment: dict[str, Any]


NAMES.extend(field.name for field in fields(Experiments))
