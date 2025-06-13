# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import List, Tuple


def compute_class_ranges(
    num_classes_for_task: int, task_id: int
) -> Tuple[int, int, List[int], List[int]]:
    """Compute class ranges for continual learning tasks.

    This function calculates the starting and ending class indices for a given task,
    as well as the current task classes and all classes seen so far.

    Args:
        num_classes_for_task (int): The number of classes for the current task.
        task_id (int): The ID of the current task.

    Returns:
        Tuple[int, int, List[int], List[int]]: A tuple containing:
            - start_idx: The starting class index for the current task
            - end_idx: The ending class index for the current task
            - current_task_classes: List of class indices for the current task
            - all_seen_classes: List of all class indices seen so far (up to and including current task)
    """
    # Calculate the starting class index based on the sum of previous tasks' classes
    start_idx = num_classes_for_task * task_id
    end_idx = start_idx + num_classes_for_task

    # Compute the current task classes and all seen classes
    current_task_classes = list(range(start_idx, end_idx))
    all_seen_classes = list(range(0, end_idx))

    return start_idx, end_idx, current_task_classes, all_seen_classes
