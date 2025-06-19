#!/bin/bash

# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Script to run all continual learning experiments
# This script first runs task0 and then tasks 1-76 using task1+.yaml

echo "Starting continual learning experiments..."

# Function to handle errors
handle_error() {
    local task_id="$1"
    echo "Error: Training failed for task_id = ${task_id}"
    echo "Please check the logs for more details"
    exit 1
}

# Run initial task (task0) with task0.yaml
echo "Running initial task (task0)..."
python src/train.py experiment=03_fig7b_continual_learning/train/task0.yaml "task_id=0" || handle_error "0"
echo "Successfully completed task 0"

# Run subsequent tasks (1-76) with task1+.yaml
echo "Running subsequent tasks (1-76)..."

for task_id in $(seq 1 76); do
    echo "Running training for task_id = ${task_id}"
    python src/train.py experiment=03_fig7b_continual_learning/train/task1+.yaml "task_id=${task_id}" || handle_error "${task_id}"
    echo "Successfully completed task ${task_id}"

    # Optional: add a small delay between tasks to allow system resources to settle
    sleep 2
done

echo "All continual learning tasks completed successfully!"
