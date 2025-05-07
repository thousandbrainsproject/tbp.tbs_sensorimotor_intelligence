# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 7.

Panel B: Line Graphs Comparing Monty and ViT in Continual Learning
 - `plot_line_graphs()`

Running the above functions requires that the following experiments have been run:
Rapid Learning (Panel A)
    TODO

Continual Learning (Panels B and C)
 - For Monty, all experiments from `monty/configs/fig7b_continual_learning.py` have been run.
 - For ViT, all experiments from `pytorch/configs/fig7b_continual_learning` have been run.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_RESULTS_DIR,
    load_vit_predictions,
    aggregate_1lm_performance_data,
)
from plot_utils import (
    TBP_COLORS,
    init_matplotlib_style,
)
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS

init_matplotlib_style()


# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YCB_NUM_CLASSES = len(SHUFFLED_YCB_OBJECTS)

"""
--------------------------------------------------------------------------------
Utilities
--------------------------------------------------------------------------------
"""


def get_monty_continual_learning_accuracy(model_id: int) -> dict[int, float]:
    """Get accuracies for all classes up to model_id.

    Args:
        model_id: The model ID to get accuracies for.
        For example, model_id=23 is the model trained on classes 1 through 23.

    Returns:
        accuracies: dict mapping class IDs to accuracies.
    """
    assert (
        0 < model_id <= YCB_NUM_CLASSES
    ), f"Model ID {model_id} must be between 1 and {YCB_NUM_CLASSES}."

    exp = f"continual_learning_dist_agent_1lm_checkpoints_task{model_id}"
    perf_df = aggregate_1lm_performance_data([exp])

    accuracies = {}
    for class_id in range(model_id + 1):
        # Filter for this class and get accuracy
        class_df = perf_df[perf_df.index.str.contains(str(class_id))]
        accuracies[class_id] = class_df["accuracy"].iloc[0]

    return accuracies


def get_vit_continual_learning_accuracy(model_id: int) -> dict[int, float]:
    """Get the accuracies for VIT for each task in the continual learning experiment.

    Args:
        model_id: The model ID to get accuracies for.
        For example, model_id=23 is the model trained on classes 1 through 23.

    Returns:
        accuracies: dict mapping class IDs to accuracies.
    """
    assert (
        0 < model_id <= YCB_NUM_CLASSES
    ), f"Model ID {model_id} must be between 1 and {YCB_NUM_CLASSES}."
    accuracies = {}
    for model_id in range(1, YCB_NUM_CLASSES + 1):
        vit_predictions_path = Path(
            DMC_RESULTS_DIR
            / "vit"
            / "logs"
            / "fig7b_continual_learning"
            / "predictions"
            / f"predictions_model{model_id-1}.csv"
        )

        df = load_vit_predictions(vit_predictions_path)

        for class_id in range(model_id + 1):
            class_df = df[df["real_class"] == class_id]
            accuracy = (
                class_df["predicted_class"] == class_df["real_class"]
            ).mean() * 100
            accuracies[class_id] = accuracy

    return accuracies


"""
--------------------------------------------------------------------------------
Panel B: Line Graphs Comparing Monty and ViT in Continual Learning
--------------------------------------------------------------------------------
"""


def plot_line_graphs() -> None:
    """Plot line graphs comparing Monty and ViT in continual learning.

    For Monty, requires the experiments from `fig7b_continual_learning` have been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig7/continual_learning`.
    """
    out_dir = OUT_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    monty_average_accuracies = []
    vit_average_accuracies = []

    for num_objects_learned in range(YCB_NUM_CLASSES):
        # Calculate average accuracy for Monty
        monty_accuracy = get_monty_continual_learning_accuracy(num_objects_learned)
        monty_average_accuracies.append(
            np.mean(monty_accuracy) if monty_accuracy else np.nan
        )

        # Calculate average accuracy for ViT
        vit_accuracy = get_vit_continual_learning_accuracy(num_objects_learned)
        vit_average_accuracies.append(np.mean(vit_accuracy) if vit_accuracy else np.nan)

    # Plot line graphs
    axes_width, axes_height = 4, 3
    fig, ax1 = plt.subplots(1, 1, figsize=(axes_width, axes_height))

    ax1.plot(
        range(1, YCB_NUM_CLASSES + 1),
        monty_average_accuracies,
        color=TBP_COLORS["blue"],
        label="Monty",
        zorder=10,  # Ensure line for Monty is on top
        clip_on=False,  # Ensure line for Monty is not clipped when at 100% accuracy
    )
    ax1.plot(
        range(1, YCB_NUM_CLASSES + 1),
        vit_average_accuracies,
        color=TBP_COLORS["purple"],
        label="Pretrained ViT",
        zorder=5,  # Ensure line for ViT is on top
        clip_on=False,  # Ensure line for ViT is not clipped when at 100% accuracy
    )

    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Accuracy on All Observed Objects (%)")
    ax1.set_xlabel("Number of Objects Learned")
    xticks = [1, 10, 20, 30, 40, 50, 60, 70, 77]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks)
    ax1.legend()

    fig.savefig(out_dir / "performance.png", bbox_inches="tight")
    fig.savefig(out_dir / "performance.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_line_graphs()
