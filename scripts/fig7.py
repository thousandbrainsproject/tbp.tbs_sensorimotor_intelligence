# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 7.

Panel A: Rapid Learning Accuracy Comparison
 - `plot_rapid_learning_accuracy()`: Compares accuracy between Monty and ViT models
   with different numbers of rotations in training data.

Panel B: Line Graphs Comparing Monty and ViT in Continual Learning
 - `plot_line_graphs()`: Shows average accuracy as more objects are learned.

Panel C: Heatmaps of ViT and Monty Accuracies during Continual Learning
 - `plot_accuracy_heatmaps()`: Visualizes accuracy for each class at each model checkpoint.

Running the above functions requires that the following experiments have been run:

1. Rapid Learning (Panel A)
   - For Monty: All experiments from `rapid_learning/dist_agent_1lm_randrot_nohyp_*rot_trained`
   - For ViT: All experiments from `fig7a_vit-b16-224-in21k_*` with different rotation counts

2. Continual Learning (Panels B and C)
   - For Monty: All experiments from `monty/configs/fig7b_continual_learning.py`
   - For ViT: All experiments from `pytorch/configs/fig7b_continual_learning`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS

try:
    from .data_utils import (
        DMC_ANALYSIS_DIR,
        DMC_RESULTS_DIR,
        aggregate_1lm_performance_data,
        load_vit_predictions,
    )
    from .plot_utils import (
        TBP_COLORS,
        init_matplotlib_style,
    )
except ImportError:
    from data_utils import (
        DMC_ANALYSIS_DIR,
        DMC_RESULTS_DIR,
        aggregate_1lm_performance_data,
        load_vit_predictions,
    )
    from plot_utils import (
        TBP_COLORS,
        init_matplotlib_style,
    )

init_matplotlib_style()


# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YCB_NUM_CLASSES = len(SHUFFLED_YCB_OBJECTS)

# Type aliases for color specifications
RGB = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]
ColorSpec = Union[str, RGB, RGBA]

"""
--------------------------------------------------------------------------------
Utilities
--------------------------------------------------------------------------------
"""


def get_monty_continual_learning_accuracy(model_id: int) -> Dict[int, float]:
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


def get_vit_continual_learning_accuracy(model_id: int) -> Dict[int, float]:
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


def create_black_to_color_gradient_colormap(
    color: ColorSpec,
) -> LinearSegmentedColormap:
    """Create a custom colormap that transitions from black to a specified color with a lighter variant.

    This function creates a three-color gradient colormap that transitions from black to the specified
    color, and then to a lighter version of that color with 50% opacity. This is useful for creating
    visually appealing heatmaps with a dark-to-light progression.

    Args:
        color: The target color to fade to from black. Can be specified as:
            - A string color name (e.g., 'blue', 'red')
            - An RGB tuple (r, g, b) with values between 0 and 1
            - An RGBA tuple (r, g, b, a) with values between 0 and 1

    Returns:
        LinearSegmentedColormap: A custom colormap that transitions from black to the specified color
        to a lighter version of that color.
    """
    # Convert color to RGBA and create a lighter version with 50% opacity
    rgba = mcolors.to_rgba(color)
    lighter_color = (rgba[0], rgba[1], rgba[2], 0.5)
    return LinearSegmentedColormap.from_list("custom", ["black", color, lighter_color])


def load_vit_rapid_learning_predictions(
    data_dir: Path,
    num_rotations: list[int],
    experiment_type: str,
) -> dict[int, float]:
    """Load ViT predictions for rapid learning experiments.
    
    Args:
        data_dir: Base directory containing ViT logs
        num_rotations: List of rotation counts to load
        experiment_type: Type of experiment ("1epoch", "25epochs", "1epoch_random_init", "75epochs_random_init")
        
    Returns:
        Dictionary mapping number of rotations to accuracy
    """
    # Select paths based on experiment type
    if experiment_type == "1epoch":
        paths = [
            data_dir / f"fig7a_vit-b16-224-in21k_1epoch_{i}rot/inference/predictions.csv"
            for i in num_rotations
        ]
    elif experiment_type == "25epochs":
        paths = [
            data_dir / f"fig7a_vit-b16-224-in21k_25epochs_{i}rot/inference/predictions.csv"
            for i in num_rotations
        ]
    elif experiment_type == "1epoch_random_init":
        paths = [
            data_dir / f"fig7a_vit-b16-224-in21k_1epoch_{i}rot_random_init/inference/predictions.csv"
            for i in num_rotations
        ]
    elif experiment_type == "75epochs_random_init":
        paths = [
            data_dir / f"fig7a_vit-b16-224-in21k_75epochs_{i}rot_random_init/inference/predictions.csv"
            for i in num_rotations
        ]
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Load data and calculate accuracies
    dfs = []
    accuracies = {}
    
    for i, path in enumerate(paths):
        if path.exists():
            df = load_vit_predictions(path)
            df["num_rotations"] = num_rotations[i]
            dfs.append(df)
            # Calculate accuracy for this rotation count
            accuracy = (df["real_class"] == df["predicted_class"]).mean() * 100
            accuracies[num_rotations[i]] = accuracy
        else:
            print(f"Warning: Path not found: {path}")
    
    return accuracies


def get_monty_rotation_error_data(df: pd.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    """Get rotation error data for Monty, including mean and std.
    
    Args:
        df: DataFrame containing Monty rotation error data
        
    Returns:
        Tuple of (means, stds) dictionaries mapping number of rotations to error values
    """
    means = {}
    stds = {}
    
    for num_rotations in df["num_rotations"].unique():
        df_subset = df[df["num_rotations"] == num_rotations]
        
        # Calculate error for correct samples only
        correct_subset = df_subset[df_subset["primary_performance"].isin(["correct", "correct_mlh"])]
        correct_samples_error = correct_subset["rotation_error"].mean() * 180 / np.pi
        
        # Store the correct samples error for plotting
        means[num_rotations] = correct_samples_error
        stds[num_rotations] = correct_subset["rotation_error"].std() * 180 / np.pi
    
    return means, stds


def get_vit_rotation_error_data(df: pd.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    """Get rotation error data for ViT, including mean and std.
    
    Args:
        df: DataFrame containing ViT rotation error data
        
    Returns:
        Tuple of (means, stds) dictionaries mapping number of rotations to error values
    """
    means = {}
    stds = {}
    
    for num_rotations in df["num_rotations"].unique():
        df_subset = df[df["num_rotations"] == num_rotations]
        
        # Calculate error for correct samples only
        correct_subset = df_subset[df_subset["real_class"] == df_subset["predicted_class"]]
        correct_samples_error = correct_subset["quaternion_error_degs"].mean()
        
        # Store the correct samples error for plotting
        means[num_rotations] = correct_samples_error
        stds[num_rotations] = correct_subset["quaternion_error_degs"].std()
    
    return means, stds


"""
--------------------------------------------------------------------------------
Panel A (left): Rapid Learning Accuracy Plot
--------------------------------------------------------------------------------
"""

def plot_rapid_learning_accuracy() -> None:
    """Plot accuracy comparison between Monty and ViT in rapid learning.

    For Monty, requires the experiments from `rapid_learning` have been run.
    For ViT, requires the experiments from `fig7a_vit-b16-224-in21k_*` have been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig7/rapid_learning`.
    """
    out_dir = OUT_DIR / "rapid_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Monty data
    monty_experiments = [
        "dist_agent_1lm_randrot_nohyp_1rot_trained",
        "dist_agent_1lm_randrot_nohyp_2rot_trained",
        "dist_agent_1lm_randrot_nohyp_4rot_trained",
        "dist_agent_1lm_randrot_nohyp_8rot_trained",
        "dist_agent_1lm_randrot_nohyp_16rot_trained",
        "dist_agent_1lm_randrot_nohyp_32rot_trained",
    ]
    monty_data = aggregate_1lm_performance_data(monty_experiments)
    
    # Extract actual rotation counts from experiment names
    import re
    monty_accuracies = {}
    for i, exp in enumerate(monty_experiments):
        match = re.search(r'_(\d+)rot_', exp)
        if match:
            num_rots = int(match.group(1))
            monty_accuracies[num_rots] = monty_data.iloc[i]["accuracy"]

    # Load ViT data
    vit_data_dir = DMC_RESULTS_DIR / "vit" / "logs"
    num_rotations = [1, 2, 4, 8, 16, 32]

    # Load ViT from pretrained 25 epochs
    vit_pretrained_25epochs_accuracies = load_vit_rapid_learning_predictions(
        vit_data_dir, num_rotations, "25epochs"
    )

    # Load ViT trained from random initialization trained for 75 epochs
    vit_from_random_init_75epochs_accuracies = load_vit_rapid_learning_predictions(
        vit_data_dir, num_rotations, "75epochs_random_init"
    )

    # Load ViT trained from random initialization trained for 1 epoch
    vit_from_random_init_1epoch_accuracies = load_vit_rapid_learning_predictions(
        vit_data_dir, num_rotations, "1epoch_random_init"
    )

    # Create figure
    axes_width, axes_height = 5, 3
    fig, ax = plt.subplots(figsize=(axes_width, axes_height))

    # Plot Monty accuracy (blue)
    ax.plot(
        list(monty_accuracies.keys()),
        list(monty_accuracies.values()),
        marker="o",
        color=TBP_COLORS["blue"],
        label="Monty",
        zorder=10,
    )

    # Plot ViT from pretrained accuracy (solid purple)
    ax.plot(
        list(vit_pretrained_25epochs_accuracies.keys()),
        list(vit_pretrained_25epochs_accuracies.values()),
        marker="o",
        color=TBP_COLORS["purple"],
        label="Pretrained ViT (25 Epochs)",
        zorder=10,
    )

    # Plot ViT from scratch accuracy (solid yellow)
    ax.plot(
        list(vit_from_random_init_75epochs_accuracies.keys()),
        list(vit_from_random_init_75epochs_accuracies.values()),
        marker="o",
        color=TBP_COLORS["yellow"],
        label="ViT (75 Epochs)",
        zorder=10,
    )

    # Plot ViT from scratch accuracy (dashed yellow)
    ax.plot(
        list(vit_from_random_init_1epoch_accuracies.keys()),
        list(vit_from_random_init_1epoch_accuracies.values()),
        marker="o",
        color=TBP_COLORS["yellow"],
        linestyle="--",
        label="ViT (1 Epoch)",
        zorder=10,
    )

    # Set x-axis tickmarks and limits
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xlabel("Number of Rotations in Training Data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

    # Set legend outside to the right
    monty_line = mlines.Line2D([], [], color=TBP_COLORS["blue"], linestyle="-", label="Monty (1 Epoch)")
    vit_pretrained_25epochs_line = mlines.Line2D([], [], color=TBP_COLORS["purple"], linestyle="-", label="Pretrained ViT (25 Epochs)")
    vit_from_random_init_75epochs_line = mlines.Line2D([], [], color=TBP_COLORS["yellow"], linestyle="-", label="ViT (75 Epochs)")
    vit_from_random_init_1epoch_line = mlines.Line2D([], [], color=TBP_COLORS["yellow"], linestyle="--", label="ViT (1 Epoch)")
    ax.legend(
        handles=[monty_line, vit_pretrained_25epochs_line, vit_from_random_init_1epoch_line, vit_from_random_init_75epochs_line],
        fontsize=7,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    # Save data as CSV
    sorted_rotations = sorted(monty_accuracies.keys())
    rapid_learning_data = pd.DataFrame({
        "num_rotations": sorted_rotations,
        "monty_accuracy": [monty_accuracies[k] for k in sorted_rotations],
        "vit_pretrained_25epochs_accuracy": [vit_pretrained_25epochs_accuracies.get(k, np.nan) for k in sorted_rotations],
        "vit_from_scratch_75epochs_accuracy": [vit_from_random_init_75epochs_accuracies.get(k, np.nan) for k in sorted_rotations],
        "vit_from_scratch_1epoch_accuracy": [vit_from_random_init_1epoch_accuracies.get(k, np.nan) for k in sorted_rotations],
    })
    rapid_learning_data.to_csv(out_dir / "fig7a_rapid_learning_accuracy_data.csv", index=False)

    # Save figures
    fig.savefig(out_dir / "rapid_learning_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "rapid_learning_accuracy.svg", bbox_inches="tight")
    plt.show()

"""
--------------------------------------------------------------------------------
Panel A (right): Rotation Error Plot
--------------------------------------------------------------------------------
"""

def plot_rotation_error() -> None:
    """Plot rotation error comparison between Monty and ViT in rapid learning.

    For Monty, requires the experiments from `rapid_learning` have been run.
    For ViT, requires the experiments from `fig7a_vit-b16-224-in21k_*` have been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig7/rapid_learning`.
    """
    out_dir = OUT_DIR / "rapid_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Monty data
    monty_experiments = [
        "dist_agent_1lm_randrot_nohyp_1rot_trained",
        "dist_agent_1lm_randrot_nohyp_2rot_trained",
        "dist_agent_1lm_randrot_nohyp_4rot_trained",
        "dist_agent_1lm_randrot_nohyp_8rot_trained",
        "dist_agent_1lm_randrot_nohyp_16rot_trained",
        "dist_agent_1lm_randrot_nohyp_32rot_trained",
    ]
    monty_data = aggregate_1lm_performance_data(monty_experiments)
    
    # Extract rotation error means from aggregated data
    import re
    monty_means = {}
    
    for i, exp in enumerate(monty_experiments):
        # Extract number of rotations from experiment name
        match = re.search(r'_(\d+)rot_', exp)
        if match:
            num_rots = int(match.group(1))
            # rotation_error.mean is already in degrees
            monty_means[num_rots] = monty_data.iloc[i]["rotation_error.mean"]

    # Load ViT data
    vit_data_dir = DMC_RESULTS_DIR / "vit" / "logs"
    num_rotations = [1, 2, 4, 8, 16, 32]

    # Load ViT from pretrained 25 epochs
    vit_pretrained_25epochs_df = pd.concat([
        load_vit_predictions(vit_data_dir / f"fig7a_vit-b16-224-in21k_25epochs_{i}rot/inference/predictions.csv")
        for i in num_rotations
    ])
    vit_pretrained_25epochs_df["num_rotations"] = np.repeat(num_rotations, len(vit_pretrained_25epochs_df) // len(num_rotations))
    vit_pretrained_25epochs_means, _ = get_vit_rotation_error_data(vit_pretrained_25epochs_df)

    # Load ViT trained from random initialization trained for 75 epochs
    vit_from_random_init_75epochs_df = pd.concat([
        load_vit_predictions(vit_data_dir / f"fig7a_vit-b16-224-in21k_75epochs_{i}rot_random_init/inference/predictions.csv")
        for i in num_rotations
    ])
    vit_from_random_init_75epochs_df["num_rotations"] = np.repeat(num_rotations, len(vit_from_random_init_75epochs_df) // len(num_rotations))
    vit_from_random_init_75epochs_means, _ = get_vit_rotation_error_data(vit_from_random_init_75epochs_df)

    # Load ViT trained from random initialization trained for 1 epoch
    vit_from_random_init_1epoch_df = pd.concat([
        load_vit_predictions(vit_data_dir / f"fig7a_vit-b16-224-in21k_1epoch_{i}rot_random_init/inference/predictions.csv")
        for i in num_rotations
    ])
    vit_from_random_init_1epoch_df["num_rotations"] = np.repeat(num_rotations, len(vit_from_random_init_1epoch_df) // len(num_rotations))
    vit_from_random_init_1epoch_means, _ = get_vit_rotation_error_data(vit_from_random_init_1epoch_df)

    # Create figure
    axes_width, axes_height = 5, 3
    fig, ax = plt.subplots(figsize=(axes_width, axes_height))

    # Plot Monty rotation error (blue)
    monty_x = sorted(monty_means.keys())
    monty_y = [monty_means[x] for x in monty_x]
    ax.plot(
        monty_x,
        monty_y,
        marker="o",
        color=TBP_COLORS["blue"],
        label="Monty",
        zorder=10,
    )

    # Plot ViT from pretrained rotation error (solid purple)
    vit_pretrained_x = sorted(vit_pretrained_25epochs_means.keys())
    vit_pretrained_y = [vit_pretrained_25epochs_means[x] for x in vit_pretrained_x]
    ax.plot(
        vit_pretrained_x,
        vit_pretrained_y,
        marker="o",
        color=TBP_COLORS["purple"],
        label="Pretrained ViT (25 Epochs)",
        zorder=10,
    )

    # Plot ViT from scratch accuracy (solid yellow)
    vit_75epochs_x = sorted(vit_from_random_init_75epochs_means.keys())
    vit_75epochs_y = [vit_from_random_init_75epochs_means[x] for x in vit_75epochs_x]
    ax.plot(
        vit_75epochs_x,
        vit_75epochs_y,
        marker="o",
        color=TBP_COLORS["yellow"],
        label="ViT (75 Epochs)",
        zorder=10,
    )

    # Plot ViT from scratch accuracy (dashed yellow)
    vit_1epoch_x = sorted(vit_from_random_init_1epoch_means.keys())
    vit_1epoch_y = [vit_from_random_init_1epoch_means[x] for x in vit_1epoch_x]
    ax.plot(
        vit_1epoch_x,
        vit_1epoch_y,
        marker="o",
        color=TBP_COLORS["yellow"],
        linestyle="--",
        label="ViT (1 Epoch)",
        zorder=10,
    )

    # Set x-axis tickmarks and limits
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xlabel("Number of Rotations in Training Data")
    ax.set_ylabel("Rotation Error (deg)")
    ax.set_ylim(0, 180)

    # Set legend outside to the right
    monty_line = mlines.Line2D([], [], color=TBP_COLORS["blue"], linestyle="-", label="Monty (1 Epoch)")
    vit_pretrained_25epochs_line = mlines.Line2D([], [], color=TBP_COLORS["purple"], linestyle="-", label="Pretrained ViT (25 Epochs)")
    vit_from_random_init_75epochs_line = mlines.Line2D([], [], color=TBP_COLORS["yellow"], linestyle="-", label="ViT (75 Epochs)")
    vit_from_random_init_1epoch_line = mlines.Line2D([], [], color=TBP_COLORS["yellow"], linestyle="--", label="ViT (1 Epoch)")
    ax.legend(
        handles=[monty_line, vit_pretrained_25epochs_line, vit_from_random_init_1epoch_line, vit_from_random_init_75epochs_line],
        fontsize=7,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    # Save data as CSV
    sorted_rotations = sorted(monty_means.keys())
    rotation_error_data = pd.DataFrame({
        "num_rotations": sorted_rotations,
        "monty_rotation_error_mean": [monty_means[k] for k in sorted_rotations],
        "vit_pretrained_25epochs_rotation_error_mean": [vit_pretrained_25epochs_means.get(k, np.nan) for k in sorted_rotations],
        "vit_from_scratch_75epochs_rotation_error_mean": [vit_from_random_init_75epochs_means.get(k, np.nan) for k in sorted_rotations],
        "vit_from_scratch_1epoch_rotation_error_mean": [vit_from_random_init_1epoch_means.get(k, np.nan) for k in sorted_rotations],
    })
    rotation_error_data.to_csv(out_dir / "fig7a_rapid_learning_rotation_error_data.csv", index=False)

    # Save figures
    fig.savefig(out_dir / "rapid_learning_rotation_error.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "rapid_learning_rotation_error.svg", bbox_inches="tight")
    plt.show()


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


"""
--------------------------------------------------------------------------------
Panel C: Heatmaps of ViT and Monty Accuracies during Continual Learning
--------------------------------------------------------------------------------
"""


def plot_accuracy_heatmaps() -> None:
    """Plot heatmaps showing both Monty and ViT's accuracy for each class at each model checkpoint.

    Output is saved to `DMC_ANALYSIS_DIR/fig7/continual_learning`.
    """

    out_dir = OUT_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create the heatmap matrices and fill them with accuracies
    monty_heatmap = np.full((YCB_NUM_CLASSES, YCB_NUM_CLASSES), np.nan)
    vit_heatmap = np.full((YCB_NUM_CLASSES, YCB_NUM_CLASSES), np.nan)

    for model_id in range(1, YCB_NUM_CLASSES + 1):
        monty_accuracy = get_monty_continual_learning_accuracy(model_id)
        for class_id, accuracy in monty_accuracy.items():
            monty_heatmap[model_id - 1, class_id] = accuracy

    for model_id in range(1, YCB_NUM_CLASSES + 1):
        vit_accuracy = get_vit_continual_learning_accuracy(model_id)
        for class_id, accuracy in vit_accuracy.items():
            vit_heatmap[model_id - 1, class_id] = accuracy

    # Create the plot with two subplots
    axes_width, axes_height = 6.5, 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(axes_width, axes_height))
    colormap = create_black_to_color_gradient_colormap(TBP_COLORS["green"])

    # Create the heatmaps using custom colormap
    _ = ax1.imshow(monty_heatmap, cmap=colormap, vmin=0, vmax=100)
    im2 = ax2.imshow(vit_heatmap, cmap=colormap, vmin=0, vmax=100)

    # Plot Params
    ax1.set_xlabel("Target Object")
    ax1.set_ylabel("Number of Objects Learned")
    ax1.set_title("Monty")

    ax2.set_xlabel("Target Object")
    ax2.set_ylabel("Number of Objects Learned")
    ax2.set_title("ViT")

    for ax in [ax1, ax2]:
        ticks = [1, 10, 20, 30, 40, 50, 60, 70, 77]
        ax.set_xticks([t - 1 for t in ticks])
        ax.set_yticks([t - 1 for t in ticks])
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)

    # Adjust the subplot layout to make room for the colorbar
    plt.subplots_adjust(left=0.08, right=0.80, wspace=0.3)

    # Add colorbar
    cax = fig.add_axes([0.82, 0.52, 0.02, 0.3])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Accuracy (%)", labelpad=5, y=0.5, fontsize=8)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("top")

    fig.savefig(out_dir / "accuracy_heatmaps.png", bbox_inches="tight")
    fig.savefig(out_dir / "accuracy_heatmaps.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # plot_line_graphs()
    # plot_accuracy_heatmaps()
    plot_rapid_learning_accuracy()
    plot_rotation_error()
