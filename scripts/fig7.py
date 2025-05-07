# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 7.

Panel B: Continual Learning
 - `plot_continual_learning()`

TODO (Hojae): Please update description below.
Running the above functions requires that the following experiments have been run:
 - `fig3_evidence_run`: For plotting the sensor path and evidence graphs + patches.
 - `dist_agent_1lm`, `dist_agent_1lm_noise`, `dist_agent_1lm_randrot_all`, and
   `dist_agent_1lm_randrot_all_noise`: For plotting the performance metrics.
 - `pretrain_dist_agent_1lm`: For plotting the known objects.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from data_utils import (
    DMC_ANALYSIS_DIR,
    load_eval_stats,
    load_vit_predictions,
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


def object_names_from_ids(ids: list[int]) -> list[str]:
    """Get the object names for a list of object IDs."""
    alphabetically_sorted_ycb_objects = sorted(SHUFFLED_YCB_OBJECTS)
    return [alphabetically_sorted_ycb_objects[id] for id in ids]


def get_monty_accuracies_for_continual_learning():
    """Get the accuracies for Monty for each task in the continual learning experiment.

    Returns:
        dict[int, dict[int, float]]: A dictionary mapping model IDs to dictionaries
        that map class IDs to accuracies.
    """
    # Load the stats.
    num_tasks = 77
    accuracies = {}
    for model_id in range(num_tasks):
        try:
            eval_stats_df = load_eval_stats(
                f"continual_learning_dist_agent_1lm_checkpoints_task{model_id}",
                add_epoch=False,
                add_episode=False,
            )
            model_accuracies = {}
            object_names = object_names_from_ids(range(model_id + 1))
            # For each class up to the current model's ID
            for class_id in range(model_id + 1):
                # Filter for this class
                class_df = eval_stats_df[
                    eval_stats_df["primary_target_object"] == object_names[class_id]
                ]
                if len(class_df) > 0:
                    # Calculate accuracy for this class
                    correct_df = class_df[
                        class_df.primary_performance.isin(["correct", "correct_mlh"])
                    ]
                    model_accuracies[class_id] = 100 * len(correct_df) / len(class_df)
                else:
                    model_accuracies[class_id] = np.nan
            accuracies[model_id + 1] = model_accuracies
        except FileNotFoundError:
            print(f"No stats found for model {model_id}")
            accuracies[model_id + 1] = {
                class_id: np.nan for class_id in range(model_id + 1)
            }
    return accuracies


def get_vit_accuracies_for_continual_learning():
    """Get the accuracies for VIT for each task in the continual learning experiment.

    Returns:
        dict[int, dict[int, float]]: A dictionary mapping model IDs to dictionaries
        that map class IDs to accuracies.
    """
    num_tasks = 77
    accuracies = {}
    for model_id in range(num_tasks):
        vit_predictions_path = Path(
            f"/Users/hlee/tbp/results/dmc/results/vit/logs/fig7b_continual_learning/predictions/predictions_model{model_id}.csv"
        )
        if not vit_predictions_path.exists():
            print(f"No predictions found for model {model_id}")
            accuracies[model_id + 1] = {
                class_id: np.nan for class_id in range(model_id + 1)
            }
            continue

        df = load_vit_predictions(vit_predictions_path)
        model_accuracies = {}

        # For each class up to the current model's ID
        for class_id in range(model_id + 1):
            # Filter for this class
            class_df = df[df["real_class"] == class_id]
            if len(class_df) > 0:
                accuracy = (
                    class_df["predicted_class"] == class_df["real_class"]
                ).mean() * 100
                model_accuracies[class_id] = accuracy
            else:
                model_accuracies[class_id] = np.nan

        accuracies[model_id + 1] = model_accuracies
    return accuracies


def plot_performance() -> None:
    """Plot core performance metrics.

    For Monty, requires the experiments from `fig7b_continual_learning` have been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig7/continual_learning/`.
    """
    # Initialize output paths.
    out_dir = OUT_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    monty_accuracies = get_monty_accuracies_for_continual_learning()
    vit_accuracies = get_vit_accuracies_for_continual_learning()

    # Initialize the plot.
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 3))

    # Extract and plot average accuracies for each model
    monty_avg_accuracies = []
    vit_avg_accuracies = []

    for model_id in range(len(monty_accuracies)):
        # Calculate average accuracy for Monty
        monty_model_accs = [
            acc for acc in monty_accuracies[model_id + 1].values() if not np.isnan(acc)
        ]
        monty_avg_accuracies.append(
            np.mean(monty_model_accs) if monty_model_accs else np.nan
        )

        # Calculate average accuracy for ViT
        vit_model_accs = [
            acc for acc in vit_accuracies[model_id + 1].values() if not np.isnan(acc)
        ]
        vit_avg_accuracies.append(np.mean(vit_model_accs) if vit_model_accs else np.nan)

    # Plot accuracy for Monty and ViT
    ax1.plot(
        range(1, len(monty_avg_accuracies) + 1),
        monty_avg_accuracies,
        color=TBP_COLORS["blue"],
        label="Monty",
        zorder=10,
        clip_on=False,
    )
    ax1.plot(
        range(1, len(vit_avg_accuracies) + 1),
        vit_avg_accuracies,
        color=TBP_COLORS["purple"],
        label="Pretrained ViT",
        zorder=5,
        clip_on=False,
    )

    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Accuracy on All Observed Objects (%)")
    ax1.set_xlabel("Number of Objects Learned")
    xticks = [1, 10, 20, 30, 40, 50, 60, 70, 77]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks)
    ax1.legend(frameon=False)

    fig.tight_layout()

    print(f"Saving to {out_dir / 'performance.png'}")
    fig.savefig(out_dir / "performance.png")
    fig.savefig(out_dir / "performance.svg")
    # plt.show()


def create_custom_colormap(color, use_three_colors=False):
    """Create a custom colormap from black to the specified color.

    Args:
        color: The target color to fade to from black
        use_three_colors: If True, creates a three-color gradient from black to color to lighter color

    Returns:
        LinearSegmentedColormap: A custom colormap
    """
    if use_three_colors:
        # Convert color to RGBA and create a lighter version with 30% opacity
        import matplotlib.colors as mcolors

        rgba = mcolors.to_rgba(color)
        lighter_color = (rgba[0], rgba[1], rgba[2], 0.5)
        return LinearSegmentedColormap.from_list(
            "custom", ["black", color, lighter_color]
        )
    else:
        return LinearSegmentedColormap.from_list("custom", ["black", color])


def plot_accuracy_heatmaps_blue():
    """Plot heatmaps showing both Monty and ViT's accuracy for each class at each model checkpoint.
    Uses a custom colormap from black to TBP blue.
    """
    # Initialize output directory
    out_dir = OUT_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the accuracies for both models
    monty_accuracies = get_monty_accuracies_for_continual_learning()
    vit_accuracies = get_vit_accuracies_for_continual_learning()

    # Create the heatmap matrices
    num_models = len(monty_accuracies)
    monty_heatmap = np.full((num_models, num_models), np.nan)
    vit_heatmap = np.full((num_models, num_models), np.nan)

    for model_id, model_accuracies in monty_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            monty_heatmap[model_id - 1, class_id] = accuracy

    for model_id, model_accuracies in vit_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            vit_heatmap[model_id - 1, class_id] = accuracy

    # Create custom colormap
    custom_cmap = create_custom_colormap(TBP_COLORS["blue"])

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))

    # Create the heatmaps using custom colormap
    im1 = ax1.imshow(monty_heatmap, cmap=custom_cmap, vmin=0, vmax=100)
    im2 = ax2.imshow(vit_heatmap, cmap=custom_cmap, vmin=0, vmax=100)

    # Set labels and titles
    ax1.set_xlabel("Target Object")
    ax1.set_ylabel("Number of Objects Learned")
    ax1.set_title("Monty")

    ax2.set_xlabel("Target Object")
    ax2.set_ylabel("Number of Objects Learned")
    ax2.set_title("ViT")

    # Set ticks
    for ax in [ax1, ax2]:
        ticks = [0, 9, 19, 29, 39, 49, 59, 69, 76]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([t + 1 for t in ticks])
        ax.set_yticklabels([t + 1 for t in ticks])

    # Adjust the subplot layout to make room for the colorbar
    plt.subplots_adjust(left=0.08, right=0.80, wspace=0.3)

    # Add colorbar
    cax = fig.add_axes([0.82, 0.52, 0.02, 0.3])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Accuracy (%)", labelpad=5, y=0.5, fontsize=8)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("top")

    fig.savefig(out_dir / "accuracy_heatmaps_blue.png", bbox_inches="tight")
    fig.savefig(out_dir / "accuracy_heatmaps_blue.svg", bbox_inches="tight")


def plot_accuracy_heatmaps_green():
    """Plot heatmaps showing both Monty and ViT's accuracy for each class at each model checkpoint.
    Uses a custom colormap from black to TBP green.
    """
    # Initialize output directory
    out_dir = OUT_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the accuracies for both models
    monty_accuracies = get_monty_accuracies_for_continual_learning()
    vit_accuracies = get_vit_accuracies_for_continual_learning()

    # Create the heatmap matrices
    num_models = len(monty_accuracies)
    monty_heatmap = np.full((num_models, num_models), np.nan)
    vit_heatmap = np.full((num_models, num_models), np.nan)

    for model_id, model_accuracies in monty_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            monty_heatmap[model_id - 1, class_id] = accuracy

    for model_id, model_accuracies in vit_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            vit_heatmap[model_id - 1, class_id] = accuracy

    # Create custom colormap
    custom_cmap = create_custom_colormap(TBP_COLORS["green"])

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))

    # Create the heatmaps using custom colormap
    im1 = ax1.imshow(monty_heatmap, cmap=custom_cmap, vmin=0, vmax=100)
    im2 = ax2.imshow(vit_heatmap, cmap=custom_cmap, vmin=0, vmax=100)

    # Set labels and titles
    ax1.set_xlabel("Target Object")
    ax1.set_ylabel("Number of Objects Learned")
    ax1.set_title("Monty")

    ax2.set_xlabel("Target Object")
    ax2.set_ylabel("Number of Objects Learned")
    ax2.set_title("ViT")

    # Set ticks
    for ax in [ax1, ax2]:
        ticks = [0, 9, 19, 29, 39, 49, 59, 69, 76]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([t + 1 for t in ticks])
        ax.set_yticklabels([t + 1 for t in ticks])

    # Adjust the subplot layout to make room for the colorbar
    plt.subplots_adjust(left=0.08, right=0.80, wspace=0.3)

    # Add colorbar
    cax = fig.add_axes([0.82, 0.52, 0.02, 0.3])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Accuracy (%)", labelpad=5, y=0.5, fontsize=8)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("top")

    fig.savefig(out_dir / "accuracy_heatmaps_green.png", bbox_inches="tight")
    fig.savefig(out_dir / "accuracy_heatmaps_green.svg", bbox_inches="tight")


def plot_accuracy_heatmaps_green_light():
    """Plot heatmaps showing both Monty and ViT's accuracy for each class at each model checkpoint.
    Uses a custom colormap from black to TBP green to light TBP green.
    """
    # Initialize output directory
    out_dir = OUT_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the accuracies for both models
    monty_accuracies = get_monty_accuracies_for_continual_learning()
    vit_accuracies = get_vit_accuracies_for_continual_learning()

    # Create the heatmap matrices
    num_models = len(monty_accuracies)
    monty_heatmap = np.full((num_models, num_models), np.nan)
    vit_heatmap = np.full((num_models, num_models), np.nan)

    for model_id, model_accuracies in monty_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            monty_heatmap[model_id - 1, class_id] = accuracy

    for model_id, model_accuracies in vit_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            vit_heatmap[model_id - 1, class_id] = accuracy

    # Create custom colormap with three colors
    custom_cmap = create_custom_colormap(TBP_COLORS["green"], use_three_colors=True)

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))

    # Create the heatmaps using custom colormap
    im1 = ax1.imshow(monty_heatmap, cmap=custom_cmap, vmin=0, vmax=100)
    im2 = ax2.imshow(vit_heatmap, cmap=custom_cmap, vmin=0, vmax=100)

    # Set labels and titles
    ax1.set_xlabel("Target Object")
    ax1.set_ylabel("Number of Objects Learned")
    ax1.set_title("Monty")

    ax2.set_xlabel("Target Object")
    ax2.set_ylabel("Number of Objects Learned")
    ax2.set_title("ViT")

    # Set ticks
    for ax in [ax1, ax2]:
        ticks = [0, 9, 19, 29, 39, 49, 59, 69, 76]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([t + 1 for t in ticks])
        ax.set_yticklabels([t + 1 for t in ticks])

    # Adjust the subplot layout to make room for the colorbar
    plt.subplots_adjust(left=0.08, right=0.80, wspace=0.3)

    # Add colorbar
    cax = fig.add_axes([0.82, 0.52, 0.02, 0.3])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Accuracy (%)", labelpad=5, y=0.5, fontsize=8)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("top")

    fig.savefig(out_dir / "accuracy_heatmaps_green_light.png", bbox_inches="tight")
    fig.savefig(out_dir / "accuracy_heatmaps_green_light.svg", bbox_inches="tight")


def plot_accuracy_heatmaps():
    """Plot heatmaps showing both Monty and ViT's accuracy for each class at each model checkpoint.

    The heatmaps will be lower triangular matrices where:
    - Rows represent model checkpoints (0 to 76)
    - Columns represent classes (0 to 76)
    - Each cell shows the accuracy of that model on that class
    - Only the lower triangle is filled since models can't be tested on classes they haven't seen yet
    """
    # Initialize output directory
    out_dir = OUT_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the accuracies for both models
    monty_accuracies = get_monty_accuracies_for_continual_learning()
    vit_accuracies = get_vit_accuracies_for_continual_learning()

    # Create the heatmap matrices
    num_models = len(monty_accuracies)
    monty_heatmap = np.full((num_models, num_models), np.nan)
    vit_heatmap = np.full((num_models, num_models), np.nan)

    for model_id, model_accuracies in monty_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            monty_heatmap[model_id - 1, class_id] = accuracy

    for model_id, model_accuracies in vit_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            vit_heatmap[model_id - 1, class_id] = accuracy

    # Create the plot with two subplots, matching the height of plot_performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    plt.subplots_adjust(left=0.08, right=0.8, wspace=0.3)

    # Create the heatmaps using Inferno colormap
    im1 = ax1.imshow(monty_heatmap, cmap="inferno", vmin=0, vmax=100)
    im2 = ax2.imshow(vit_heatmap, cmap="inferno", vmin=0, vmax=100)

    # Set labels and titles
    ax1.set_xlabel("Target Object")
    ax1.set_ylabel("Number of Objects Learned")
    ax1.set_title("Monty")

    ax2.set_xlabel("Target Object")
    ax2.set_ylabel("Number of Objects Learned")
    ax2.set_title("ViT")

    # Set ticks
    for ax in [ax1, ax2]:
        ticks = [0, 9, 19, 29, 39, 49, 59, 69, 76]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([t + 1 for t in ticks])
        ax.set_yticklabels([t + 1 for t in ticks])
        ax.grid(True, linestyle="-", alpha=0.3)

    # Move colorbar to the top right outside the ViT subplot
    cax = fig.add_axes([0.8, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Accuracy (%)", labelpad=5, y=0.5, fontsize=8)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("top")

    fig.savefig(out_dir / "accuracy_heatmaps.png", bbox_inches="tight")
    fig.savefig(out_dir / "accuracy_heatmaps.svg", bbox_inches="tight")


def plot_accuracy_heatmaps_gray_bg():
    """Plot heatmaps showing both Monty and ViT's accuracy for each class at each model checkpoint.

    The heatmaps will be lower triangular matrices where:
    - Rows represent model checkpoints (0 to 76)
    - Columns represent classes (0 to 76)
    - Each cell shows the accuracy of that model on that class
    - Only the lower triangle is filled since models can't be tested on classes they haven't seen yet
    - Uses a gray background only inside the plot area to make the yellow in inferno colormap pop out
    """
    # Initialize output directory
    out_dir = OUT_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the accuracies for both models
    monty_accuracies = get_monty_accuracies_for_continual_learning()
    vit_accuracies = get_vit_accuracies_for_continual_learning()

    # Create the heatmap matrices
    num_models = len(monty_accuracies)
    monty_heatmap = np.full((num_models, num_models), np.nan)
    vit_heatmap = np.full((num_models, num_models), np.nan)

    for model_id, model_accuracies in monty_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            monty_heatmap[model_id - 1, class_id] = accuracy

    for model_id, model_accuracies in vit_accuracies.items():
        for class_id, accuracy in model_accuracies.items():
            vit_heatmap[model_id - 1, class_id] = accuracy

    # Create the plot with two subplots, matching the height of plot_performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))

    # Set gray background only for the plot areas
    ax1.set_facecolor("#dddddd")
    ax2.set_facecolor("#dddddd")

    # Create the heatmaps using Inferno colormap
    im1 = ax1.imshow(monty_heatmap, cmap="inferno", vmin=0, vmax=100)
    im2 = ax2.imshow(vit_heatmap, cmap="inferno", vmin=0, vmax=100)

    # Set labels and titles
    ax1.set_xlabel("Target Object")
    ax1.set_ylabel("Number of Objects Learned")
    ax1.set_title("Monty")

    ax2.set_xlabel("Target Object")
    ax2.set_ylabel("Number of Objects Learned")
    ax2.set_title("ViT")

    # Set ticks - shifted by 1 to better align with data
    for ax in [ax1, ax2]:
        ticks = [0, 9, 19, 29, 39, 49, 59, 69, 76]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([t + 1 for t in ticks])
        ax.set_yticklabels([t + 1 for t in ticks])

    # Adjust the subplot layout to make room for the colorbar
    plt.subplots_adjust(left=0.08, right=0.80, wspace=0.3)

    # Add colorbar with adjusted position
    cax = fig.add_axes([0.82, 0.52, 0.02, 0.3])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Accuracy (%)", labelpad=5, y=0.5, fontsize=8)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("top")

    # Save the plot with bbox_inches='tight' to ensure everything is included
    fig.savefig(out_dir / "accuracy_heatmaps_gray_bg.png", bbox_inches="tight")
    fig.savefig(out_dir / "accuracy_heatmaps_gray_bg.svg", bbox_inches="tight")


if __name__ == "__main__":
    plot_performance()
    plot_accuracy_heatmaps()
    plot_accuracy_heatmaps_gray_bg()
    plot_accuracy_heatmaps_blue()
    plot_accuracy_heatmaps_green()
    plot_accuracy_heatmaps_green_light()
    # plot_monty_accuracy_heatmap()
    # plot_vit_accuracy_heatmap()
