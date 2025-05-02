# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 5.

Panel B: 8-patch view finder
 - `plot_8lm_patches()`

Panel C: Accuracy
 - `plot_accuracy()`

Panel C: Steps
 - `plot_steps()`

Running the above functions requires that the following experiments have been run:
 - `fig5_visualize_8lm_patches`
 - `dist_agent_1lm_randrot_noise`
 - `dist_agent_2lm_randrot_noise`
 - `dist_agent_4lm_randrot_noise`
 - `dist_agent_8lm_randrot_noise`
 - `dist_agent_16lm_randrot_noise`
"""

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    get_frequency,
    load_eval_stats,
)
from plot_utils import (
    TBP_COLORS,
    axes3d_set_aspect_equal,
    init_matplotlib_style,
    violinplot,
)

init_matplotlib_style()
np.random.seed(0)

OUT_DIR = DMC_ANALYSIS_DIR / "fig5"
OUT_DIR.mkdir(parents=True, exist_ok=True)


"""
--------------------------------------------------------------------------------
Utilities
--------------------------------------------------------------------------------
"""


def reduce_eval_stats(eval_stats: pd.DataFrame) -> pd.DataFrame:
    """Reduce the eval stats dataframe to a single row per episode.

    The main purpose of this function is to classify an episode as either "correct"
    or "confused" based on the number of correct and confused LMs in an episode (or
    "correct_mlh" and "confused_mlh" for timed-out episodes).

    Args:
        eval_stats: The eval stats dataframe.

    Returns:
        pd.DataFrame: A dataframe with a single row per episode.
    """
    performance_options = (
        "correct",
        "confused",
        "no_match",
        "correct_mlh",
        "confused_mlh",
        "time_out",
        "pose_time_out",
        "no_label",
        "patch_off_object",
    )

    episodes = np.arange(eval_stats.episode.max() + 1)
    assert np.array_equal(eval_stats.episode.unique(), episodes)  # sanity check
    n_episodes = len(episodes)

    # Columns of output dataframe. More are added later.
    output_data = {
        "primary_performance": np.zeros(n_episodes, dtype=object),
    }
    for name in performance_options:
        output_data[f"n_{name}"] = np.zeros(n_episodes, dtype=int)

    episode_groups = eval_stats.groupby("episode")
    for episode, df in episode_groups:
        # Find one result given many LM results.
        row = {}

        perf_counts = {key: 0 for key in performance_options}
        perf_counts.update(df.primary_performance.value_counts())
        # found = []
        for name in performance_options:
            row[f"n_{name}"] = perf_counts[name]

        # Decide performance based on the number of correct/confused LM and
        # correct_mlh/confused_mlh LMs. The rules are as follows:
        #
        # 1. If there are any correct or confused LMs (i.e., the episode did not time-out),
        #    then we apply the following rules.
        #   1.1 |correct LMs| > |confused-LMs| : correct
        #   2.1 |correct LMs| < |confused-LMs| : confused
        #   3.1 |correct LMs| = |confused-LMs| : we break the tie by comparing LMs
        #      that timed out.
        #      3.1.1 |correct_mlh LMs| > |confused_mlh LMs| : correct
        #      3.2.1 |correct_mlh LMs| < |confused_mlh LMs| : confused
        #      3.3.1 |correct_mlh LMs| = |confused_mlh LMs| : we break the tie by
        #           randomly choosing correct or confused.
        #
        # 2. If the episode timed out, then we apply the following rules:
        #   2.1. |correct_mlh LMs| > |confused_mlh LMs| : correct
        #   2.2. |correct_mlh LMs| < |confused_mlh LMs| : confused
        #   2.3. |correct_mlh LMs| = |confused_mlh LMs| : we break the tie by
        #        randomly choosing correct_mlh or confused_mlh.
        #
        # 3. In the highly unusual case where no LMs terminated with
        #    correct[_mlh] or confused[_mlh], then we assign performance based on
        #    precedence rules of the remaining options. The following is a list
        #    of remaining options in order of precedence (highest to lowest):
        #     - time_out
        #     - pose_time_out
        #     - no_label
        #     - patch_off_object
        #

        # Case 1: There are some correct or confused LMs.
        if row["n_correct"] > 0 or row["n_confused"] > 0:
            if row["n_correct"] > row["n_confused"]:
                performance = "correct"
            elif row["n_correct"] < row["n_confused"]:
                performance = "confused"
            else:
                # Break the tie by comparing timed-out LMs.
                if row["n_correct_mlh"] > row["n_confused_mlh"]:
                    performance = "correct"
                elif row["n_correct_mlh"] < row["n_confused_mlh"]:
                    performance = "confused"
                else:
                    # Break the tie by flipping a coin.
                    if np.random.rand() < 0.5:
                        performance = "correct"
                    else:
                        performance = "confused"

        # Case 2: The episode timed out.
        elif row["n_correct_mlh"] > 0 or row["n_confused_mlh"] > 0:
            if row["n_correct_mlh"] > row["n_confused_mlh"]:
                performance = "correct_mlh"
            elif row["n_correct_mlh"] < row["n_confused_mlh"]:
                performance = "confused_mlh"
            else:
                # Break the tie by flipping a coin.
                if np.random.rand() < 0.5:
                    performance = "correct_mlh"
                else:
                    performance = "confused_mlh"

        # Case 3: No LMs terminated with correct[_mlh] or confused[_mlh].
        else:
            # Apply precedence rules.
            performance = None
            for name in ("time_out", "pose_time_out", "no_label", "patch_off_object"):
                if row[f"n_{name}"] > 0:
                    performance = name
                    break
            if performance is None:
                raise ValueError("No valid performance value found")

        row["primary_performance"] = performance

        for key, val in row.items():
            output_data[key][episode] = val

    # Add episode data not specific to the LM.
    output_data["monty_matching_steps"] = (
        episode_groups.monty_matching_steps.first().values
    )
    output_data["primary_target_object"] = (
        episode_groups.primary_target_object.first().values
    )
    output_data["primary_target_rotation"] = (
        episode_groups.primary_target_object.first().values
    )
    output_data["episode"] = episode_groups.episode.first().values
    output_data["epoch"] = episode_groups.epoch.first().values

    out = pd.DataFrame(output_data)
    return out


def aggregate_multilm_performance_data(experiments: Iterable[str]) -> pd.DataFrame:
    """Save the performance table for the single LM experiments.

    Output is saved to `DMC_ANALYSIS_DIR/fig3/performance/single_lm_performance.csv`.
    """

    columns = {
        "accuracy": [],
        "percent.correct": [],
        "percent.correct_mlh": [],
        "n_steps": [],
        "n_steps.mean": [],
        "n_steps.median": [],
        "rotation_error": [],
        "rotation_error.mean": [],
        "rotation_error.median": [],
    }
    for exp in experiments:
        eval_stats = load_eval_stats(exp)
        reduced_stats = reduce_eval_stats(eval_stats)
        accuracy = 100 * get_frequency(
            reduced_stats["primary_performance"], ("correct", "correct_mlh")
        )
        percent_correct = 100 * get_frequency(
            reduced_stats["primary_performance"], "correct"
        )
        percent_correct_mlh = 100 * get_frequency(
            reduced_stats["primary_performance"], "correct_mlh"
        )
        n_steps = eval_stats["num_steps"]
        rotation_error = np.degrees(eval_stats.rotation_error.dropna())

        columns["accuracy"].append(accuracy)
        columns["percent.correct"].append(percent_correct)
        columns["percent.correct_mlh"].append(percent_correct_mlh)

        columns["n_steps"].append(n_steps)
        columns["n_steps.mean"].append(n_steps.mean())
        columns["n_steps.median"].append(n_steps.median())

        columns["rotation_error"].append(rotation_error)
        columns["rotation_error.mean"].append(rotation_error.mean())
        columns["rotation_error.median"].append(rotation_error.median())

    return pd.DataFrame(columns, index=experiments)


"""
--------------------------------------------------------------------------------
Panel B: 8-patch view finder
--------------------------------------------------------------------------------
"""


def plot_8lm_patches():
    """Plot the 8-SM + view_finder visualization for figure 4.

    Uses data from the experiment `fig4_visualize_8lm_patches` defined in
    `configs/visualizations.py`. This function renders the sensor module
    RGBA data in the scene (in 3D) and overlays the sensor module's patch
    boundaries.

    Output is saved to `DMC_ANALYSIS_DIR/fig5/8lm_patches`.

    """
    # Initialize output directory.
    out_dir = OUT_DIR / "8lm_patches"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load the detailed stats.
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig5_visualize_8lm_patches"
    detailed_stats_path = experiment_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    def pull(sm_num: int):
        """Helper function for extracting necessary sensor data."""
        sm_dict = stats[f"SM_{sm_num}"]
        # Extract RGBA sensor patch.
        rgba_2d = np.array(sm_dict["raw_observations"][0]["rgba"]) / 255.0
        n_rows, n_cols = rgba_2d.shape[0], rgba_2d.shape[1]

        # Extract locations and on-object filter.
        semantic_3d = np.array(sm_dict["raw_observations"][0]["semantic_3d"])
        pos_1d = semantic_3d[:, 0:3]
        pos_2d = pos_1d.reshape(n_rows, n_cols, 3)
        on_object_1d = semantic_3d[:, 3].astype(int) > 0
        on_object_2d = on_object_1d.reshape(n_rows, n_cols)

        # Filter out points that aren't on-object. Yields a flat list of points/colors.
        return rgba_2d, pos_2d, on_object_2d

    # Create a 3D plot of the semantic point cloud
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=90, azim=-90, roll=0)
    ax.set_proj_type("persp", focal_length=0.125)
    ax.dist = 4.55

    # Render the view finder's RGBA data in the scene.
    rgba_2d, pos_2d, on_object_2d = pull(8)
    rows, cols = np.where(on_object_2d)
    pos_valid_1d = pos_2d[on_object_2d]
    rgba_valid_1d = rgba_2d[on_object_2d]
    ax.scatter(
        pos_valid_1d[:, 0],
        pos_valid_1d[:, 1],
        pos_valid_1d[:, 2],
        c=rgba_valid_1d,
        marker="o",
        alpha=0.3,
        zorder=5,
        s=10,
        edgecolors="none",
    )

    # Render patches and patch boundaries for all sensors.
    for i in range(8):
        # Load sensor data.
        rgba_2d, pos_2d, on_object_2d = pull(i)
        rows, cols = np.where(on_object_2d)
        pos_valid_1d = pos_2d[on_object_2d]
        rgba_valid_1d = rgba_2d[on_object_2d]

        # Render the patch.
        ax.scatter(
            pos_valid_1d[:, 0],
            pos_valid_1d[:, 1],
            pos_valid_1d[:, 2],
            c=rgba_valid_1d,
            marker="o",
            alpha=1,
            zorder=10,
            edgecolors="none",
            s=1,
        )

        # Draw the patch boundaries (complicated).
        n_rows, n_cols = on_object_2d.shape
        row_mid, col_mid = n_rows // 2, n_cols // 2
        n_pix_on_object = on_object_2d.sum()

        if n_pix_on_object == 0:
            contours = []
        elif n_pix_on_object == on_object_2d.size:
            temp = np.zeros((n_rows, n_cols), dtype=bool)
            temp[0, :] = True
            temp[-1, :] = True
            temp[:, 0] = True
            temp[:, -1] = True
            contours = [np.argwhere(temp)]
        else:
            contours = skimage.measure.find_contours(
                on_object_2d, level=0.5, positive_orientation="low"
            )
            contours = [] if contours is None else contours

        for ct in contours:
            row_mid, col_mid = n_rows // 2, n_cols // 2

            # Contour may be floating point (fractional indices from scipy). If so,
            # round rows/columns towards the center of the patch.
            if not np.issubdtype(ct.dtype, np.integer):
                # Round towards the center.
                rows, cols = ct[:, 0], ct[:, 1]
                rows_new, cols_new = np.zeros_like(rows), np.zeros_like(cols)
                rows_new[rows >= row_mid] = np.floor(rows[rows >= row_mid])
                rows_new[rows < row_mid] = np.ceil(rows[rows < row_mid])
                cols_new[cols >= col_mid] = np.floor(cols[cols >= col_mid])
                cols_new[cols < col_mid] = np.ceil(cols[cols < col_mid])
                ct_new = np.zeros_like(ct, dtype=int)
                ct_new[:, 0] = rows_new.astype(int)
                ct_new[:, 1] = cols_new.astype(int)
                ct = ct_new

            # Drop any points that happen to be off-object (it's possible that
            # some boundary points got rounded off-object).
            points_on_object = on_object_2d[ct[:, 0], ct[:, 1]]
            ct = ct[points_on_object]

            # In order to plot the boundary as a line, we need the points to
            # be in order. We can order them by associating each point with its
            # angle from the center of the patch. This isn't a general solution,
            # but it works here.
            Y, X = row_mid - ct[:, 0], ct[:, 1] - col_mid  # pixel to X/Y coords.
            theta = np.arctan2(Y, X)
            sort_order = np.argsort(theta)
            ct = ct[sort_order]

            # Finally, plot the contour.
            xyz = pos_2d[ct[:, 0], ct[:, 1]]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="k", linewidth=3, zorder=20)

    axes3d_set_aspect_equal(ax)
    ax.axis("off")

    fig.savefig(out_dir / "8lm_patches.png")
    fig.savefig(out_dir / "8lm_patches.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel C: Accuracy
--------------------------------------------------------------------------------
"""


def plot_accuracy():
    """Plot accuracy of 1-LM and multi-LM experiments.

    Requires the following experiments to have been run:
    - `dist_agent_1lm_randrot_noise`
    - `dist_agent_2lm_randrot_noise`
    - `dist_agent_4lm_randrot_noise`
    - `dist_agent_8lm_randrot_noise`
    - `dist_agent_16lm_randrot_noise`

    Output is saved to `DMC_ANALYSIS_DIR/fig5/performance`.

    """
    # Initialize output directory.
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)

    experiments = [
        "dist_agent_1lm_randrot_noise",
        "dist_agent_2lm_randrot_noise",
        "dist_agent_4lm_randrot_noise",
        "dist_agent_8lm_randrot_noise",
        "dist_agent_16lm_randrot_noise",
    ]
    performance = aggregate_multilm_performance_data(experiments)

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3), sharex=True)
    top_ax, bottom_ax = axes
    fig.subplots_adjust(hspace=0.05)

    # Plot params.
    ylims = [(0, 25), (75, 100)]
    xticks = np.arange(5)

    # 1-LM
    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        ax.bar(
            xticks[0],
            [performance.accuracy[0]],
            color=TBP_COLORS["blue"],
            width=0.8,
            label="no voting",
        )

    # Multi-LM
    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        ax.bar(
            xticks[1:],
            performance.accuracy[1:],
            color=TBP_COLORS["purple"],
            width=0.8,
            label="voting",
        )

    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        ax.set_ylim(ylims[ax_num])

    # Sets parameters for both x-axes (they're shared, so removing ticks for the
    # top plot removes ticks for the bottom plot).
    bottom_ax.set_xlabel("Num. LMs")
    bottom_ax.set_xticks(xticks)
    bottom_ax.set_xticklabels(["1", "2", "4", "8", "16"])
    bottom_ax.legend(loc="lower right")

    bottom_ax.set_ylabel("% Correct")
    bottom_ax.set_yticks([0, 10, 20])
    top_ax.spines.bottom.set_visible(False)
    top_ax.set_yticks([80, 90, 100])

    # Draw y-axis divider markers.
    marker_kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=8,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    top_ax.plot([0], [0], transform=top_ax.transAxes, **marker_kwargs)
    bottom_ax.plot([0], [1], transform=bottom_ax.transAxes, **marker_kwargs)

    fig.savefig(out_dir / "accuracy.png")
    fig.savefig(out_dir / "accuracy.svg")
    plt.show()



"""
--------------------------------------------------------------------------------
Panel C: Steps
--------------------------------------------------------------------------------
"""


def plot_steps():
    """Plot the number of steps taken by 1-LM and multi-LM experiments.

    Requires the following experiments to have been run:
    - `dist_agent_1lm_randrot_noise`
    - `dist_agent_2lm_randrot_noise`
    - `dist_agent_4lm_randrot_noise`
    - `dist_agent_8lm_randrot_noise`
    - `dist_agent_16lm_randrot_noise`

    Output is saved to `DMC_ANALYSIS_DIR/fig5/performance`.

    """
    # Initialize output directory.
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)

    experiments = [
        "dist_agent_1lm_randrot_noise",
        "dist_agent_2lm_randrot_noise",
        "dist_agent_4lm_randrot_noise",
        "dist_agent_8lm_randrot_noise",
        "dist_agent_16lm_randrot_noise",
    ]
    performance = aggregate_multilm_performance_data(experiments)

    fig = plt.figure(figsize=(3.4, 3))
    gs = fig.add_gridspec(3, 1)  # 3 rows, bottom plot will take 2 rows

    # Create the two subplots with shared x-axis
    top_ax = fig.add_subplot(gs[0, 0])  # Top subplot takes 1/3
    bottom_ax = fig.add_subplot(gs[1:, 0], sharex=top_ax)  # Bottom subplot takes 2/3
    fig.subplots_adjust(hspace=0.05)

    # Plot params.
    ylims = [(0, 110), (440, 500)]
    yticks = [
        [0, 25, 50, 75, 100],
        [450, 475, 500],
    ]
    xticks = np.arange(5)

    # Plot distribution of n_steps for 1-LM
    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        violinplot(
            [performance.n_steps[0]],
            [xticks[0]],
            color=TBP_COLORS["blue"],
            width=0.8,
            showmedians=True,
            median_style=dict(color="lightgray", lw=2),
            bw_method=0.1,
            ax=ax,
        )

    # Plot distribution of n_steps for >1 LM
    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        violinplot(
            performance.n_steps[1:],
            xticks[1:],
            color=TBP_COLORS["purple"],
            width=0.8,
            showmedians=True,
            median_style=dict(color="lightgray", lw=2),
            bw_method=bw_method,
            ax=ax,
        )

    # Plot a line connecting the means across all conditions.
    means = performance["n_steps.mean"]
    for ax in [bottom_ax, top_ax]:
        ax.scatter(
            xticks,
            means,
            color="black",
            marker="o",
            s=20,
        )
        ax.plot(xticks, means, color="black", linestyle="-", linewidth=2, zorder=10)

    # Set y-limits and ticks.
    for ax_num, ax in enumerate([bottom_ax, top_ax]):
        ax.set_ylim(ylims[ax_num])
        ax.set_yticks(yticks[ax_num])

    # Sets parameters for both x-axes (they're shared, so removing ticks for the
    # top plot removes ticks for the bottom plot).
    bottom_ax.set_xlabel("Num. LMs")
    bottom_ax.set_xticks(xticks)
    bottom_ax.set_xticklabels(["1", "2", "4", "8", "16"])

    bottom_ax.set_ylabel("Steps")
    top_ax.spines.bottom.set_visible(False)

    # Draw y-axis divider markers.
    marker_kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=8,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    top_ax.plot([0], [0], transform=top_ax.transAxes, **marker_kwargs)
    bottom_ax.plot([0], [1], transform=bottom_ax.transAxes, **marker_kwargs)

    fig.savefig(out_dir / "steps.png")
    fig.savefig(out_dir / "steps.svg")
    plt.show()


def save_performance_table() -> None:
    """Save the performance metrics to a CSV file."""
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(exist_ok=True, parents=True)
    experiments = [
        "dist_agent_1lm_randrot_noise",
        "dist_agent_2lm_randrot_noise",
        "dist_agent_4lm_randrot_noise",
        "dist_agent_8lm_randrot_noise",
        "dist_agent_16lm_randrot_noise",
    ]
    performance = aggregate_multilm_performance_data(experiments)
    performance = performance.drop(columns=["n_steps", "rotation_error"])
    performance.to_csv(out_dir / "performance.csv")


if __name__ == "__main__":
    plot_8lm_patches()
    plot_accuracy()
    plot_steps()
    save_performance_table()
