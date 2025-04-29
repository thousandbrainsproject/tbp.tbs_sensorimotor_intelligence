# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 3.

Panel A: Known Objects
 - `plot_known_objects()`

Panel B: Evidence Graphs and Patches
 - `plot_evidence_graphs_and_patches()`

Panel C: Sensor Path
 - `plot_sensor_path()`

Panel D: Performance
 - `plot_performance()`
 - `draw_icons()`

Running the above functions requires that the following experiments have been run:
 - `pretrain_dist_agent_1lm`: For plotting the known objects.
 - `fig3_evidence_run`: For plotting the sensor path and evidence graphs + patches.
 - `dist_agent_1lm`, `dist_agent_1lm_noise`, `dist_agent_1lm_randrot_all`, and
   `dist_agent_1lm_randrot_all_noise`: For plotting the performance metrics.

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    load_eval_stats,
    load_object_model,
)
from plot_utils import (
    TBP_COLORS,
    SensorModuleData,
    axes3d_clean,
    axes3d_set_aspect_equal,
    init_matplotlib_style,
    violinplot,
)

init_matplotlib_style()


# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


"""
--------------------------------------------------------------------------------
Panel A: Known Objects
--------------------------------------------------------------------------------
"""


def plot_known_objects():
    """Plot the "known objects" for panel A.

    Requires the experiment `dist_agent_1lm` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig3/known_objects/`.
    """
    # Initialize output paths.
    out_dir = OUT_DIR / "known_objects"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(5, 4), subplot_kw={"projection": "3d"})
    mug = load_object_model("dist_agent_1lm", "mug")
    bowl = load_object_model("dist_agent_1lm", "bowl")
    golf_ball = load_object_model("dist_agent_1lm", "golf_ball")

    axes[0].scatter(mug.x, mug.y, mug.z, color=mug.rgba, alpha=0.5, s=5, linewidth=0)
    axes[1].scatter(
        bowl.x, bowl.y, bowl.z, color=bowl.rgba, alpha=0.5, s=5, linewidth=0
    )
    axes[2].scatter(
        golf_ball.x,
        golf_ball.y,
        golf_ball.z,
        color=golf_ball.rgba,
        alpha=0.5,
        s=5,
        linewidth=0,
    )

    for ax in axes:
        axes3d_clean(ax, grid=False)
        axes3d_set_aspect_equal(ax)
        ax.view_init(115, -90, 0)

    fig.savefig(out_dir / "known_objects.png")
    fig.savefig(out_dir / "known_objects.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel B: Evidence Graphs and Patches
--------------------------------------------------------------------------------
"""


def plot_evidence_graphs_and_patches():
    """Plot the evidence graphs and patches for panel D.

    Requires the experiment `fig3_evidence_run` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig3/evidence_graphs_and_patches/`.
    """
    # Initialize output paths.
    main_out_dir = OUT_DIR / "evidence_graphs_and_patches"
    main_out_dir.mkdir(parents=True, exist_ok=True)

    # Load the stats.
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_evidence_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[0]

    object_names = ["mug", "bowl", "golf_ball"]
    steps = np.array([0, 10, 20, 39, 40])
    steps = np.arange(41)
    n_steps = len(steps)

    n_rows = n_cols = 64
    center_loc = n_rows // 2 * n_cols + n_cols // 2
    centers = np.zeros((n_steps, 3))
    for i in range(n_steps):
        arr = np.array(stats["SM_0"]["raw_observations"][i]["semantic_3d"])
        centers[i, 0] = arr[center_loc, 0]
        centers[i, 1] = arr[center_loc, 1]
        centers[i, 2] = arr[center_loc, 2]

    # Extract evidence values for all objects.
    all_evidences = stats["LM_0"]["evidences"]
    all_possible_locations = stats["LM_0"]["possible_locations"]
    all_possible_rotations = stats["LM_0"]["possible_rotations"]
    objects = {name: load_object_model("dist_agent_1lm", name) for name in object_names}
    for name, obj in objects.items():
        obj.evidences, obj.locations = [], []
        for i in range(n_steps):
            obj.evidences.append(all_evidences[i][name])
            obj.locations.append(all_possible_locations[i][name])
        obj.evidences = np.array(obj.evidences)
        obj.locations = np.array(obj.locations)
        obj.rotation = np.array(all_possible_rotations[0][name])

    # Plot evidence graphs for each object and step individually.
    out_dir = main_out_dir / "evidence_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    for i, step in enumerate(steps):
        fig, axes = plt.subplots(
            1, len(objects) + 1, figsize=(12, 6), subplot_kw=dict(projection="3d")
        )

        # Plot mug with current observation location.
        ax = axes[0]
        mug = objects["mug"]
        ax.scatter(
            mug.x,
            mug.y,
            mug.z,
            color="gray",
            alpha=0.6,
            s=1,
            linewidths=0,
        )
        center = centers[step]
        ax.scatter(center[0], center[1], center[2], color="red", s=50)
        ax.axis("off")
        ax.view_init(100, -100, -10)
        ax.set_xlim(-0.119, 0.119)
        ax.set_ylim(1.5 - 0.119, 1.5 + 0.119)
        ax.set_zlim(-0.119, 0.119)

        # Make colormap for this step.
        all_evidences = [obj.evidences[step].flatten() for obj in objects.values()]
        all_evidences = np.concatenate(all_evidences)
        evidences_min = np.percentile(all_evidences, 2.5)
        evidences_max = np.percentile(all_evidences, 99.99)
        scalar_map = plt.cm.ScalarMappable(
            cmap="inferno", norm=plt.Normalize(vmin=evidences_min, vmax=evidences_max)
        )

        for j, obj in enumerate(objects.values()):
            ax = axes[j + 1]
            locations = obj.locations[step]
            n_points = locations.shape[0] // 2
            locations = locations[:n_points]
            evidences = obj.evidences[step]
            ev1 = evidences[:n_points]
            ev2 = evidences[n_points:]
            stacked = np.hstack([ev1[:, np.newaxis], ev2[:, np.newaxis]])
            evidences = stacked.max(axis=1)

            colors = evidences
            sizes = np.log(evidences - evidences.min() + 1) * 10
            alphas = np.array(evidences)
            alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
            x, y, z = locations[:, 0], locations[:, 1], locations[:, 2]
            ax.scatter(
                x,
                y,
                z,
                c=colors,
                cmap="inferno",
                alpha=alphas,
                vmin=evidences_min,
                vmax=evidences_max,
                s=sizes,
                linewidths=0,
            )

            # Plot highest evidence location separately.
            ind_ev_max = evidences.argmax()
            ev = evidences[ind_ev_max]
            x, y, z = x[ind_ev_max], y[ind_ev_max], z[ind_ev_max]
            sizes = sizes[ind_ev_max]
            ax.scatter(
                x,
                y,
                z,
                c=ev,
                cmap="inferno",
                alpha=1,
                vmin=evidences_min,
                vmax=evidences_max,
                s=sizes,
                linewidths=0.5,
                edgecolor="black",
            )

            ax.view_init(100, -100, -10)
            axes3d_clean(ax, grid=False)
            axes3d_set_aspect_equal(ax)
            ax.set_xlim(-0.119, 0.119)
            ax.set_ylim(1.5 - 0.119, 1.5 + 0.119)
            ax.set_zlim(-0.119, 0.119)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
        fig.suptitle(f"step {step}")
        fig.savefig(png_dir / f"evidence_graphs_{step}.png")
        fig.savefig(svg_dir / f"evidence_graphs_{step}.svg")
        plt.close(fig)

    # Plot the colorbar.
    fig, ax = plt.subplots(1, 1, figsize=(1, 2))
    cbar = plt.colorbar(scalar_map, ax=ax, orientation="vertical", label="Evidence")
    ax.remove()  # Remove the empty axes, we just want the colorbar
    cbar.set_ticks([])
    cbar.set_label("")
    fig.tight_layout()
    fig.savefig(out_dir / "colorbar.png")
    fig.savefig(out_dir / "colorbar.svg")
    plt.close(fig)

    # Extract RGBA patches for sensor module 0.
    rgba_patches = []
    for ind in range(n_steps):
        rgba_patches.append(np.array(stats["SM_0"]["raw_observations"][ind]["rgba"]))
    rgba_patches = np.array(rgba_patches)

    # Save the RGBA patches.
    out_dir = main_out_dir / "patches"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        patch = rgba_patches[step]
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(patch)
        ax.axis("off")
        fig.tight_layout(pad=0.0)
        fig.savefig(png_dir / f"patch_step_{step}.png")
        fig.savefig(svg_dir / f"patch_step_{step}.svg")
        plt.close(fig)

"""
--------------------------------------------------------------------------------
Panel C: Sensor Path
--------------------------------------------------------------------------------
"""


def plot_sensor_path():
    """Plot the sensor path for panel A.

    Requires the experiment `fig3_evidence_run` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig3/sensor_path/`.
    """
    # Initialize output paths.
    out_dir = OUT_DIR / "sensor_path"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the stats.
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_evidence_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[0]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), subplot_kw={"projection": "3d"})

    model = load_object_model("dist_agent_1lm", "mug")
    ax.scatter(
        model.x,
        model.y,
        model.z,
        color=model.rgba,
        alpha=0.35,
        s=4,
        edgecolor="none",
    )
    sm = SensorModuleData(stats["SM_0"])
    sm.plot_sensor_path(ax, steps=36)

    ax.set_proj_type("persp", focal_length=0.5)
    ax.view_init(115, -90, 0)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)
    fig.savefig(out_dir / "sensor_path.png")
    fig.savefig(out_dir / "sensor_path.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel D: Performance
--------------------------------------------------------------------------------
"""


def plot_performance() -> None:
    """Plot core performance metrics.

    Requires the experiments `dist_agent_1lm`, `dist_agent_1lm_noise`,
    `dist_agent_1lm_randrot_all`, and `dist_agent_1lm_randrot_all_noise` have been
    run.

    Output is saved to `DMC_ANALYSIS_DIR/fig3/performance/`.
    """
    # Initialize output paths.
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the stats.
    dataframes = [
        load_eval_stats("dist_agent_1lm"),
        load_eval_stats("dist_agent_1lm_noise_all"),
        load_eval_stats("dist_agent_1lm_randrot_14"),
        load_eval_stats("dist_agent_1lm_randrot_14_noise_all"),
        load_eval_stats("dist_agent_1lm_randrot_14_color_clamped"),
        load_eval_stats("dist_agent_1lm_randrot_14_noise_all_color_clamped"),
        load_eval_stats("dist_agent_1lm_randrot_14_noise_all_color_clamped_hsv"),
    ]
    accuracy, rotation_error = [], []
    for i, df in enumerate(dataframes):
        sub_df = df[df.primary_performance.isin(["correct", "correct_mlh"])]
        accuracy.append(100 * len(sub_df) / len(df))
        rotation_error.append(np.degrees(sub_df.rotation_error))

    # Initialize the plot.
    axes_width, axes_height = 2.81, 1.75
    axes_frac = 0.7
    axes_loc = (1 - axes_frac) / 2
    fig = plt.figure(figsize=(axes_width / axes_frac, axes_height / axes_frac))
    ax1 = fig.add_axes([axes_loc, axes_loc, axes_frac, axes_frac])

    # fig, ax1 = plt.subplots(1, 1, figsize=(3.5, 3))
    ax2 = ax1.twinx()

    # Params
    bar_width = 0.4
    violin_width = 0.4
    gap = 0.04
    xticks = np.arange(len(dataframes)) * 1.3
    bar_positions = xticks - bar_width / 2 - gap / 2
    violin_positions = xticks + violin_width / 2 + gap / 2
    median_style = dict(color="lightgray", lw=1, ls="-")

    # Plot accuracy bars
    ax1.bar(
        bar_positions,
        accuracy,
        color=TBP_COLORS["blue"],
        width=bar_width,
    )
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("% Correct")

    # Plot rotation error violins
    violinplot(
        rotation_error,
        violin_positions,
        width=violin_width,
        color=TBP_COLORS["purple"],
        showextrema=False,
        showmedians=True,
        median_style=median_style,
        ax=ax2,
    )

    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_ylim(0, 180)
    ax2.set_ylabel("Rotation Error (deg)")

    ax1.set_xticks(xticks)
    xticklabels = [
        "base",
        "noise",
        "RR",
        "noise+RR",
        "RR\n(hsv clamped)",
        "noise+RR\n(hue clamped)",
        "noise+RR\n(hsv clamped)",
    ]
    ax1.set_xticklabels(xticklabels, rotation=45, ha="center")

    ax1.spines["right"].set_visible(True)
    ax2.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(out_dir / "performance.png")
    fig.savefig(out_dir / "performance.svg")
    plt.show()


def draw_icons():
    """Draw the object models under the x-axis for panel B.

    Requires the experiment `pretrain_dist_agent_1lm` has been run.

    Output is saved to `DMC_ANALYSIS_DIR/fig3/icons`.
    """
    # Initialize output paths.
    out_dir = OUT_DIR / "icons"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(0)
    standard_noise_params = {"location": 0.002, "features": {"hsv": 0.1}}
    all_params = [
        {
            "label": "base",
            "noise": {},
            "rotation": (0, 0, 0),
        },
        {
            "label": "noise",
            "noise": standard_noise_params,
            "rotation": (0, 0, 0),
        },
        {
            "label": "RR",
            "noise": {},
            "rotation": [45, 10, 30],
        },
        {
            "label": "RR + noise",
            "noise": standard_noise_params,
            "rotation": [45, 10, 30],
        },
        {
            "label": "noise+RR\n(hsv clamped)",
            "noise": standard_noise_params,
            "rotation": [45, 10, 30],
        },
        {
            "label": "noise+RR\n(hue clamped)",
            "noise": standard_noise_params,
            "rotation": [45, 10, 30],
        },
    ]

    # Load the pretrained model.
    model = load_object_model("dist_agent_1lm", "mug")
    model = model - [0, 1.5, 0]

    # Plot the icons.
    fig, axes = plt.subplots(
        1, len(all_params), figsize=(8, 4), subplot_kw={"projection": "3d"}
    )
    for i, params in enumerate(all_params):
        ax = axes[i]
        noise_params = params["noise"]

        obj = model.copy()
        obj = obj.rotated(params["rotation"], degrees=True)
        if "location" in noise_params:
            noise = rng.normal(0, noise_params["location"], obj.pos.shape)
            obj.pos = obj.pos + noise

        if "features" in noise_params and "hsv" in noise_params["features"]:
            hsv_std = noise_params["features"]["hsv"]

            # Convert RGB to HSV
            rgb = model.rgba[:, :3]
            hsv = np.zeros_like(rgb)
            for j in range(len(rgb)):
                hsv[j] = matplotlib.colors.rgb_to_hsv(rgb[j])

            # Add HSV noise
            noise = rng.normal(0, hsv_std, hsv.shape)
            hsv = hsv + noise
            hsv = np.clip(hsv, 0, 1)

            # Convert back to RGB
            rgba = np.ones((len(hsv), 4))
            for j in range(len(hsv)):
                rgba[j, :3] = matplotlib.colors.hsv_to_rgb(hsv[j])
            obj.rgba = rgba

        if params["label"] == "noise+RR\n(hue clamped)":
            model_hsv = matplotlib.colors.rgb_to_hsv(model.rgba[:, :3])
            model_hsv[:, 0] = 0.667
            obj.rgba[:, :3] = matplotlib.colors.hsv_to_rgb(model_hsv)

        elif params["label"] == "noise+RR\n(hsv clamped)":
            clamped_color = matplotlib.colors.hsv_to_rgb([0.667, 1.0, 1.0])
            rgb = np.broadcast_to(clamped_color, obj.rgba[:, :3].shape)
            obj.rgba[:, :3] = rgb

        ax.scatter(obj.x, obj.y, obj.z, c=obj.rgba, alpha=0.5, edgecolors="none", s=2)
        ax.set_title(params["label"])
        axes3d_set_aspect_equal(ax)
        ax.axis("off")
        ax.view_init(90, -90)

    fig.savefig(out_dir / "icons.png")
    fig.savefig(out_dir / "icons.svg")
    plt.show()


if __name__ == "__main__":
    # plot_known_objects()
    # plot_evidence_graphs_and_patches()
    # plot_sensor_path()
    # plot_performance()
    draw_icons()
