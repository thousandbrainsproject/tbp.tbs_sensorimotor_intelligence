# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""This module defines functions used to generate images for figure 8.

Panel A: Training FLOPs in Monty and ViT
Panel B: Inference FLOPs vs. Accuracy in Monty and ViT
Panel C: Inference FLOPs vs. Rotation Error in Monty and ViT
"""

from __future__ import annotations

from brokenaxes import brokenaxes
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Union
import torch

from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_RESULTS_DIR,
    load_eval_stats,
    load_floppy_traces,
    load_monty_training_flops,
    load_vit_predictions,
    aggregate_1lm_performance_data,
)
from plot_utils import (
    TBP_COLORS,
    init_matplotlib_style,
)
from count_vit_flops import (
    vit_b16,
    vit_l16,
    vit_b32,
    vit_l32,
    vit_h14,
    input_shape,
    get_forward_flops,
    calculate_training_flops,
    get_imagenet21k_pretraining_flops,
)
init_matplotlib_style()


# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

"""
--------------------------------------------------------------------------------
Utilities
--------------------------------------------------------------------------------
"""
def load_training_data() -> dict[str, float]:
    """Get the FLOPs values for plotting.
    
    Returns:
        tuple containing:
        - vit_inference_1image: FLOPs for ViT inference on one image
        - vit_fintune_pretrained: FLOPs for ViT finetuning from pretrained
        - vit_pretrain_pretrained: FLOPs for ViT pretraining
        - vit_fintune_scratch: FLOPs for ViT finetuning from scratch
        - monty_train: FLOPs for Monty training
    """
    vit_inference_1image = get_forward_flops(
        model = vit_b16,
        input_shape = input_shape,
    )
    vit_fintune_pretrained = calculate_training_flops(
        model = vit_b16,
        input_shape = input_shape,
        num_images = 77 * 14, # 77 classes * 14 rotations per class
        num_epochs = 25, # 25 epochs for finetuning
    )
    vit_pretrain_pretrained = get_imagenet21k_pretraining_flops(
        model = vit_b16,
    )
    vit_fintune_scratch = calculate_training_flops(
        model = vit_b16,
        input_shape = input_shape,
        num_images = 77 * 14, # 77 classes * 14 rotations per class
        num_epochs = 75, # 75 epochs for random init
    )
    
    # Calculate Monty FLOPs
    # monty_train = load_monty_training_flops()["flops"]
    monty_train = 2.52e11 # TODO (Hojae): Update this value once Monty training FLOPs PR is merged
    
    return {
        "vit_inference_1image": vit_inference_1image,
        "vit_fintune_pretrained": vit_fintune_pretrained,
        "vit_pretrain_pretrained": vit_pretrain_pretrained,
        "vit_fintune_scratch": vit_fintune_scratch,
        "monty_train": monty_train,
    }

def format_flops(flops: Union[int, float]) -> str:
    """Format FLOPs value for display in scientific notation.
    
    Args:
        flops: Number of FLOPs to format.
        
    Returns:
        Formatted string representation of FLOPs in scientific notation.
    """
    # Convert to float if it's a string
    if isinstance(flops, int):
        flops = float(flops)

    if flops == 0:
        return "0"
    
    # Calculate the exponent
    exponent = int(np.floor(np.log10(abs(flops))))
    
    # Calculate the coefficient
    coefficient = flops / (10 ** exponent)
    
    # Format with appropriate precision
    if coefficient >= 10:
        coefficient /= 10
        exponent += 1
    
    return f"{coefficient:.2f} × 10$^{{{exponent}}}$"

def get_monty_flops_accuracy_data(exp_names: list[str]) -> pd.DataFrame:
    """Get Monty FLOPs and accuracy data from experiments.
    
    Args:
        exp_names: List of experiment names for Monty.
        
    Returns:
        DataFrame containing FLOPs and accuracy data for Monty experiments.
    """
    return load_eval_stats(exp_names)


def aggregate_vit_predictions(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Aggregate ViT predictions into a single row with model performance metrics.
    
    Args:
        df: DataFrame from load_vit_predictions containing prediction results.
        
    Returns:
        Single-row DataFrame with columns: model, accuracy, rotation_error
    """    
    # Calculate accuracy
    accuracy = (df["predicted_class"] == df["real_class"]).mean() * 100  # Convert to percentage
    
    # Calculate rotation error using geodesic loss
    predicted_quaternions = torch.tensor(np.stack(df["predicted_quaternion"].values))
    real_quaternions = torch.tensor(np.stack(df["real_quaternion"].values))
    # Compute the dot product
    dot_product = torch.sum(predicted_quaternions * real_quaternions, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle = 2 * torch.acos(torch.abs(dot_product))
    # Convert to degrees
    angle = angle * 180 / np.pi
    angle = angle.mean()
    
    # Calculate FLOPs (Inference)
    if "ViT-B/16" in model_name:
        flops = get_forward_flops(vit_b16, input_shape)
    elif "ViT-L/16" in model_name:
        flops = get_forward_flops(vit_l16, input_shape)
    elif "ViT-B/32" in model_name:
        flops = get_forward_flops(vit_b32, input_shape)
    elif "ViT-L/32" in model_name:
        flops = get_forward_flops(vit_l32, input_shape)
    elif "ViT-H/14" in model_name:
        flops = get_forward_flops(vit_h14, input_shape)

    # Return single row DataFrame
    return pd.DataFrame({
        "model": [model_name],
        "accuracy": [accuracy],
        "rotation_error": [angle],
        "flops": [flops]
    })


"""
--------------------------------------------------------------------------------
Panel A: Training FLOPs in Monty and ViT
--------------------------------------------------------------------------------
"""

def plot_training_flops() -> None:
    """Plot the training FLOPs for Monty and ViT.
    
    Output is saved to `DMC_ANALYSIS_DIR/fig8/training_flops`.
    """
    out_dir = OUT_DIR / "training_flops"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get FLOPs values from data function
    data = load_training_data()

    # Prepare data for plotting
    models = [
        "ViT\n (Pretrained)",
        "ViT",
        "Monty"
    ]
    # Stacked bars for the first entry
    finetune = data["vit_fintune_pretrained"]
    pretrain = data["vit_pretrain_pretrained"]
    flops = [
        [finetune, pretrain],  # Stacked for first
        [data["vit_fintune_scratch"], 0],  # Only one bar for second
        [data["monty_train"], 0]  # Only one bar for third
    ]
    # Color assignments
    stack_colors = [TBP_COLORS["purple"], TBP_COLORS["yellow"]]  # finetune, pretrain
    colors = [TBP_COLORS["purple"], TBP_COLORS["blue"]]  # for single bars: From Scratch, Monty

    # Plot
    fig, ax = plt.subplots(figsize=(7, 3))
    y_pos = np.arange(len(models))
    # Plot stacked bar for first entry
    bar1 = ax.barh(y_pos[0], flops[0][0], color=stack_colors[0], label="Finetuning")
    bar2 = ax.barh(y_pos[0], flops[0][1], left=flops[0][0], color=stack_colors[1], label="Pretraining")
    # Plot single bars for the rest
    ax.barh(y_pos[1], flops[1][0], color=colors[0], label="From Scratch")
    ax.barh(y_pos[2], flops[2][0], color=colors[1], label="Monty")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, ha='center')
    ax.set_xscale('log')
    ax.set_xlabel('Training FLOPs')
    
    # Configure minor ticks for log scale (ticks only, no labels)
    ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.tick_params(axis='x', which='minor', labelbottom=False)
    ax.tick_params(axis='y', which='major', pad=25)
    
    # Add text labels with consistent placement (all outside bars)
    ax.text(flops[0][0]*0.8, y_pos[0], format_flops(finetune), va='center', ha='right', fontsize=7, color='white', rotation=270)
    ax.text(flops[0][0] + flops[0][1]*0.8, y_pos[0], format_flops(pretrain), va='center', ha='right', fontsize=7, color='black', rotation=270)
    ax.text(flops[1][0] * 0.8, y_pos[1], format_flops(flops[1][0]), va='center', ha='right', fontsize=7, color='white', rotation=270)
    ax.text(flops[2][0] * 0.8, y_pos[2], format_flops(flops[2][0]), va='center', ha='right', fontsize=7, color='black', rotation=270)

    # For ViT, add text annotation for Finetuning and Pretraining with more spacing
    ax.text(flops[0][0]*0.5, y_pos[0], 'Finetuning', va='center', ha='right', fontsize=7, color='white', rotation=270)
    ax.text(flops[0][0] + flops[0][1]*0.5, y_pos[0], 'Pretraining', va='center', ha='right', fontsize=7, color='black', rotation=270)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(out_dir / "training_flops.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / "training_flops.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


"""
--------------------------------------------------------------------------------
Panel B: Inference FLOPs vs. Accuracy in Monty and ViT
--------------------------------------------------------------------------------
"""


def plot_inference_flops_vs_accuracy() -> None:
    """Plot the inference FLOPs vs. accuracy for Monty and ViT.

    Output is saved to `DMC_ANALYSIS_DIR/fig8/inference_flops_vs_accuracy`.
    """
    out_dir = OUT_DIR / "inference_flops_vs_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data using utility functions
    hypothesis_testing_exp = "dist_agent_1lm_randrot_x_percent_20_floppy"
    random_walk_exp = "dist_agent_1lm_randrot_nohyp_x_percent_20_floppy"
    hypothesis_testing_flops = load_floppy_traces(DMC_RESULTS_DIR / hypothesis_testing_exp)["flops_mean"].values[0]
    random_walk_flops = load_floppy_traces(DMC_RESULTS_DIR / random_walk_exp)["flops_mean"].values[0]
    hypothesis_testing_df = aggregate_1lm_performance_data([hypothesis_testing_exp])
    hypothesis_testing_df["flops"] = hypothesis_testing_flops
    random_walk_df = aggregate_1lm_performance_data([random_walk_exp])
    random_walk_df["flops"] = random_walk_flops

    # Load ViT predictions
    vit_results_dir = Path("~/tbp/results/dmc/results/vit/logs_reproduction").expanduser()
    vit_b16_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-b16-224-in21k" / "inference" / "predictions.csv")
    vit_b16_pretrained_df = aggregate_vit_predictions(vit_b16_pretrained_df, "ViT-B/16")
    vit_l16_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-l16-224-in21k" / "inference" / "predictions.csv")
    vit_l16_pretrained_df = aggregate_vit_predictions(vit_l16_pretrained_df, "ViT-L/16")
    vit_b32_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-b32-224-in21k" / "inference" / "predictions.csv")
    vit_b32_pretrained_df = aggregate_vit_predictions(vit_b32_pretrained_df, "ViT-B/32")
    vit_l32_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-l32-224-in21k" / "inference" / "predictions.csv")
    vit_l32_pretrained_df = aggregate_vit_predictions(vit_l32_pretrained_df, "ViT-L/32")
    vit_h14_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-h14-224-in21k" / "inference" / "predictions.csv")
    vit_h14_pretrained_df = aggregate_vit_predictions(vit_h14_pretrained_df, "ViT-H/14")
    vit_b16_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-b16-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_b16_scratch_df = aggregate_vit_predictions(vit_b16_scratch_df, "ViT-B/16 (Random Init)")
    vit_l16_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-l16-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_l16_scratch_df = aggregate_vit_predictions(vit_l16_scratch_df, "ViT-L/16 (Random Init)")
    vit_b32_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-b32-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_b32_scratch_df = aggregate_vit_predictions(vit_b32_scratch_df, "ViT-B/32 (Random Init)")
    vit_l32_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-l32-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_l32_scratch_df = aggregate_vit_predictions(vit_l32_scratch_df, "ViT-L/32 (Random Init)")
    vit_h14_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-h14-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_h14_scratch_df = aggregate_vit_predictions(vit_h14_scratch_df, "ViT-H/14 (Random Init)")

    # Concatenate all ViT predictions
    pretrained_vit_df = pd.concat([vit_b16_pretrained_df, vit_l16_pretrained_df, vit_b32_pretrained_df, vit_l32_pretrained_df, vit_h14_pretrained_df])
    scratch_vit_df = pd.concat([vit_b16_scratch_df, vit_l16_scratch_df, vit_b32_scratch_df, vit_l32_scratch_df, vit_h14_scratch_df])

    # Plot Params
    axes_width, axes_height = 4, 3
    broken_axes_ylims = ((0, 12), (48, 100))
    broken_axes_hspace = 0.05
    marker_size = 30
    zorder = 3

    # Create figure with broken axes
    fig = plt.figure(figsize=(axes_width, axes_height))
    bax = brokenaxes(ylims=broken_axes_ylims, hspace=broken_axes_hspace)

    # Create scatter plot
    bax.scatter(hypothesis_testing_df['flops'].values[0], hypothesis_testing_df['accuracy'].values[0], color=TBP_COLORS['blue'], label='Monty', s=marker_size, zorder=zorder)
    bax.scatter(random_walk_df['flops'].values[0], random_walk_df['accuracy'].values[0], color=TBP_COLORS['blue'], label='Monty', s=marker_size, zorder=zorder)
    bax.scatter(pretrained_vit_df['flops'], pretrained_vit_df['accuracy'], color=TBP_COLORS['purple'], label='ViT (Pretrained)', s=marker_size, zorder=zorder)
    bax.scatter(scratch_vit_df['flops'], scratch_vit_df['accuracy'], color=TBP_COLORS['purple'], label='ViT (Random Init)', s=marker_size, zorder=zorder)
    # Add horizontal and vertical lines from Random Walk point
    random_walk_point = random_walk_flops
    
    # Draw horizontal line
    bax.plot([0, random_walk_point], 
             [random_walk_df['accuracy'].values[0], random_walk_df['accuracy'].values[0]], 
             color=TBP_COLORS['blue'], 
             linestyle='--', 
             alpha=0.5, 
             zorder=2)
    
    # Draw vertical line
    bax.plot([random_walk_point, random_walk_point], 
             [random_walk_df['accuracy'], 100], 
             color=TBP_COLORS['blue'], 
             linestyle='--', 
             alpha=0.5, 
             zorder=2)
    
    # Fill the rectangle top left corner
    bax.fill_between([0, random_walk_point], 
                     [random_walk_df['accuracy'].values[0], random_walk_df['accuracy'].values[0]], 
                     [100, 100],
                     color=TBP_COLORS['blue'], 
                     alpha=0.1, 
                     zorder=1)
    
    # Add annotations for Monty points
    bax.annotate('Hypothesis-Testing Policy', 
                (hypothesis_testing_df['flops'], hypothesis_testing_df['accuracy']),
                xytext=(7, 0), textcoords='offset points',
                color=TBP_COLORS['blue'],
                fontsize=6)
    bax.annotate('Random Walk', 
                (random_walk_df['flops'], random_walk_df['accuracy']),
                xytext=(7, 0), textcoords='offset points',
                color=TBP_COLORS['blue'],
                fontsize=6)
    
    # Add annotations for ViT points
    for i, row in pretrained_vit_df.iterrows():
        if row['model'] == 'ViT-L/32' or row['model'] == 'ViT-H/14':
            # Make is lower left
            bax.annotate(row['model'],
                        (row['flops'], row['accuracy']),
                        xytext=(-10, -10), textcoords='offset points',
                        color=TBP_COLORS['purple'],
                        fontsize=6)
        else:
            bax.annotate(row['model'],
                        (row['flops'], row['accuracy']),
                        xytext=(7, 0), textcoords='offset points',
                        color=TBP_COLORS['purple'],
                        fontsize=6)

    bax.set_xscale('log')
    bax.set_xlabel('Inference FLOPs')
    bax.set_ylabel('Accuracy (%)')

    # Set x-axis ticks and limits with LaTeX formatting for superscripts
    # xticks = [5e9, 5e10, 5e11]
    # xticklabels = [r'$5\times10^9$', r'$5\times10^{10}$', r'$5\times10^{11}$']
    # bax.set_xlim(5e9, 5e11)
    # for ax in bax.axs:
    #     ax.set_xticks(xticks)
    #     ax.xaxis.set_major_locator(FixedLocator(xticks))
    #     ax.set_xticklabels(xticklabels)

    bax.legend(loc="lower right", fontsize=6)

    fig.savefig(out_dir / "inference_flops_vs_accuracy.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / "inference_flops_vs_accuracy.pdf", dpi=DPI, bbox_inches="tight")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel C: Inference FLOPs vs. Rotation Error in Monty and ViT
--------------------------------------------------------------------------------
"""

def plot_inference_flops_vs_rotation_error() -> None:
    """Plot the inference FLOPs vs. rotation error for Monty and ViT.
    
    Args:
        monty_exps: List of experiment names for Monty.
        pretrained_vit_exps: List of experiment names for pretrained ViT.
        vit_exps: List of experiment names for ViT.

    Output is saved to `DMC_ANALYSIS_DIR/fig8/inference_flops_vs_rotation_error`.
    """
    out_dir = OUT_DIR / "inference_flops_vs_rotation_error"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data using utility functions
    hypothesis_testing_exp = "dist_agent_1lm_randrot_x_percent_20_floppy"
    random_walk_exp = "dist_agent_1lm_randrot_nohyp_x_percent_20_floppy"
    hypothesis_testing_flops = load_floppy_traces(DMC_RESULTS_DIR / hypothesis_testing_exp)["flops_mean"].values[0]
    random_walk_flops = load_floppy_traces(DMC_RESULTS_DIR / random_walk_exp)["flops_mean"].values[0]
    hypothesis_testing_df = aggregate_1lm_performance_data([hypothesis_testing_exp])
    hypothesis_testing_df["flops"] = hypothesis_testing_flops
    random_walk_df = aggregate_1lm_performance_data([random_walk_exp])
    random_walk_df["flops"] = random_walk_flops

    # Load ViT predictions
    vit_results_dir = Path("~/tbp/results/dmc/results/vit/logs_reproduction").expanduser()
    vit_b16_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-b16-224-in21k" / "inference" / "predictions.csv")
    vit_b16_pretrained_df = aggregate_vit_predictions(vit_b16_pretrained_df, "ViT-B/16")
    vit_l16_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-l16-224-in21k" / "inference" / "predictions.csv")
    vit_l16_pretrained_df = aggregate_vit_predictions(vit_l16_pretrained_df, "ViT-L/16")
    vit_b32_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-b32-224-in21k" / "inference" / "predictions.csv")
    vit_b32_pretrained_df = aggregate_vit_predictions(vit_b32_pretrained_df, "ViT-B/32")
    vit_l32_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-l32-224-in21k" / "inference" / "predictions.csv")
    vit_l32_pretrained_df = aggregate_vit_predictions(vit_l32_pretrained_df, "ViT-L/32")
    vit_h14_pretrained_df = load_vit_predictions(vit_results_dir / "fig8b_vit-h14-224-in21k" / "inference" / "predictions.csv")
    vit_h14_pretrained_df = aggregate_vit_predictions(vit_h14_pretrained_df, "ViT-H/14")
    vit_b16_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-b16-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_b16_scratch_df = aggregate_vit_predictions(vit_b16_scratch_df, "ViT-B/16 (Random Init)")
    vit_l16_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-l16-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_l16_scratch_df = aggregate_vit_predictions(vit_l16_scratch_df, "ViT-L/16 (Random Init)")
    vit_b32_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-b32-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_b32_scratch_df = aggregate_vit_predictions(vit_b32_scratch_df, "ViT-B/32 (Random Init)")
    vit_l32_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-l32-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_l32_scratch_df = aggregate_vit_predictions(vit_l32_scratch_df, "ViT-L/32 (Random Init)")
    vit_h14_scratch_df = load_vit_predictions(vit_results_dir / "fig8b_vit-h14-224-in21k_random_init" / "inference" / "predictions.csv")
    vit_h14_scratch_df = aggregate_vit_predictions(vit_h14_scratch_df, "ViT-H/14 (Random Init)")

    # Concatenate all ViT predictions
    pretrained_vit_df = pd.concat([vit_b16_pretrained_df, vit_l16_pretrained_df, vit_b32_pretrained_df, vit_l32_pretrained_df, vit_h14_pretrained_df])
    scratch_vit_df = pd.concat([vit_b16_scratch_df, vit_l16_scratch_df, vit_b32_scratch_df, vit_l32_scratch_df, vit_h14_scratch_df])

    # Plot Params
    axes_width, axes_height = 4, 3
    marker_size = 30
    zorder = 3

    # Create figure with regular axes
    fig, ax = plt.subplots(figsize=(axes_width, axes_height))

    # Create scatter plot
    ax.scatter(hypothesis_testing_df['flops'].values[0], hypothesis_testing_df['rotation_error.mean'].values[0], color=TBP_COLORS['blue'], label='Monty', s=marker_size, zorder=zorder)
    ax.scatter(random_walk_df['flops'].values[0], random_walk_df['rotation_error.mean'].values[0], color=TBP_COLORS['blue'], label='Monty', s=marker_size, zorder=zorder)
    ax.scatter(pretrained_vit_df['flops'], pretrained_vit_df['rotation_error'], color=TBP_COLORS['purple'], label='ViT (Pretrained)', s=marker_size, zorder=zorder)
    ax.scatter(scratch_vit_df['flops'], scratch_vit_df['rotation_error'], color=TBP_COLORS['purple'], label='ViT (Random Init)', s=marker_size, zorder=zorder)  
    # Add horizontal and vertical lines from Random Walk point
    random_walk_point = random_walk_flops
    
    # Draw horizontal line
    ax.plot([0, random_walk_point], 
            [random_walk_df['rotation_error.mean'].values[0], random_walk_df['rotation_error.mean'].values[0]], 
            color=TBP_COLORS['blue'], 
            linestyle='--', 
            alpha=0.5, 
            zorder=2)
    
    # Draw vertical line
    ax.plot([random_walk_point, random_walk_point], 
            [0, random_walk_df['rotation_error.mean'].values[0]], 
            color=TBP_COLORS['blue'], 
            linestyle='--', 
            alpha=0.5, 
            zorder=2)
    
    # Fill the rectangle top right corner
    ax.fill_between([0, random_walk_point], 
                    [random_walk_df['rotation_error.mean'].values[0], random_walk_df['rotation_error.mean'].values[0]], 
                    [0, 0],
                    color=TBP_COLORS['blue'], 
                    alpha=0.1, 
                    zorder=1)
    
    # Add annotations for Monty points
    ax.annotate('Hypothesis-Testing Policy', 
                (hypothesis_testing_df['flops'], hypothesis_testing_df['rotation_error.mean']),
                        xytext=(0, -10), textcoords='offset points',
                        color=TBP_COLORS['blue'],
                        fontsize=6)
    ax.annotate('Random Walk', 
                (random_walk_df['flops'], random_walk_df['rotation_error.mean']),
                        xytext=(7, 0), textcoords='offset points',
                        color=TBP_COLORS['blue'],
                        fontsize=6)
    
    # Add annotations for ViT points
    for i, row in pretrained_vit_df.iterrows():
        if row['model'] == 'ViT-L/32':
            ax.annotate(row['model'],
                        (row['flops'], row['rotation_error']),
                        xytext=(-20, 16), textcoords='offset points',
                        color=TBP_COLORS['purple'],
                        fontsize=6)
        elif row["model"] == "ViT-B/16":
            ax.annotate(row['model'],
                        (row['flops'], row['rotation_error']),
                        xytext=(0, -10), textcoords='offset points',
                        color=TBP_COLORS['purple'],
                        fontsize=6)
        else:
            ax.annotate(row['model'],
                        (row['flops'], row['rotation_error']),
                        xytext=(-10, -10), textcoords='offset points',
                        color=TBP_COLORS['purple'],
                        fontsize=6)

    ax.set_xscale('log')
    ax.set_xlabel('Inference FLOPs')
    ax.set_ylabel('Rotation Error (deg)')
    ax.set_ylim(0, 180)
    
    # Set x-axis ticks and limits with LaTeX formatting for superscripts
    # xticks = [5e9, 5e10, 5e11]
    # xticklabels = [r'$5\times10^9$', r'$5\times10^{10}$', r'$5\times10^{11}$']
    # ax.set_xlim(5e9, 5e11)
    # ax.set_xticks(xticks)
    # ax.xaxis.set_major_locator(FixedLocator(xticks))
    # ax.set_xticklabels(xticklabels)

    ax.legend(loc="upper right", fontsize=6)

    fig.savefig(out_dir / "inference_flops_vs_rotation_error.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / "inference_flops_vs_rotation_error.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # plot_training_flops()
    
    plot_inference_flops_vs_accuracy()
    # plot_inference_flops_vs_rotation_error()
