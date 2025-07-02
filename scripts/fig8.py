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

from plot_utils import format_flops

from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_RESULTS_DIR,
    DMC_ROOT_DIR,
    load_eval_stats,
    load_floppy_traces,
    load_vit_predictions,
    aggregate_1lm_performance_data,
    aggregate_vit_predictions,
)
from plot_utils import (
    TBP_COLORS,
    init_matplotlib_style,
)
init_matplotlib_style()


# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600

"""
--------------------------------------------------------------------------------
Utilities
--------------------------------------------------------------------------------
"""
def prepare_fig8a_data() -> pd.DataFrame:
    """Load training FLOPs data for Figure 8a from saved CSV files and Monty traces.
    
    This function reads ViT FLOPs from fig8a_results.csv and loads Monty training FLOPs
    from floppy traces in the dist_agent_1lm_floppy directory.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['model_name', 'total_flops', 'model_type']
                     where rows represent different models.
    """
    results = []
    
    # Load ViT FLOPs from CSV file
    try:
        vit_csv_path = DMC_RESULTS_DIR / "fig8a_results.csv"
        vit_df = pd.read_csv(vit_csv_path)
        # Assume CSV has columns: flops, model
        results.append(vit_df)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {vit_csv_path}. Please run `count_vit_flops.py` to compute ViT training FLOPs.")
    
    # Load Monty training FLOPs using load_floppy_traces
    monty_exp_path = DMC_ROOT_DIR / "pretrained_models" / "dist_agent_1lm_floppy"
    try:
        monty_flops_df = load_floppy_traces(monty_exp_path)
        monty_data = pd.DataFrame({
            'model': ['Monty'],
            'flops': [monty_flops_df['total_flops'].iloc[0]],
        })
        results.append(monty_data)
    except (FileNotFoundError, KeyError, IndexError) as e:
        raise FileNotFoundError(f"Could not load Monty floppy traces from {monty_exp_path}. Error: {e}")

    # combine results and save to csv in out_dir
    combined_data = pd.concat(results, ignore_index=True)
    combined_data.to_csv(OUT_DIR / "fig8a_training_flops.csv", index=False)
    print(f"Saved ViT and Monty training FLOPs data to {OUT_DIR / 'fig8a_training_flops.csv'}. This will be used for plotting.")
    return combined_data


def prepare_fig8b_data() -> pd.DataFrame:
    """Load inference FLOPs, accuracy, and rotation error data for Figure 8b.
    
    This function reads ViT inference FLOPs from fig8b_results.csv, loads ViT predictions
    from various model directories, loads Monty performance data from experiment directories,
    and combines all data into a single DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['model', 'accuracy', 'rotation_error', 'flops', 'model_type']
                     where rows represent different models and experiments.
    """
    results = []
    
    # Load ViT inference FLOPs from CSV file
    try:
        vit_csv_path = DMC_RESULTS_DIR / "fig8b_results.csv"
        vit_flops_df = pd.read_csv(vit_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {vit_csv_path}. Please run the appropriate script to compute ViT inference FLOPs.")
    
    # ViT model configurations
    vit_models = ['b32', 'b16', 'l32', 'l16', 'h14']
    vit_results_dir = Path("~/tbp/results/dmc/results/vit/logs").expanduser()
    
    # Process ViT models (both pretrained and random_init versions)
    for model in vit_models:
        model_name = f"vit-{model}-224-in21k"
        
        # Format display name with slash (e.g., 'b32' -> 'ViT-B/32')
        model_size = model[0].upper()  # 'b' -> 'B', 'l' -> 'L', 'h' -> 'H'
        patch_size = model[1:]  # '32', '16', '14'
        display_name = f"ViT-{model_size}/{patch_size}"
        
        # Process pretrained version
        pretrained_path = vit_results_dir / f"fig8b_{model_name}" / "inference" / "predictions.csv"
        if pretrained_path.exists():
            try:
                predictions_df = load_vit_predictions(pretrained_path)
                # Get FLOPs for this model from the CSV
                model_flops = None
                if not vit_flops_df.empty:
                    flops_row = vit_flops_df[vit_flops_df['model'].str.contains(model, case=False, na=False)]
                    if not flops_row.empty:
                        model_flops = flops_row['flops'].iloc[0]
                
                vit_data = aggregate_vit_predictions(predictions_df, display_name, model_flops)
                vit_data['model_type'] = 'ViT (Pretrained)'
                results.append(vit_data)
            except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                print(f"Warning: Could not load pretrained ViT {model} predictions: {e}")
        
        # Process random_init version
        random_init_path = vit_results_dir / f"fig8b_{model_name}_random_init" / "inference" / "predictions.csv"
        if random_init_path.exists():
            try:
                predictions_df = load_vit_predictions(random_init_path)
                # Use same FLOPs as pretrained (inference FLOPs should be the same)
                model_flops = None
                if not vit_flops_df.empty:
                    flops_row = vit_flops_df[vit_flops_df['model'].str.contains(model, case=False, na=False)]
                    if not flops_row.empty:
                        model_flops = flops_row['flops'].iloc[0]
                
                vit_data = aggregate_vit_predictions(predictions_df, f"{display_name} (Random Init)", model_flops)
                vit_data['model_type'] = 'ViT (Random Init)'
                results.append(vit_data)
            except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                print(f"Warning: Could not load random_init ViT {model} predictions: {e}")
    
    # Process Monty experiments
    # Hypothesis testing experiment
    hyp_exp_name = "dist_agent_1lm_randrot_x_percent_20_floppy"
    try:
        hyp_flops_df = load_floppy_traces(DMC_RESULTS_DIR / hyp_exp_name)
        hyp_performance_df = aggregate_1lm_performance_data([hyp_exp_name])
        
        monty_hyp_data = pd.DataFrame({
            'model': ['Monty (Hypothesis Testing)'],
            'accuracy': [hyp_performance_df['accuracy'].iloc[0]],
            'rotation_error': [hyp_performance_df['rotation_error.mean'].iloc[0]],
            'flops': [hyp_flops_df['flops_mean'].iloc[0]],
            'model_type': ['Monty']
        })
        results.append(monty_hyp_data)
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"Warning: Could not load Monty hypothesis testing data: {e}")
    
    # No hypothesis testing experiment (random walk)
    no_hyp_exp_name = "dist_agent_1lm_randrot_nohyp_x_percent_20_floppy"
    try:
        no_hyp_flops_df = load_floppy_traces(DMC_RESULTS_DIR / no_hyp_exp_name)
        no_hyp_performance_df = aggregate_1lm_performance_data([no_hyp_exp_name])
        
        monty_no_hyp_data = pd.DataFrame({
            'model': ['Monty (Random Walk)'],
            'accuracy': [no_hyp_performance_df['accuracy'].iloc[0]],
            'rotation_error': [no_hyp_performance_df['rotation_error.mean'].iloc[0]],
            'flops': [no_hyp_flops_df['flops_mean'].iloc[0]],
            'model_type': ['Monty']
        })
        results.append(monty_no_hyp_data)
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"Warning: Could not load Monty no hypothesis testing data: {e}")
    
    # Combine all results
    if not results:
        raise ValueError("No data could be loaded for any models. Please check the data paths and files.")
    
    combined_data = pd.concat(results, ignore_index=True)
    
    # Save to CSV
    output_path = OUT_DIR / "fig8b_inference_data.csv"
    combined_data.to_csv(output_path, index=False)
    print(f"Saved Figure 8b inference data to {output_path}")
    
    return combined_data





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

    # Load FLOPs data using new data loading function
    data_df = prepare_fig8a_data()

    # Extract data from DataFrame
    model_data = {}
    for _, row in data_df.iterrows():
        model_data[row['model']] = row['flops']
    
    # Prepare data for plotting
    models = [
        "ViT\n (Pretrained)",
        "ViT",
        "Monty"
    ]
    
    # Extract specific model data based on the CSV structure you provided
    vit_scratch_flops = model_data.get('vit-b16_scratch_training_flops')
    vit_finetuning_flops = model_data.get('vit-b16_pretrained_tuning_flops')  
    vit_pretraining_flops = model_data.get('pretraining_flops')
    monty_flops = model_data.get('Monty')
    
    # Check if we found all required data
    if vit_scratch_flops is None:
        raise ValueError(f"Could not find 'vit-b16_scratch_training_flops' in data. Available models: {list(model_data.keys())}")
    if vit_finetuning_flops is None:
        raise ValueError(f"Could not find 'vit-b16_pretrained_tuning_flops' in data. Available models: {list(model_data.keys())}")
    if vit_pretraining_flops is None:
        raise ValueError(f"Could not find 'pretraining_flops' in data. Available models: {list(model_data.keys())}")
    if monty_flops is None:
        raise ValueError(f"Could not find 'Monty' in data. Available models: {list(model_data.keys())}")
    
    # Prepare flops data for plotting
    # ViT (Pretrained) = finetuning + pretraining (stacked)
    # ViT (From Scratch) = scratch training only
    # Monty = monty training only
    flops = [
        [vit_finetuning_flops, vit_pretraining_flops],  # Stacked for ViT (Pretrained)
        [vit_scratch_flops, 0],  # Only one bar for ViT (From Scratch)
        [monty_flops, 0]  # Only one bar for Monty
    ]
    # Color assignments
    stack_colors = [TBP_COLORS["purple"], TBP_COLORS["yellow"]]  # finetune, pretrain
    colors = [TBP_COLORS["purple"], TBP_COLORS["blue"]]  # for single bars: From Scratch, Monty

    # Plot
    fig, ax = plt.subplots(figsize=(7, 3))
    y_pos = np.arange(len(models))
    # Plot stacked bar for first entry
    ax.barh(y_pos[0], flops[0][0], color=stack_colors[0], label="Finetuning")
    ax.barh(y_pos[0], flops[0][1], left=flops[0][0], color=stack_colors[1], label="Pretraining")
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
    ax.text(flops[0][0]*0.8, y_pos[0], format_flops(vit_finetuning_flops), va='center', ha='right', fontsize=7, color='white', rotation=270)
    ax.text(flops[0][0] + flops[0][1]*0.8, y_pos[0], format_flops(vit_pretraining_flops), va='center', ha='right', fontsize=7, color='black', rotation=270)
    ax.text(flops[1][0] * 0.8, y_pos[1], format_flops(flops[1][0]), va='center', ha='right', fontsize=7, color='white', rotation=270)
    ax.text(flops[2][0] * 0.8, y_pos[2], format_flops(flops[2][0]), va='center', ha='right', fontsize=7, color='black', rotation=270)

    # For ViT, add text annotation for Finetuning and Pretraining with more spacing
    ax.text(flops[0][0]*0.4, y_pos[0], 'Finetuning', va='center', ha='right', fontsize=7, color='white', rotation=270)
    ax.text(flops[0][0] + flops[0][1]*0.4, y_pos[0], 'Pretraining', va='center', ha='right', fontsize=7, color='black', rotation=270)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(out_dir / "training_flops.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / "training_flops.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training FLOPs plot to {out_dir / 'training_flops.png'}")


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

    # Load consolidated data
    data_df = prepare_fig8b_data()
    
    # Filter data for different model types
    monty_data = data_df[data_df['model_type'] == 'Monty']
    pretrained_vit_df = data_df[data_df['model_type'] == 'ViT (Pretrained)']
    scratch_vit_df = data_df[data_df['model_type'] == 'ViT (Random Init)']
    
    # Extract specific Monty experiments
    hypothesis_testing_df = monty_data[monty_data['model'] == 'Monty (Hypothesis Testing)']
    random_walk_df = monty_data[monty_data['model'] == 'Monty (Random Walk)']
    
    # Get random walk FLOPs for plotting lines
    if not random_walk_df.empty:
        random_walk_flops = random_walk_df['flops'].values[0]
    else:
        print("Warning: No Random Walk data found for plotting reference lines")
        random_walk_flops = 0

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
    bax.scatter(random_walk_df['flops'].values[0], random_walk_df['accuracy'].values[0], color=TBP_COLORS['blue'], s=marker_size, zorder=zorder)  # No label to avoid duplicate
    bax.scatter(pretrained_vit_df['flops'], pretrained_vit_df['accuracy'], color=TBP_COLORS['purple'], label='ViT (Pretrained)', s=marker_size, zorder=zorder)
    bax.scatter(scratch_vit_df['flops'], scratch_vit_df['accuracy'], facecolors='none', edgecolors=TBP_COLORS['purple'], label='ViT (Random Init)', s=marker_size, zorder=zorder+10)
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
                xytext=(7, 2), textcoords='offset points',
                color=TBP_COLORS['blue'],
                fontsize=6)
    bax.annotate('Random Walk', 
                (random_walk_df['flops'], random_walk_df['accuracy']),
                xytext=(7, 0), textcoords='offset points',
                color=TBP_COLORS['blue'],
                fontsize=6)
    
    # Add annotations for ViT points (pretrained)
    for _, row in pretrained_vit_df.iterrows():
        if row['model'] == 'ViT-L/32':
            bax.annotate(row['model'],
                        (row['flops'], row['accuracy']),
                        xytext=(7, -7), textcoords='offset points',
                        color=TBP_COLORS['purple'],
                        fontsize=6)
        else:
            bax.annotate(row['model'],
                    (row['flops'], row['accuracy']),
                        xytext=(7, 0), textcoords='offset points',
                        color=TBP_COLORS['purple'],
                        fontsize=6)
    
    
    # Add annotations for ViT points (random init)
    for _, row in scratch_vit_df.iterrows():
        # Extract base model name from 'ViT-B/32 (Random Init)' format
        model_name = row['model'].replace(' (Random Init)', '')
        bax.annotate(model_name,
                    (row['flops'], row['accuracy']),
                    xytext=(7, 0), textcoords='offset points',
                    color=TBP_COLORS['purple'],
                    fontsize=6)

    bax.set_xscale('log')
    bax.set_xlabel('Inference FLOPs')
    bax.set_ylabel('Accuracy (%)')

    # Set x-axis ticks and limits with LaTeX formatting for superscripts
    xticks = [5e9, 5e10, 5e11]
    xticklabels = [r'$5\times10^9$', r'$5\times10^{10}$', r'$5\times10^{11}$']
    bax.set_xlim(4e9, 5e11)
    for ax in bax.axs:
        ax.set_xticks(xticks)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.set_xticklabels(xticklabels)

    bax.legend(loc="lower right", fontsize=6)

    fig.savefig(out_dir / "inference_flops_vs_accuracy.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / "inference_flops_vs_accuracy.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()


"""
--------------------------------------------------------------------------------
Panel C: Inference FLOPs vs. Rotation Error in Monty and ViT
--------------------------------------------------------------------------------
"""

def plot_inference_flops_vs_rotation_error() -> None:
    """Plot the inference FLOPs vs. rotation error for Monty and ViT.
    
    Output is saved to `DMC_ANALYSIS_DIR/fig8/inference_flops_vs_rotation_error`.
    """
    out_dir = OUT_DIR / "inference_flops_vs_rotation_error"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load consolidated data
    data_df = prepare_fig8b_data()
    
    # Filter data for different model types
    monty_data = data_df[data_df['model_type'] == 'Monty']
    pretrained_vit_df = data_df[data_df['model_type'] == 'ViT (Pretrained)']
    scratch_vit_df = data_df[data_df['model_type'] == 'ViT (Random Init)']
    
    # Extract specific Monty experiments
    hypothesis_testing_df = monty_data[monty_data['model'] == 'Monty (Hypothesis Testing)']
    random_walk_df = monty_data[monty_data['model'] == 'Monty (Random Walk)']
    
    # Get random walk FLOPs for plotting lines
    if not random_walk_df.empty:
        random_walk_flops = random_walk_df['flops'].values[0]
    else:
        print("Warning: No Random Walk data found for plotting reference lines")
        random_walk_flops = 0

    # Plot Params
    axes_width, axes_height = 4, 3
    marker_size = 30
    zorder = 3

    # Create figure with regular axes
    fig, ax = plt.subplots(figsize=(axes_width, axes_height))

    # Create scatter plot
    ax.scatter(hypothesis_testing_df['flops'].values[0], hypothesis_testing_df['rotation_error'].values[0], color=TBP_COLORS['blue'], label='Monty', s=marker_size, zorder=zorder)
    ax.scatter(random_walk_df['flops'].values[0], random_walk_df['rotation_error'].values[0], color=TBP_COLORS['blue'], s=marker_size, zorder=zorder)  # No label to avoid duplicate
    # Plot pretrained ViT first with lower zorder
    ax.scatter(pretrained_vit_df['flops'], pretrained_vit_df['rotation_error'], color=TBP_COLORS['purple'], label='ViT (Pretrained)', s=marker_size, zorder=zorder)
    # Plot scratch ViT last with higher zorder and thicker edge
    ax.scatter(scratch_vit_df['flops'], scratch_vit_df['rotation_error'], facecolors='white', edgecolors=TBP_COLORS['purple'], label='ViT (Random Init)', s=marker_size, zorder=zorder)
    
    # Add horizontal and vertical lines from Random Walk point
    random_walk_point = random_walk_flops
    
    # Draw horizontal line
    ax.plot([0, random_walk_point], 
            [random_walk_df['rotation_error'].values[0], random_walk_df['rotation_error'].values[0]], 
            color=TBP_COLORS['blue'], 
            linestyle='--', 
            alpha=0.5, 
            zorder=2)
    
    # Draw vertical line
    ax.plot([random_walk_point, random_walk_point], 
            [0, random_walk_df['rotation_error'].values[0]], 
            color=TBP_COLORS['blue'], 
            linestyle='--', 
            alpha=0.5, 
            zorder=2)
    
    # Fill the rectangle top right corner
    ax.fill_between([0, random_walk_point], 
                    [random_walk_df['rotation_error'].values[0], random_walk_df['rotation_error'].values[0]], 
                    [0, 0],
                    color=TBP_COLORS['blue'], 
                    alpha=0.1, 
                    zorder=1)
    
    # Add annotations for Monty points
    ax.annotate('Hypothesis-Testing Policy', 
                (hypothesis_testing_df['flops'], hypothesis_testing_df['rotation_error']),
                        xytext=(0, -10), textcoords='offset points',
                        color=TBP_COLORS['blue'],
                        fontsize=6)
    ax.annotate('Random Walk', 
                (random_walk_df['flops'], random_walk_df['rotation_error']),
                        xytext=(7, 0), textcoords='offset points',
                        color=TBP_COLORS['blue'],
                        fontsize=6)
    
    # Add annotations for ViT points (pretrained)
    for _, row in pretrained_vit_df.iterrows():
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
    
    # Add annotations for ViT points (random init)
    # for _, row in scratch_vit_df.iterrows():
    #     # Extract base model name from 'ViT-B/32 (Random Init)' format
    #     model_name = row['model'].replace(' (Random Init)', '')
    #     if model_name == 'ViT-L/32':
    #         ax.annotate(model_name,
    #                     (row['flops'], row['rotation_error']),
    #                     xytext=(-20, 16), textcoords='offset points',
    #                     color=TBP_COLORS['purple'],
    #                     fontsize=6)
    #     elif model_name == "ViT-B/16":
    #         ax.annotate(model_name,
    #                     (row['flops'], row['rotation_error']),
    #                     xytext=(0, -10), textcoords='offset points',
    #                     color=TBP_COLORS['purple'],
    #                     fontsize=6)
    #     else:
    #         ax.annotate(model_name,
    #                     (row['flops'], row['rotation_error']),
    #                     xytext=(-10, -10), textcoords='offset points',
    #                     color=TBP_COLORS['purple'],
    #                     fontsize=6)

    ax.set_xscale('log')
    ax.set_xlabel('Inference FLOPs')
    ax.set_ylabel('Rotation Error (deg)')
    ax.set_ylim(0, 180)
    
    # Set x-axis ticks and limits with LaTeX formatting for superscripts
    xticks = [5e9, 5e10, 5e11]
    xticklabels = [r'$5\times10^9$', r'$5\times10^{10}$', r'$5\times10^{11}$']
    ax.set_xlim(4e9, 5e11)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.set_xticklabels(xticklabels)

    ax.legend(loc="upper right", fontsize=6)

    fig.savefig(out_dir / "inference_flops_vs_rotation_error.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / "inference_flops_vs_rotation_error.pdf", dpi=DPI, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_training_flops()
    
    # plot_inference_flops_vs_accuracy()
    # plot_inference_flops_vs_rotation_error()
