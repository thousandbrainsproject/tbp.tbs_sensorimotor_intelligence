#!/usr/bin/env python3
"""
Script to analyze FLOPs for floppy experiment results.

This script loads and analyzes FLOP traces from a specified experiment directory
using the load_floppy_traces function from data_utils.py.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.data_utils import load_floppy_traces

def analyze_flops(experiment_path: str):
    """
    Analyze FLOPs data for a given experiment.
    
    Args:
        experiment_path (str): Path to the experiment directory
    """
    print(f"Analyzing FLOPs for experiment: {experiment_path}")
    print("=" * 60)
    
    try:
        # Load the floppy traces data
        flops_df = load_floppy_traces(experiment_path)
        
        # Display basic statistics
        print("Basic FLOP Statistics:")
        print("-" * 30)
        print(f"Experiment: {flops_df['experiment'].iloc[0]}")
        print(f"Mean FLOPs per episode: {flops_df['flops_mean'].iloc[0]:,.2f}")
        print(f"Standard deviation: {flops_df['flops_std'].iloc[0]:,.2f}")
        print()
        
        # Load individual flop trace files for more detailed analysis
        exp_path = Path(experiment_path)
        if not exp_path.exists():
            from data_utils import DMC_RESULTS_DIR
            exp_path = DMC_RESULTS_DIR / experiment_path
        
        print("Detailed Analysis:")
        print("-" * 30)
        
        # Find all flop trace files
        flop_files = list(exp_path.glob("flop_traces*.csv"))
        print(f"Found {len(flop_files)} flop trace files")
        
        if flop_files:
            # Load and analyze individual files
            all_flops = []
            all_methods = []
            
            for file in flop_files:
                print(f"Processing {file.name}...")
                df = pd.read_csv(file)
                
                # Filter for experiment.run_episode method
                run_episode_df = df[df["method"] == "experiment.run_episode"]
                
                if not run_episode_df.empty:
                    episode_flops = run_episode_df["flops"].tolist()
                    all_flops.extend(episode_flops)
                    all_methods.extend(run_episode_df["method"].tolist())
                    
                    print(f"  Episodes in file: {len(episode_flops)}")
                    print(f"  Mean FLOPs: {np.mean(episode_flops):,.2f}")
                    print(f"  Min FLOPs: {np.min(episode_flops):,.2f}")
                    print(f"  Max FLOPs: {np.max(episode_flops):,.2f}")
            
            if all_flops:
                print()
                print("Overall Statistics:")
                print("-" * 30)
                print(f"Total episodes analyzed: {len(all_flops)}")
                print(f"Overall mean FLOPs: {np.mean(all_flops):,.2f}")
                print(f"Total FLOPs: {np.sum(all_flops):,.2f}")
                print(f"Overall std FLOPs: {np.std(all_flops):,.2f}")
                print(f"Overall median FLOPs: {np.median(all_flops):,.2f}")
                print(f"Overall min FLOPs: {np.min(all_flops):,.2f}")
                print(f"Overall max FLOPs: {np.max(all_flops):,.2f}")
                
                # Create a simple histogram
                plt.figure(figsize=(10, 6))
                plt.hist(all_flops, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('FLOPs per Episode')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of FLOPs per Episode\n{exp_path.name}')
                plt.grid(True, alpha=0.3)
                
                # Add statistics to the plot
                plt.axvline(np.mean(all_flops), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_flops):,.0f}')
                plt.axvline(np.median(all_flops), color='green', linestyle='--', 
                           label=f'Median: {np.median(all_flops):,.0f}')
                plt.legend()
                
                # Save the plot
                output_file = f"flops_analysis_{exp_path.name}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"\nHistogram saved as: {output_file}")
                plt.show()
            else:
                print("No experiment.run_episode data found in flop trace files")
        else:
            print("No flop trace files found in the experiment directory")
            
    except Exception as e:
        print(f"Error analyzing FLOPs: {e}")
        return None
    
    return flops_df

def main():
    """Main function to run the FLOP analysis."""
    
    # Default experiment path
    default_path = "/Users/hlee/tbp/results/dmc/results/dist_agent_1lm_randrot_nohyp_x_percent_20_floppy"
    default_path = "/Users/hlee/tbp/results/dmc/results/dist_agent_1lm_randrot_x_percent_20_floppy"

    # Allow command line argument for different path
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
    else:
        experiment_path = default_path
    
    print("FLOP Analysis Script")
    print("=" * 60)
    print(f"Target experiment: {experiment_path}")
    print()
    
    # Run the analysis
    result = analyze_flops(experiment_path)
    
    if result is not None:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed. Please check the experiment path and data files.")

if __name__ == "__main__":
    main() 