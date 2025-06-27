"""
Script to count FLOPs associated with KDTree construction during model initialization.

# This standalone script is used to measure the FLOPs associated with KDTree construction 
# during model initialization in MontyExperiment. In actual deployments, Monty builds the 
# KD-tree as part of a pre-inference stage (i.e., during model initialization rather than 
# during training or inference). Because of this, Floppy does not track these operations 
# as part of its runtime FLOP counting. This script manually triggers model initialization 
# from a config, including `load_state_dict`, to account for KDTree-related computations.

The FLOP counting formula used is: (5 + k)n * log2(n)
where:
- n = number of points in the graph
- k = dimensionality (3 for x,y,z coordinates)
- Formula represents construction cost for balanced KD-tree
For full details, see the user guide in the Floppy repository at
https://github.com/thousandbrainsproject/tbp.floppy/blob/main/docs/user_guide/index.md

Call chain traced:
KDTree -> set_graph -> _initialize_model_with_graph -> _add_graph_to_memory -> load_state_dict
"""

import copy
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from scipy.spatial import KDTree
import pandas as pd
from datetime import datetime

# Setup paths to access monty modules
current_dir = Path(__file__).parent
monty_path = current_dir.parent.parent / "tbp.monty"
tbs_monty_path = current_dir.parent / "monty"
sys.path.insert(0, str(monty_path / "src"))
sys.path.insert(0, str(tbs_monty_path))

# Import required monty modules
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.models.object_model import GridObjectModel
from configs.common import DMC_RESULTS_DIR

# Import config loading utilities from local configs
try:
    from configs import CONFIGS
    print(f"Successfully loaded {len(CONFIGS)} configs from local configs module")
except ImportError as e:
    print(f"Warning: Could not import local configs: {e}")
    CONFIGS = {}
    print("CONFIGS will be empty. Make sure you're running from the correct directory.")


class KDTreeFlopCounter:
    """Simple FLOP counter for KDTree construction operations."""
    
    def __init__(self):
        self.total_flops = 0
        self.kdtree_constructions = []
        self.objects_processed = []
    
    def reset(self):
        """Reset the counter for a new measurement."""
        self.total_flops = 0
        self.kdtree_constructions.clear()
        self.objects_processed.clear()
    
    def count_kdtree_construction(self, points, object_name="unknown"):
        """
        Count FLOPs for KDTree construction.
        
        Args:
            points: Array of points used to construct the KDTree
            object_name: Name of the object being processed
            
        Returns:
            int: Number of FLOPs for this construction
        """
        n = len(points)
        if n <= 1:
            return 0
            
        # Get dimensionality (should be 3 for x,y,z coordinates)
        k = points.shape[1] if hasattr(points, 'shape') and len(points.shape) > 1 else 3
        
        # Calculate FLOPs using the formula: (5 + k)n * log2(n)
        flops = int((5 + k) * n * np.log2(n))
        
        self.total_flops += flops
        self.kdtree_constructions.append({
            'object_name': object_name,
            'num_points': n,
            'dimensionality': k,
            'flops': flops
        })
        self.objects_processed.append(object_name)
        
        print(f"KDTree construction for {object_name}: {n} points, {k}D -> {flops:,} FLOPs")
        return flops
    
    def get_summary(self):
        """Get a summary of all FLOP counts."""
        return {
            'total_flops': self.total_flops,
            'num_objects': len(set(self.objects_processed)),
            'constructions': self.kdtree_constructions
        }


# Global FLOP counter instance
flop_counter = KDTreeFlopCounter()


def patch_kdtree_construction():
    """
    Patch KDTree construction in GridObjectModel to count FLOPs.
    
    This monkey-patches the set_graph method to intercept KDTree constructions.
    """
    original_set_graph = GridObjectModel.set_graph
    
    def counting_set_graph(self, graph):
        """Wrapped set_graph method that counts KDTree construction FLOPs."""
        # Call the original method
        result = original_set_graph(self, graph)
        
        # If a KDTree was constructed, count the FLOPs
        if hasattr(self, '_location_tree') and self._location_tree is not None:
            if hasattr(self, '_graph') and self._graph is not None:
                points = self._graph.pos
                if points is not None and len(points) > 1:
                    flop_counter.count_kdtree_construction(
                        points=points,
                        object_name=getattr(self, 'object_id', 'unknown')
                    )
        
        return result
    
    # Apply the patch
    GridObjectModel.set_graph = counting_set_graph


def load_config_by_name(config_name: str) -> Dict[str, Any]:
    """
    Load an experiment config by name from the available config groups.
    
    Args:
        config_name: Name of the experiment config (e.g., "base_config_10distinctobj_surf_agent")
        
    Returns:
        dict: The experiment configuration
        
    Raises:
        ValueError: If the config name is not found
    """
    if not CONFIGS:
        raise ImportError(
            "Could not import config loading utilities. "
            "Make sure you're running from the correct directory with access to configs"
        )
    
    try:
        # Try to find the config in the CONFIGS dictionary
        if config_name in CONFIGS:
            return CONFIGS[config_name]
        else:
            # If not found directly, list available configs for helpful error message
            available_configs = list(CONFIGS.keys())
            raise ValueError(
                f"Config '{config_name}' not found. Available configs: {available_configs[:10]}..."
                f" (showing first 10 of {len(available_configs)} total)"
            )
    except Exception as e:
        raise ValueError(f"Error loading config '{config_name}': {e}")


def extract_model_path_from_config(config: Dict[str, Any]) -> str:
    """
    Extract the model path from an experiment config.
    
    Args:
        config: The experiment configuration dictionary
        
    Returns:
        str: Path to the model file
        
    Raises:
        ValueError: If no model path is found in the config
    """
    model_path = None
    
    # Check experiment_args for model_name_or_path
    if "experiment_args" in config:
        exp_args = config["experiment_args"]
        if hasattr(exp_args, 'model_name_or_path'):
            model_path = exp_args.model_name_or_path
        elif isinstance(exp_args, dict) and "model_name_or_path" in exp_args:
            model_path = exp_args["model_name_or_path"]
    
    if not model_path:
        raise ValueError(
            "No model path found in config. Make sure the config has 'experiment_args' "
            "with 'model_name_or_path' specified."
        )
    
    # Convert to full path if it's just a directory
    if os.path.isdir(model_path):
        model_file = os.path.join(model_path, "model.pt")
        if os.path.exists(model_file):
            model_path = model_file
        else:
            raise FileNotFoundError(f"Model file not found at {model_file}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model_path


def count_model_init_flops_from_config(config_name: str) -> Dict[str, Any]:
    """
    Initialize a MontyExperiment model from a config name and count FLOPs for KDTree construction.
    
    Args:
        config_name: Name of the experiment config
        
    Returns:
        dict: Summary of FLOP counts
    """
    # Load the config by name
    config = load_config_by_name(config_name)
    
    model_path = extract_model_path_from_config(config)
    
    flop_counter.reset()
    
    patch_kdtree_construction()
    
    # Modify config to avoid running actual training/evaluation
    modified_config = copy.deepcopy(config)
    
    # Ensure we're only loading the model, not training or evaluating
    if "experiment_args" in modified_config:
        modified_config["experiment_args"].do_train = False
        modified_config["experiment_args"].do_eval = False
        modified_config["experiment_args"].n_train_epochs = 0
        modified_config["experiment_args"].n_eval_epochs = 0
    
    # Initialize the experiment
    experiment = MontyExperiment()
    
    try:
        with experiment:
            # This will trigger model initialization and load_state_dict
            experiment.setup_experiment(modified_config)
    except Exception as e:
        print(f"Error during model setup_experiment: {e}")
        raise e
    
    # Get the summary
    summary = flop_counter.get_summary()
    summary['config_name'] = config_name
    summary['model_path'] = model_path
    return summary


def save_summary_to_csv(summary: Dict[str, Any], output_file: str = None):
    """
    Save the FLOP counting summary to a CSV file.
    
    Args:
        summary: Dictionary containing the FLOP counting results
        output_file: Optional path to output file. If None, generates a timestamped filename in DMC_RESULTS_DIR.
    """
    if output_file is None:
        # Generate timestamped filename in DMC_RESULTS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = DMC_RESULTS_DIR / f"flop_counting_results_{timestamp}.csv"
    
    # Create a list of dictionaries for each construction
    rows = []
    for const in summary['constructions']:
        row = {
            'config_name': summary['config_name'],
            'model_path': summary['model_path'],
            'total_flops': summary['total_flops'],
            'num_objects': summary['num_objects'],
            'object_name': const['object_name'],
            'num_points': const['num_points'],
            'dimensionality': const['dimensionality'],
            'object_flops': const['flops']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


def main():
    """Main function to run the FLOP counting."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Count FLOPs for KDTree construction during model initialization"
    )
    parser.add_argument(
        "-e", "--experiment", 
        type=str,
        required=True,
        help="Name of the experiment config"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help=f"Output CSV file path (default: {DMC_RESULTS_DIR}/flop_counting_results_<timestamp>.csv)"
    )
    
    args = parser.parse_args()
    
    try:
        # Count the FLOPs
        summary = count_model_init_flops_from_config(args.experiment)
        
        # Save to CSV
        save_summary_to_csv(summary, args.output)
        
        return summary
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 