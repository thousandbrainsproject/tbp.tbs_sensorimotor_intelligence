import os
from pathlib import Path

import pprint
import numpy as np

import torch

"""
Count the number of parameters in a Monty model.

The Monty model considers of a series of learned graphs, which are started in nested
dictionaries like:
graph = state_dict["lm_dict"][lm_id]["graph_memory"]["mug"][f"patch_{lm_id}"]

A typical graph has the following structure:
'hammer': {'patch': Data(
  x=[1188, 28],
  pos=[1188, 3],
  norm=[1188, 3],
  feature_mapping={
    node_ids=[2],
    pose_vectors=[2],
    pose_fully_defined=[2],
    on_object=[2],
    object_coverage=[2],
    rgba=[2],
    hsv=[2],
    principal_curvatures=[2],
    principal_curvatures_log=[2],
    gaussian_curvature=[2],
    mean_curvature=[2],
    gaussian_curvature_sc=[2],
    mean_curvature_sc=[2]
  },
  edge_index=[2, 13068],
  edge_attr=[13068, 3]
)},

We want to account for all of the features stored at every point
in the graph.

In particular, we need to count all of the floating point parameters, including e.g.
within pose_vectors, there will be 2x arrays of 3 floats each. All values are either
Python floats or numpy arrays. 

The exception is edges, which we can ignore.
"""

# Environment variables.
# Define the root directory for DMC (DeepMind Control) related files
# 1. os.environ.get("DMC_ROOT_DIR", "~/tbp/results/dmc") - Get the value of the DMC_ROOT_DIR 
#    environment variable, or use "~/tbp/results/dmc" as the default if not set
# 2. Path(...) - Convert the string to a Path object for better path manipulation
# 3. .expanduser() - Expand the tilde (~) in the path to the user's home directory
DMC_ROOT_DIR = Path(os.environ.get("DMC_ROOT_DIR", "~/tbp/results/dmc")).expanduser()

model_path = DMC_ROOT_DIR / "pretrained_models/dist_agent_1lm/pretrained/model.pt"

# Use pprint to understand the structure of the model, without printing all of the
# actual parameters.
state_dict = torch.load(model_path)
# pprint.pprint(state_dict)


sample_graph = state_dict["lm_dict"][0]["graph_memory"]["mug"]["patch"]

print(sample_graph)
# Access attributes directly from the Data object
print("Feature mapping keys:", sample_graph.feature_mapping.keys())

for key in sample_graph.feature_mapping.keys():
    print("\nValue of", key)
    print(sample_graph.feature_mapping[key])

# Assuming 'pose_vectors' is a direct attribute holding the tensor data
# (as is common in torch_geometric Data objects, alongside 'x', 'pos', etc.)
# If pose_vectors data is stored elsewhere (e.g., sliced from 'x' based on feature_mapping),
# this would need adjustment based on how the Data object was constructed.
# However, accessing direct attributes is the standard way.
print("Shape of pose_vectors:", len(sample_graph.feature_mapping["pose_vectors"]))
print("Pose vectors:", sample_graph.feature_mapping["pose_vectors"])
