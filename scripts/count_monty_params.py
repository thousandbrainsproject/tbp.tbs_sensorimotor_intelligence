# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""This script counts the number of parameters in a Monty model, specifically the
dist_agent_1lm model. This is a result (4 million parameters) referenced in the
Continual Learning section of the paper.

The Monty model consists of a series of learned graphs for different objects. These
are stored in a nested state dictionary, e.g.:

    graph = state_dict["lm_dict"][lm_id]["graph_memory"]["mug"]["patch"]

This script will count all the numerical values stored in these graphs, as a measure
of the total number of parameters in the model.

For clarity, we begin by breaking down the structure of the state dictionary.

A Monty model's state_dict contains the following keys:
- 'lm_dict'
- 'sm_dict'
- 'motor_system_dict'
- 'lm_to_lm_matrix'
- 'lm_to_lm_vote_matrix'
- 'sm_to_lm_matrix'

All of the information we are concerned with is in lm_dict. sm_dict and
motor_system_dict contain information for logging and visualization purposes. The
connectivity information in lm_to_lm_matrix, lm_to_lm_vote_matrix, and sm_to_lm_matrix
is a negligible fraction of data, particularly in single-LM models.

Within lm_dict, lm_id is the index of the Learning Module(s), of which only 0 is
relevant for single-LM models. lm_dict does not contain any other keys.

state_dict["lm_dict"][lm_id] contains:
- 'graph_memory'
- 'target_to_graph_id'
- 'graph_id_to_target'
The latter two are mappings between the object name and the index of the graph, which
are needed to assess the performance of the model. They are of negligible size.

state_dict["lm_dict"][lm_id]["graph_memory"] contains keys corresponding
to all of the learned objects (e.g. 77 in the case of YCB).

Each graph key (e.g. state_dict["lm_dict"][lm_id]["graph_memory"]["hammer"])
is associated with a key corresponding to the sensory channel that was used to learn the
object. This is typically "patch", but could be indexed (e.g. "patch_0", "patch_1") in
multi-LM (hierarchical or voting) models.

A graph can therefore be accessed with e.g.
state_dict["lm_dict"][lm_id]["graph_memory"]["hammer"]["patch"].

The accessed graph is a torch_geometric Data object, which has the following structure:

Data(
    x=[1188, 28],  # 28 features per node; not all of these are used in inference,
    # but for simplicity we count all of them.
    pos=[1188, 3],  # Position of each node in an object's reference frame
    norm=[1188, 3],  # The norm of each node; this is redundant with the norm in x,
    # and is in general never accessed directly during inference.

    # feature_mapping stores the mapping between the array indices, and the features
    # they represent. For example, "pose_vectors" has values [1, 10], indicating that
    # indices 1:10 in x store the pose vectors of the node. These are stored as utility
    # variables as Monty frequently switches between using the full x array, vs indexing
    # individual values.
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

    # Edges are not used in Monty models at the moment and so can be ignored.
    edge_index=[2, 13068],
    edge_attr=[13068, 3]
)
"""

import os
from pathlib import Path

import torch

# Get the value of the DMC_ROOT_DIR environment variable, or use "~/tbp/results/dmc" as
# the default if not set
DMC_ROOT_DIR = Path(os.environ.get("DMC_ROOT_DIR", "~/tbp/results/dmc")).expanduser()
# We use the single-LM, distant-agent model.
SINGLE_LM_MODEL_PATH = (DMC_ROOT_DIR /
                        "pretrained_models/dist_agent_1lm/pretrained/model.pt")
SENSORY_CHANNEL = "patch"
LM_ID = 0

def count_monty_model_params(graph_memory: dict) -> int:
    """
    Count the number of parameters in a Monty model.

    Args:
        graph_memory (dict): The graph memory dictionary of the model.

    Returns:
        The total number of parameters in the model.
    """

    total_params = 0

    # Iterate through the learned objects
    for object_data in graph_memory.values():
        torch_geometric_data = object_data[SENSORY_CHANNEL]

        total_params += torch_geometric_data.x.numel()
        total_params += torch_geometric_data.pos.numel()

        # Each feature mapping key has two values (the upper and lower bounds of the
        # indices in the x array that store the feature).
        total_params += len(torch_geometric_data.feature_mapping.keys()) * 2

        # Due to redundancy/not being used, norm and edge_index/edge_attr are not
        # counted.

    return total_params


if __name__ == "__main__":
    graph_memory = torch.load(SINGLE_LM_MODEL_PATH)["lm_dict"][LM_ID]["graph_memory"]

    total_params = count_monty_model_params(graph_memory)

    print(f"Total number of parameters: {total_params}")
    print(f"\nIn millions: {total_params / 1e6:.2f}M")
