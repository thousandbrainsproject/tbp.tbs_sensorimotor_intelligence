# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from .fig3_robust_sensorimotor_inference import CONFIGS as FIG_3_CONFIGS
from .fig4_structured_object_representations import CONFIGS as FIG_4_CONFIGS
from .fig5_rapid_inference_with_model_based_policies import CONFIGS as FIG_5_CONFIGS
from .fig6_rapid_inference_with_voting import CONFIGS as FIG_6_CONFIGS
from .fig7a_rapid_learning import CONFIGS as FIG_7_RAPID_LEARNING_CONFIGS
from .fig7b_continual_learning import CONFIGS as FIG_7_CONTINUAL_LEARNING_CONFIGS
from .fig8_flops import CONFIGS as FIG_8_CONFIGS
from .pretraining_experiments import CONFIGS as PRETRAINING_CONFIGS
from .view_finder_images import CONFIGS as VIEW_FINDER_CONFIGS
from .visualizations import CONFIGS as VISUALIZATION_CONFIGS

CONFIGS = dict()
CONFIGS.update(PRETRAINING_CONFIGS)
CONFIGS.update(FIG_3_CONFIGS)
CONFIGS.update(FIG_4_CONFIGS)
CONFIGS.update(FIG_5_CONFIGS)
CONFIGS.update(FIG_6_CONFIGS)
CONFIGS.update(FIG_7_CONTINUAL_LEARNING_CONFIGS)
CONFIGS.update(FIG_7_RAPID_LEARNING_CONFIGS)
CONFIGS.update(FIG_8_CONFIGS)
CONFIGS.update(VIEW_FINDER_CONFIGS)
CONFIGS.update(VISUALIZATION_CONFIGS)