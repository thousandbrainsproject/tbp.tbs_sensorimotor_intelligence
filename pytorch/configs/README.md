# Configuration Structure

This project uses [Hydra](https://hydra.cc/) for managing configurations. Hydra provides a flexible way to organize and override configurations through YAML files and command-line arguments.

## Directory Structure

```
configs/
├── callbacks/              # Lightning callback configurations
│   ├── definitions/       # Individual callback definitions
│   └── presets/          # Pre-configured callback combinations
├── data/                  # Dataset configurations
├── experiment/            # Experiment-specific configurations
│   ├── 01_hyperparameter_optimization/  # Systematic hyperparameter tuning
│   │   ├── 01_early_stopping/          # Early stopping experiments
│   │   ├── 02_gradient_clipping/       # Gradient clipping tests
│   │   ├── 03_frozen_backbone/         # Backbone freezing evaluation
│   │   ├── 04_data_augmentation/       # Data augmentation studies
│   │   ├── 05_multilayer_heads/        # Head architecture experiments
│   │   ├── 06_warmup_cosine_decay/     # Learning rate scheduling
│   │   ├── 07_adamw_optimizer/         # Optimizer selection
│   │   ├── 08_differential_lrs/        # Differential learning rates
│   │   ├── 09_rotation_weights/        # Rotation loss weighting
│   │   ├── 10_ablate_data_aug/        # Data augmentation ablation
│   │   ├── 11_learning_rates/         # Learning rate search
│   │   ├── 12_learning_rates_p2/      # Extended learning rate search
│   │   ├── 13_baseline_models/        # Baseline model configurations
│   │   └── 14_train_from_scratch_lrs/ # Training from scratch
│   ├── 02_fig7a_rapid_learning/        # Rapid learning experiments (Fig 7a)
│   │   ├── pretrained/               # Experiments with pretrained models
│   │   │   ├── train/               # Training configurations
│   │   │   └── inference/           # Inference configurations
│   │   └── random_init/             # Experiments with random initialization
│   │       ├── train/               # Training configurations
│   │       └── inference/           # Inference configurations
│   ├── 03_fig7b_continual_learning/   # Continual learning experiments
│   ├── 04_fig8b_flops/                # FLOP measurement experiments (Fig 8b)
│   │   ├── pretrained/               # Experiments with pretrained models
│   │   │   ├── train/               # Training configurations
│   │   │   └── inference/           # Inference configurations
│   │   └── random_init/             # Experiments with random initialization
│   │       ├── train/               # Training configurations
│   │       └── inference/           # Inference configurations
│   └── 05_sanity_check/               # Validation experiments
├── extras/               # Additional configuration options
├── hydra/                # Hydra-specific settings
├── logger/               # Logging configurations (WandB, etc.)
├── model/                # Model architectures and parameters
├── paths/                # Path configurations
└── trainer/              # Training configurations
```
