#!/bin/bash

# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Script to run all flops experiments for fig8b
# This script runs training and evaluation for various model architectures with both pretrained and random initialization

# Define the models to test
MODELS=("vit-b16-224-in21k" "vit-b32-224-in21k" "vit-h14-224-in21k" "vit-l16-224-in21k" "vit-l32-224-in21k")

echo "Starting flops experiments..."

# Section 1: Training with pretrained initialization
echo "Section 1: Training with pretrained initialization..."

# Train different model architectures
for model in "${MODELS[@]}"; do
    echo "Training pretrained ${model}..."
    python src/train.py experiment=04_fig8b_flops/pretrained/train/${model} paths=reproduction
done

# Section 2: Training with random initialization
echo "Section 2: Training with random initialization..."

# Train different model architectures
for model in "${MODELS[@]}"; do
    echo "Training randomly initialized ${model}..."
    python src/train.py experiment=04_fig8b_flops/random_init/train/${model} paths=reproduction
done

# Section 3: Evaluation of pretrained models
echo "Section 3: Evaluating pretrained models..."

# Evaluate different model architectures
for model in "${MODELS[@]}"; do
    echo "Testing pretrained ${model}..."
    python src/eval_standard.py experiment=04_fig8b_flops/pretrained/inference/${model} paths=reproduction
done

# Section 4: Evaluation of randomly initialized models
echo "Section 4: Evaluating randomly initialized models..."

# Evaluate different model architectures
for model in "${MODELS[@]}"; do
    echo "Testing randomly initialized ${model}..."
    python src/eval_standard.py experiment=04_fig8b_flops/random_init/inference/${model} paths=reproduction
done

echo "All flops experiments completed!" 