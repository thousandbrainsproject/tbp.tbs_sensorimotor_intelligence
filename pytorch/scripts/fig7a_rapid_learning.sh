#!/bin/bash

# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Script to run all rapid training experiments for fig7a
# This script runs training and evaluation for various rotation settings with both pretrained and random initialization

echo "Starting rapid learning experiments..."

# Section 1: Training with pretrained initialization
echo "Section 1: Training with pretrained initialization (25 epochs)..."

for rot in 1 2 4 8 16 32; do
    echo "Training pretrained model with ${rot} rotations for 25 epochs..."
    python src/train.py \
        experiment=02_fig7a_rapid_learning/pretrained/train/vit-b16-224-in21k_25epochs_n_rot \
        data.num_rotations_for_train=${rot} \
        paths=reproduction
done

# Section 2: Training with random initialization
echo "Section 2: Training with random initialization (75 epochs)..."

for rot in 1 2 4 8 16 32; do
    echo "Training randomly initialized model with ${rot} rotations for 75 epochs..."
    python src/train.py \
        experiment=02_fig7a_rapid_learning/random_init/train/vit-b16-224-in21k_75epochs_n_rot \
        data.num_rotations_for_train=${rot} \
        paths=reproduction
done

echo "Section 3: Training with random initialization (1 epoch)..."

for rot in 1 2 4 8 16 32; do
    echo "Training randomly initialized model with ${rot} rotations for 1 epoch..."
    python src/train.py \
        experiment=02_fig7a_rapid_learning/random_init/train/vit-b16-224-in21k_1epochs_n_rot \
        data.num_rotations_for_train=${rot} \
        paths=reproduction
done

# Section 4: Evaluation of pretrained models
echo "Section 4: Evaluating pretrained models (25 epochs)..."

for rot in 1 2 4 8 16 32; do
    echo "Testing pretrained model with ${rot} rotations (25 epochs)..."
    python src/eval_standard.py \
        experiment=02_fig7a_rapid_learning/pretrained/inference/vit-b16-224-in21k_25epochs_n_rot \
        data.num_rotations_for_train=${rot} \
        paths=reproduction
done

# Section 5: Evaluation of randomly initialized models
echo "Section 5: Evaluating randomly initialized models (75 epochs)..."

for rot in 1 2 4 8 16 32; do
    echo "Testing randomly initialized model with ${rot} rotations (75 epochs)..."
    python src/eval_standard.py \
        experiment=02_fig7a_rapid_learning/random_init/inference/vit-b16-224-in21k_75epochs_n_rot \
        data.num_rotations_for_train=${rot} \
        paths=reproduction
done

echo "Section 6: Evaluating randomly initialized models (1 epoch)..."

for rot in 1 2 4 8 16 32; do
    echo "Testing randomly initialized model with ${rot} rotations (1 epoch)..."
    python src/eval_standard.py \
        experiment=02_fig7a_rapid_learning/random_init/inference/vit-b16-224-in21k_1epoch_n_rot \
        data.num_rotations_for_train=${rot} \
        paths=reproduction
done

echo "All rapid training experiments completed!"
