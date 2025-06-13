#!/bin/bash

# Script to run all rapid training experiments for fig7a
# This script runs training and evaluation for various rotation settings with both pretrained and random initialization

echo "Starting rapid learning experiments..."

# Section 1: Training with pretrained initialization
echo "Section 1: Training with pretrained initialization (25 epochs)..."

for rot in 1 2 4 8 16 32; do
    echo "Training pretrained model with ${rot} rotations for 25 epochs..."
    python src/train.py \
        experiment=fig7a_pretrained/train/vit-b16-224-in21k_25epochs_n_rot \
        data.num_rotations_for_train=${rot}
done

echo "Section 2: Training with pretrained initialization (1 epoch)..."

for rot in 1 2 4 8 16 32; do
    echo "Training pretrained model with ${rot} rotations for 1 epoch..."
    python src/train.py \
        experiment=fig7a_pretrained/train/vit-b16-224-in21k_1epochs_n_rot \
        data.num_rotations_for_train=${rot}
done

# Section 3: Training with random initialization
echo "Section 3: Training with random initialization (25 epochs)..."

for rot in 1 2 4 8 16 32; do
    echo "Training randomly initialized model with ${rot} rotations for 25 epochs..."
    python src/train.py \
        experiment=fig7a_random_init/train/vit-b16-224-in21k_25epochs_n_rot \
        data.num_rotations_for_train=${rot}
done

echo "Section 4: Training with random initialization (1 epoch)..."

for rot in 1 2 4 8 16 32; do
    echo "Training randomly initialized model with ${rot} rotations for 1 epoch..."
    python src/train.py \
        experiment=fig7a_random_init/train/vit-b16-224-in21k_1epochs_n_rot \
        data.num_rotations_for_train=${rot}
done

# Section 5: Evaluation of pretrained models
echo "Section 5: Evaluating pretrained models (25 epochs)..."

for rot in 1 2 4 8 16 32; do
    echo "Testing pretrained model with ${rot} rotations (25 epochs)..."
    python src/eval.py \
        experiment=fig7a_pretrained/inference/vit-b16-224-in21k_25epochs_n_rot \
        data.num_rotations_for_train=${rot}
done

echo "Section 6: Evaluating pretrained models (1 epoch)..."

for rot in 1 2 4 8 16 32; do
    echo "Testing pretrained model with ${rot} rotations (1 epoch)..."
    python src/eval.py \
        experiment=fig7a_pretrained/inference/vit-b16-224-in21k_1epoch_n_rot \
        data.num_rotations_for_train=${rot}
done

# Section 7: Evaluation of randomly initialized models
echo "Section 7: Evaluating randomly initialized models (25 epochs)..."

for rot in 1 2 4 8 16 32; do
    echo "Testing randomly initialized model with ${rot} rotations (25 epochs)..."
    python src/eval.py \
        experiment=fig7a_random_init/inference/vit-b16-224-in21k_25epochs_n_rot \
        data.num_rotations_for_train=${rot}
done

echo "Section 8: Evaluating randomly initialized models (1 epoch)..."

for rot in 1 2 4 8 16 32; do
    echo "Testing randomly initialized model with ${rot} rotations (1 epoch)..."
    python src/eval.py \
        experiment=fig7a_random_init/inference/vit-b16-224-in21k_1epoch_n_rot \
        data.num_rotations_for_train=${rot}
done

echo "All rapid training experiments completed!"
