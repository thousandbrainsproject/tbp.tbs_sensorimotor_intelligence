# 🔧 ViT Configurations, Hyperparameter Optimizations, and Sanity Checks

This document organizes the steps we have taken to establish a reasonable baseline for ViT model on RGBD image classification and rotation prediction. The ViT baseline is built on `vit-b16-224-in21k` (from HuggingFace), with the goal of developing solid and fair baseline, balancing between SOTA-chasing and overly weak baselines.

## 📝 Starting Configuration (`starting_config.yaml`)

The starting configuration defines the following:

- RGBD_ViT model with `vit-b16-224-in21k` backbone from HuggingFace with the following modifications:
  - Two "simple" heads (`nn.Linear`) for classification and quaternion prediction
  - Full training of all parameters enabled
  - RGBD patch embedding initialized from pretrained RGB weights + mean of the pretrained RGB weights for Depth Channel:

```python
    new_projection = nn.Conv2d(in_channels=4,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    # Copy RGB weights and initialize depth channel
    with torch.no_grad():
        depth_weight = rgb_weights.mean(dim=1, keepdim=True)  # (out_c, 1, k, k)
        new_weight = torch.cat(
            [rgb_weights, depth_weight], dim=1
        )
```

- Dataset and DataLoaders:
  - Dataset contains 1,078 images (77 objects x 14 rotations), randomly split into train/val with 80/20 ratio
  - Test dataset contains 385 images (77 objects x 5 random rotations)
  - Batch size of 128
- Optimizers and Schedulers:
  - Adam Optimizer (`lr: 0.001`) with ReduceLROnPlateau Scheduler (`min_lr = 1e-6`, `factor=0.1`, `patience=10`)
  - Early Stopping enabled (monitoring `val/loss` with `patience=20`)
- Combined loss function with $\\lambda\_\\text{rot} = 1.0$:
  $$L = \\text{CrossEntropyLoss} + \\lambda\_\\text{rot} * \\text{Geodesic Loss}$$
  - Note: $\\lambda\_\\text{rot} = 1.0$ balances the values of CrossEntropyLoss and Geodesic Loss

## 🔁 Incremental Optimization and Sanity Checks

We evaluated the following optimizations, applied **incrementally** from the starting configuration. We report the `val/class_acc` (classification accuracy on validation set) and `test/class_acc` (classification accuracy on test set).

**Note 1**: We use metrics from validation set to pick the best model (picking by `test/class_acc` would be considered "cheating"). Test metrics are included only for reference.

**Note 2**: We omitted `val/rotation_error` and `test/rotation_error` because these were consistently very high and thus weren't meaningful in choosing the best model.

### 1. Early Stopping

- **Reason**: Verify we aren't missing better models by training longer (up to `max_epochs = 200`)
- **Results**:
  - With Early Stopping (58 epochs): `val/class_acc` = 75.5%; `test/class_acc` = 72.5%
  - Without Early Stopping: Training becomes unstable and model diverges
- **Decision**: Keep Early Stopping enabled
- **Impact**: Reduces training FLOPs by ~3x (73 epochs vs 200) but maintains model quality

### 2. Gradient Clipping

- **Reason**: Regularize ViT model and prevent divergence
- **Results**:
  - With gradient clipping (1.0): `val/class_acc` = 77.3%; `test/class_acc` = 76.1%
- **Decision**: Enable gradient clipping with value 1.0

### 3. Freezing Backbone

- **Reason**: Common practice in finetuning to freeze backbone or early layers
- **Results**:
  - With frozen backbone: `val/class_acc` = 76.9%; `test/class_acc` = 71.6%
- **Decision**: Keep all parameters trainable

### 4. Data Augmentation

- **Reason**: Standard practice for finetuning on small datasets
- **Implementation**: Only rotation-invariant augmentations:

```python
self.rgb_transform: T.Compose = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
```

- **Results**:
  - With augmentation: `val/class_acc` = 87.5%; `test/class_acc` = 78.4%
- **Final Decision**: Omit data augmentation to match training conditions with other models

### 5. Sophisticated Heads

- **Reason**: Simple `nn.Linear` may be too naive
- **Results**:
  - Linear -> GELU -> Dropout -> Linear: `val/class_acc` = 78.7%; `test/class_acc` = 75.1%
  - LayerNorm -> Linear: `val/class_acc` = 88.9%; `test/class_acc` = 78.2%
  - Linear -> LayerNorm -> GELU -> Dropout -> Linear: `val/class_acc` = 87.0%; `test/class_acc` = 76.6%
- **Decision**: Use LayerNorm -> Linear architecture

### 6. Warmup + Cosine Decay LR Schedule

- **Reason**: More standard in ViT finetuning than ReduceLROnPlateau
- **Results**:
  - `val/class_acc` = 88.4%; `test/class_acc` = 80.5%
- **Decision**: Keep Warmup + Cosine Decay schedule

### 7. AdamW Optimizer

- **Reason**: Standard practice, properly applies weight decay unlike Adam
- **Results**:
  - `val/class_acc` = 90.3%; `test/class_acc` = 86.8%
- **Decision**: Use AdamW

### 8. Differential Learning Rates

- **Reason**: Common to use lower LR for pretrained backbone vs new heads
- **Results**:
  - `val/class_acc` = 87.5%; `test/class_acc` = 86.0%
- **Decision**: Use same learning rate for all parameters

### 9. Loss Weight Tuning ($\\lambda\_\\text{rot}$)

- **Reason**: Validate empirical choice of $\\lambda\_\\text{rot} = 1.0$
- **Results**:
  - $\\lambda\_\\text{rot} = 0.1$: `val/class_acc = 92.6%`, `val/rotation_error = 113.7°`
  - $\\lambda\_\\text{rot} = 1.0$: `val/class_acc = 90.3%`, `val/rotation_error = 109.2°`
  - $\\lambda\_\\text{rot} = 3.0$: `val/class_acc = 89.4%`, `val/rotation_error = 106.2°`
  - $\\lambda\_\\text{rot} = 5.0$: `val/class_acc = 88.9%`, `val/rotation_error = 105.8°`
- **Decision**: Keep $\\lambda\_\\text{rot} = 1.0$

### 10. Learning Rate Optimization

- **Initial Search**: [0.001, 0.0003, 0.0001, 0.00003, 0.00001]
  - Best: lr = 0.0003 with `val/class_acc` = 92.1%
- **Fine Search**: [0.0002, 0.0004, 0.0005, 0.0006, 0.0007]
  - Best: lr = 0.0005 with `val/class_acc` = 93.1%

## 📊 Training Duration Analysis

Analyzed optimal training duration for different ViT variants:

- vit-b32: Best at epoch 30 (`val/class_acc` = 92.1%)
- vit-b16: Best at epoch 20
- vit-l32: Best at epoch 6 (`val/class_acc` = 92.6%)
- vit-l16: Best at epoch 12 (`val/class_acc` = 94.0%)
- vit-h14: Best at epoch 28 (`val/class_acc` = 94.0%)

**Decision**: Standardize on 25 epochs for consistency and stability

## 🎯 Final Configuration (`optimized_config.yaml`)

The optimized configuration includes:

- Early Stopping: Enabled
- Gradient Clipping: 1.0
- Full parameter updates (no frozen layers)
- No data augmentation
- LayerNorm -> Linear architecture for both heads
- Warmup + Cosine Decay LR Schedule
- AdamW Optimizer:
  - Learning rate: 0.0005
  - Weight decay: 0.01
  - Betas: [0.9, 0.999]
  - Epsilon: 1e-8
- Loss weight $\\lambda\_\\text{rot} = 1.0$
- Training duration: 25 epochs

## 🔬 Training from Scratch Results

Learning rate search for training ViT-B16 from scratch:

- lr = 5e-3: `val/class_acc` = 37.5%
- lr = 1e-3: `val/class_acc` = 71.8%
- lr = 5e-4: `val/class_acc` = 72.2%
- lr = 1e-4: `val/class_acc` = 75.9%
- lr = 5e-5: `val/class_acc` = 77.8%
- lr = 1e-5: `val/class_acc` = 78.7% (best, epoch 75)
- lr = 5e-6: `val/class_acc` = 72.2%
