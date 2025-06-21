# Reproducing Vision Transformer (ViT) Results

This guide provides step-by-step instructions to reproduce the Vision Transformer experiments for object classification and pose estimation using RGB-D inputs.

## 📋 Prerequisites

- Conda/Miniconda installed
- Access to the YCB dataset (RGB-D images from Habitat simulator)
- WandB account for experiment tracking (optional but recommended)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**One-command setup and verification:**
```bash
# Clone the repository (if not already done)
cd tbp.tbs_sensorimotor_intelligence/pytorch

# Run automated setup and verification (5-15 minutes)
./scripts/setup_and_test.sh
```

This will automatically:
- ✅ Check conda installation
- ✅ Create/activate `vit` environment  
- ✅ Verify all dependencies
- ✅ Run end-to-end tests
- ✅ Show next steps for running experiments

**Run ALL experiments (4-8 hours):**
```bash
# Complete reproduction suite with limited continual learning
./scripts/run_all_experiments.sh
```

### Option 2: Manual Setup

```bash
# Clone the repository (if not already done)
cd tbp.tbs_sensorimotor_intelligence/pytorch

# Create and activate conda environment
conda env create -f environment.yaml
conda activate vit

# Set PROJECT_ROOT environment variable (critical for proper paths)
export PROJECT_ROOT=$(pwd)

# Verify setup
python scripts/verify_setup.py

# Run quick test (optional but recommended)
./scripts/quick_test.sh
```

**Important**: Always run experiments from the `tbp.tbs_sensorimotor_intelligence/pytorch` directory where the `.project-root` file is located.

### 2. Data Setup

The experiments expect data in the following structure:
```
~/tbp/results/dmc/data/view_finder_images/
├── view_finder_32/
│   └── view_finder_rgbd/
├── view_finder_base/
│   └── view_finder_rgbd/
└── view_finder_randrot/
    └── view_finder_rgbd/
```

If your data is located elsewhere, you can either:
- **Option A**: Create symlinks to match the expected structure
- **Option B**: Override the data path when running experiments:
  ```bash
  python src/train.py experiment=<config> paths.data_dir="/path/to/your/data"
  ```

### 3. WandB Setup (Optional but Recommended)

```bash
# Login to WandB
wandb login

# Or set your API key
export WANDB_API_KEY=your_api_key_here
```

## 🧪 Running Experiments

### Individual Experiments

**Training a single model:**
```bash
python src/train.py experiment=04_fig8b_flops/pretrained/train/vit-b16-224-in21k
```

**Evaluating a trained model:**
```bash
python src/eval.py experiment=04_fig8b_flops/pretrained/inference/vit-b16-224-in21k
```

**Overriding parameters:**
```bash
python src/train.py experiment=<config> \
    trainer.max_epochs=50 \
    data.batch_size=64 \
    model.net.model_name=vit-l16-224-in21k
```

### Batch Experiments

We provide comprehensive scripts for reproducing all results:

#### All Experiments (Complete Reproduction)
```bash
# Run ALL experiments with setup verification (4-8 hours)
./scripts/run_all_experiments.sh
```

This comprehensive script runs:
- **Figure 7a**: Rapid learning experiments (all rotation counts)
- **Figure 7b**: Continual learning (first 5 tasks only for demo)
- **Figure 8b**: FLOP analysis (all ViT architectures)

#### Individual Experiment Suites

We also provide individual experiment scripts corresponding to different research figures:

#### Figure 7a: Rapid Learning Experiments
Tests learning efficiency with different amounts of training data (rotation counts):

```bash
chmod +x scripts/fig7a_rapid_learning.sh
./scripts/fig7a_rapid_learning.sh
```

This runs:
- **Pretrained models**: 25 epochs with 1, 2, 4, 8, 16, 32 rotations
- **Random init models**: 75 epochs and 1 epoch with same rotation counts
- **Evaluation**: All trained models on test set

#### Figure 7b: Continual Learning Experiments
Tests continual learning across sequential tasks:

```bash
chmod +x scripts/fig7b_continual_learning.sh
./scripts/fig7b_continual_learning.sh
```

#### Figure 8b: FLOP Analysis Experiments
Compares different ViT architectures (B/16, B/32, L/16, L/32, H/14):

```bash
chmod +x scripts/fig8b_flops.sh
./scripts/fig8b_flops.sh
```

This runs:
- **Training**: All 5 ViT variants with both pretrained and random initialization
- **Evaluation**: All trained models with FLOP counting

### Custom Experiment Scripts

You can create your own batch experiments. Here's a template:

```bash
#!/bin/bash
# custom_experiment.sh

echo "Starting custom experiments..."

# Train multiple models
MODELS=("vit-b16-224-in21k" "vit-b32-224-in21k")
for model in "${MODELS[@]}"; do
    echo "Training ${model}..."
    python src/train.py \
        experiment=04_fig8b_flops/pretrained/train/${model} \
        trainer.max_epochs=10
done

# Evaluate models
for model in "${MODELS[@]}"; do
    echo "Evaluating ${model}..."
    python src/eval.py \
        experiment=04_fig8b_flops/pretrained/inference/${model}
done

echo "Custom experiments completed!"
```

## 📊 Analyzing Results

### WandB Dashboard

1. **Access your project**: Go to https://wandb.ai and navigate to project `benchmark_vit`

2. **Compare runs**: 
   - Use the "Table" view for side-by-side comparison
   - Sort by "Name" to group related experiments
   - Filter by tags (e.g., `fig7a`, `fig8b`) to focus on specific experiment types

3. **Key metrics to monitor**:
   - `val/class_acc`: Validation classification accuracy (primary metric)
   - `test/class_acc`: Test classification accuracy 
   - `val/loss`: Validation loss
   - `test/rotation_error`: Rotation prediction error in degrees
   - Training time and computational metrics

### Example WandB Comparison Workflow

1. **Filter experiments**:
   ```
   Tags: fig8b
   ```

2. **Sort by name** to group experiments by model type:
   ```
   fig8b_vit-b16-224-in21k
   fig8b_vit-b32-224-in21k  
   fig8b_vit-h14-224-in21k
   ...
   ```

3. **Create comparison table**:
   - Select runs to compare
   - Add key columns: `val/class_acc`, `test/class_acc`, `trainer/max_epochs`
   - Export results for further analysis

### Local Results Analysis

Training and evaluation outputs are saved to:
```
~/tbp/results/dmc/results/vit/logs/<experiment_name>/
├── checkpoints/          # Model checkpoints
├── inference/            # Evaluation results
└── <experiment_name>.log # Training logs
```

## 🔧 Configuration System

This project uses [Hydra](https://hydra.cc/) for configuration management. Key config locations:

```
configs/
├── experiment/           # Pre-defined experiment configurations
│   ├── 01_hyperparameter_optimization/
│   ├── 02_fig7a_rapid_learning/
│   ├── 03_fig7b_continual_learning/
│   └── 04_fig8b_flops/
├── model/               # Model architecture configs
├── data/                # Dataset configurations  
├── trainer/             # Training configurations
└── callbacks/           # Callback configurations
```

### Creating Custom Configurations

1. **Copy existing config**:
   ```bash
   cp configs/experiment/04_fig8b_flops/pretrained/train/vit-b16-224-in21k.yaml \
      configs/experiment/my_custom_experiment.yaml
   ```

2. **Modify parameters** as needed

3. **Run with your config**:
   ```bash
   python src/train.py experiment=my_custom_experiment
   ```

## 🐛 Troubleshooting

### Common Issues

**1. Path errors**:
```
KeyError: 'PROJECT_ROOT'
```
**Solution**: Make sure you're in the right directory and set PROJECT_ROOT:
```bash
cd tbp.tbs_sensorimotor_intelligence/pytorch
export PROJECT_ROOT=$(pwd)
```

**2. Data not found**:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../view_finder_rgbd'
```
**Solution**: Check data path configuration:
```bash
python src/train.py experiment=<config> paths.data_dir="/correct/path/to/data"
```

**3. CUDA/MPS device issues**:
**Solution**: Override trainer accelerator:
```bash
# For CPU
python src/train.py experiment=<config> trainer.accelerator=cpu

# For CUDA
python src/train.py experiment=<config> trainer.accelerator=gpu

# For Apple Silicon
python src/train.py experiment=<config> trainer.accelerator=mps
```

**4. Out of memory**:
**Solution**: Reduce batch size or accumulate gradients:
```bash
python src/train.py experiment=<config> \
    data.batch_size=32 \
    trainer.accumulate_grad_batches=4
```

### Debugging Tips

1. **Dry run**: Test configs without training:
   ```bash
   python src/train.py experiment=<config> trainer.fast_dev_run=true
   ```

2. **Verbose logging**: Enable debug mode:
   ```bash
   python src/train.py experiment=<config> hydra.verbose=true
   ```

3. **Check config resolution**:
   ```bash
   python src/train.py experiment=<config> --cfg job
   ```

## 📈 Expected Results

### Hyperparameter Optimization Results
After running the full optimization pipeline (`01_hyperparameter_optimization/`), you should see:

- **Best ViT-B16 performance**: ~93% validation accuracy
- **Optimal learning rate**: 5e-4 with AdamW optimizer
- **Training duration**: ~20-25 epochs for convergence

### Model Architecture Comparison (Fig 8b)
Expected test accuracies for pretrained models:

| Model | Test Accuracy | Parameters |
|-------|---------------|------------|
| ViT-B/32 | ~90-92% | 88M |
| ViT-B/16 | ~92-94% | 86M |
| ViT-L/32 | ~93-95% | 307M |
| ViT-L/16 | ~94-96% | 304M |
| ViT-H/14 | ~94-96% | 632M |

*Note: Exact numbers may vary slightly due to randomness in training.*

## 🤝 Contributing

When adding new experiments:

1. Create configuration files in appropriate `configs/experiment/` subdirectory
2. Update batch scripts in `scripts/` if needed  
3. Document expected results and any special requirements
4. Test with a small dry run before full execution

## 📝 Citation

If you use this code, please cite the original work and acknowledge the Vision Transformer implementation and YCB dataset sources as mentioned in the main README. 