# Vision Transformer for Object Classification and Pose Estimation

A Vision Transformer (ViT) implementation for joint object classification and pose estimation using RGB-D inputs. 

## 🚀 Quick Start

For detailed reproduction instructions, see **[REPRODUCE_RESULTS.md](REPRODUCE_RESULTS.md)**.

### Verify Your Setup
```bash
cd tbp.tbs_sensorimotor_intelligence/pytorch
conda env create -f environment.yaml
conda activate vit
export PROJECT_ROOT=$(pwd)

# Quick verification
python scripts/verify_setup.py

# End-to-end test  
./scripts/quick_test.sh
```

### Run Experiments
```bash
# Single experiment
python src/train.py experiment=04_fig8b_flops/pretrained/train/vit-b16-224-in21k

# Batch experiments
./scripts/fig8b_flops.sh         # Model architecture comparison
./scripts/fig7a_rapid_learning.sh   # Rapid learning experiments
```

## Set up Conda Environment

```bash
# Clone project
git clone https://github.com/thousandbrainsproject/tbp.tbs_sensorimotor_intelligence.git
cd pytorch

# Create conda environment
conda env create -f environment.yaml
conda activate vit

# Set PROJECT_ROOT (important!)
export PROJECT_ROOT=$(pwd)
```

## Usage

All experiments can be run from the project root directory (`tbp.tbs_sensorimotor_intelligence/pytorch`):

```bash
python src/train.py experiment=your_exp_config
# Example
python src/train.py experiment=04_fig8b_flops/pretrained/train/vit-b16-224-in21k
```

Available experiment scripts:
```bash
./scripts/fig7a_rapid_learning.sh   # Rapid learning with different data amounts
./scripts/fig7b_continual_learning.sh  # Continual learning across tasks  
./scripts/fig8b_flops.sh           # Model architecture comparison
```

## Architecture

The model architecture consists of:

- Vision Transformer (ViT) backbone
- Custom RGB-D patch embedding layer
- Dual heads for classification and rotation
- Quaternion-based pose representation

## Datasets

The project uses RGB-D images extracted the YCB dataset using the Habitat simulator. The dataset structure is as follows:

```
view_finder_images/
├── view_finder_32/
│   └── view_finder_rgbd/
├── view_finder_base/
│   └── view_finder_rgbd/
└── view_finder_randrot/
    └── view_finder_rgbd/
```

## Acknowledgements

- Vision Transformer (ViT) implementation based on the original paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- YCB Dataset: [Yale-CMU-Berkeley Object and Model Set](https://www.ycbbenchmarks.com/)
- Template Repository: [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
