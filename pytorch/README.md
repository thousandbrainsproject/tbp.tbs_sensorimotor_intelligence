# Vision Transformer for Object Classification and Pose Estimation

A Vision Transformer (ViT) implementation for joint object classification and pose estimation using RGB-D inputs. 

## Set up Conda Environment

```bash
# Clone project
git clone https://github.com/thousandbrainsproject/tbp.tbs_sensorimotor_intelligence.git
cd pytorch

# Create conda environment
conda env create -f environment.yaml -n vit

# Activate environment
conda activate vit
```

## Usage

All experiments can be run following the below syntax from the project root directory (`tbp.tbs_sensorimotor_intelligence/pytorch`):

```bash
python src/train.py experiment=your_exp_config
# Example
python src/train.py experiment=04_fig8b_flops/pretrained/train/vit-b16-224-in21k.yaml
```

To run all training and inference experiments, we have created a bash script to automatically run the relevant experiments in the `scripts/` directory.
```bash
chmod +x ./scripts/fig8b_flops.sh
./scripts/fig8b_flops.sh
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
