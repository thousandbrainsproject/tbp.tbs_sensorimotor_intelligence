# Monty Experiments

This directory contains all experiments from the paper that use the Monty framework.

## Environment Installation

The environment for this project is managed with [conda](https://www.anaconda.com/download/success).

To create the environment, run the following command, tailored to your system:

### ARM64 (Apple Silicon) (zsh shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init zsh
conda activate tbs_sensorimotor_intelligence
conda config --env --set subdir osx-64
```

### ARM64 (Apple Silicon) (bash shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init
conda activate tbs_sensorimotor_intelligence
conda config --env --set subdir osx-64
```

### Intel (zsh shell)
```
conda env create -f environment.yml
conda init zsh
conda activate tbs_sensorimotor_intelligence
```

### Intel (bash shell)
```
conda env create -f environment.yml
conda init
conda activate tbs_sensorimotor_intelligence
```

## Further Setup

The results of DMC experiments are stored under `DMC_ROOT_DIR`. By default, this is `~/tbp/results/dmc`, but you can change this by setting the `DMC_ROOT_DIR` environment variable. It has the following structure:

```
DMC_ROOT_DIR/
    pretrained_models/
    results/
    view_finder_images/        
```

The `pretrained_models` directory should contain the pre-trained models for Monty experiments. The `results` directory contains the results of evaluation experiments. The `view_finder_images` directory contains the images used for the view finder experiments.

Pretrained models can be downloaded from `TODO create link`.

Figures and tables are stored under `DMC_ANALYSIS_DIR`. By default, this is `~/tbp/results/dmc_analysis`, but you can change this by setting the `DMC_ANALYSIS_DIR` environment variable.

## Experiments

Experiments are defined in the `configs` directory, including for pre-training models from scratch.

After installing the environment, to run an experiment, run:

```bash
python run.py -e <experiment_name>
```

To run an experiment where episodes are executed in parallel, run:

```bash
python run_parallel.py -e <experiment_name> -n <num_parallel>
```

## Reproducting Figures from the Paper

Analaysis scripts to generate the figures are in the `scripts` directory. The data for figures can either be generated via `configs`, or you can directly download the pre-trained models at `link-to-be-confirmed` and the raw results from experiments at `link-to-be-confirmed`.
