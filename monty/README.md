# Monty

This directory contains experiments that use the Monty framework.

## Installation

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

## Experiments

Experiments are defined in the `configs` directory.

After installing the environment, to run an experiment, run:

```bash
python run.py -e <experiment_name>
```

To run an experiment where episodes are executed in parallel, run:

```bash
python run_parallel.py -e <experiment_name> -n <num_parallel>
```
