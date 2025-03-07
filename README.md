# tbp.tbs_sensorimotor_intelligence

This repository contains code to replicate experiments from our paper, "Thousand Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference".

Experiments make use of either the Monty framework, or Pytorch in the case of deep learning models. These can be found under their respective directories `monty` and `pytorch`.

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

For instructions on how to run experiments, please refer to the respective README files in the `monty` and `pytorch` directories.

## Development

After installing the environment, you can run the following commands to check your code.

### Run formatter

```bash
ruff format
```

### Run style checks

```bash
ruff check
```

### Run dependency checks

```bash
deptry .
```

### Run static type checks

```bash
mypy .
```

### Run tests

```bash
pytest
```
