# Causally Emergent Representations

This repository contains the implementation of the paper "Learning diverse causally emergent representations from time series data". The code provides methods to learn emergent features from multivariate time series data using information-theoretic objectives.

## Overview

This codebase implements a novel method for discovering emergent variables in time series data by combining:
- Recent information-theoretic characterizations of emergence
- Differentiable mutual information estimators
- Deep learning architectures

The key innovation is an end-to-end differentiable architecture that can learn maximally emergent representations from multivariate time series data.

## Getting Started

weights and biases (https://wandb.ai) is used to log experimental results so a wandb account is needed

### Installation
```bash
git clone [repository-url]
cd [repository-name]
pip install -r requirements.txt
pip install -e .
```

### Dataset Setup

The datasets used in this research are hosted on Zenodo. To set up the datasets:

1. Install the required packages for data download:
```bash
pip install requests tqdm
```

2. Run the data download script:
```bash
python data/download_data.py
```

This will download and organize all necessary datasets in the `data/` directory. For more details about the datasets and their structure, see `data/README.md`.

## Code Structure

The main components of the codebase are:

- `custom_datasets.py`: Dataset classes for loading and preprocessing FMRI, MEG, and Game of Life data
- `models.py`: Neural network architectures including skip connection networks and critics
- `trainers.py`: Training loops and optimization logic for learning emergent features
- `experiments/`: Scripts for running experiments on different datasets
  - `FMRI/`: FMRI emergence experiments
  - `MEG/`: MEG emergence experiments 
  - `GOL/`: Game of Life emergence experiments

Each experiment follows a similar pattern:
1. Load and preprocess the dataset
2. Configure the model architecture and training parameters
3. Train the feature network to learn emergent representations
4. Log results and visualizations to wandb

The core training loop uses mutual information estimators and gradient-based optimization to discover maximally emergent features from the input time series data.

This codebase currently contains some datasets but these will be moved to s3 for convenience.
