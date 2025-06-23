# HPGNN-Link: Interpretable Link Prediction via Prototype and Hypergraph Learning

## Getting Started

### Requirements

- Our code has been tested with
  - Python = 3.7.16
  - PyTorch = 1.12.0
  - DGL = 0.9.1
  - torch-geometric = 2.3.1
  - networkx = 2.6.3

### Datasets

Datasets used in the paper are under `datasets/`.

### Run Model

- Train model

  - Run `run_model.py` as the examples below

    ```
    python run_model.py
    ```

### Trained Model
The trained models are under `save_model/`. Currently, we provide trained model on the FrenchRoyalty dataset, and models for other datasets will be open-sourced once the paper has been accepted.
