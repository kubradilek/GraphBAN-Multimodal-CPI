# GraphBAN-Multimodal-CPI

This repository contains the implementation of the multimodal compound–protein interaction (CPI) prediction framework developed as part of my PhD thesis at the University of Nottingham.

The framework extends GraphBAN by integrating:

- ESMC protein language model embeddings
- SaProt structure-aware protein embeddings
- Graph convolutional networks (GCN) for molecular representation
- Bilinear attention networks (BAN)
- Contrastive learning
- Latent diffusion regularisation
- Optuna hyperparameter optimisation

## Repository structure

```
data_processing/
    Protein feature extraction using ESMC and SaProt

model_training/
    GraphBAN model training and optimisation

prediction/
    Prediction and attention analysis
```

## Requirements

Main Python packages include:

- Python 3.10+
- PyTorch
- DGL
- DGLLife
- RDKit
- Transformers
- ESM
- Optuna
- Scikit-learn
- Pandas
- NumPy

## Citation

If you use this repository, please cite the associated PhD thesis.
