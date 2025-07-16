# 🧪 HOMO–LUMO Gap Models

A machine learning pipeline for predicting **HOMO–LUMO energy gaps** using 2D and (eventually) 3D molecular features. This repository supports reproducible experiments, model evaluation, and integration with an interactive web app.


### Problem Statement & Motivation

Accurately predicting quantum chemical properties like the HOMO–LUMO gap is essential for advancing materials science, drug discovery, and electronics. The size of the gap offers insight into a molecule's reactivity and stability.

While Density Functional Theory (DFT) provides reliable results, it is computationally expensive for large-scale screening. This project explores machine learning alternatives that are lightweight, interpretable, and feasible to run on modest hardware.

### Related Work & Key Gap

Past research shows:
- DFT is accurate but slow and resource-intensive
- ML models (kernel methods, GNNs) are faster, but can require expensive compute and complex setup

Gap Addressed: This project aims to develop fast, accessible ML models that can serve as a practical alternative to DFT-based gap prediction—scalable for screening pipelines and deployable via a lightweight app.


## Methodology & Evaluation

This repository includes:
- Baseline 2D models using RDKit fingerprints and Coulomb matrices
- Graph Neural Networks (GNNs) using ChemML and OGB-based graph encodings
- A final hybrid GNN model combining structural embeddings and cheminformatics descriptors

| Metric   | Best Model (Hybrid GNN) |
|----------|-------------------------|
| **MAE**  | 0.159 eV                |
| **RMSE** | 0.234 eV                |
| **R²**   | 0.965                   |

Models are evaluated using:
- Parity and residual plots
- MAE, RMSE, R²
- Error inspection on high-error molecules
- Train/val/test splits aligned with OGB standards


## App Deployment

An interactive web application is available for real-time prediction and visualization:

**Live App**: [HOMO–LUMO Gap Predictor on Hugging Face](https://huggingface.co/spaces/MooseML/homo-lumo-gap-predictor)

### App Features:
- SMILES input for any organic molecule
- Real-time HOMO–LUMO gap prediction
- Molecular visualization
- CSV result export for local logging


## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

This project was developed and tested locally with the following setup:

* **Python version**: `3.8.20`
* **GPU**: `NVIDIA RTX 3070 Ti`
* **CUDA**: `11.8`

Key libraries:

* **PyTorch**: `torch==2.4.1+cu118`
* **PyTorch Geometric**: `torch-geometric==2.6.1`
* **TensorFlow**: `2.10.0`

> Additional libraries used include `torch_scatter`, `torch_sparse`, `torch_cluster`, `torchvision`, and `torchaudio`, all built with CUDA 11.8 compatibility.

## Next Steps

Planned future work includes:

* Integrating **3D molecular geometry** 
* Evaluating 3D-aware architectures like **SchNet**, **DimeNet**, or **SE(3)-equivariant GNNs**
* Expanding app functionality for larger file uploads and faster inference  


## Related Resources

* **[Hugging Face App – homo-lumo-gap-predictor](https://huggingface.co/spaces/MooseML/homo-lumo-gap-predictor)**
* **GitHub App Repo:** [MooseML/homo-lumo-gap-predictor](https://github.com/MooseML/homo-lumo-gap-predictor)
* **Model Repo (this):** [MooseML/homo-lumo-gap-models](https://github.com/MooseML/homo-lumo-gap-models)
