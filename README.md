# RNA 3D Folding Solution Walkthrough

I have implemented a complete solution for the **Stanford RNA 3D Folding** competitions (Part 1 and 2), designed to run on Google Colab.

## 1. Solution Overview

The solution uses a **RibonanzaNet** backbone, which was the state-of-the-art architecture from the **Stanford Ribonanza RNA Folding (2D)** competition. This backbone is adapted to predict 3D geometric features.

### Key Components
-   **Backbone**: A Transformer with **Pairwise Attention** to capture long-range interactions (crucial for RNA secondary and tertiary structure).
-   **Heads**:
    -   **Distogram**: Predicts the distance between all pairs of residues.
    -   **Torsion**: Predicts the 7 torsion angles ($\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \chi$) for each residue.
-   **Structure Module (NeRF)**: A differentiable recreation of the 3D backbone coordinates from the predicted torsion angles.

## 2. Files and Artifacts

| File | Description |
| :--- | :--- |
| `colab_notebook.ipynb` | **Start Here**. The main Jupyter notebook. |
| `rna_model.py` | Contains the `RNAModel`, `RibonanzaBlock`, and `nerf_build` functions. |
| `colab_train.py` | Contains the `RNADataset`, `loss_fn`, and training loop. |

## 3. How to Run on Colab

1.  **Download** the `colab_notebook.ipynb` file.
2.  **Upload** it to [Google Colab](https://colab.research.google.com/).
3.  **Set Runtime to GPU**: Go to `Runtime` > `Change runtime type` > Select `T4 GPU`.
4.  **Run All Cells**: The notebook will:
    -   Install `torch` and `biopython`.
    -   Write the helper python files to the Colab instance.
    -   Initialize the model.
    -   Run a training loop on generated mock data.
    -   Output the shape of the predicted 3D coordinates.

## 4. Adapting for Part 2 (New Challenges)

This baseline is ready for **Part 2** with the following next steps (outlined in the notebook comments):
1.  **Load Real Data**: Replace `RNADataset` with a proper loader for the competition's `.parquet` files.
2.  **Template Integration**: For Part 2, you can extend the `RibonanzaBlock` to accept template coordinates (from PDB search) as an extra input channel.
3.  **Refinement**: Implement an equivariant GNN (like EGNN) on top of the NeRF output for fine-grained atomic refinement.
