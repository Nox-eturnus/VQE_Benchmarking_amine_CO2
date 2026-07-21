# VQE Carbon Capture Study: Ten Amines + CO2

This repository contains a Jupyter Notebook that implements a Variational Quantum Eigensolver (VQE) to study the interaction of various amines with carbon dioxide (CO2). This workflow is part of a quantum computational chemistry study on carbon capture. 

## Repository Contents
- **`VQE_Amines_Workflow.ipynb`**: The Jupyter Notebook containing the VQE workflow. It includes DFT-optimized coordinates for the amines, the shared CO2 monomer, pre-reactive complexes, and the quantum simulation routines using Qiskit.

## Prerequisites
To run the code, you need Python installed on your system. You can run the `.ipynb` file using Jupyter Notebook, JupyterLab, VS Code, or Google Colab.

The notebook is self-contained and features an initial setup block that will automatically attempt to install the required Python packages via `pip`. The core dependencies include:
- `qiskit` (>=2.0, <3.0)
- `qiskit-nature` (>=0.7.2)
- `qiskit-algorithms` (>=0.3.0)
- `qiskit-aer` (>=0.15)
- `pyscf` (>=2.4.0)
- `matplotlib`
- `numpy`

## How to Run Locally
1. **Clone or Download** this directory to your local machine.
2. **Open a Terminal or Command Prompt** and navigate to this directory.
3. (Optional but recommended) **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
4. **Install Jupyter** (if you don't have it installed):
   ```bash
   pip install notebook
   ```
5. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
6. **Open and Run:** In your browser, click on `VQE_Amines_Workflow.ipynb` to open it. Run the notebook cell by cell (`Shift + Enter`). The first code cell will automatically install the necessary dependencies for you.

## How to Run on Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Select **File > Upload notebook**.
3. Upload `VQE_Amines_Workflow.ipynb` from this directory.
4. Go to **Runtime > Run all** (or press `Ctrl+F9`) to execute the workflow in your browser. The setup cell is pre-configured to install everything required on Colab.
