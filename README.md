# Code for Generalization of Clifford Data Regression to the Pauli Error Channel: A Pauli-Based Mitigation Approach

This repository contains the necessary Python code to reproduce the results presented in the paper:

> **“Generalization of Clifford Data Regression to the Pauli Error Channel: A Pauli-Based Mitigation Approach”**  
> *Francesc Sabater and Carlos A. Riofrío*

The work explores and compares two learning-based quantum error mitigation techniques:
- **Direct Clifford Data Regression (CDR)**  
- **Pauli-based CDR**, a generalization that applies mitigation on decomposed Pauli terms, achieving higher accuracy under realistic noise models.

The repository is organized into two folders:
1. **`VQE/`** – Contains scripts and workflows for Variational Quantum Eigensolver (VQE) simulations and mitigation.
2. **`QAOA/`** – Contains scripts for Quantum Approximate Optimization Algorithm (QAOA) studies and histogram mitigation.

Each folder includes its own `README.md` with detailed instructions on how to execute the simulations, train mitigation models, and generate figures corresponding to the paper’s results.

---

## How to Reproduce the Results

To reproduce the numerical and hardware results presented in the paper follow the step-by-step workflows described in `VQE/README.md` and `QAOA/README.md`.

The datasets and figures produced will reproduce the results shown in the paper.

---

## Reference

If you use this repository or its code, please cite: (paper to be published soon)


## License

This repository is licensed under the **Apache License, Version 2.0**.  
You may obtain a copy of the license at:

> http://www.apache.org/licenses/LICENSE-2.0

---
