# QAOA Study Workflow

This folder contains the scripts used to generate and analyze the numerical 
simulations for the QAOA study. To reproduce the results, run the scripts 
in the following order:

1. **optimal_parameters_QAOA.py**  
   Performs noiseless optimization of the QAOA ansatz.  
   - Generates multiple optimized parameter sets (instances) for each problem size or parameter configuration.  
   - These optimized circuits serve as the baseline for later noisy simulations and error mitigation.  

2. **generate_training_data_counts_p.py**  
   Generates training circuits and corresponding measurement counts.  
   - Constructs near-Clifford training circuits by replacing a fraction of non-Clifford gates.  
   - Executes these circuits to collect measurement counts in the Z and X bases.  
   - Produces training datasets needed to fit the learning-based error mitigation models.  

3. **learning_ansatz_direct_p.py**  
   Fits the Direct CDR model.  
   - Uses training data to learn the linear ansatz mapping between noisy and ideal p(bitsrings).  
   - Stores the fitted parameters for later application in the Direct mitigation method.  

4. **learning_ansatz_pauli_p.py**  
   Fits the Pauli CDR model.  
   - Learns independent linear mappings for each Pauli term in the p(bitstrings)  
   - Provides the fitted parameters for the Pauli mitigation method.  

5. **results.py**  
   Runs the optimized QAOA circuits with noise and applies error mitigation.  
   - Collects noisy measurement results for the circuits of interest.  
   - Applies both Direct and Pauli mitigation using the parameters learned in previous steps.  
   - Outputs the mitigated results for analysis.  

6. **plot.py**  
   Produces figures summarizing the QAOA study results.  
   - Plots errors (e.g., relative error in expectation values or distributions) as a function of problem parameters.  
   - Compares unmitigated results with Direct and Pauli mitigation approaches.  
   - Generates final figures for inclusion in reports or publications.  

---

## Notes
- Run the scripts in the above order for a consistent workflow.  
- Optional benchmarking (e.g., ZNE) is not included in this folder.  
- Dependencies: Qiskit, Mitiq, NumPy, Matplotlib.  

