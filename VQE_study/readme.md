# VQE Simulations Workflow

This repository contains the Python scripts used to generate the numerical simulations 
and figures presented in the manuscript (VQE). To reproduce the results, the scripts should 
be executed in the following order:

1. **optimal_parameters_VQE.py**  
   Performs the noiseless optimization of the VQE ansatz.  
   - Generates 10 independent optimization instances for each value of the parameter \(g\).  
   - Saves the optimized variational parameters that will later be used for noisy simulations 
     and error mitigation.  
   - This step ensures that the cost function is minimized in the absence of noise, so that 
     comparisons across methods are fair.  

2. **generate_training_data_counts.py**  
   Generates the training circuits and their corresponding measurement counts.  
   - Constructs near-Clifford training circuits by replacing a fraction of non-Clifford gates.  
   - Computes measurement outcomes (counts) in both Z and X bases, for both noisy execution 
     and noiseless Clifford simulation.  
   - Stores the training datasets that are required to fit the learning-based mitigation models.  

3. **learning_ansatz.py**  
   Learns the optimal parameters of the linear ansatz used for error mitigation.  
   - Fits the mapping between noisy and noiseless expectation values using the training data.  
   - Supports both the **Direct** and **Pauli** methods, storing the fitted parameters for later use.  
   - This step provides the classical post-processing models that will be applied to mitigate errors.  

4. **results.py**  
   Runs the target (optimized) circuits and applies error mitigation.  
   - Executes the circuits with noise and collects the raw measurement data.  
   - Applies the previously learned ansatz (Direct or Pauli) to mitigate expectation values.  
   - Outputs the final mitigated results for each value of \(g\).  

5. **zne.py** *(optional)*  
   Runs Zero-Noise Extrapolation (ZNE) as a benchmark method.  
   - Implements Richardson extrapolation using unitary folding to artificially increase circuit depth.  
   - Produces mitigated expectation values for comparison with CDR-based approaches.  
   - This step is optional and mainly serves as a reference to validate the improvements 
     achieved by learning-based methods.  

6. **plot.py**  
   Generates the figures used in the manuscript.  
   - Plots the relative error in the expectation value of the Hamiltonian as a function of \(g\).  
   - Can also plot comparisons across different mitigation strategies (Direct, Pauli, ZNE, or unmitigated).  
   - Produces the final figures that summarize the performance of each method.  
