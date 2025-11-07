import itertools
from copy import deepcopy
import numpy as np 
import json
import os


from qiskit import qpy
from qiskit import  transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler import CouplingMap

from mitiq import cdr



def hamiltonian(n_qubits): 
    """
    Build the transverse-field Ising Hamiltonian for a given number of qubits.

    The Hamiltonian is defined as:
        H = -J ∑ Z_i Z_{i+1} - g ∑ X_i
    where J and g are global parameters.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.

    Returns
    -------
    SparsePauliOp
        Hamiltonian expressed as a sum of Pauli operators with coefficients.
    """


    # Pauli terms and coefficients
    pauli_terms = []
    coefficients = []

    # Add ZZ terms for nearest neighbors
    for i in range(n_qubits - 1):
        pauli_string = ["I"] * n_qubits
        pauli_string[i] = "Z"
        pauli_string[i + 1] = "Z"
        pauli_terms.append("".join(pauli_string))
        coefficients.append(-J_param)

    # Add X terms for external field
    for i in range(n_qubits):
        pauli_string = ["I"] * n_qubits
        pauli_string[i] = "X"
        pauli_terms.append("".join(pauli_string))
        coefficients.append(-g)

    # Construct the Hamiltonian
    hamiltonian = SparsePauliOp.from_list(list(zip(pauli_terms, coefficients)))

    return hamiltonian

def circuit_to_mitigate(n_qubits, hamiltonian=hamiltonian):
    """
    Initializes the circuit we want to error mitigate.

    Args: 
        n_qubits (int): Number of qubits for the circuit.
        hamiltonian: Hamiltonian object that defines the system (e.g., SparsePauliOp).
    
    Returns: 
        qc (qiskit.QuantumCircuit): The initialized ansatz circuit.
    """

    
    hamiltonian_of_interest=hamiltonian(n_qubits)
    # Check if the number of qubits in the Hamiltonian matches n_qubits
    if hamiltonian_of_interest.num_qubits != n_qubits:
        raise ValueError("Number of qubits in the Hamiltonian does not match the specified n_qubits.")
    
    # Create an EfficientSU2 ansatz circuit
    ansatz = EfficientSU2(hamiltonian_of_interest.num_qubits)

    return ansatz



def executor(circuit, pauli_string, shots=10000, device_backend=FakeAlmadenV2(),counts=False): 
    '''This function recieves a circuit and outputs the expectation value we want to mitigate. 
    Args: 
        circuit: A qiskit.QuantumCircuit 
        pauli_string: Expectation value that we want to measure
        shots: Number of shots, int. Default is 1000 
        device backend=None: Indicates the noisy simulated abckend that we will be using, If None, no noise model is applied
        counts: Default to False. If true returns the counts for standarised pauli string
    OUTPUT: 
        expectation_value: Expectation value that we want to mitigate, in this case ZZ in qubits 2,3  

    '''

    n_qubits=circuit.num_qubits
    if len(pauli_string)!=n_qubits: 
        raise ValueError("Pauli string does not act on the same number of qubits as there are in the circuit")
    
    if counts: 
        circuit_change_basis=change_basis_all(circuit,pauli_string) #add change of basis required to measure in the correspondant basis
    else: 
        circuit_change_basis=change_basis(circuit,pauli_string)
    circuit_change_basis.measure_all()
    #print(circuit_change_basis.draw())

    if device_backend==None: #No noise
        # Transpile for simulator
        simulator = AerSimulator()
        transpiled_circuit = transpile(circuit_change_basis, simulator)
        # Run and get counts
        result = simulator.run(transpiled_circuit,shots=shots).result()
        counts = result.get_counts()

    else: 
        #start_time_1 = time.time()
        simulator = AerSimulator.from_backend(device_backend)
        transpiled_circuit = transpile(circuit_change_basis, simulator)
        result = simulator.run(transpiled_circuit,shots=shots).result()
        counts = result.get_counts()


    
    if counts: 
        return counts 
    else: 
        expectation_value=calculate_expect_value_from_counts(counts,pauli_string,shots)
        return expectation_value



def change_basis(circuit,pauli_string): 
    '''Applies additional gates to the circuit in order to change basis to perform the correct measurment we are interested in. 
    Args: 
        circuit: qiskit.QuantumCircuit to add gates to 
        pauli_string: Pauli string inidcating the expected value we want to measure (same lenght as num_qubits)
    Return: 
        modified_circuit: qiskit.QuantumCircuit with added gates.
    '''
    modified_circuit=deepcopy(circuit)

    
    for i in range(1,len(pauli_string)+1): 
        if pauli_string[-i]=='X': 
            modified_circuit.h(i-1)
        

    #measure in X basis for qubit 2 (third)
    

    return modified_circuit 

def change_basis_all(circuit,pauli_string): 
    '''Applies additional gates to the circuit in order to change basis to perform the correct measurment we are interested in. 
    Args: 
        circuit: qiskit.QuantumCircuit to add gates to 
        pauli_string: Pauli string inidcating the expected value we want to measure (same lenght as num_qubits)
    Return: 
        modified_circuit: qiskit.QuantumCircuit with added gates.

    FUNCTION VALID ONLY FOR OBSERVABLES CONTAING ALL X OR ALL Z'S; MEAUSURES ALL QUBITS IN SAME BASIS 
    '''
    modified_circuit=deepcopy(circuit)
    if 'X' in pauli_string: 
        for i in range(len(pauli_string)): 
            modified_circuit.h(i)
    return modified_circuit
    

def calculate_expect_value_from_counts(counts,pauli_string,shots):
    '''Calculates the expectation value of a given observable given the counts results. 
    Args: 
        counts: Dict. {'0000' : 1000, '0100' : 9998, ... } !!!! 0100 the 1 refers to the third qubit (-3) !!!! 
        pauli_string: pauli_string that we want to measure
        shots: Int. Number of shots
    Returns: 
        expectation value: Float.  A given expectation value 
    '''
    expectation_value = 0
    for outcome in counts:
        operator_value=1
        for i in range(len(pauli_string)):
            if pauli_string[i]!='I': 
                qubit_value=int(outcome[i])
                operator_value=operator_value*-2*(qubit_value-1/2)
        expectation_value+=operator_value*counts[outcome]
    expectation_value=expectation_value/shots

    ''' Calculate expectation value of XX for the second and third qubits of 4 possible 
    expectation_value = 0
    for outcome in counts:
        # Extract the state of the second and third qubits
        # Assume that the qubits are indexed from right to left (0 to n-1)
        second_qubit = int(outcome[-4])
        third_qubit = int(outcome[-3])
        
        # Assign value based on ZZ measurement outcomes
        zx_value= -(2*(second_qubit-1/2))*(2*(third_qubit-1/2))
        #print(zz_value)
        # Update the expectation value
        expectation_value +=zx_value * counts[outcome]

    # Normalize by the number of shots to get the expectation value
    expectation_value = expectation_value/shots
    '''

    #Probability of obtaining 00

    #expectation_value=counts['00']/shots
            
    return expectation_value







def pauli_observable(pauli_string, num_qubits): 
    '''Returns a Pauli Observable created from a single Pauli string
    Args: 
        pauli_string: String indicating the pauli operator. CAUTION: In IZXI, Z is acting on the third qubit (-3) and X is acting on the second one (-2)
        num_qubits: Number of qubits in the circuit 
    Returns: 
        observable: qiskit.quantum_info.Pauli object
    '''

    # Check if the pauli_string is a string
    if not isinstance(pauli_string, str): 
        raise ValueError("Pauli string must be a string")

    # Check if the length of pauli_string matches num_qubits
    #if len(pauli_string) != num_qubits: 
    #    raise ValueError("Pauli string does not act on the same number of qubits as there are in the circuit")
    
    # Create and return the Pauli observable

    #we could also return a np.matrix with 
    #return Pauli(pauli_string).to_matrix()
    return Pauli(pauli_string)


def number_to_n_digit_binary(number, n):
    '''Takes a basis index number and converts it to the correspondent computational basis element for n qubits. 
    Args: 
        number: Int. Index of the basis element
        n: Int. number of qubits 
    Returns: Str. String of measured qubits. (already in the qiskit ordering)

    '''
    # Convert the number to binary and remove the '0b' prefix
    binary_str = bin(number)[2:]
    
    # Pad the binary string with leading zeros to make it n digits long
    padded_binary_str = binary_str.zfill(n)
    
    return padded_binary_str




def create_nearest_neighbor_coupling_map(n):
    # Define nearest-neighbor connections for n qubits
    connections = [[i, i + 1] for i in range(n - 1)]
    # Create the CouplingMap with the connections
    coupling_map = CouplingMap(connections)
    return coupling_map



def order_counts(counts1, counts2, counts3, n_qubits):
    """
    Orders three dictionaries containing bitstring counts for quantum experiments 
    and ensures that all dictionaries have the same bitstrings, filling in 
    any missing bitstrings with a count of 0.

    Args:
    counts1 (dict): First dictionary of bitstring counts from a quantum experiment.
    counts2 (dict): Second dictionary of bitstring counts from a quantum experiment.
    counts3 (dict): Third dictionary of bitstring counts from a quantum experiment.
    n_qubits (int): Number of qubits used in the experiment, determines the bitstring length.

    Returns:
    tuple: Three ordered dictionaries (counts1, counts2, and counts3) where all possible 
           bitstrings of length `n_qubits` are present, with missing bitstrings filled in with a count of 0.
    """
    
    # Generate all possible bitstrings of length n_qubits
    bitstrings = [''.join(bitstring) for bitstring in itertools.product('01', repeat=n_qubits)]
    
    # Create new dictionaries that include all bitstrings with counts, filling missing ones with 0
    ordered_counts1 = {bitstring: counts1.get(bitstring, 0) for bitstring in bitstrings}
    ordered_counts2 = {bitstring: counts2.get(bitstring, 0) for bitstring in bitstrings}
    ordered_counts3 = {bitstring: counts3.get(bitstring, 0) for bitstring in bitstrings}
    
    return ordered_counts1, ordered_counts2, ordered_counts3


def write_in_file(file_name, lists, header=None):
    '''This function creates a file named file_name.txt and writes the data indicated by lists.
    After writing the data, the function closes the file.
    
    Args: 
        file_name (str): The name of the file (without .txt extension).
        lists (list of lists): A list of lists where each inner list represents a column in the file.
                               All inner lists should have the same length.
        header (str or None): If provided, this will be written as the header line in the file.
    '''
    os.makedirs(os.path.dirname(f"{file_name}.txt"), exist_ok=True)
    
    # Check that all lists are of the same length
    if not all(len(lst) == len(lists[0]) for lst in lists):
        raise ValueError("All lists must have the same length")
    
    # Open the file for writing
    with open(f"{file_name}.txt", "w") as file:
        # Write the header if provided
        if header:
            file.write(f"{header}\n")
        
        # Write data row by row
        for row in zip(*lists):  # zip transposes the list of lists
            file.write("\t".join(map(str, row)) + "\n")  # Join elements with tabs and write to file


def read_from_file(file_name, header=None):
    '''This function opens a file named file_name.txt and reads the data.
    After reading the data, the function closes the file.
    
    Args: 
        file_name (str): The name of the file (without .txt extension).
        header (str or None): If provided, the first line of the file is omitted.
        
    Returns: 
        List of numpy arrays: Each array contains the elements of a column, converted to floats.
    '''
    
    with open(f"{file_name}.txt", "r") as file:
        # Read all lines from the file
        lines = file.readlines()
        
        # Skip the header if provided
        if header:
            lines = lines[1:]
        
        # Split each line into individual elements (assuming tab-delimited)
        # Convert each element to float
        data = [list(map(float, line.strip().split('\t'))) for line in lines]
        
        # Transpose the rows back to columns (the reverse of zip(*lists))
        columns = list(map(list, zip(*data)))
    
    # Convert each column list to a NumPy array
    return [np.array(column) for column in columns]


def save_list_of_dicts_to_file(list_of_dicts, filename):
    """
    Saves a list of dictionaries to a text file in JSON format.
    Automatically appends .txt extension if not provided.

    Args:
    list_of_dicts (list): List of dictionaries to save.
    filename (str): Name of the text file to save the data.
    """
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as file:
        json.dump(list_of_dicts, file, indent=4)  # indent=4 for pretty printing


def cost_function_clifford(circuit, cost_hamiltonian_function, shots): 
    """
    Estimate the expectation value of a Hamiltonian using a near-Clifford simulator.

    This function evaluates the expectation value of the Hamiltonian defined by
    `cost_hamiltonian_function` on the provided circuit. The circuit is executed
    in the Z and X measurement bases, and the corresponding Pauli terms are
    reconstructed from the measurement outcomes.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        The near-Clifford quantum circuit to be evaluated.
    cost_hamiltonian_function : callable
        Function that generates the cost Hamiltonian for a given number of qubits.
        Must return a `SparsePauliOp` or equivalent object with `.paulis` and `.coeffs`.
    shots : int
        Number of measurement shots per basis.

    Returns
    -------
    float
        Estimated expectation value of the Hamiltonian for the given circuit.

    Raises
    ------
    ValueError
        If a Pauli term other than those containing 'Z' or 'X' is encountered.

    Notes
    -----
    - Uses `executor` to run the circuit with measurements in the Z and X bases.
    - Assumes the existence of `calculate_expect_value_from_counts` to compute
      expectation values from raw measurement counts.
    - Currently only supports Hamiltonians with Pauli-X and Pauli-Z terms.
    """


    num_qubits=circuit.num_qubits
    
    ham_sparse_pauli_op=cost_hamiltonian_function(num_qubits)
   
    pauli_strings = ham_sparse_pauli_op.paulis.to_labels()  # List of Pauli strings
    coeff_list = np.real(ham_sparse_pauli_op.coeffs )               # Corresponding coefficients
    
    hamiltonian_value=0
    #Z pauli strings 
    pauli_string='Z'*n_qubits
    #countsZ=near_clifford_simulator_qrack(circuit,pauli_string,shots=shots,counts=True) 
    countsZ=executor(circuit,pauli_string,shots=shots,device_backend=None,counts=True)
    #X pauli strings
    pauli_string='X'*n_qubits
    #countsX=near_clifford_simulator_qrack(circuit,pauli_string,shots=shots,counts=True) 
    countsX=executor(circuit,pauli_string,shots=shots,device_backend=None,counts=True)

    for element,weight in zip(pauli_strings,coeff_list): 
        if 'Z' in element: 
            expected_value = calculate_expect_value_from_counts(countsZ, element, shots)
        elif 'X' in element:
            expected_value = calculate_expect_value_from_counts(countsX, element, shots)
        else:
            raise ValueError("element must contain either 'Z' or 'X'.")
        
        hamiltonian_value+=weight*expected_value

    return hamiltonian_value

def generate_training_circuits(circuit,
                           num_training_circuits, fraction_non_clifford,
                           ideal_simulator=None, observable=None, shots=10000):
    """
    Generate a set of training circuits for Clifford Data Regression (CDR).

    Starting from the original circuit, this function produces a larger pool of
    near-Clifford training circuits by randomly replacing a fraction of
    non-Clifford gates. The circuits are then post-selected based on their
    (classically estimated) low-energy expectation values, ensuring that the
    final training set contains circuits close to the target ground state.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Original parameterized quantum circuit.
    num_training_circuits : int
        Number of training circuits to return after post-selection.
    fraction_non_clifford : float
        Fraction of non-Clifford gates in the original circuit that are not
        replaced by Clifford gates.
    ideal_simulator : callable, optional
        Not currently used in this implementation. Reserved for future
        extensions to compute exact expectation values.
    observable : object, optional
        Auxiliary variable for `ideal_simulator` (unused here).
    shots : int, default=10000
        Number of measurement shots for evaluating low-energy circuits.

    Returns
    -------
    list[qiskit.QuantumCircuit]
        A list of `num_training_circuits` near-Clifford training circuits,
        sorted by their estimated energy.

    Notes
    -----
    - Internally generates 10× more circuits than needed and post-selects those
      with the lowest energy, following Ref.~[Czarnik_2021].
    - Relies on `cdr.clifford_training_data.generate_training_circuits` to
      construct candidate circuits and `cost_function_clifford` to estimate
      their energy.
    """

    
    basis_gates = ['rz', 'sx', 'cx']  # 'sx' represents the √X gate in Qiskit
    # Transpile the circuit into the desired gate set
    coupling_map = create_nearest_neighbor_coupling_map(n_qubits)
    new_qiskit_circuit = transpile(circuit, basis_gates=basis_gates,coupling_map=coupling_map)
    #print(new_qiskit_circuit)

    pauli_string='I'*circuit.num_qubits #no change of basis

    
    #Generate training circuits (with low energy)
    training_circuits=cdr.clifford_training_data.generate_training_circuits(new_qiskit_circuit,num_training_circuits*10,fraction_non_clifford,method_select='uniform', method_replace='uniform', random_state=None)
    print('Pretraining circuits generated')
    #Training data
    pre_x_cliff_list=[]
    for i in range(num_training_circuits*10):       
        modified_circuit=training_circuits[i]
        x_cliff=cost_function_clifford(modified_circuit,hamiltonian,shots=shots)
        pre_x_cliff_list.append(x_cliff)

    # Combine the training_circuits and pre_x_cliff_list into a single list of tuples
    circuit_x_cliff_pairs = list(zip(training_circuits, pre_x_cliff_list))
    # Sort the list of tuples based on the x_cliff values
    circuit_x_cliff_pairs.sort(key=lambda x: x[1])
    # Extract the sorted training_circuits list
    sorted_training_circuits = [circuit for circuit, x_cliff in circuit_x_cliff_pairs][:num_training_circuits]
    print('Training circuits generated')

    return sorted_training_circuits


def generate_training_counts_from_training_circuits(training_circuits,
                           ideal_simulator=None, observable=None, shots=10000): 
    """
    Generate training data from a list of training circuits.

    For each training circuit, this function executes the circuit in both the Z and X
    measurement bases on a noisy backend and on a noiseless (Clifford) simulator. It
    returns the corresponding counts, optionally including ideal counts if an ideal
    simulator is provided.

    Parameters
    ----------
    training_circuits : list[QuantumCircuit]
        List of training circuits to be executed.
    ideal_simulator : callable, optional
        Function to compute ideal expectation values (without shot noise).
        If None (default), the Clifford simulator results are used as a proxy.
    observable : object, optional
        Auxiliary variable passed to the ideal simulator, if provided.
    shots : int, default=10000
        Number of measurement shots per circuit execution.

    Returns
    -------
    counts_noisy_list : list[dict]
        Noisy counts in the Z basis for each training circuit.
    counts_cliff_list : list[dict]
        Noiseless (Clifford simulator) counts in the Z basis.
    counts_y_list : list[dict]
        Ideal counts if `ideal_simulator` is provided, otherwise same as `counts_cliff_list`.
    counts_noisy_list_X : list[dict]
        Noisy counts in the X basis for each training circuit.
    counts_cliff_list_X : list[dict]
        Noiseless (Clifford simulator) counts in the X basis.
    counts_y_list_X : list[dict]
        Ideal counts in the X basis if `ideal_simulator` is provided, otherwise same as `counts_cliff_list_X`.

    Notes
    -----
    - Assumes the existence of helper functions:
      `executor(circuit, pauli_string, shots, device_backend, counts=True)` and
      `order_counts(noisy_counts, cliff_counts, y_counts, num_qubits)`.
    - Uses a global `device_backend` and `circuit` object inside the loop, which should
      be defined externally or passed explicitly for consistency.
    """

    
    sorted_training_circuits=training_circuits

    counts_noisy_list=[] #initialize training data 
    counts_cliff_list=[] 
    counts_y_list =[] 


    counts_noisy_list_X=[] #initialize training data 
    counts_cliff_list_X=[] 
    counts_y_list_X =[] 

    for i in range(num_training_circuits): 
        if (i + 1) % 10 == 0 or i == num_training_circuits - 1:  # Print every 10 iterations and at the last iteration
            # Calculate the percentage of progress
            progress = (i + 1) / num_training_circuits * 100
            # Print progress and remaining percentage
            print(f"Iteration {i + 1}: {progress:.2f}% completed, {100 - progress:.2f}% remaining.")
        
        modified_circuit=sorted_training_circuits[i]
        
        pauli_string='I'*circuit.num_qubits #no change of basis
        noisy_counts=executor(modified_circuit,pauli_string,shots=shots,device_backend=device_backend,counts=True) 
        cliff_counts=executor(modified_circuit,pauli_string,shots=shots,device_backend=None,counts=True)
        if ideal_simulator!=None: 
            #y=ideal_simulator(modified_circuit,observable) 
            y_counts=cliff_counts
        else: 
            y_counts=cliff_counts

        o_noisy_counts,o_cliff_counts,o_y_counts=order_counts(noisy_counts,cliff_counts,y_counts,modified_circuit.num_qubits)

        counts_noisy_list.append(o_noisy_counts)
        counts_cliff_list.append(o_cliff_counts)
        counts_y_list.append(o_y_counts)


        pauli_string='X'*circuit.num_qubits #no change of basis
        noisy_counts=executor(modified_circuit,pauli_string,shots=shots,device_backend=device_backend,counts=True) 
        cliff_counts=executor(modified_circuit,pauli_string,shots=shots,device_backend=None,counts=True)
        if ideal_simulator!=None: 
            #y=ideal_simulator(modified_circuit,observable) 
            y_counts=cliff_counts
        else: 
            y_counts=cliff_counts

        o_noisy_counts,o_cliff_counts,o_y_counts=order_counts(noisy_counts,cliff_counts,y_counts,modified_circuit.num_qubits)

        counts_noisy_list_X.append(o_noisy_counts)
        counts_cliff_list_X.append(o_cliff_counts)
        counts_y_list_X.append(o_y_counts)


    return counts_noisy_list, counts_cliff_list, counts_y_list, counts_noisy_list_X, counts_cliff_list_X, counts_y_list_X



    



if __name__ == "__main__":
    shots=10**4  #Shots make it super slow
    device_backend=FakeAlmadenV2()
    
    n_qubits=10
    g_list=np.arange(-2, 2.01, 0.25) 
    J_param=1.0 

    num_training_circuits=150
    fraction_non_clifford=0.2
    for g in g_list:
    
        file_name='optimal_parameters/optimal_parameters_VQE_g_shots'+str(shots)+'_qubits'+str(n_qubits)+'_noiseless_g'+str(g)+'_J'+str(J_param)
        data=read_from_file(file_name,header=True)
        instances=data[0]
        param_lists = data[1:-1]  


        
        for i in range(len(instances)): 
            optimal_params = [param_list[i] for param_list in param_lists]
            instance=instances[i]
            print(g,'Instance = '+str(instance))
            circuit=circuit_to_mitigate(n_qubits)
            optimal_circuit=circuit.assign_parameters(optimal_params) 
            #print(f"Instance {instance}, Optimal Params: {optimal_params}")
            #print(len(optimal_params))

        
            training_circuits=generate_training_circuits(optimal_circuit,num_training_circuits,fraction_non_clifford,shots=shots,ideal_simulator=False)
            
            folder_name = 'training_circuits_files'
            os.makedirs(folder_name, exist_ok=True)
            file_name='set_of_T_circuits_lowE_instance'+str(instance)+'_qubits'+str(n_qubits)+'_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            full_path = os.path.join(folder_name, file_name + '.qpy')
            # Save the circuits to the specified folder
            with open(full_path, 'wb') as file:
                qpy.dump(training_circuits, file)
            
            noisy_counts,clifford_counts,ideal_counts,noisy_counts_X,clifford_counts_X,_ =generate_training_counts_from_training_circuits(training_circuits,shots=shots,ideal_simulator=False)

            file_name='training_counts/training_lowE_noisy_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            save_list_of_dicts_to_file(noisy_counts,file_name)

            file_name='training_counts/training_lowE_cliff_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            save_list_of_dicts_to_file(clifford_counts,file_name)

            file_name='training_counts/trainingX_lowE_noisy_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            save_list_of_dicts_to_file(noisy_counts_X,file_name)


            file_name='training_counts/trainingX_lowE_cliff_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            save_list_of_dicts_to_file(clifford_counts_X,file_name)

            





    
        









            


