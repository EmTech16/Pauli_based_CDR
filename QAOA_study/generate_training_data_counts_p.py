import itertools
from copy import deepcopy
import numpy as np 
import json
import os

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import  SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import CouplingMap

from mitiq import cdr




def hamiltonian1(num_qubits): 
    """
    Construct a Hamiltonian with nearest-neighbor ZZ interactions.

    The Hamiltonian has the form:
        H = ∑ Z_i Z_{i+1},
    where the sum runs over all pairs of adjacent qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system.

    Returns
    -------
    SparsePauliOp
        Hamiltonian expressed as a sum of Pauli strings with coefficients.
    pauli_strings : list[str]
        List of Pauli strings representing the ZZ interaction terms.
    coeff_list : numpy.ndarray
        Array of coefficients (set to +1 for each ZZ term).

    Notes
    -----
    - The sign convention may differ from Qiskit QAOA implementations,
    hence coefficients are explicitly set to +1 here.
    - Only nearest-neighbor couplings are included.
    """

    pauli_list=[]
    coeff_list=[]
     #Now ZZ terms 
    for i in range(num_qubits-1): 
        paulis = ["I"] * num_qubits  # Create a list of 'I's
        paulis[i] = 'Z'  # Modify the i-th element
        paulis[i+1] = 'Z'
        pauli_list.append(paulis)
        coeff_list.append(1) # #to account for different sign in qiski qaoa

    coeff_list=np.asarray(coeff_list)

    pauli_strings = [''.join(sublist) for sublist in pauli_list]

    return SparsePauliOp(pauli_strings,coeff_list),pauli_strings,coeff_list

def hamiltonian2(num_qubits, g=2):
    """
    Construct a Hamiltonian with single-qubit X terms.

    The Hamiltonian has the form:
        H = g ∑ X_i,
    where the sum runs over all qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system.
    g : float, default=2
        Coupling strength (coefficient) for each X term.

    Returns
    -------
    SparsePauliOp
        Hamiltonian expressed as a sum of Pauli strings with coefficients.
    pauli_strings : list[str]
        List of Pauli strings representing the X terms.
    coeff_list : list[float]
        List of coefficients (set to g for each X term).

    Notes
    -----
    - The sign convention may differ from Qiskit QAOA implementations,
      hence coefficients are explicitly set to +g here.
    """

    pauli_list=[]
    coeff_list=[]
    #Start with X terms 
    for i in range(num_qubits): 
        paulis = ["I"] * num_qubits  # Create a list of 'I's
        paulis[i] = 'X'  # Modify the i-th element
        pauli_list.append(paulis)
        coeff_list.append(g) #to account for different sign in qiski qaoa
    pauli_strings = [''.join(sublist) for sublist in pauli_list]

    return SparsePauliOp(pauli_strings,coeff_list),pauli_strings,coeff_list


def hamiltonian(num_qubits, g=2): 
    """
    Construct the Hamiltonian whose ground state is targeted.

    By default, this function generates a 1D Ising-type Hamiltonian 
    with nearest-neighbor ZZ interactions:
        H = -∑ Z_i Z_{i+1}.
    
    An optional transverse-field term (g ∑ X_i) is included in the code 
    but currently commented out.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system.
    g : float, default=2
        Coupling strength for the (optional) X terms.

    Returns
    -------
    SparsePauliOp
        Hamiltonian expressed as a sum of Pauli strings with coefficients.
    pauli_strings : list[str]
        List of Pauli strings representing the interaction terms.
    coeff_list : numpy.ndarray
        Array of coefficients corresponding to each Pauli string.

    """
    pauli_list=[]
    coeff_list=[]
    
    

    #Now ZZ terms 
    for i in range(num_qubits-1): 
        paulis = ["I"] * num_qubits  # Create a list of 'I's
        paulis[i] = 'Z'  # Modify the i-th element
        paulis[i+1] = 'Z'
        pauli_list.append(paulis)
        coeff_list.append(-1) 

    coeff_list=np.asarray(coeff_list)

    pauli_strings = [''.join(sublist) for sublist in pauli_list]

    return SparsePauliOp(pauli_strings,coeff_list),pauli_strings,coeff_list

def cost_function(params, intial_ansatz, num_qubits, cost_hamiltonian_function, device_backend, shots): 
    """
    Compute the expectation value of a Hamiltonian for a parameterized ansatz.

    Parameters
    ----------
    params : array-like
        Parameter values for the ansatz circuit.
    intial_ansatz : qiskit.QuantumCircuit
        Parameterized ansatz circuit before binding.
    num_qubits : int
        Number of qubits.
    cost_hamiltonian_function : callable
        Function returning (SparsePauliOp, pauli_strings, coeff_list).
    device_backend : Backend
        Quantum backend or simulator.
    shots : int
        Number of measurement shots.

    Returns
    -------
    float
        Estimated expectation value of the Hamiltonian.
    """


    parametrised_circuit=deepcopy(intial_ansatz)
    parametrised_circuit=parametrised_circuit.assign_parameters(params)
    
    _,pauli_strings,coeff_list=cost_hamiltonian_function(num_qubits)
    
    hamiltonian=0

    #Z pauli strings 
    counts=executor(parametrised_circuit,pauli_strings[0],shots=shots,device_backend=device_backend,counts=True) 
    for element,weight in zip(pauli_strings,coeff_list): 
        expected_value=calculate_expect_value_from_counts(counts,element,shots)
        hamiltonian+=weight*expected_value
    
    #print(params,hamiltonian)
    return hamiltonian
    

def noisy_expected_hamiltonian(circuit,num_qubits,cost_hamiltonian_function,device_backend,shots):
    '''Computes the expectation value of the energy'''
    _,pauli_strings,coeff_list=cost_hamiltonian_function(num_qubits)
    
    hamiltonian=0

    #Z pauli strings 
    counts=executor(circuit,pauli_strings[0],shots=shots,device_backend=device_backend,counts=True) 
    for element,weight in zip(pauli_strings,coeff_list): 
        expected_value=calculate_expect_value_from_counts(counts,element,shots)
        hamiltonian+=weight*expected_value
    
    return hamiltonian


def cliff_expected_hamiltonian(circuit,num_qubits,cost_hamiltonian_function,shots):
    '''Computes the expectation value of the energy'''
    _,pauli_strings,coeff_list=cost_hamiltonian_function(num_qubits)

    hamiltonian=0



    #Z pauli strings 
    #counts=near_clifford_simulator_qrack(circuit,pauli_strings[0],shots=shots,counts=True)
    counts=executor(circuit,pauli_strings[0],shots=shots,device_backend=None,counts=True)
    for element,weight in zip(pauli_strings,coeff_list): 
        expected_value=calculate_expect_value_from_counts(counts,element,shots)
        hamiltonian+=weight*expected_value
    
    return hamiltonian



def circuit_to_mitigate(num_qubits,cost_hamiltonian_function=hamiltonian1,mixer_hamiltonian_fucntion=hamiltonian2,reps=2): 
    '''Intialises the circuit that we want to error mitigate
    Args: 
        num_qubits: 
        cost_hamiltonian: function that initialised the camiltonian
        reps: Number of layers in the QAOA ansatz
    
    OUTPUT: 
        qc: qiskit.QuantumCircuit
    '''
    cost_hamiltonian=cost_hamiltonian_function(num_qubits)[0]
    mixer_hamiltonian=mixer_hamiltonian_fucntion(num_qubits)[0]



    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian,mixer_operator=mixer_hamiltonian, reps=reps)

    #circuit.draw('mpl')
    #plt.show()

    return circuit




def executor(circuit, pauli_string, shots=10000, device_backend=None,counts=False): 
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
        #end_time_1 = time.time()
        #execution_time_1 = end_time_1 - start_time_1
        #print(f"Execution time for first block: {execution_time_1:.4f} seconds")

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


def generate_training_counts_from_training_circuits(training_circuits,
                           ideal_simulator=None,observable=None,shots=10000): 
    '''This function generate the training data X_noisy, X_cliff that will be later used for performing some type of fitting. 
    Args: 
        device backend: Indicates the noisy simulated abckend that we will be using, If None, no noise model is applied. Example: FakeVigoV2()
        circuit: Original circuit from wich we want to generate the training data. 
        num_training_circuits: Number of training circuits 
        fraction_non_clifford: Fraction of non clifford gates from the original circuit that are not replaced by clifford gates. 
        ideal_simulator: Func. None by default. If set to a function then the output contains the ideal expectation value (without shot nosie) computed with the provided ideal simulator. 
        observable:  Auxiliar variable for ideal simulator. 
        shots: num of shots for noisy executor and non clifford executor
        MORE TO COME (observable.... )

    Returns: 
        counts_noisy_list:  List of len(num_training_circuits) containing  the noisy counts of each circuit 
        counts_cliff_list:  List of len(num_training_circuits) containing  the simualted with near_clifford simualtor counts of each circuit
        counts_list:        List of len(num_training_circuits) conatining the ideal valuues. Y=clif_counts if no ideal simulator is provided. 

        
    FUNCTION TO BE IMPROVED WITH MORE OPTIONS 
    The function generates the training circuits using cdr.clifford_training_data.generate_training_circuits from Mitiq 
    '''
    
    sorted_training_circuits=training_circuits

    counts_noisy_list=[] #initialize training data 
    counts_cliff_list=[] 
    counts_y_list =[] 


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

    return counts_noisy_list, counts_cliff_list, counts_y_list

    


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




def generate_training_circuits(circuit,
                           num_training_circuits, fraction_non_clifford,
                           ideal_simulator=None, observable=None, shots=10000):
    """
    Generate near-Clifford training circuits for CDR.

    The function creates an enlarged pool of training circuits by replacing a
    fraction of non-Clifford gates in the original circuit, estimates their
    energies classically, and post-selects the lowest-energy ones.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Original ansatz circuit.
    num_training_circuits : int
        Number of training circuits to keep after post-selection.
    fraction_non_clifford : float
        Fraction of non-Clifford gates left unchanged.
    ideal_simulator : callable, optional
        Placeholder for future use (not used here).
    observable : object, optional
        Placeholder for future use (not used here).
    shots : int, default=10000
        Shots for estimating Clifford circuit energies.

    Returns
    -------
    list[qiskit.QuantumCircuit]
        List of post-selected training circuits.
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
        x_cliff=cliff_expected_hamiltonian(modified_circuit,modified_circuit.num_qubits,hamiltonian,shots=shots)
        pre_x_cliff_list.append(x_cliff)

    # Combine the training_circuits and pre_x_cliff_list into a single list of tuples
    circuit_x_cliff_pairs = list(zip(training_circuits, pre_x_cliff_list))
    # Sort the list of tuples based on the x_cliff values
    circuit_x_cliff_pairs.sort(key=lambda x: x[1])
    # Extract the sorted training_circuits list
    sorted_training_circuits = [circuit for circuit, x_cliff in circuit_x_cliff_pairs][:num_training_circuits]
    print('Training circuits generated')

    return sorted_training_circuits




if __name__ == "__main__":
    shots=10**5#Shots make it super slow
    device_backend=FakeAlmadenV2()
    n_qubits=10
    p_list=list(range(1,11))

    for p in p_list: 
    
        file_name='optimal_parameters/optimal_parameters_shots'+str(shots)+'_qubits'+str(n_qubits)+'_p'+str(p)+'_noiseless'
        data=read_from_file(file_name,header=True)
        instances=data[0]
        param_lists = data[1:2*p + 1]  

    
        for i in range(len(instances)): 
            optimal_params = [param_list[i] for param_list in param_lists]
            instance=instances[i]
            print('Instance = '+str(instance)+'p'+str(p))
            circuit=circuit_to_mitigate(n_qubits,reps=p)
            optimal_circuit=circuit.assign_parameters(optimal_params) 
            #print(f"Instance {instance}, Optimal Params: {optimal_params}")

            num_training_circuits=150
            fraction_non_clifford=0.2

            training_circuits=generate_training_circuits(optimal_circuit,num_training_circuits,fraction_non_clifford,shots=shots,ideal_simulator=False)

            noisy_counts,clifford_counts,ideal_counts =generate_training_counts_from_training_circuits(training_circuits,shots=shots,ideal_simulator=False)

            file_name='training_counts/training_lowE_noisy_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_p'+str(p)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)
            save_list_of_dicts_to_file(noisy_counts,file_name)


            file_name='training_counts/training_lowE_cliff_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_p'+str(p)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)
            save_list_of_dicts_to_file(clifford_counts,file_name)







  
    









        


