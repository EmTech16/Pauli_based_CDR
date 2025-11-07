import os
import itertools
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')  
import numpy as np 


from qiskit import  transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp,Statevector
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import CouplingMap



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
        transpiled_circuit = transpile(circuit_change_basis, simulator)#,optimization_level=optimal_transpile_level)
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






def order_counts(counts1, counts2, n_qubits):
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
    
    return ordered_counts1, ordered_counts2


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
        List of numpy arrays or lists: Each array contains the elements of a column. 
        If a column has strings that cannot be converted to floats, that column is returned as a list of strings.
    '''
    
    with open(f"{file_name}.txt", "r") as file:
        # Read all lines from the file
        lines = file.readlines()
        
        # Skip the header if provided
        if header:
            lines = lines[1:]
        
        # Split each line into individual elements (assuming tab-delimited)
        data = [line.strip().split('\t') for line in lines]
        
        # Transpose the rows back to columns
        columns = list(map(list, zip(*data)))
    
    # Try to convert each element of a column to float
    def try_convert_column(column):
        try:
            return np.array([float(x) for x in column])
        except ValueError:
            # If conversion to float fails, return the column as a list of strings
            return column
    
    # Apply try_convert_column to each column
    converted_columns = [try_convert_column(column) for column in columns]
    
    return converted_columns



def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.

    Parameters:
        p (list or numpy array): The first probability distribution (e.g., P).
        q (list or numpy array): The second probability distribution (e.g., Q).

    Returns:
        float: The KL divergence D(P || Q).
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)

    # Add a small constant to avoid zero probabilities
    epsilon=1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # Normalize the distributions to ensure they sum to 1
    p /= np.sum(p)
    q /= np.sum(q)

    # Compute KL divergence
    kl_div = np.sum(p * np.log(p / q))
    return kl_div


def get_energy_from_bitstring(bitstring): 
    """
    Computes the expected value (energy) of the Hamiltonian for a given bitstring state.

    This function takes a bitstring, converts it into a corresponding quantum state (in 
    the computational basis), and calculates the expected value of the Hamiltonian operator 
    for this state. The Hamiltonian is represented as a sum of Pauli strings, and it is 
    constructed internally using the `hamiltonian()` function, which returns the operator 
    along with the Pauli strings and coefficients used in the construction.

    Args:
        bitstring (str): A string representing a quantum state in the computational basis, 
                         consisting of '0's and '1's (e.g., '01110').

    Returns:
        float: The expected value (energy) of the Hamiltonian for the given bitstring state.

    Notes:
        - The function assumes the bitstring is in Qiskit's convention (big-endian format).
        - The `hamiltonian()` function is expected to return the Hamiltonian operator, Pauli 
          strings, and coefficients.
    """
    num_qubits=len(bitstring)
    hamiltonian_operator,pauli_strings,coeff_list=hamiltonian(num_qubits)

    state = Statevector.from_label(bitstring)
    operator_matrix=hamiltonian_operator.to_matrix()

    expected_value = state.expectation_value(operator_matrix)
    return float(expected_value) #float because is hermitian
    

def linear_ansatz(p_noisy, a, b):
    '''
    Args: 
        a, b: floats (parameters to be optimized)
        p_noisy: probability to mitigate (x-data)
    Returns: 
        p_mit: mitigated probability (y-data)
    '''
    return a * p_noisy + b

def linear_ansatz_pauli(pauli_noisy, a):
    '''
    Args: 
        a: Float (parameter to be optimized)
        pauli_noisy: pauli to mitigate (x-data)
    Returns: 
        pauli_mit: mitigated probability (y-data)
    '''
    return pauli_noisy/a

    
def expected_value_pauli_string_bitstring(pauli_string,bitstring): 
    '''This function takes as input a pauli_string and a bittring and outpus <bitstrin|Pauli|bitstring>#
    Args: 
        pauli_string: String 
        bitstring: string 
    Returns: 
        <bitstrin|Pauli|bitstring>: Float 
    '''
    expected_value=1
    if 'X' in pauli_string or 'Y' in pauli_string:
        return 0
    else:
        for operator, bit in zip(pauli_string,bitstring): 
            if operator=='Z' and bit=='1': 
                expected_value=expected_value*-1
    

    return expected_value

def expected_value_bitstring_from_paulis(bitstring,pauli_strings,pauli_values): 
    '''This function take a bitstring and outputs the exoectation value of that bitstring obatined from the expectation values of paulis 
    args: 
        bitstring: String 
        pauli_strins: List of Strings 
        pauli_values: List of expected values of pauli  
        '''
    
    # Check if pauli_strings and pauli_values have the same length
    if len(pauli_strings) != len(pauli_values):
        raise ValueError("pauli_strings and pauli_values must have the same length")
    

    expected_bitstring=0 
    for pauli_string,pauli_value in zip(pauli_strings,pauli_values): 
        expected_bitstring+=expected_value_pauli_string_bitstring(pauli_string,bitstring)*pauli_value 

    return expected_bitstring/2**n_qubits


def create_nearest_neighbor_coupling_map(n):
    # Define nearest-neighbor connections for n qubits
    connections = [[i, i + 1] for i in range(n - 1)]
    # Create the CouplingMap with the connections
    coupling_map = CouplingMap(connections)
    return coupling_map


def create_nearest_neighbor_coupling_map(n):
    # Define nearest-neighbor connections for n qubits
    connections = [[i, i + 1] for i in range(n - 1)]
    # Create the CouplingMap with the connections
    coupling_map = CouplingMap(connections)
    return coupling_map



def expected_value_and_error(samples):
    """
    Calculate the expected value and statistical error of a given list of samples.

    Parameters:
        samples (list or numpy array): A list or array of sample values.

    Returns:
        tuple: (expected_value, statistical_error)
            - expected_value: The mean of the samples.
            - statistical_error: The standard error of the mean.
    """
    # Convert samples to a NumPy array for convenience
    samples = np.array(samples, dtype=np.float64)
    
    # Compute the expected value (mean)
    expected_value = np.mean(samples)
    
    # Compute the statistical error (standard error of the mean)
    statistical_error = np.std(samples, ddof=1) / np.sqrt(len(samples))
    
    return expected_value, statistical_error



if __name__ == "__main__":
    shots=10**5 #Training shots
    actual_shots=shots #Implementation shots
    n_qubits=10
    d=2**n_qubits
    num_training_circuits=150
    fraction_non_clifford=0.2

    device_backend=FakeAlmadenV2()
    p_list=list(range(1,11))

    noisy_euclidean=[]
    cdr_euclidean_pauli=[]
    cdr_euclidean_bit=[]

    noisy_kl=[]
    cdr_kl_pauli=[]
    cdr_kl_bit=[]

    noisy_euclidean_pm=[]
    cdr_euclidean_pauli_pm=[]
    cdr_euclidean_bit_pm=[]

    noisy_kl_pm=[]
    cdr_kl_pauli_pm=[]
    cdr_kl_bit_pm=[]
    
        
    for p in p_list: 
    
        file_name='optimal_parameters/optimal_parameters_shots'+str(shots)+'_qubits'+str(n_qubits)+'_p'+str(p)+'_noiseless'
        data=read_from_file(file_name,header=True)
        instances=data[0]
        param_lists = data[1:2*p + 1]  

        noisy_euclidean_p=[]
        cdr_euclidean_pauli_p=[]
        cdr_euclidean_bit_p=[]
    
        noisy_kl_p=[]
        cdr_kl_pauli_p=[]
        cdr_kl_bit_p=[]
        #num_instances=1

        #print(num_instances)

        for l in range(10): 
            optimal_params = [param_list[l] for param_list in param_lists]
            instance=instances[l]
            print('Instance = '+str(instance))
            circuit=circuit_to_mitigate(n_qubits,reps=p)
            optimal_circuit=circuit.assign_parameters(optimal_params) 


            #----------------------------------------NOISY RESULT----------------------------------------------------------------------
            #Result non mitigated
            pauli_string='I'*n_qubits #no change of basis
            #noisy_counts=executor_depolarising(optimal_circuit,pauli_string,shots=actual_shots) #dictionary
            noisy_counts=executor(optimal_circuit,pauli_string,shots=shots,device_backend=device_backend,counts=True)
            y_counts=executor(optimal_circuit,pauli_string,shots=actual_shots,device_backend=None,counts=True) #ideal dictionary

            noisy_o_counts,ideal_o_counts=order_counts(noisy_counts,y_counts,n_qubits)
            #print(p)
            # Sort items by value in descending order, then slice the top 10
            top_10 = sorted(y_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            # Print each of the top 10 key-value pairs
            for key, value in top_10:
                print(f"{key}: {value}")
        
            #vectors of probabilities
            noisy_count_vector=np.array(list(noisy_o_counts.values()))/actual_shots
            ideal_count_vector=np.array(list(ideal_o_counts.values()))/actual_shots


            #----------------------------------------------PAULI CDR---------------------------------------------------------------------------------
            file_name='learned_ansatz/paulis_lowE_linear_ansatz_optimal_parameters_'+str(shots)+'_qubits'+str(n_qubits)+'_p'+str(p)+'instance'+str(instance)
            data=read_from_file(file_name,header=True)
            pauli_operators=data[0]
            #optimal_pauli_params=data[1]
            optimal_a_pauli=data[1]
            optimal_b_pauli=data[2]

            bitstrings=list(noisy_o_counts.keys()) 
            

            noisy_paulis=[]
            ideal_paulis=[]
            mit_paulis=[]
            #for pauli_string,parmater in zip(pauli_operators,optimal_pauli_params): 
            for pauli_string,parmater_a,parmater_b in zip(pauli_operators,optimal_a_pauli,optimal_b_pauli): 
                noisy_pauli=calculate_expect_value_from_counts(noisy_o_counts,pauli_string,shots=shots)
                noisy_paulis.append(noisy_pauli) #noisy paulis 
                ideal_paulis.append(calculate_expect_value_from_counts(ideal_o_counts,pauli_string,shots=shots)) #ideal paulis
                #mit_paulis.append(linear_ansatz_pauli(noisy_pauli,parmater)) #we get the mitigated paulis
                mit_paulis.append(linear_ansatz(noisy_pauli,parmater_a,parmater_b)) #we get the mitigated paulis


            #original_pauli_error=[abs(x-y) for x,y in zip(noisy_paulis,ideal_paulis)]
            #mit_pauli_error=[abs(x-y) for x,y in zip(mit_paulis,ideal_paulis)]
            #we have mitigated paulis, now let's get probabilties 
        
            ideal_p_vector_paulis=[expected_value_bitstring_from_paulis(bitstring,pauli_operators,ideal_paulis) for bitstring in bitstrings]
            noisy_p_vector_paulis=[expected_value_bitstring_from_paulis(bitstring,pauli_operators,noisy_paulis) for bitstring in bitstrings]
            #cdr_p_vector_paulis = [max(expected_value_bitstring_from_paulis(bitstring,pauli_operators,mit_paulis),0) for bitstring in bitstrings]  #only positive value,s if negative then zero 
            #cdr_p_vector_paulis=cdr_p_vector_paulis/sum(cdr_p_vector_paulis) 
            cdr_p_vector_paulis= [expected_value_bitstring_from_paulis(bitstring,pauli_operators,mit_paulis) for bitstring in bitstrings] #no caping, no normalisation


            #----------------------------------------------BIT CDR---------------------------------------------------------------------------------
            file_name='learned_ansatz/bitstrings_lowE_linear_ansatz_optimal_parameters_'+str(shots)+'_qubits'+str(n_qubits)+'_p'+str(p)+'instance'+str(instance)
            data=read_from_file(file_name,header=True)
            a_optimal_list=data[1]
            b_optimal_list=data[2]


            bitstrings=list(noisy_o_counts.keys()) 
            
        

            mitigated_count_vector_bit=[] #we generate the mitigated vector
            for p_noisy,a,b in zip(noisy_count_vector,a_optimal_list,b_optimal_list):
                p_mit=linear_ansatz(p_noisy,a,b)
                #if p_mit>0: 
                #    mitigated_count_vector_bit.append(p_mit) 
                #else: 
                #    mitigated_count_vector_bit.append(0) #if probability is negative we keep it at zero

                mitigated_count_vector_bit.append(p_mit) #no capping
        
            #cdr_p_count_vector_bit =mitigated_count_vector_bit/sum(mitigated_count_vector_bit) #impose normalisation
    
            cdr_p_count_vector_bit = mitigated_count_vector_bit # no capping no normalisation 


            original_error=[(x-y)**2 for x,y in zip(noisy_count_vector,ideal_count_vector)]
            cdr_erorr=[(x-y)**2 for x,y in zip(cdr_p_vector_paulis,ideal_count_vector)]
            cdr_bit_error=[(x-y)**2 for x,y in zip(cdr_p_count_vector_bit,ideal_count_vector)]

            noisy_euclidean_p.append(np.sqrt(sum(original_error)))
            cdr_euclidean_pauli_p.append((sum(cdr_erorr)))
            cdr_euclidean_bit_p.append(sum(cdr_bit_error))

            noisy_kl_p.append(kl_divergence(ideal_count_vector,noisy_count_vector))
            cdr_kl_pauli_p.append(kl_divergence(ideal_count_vector,cdr_p_vector_paulis))
            cdr_kl_bit_p.append(kl_divergence(ideal_count_vector,cdr_p_count_vector_bit))




        noisy_euclidean_p_f,pm_noisy_euclidean_p_f=expected_value_and_error(noisy_euclidean_p)   
        cdr_euclidean_p_f,pm_cdr_euclidean_p_f=expected_value_and_error(cdr_euclidean_pauli_p)
        cdr_euclidean_p_f_bit,pm_cdr_euclidean_p_f_bit=expected_value_and_error(cdr_euclidean_bit_p)


        noisy_kl_p_f,pm_noisy_kl_p_f=expected_value_and_error(noisy_kl_p)
        cdr_kl_p_f,pm_cdr_kl_p_f=expected_value_and_error(cdr_kl_pauli_p)
        cdr_kl_p_f_bit,pm_cdr_kl_p_f_bit=expected_value_and_error(cdr_kl_bit_p)
        
    

        noisy_euclidean.append(noisy_euclidean_p_f)
        noisy_euclidean_pm.append(pm_noisy_euclidean_p_f)

        cdr_euclidean_pauli.append(cdr_euclidean_p_f)
        cdr_euclidean_pauli_pm.append(pm_cdr_euclidean_p_f)

        cdr_euclidean_bit.append(cdr_euclidean_p_f_bit)
        cdr_euclidean_bit_pm.append(pm_cdr_euclidean_p_f_bit)


        noisy_kl.append(noisy_kl_p_f)
        noisy_kl_pm.append(pm_noisy_kl_p_f)

        cdr_kl_pauli.append(cdr_kl_p_f)
        cdr_kl_pauli_pm.append(pm_cdr_kl_p_f)

        cdr_kl_bit.append(cdr_kl_p_f_bit)
        cdr_kl_bit_pm.append(pm_cdr_kl_p_f_bit)


    file_name='results/results_average_over_instances_LINEAR_ANSATZ_errors_'+str(n_qubits)+'shots'+str(shots)+'num_training_circuits_'+str(num_training_circuits)+'fnc'+str(fraction_non_clifford)
    write_in_file(file_name,[p_list,noisy_euclidean,cdr_euclidean_bit,cdr_euclidean_pauli,noisy_kl,cdr_kl_bit,cdr_kl_pauli,noisy_euclidean_pm,cdr_euclidean_bit_pm,cdr_euclidean_pauli_pm,noisy_kl_pm,cdr_kl_bit_pm,cdr_kl_pauli_pm],header='#p, idela E, encoded E, noisy E, encoded euclidean, noisy euclidean ')



            
