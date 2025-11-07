import os 
import itertools
from copy import deepcopy
import numpy as np 
import json
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2

from qiskit.circuit.library import EfficientSU2


from scipy.optimize import curve_fit


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




def expectation_values_to_mitigate(Zcounts, Xcounts, num_qubits, cost_hamiltonian_function): 
    """
    Compute expectation values of all Pauli strings in the Hamiltonian, 
    along with the total Hamiltonian expectation value.

    For each Pauli string in the Hamiltonian, this function computes the
    expectation value using pre-collected measurement counts in the Z and X
    bases. The results are returned together with the Hamiltonian expectation
    value.

    Parameters
    ----------
    Zcounts : dict
        Measurement counts obtained in the Z basis.
    Xcounts : dict
        Measurement counts obtained in the X basis.
    num_qubits : int
        Number of qubits in the circuit.
    cost_hamiltonian_function : callable
        Function that generates the cost Hamiltonian for a given number of qubits.
        Must return a `SparsePauliOp` or equivalent with `.paulis` and `.coeffs`.

    Returns
    -------
    list_of_pauli_strings : list[str]
        List of Pauli strings involved in the Hamiltonian, plus an entry `'H'`
        corresponding to the total Hamiltonian.
    list_of_pauli_strings_values : list[float]
        Expectation values of the Pauli strings, with the final entry being the
        Hamiltonian expectation value.

    Notes
    -----
    - Uses `calculate_expect_value_from_counts` to evaluate expectation values
      from measurement counts.
    - Currently only supports Pauli terms containing 'Z' or 'X'.
    - The symbol `'H'` is appended to the list to represent the total Hamiltonian.
    """


    ham_sparse_pauli_op=cost_hamiltonian_function(num_qubits)
   
    pauli_strings = ham_sparse_pauli_op.paulis.to_labels()  # List of Pauli strings
    coeff_list = np.real(ham_sparse_pauli_op.coeffs )               # Corresponding coefficients

    hamiltonian_value=0
    list_of_pauli_strings=[]
    list_of_pauli_strings_values=[]

    for element,weight in zip(pauli_strings,coeff_list): 
        if 'Z' in element: 
            expected_value=calculate_expect_value_from_counts(Zcounts,element,shots)
            list_of_pauli_strings.append(element)
            list_of_pauli_strings_values.append(expected_value)
            hamiltonian_value+=weight*expected_value
        if 'X' in element: 
            expected_value=calculate_expect_value_from_counts(Xcounts,element,shots)
            hamiltonian_value+=weight*expected_value
            list_of_pauli_strings.append(element)
            list_of_pauli_strings_values.append(expected_value)

    list_of_pauli_strings.append('H')
    list_of_pauli_strings_values.append(hamiltonian_value)
    return list_of_pauli_strings,list_of_pauli_strings_values


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
    
    with open(filename, 'w') as file:
        json.dump(list_of_dicts, file, indent=4)  # indent=4 for pretty printing


def load_list_of_dicts_from_file(filename):
    """
    Reads a list of dictionaries from a text file in JSON format and returns it.
    Automatically appends .txt extension if not provided.

    Args:
    filename (str): Name of the text file to read the data from.
    
    Returns:
    list: The list of dictionaries recovered from the file.
    """
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    with open(filename, 'r') as file:
        return json.load(file)



def linear_ansatz(p_noisy, a, b):
    '''
    Args: 
        a, b: floats (parameters to be optimized)
        p_noisy: probability to mitigate (x-data)
    Returns: 
        p_mit: mitigated probability (y-data)
    '''
    return a * p_noisy + b


def linear_ansatz_pauli(pauli_noisy, a,):
    '''
    Args: 
        a: Float (parameter to be optimized)
        pauli_noisy: pauli to mitigate (x-data)
    Returns: 
        pauli_mit: mitigated probability (y-data)
    '''
    return pauli_noisy/a





        


    




 

    
if __name__ == "__main__":
    shots=10**4#Shots make it super slow
    device_backend=FakeAlmadenV2()
    
    n_qubits=10
    g_list=np.arange(-2, 2.01, 0.25) 
    J_param=1.0 


    for g in g_list: 
    
        file_name='optimal_parameters/optimal_parameters_VQE_g_shots'+str(shots)+'_qubits'+str(n_qubits)+'_noiseless_g'+str(g)+'_J'+str(J_param)
        data=read_from_file(file_name,header=True)
        instances=data[0]
        param_lists = data[1:-1]  

        num_training_circuits=150
        fraction_non_clifford=0.2

        
        for i in range(len(instances)): 
            optimal_params = [param_list[i] for param_list in param_lists]
            instance=instances[i]
            print(g,'Instance = '+str(instance))
            circuit=circuit_to_mitigate(n_qubits)
            optimal_circuit=circuit.assign_parameters(optimal_params) 
            #print(f"Instance {instance}, Optimal Params: {optimal_params}")


            file_name='training_counts/training_lowE_noisy_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            noisy_counts=load_list_of_dicts_from_file(file_name)


            file_name='training_counts/training_lowE_cliff_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            cliff_counts=load_list_of_dicts_from_file(file_name)

            file_name='training_counts/trainingX_lowE_noisy_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            noisy_counts_X=load_list_of_dicts_from_file(file_name)


            file_name='training_counts/trainingX_lowE_cliff_data_instance'+str(instance)+'_shots'+str(shots)+'_qubits'+str(n_qubits)+'_fakeAlmaden_frac_non_clifford_'+str(fraction_non_clifford)+'num_training_circuits'+str(num_training_circuits)+'g'+str(g)+'_J'+str(J_param)
            cliff_counts_X=load_list_of_dicts_from_file(file_name)


            noisy_pauli_values=[] #list containing num_training_circuits vectors 
            for noisy_count,noisy_count_X in zip(noisy_counts,noisy_counts_X): 
                pauli_strings,pauli_values=expectation_values_to_mitigate(noisy_count,noisy_count_X,n_qubits,hamiltonian)
                noisy_pauli_values.append(pauli_values)

            cliff_pauli_values=[] #list containing num_training_circuits vectors 
            for cliff_count,cliff_count_X in zip(cliff_counts,cliff_counts_X): 
                pauli_strings,pauli_values=expectation_values_to_mitigate(cliff_count,cliff_count_X,n_qubits,hamiltonian)
                cliff_pauli_values.append(pauli_values)


            optimal_a_list=[]
            optimal_b_list=[] #list of opitmal paremeters for each bitstring

            num_pauli_operators=len(pauli_strings)-1

            #print('Num circuits per pauli', num_circuits_per_pauli)
            i=-1
            for pauli_string in pauli_strings: 
                i=i+1
               
                #for each probability 
                noisy_values_fit=[element[i] for element in noisy_pauli_values]
                cliff_values_fit=[element[i] for element in cliff_pauli_values]

                # Use curve_fit to find optimal values of a and b
                if i!=len(pauli_strings)-1: 
                   
                    params, covariance = curve_fit(linear_ansatz, noisy_values_fit, cliff_values_fit , p0=[1.0,0.0])
                    # Extract the optimized values of a and b
                    a_opt, b_opt = params
                    optimal_a_list.append(a_opt)
                    optimal_b_list.append(b_opt)
                else: 
                    print(pauli_string, 'Hamiltonian')
                    params, covariance = curve_fit(linear_ansatz, noisy_values_fit, cliff_values_fit , p0=[1.0,0.0])
                    # Extract the optimized values of a and b
                    a_opt, b_opt = params
                    optimal_a_list.append(a_opt)
                    optimal_b_list.append(b_opt)


            file_name='learned_ansatz/learningVQE_LINEAR_ANSATZ_lowE_optimal_parameters_'+str(shots)+'_qubits'+str(n_qubits)+'instance'+str(instance)+'g'+str(g)+'_J'+str(J_param)+'150training'
            write_in_file(file_name,[pauli_strings,optimal_a_list,optimal_b_list],header='#bitstring  ,optimal a _list, optimal_b_list')

                


            






    
        









            


