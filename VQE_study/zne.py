import itertools
from copy import deepcopy
import numpy as np 
import json
import os 
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit.circuit.library import  EfficientSU2

from scipy.optimize import curve_fit

from mitiq.zne.inference import RichardsonFactory as RF





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


    
def expectation_values_to_mitigate(Zcounts, Xcounts, num_qubits, cost_hamiltonian_function,shots): 
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


    print('Shots expected value from counts: ', shots)
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


def compute_hamiltonian_from_paulis(pauli_list, num_qubits, cost_hamiltonian_function): 
    """
    Compute the expectation value of the Hamiltonian from precomputed Pauli expectations.

    Given a list of expectation values for each Pauli string in the Hamiltonian, 
    this function reconstructs the total Hamiltonian expectation value by 
    combining them with the corresponding coefficients.

    Parameters
    ----------
    pauli_list : list[float]
        List of expectation values for the Pauli strings in the Hamiltonian, 
        ordered consistently with the Pauli terms returned by `cost_hamiltonian_function`.
    num_qubits : int
        Number of qubits in the system.
    cost_hamiltonian_function : callable
        Function that generates the cost Hamiltonian for a given number of qubits.
        Must return a `SparsePauliOp` or equivalent with `.paulis` and `.coeffs`.

    Returns
    -------
    float
        The expectation value of the Hamiltonian.

    Notes
    -----
    - Assumes that `pauli_list` has the same ordering as the Pauli terms in the 
      Hamiltonian.
    - Only uses linear combination of Pauli expectations and coefficients, with 
      no additional quantum execution required.
    """

    ham_sparse_pauli_op=cost_hamiltonian_function(num_qubits)
   
    pauli_strings = ham_sparse_pauli_op.paulis.to_labels()  # List of Pauli strings
    coeff_list = np.real(ham_sparse_pauli_op.coeffs )               # Corresponding coefficients

    
    hamiltonian_value=0

    i=-1 

    for element,weight in zip(pauli_strings,coeff_list): 
        i=i+1 
        expected_value=pauli_list[i]
        hamiltonian_value+=weight*expected_value
    
    return hamiltonian_value




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
       

    
    if counts: 
        return counts 
    else: 
        expectation_value=calculate_expect_value_from_counts(counts,pauli_string,shots)
        return expectation_value


def scale_transpiled_circuit(circuit,scale_factor): 
    '''Scales transpiled circuit by local folding UUdaggerU 
    Only accepts scale factor 1,3,5 '''

    if scale_factor not in {1, 3, 5,7}:
        raise ValueError(f"Invalid value for 'scale_facor: {scale_factor}. Must be 1, 3, or 5.")
    
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    
    for instr, qargs, cargs in circuit.data:
        #print(f"Gate: {instr.name}, Qubits: {[q._index for q in qargs]}, Params: {instr.params}")
        qubit_indices = [q._index for q in qargs]  # or q._index for older versions
        
        if instr.name == 'u3' or instr.name == 'u':
            theta, phi, lam = instr.params

            if scale_factor == 1:
                new_circuit.append(instr, qargs, cargs)

            elif scale_factor == 3:
                new_circuit.append(instr, qargs, cargs)
                new_circuit.append(instr.inverse(), qargs, cargs)
                new_circuit.append(instr, qargs, cargs)

            elif scale_factor == 5:
                new_circuit.append(instr, qargs, cargs)
                new_circuit.append(instr.inverse(), qargs, cargs)
                new_circuit.append(instr, qargs, cargs)
                new_circuit.append(instr.inverse(), qargs, cargs)
                new_circuit.append(instr, qargs, cargs)

            elif scale_factor==7: 
                new_circuit.append(instr, qargs, cargs)
                new_circuit.append(instr.inverse(), qargs, cargs)
                new_circuit.append(instr, qargs, cargs)
                new_circuit.append(instr.inverse(), qargs, cargs)
                new_circuit.append(instr, qargs, cargs)
                new_circuit.append(instr.inverse(), qargs, cargs)
                new_circuit.append(instr, qargs, cargs)


        elif instr.name == 'cx':
            ctrl, tgt = qubit_indices  # CX has 2 qubits

            if scale_factor == 1:
                new_circuit.cx(ctrl, tgt)

            elif scale_factor == 3:
                new_circuit.cx(ctrl, tgt)
                new_circuit.cx(ctrl, tgt)  # inverse
                new_circuit.cx(ctrl, tgt)

            elif scale_factor == 5:
                new_circuit.cx(ctrl, tgt)
                new_circuit.cx(ctrl, tgt)  # inverse
                new_circuit.cx(ctrl, tgt)
                new_circuit.cx(ctrl, tgt)  # inverse
                new_circuit.cx(ctrl, tgt)

            elif scale_factor==7: 
                new_circuit.cx(ctrl, tgt)
                new_circuit.cx(ctrl, tgt)  # inverse
                new_circuit.cx(ctrl, tgt)
                new_circuit.cx(ctrl, tgt)  # inverse
                new_circuit.cx(ctrl, tgt)
                new_circuit.cx(ctrl, tgt)  # inverse
                new_circuit.cx(ctrl, tgt)

        else:
            if instr.name in {'barrier', 'measure'}:
                new_circuit.append(instr, qargs, cargs)
            else:
                raise ValueError(f"Unsupported gate '{instr.name}' encountered in circuit.")


    return new_circuit

def executor_zne(circuit,scale_factor, pauli_string, shots=10000, device_backend=None,counts=False): 
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
    print(shots)
    
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
        transpiled_circuit.global_phase = 0
        scaled_circuit=scale_transpiled_circuit(transpiled_circuit,scale_factor)
        # Run and get counts
        result = simulator.run(scaled_circuit,shots=shots).result()
        counts = result.get_counts()

    else: 
        #start_time_1 = time.time()
        simulator = AerSimulator.from_backend(device_backend)
        transpiled_circuit = transpile(circuit_change_basis, simulator)
        
        transpiled_circuit.global_phase = 0
        scaled_circuit=scale_transpiled_circuit(transpiled_circuit,scale_factor)
        result = simulator.run(scaled_circuit,shots=shots).result()
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




def order_counts(counts1, counts2 , n_qubits):
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
    #ordered_counts3 = {bitstring: counts3.get(bitstring, 0) for bitstring in bitstrings}
    
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




def zne_extrapolate(scale_factors, noisy_energies, method='exponential'):
    """
    Extrapolate to zero noise using linear, exponential, Richardson, or quadratic polynomial methods.

    Parameters:
    -----------
    scale_factors : list or array
        Noise scale factors (e.g., [1, 3, 5]).

    noisy_energies : list or array
        Energy values at each noise scale factor.

    method : str
        'linear', 'exponential', 'richardson', or 'poly2'

    Returns:
    --------
    zero_noise_energy : float
        Extrapolated energy at scale factor 0.
    """
    scale_factors = np.array(scale_factors)
    noisy_energies = np.array(noisy_energies)

    if method == 'linear':
        # Linear fit: E(s) = a * s + b
        coeffs = np.polyfit(scale_factors, noisy_energies, deg=1)
        return coeffs[1]  # y-intercept: E(0)

    elif method == 'exponential':
        # Exponential fit: E(s) = a * exp(b * s) + c
        def exp_func(s, a, b, c):
            return a * np.exp(b * s) + c

        try:
            popt, _ = curve_fit(exp_func, scale_factors, noisy_energies, p0=(1, -0.1, 0.), maxfev=10000)
            return exp_func(0, *popt)
        except RuntimeError as e:
            print("RuntimeError during exponential fit:")
            print("Noisy energies:", noisy_energies)
            print("Scale factors:", scale_factors)
            return noisy_energies[0]

    elif method == 'richardson':
        # Richardson extrapolation (requires 3 scale factors)
        if len(scale_factors) != 3:
            raise ValueError("Richardson extrapolation requires exactly 3 scale factors.")
        s1, s2, s3 = scale_factors
        e1, e2, e3 = noisy_energies

        zero_noise_energy = (e1 * (s2**2 - s3**2) - e2 * (s1**2 - s3**2) + e3 * (s1**2 - s2**2)) / (
            (s2 - s1) * (s3 - s1) * (s3 - s2)
        )
        return zero_noise_energy

    elif method == 'poly2':
        # Quadratic fit: E(s) = a*s^2 + b*s + c
        if len(scale_factors) < 3:
            raise ValueError("Polynomial extrapolation requires at least 3 scale factors.")
        coeffs = np.polyfit(scale_factors, noisy_energies, deg=2)
        return np.polyval(coeffs, 0.0)  # E(0)

    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    shots=377500 #Shots make it super slow
    device_backend=FakeAlmadenV2()

    zne_shots=shots

    g_list=np.arange(-2, 2.01, 0.25) 
    J_param=1.0 

    
    relative_error_noisy=[]
    relative_error_direct_zne=[]
    relative_error_pauli_zne=[]

    relative_error_noisy_pm=[]
    relative_error_direct_pm_zne=[]
    relative_error_pauli_pm_zne=[]
    
        
    for g in g_list: 

        ideal_hamiltonian_list_j=[]
        noisy_hamiltonian_list_j=[]
        mitigated_hamiltonian_direct_list_j_zne=[]
        mitigatated_hamiltonian_pauli_list_j_zne=[]

        
            
        for l in range(10): 
            n_qubits=10
        
            file_name='optimal_parameters/optimal_parameters_VQE_g_shots'+str(shots)+'_qubits'+str(n_qubits)+'_noiseless_g'+str(g)+'_J'+str(J_param)
            data=read_from_file(file_name,header=True)
            instances=data[0]
            param_lists = data[1:-1]
            energy_list=data[-1]

            num_training_circuits=150
            fraction_non_clifford=0.2
        
            print(g,l)

            optimal_params = [param_list[l] for param_list in param_lists]
            instance=instances[l]
            
            circuit=circuit_to_mitigate(n_qubits)
            optimal_circuit=circuit.assign_parameters(optimal_params) 


            pauli_string='I'*n_qubits #no change of basis
            noisy_counts=executor(optimal_circuit,pauli_string,shots=shots,device_backend=device_backend,counts=True) #dictionary
            y_counts=executor(optimal_circuit,pauli_string,shots=shots,device_backend=None,counts=True) #ideal dictionary
            noisy_o_countsZ,ideal_o_countsZ=order_counts(noisy_counts,y_counts,n_qubits)
            

            pauli_string='X'*n_qubits #no change of basis
            noisy_counts=executor(optimal_circuit,pauli_string,shots=shots,device_backend=device_backend,counts=True) #dictionary
            y_counts=executor(optimal_circuit,pauli_string,shots=shots,device_backend=None,counts=True) #ideal dictionary
            noisy_o_countsX,ideal_o_countsX=order_counts(noisy_counts,y_counts,n_qubits)

            
            # Noisy values: 

            pauli_strings, noisy_results=expectation_values_to_mitigate(noisy_o_countsZ,noisy_o_countsX,n_qubits,hamiltonian,shots=shots)
            pauli_strings, ideal_results=expectation_values_to_mitigate(ideal_o_countsZ,ideal_o_countsX,n_qubits,hamiltonian,shots=shots)

            noisy_hamiltonian_list_j.append(noisy_results[-1])
            ideal_hamiltonian_list_j.append(ideal_results[-1])


            for scale_factor in [1,3,5,7]: 

                pauli_string='I'*n_qubits #no change of basis
                zne_counts=executor_zne(optimal_circuit,scale_factor,pauli_string,shots=zne_shots,device_backend=device_backend,counts=True) #dictionary
                zne_o_countsZ,_=order_counts(zne_counts,zne_counts,n_qubits)

                pauli_string='X'*n_qubits #no change of basis
                zne_countsX=executor_zne(optimal_circuit,scale_factor,pauli_string,shots=zne_shots,device_backend=device_backend,counts=True) #dictionary
                zne_o_countsX,_=order_counts(zne_countsX,zne_countsX,n_qubits)

                pauli_strings, zne_results = expectation_values_to_mitigate(zne_o_countsZ,zne_o_countsX,n_qubits,hamiltonian,shots=zne_shots)

                if scale_factor==1: zne_results_1=zne_results
                if scale_factor==3: zne_results_3=zne_results
                if scale_factor==5: zne_results_5=zne_results 
                if scale_factor==7: zne_results_7=zne_results             


            mitigated_results=[]

            i=-1
            for pauli_string in pauli_strings: 
                i=i+1
                noisy_values_scaled=[zne_results_1[i],zne_results_3[i],zne_results_5[i],zne_results_7[i]]
                scale_factors=[1,3,5,7]
                #mitigated_results.append(zne_extrapolate(scale_factors, noisy_values_scaled, method='poly2'))
                #print(noisy_values_scaled)
                mitigated_results.append(RF.extrapolate(scale_factors,noisy_values_scaled))

            
            mitigated_hamiltonian_direct_list_j_zne.append(mitigated_results[-1])
            mitigatated_hamiltonian_pauli_list_j_zne.append(compute_hamiltonian_from_paulis(mitigated_results,n_qubits,hamiltonian)) 


        relative_error_noisy_j=[np.abs(i_h-n_h)*100/np.abs(i_h) for i_h,n_h in zip(ideal_hamiltonian_list_j,noisy_hamiltonian_list_j)]
        relative_error_direct_j=[np.abs(i_h-n_h)*100/np.abs(i_h) for i_h,n_h in zip(ideal_hamiltonian_list_j,mitigated_hamiltonian_direct_list_j_zne)]
        relative_error_pauli_j=[np.abs(i_h-n_h)*100/np.abs(i_h) for i_h,n_h in zip(ideal_hamiltonian_list_j,mitigatated_hamiltonian_pauli_list_j_zne)]


        noisy,noisy_pm=expected_value_and_error(relative_error_noisy_j)
        direct,direct_pm=expected_value_and_error(relative_error_direct_j)
        pauli,pauli_pm=expected_value_and_error(relative_error_pauli_j)

 
        relative_error_noisy.append(noisy)
        relative_error_noisy_pm.append(noisy_pm)

        relative_error_direct_zne.append(direct)
        relative_error_direct_pm_zne.append(direct_pm)

        relative_error_pauli_zne.append(pauli)
        relative_error_pauli_pm_zne.append(pauli_pm)


    file_name='results/results_ZNE_7_ZNE_shots'+str(zne_shots)+'_mitiq_rich_relative_error_g_average_instances_VQE_'+str(shots)+'_qubits'+str(n_qubits)
    write_in_file(file_name,[g_list,relative_error_noisy,relative_error_noisy_pm,relative_error_direct_zne,relative_error_direct_pm_zne,relative_error_pauli_zne,relative_error_pauli_pm_zne],header='#g, ideal, noisy, mit, mit pauli ')

