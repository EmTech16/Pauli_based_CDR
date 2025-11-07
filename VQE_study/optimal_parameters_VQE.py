from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np 
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import  SparsePauliOp
from qiskit.circuit.library import  EfficientSU2
from scipy.optimize import minimize
import os

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



def cost_function(params, intial_ansatz, num_qubits, cost_hamiltonian_function, device_backend, shots): 
    """
    Compute the expectation value of a cost Hamiltonian for a parameterized circuit.

    The ansatz is parameterized with `params`, executed on the given backend, and 
    measured in the Z and X bases. Expectation values of the corresponding Pauli terms 
    are combined with their coefficients to estimate the total energy.

    Parameters
    ----------
    params : array-like
        Parameter values for the ansatz circuit.
    intial_ansatz : qiskit.QuantumCircuit
        Parameterized quantum circuit before binding parameters.
    num_qubits : int
        Number of qubits.
    cost_hamiltonian_function : callable
        Function that returns the Hamiltonian (as a SparsePauliOp).
    device_backend : Backend
        Quantum backend or simulator to run the circuits.
    shots : int
        Number of measurement shots.

    Returns
    -------
    float
        Estimated expectation value of the Hamiltonian.
    """


    parametrised_circuit=deepcopy(intial_ansatz)
    parametrised_circuit=parametrised_circuit.assign_parameters(params)
    
    ham_sparse_pauli_op=cost_hamiltonian_function(num_qubits)
   
    pauli_strings = ham_sparse_pauli_op.paulis.to_labels()  # List of Pauli strings
    coeff_list = np.real(ham_sparse_pauli_op.coeffs )               # Corresponding coefficients

    hamiltonian_value=0

    pauli_string='I'*n_qubits
    Zcounts=executor(parametrised_circuit,pauli_string,shots=shots,device_backend=device_backend,counts=True) 

    pauli_string='X'*n_qubits #no change of basis
    Xcounts=executor(parametrised_circuit,pauli_string,shots=shots,device_backend=device_backend,counts=True) 
    #print(pauli_strings)
    for element,weight in zip(pauli_strings,coeff_list): 
        if 'Z' in element: 
            expected_value=calculate_expect_value_from_counts(Zcounts,element,shots)
            hamiltonian_value+=weight*expected_value
        if 'X' in element: 
            expected_value=calculate_expect_value_from_counts(Xcounts,element,shots)
            hamiltonian_value+=weight*expected_value
            
    print(hamiltonian_value)
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





if __name__ == "__main__":
    
    
    shots=10**4 #Shots make it super slow
    device_backend=None #noiseles optimization
    n_qubits=10
    g=1
    g_list=np.arange(-2, 2.01, 0.25) 
    J_param=1.0 

    for g in g_list:
        num_instances = 10
        print(num_instances,g)
        instances = range(1, num_instances + 1)

        circuit=circuit_to_mitigate(n_qubits)
        num_parameteres=circuit.num_parameters


        # List of lists to store the parameters for each instance
        param_lists = [[] for _ in range(num_parameteres)]  # 2 * p because each layer has two parameters

        energy_list = []

        for instance in instances:
            init_params=[np.random.uniform(0,2*np.pi) for i in range(num_parameteres)]
            result=minimize(cost_function,init_params,args=(circuit,n_qubits,hamiltonian,device_backend,shots), method="COBYLA",tol=0.01)
            energy_list.append(result.fun)
            optimal_params=result.x 

            # Append the optimal parameters to the corresponding lists
            for idx, param in enumerate(optimal_params):
                param_lists[idx].append(param)            
        
            print(g,result)
            print(g,optimal_params)
        
        file_name='optimal_parameters/optimal_parameters_VQE_g_shots'+str(shots)+'_qubits'+str(n_qubits)+'_noiseless_g'+str(g)+'_J'+str(J_param)
        write_in_file(file_name,[instances] + param_lists + [energy_list],header='#instance, a, b, c, d, ..., energy')




    



  
    









        


