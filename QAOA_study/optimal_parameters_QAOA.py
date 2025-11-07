from copy import deepcopy
import numpy as np 
import os 

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

from scipy.optimize import minimize




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
    shots=10**5#Shots make it super slow
    device_backend=None #noiseles optimization
    n_qubits=10
    for p in range(1,11): # or any other value of p
        num_instances = 10
        instances = range(1, num_instances + 1)

        # List of lists to store the parameters for each instance
        param_lists = [[] for _ in range(2 * p)]  # 2 * p because each layer has two parameters

        energy_list = []

        for instance in instances:
            circuit=circuit_to_mitigate(n_qubits,reps=p)
            init_params=[np.random.uniform(0,2*np.pi),np.random.uniform(0,2*np.pi)]*p
            result=minimize(cost_function,init_params,args=(circuit,n_qubits,hamiltonian,device_backend,shots), method="COBYLA")
            energy_list.append(result.fun)
            optimal_params=result.x 

            # Append the optimal parameters to the corresponding lists
            for idx, param in enumerate(optimal_params):
                param_lists[idx].append(param)

            print(result)
            print(optimal_params)
        
        file_name='optimal_parameters/optimal_parameters_shots'+str(shots)+'_qubits'+str(n_qubits)+'_p'+str(p)+'_noiseless'
        write_in_file(file_name,[instances] + param_lists + [energy_list],header='#instance, a, b, c, d, ..., energy')
        


        

        



    
        









            


