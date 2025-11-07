import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



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


if __name__ == "__main__":
    optimal_shots=10**5 #Training shots
    
    n_qubits=10
    d=2**n_qubits
    shots=10**5
    num_training_circuits=150
    fraction_non_clifford=0.2


    file_name='optimal_parameters/optimal_parameters_shots'+str(shots)+'_qubits'+str(n_qubits)+'_p'+str(3)+'_noiseless'
    data=read_from_file(file_name,header=True)
    instances=data[0]

   

file_name='results/results_average_over_instances_LINEAR_ANSATZ_errors_'+str(n_qubits)+'shots'+str(shots)+'num_training_circuits_'+str(num_training_circuits)+'fnc'+str(fraction_non_clifford)
data=read_from_file(file_name,header=True)

p_list=data[0]
noisy_euclidean=data[1]
cdr_euclidean_bit=data[2]
cdr_euclidean_pauli=data[3]

noisy_euclidean_pm=data[7]
cdr_euclidean_bit_pm=data[8]
cdr_euclidean_pauli_pm=data[9]




# Main plot
plt.figure(figsize=(8, 6))

plt.errorbar(p_list, noisy_euclidean, yerr=noisy_euclidean_pm, 
        label=r'Noisy', marker='s', markerfacecolor='none', linewidth=0.5, markersize=6, capsize=4, linestyle='-')

plt.errorbar(p_list, cdr_euclidean_bit, yerr=cdr_euclidean_bit_pm, 
        label=r'Direct', marker='o', markerfacecolor='none', color='green', linewidth=0.5, markersize=6, capsize=4, linestyle='-')
        

plt.errorbar(p_list, cdr_euclidean_pauli, yerr=cdr_euclidean_pauli_pm, 
                label=r'Pauli', marker='x', markerfacecolor='none', color='orange', linewidth=0.5, markersize=6, capsize=4, linestyle='-')


# Adding labels and a legend
plt.ylabel(r'$\text{Euc}$', fontsize=14)
plt.xlabel(r'$\ell$', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14) 
plt.legend(loc=(0.8, 0.4), fontsize=12)

# Inset Zoom Plot (Middle-Right Position)
axins = inset_axes(
    plt.gca(),
    width="50%",
    height="40%",
    bbox_to_anchor=(0.08, 0.01, 0.95, 0.95),  # (x, y, width, height) in axes fraction
    bbox_transform=plt.gca().transAxes,
    loc="upper left",
    borderpad=0
)

axins.errorbar(p_list, cdr_euclidean_bit, yerr=cdr_euclidean_bit_pm, 
               marker='o', markerfacecolor='none', color='green', linewidth=0.5, markersize=6, capsize=4, linestyle='-')

axins.errorbar(p_list, cdr_euclidean_pauli, yerr=cdr_euclidean_pauli_pm, 
               marker='x', markerfacecolor='none', color='orange', linewidth=0.5, markersize=6, capsize=4, linestyle='-')


# Adjusting inset axes limits (zoom-in range)
axins.set_xlim(min(p_list)-1, max(p_list)+1)  # Modify to focus on a specific range
axins.set_ylim(-0.005, max(max(cdr_euclidean_pauli), max(cdr_euclidean_bit)) * 1.2+0.005)  # Adjust zoom

axins.set_xticks(p_list[1::2])
axins.set_xticklabels([str(int(p)) for p in p_list[1::2]], fontsize=9)
# **Add labels to the inset**
#axins.set_xlabel(r'p', fontsize=10)  # X-axis label
#axins.set_ylabel(r'Error', fontsize=10)  # Y-axis label

# Mark the zoom-in location
mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")

# Save and close
plt.savefig('INSET_pauli_vs_bit_euclidean_'+str(n_qubits)+'shots'+str(shots)+'num_training_circuits'+str(num_training_circuits)+'fnc'+str(fraction_non_clifford)+'.png', dpi=300)
plt.show()
#plt.close()

