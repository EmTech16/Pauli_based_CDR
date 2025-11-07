import matplotlib.pyplot as plt
import numpy as np


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
    shots=10**4 #Training shots

    plot_ZNE=False

    n_qubits=10

    file_name='results/results_relative_error_g_average_instances_VQE_'+str(shots)+'_qubits'+str(n_qubits)

    data=read_from_file(file_name,header=True)
    g_list=data[0]
    relative_error_noisy = data[1]
    relative_error_noisy_pm = data[2]

    relative_error_direct = data[3] 
    relative_error_direct_pm = data[4]

    relative_error_pauli = data[5]
    relative_error_pauli_pm = data[6]



    if plot_ZNE: 

        zne_shots=377500
        file_name='results/results_ZNE_7_ZNE_shots'+str(zne_shots)+'_mitiq_rich_relative_error_g_average_instances_VQE_'+str(shots)+'_qubits'+str(n_qubits)
        data=read_from_file(file_name,header=True)


        relative_error_direct_zne = data[3] 
        relative_error_direct_zne_pm = data[4]


    # Main figure
    plt.figure(figsize=(8, 6))

    # Error bars for different errors
    plt.errorbar(g_list, relative_error_noisy, yerr=relative_error_noisy_pm, 
            label=r'$ \Delta E_o/E  $', marker='s', markerfacecolor='none', linewidth=0.5, 
            color='blue', markersize=6, capsize=4, linestyle='-')
    plt.errorbar(g_list, relative_error_direct, yerr=relative_error_direct_pm, 
            label=r'$ \Delta E_D/E  $', marker='o', markerfacecolor='none', color='green',
            linewidth=0.5, markersize=6, capsize=4, linestyle='-')
    plt.errorbar(g_list, relative_error_pauli, yerr=relative_error_pauli_pm, 
            label=r'$ \Delta E_P/E $', marker='x', markerfacecolor='none', color='orange', 
            linewidth=0.5, markersize=6, capsize=4, linestyle='-')#
    
    if plot_ZNE:
        plt.errorbar(g_list, relative_error_direct_zne, yerr=relative_error_direct_zne_pm, 
                label=r'$ \Delta E_Z/E $', marker='^', markerfacecolor='none', color='red', 
                linewidth=0.5, markersize=6, capsize=4, linestyle='-')


    # Labels and legend
    plt.ylabel(r'$ \Delta E/E (\%)$', fontsize=14)
    plt.xlabel(r'$g/J$', fontsize=14)
    plt.xticks(g_list[::2], fontsize=14)
    plt.yticks(fontsize=14)
    #plt.legend(loc='best', fontsize=12)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.77, 0.75))

    # Save figure
    plt.savefig('VQE_result.png')
    plt.show()



