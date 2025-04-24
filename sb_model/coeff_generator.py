import numpy as np
import os
import su_basis
import check_basis

def calculate_su_coefficients(matrix, basis_matrices):
    """Calculate SU coefficients for a given density matrix using SU basis matrices."""
    coefficients = []
    for B_i in basis_matrices:
        # Calculate coefficient as Tr(matrix * B_i) / Tr(B_i^2)
        a_i = np.trace(matrix @ B_i) / np.trace(B_i @ B_i)
        coefficients.append(a_i)
    return coefficients

def process_and_save_all_files(input_folder, output_folder, n, generate_su_basis):
    # Generate SU(n) basis matrices
    su_basis = generate_su_basis(n)
    check_basis.check(su_basis)
    
    #su_basis_complete = su_basis.copy()
    # Identity matrix (part of the basis)
    #identity_matrix = np.eye(n, dtype=complex)
    #su_basis_complete.append(identity_matrix)

    # Process all .npy files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            file_path = os.path.join(input_folder, filename)
            print(f"\nProcessing file: {file_path}")

            # Load the .npy file
            data = np.load(file_path)

            # Separate time and matrix data
            rho_data = data[:, 1:]  # Density matrix data
            time_data = data[:, 0]  # Time values
            
            # Ensure that rho_values has the correct number of elements for an n x n matrix
            assert n * n == rho_data.shape[1], "The number of rho columns must be a perfect square for an n x n matrix."

            # Prepare to store coefficients for each time step
            coefficients_all = []

            # Loop over each density matrix (each time step)
            for i, row in enumerate(rho_data):
                # Assume row is reshaped to an (n x n) density matrix
                matrix = row.reshape((n, n))
                # Extract only the real part of the diagonal elements
                real_diag_elements = np.real(np.diag(matrix))
                
                # Replace the diagonal elements in the matrix with the real parts
                np.fill_diagonal(matrix, real_diag_elements)
                #print("Updated matrix with real diagonal elements:\n", matrix)
                # Calculate SU coefficients for this density matrix
                coefficients = calculate_su_coefficients(matrix, su_basis)
                
                #coefficients.append(1/n)

                #print('coeff:', coefficients)
                # Reconstruct the matrix using the coefficients
                #reconstructed_matrix = sum(a * B for a, B in zip(coefficients, su_basis_complete))

                # Print the results for each time step
                #print(f"\nTime: {time_data[i]}")
                #print("Original Density Matrix:\n", matrix)
                #print("Reconstructed Matrix:\n", reconstructed_matrix)
                #print("Trace of reconstructed rho:", np.trace(reconstructed_matrix))
                # Append coefficients with corresponding time
                coefficients_all.append([time_data[i], *coefficients])

            # Convert the coefficients list to a NumPy array
            coefficients_all = np.array(coefficients_all)

            # Save the coefficients to a new .npy file in the output folder
            output_file_path = os.path.join(output_folder, filename.replace('.npy', '_coefficients.npy'))
            np.save(output_file_path, coefficients_all)
            print(f"Coefficients saved to: {output_file_path}")

# Example usage
input_folder = '/home/dell/arif/pypackage/sb/data/training_data/combined'  # Rho .npy files
output_folder = 'sb'
n_states = 2  # Number of states: Dimension of SU(n) group
process_and_save_all_files(input_folder, output_folder, n_states, su_basis.generate_su_basis)
