import numpy as np
import tensorflow as tf

# supress tf warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def pinn_loss(y_true: tf.Tensor,
              y_pred: tf.Tensor, 
              n_states: int, 
              lambda_1: float, 
              lambda_2: float,
              lambda_3: float,
              lambda_4: float):
    def reconstruct_density_matrix(pred_vector, n_states):
        """
        Reconstruct a density matrix from a flattened prediction vector for an arbitrary N x N matrix.
        
        Args:
            pred_vector: Flattened prediction vector containing real and imaginary parts of diagonal and upper-off diagonal elements.
                         Shape: (batch_size, n_steps, num_elements), where num_elements = n_states + (n_states*(n_states-1))/2*2
            n_states: The number of states in the system (e.g., 2 for a 2x2 matrix).
        
        Returns:
            A tensor of reconstructed density matrices of shape (batch_size, n_steps, n_states, n_states).
        """

        num_elements = pred_vector.shape[-1]
        n_steps = num_elements // n_states**2
        batch_size = tf.shape(pred_vector)[0]
        # Prepare a TensorArray for the reconstructed matrices
        rho_matrices = tf.TensorArray(dtype=tf.complex64, size=n_steps)

        a, b = 0, n_states**2  # Indices for slicing `pred_vector`

        for t in range(n_steps):
            # Extract the prediction vector for the current time step
            y = pred_vector[:, a:b]  # Shape: (batch_size, n_states^2)
            # Initialize a batch-specific density matrix for this time step
            rho_matrix = tf.zeros((batch_size, n_states, n_states), dtype=tf.complex64)

            # Fill diagonal elements
            for i in range(batch_size):
                rho_i = y[i, :]  # Loop over batch
                # Diagonal elements
                idx = 0
                for j in range(n_states):
                    if j > 0:  #
                        idx += 2 * (n_states - j) + 1
                    rho_matrix = tf.tensor_scatter_nd_update(
                        rho_matrix,
                        [[i, j, j]],
                        [tf.cast(rho_i[idx], tf.complex64)],
                    )

                # Off-diagonal elements
                idx = 1 # Start of off-diagonal elements
                for row in range(n_states - 1):
                    for col in range(row + 1, n_states):
                        real_value = tf.cast(rho_i[idx], tf.complex64)
                        imag_value = tf.cast(rho_i[idx + 1], tf.complex64)

                        # Upper triangle
                        rho_matrix = tf.tensor_scatter_nd_update(
                            rho_matrix,
                            [[i, row, col]],
                            [real_value + 1j * imag_value],
                        )

                        # Lower triangle (conjugate symmetry)
                        rho_matrix = tf.tensor_scatter_nd_update(
                            rho_matrix,
                            [[i, col, row]],
                            [real_value - 1j * imag_value],
                        )
                        if col == n_states -1: # skip diagonal terms
                            idx += 3
                        else:
                            idx +=2  # Move to the next real/imag pair

            # Store the reconstructed matrix for this time step
            rho_matrices = rho_matrices.write(t, rho_matrix)

            a = b  # Move to the next time step
            b += n_states**2

        # Stack the reconstructed matrices along the time dimension (batch_size, nsteps, n_states, n_states)
        # swapping the dimensions of batch_size and nsteps by perm
        return tf.transpose(rho_matrices.stack(), perm=[1, 0, 2, 3])
    
    #Combine MSE loss with constraints for all time steps
    print("custom_loss.pinn_loss: Running with pinn loss") 
    density_matrices = reconstruct_density_matrix(y_pred, n_states)
    
    #
    hermitian_conj = tf.linalg.adjoint(density_matrices)  # Hermitian check
    hermitian_diff = tf.reduce_mean(tf.square(tf.abs(density_matrices - hermitian_conj)))

    # Calculate the eigenvalues using tf.linalg.eigvals
    eigenvalues = tf.math.real(tf.linalg.eigvalsh(density_matrices))  # Positivity check
    positivity_loss = tf.reduce_mean(tf.square(tf.nn.relu(-eigenvalues)))

    # Additional eigenvalue regularization to ensure eigenvalues are in [0, 1]
    eigenvalue_clip = tf.clip_by_value(eigenvalues, 0.0, 1.0)
    eigenvalue_regularization = tf.reduce_mean(tf.square(eigenvalue_clip - eigenvalues))
    
    # Apply trace constraint (penalize the trace being different from 1)
    trace_loss = tf.reduce_mean(tf.square(tf.math.real(tf.linalg.trace(density_matrices)) - 1))  # Trace must be 1
    # Compute the Mean Squared Error loss (MSE)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # MSE between true and predicted density matrices

    # Total loss
    return mse_loss + lambda_1 * trace_loss + lambda_2* hermitian_diff + lambda_3 * positivity_loss + lambda_4 * eigenvalue_regularization

def su_loss(y_true: tf.Tensor,
            y_pred: tf.Tensor,
            basis_matrices: np.ndarray,
            n_states: int,
            lambda_1: float,
            lambda_2: float,
            lambda_3: float):
    """
    Custom loss function for SU(n) reconstruction.

    Parameters:
    - y_true: True target tensor (not directly used in this loss example).
    - y_pred: Predicted coefficients tensor (batch_size, time_steps * (n_states**2 - 1)).
    - basis_matrices: List of SU(n) basis matrices (n_states**2 elements, including identity).
    - n_states: Number of quantum states (dimension of the density matrix).
    - lambda_1: Regularization weight for trace normalization.
    - lambda_2: Regularization weight for Hermitian positivity.

    Returns:
    - Loss value (scalar).
    """

    print("Running SU(n) custom loss...")

    # Number of time steps
    num_elements = y_pred.shape[1]  # Total number of elements in the prediction vector
    n_steps = num_elements // (n_states**2 - 1)  # Number of time steps

    # Reshape y_pred to (batch_size, time_steps, n_states**2 - 1)
    pred_vector_reshaped = tf.reshape(y_pred, [tf.shape(y_pred)[0], n_steps, n_states**2 - 1])

    # Step 1: Append 1/n_states to each coefficient set for normalization
    # Shape after concat: (batch_size, n_steps, n_states**2)
    coefficients_with_norm = tf.concat(
        [pred_vector_reshaped, tf.fill([tf.shape(pred_vector_reshaped)[0], n_steps, 1], 1 / n_states)],
        axis=-1
    )

    # Step 2: Convert coefficients to complex numbers
    coefficients_with_norm = tf.cast(coefficients_with_norm, tf.complex64)


    # Step 3: Convert basis_matrices to tensor
    # Shape: (n_states**2, n_states, n_states)
    basis_matrices_tensor = tf.stack([tf.convert_to_tensor(B, dtype=tf.complex64) for B in basis_matrices], axis=0)

    # Expand coefficients for broadcasting: (batch_size, n_steps, n_states**2, 1, 1)
    coefficients_expanded = tf.expand_dims(tf.expand_dims(coefficients_with_norm, -1), -1)

    # Multiply and sum over the basis matrices: (batch_size, n_steps, n_states, n_states)
    reconstructed_rho_matrices = tf.reduce_sum(coefficients_expanded * basis_matrices_tensor, axis=-3)
    #
    hermitian_conj = tf.linalg.adjoint(reconstructed_rho_matrices)  # Hermitian check
    hermitian_diff = tf.reduce_mean(tf.square(tf.abs(reconstructed_rho_matrices - hermitian_conj)))

    eigenvalues = tf.math.real(tf.linalg.eigvalsh(reconstructed_rho_matrices))  # Positivity check
    positivity_loss = tf.reduce_mean(tf.square(tf.nn.relu(-eigenvalues)))

    # Additional eigenvalue regularization to ensure eigenvalues are in [0, 1]
    eigenvalue_clip = tf.clip_by_value(eigenvalues, 0.0, 1.0)
    eigenvalue_regularization = tf.reduce_mean(tf.square(eigenvalue_clip - eigenvalues))

    # Compute the Mean Squared Error loss (MSE)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))# MSE between true and predicted density matrices
    # Total loss: sum of all penalties and MSE
    total_loss = mse_loss + lambda_1*hermitian_diff + lambda_2*positivity_loss + lambda_3*eigenvalue_regularization

    return total_loss

