import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable


def reconstruct_y_full(y, n_states):
    """
    Reconstruct the full flattened vector (y_full) from y (the prediction vector)
    by inserting the last diagonal element computed as 1 minus the sum of the given diagonal elements.

    Args:
        y: Tensor of shape (batch_size, n_states^2 - 1). For each density matrix sample,
           the ordering is:
             [rho_11,
              (rho_12_real, rho_12_imag, rho_13_real, rho_13_imag, ..., rho_1n_real, rho_1n_imag),
              rho_22,
              (rho_23_real, rho_23_imag, ..., rho_2n_real, rho_2n_imag),
              ...,
              rho_(n-1,n-1),
              (rho_(n-1,n)_real, rho_(n-1,n)_imag)]
           Note that the last diagonal element rho_nn is omitted.
        n_states: Dimension of the density matrix.

    Returns:
        y_full: Numpy array of size (1, n_states^2) containing the full vector with
                the last diagonal element inserted.
    """
    # List to collect the full vector entries.
    # We'll also collect the diagonal entries to later compute the last one.
    full_entries = []
    diag_entries = []  # collected diagonal elements for rows 0 ... n-2

    # Pointer into y.
    idx = 0
    # Loop over rows 0 to n_states-2 (last row diagonal is missing)
    for i in range(n_states - 1):
        # Extract diagonal element for row i.
        diag_val = y[:, idx]  # shape (batch_size,)
        full_entries.append(diag_val)
        diag_entries.append(diag_val)
        idx += 1

        # For each off-diagonal element in row i (columns i+1 to n_states-1)
        for j in range(i + 1, n_states):
            # Real and imaginary parts.
            real_part = y[:, idx]
            imag_part = y[:, idx + 1]
            full_entries.append(real_part)
            full_entries.append(imag_part)
            idx += 2

    # Compute the last diagonal element: 1 minus the sum of all previous diagonal elements.
    diag_last = 1.0 - tf.add_n(diag_entries)
    full_entries.append(diag_last)

    # Stack along axis=1 to create the full vector for each sample.
    y_full = np.stack(full_entries, axis=1)
    return y_full

def OSTL(Xin: np.ndarray,  
        n_states: int, 
        time: float,  
        QDmodelIn: str,
        systemType: str,
        traj_output_file: str,
        pinn: str):

    #
    # check whether the model exists in the current directory
    #
    
    model = tf.keras.models.load_model(QDmodelIn, compile=False)
    #Show the model architecture
    model.summary()
    #
    print('ml_dyn.OSTL: Running dynamics with OSTL approach ......')
    #
    #
    # normalizing the time feature using logistic function 
    #
    x_pred = Xin[:]
    x_pred = x_pred.reshape(1, Xin.shape[0],1) # reshape the input 
    yhat = model.predict(x_pred, verbose=0)
    n_time_steps = int((yhat.shape[1] + 401)/n_states**2) 
    time_range = np.linspace(0, time, n_time_steps)
    print(n_time_steps)
    y = np.zeros((1, n_states**2), dtype=float)
    y_tmp = np.zeros((1, n_states**2 - 1), dtype=float)
    y1 = np.zeros((n_time_steps, n_states, n_states), dtype=complex)

    a = 0; b = n_states**2 -1;

    for i in range(0, n_time_steps):
        y_tmp[0,:] = yhat[0, a:b]
        
        a = b  # Move to the next time step for next iteration
        b += n_states**2 - 1
        
        y_full = reconstruct_y_full(y_tmp, n_states)

        y = y_full
        
        print(y)

        # Reconstruct the density matrix for the current time step
        rho_matrix = np.zeros((n_states, n_states), dtype=complex)
    
        # Fill the diagonal elementS
        idx = 0
        for j in range(n_states):
            if j > 0:  #
                idx += 2 * (n_states - j) + 1

            rho_matrix[j, j] = y[0, idx]  

        # Fill the off-diagonal elements (real + imaginary)
        idx = 1  # Start after the diagonal elements
        for row in range(0, n_states-1):
            for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                real_part = y[0, idx]
                imag_part = y[0, idx + 1]
                rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                if col == n_states -1:
                    idx += 3        # skip diagonal terms
                else:
                    idx += 2  # Move to the next real/imag pair
        
        # Store the reconstructed matrix in the 3D array
        
        y1[i] = rho_matrix
   
    trace = np.trace(y1[:,:,:], axis1=1, axis2=2)
    np.savez('trace_'+traj_output_file, time=time_range, trace=np.real(trace))
    print('ml_dyn.OSTL: Trace is saved in a file  "' + "trace_"+traj_output_file + '"')
    eig, vec = np.linalg.eig(y1[:,:, :])
    np.savez('eig_'+traj_output_file, time=time_range, eig=np.real(eig))
    print('ml_dyn.coeff_OSTL: Eigen values is saved in a file  "' + "eig_"+traj_output_file + '"')   
    np.savez(traj_output_file, time=time_range, rho=y1)
    print('ml_dyn.OSTL: Dynamics is saved in a file  "' "dynamics_" + traj_output_file + '"')


n_states = 2
time = 20
pinn = 'False'
time_step = 0.05
QDmodelIn = 'cnn_lstm_model-507-tloss-5.030e-06-vloss-6.616e-06.keras'
systemType = 'SB'
traj_output_file = 'sym_punn'
Xin = np.array([0, 1, 9.0/10, 0.6, 1.0])

OSTL(Xin, n_states,
    time,
    QDmodelIn,
    systemType,
    traj_output_file,
    pinn)

traj_output_file = 'asym_punn'
Xin = np.array([1, 1, 9.0/10, 0.6, 1.0])

OSTL(Xin, n_states,
    time,
    QDmodelIn,
    systemType,
    traj_output_file,
    pinn)
