import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

models = ['MLP_model-274-tloss-1.556e-05-vloss-2.148e-05.keras',
 'LSTM_model-522-tloss-1.570e-05-vloss-2.129e-05.keras', 
 'CNN_model-233-tloss-1.595e-05-vloss-2.212e-05.keras', 
 'CNN_LSTM_model-284-tloss-1.544e-05-vloss-2.320e-05.keras']

nn_models = ['mlp', 'lstm', 'cnn', 'cnn_lstm']

# Directory where your models are saved
system = 'SB'
   
def OSTL(Xin: np.ndarray,  
        n_states: int, 
        time: float,  
        QDmodelIn: str,
        systemType: str,
        traj_output_file: str):

    #
    # check whether the model exists in the current directory
    #
    model = load_model(QDmodelIn, compile=False)
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
    n_time_steps = int(yhat.shape[1]/n_states**2)
    time_range = np.linspace(0, time, n_time_steps)
    y = np.zeros((1, n_states**2), dtype=float)
    y1 = np.zeros((n_time_steps, n_states, n_states), dtype=complex)

    a = 0; b = n_states**2;
    for i in range(0, n_time_steps):
        y[0,:] = yhat[0, a:b]
        a = a + n_states**2
        b = b + n_states**2

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
    np.savez('dyn_'+traj_output_file, time=time_range, rho=y1)
    print('ml_dyn.OSTL: Dynamics is saved in a file  "' + 'dyn_' + traj_output_file + '"')

n_states = 2
time = 20
time_step = 0.05
systemType = 'SB'


for (nn_model, model_path) in (nn_models, models):

    traj_output_file = nn_model + '_sb_sym_punn'
    Xin = np.array([0, 1, 9.0/10, 0.6, 1.0])
    OSTL(Xin, n_states,
        time,
        model_path,
        systemType,
        traj_output_file)

for (nn_model, model_path) in (nn_models, models):

    traj_output_file = nn_model + '_sb_asym_punn'
    Xin = np.array([1, 1, 9.0/10, 0.6, 1.0])
    OSTL(Xin, n_states,
        time,
        model_path,
        systemType,
        traj_output_file)

