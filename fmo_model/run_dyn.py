import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import su_basis
from tensorflow.keras.saving import register_keras_serializable
import custom_loss as ml_loss

models = ['PUNN_model-91-tloss-1.095e-05-vloss-1.245e-05.keras',
 'PINN_model-1063-tloss-1.077e-05-vloss-1.292e-05.keras', 
 'su7-PINN_model-142-tloss-1.059e-05-vloss-1.944e-05.keras',
 'su7-PUNN_model-157-tloss-1.008e-05-vloss-2.274e-05.keras']

# Directory where your models are saved
ml_model = 'cnn_lstm'
system = 'FMO'


@register_keras_serializable(package='run_dyn')
def custom_loss(y_true, y_pred):
    if pinn == 'True':
        loss = ml_loss.pinn_loss(y_true, y_pred, n_states, lambda_1=0.5, lambda_2=0.5, lambda_3=0.5)
    if pinn == 'coeff':
       print('Running with SU custom loss: mse + other losses (except trace loss)')
       basis_matrices = su_basis.generate_su_basis(n_states)
       # Identity matrix (part of the basis)
       identity_matrix = np.eye(n_states, dtype=complex)
       basis_matrices.append(identity_matrix)
       loss = ml_loss.su_loss(y_true, y_pred, basis_matrices, n_states, lambda_1=0.5, lambda_2=0.5)
    else:
        tot_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return tot_loss
   
def OSTL_coeff(Xin: np.ndarray,  
        n_states: int, 
        time: float,  
        time_step: float, 
        QDmodelIn: str,
        systemType: str,
        traj_output_file: str):
   
    path_to_model = QDmodelIn
    #
    # check whether the model exists in the current directory
    #

    model = tf.keras.models.load_model(QDmodelIn, custom_objects={'custom_loss': custom_loss})

    #Show the model architecture
    model.summary()
    #
    print('ml_dyn.OSTL: Running dynamics with OSTL approach ......')
    #
    
    basis_matrices = su_basis.generate_su_basis(n_states)
    
    # Identity matrix (part of the basis)
    identity_matrix = np.eye(n_states, dtype=complex)
    basis_matrices.append(identity_matrix)
    time_range = 0
    tt = time_range
    for i in range(0, int(time/time_step)-1):
        tt += time_step
        time_range = np.append(time_range, tt)
    nsteps = len(time_range)
    y = np.zeros((nsteps, n_states, n_states), dtype=complex)
    #
    # normalizing the time feature using logistic function 
    #
    x_pred = Xin[:]
    x_pred = x_pred.reshape(1, Xin.shape[0],1) # reshape the input 
    yhat = model.predict(x_pred, verbose=0)
    a = 0; b = n_states**2-1;
    for i in range(0, nsteps):
        coefficients_i= yhat[0, a:b]
        coefficients_i = np.append(coefficients_i, 1/n_states)
        # Reconstruct the matrix using the coefficients
        reconstructed_rho = sum(a * B for a, B in zip(coefficients_i, basis_matrices))
        y[i, :, :] = reconstructed_rho
        a = a + n_states**2-1
        b = b + n_states**2-1
    
    trace = np.trace(y[:,:, :], axis1=1, axis2=2)
    np.savez('trace_'+traj_output_file, time=time_range, trace=np.real(trace))
    print('ml_dyn.coeff_OSTL: Trace is saved in a file  "' + "trace_"+traj_output_file + '"')   
    eig, vec = np.linalg.eig(y[:,:, :])
    np.savez('eig_'+traj_output_file, time=time_range, eig=np.real(eig))
    print('ml_dyn.coeff_OSTL: Eigen values is saved in a file  "' + "eig_"+traj_output_file + '"')   
    np.savez(traj_output_file, time=time_range, rho=y)
    print('ml_dyn.coeff_OSTL: Dynamics is saved in a file  "' + traj_output_file + '"')

def OSTL(Xin: np.ndarray,  
        n_states: int, 
        time: float,  
        QDmodelIn: str,
        systemType: str,
        traj_output_file: str,
        pinn: str):

    path_to_model = QDmodelIn
    #
    # check whether the model exists in the current directory
    #
    
    model = tf.keras.models.load_model(QDmodelIn, custom_objects={'custom_loss': custom_loss})
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
    np.savez(traj_output_file, time=time_range, rho=y1)
    print('ml_dyn.OSTL: Dynamics is saved in a file  "' + traj_output_file + '"')

for idx, model_path in enumerate(models):
    if idx == 0:
        method = 'punn'
    elif idx == 1:
        method = 'pinn'
    else:
        method = 'su2-punn-pinn'
    
    if method == 'su2-punn-pinn':
        if system == 'FMO':
            n_states = 7
            time = 1
            time_step = 0.005
            QDmodelIn = model_path
            systemType = 'FMO'
            if idx==2:
                traj_output_file = 'dynamics_fmo_7_1_su7-PINN'
            if idx==3:
                traj_output_file = 'dynamics_fmo_7_1_su7-PUNN'

            Xin = np.array([0.1, 400/500, 40/520, 90/510]) # normalized excitation index, gamma, lambda, T
            OSTL_coeff(Xin, n_states,
                    time,
                    time_step,
                    QDmodelIn,
                    systemType,
                    traj_output_file)
            if idx==2: 
                traj_output_file = 'dynamics_fmo_7_6_su7-PINN'
            if idx==3:
                traj_output_file = 'dynamics_fmo_7_6_su7-PUNN'

            Xin = np.array([0.6, 400/500, 40/520, 90/510]) 
            OSTL_coeff(Xin, n_states,
                    time,
                    time_step,
                    QDmodelIn,
                    systemType,
                    traj_output_file)
    
    if method == 'pinn':
    
        if system == 'FMO':
            n_states = 7
            time = 1 
            pinn = 'True'
            QDmodelIn = model_path
            systemType = 'FMO'
            traj_output_file = 'dynamics_fmo_7_1_PINN'
            Xin = np.array([0.1, 400/500, 40/520, 90/510])
    
            OSTL(Xin, n_states,
                time,
                QDmodelIn,
                systemType,
                traj_output_file,
                pinn)
    
            traj_output_file = 'dynamics_fmo_7_6_PINN'
            Xin = np.array([0.6, 400/500, 40/520, 90/510])
    
            OSTL(Xin, n_states,
                time,
                QDmodelIn,
                systemType,
                traj_output_file,
                pinn)
    
    if method == 'punn':
    
        if system == 'FMO':
            n_states = 7
            time = 1 
            pinn = 'False'
            QDmodelIn = model_path
            systemType = 'FMO'
            traj_output_file = 'dynamics_fmo_7_1_PUNN'
            Xin = np.array([0.1, 400/500, 40/520, 90/510])
    
            OSTL(Xin, n_states,
                time,
                QDmodelIn,
                systemType,
                traj_output_file,
                pinn)
    
            traj_output_file = 'dynamics_fmo_7_6_PUNN'
            Xin = np.array([0.6, 400/500, 40/520, 90/510])
    
            OSTL(Xin, n_states,
                time,
                QDmodelIn,
                systemType,
                traj_output_file,
                pinn)
