import numpy as np
import hyperopt_optim
import ml_models



ml_model = 'cnn_lstm'
n_states = 2
method = 'normal'

def training_step():
    
    # Reshape for LSTM: [samples, timesteps, features]
    # Assuming each sample corresponds to one timestep
    #hyperopt_optim.optimize(x, y, n_states, ml_model, 100, 30, pinn)
    if ml_model == 'cnn_lstm':
        ml_models.CNN_LSTM_train(x, y, n_states, 3000, 500, pinn)
    if ml_model == 'lstm':
        ml_models.LSTM_train(x, y, 2000, 500)
    

if method == 'coeff':
    pinn = method
    x = np.load('x_sb_coeff.npy')
    y = np.load('y_sb_coeff.npy')

    training_step()

if method == 'pinn':
    pinn = 'True'
    x = np.load('x_sb_normal_pinn.npy')
    y = np.load('y_sb_normal_pinn.npy')

    training_step()

if method == 'normal':
    pinn = 'False'
    x = np.load('x_sb_normal_pinn.npy')
    y = np.load('y_sb_normal_pinn.npy')

    training_step()
