import numpy as np
import hyperopt_optim
import ml_models



n_states = 2
x_file = 'x_punn-pinn.npy'
y_file = 'y_punn-pinn.npy'


nn_models = ['mlp', 'lstm', 'cnn', 'cnn_lstm']

def training_step(x_file, y_file, nn_model):
    
    x = np.load(x_file)
    y = np.load(y_file)
    if nn_model == 'cnn_lstm':
        ml_models.CNN_LSTM_train(x, y, 3000, 500)
    if nn_model == 'lstm':
        ml_models.LSTM_train(x, y, 2000, 500)
    if nn_model == 'cnn':
        ml_models.CNN_train(x, y, 3000, 500)
    if nn_model == 'mlp':
        ml_models.MLP_train(x, y, 2000, 500)
 
for nn_model in nn_models:

    training_step(x_file, y_file, nn_model)
