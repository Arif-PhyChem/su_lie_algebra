import numpy as np
import hyperopt_optim
import ml_models



n_states = 2

methods = ['punn', 'pinn', 'su2-punn', 'su2-pinn']
constraints = ['No', 'Yes', 'No', 'Yes']

def training_step(x_file, y_file, n_states, method, constraint):
    
    x = np.load(x_file)
    y = np.load(y_file)
  
    ml_models.CNN_LSTM_train(x, y, n_states, 3000, 500, method, constraint)
    

for (method, constraint) in (methods, constraints):

    if method == 'punn' or method == 'pinn':
        x_file = 'x_sb_punn-pinn.npy'
        y_file = 'y_sb_punn-pinn.npy'

        training_step(x_file, y_file, n_states, method,  constraint)


    if method == 'su2-punn' or  method == 'su2-pinn':
        x_file = 'x_sb_su2-punn-pinn.npy'
        y_file = 'y_sb_su2-punn-pinn.npy'

        training_step(x_file, y_file, n_states, method, constraint)
