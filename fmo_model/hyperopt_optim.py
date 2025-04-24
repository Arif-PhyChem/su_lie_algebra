import os
import tensorflow as tf
import keras as keras
import pickle
import numpy as np
import su_basis
import custom_loss as ml_loss
from keras.layers import Dense, BatchNormalization
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import MaxPooling1D
from keras.layers import Conv1D, LSTM
from hyperopt import fmin, hp, Trials, STATUS_OK, tpe
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable

def optimize(x: np.ndarray, 
            y: np.ndarray,
            n_states: int,
            ml_model: str,
            epochs: int, 
            max_evals: int,
            pinn: str
            ):
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)
    
    if ml_model == 'cnn_lstm':
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
        x_val  = x_val.reshape(x_val.shape[0], x_val.shape[1],1)

    ######################################################
    print('hyperopt_optim: Optimizing the Neural Network with hyperopt library')
    print('hyperopt_optim: Setting Optimizer to Adam and loss to mean square error (mse)')
    print('hyperopt_optim: We do not optimize the activation function and set it equal to Relu')  
    print('hyperopt_optim: Maximum number of evaluations =', max_evals)
    print('hyperopt_optim: Each evaluation runs for ' + str(epochs) + ' epochs')
    print('=================================================================')
    #####################################################

    if pinn == 'coeff':
        basis_matrices = su_basis.generate_su_basis(n_states)
        # Identity matrix (part of the basis)
        identity_matrix = np.eye(n_states, dtype=complex)
        basis_matrices.append(identity_matrix)

    @register_keras_serializable(package='ml_models')
    def custom_loss(y_true, y_pred):
        if pinn == 'True':
            #print('Running with PINN custom loss: mse + other losses')
            #mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            #trace_penalty = 0.0
            #diagonal_idx = [i * (n_states * 2 - i) for i in range(n_states)]
            #for kk in range(0, y_pred.shape[-1]//n_states**2):
            #    trace_t = 0.0
            #    # calculate trace for each time step
            #    for idx in diagonal_idx: 
            #        trace_t += y_pred[:, kk * n_states**2 + idx]
            #    trace_penalty += tf.reduce_mean(tf.square(1- trace_t))
            #trace_penalty /= y_pred.shape[-1]//n_states**2
            
            #tot_loss = 2.0*mse_loss + 1.0*trace_penalty
            loss = ml_loss.pinn_loss(y_true, y_pred, n_states, lambda_1=0.5, lambda_2=0.5, lambda_3=0.5)
        elif pinn == 'coeff':
            #print('Running with SU custom loss: mse + other losses (except trace loss)')
            loss = ml_loss.su_loss(y_true, y_pred, basis_matrices, n_states, lambda_1=0.5, lambda_2=0.5)
        else:
            print('Running with MSE loss')
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return loss

    if ml_model == 'cnn_lstm':
        kernel_choice = x_train.shape[1] - 1
        space = {'Conv1D': hp.choice('Conv1D',  np.arange(16, 256, 16)),
            'Conv1D_1': hp.choice('Conv1D_1',   np.arange(16, 256, 16)),
            'Conv1D_2': hp.choice('Conv1D_2',   np.arange(16, 256, 16)),
            'lstm_units': hp.choice('lstm_units', np.arange(16, 512, 32)),
            'lstm_units_1': hp.choice('lstm_units_1', np.arange(16, 512, 32)),
            'lstm_units_2': hp.choice('lstm_units_2', np.arange(16, 512, 32)),
            'Dense': hp.choice('Dense', np.arange(16, 512, 32)),
            'Dense_1': hp.choice('Dense_1', np.arange(16, 512, 32)),
            'Dense_2': hp.choice('Dense_2', np.arange(16, 512, 32)),
            'if_cnn': hp.choice('if_cnn', [{'layers': 'two', }, {'layers': 'three'}]), 
            'if_dense': hp.choice('if_dense', [{'layers': 'two', }, {'layers': 'three'}]),
            'if_lstm': hp.choice('if_lstm', [{'layers': 'one', }, {'layers': 'two', }, {'layers': 'three'}]),
            'kernel_size': hp.uniform('kernel_size', 1, kernel_choice),
            'kernel_size_1': hp.uniform('kernel_size_1', 1, kernel_choice),
            'kernel_size_2': hp.uniform('kernel_size_2', 1, kernel_choice),
            'learning_rate': hp.choice('learning_rate', [10**-5, 10**-4, 10**-3]),
            'batch_size': hp.choice('batch_size', np.arange(16, 64, 16)),
            'activation': 'relu'
        } 

    if ml_model == 'lstm':
        space = {'Dense': hp.choice('Dense', [8, 16,32,64,128,256,512]),
            'Dense_1': hp.choice('Dense_1', [8, 16,32,64,128,256,512]),
            'Dense_2': hp.choice('Dense_2', [8, 16,32,64,128,256,512]),
            'lstm_units': hp.choice('lstm_units', np.arange(16, 512, 32)),
            'lstm_units_1': hp.choice('lstm_units_1', np.arange(16, 512, 32)),
            'lstm_units_2': hp.choice('lstm_units_2', np.arange(16, 512, 32)),
            'if_1': hp.choice('if_1', [{'layers': 'two', }, {'layers': 'three'}]),
            'if_lstm': hp.choice('if_lstm', [{'layers': 'one'}, {'layers': 'two'}, {'layers': 'three'}]),
            'learning_rate': hp.choice('learning_rate', [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]),
            'batch_size': hp.choice('batch_size', [32, 64,128,256,512]),
            'activation': 'relu'
        } 

    def optimize_model(params):

        model = Sequential()
        if ml_model == 'cnn_lstm':
            model.add(Conv1D(params['Conv1D'], kernel_size=int(params['kernel_size']), input_shape=(x_train.shape[1],1)))
            model.add(Activation(params['activation']))
            model.add(BatchNormalization())
            model.add(Conv1D(params['Conv1D_1'], kernel_size=int(params['kernel_size_1']), padding='same'))
            model.add(Activation(params['activation']))
            model.add(BatchNormalization())
            if params['if_cnn']['layers'] == 'three':
                model.add(Conv1D(params['Conv1D_2'], kernel_size=int(params['kernel_size_2']), padding='same'))
                model.add(Activation(params['activation']))
                model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            if params['if_lstm']['layers'] == 'one':
                model.add(LSTM(int(params['lstm_units']), return_sequences=False))

            if params['if_lstm']['layers']  == 'two':
                model.add(LSTM(int(params['lstm_units']), return_sequences=True))
                model.add(BatchNormalization())
                model.add(LSTM(int(params['lstm_units_1']), return_sequences=False))

            if params['if_lstm']['layers']  == 'three':
                model.add(LSTM(int(params['lstm_units']), return_sequences=True))
                model.add(BatchNormalization())
                model.add(LSTM(int(params['lstm_units_1']), return_sequences=True))
                model.add(BatchNormalization())
                model.add(LSTM(int(params['lstm_units_2']), return_sequences=False))
                
            model.add(BatchNormalization())

        if ml_model == 'lstm':
            if params['if_lstm']['layers'] == 'one':
                model.add(LSTM(int(params['lstm_units']), return_sequences=False, input_shape=(x_train.shape[1],1)))

            if params['if_lstm']['layers']  == 'two':
                model.add(LSTM(int(params['lstm_units']), return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(LSTM(int(params['lstm_units_1']), return_sequences=False))

            if params['if_lstm']['layers']  == 'three':
                model.add(LSTM(int(params['lstm_units']), return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(LSTM(int(params['lstm_units_1']), return_sequences=True))
                model.add(LSTM(int(params['lstm_units_2']), return_sequences=False))
        
        model.add(Dense(params['Dense']))
        model.add(Activation(params['activation']))
        model.add(BatchNormalization())
        model.add(Dense(params['Dense_1']))
        model.add(Activation(params['activation']))
        model.add(BatchNormalization())
        if params['if_dense']['layers'] == 'three':
            model.add(Dense(params['Dense_2']))
            model.add(Activation(params['activation']))
            model.add(BatchNormalization())
        model.add(Dense(y_train.shape[1], activation='linear'))

        adam = keras.optimizers.Adam(learning_rate=params['learning_rate'])

        
        # Create dataset pipelines for training and validation
        Batch_size = int(params['batch_size'])
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (train_dataset
                 .shuffle(buffer_size=10000)
                 .batch(Batch_size)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(Batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        model.compile(loss=custom_loss, optimizer=adam)
        
        print(model.summary())
        
        model.fit(train_dataset, 
                validation_data=val_dataset,
                epochs=epochs,
                verbose=2)
        
        loss= model.evaluate(val_dataset, verbose=0)
        return {'loss': loss, 'status': STATUS_OK, 'model': model}

    #if __name__ == '__main__':
    trials = Trials()
    best_run = fmin(optimize_model,
                        space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials)
    if ml_model == 'cnn_lstm':
        f = open("best_cnn_lstm_params.pkl", "wb")
    if ml_model == 'lstm':
        f = open("best_lstm_params.pkl", "wb")
    pickle.dump(best_run, f)
    f.close()
