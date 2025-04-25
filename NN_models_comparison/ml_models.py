import os
import csv
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D, LSTM
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable

# Custom callback to save training and validation loss
class LossHistoryCallback(keras.callbacks.Callback):
    def __init__(self, log_file="loss_history.csv"):
        super().__init__()
        self.log_file = log_file
        self.epoch_loss = []
        self.epoch_val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        self.epoch_loss.append(train_loss)
        self.epoch_val_loss.append(val_loss)

        # Save the losses to a CSV file
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if epoch == 0:  # Add headers only for the first epoch
                writer.writerow(["Epoch", "Training Loss", "Validation Loss"])
            writer.writerow([epoch + 1, train_loss, val_loss])

# Instantiate the loss history callback
loss_history_callback = LossHistoryCallback(log_file="loss_history.csv")

def CNN_train(x: np.ndarray, 
            y: np.ndarray,
            epochs: int,
            patience: int
            ):
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)
     
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
    x_val  = x_val.reshape(x_val.shape[0], x_val.shape[1],1)

    optim_param_file = "best_cnn_params.pkl"
    
    print('=================================================================')
    print('ml_mode.cnn_lstm: Looking for', optim_param_file)
    model = Sequential()

 
    if os.path.isfile(optim_param_file):
        print('ml_models.cnn: loading hyperparameters from', optim_param_file)
        print('=================================================================')
        #f = open(optim_param_file, 'rb')
        #hyper_param = pickle.load(f)
        #print('hyperparameters = ', hyper_param)
        #f.close()
        #filter_choice = np.arange(16, 256, 16)
        #dense_choice = np.arange(16, 512, 32)
        #lr_choice = [10**-5, 10**-4, 10**-3]
        #batch_size = np.arange(32, 512, 32)
        #lstm_choice = np.arange(16, 512, 32)

        #Conv1D_0 = filter_choice[hyper_param['Conv1D']] 
        #Conv1D_1 = filter_choice[hyper_param['Conv1D_1']]
        #Conv1D_2 = filter_choice[hyper_param['Conv1D_2']]
        #Dense_0 = dense_choice[hyper_param['Dense']]
        #Dense_1 = dense_choice[hyper_param['Dense_1']]
        #Dense_2 = dense_choice[hyper_param['Dense_2']]
        #Batch_size = batch_size[hyper_param['batch_size']] 
        #lstm_units_0 = int(lstm_choice[hyper_param['lstm_units']])
        #If_cnn = hyper_param['if_cnn']
        #If_dense = hyper_param['if_dense']
        #If_lstm = hyper_param['if_lstm']
        #Kernel_0 = int(hyper_param['kernel_size'])
        #Kernel_1 = int(hyper_param['kernel_size_1'])
        #Kernel_2 = int(hyper_param['kernel_size_2'])
        #Lr_rate = lr_choice[hyper_param['learning_rate']]

        #print('=================================================================')
        #print('ml_models.cnn: Running wth EarlyStopping of patience =', patience)
        #print('ml_models.cnn: Running with batch size =', Batch_size , 'and epochs =', epochs)
        #print('=================================================================')

        #model.add(Conv1D(Conv1D_0, kernel_size=Kernel_0, activation ='relu', input_shape=(x_train.shape[1],1)))
        #model.add(BatchNormalization())
        #model.add(Conv1D(Conv1D_1, kernel_size=Kernel_1, activation = 'relu', padding='same'))
        #model.add(BatchNormalization())
        #if If_cnn == 1:
        #    model.add(Conv1D(Conv1D_2, kernel_size=Kernel_2, activation = 'relu', padding='same'))
        #    model.add(BatchNormalization())

        #model.add(MaxPooling1D(pool_size=2))
        #model.add(Flatten())
        #model.add(BatchNormalization())
        #model.add(Dense(Dense_0, activation = 'relu'))
        #model.add(BatchNormalization())
        #model.add(Dense(Dense_1, activation = 'relu'))
        #model.add(BatchNormalization())
        #if If_dense == 1: 
        #    model.add(Dense(Dense_2, activation = 'relu'))
        #    model.add(BatchNormalization())
        #model.add(Dense(y_train.shape[1], activation='linear'))
        #adam = keras.optimizers.Adam(learning_rate=Lr_rate)
    else:
        print('=================================================================')
        print('ml_models.cnn_lstm: '+ str(optim_param_file) +  ' not found, thus training CNN model with the default structure')
        print('=================================================================')
        print('ml_models.cnn_lstm: Running wth EarlyStopping of patience =', patience)
        print('ml_models.cnn_lstm: Running with batch size = 64 and epochs =', epochs)
        print('=================================================================')
        model.add(Conv1D(80, kernel_size=3, activation ='relu', input_shape=(x_train.shape[1],1)))
        model.add(Conv1D(110, kernel_size=3, activation = 'relu', padding='same'))
        model.add(Conv1D(80, kernel_size=3, activation = 'relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))
        adam = keras.optimizers.Adam(learning_rate=10**-3)


    # Create dataset pipelines for training and validation
    if os.path.isfile(optim_param_file):
        Batch_size = Batch_size
    else:
        Batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset
                 .shuffle(buffer_size=10000)
                 .batch(Batch_size)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(Batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
   
    model.compile(loss='mse', optimizer=adam)

    print(model.summary())
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
  
    models_dir =  "cnn_trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Directory",  models_dir, "is created sucessfully where the trained models will be saved")
    else:
        print("Directory",  models_dir, "already exists where the trained models will be saved")

    filepath=models_dir+"/cnn_model-{epoch:02d}-tloss-{loss:.3e}-vloss-{val_loss:.3e}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, loss_history_callback, es]

    if os.path.isfile(optim_param_file):
        model.fit(train_dataset,  
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks_list) 
    else:
        model.fit(train_dataset,  
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks_list) 


def CNN_LSTM_train(x: np.ndarray, 
            y: np.ndarray,
            epochs: int,
            patience: int
            ):
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)
     
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
    x_val  = x_val.reshape(x_val.shape[0], x_val.shape[1],1)

    optim_param_file = "best_cnn_lstm_params.pkl"
    
    print('=================================================================')
    print('ml_mode.cnn_lstm: Looking for', optim_param_file)
    model = Sequential()
 
    if os.path.isfile(optim_param_file):
        print('ml_models.cnn: loading hyperparameters from', optim_param_file)
        print('=================================================================')
        f = open(optim_param_file, 'rb')
        hyper_param = pickle.load(f)
        print('hyperparameters = ', hyper_param)
        f.close()
        filter_choice = np.arange(16, 256, 16)
        dense_choice = np.arange(16, 512, 32)
        lr_choice = [10**-5, 10**-4, 10**-3]
        batch_size = np.arange(32, 512, 32)
        lstm_choice = np.arange(16, 512, 32)

        Conv1D_0 = filter_choice[hyper_param['Conv1D']] 
        Conv1D_1 = filter_choice[hyper_param['Conv1D_1']]
        Conv1D_2 = filter_choice[hyper_param['Conv1D_2']]
        Dense_0 = dense_choice[hyper_param['Dense']]
        Dense_1 = dense_choice[hyper_param['Dense_1']]
        Dense_2 = dense_choice[hyper_param['Dense_2']]
        Batch_size = batch_size[hyper_param['batch_size']] 
        lstm_units_0 = int(lstm_choice[hyper_param['lstm_units']])
        If_cnn = hyper_param['if_cnn']
        If_dense = hyper_param['if_dense']
        If_lstm = hyper_param['if_lstm']
        Kernel_0 = int(hyper_param['kernel_size'])
        Kernel_1 = int(hyper_param['kernel_size_1'])
        Kernel_2 = int(hyper_param['kernel_size_2'])
        Lr_rate = lr_choice[hyper_param['learning_rate']]

        print('=================================================================')
        print('ml_models.cnn: Running wth EarlyStopping of patience =', patience)
        print('ml_models.cnn: Running with batch size =', Batch_size , 'and epochs =', epochs)
        print('=================================================================')

        model.add(Conv1D(Conv1D_0, kernel_size=Kernel_0, activation ='relu', input_shape=(x_train.shape[1],1)))
        model.add(BatchNormalization())
        model.add(Conv1D(Conv1D_1, kernel_size=Kernel_1, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        if If_cnn == 1:
            model.add(Conv1D(Conv1D_2, kernel_size=Kernel_2, activation = 'relu', padding='same'))
            model.add(BatchNormalization())

        model.add(MaxPooling1D(pool_size=2))
        
        if If_lstm == 0:
            model.add(LSTM(lstm_units_0, return_sequences=False))
        if If_lstm == 1:
            lstm_units_1 = int(lstm_choice[hyper_param['lstm_units_1']])
            model.add(LSTM(lstm_units_0, return_sequences=True))
            model.add(BatchNormalization())
            model.add(LSTM(lstm_units_1, return_sequences=False))
        if If_lstm == 2:
            lstm_units_1 = int(lstm_choice[hyper_param['lstm_units_1']])
            lstm_units_2 = int(lstm_choice[hyper_param['lstm_units_2']])
            model.add(LSTM(lstm_units_0, return_sequences=True))
            model.add(BatchNormalization())
            model.add(LSTM(lstm_units_1, return_sequences=True))
            model.add(BatchNormalization())
            model.add(LSTM(lstm_units_2, return_sequences=False))
        
        model.add(BatchNormalization())
        model.add(Dense(Dense_0, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dense(Dense_1, activation = 'relu'))
        model.add(BatchNormalization())
        if If_dense == 1: 
            model.add(Dense(Dense_2, activation = 'relu'))
            model.add(BatchNormalization())
        model.add(Dense(y_train.shape[1], activation='linear'))
        adam = keras.optimizers.Adam(learning_rate=Lr_rate)
    else:
        print('=================================================================')
        print('ml_models.cnn_lstm: '+ str(optim_param_file) +  ' not found, thus training CNN model with the default structure')
        print('=================================================================')
        print('ml_models.cnn_lstm: Running wth EarlyStopping of patience =', patience)
        print('ml_models.cnn_lstm: Running with batch size = 64 and epochs =', epochs)
        print('=================================================================')
        model.add(Conv1D(80, kernel_size=3, activation ='relu', input_shape=(x_train.shape[1],1)))
        model.add(Conv1D(110, kernel_size=3, activation = 'relu', padding='same'))
        model.add(Conv1D(80, kernel_size=3, activation = 'relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(112,  return_sequences=False))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))
        adam = keras.optimizers.Adam(learning_rate=10**-3)


    # Create dataset pipelines for training and validation
    if os.path.isfile(optim_param_file):
        Batch_size = Batch_size
    else:
        Batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset
                 .shuffle(buffer_size=10000)
                 .batch(Batch_size)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(Batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
   
    model.compile(loss='mse', optimizer=adam)

    print(model.summary())
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
  
    models_dir =  "cnn_lstm_trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Directory",  models_dir, "is created sucessfully where the trained models will be saved")
    else:
        print("Directory",  models_dir, "already exists where the trained models will be saved")

    filepath=models_dir+"/cnn_lstm_model-{epoch:02d}-tloss-{loss:.3e}-vloss-{val_loss:.3e}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, loss_history_callback, es]

    if os.path.isfile(optim_param_file):
        model.fit(train_dataset,  
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks_list) 
    else:
        model.fit(train_dataset,  
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks_list) 

def LSTM_train(x: np.ndarray, 
            y: np.ndarray,
            epochs: int,
            patience: int):
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)

    print('=================================================================')
    print('ml_models.lstm: Running wth EarlyStopping of patience =', patience)
    print('ml_models.lstm: Running with batch size = 64 and epochs =', epochs)
    print('=================================================================')

    
    optim_param_file = "best_lstm_params.pkl"
    
    print('=================================================================')
    print('ml_model.lstm: Looking for',optim_param_file)
    
    
    model = Sequential()
    
    if os.path.isfile(optim_param_file):
        print('ml_models.lstm: loading hyperparameters from', optim_param_file)
        print('=================================================================')
        f = open('best_lstm_params.pkl', 'rb')
        hyper_param = pickle.load(f)
        f.close()
        dense_choice = [8,16,32,64,128,256,512]
        lr_choice = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
        batch_size = [8,16,32,64,128,256,512]
        lstm_choice = np.arange(16, 512, 32)

        lstm_units_0 = int(lstm_choice[hyper_param['lstm_units']])
        Dense_0 = dense_choice[hyper_param['Dense']]
        Dense_1 = dense_choice[hyper_param['Dense_1']]
        Batch_size = batch_size[hyper_param['batch_size']] 
        If_lstm = hyper_param['if_lstm']
        If_1 = hyper_param['if_1']
        Lr_rate = lr_choice[hyper_param['learning_rate']]

        print('=================================================================')
        print('ml_models.lstm: Running wth EarlyStopping of patience =', patience)
        print('ml_models.lstm: Running with batch size =', Batch_size , 'and epochs =', epochs)
        print('=================================================================')


        if If_lstm == 0:
            model.add(LSTM(lstm_units_0, return_sequences=False, input_shape=(x_train.shape[1],1)))

        if If_lstm == 1:
            lstm_units_1 = int(lstm_choice[hyper_param['lstm_units_1']])
            model.add(LSTM(lstm_units_0, return_sequences=True, input_shape=(x_train.shape[1],1)))
            model.add(LSTM(lstm_units_1, return_sequences=False))

        if If_lstm == 2:
            lstm_units_1 = int(lstm_choice[hyper_param['lstm_units_1']])
            lstm_units_2 = int(lstm_choice[hyper_param['lstm_units_2']])
            model.add(LSTM(lstm_units_0, return_sequences=True, input_shape=(x_train.shape[1],1)))
            model.add(LSTM(lstm_units_1, return_sequences=True))
            model.add(LSTM(lstm_units_2, return_sequences=False))
        
        model.add(Dense(Dense_0, activation = 'relu'))
        model.add(Dense(Dense_1, activation = 'relu'))
        
        if If_1 == 1: 
            Dense_2 = dense_choice[hyper_param['Dense_2']]
            model.add(Dense(Dense_2, activation = 'relu'))
        
        model.add(Dense(y_train.shape[1], activation='linear'))
       
        adam = keras.optimizers.Adam(learning_rate=Lr_rate)
    
    else:
        print('=================================================================')
        print('ml_models.lstm: '+ str(optim_param_file) +  ' not found, thus training LSTM model with the default structure')
        print('=================================================================')
        print('ml_models.lstm: Running wth EarlyStopping of patience =', patience)
        print('ml_models.lstm: Running with batch size = 64 and epochs =', epochs)
        print('=================================================================')

        model.add(LSTM(112,  return_sequences=False, input_shape=(x_train.shape[1],1)))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))
        adam = keras.optimizers.Adam(learning_rate=10**-3)
    
    
    # Create dataset pipelines for training and validation
    if os.path.isfile(optim_param_file):
        Batch_size = Batch_size
    else:
        Batch_size = 64

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset
                 .shuffle(buffer_size=10000)
                 .batch(Batch_size)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(Batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    print(model.summary())
   
    model.compile(loss='mse', optimizer=adam)

    
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
    
    models_dir =  "lstm_trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Directory",  models_dir, "is created sucessfully where the trained models will be saved")
    else:
        print("Directory",  models_dir, "already exists where the trained models will be saved")

    filepath=models_dir+"/lstm_model-{epoch:02d}-tloss-{loss:.3e}-vloss-{val_loss:.3e}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, loss_history_callback, es]
  
    if os.path.isfile(optim_param_file):
        model.fit(train_dataset,  
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=[callbacks_list, es]) 
    else:
        model.fit(train_dataset,  
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=[callbacks_list, es])


def MLP_train(x: np.ndarray, 
            y: np.ndarray,
            epochs: int,
            patience: int):
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)

    print('=================================================================')
    print('ml_models.lstm: Running wth EarlyStopping of patience =', patience)
    print('ml_models.lstm: Running with batch size = 64 and epochs =', epochs)
    print('=================================================================')

    
    optim_param_file = "best_mlp_params.pkl"
    
    print('=================================================================')
    print('ml_model.lstm: Looking for',optim_param_file)
    
    
    model = Sequential()
    
    if os.path.isfile(optim_param_file):
        print('ml_models.lstm: loading hyperparameters from', optim_param_file)
        print('=================================================================')
        #f = open('best_lstm_params.pkl', 'rb')
        #hyper_param = pickle.load(f)
        #f.close()
        #dense_choice = [8,16,32,64,128,256,512]
        #lr_choice = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
        #batch_size = [8,16,32,64,128,256,512]
        #lstm_choice = np.arange(16, 512, 32)

        #lstm_units_0 = int(lstm_choice[hyper_param['lstm_units']])
        #Dense_0 = dense_choice[hyper_param['Dense']]
        #Dense_1 = dense_choice[hyper_param['Dense_1']]
        #Batch_size = batch_size[hyper_param['batch_size']] 
        #If_lstm = hyper_param['if_lstm']
        #If_1 = hyper_param['if_1']
        #Lr_rate = lr_choice[hyper_param['learning_rate']]

        #print('=================================================================')
        #print('ml_models.lstm: Running wth EarlyStopping of patience =', patience)
        #print('ml_models.lstm: Running with batch size =', Batch_size , 'and epochs =', epochs)
        #print('=================================================================')


        #model.add(Dense(Dense_0, activation = 'relu'))
        #model.add(Dense(Dense_1, activation = 'relu'))
        #
        #if If_1 == 1: 
        #    Dense_2 = dense_choice[hyper_param['Dense_2']]
        #    model.add(Dense(Dense_2, activation = 'relu'))
        #
        #model.add(Dense(y_train.shape[1], activation='linear'))
       
        #adam = keras.optimizers.Adam(learning_rate=Lr_rate)
    
    else:
        print('=================================================================')
        print('ml_models.mlp: '+ str(optim_param_file) +  ' not found, thus training MLP model with the default structure')
        print('=================================================================')
        print('ml_models.mlp: Running wth EarlyStopping of patience =', patience)
        print('ml_models.mlp: Running with batch size = 64 and epochs =', epochs)
        print('=================================================================')

        model.add(Dense(256, activation = 'relu', input_shape=(x_train.shape[1], )))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))
        adam = keras.optimizers.Adam(learning_rate=10**-3)
    
    
    # Create dataset pipelines for training and validation
    if os.path.isfile(optim_param_file):
        Batch_size = Batch_size
    else:
        Batch_size = 64

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset
                 .shuffle(buffer_size=10000)
                 .batch(Batch_size)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(Batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    print(model.summary())

    model.compile(loss='mse', optimizer=adam)

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
    
    models_dir =  "mlp_trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Directory",  models_dir, "is created sucessfully where the trained models will be saved")
    else:
        print("Directory",  models_dir, "already exists where the trained models will be saved")

    filepath=models_dir+"/mlp_model-{epoch:02d}-tloss-{loss:.3e}-vloss-{val_loss:.3e}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, loss_history_callback, es]

    if os.path.isfile(optim_param_file):
        model.fit(train_dataset,  
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=[callbacks_list, es]) 
    else:
        model.fit(train_dataset,  
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=[callbacks_list, es]) 
