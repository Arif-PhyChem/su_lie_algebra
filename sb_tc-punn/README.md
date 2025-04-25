# SB_TC-PUNN

This folder contains a collection of files for TC-PUNN tested on SB model.

## Files

1. **pred_data**
    - Contains data predicted by the models, including dynamics data, eigenvalues of the Reduced Density Matrix (RDM) at each time step, trace, and other related information.

2. **hyperopt_optim.py**
    - Performs Hyperopt optimization of the neural network architecture. (Note: Optimization is not performed in this implementation.)

3. **ml_models.py**
    - Contains the CNN-LSTM neural network architecture used in the project.

4. **prep_input.py**
    - Prepares the training data for the models.

5. **prep_data.py**
    - Calls `prep_input.py` to handle data preparation tasks.

6. **run_dyn.py**
    - Predicts dynamics using the trained models.

6. **train_model.py**
    - Trains a CNN-LSTM model on the prepared training data.

7. **plot_all.py:** 
    - Plotting dynamics, trace and eigenvalues

## Trained Models

We used the following models with comparable training and validation loss (The 1st number shows epoch, 2nd number shows training loss (tloss) and the 3rd number shows  validation loss (vloss)):

1. **cnn_lstm_model-507-tloss-5.030e-06-vloss-6.616e-06**