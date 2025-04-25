# NN_models_comparison

This folder contains a collection of files and trained models comparing four NN models (MLP, LSTM, CNN and CNN-LSTM). This comparison is only done for PUNN approach.

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

7. **dyn_plot.py:** 
    - Plotting dynamics 
8. **eig_plot.py:** 
    - Plotting spectrum of eigenvalues
9. **trace_plot.py:** 
    - Plotting Trace
10. **lr_curve.py:** 
    - Plotting the loss vs number of epochs
11. **mae.py:** 
    - Calculating MAE for the test trajectory
12. **x_punn-pinn.npy:**
    - Training data X
13. **y_punn-pinn.npy:**
    - Training data Y

## Trained Models

We used the following models with comparable training and validation loss (The 1st number shows epoch, 2nd number shows training loss (tloss) and the 3rd number shows  validation loss (vloss)):

1. **MLP_model-274-tloss-1.556e-05-vloss-2.148e-05**
2. **LSTM_model-522-tloss-1.570e-05-vloss-2.129e-05**
3. **CNN_model-233-tloss-1.595e-05-vloss-2.212e-05**
4. **CNN_LSTM_model-284-tloss-1.544e-05-vloss-2.320e-05**