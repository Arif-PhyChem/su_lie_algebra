# sb_model

This folder contains a collection of files and trained models for SB model

## Files

1. **pred_data**
    - Contains data predicted by the models, including dynamics data, eigenvalues of the Reduced Density Matrix (RDM) at each time step, trace, and other related information.

2. **su_basis.py**
    - Generates the `su(n)` operators for a trajectory of RDM evolution.

3. **coeff_generator.py**
    - Generates `su(n)` coefficients corresponding to each `su(n)` operator. These coefficients are used to train the model.

4. **check_basis.py**
    - Verifies whether a generated `su(n)` operator satisfies the required conditions. (Refer to the corresponding article for more details.)

5. **custom_loss.py**
    - Implements a custom loss function with all constraints for both Physics-Informed Neural Networks (PINN) and `su(n)`-PINN.

6. **hyperopt_optim.py**
    - Performs Hyperopt optimization of the neural network architecture. (Note: Optimization is not performed in this implementation.)

7. **ml_models.py**
    - Contains the CNN-LSTM neural network architecture used in the project.

8. **prep_input.py**
    - Prepares the training data for the models.

9. **prep_data.py**
    - Calls `prep_input.py` to handle data preparation tasks.

10. **run_dyn.py**
    - Predicts dynamics using the trained models.

11. **train_model.py**
    - Trains a CNN-LSTM model on the prepared training data.

## Trained Models

We used the following models with comparable training and validation loss (The 1st number shows epoch, 2nd number shows training loss (tloss) and the 3rd number shows  validation loss (vloss)):

1. **PUNN_cnn_lstm_model-284-tloss-1.544e-05-vloss-2.320e-05**
2. **su2-PUNN_cnn_lstm_model-464-tloss-1.152e-05-vloss-3.191e-05**
3. **PINN_cnn_lstm_model-990-tloss-1.611e-05-vloss-2.381e-05**
4. **su2-PINN_cnn_lstm_model-359-tloss-1.382e-05-vloss-3.464e-05**
