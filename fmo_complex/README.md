# FMO_complex

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

1. **PUNN_model-91-tloss-1.095e-05-vloss-1.245e-05**
2. **su7-PUNN_model-157-tloss-1.008e-05-vloss-2.274e-05**
3. **PINN_model-1063-tloss-1.077e-05-vloss-1.292e-05**
4. **su7-PINN_model-142-tloss-1.059e-05-vloss-1.944e-05**
