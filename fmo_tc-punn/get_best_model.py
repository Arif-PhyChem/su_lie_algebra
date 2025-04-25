import re
import os

def best_model(model_dir: str,
               ml_model: str):

  # Define a regex pattern to extract the epoch number from filenames
  if ml_model == 'cnn_lstm':
        pattern = re.compile(r'cnn_lstm_model-(\d+)-tloss-\d+\.\d+e-\d+-vloss-\d+\.\d+e-\d+\.keras')
  if ml_model == 'lstm':
        pattern = re.compile(r'lstm_model-(\d+)-tloss-\d+\.\d+e-\d+-vloss-\d+\.\d+e-\d+\.keras')

  # List all files in the directory
  model_files = os.listdir(model_dir)

  # Extract epoch numbers without loading models
  max_epoch = 0
  latest_model_file = None
  for filename in model_files:
    match = pattern.match(filename)
    if match:
      epoch = int(match.group(1))  # Extract the epoch number
      if epoch > max_epoch:
        max_epoch = epoch
        latest_model_file = filename

  # Check if any models found
  if not latest_model_file:
    raise ValueError("No model files found in the directory.")

  # Build path to the model with highest epoch
  latest_model_path = os.path.join(model_dir, latest_model_file)

  # Return information about the best model (path and epoch)
  return latest_model_path, max_epoch
