"""
This script uses Optuna to tune hyperparameters for the LSTM model.
It updates config.py with the best-found hyperparameters.
"""

import optuna
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
import numpy as np
import random

from training import prepare_data_for_training
from data_processing.data_fetch import fetch_stock_data
from data_processing.data_transform import transform_data
from data.config import TICKERS, START_DATE, END_DATE

import os
import re

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Fetch and transform data once to avoid repeating for each trial
print(f"Fetching data for tickers: {TICKERS} from {START_DATE} to {END_DATE}...")
raw_data = fetch_stock_data(TICKERS, START_DATE, END_DATE)
print("Data fetching completed.")

print("\nTransforming data...")
transformed_data = transform_data(raw_data)
print("Data transformation completed.")


def objective(trial):
    # Hyperparameters to tune
    hyperparams = {
        'LOOK_BACK': trial.suggest_int('LOOK_BACK', 30, 120),
        'LSTM_LAYERS': trial.suggest_categorical('LSTM_LAYERS', [1, 2]),
        'LSTM_UNITS_LAYER_1': trial.suggest_int('LSTM_UNITS_LAYER_1', 32, 128),
        'LSTM_UNITS_LAYER_2': trial.suggest_int('LSTM_UNITS_LAYER_2', 32, 128),
        'DROPOUT_RATE': trial.suggest_float('DROPOUT_RATE', 0.0, 0.5),
        'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [16, 32, 64]),
        'LEARNING_RATE': trial.suggest_float('LEARNING_RATE', 1e-5, 1e-2, log=True),
        'OPTIMIZER': trial.suggest_categorical('OPTIMIZER', ['Adam', 'RMSprop']),
    }

    print(f"\nStarting trial with parameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")

    # Prepare data with the selected look_back window
    x_train, y_train, x_val, y_val, _, _, _ = prepare_data_for_training(
        transformed_data, look_back=hyperparams['LOOK_BACK'])

    # Define the model
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units=hyperparams['LSTM_UNITS_LAYER_1'],
                   return_sequences=(hyperparams['LSTM_LAYERS'] > 1),
                   input_shape=(hyperparams['LOOK_BACK'], 1)))
    if hyperparams['DROPOUT_RATE'] > 0:
        model.add(Dropout(hyperparams['DROPOUT_RATE']))

    # Second LSTM layer (if applicable)
    if hyperparams['LSTM_LAYERS'] > 1:
        model.add(LSTM(units=hyperparams['LSTM_UNITS_LAYER_2']))
        if hyperparams['DROPOUT_RATE'] > 0:
            model.add(Dropout(hyperparams['DROPOUT_RATE']))

    # Output layer
    model.add(Dense(1))

    # Select optimizer
    if hyperparams['OPTIMIZER'] == 'Adam':
        optimizer = Adam(learning_rate=hyperparams['LEARNING_RATE'])
    else:
        optimizer = RMSprop(learning_rate=hyperparams['LEARNING_RATE'])

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
    pruning_callback = TFKerasPruningCallback(trial, monitor='val_loss')

    callbacks = [early_stopping, lr_scheduler, pruning_callback]

    # Train the model with the current trial's batch size and epochs
    try:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=100,  # Set high epochs; early stopping will handle the rest
            batch_size=hyperparams['BATCH_SIZE'],
            callbacks=callbacks,
            verbose=0
        )

        # Get the best validation loss of this trial
        val_loss = min(history.history['val_loss'])
        print(f"Trial completed with best validation loss: {val_loss}")

        return val_loss
    except optuna.exceptions.TrialPruned:
        print("Trial was pruned.")
        raise


if __name__ == "__main__":
    print("\nStarting hyperparameter tuning with Optuna...")
    # Create a study to minimize the validation loss
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50)

    # Display the best trial details after optimization
    best_trial = study.best_trial
    print("\nHyperparameter tuning completed.")
    print(f"Best trial number: {best_trial.number}")
    print(f"Best trial value (validation loss): {best_trial.value}")
    print(f"Best trial parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Update the hyperparameters in config.py
    config_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'config.py')

    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config_content = f.read()

        # For each hyperparameter, replace its value in the config content
        for key, value in best_trial.params.items():
            # Build a regex pattern to match the variable assignment, capturing comments
            pattern = rf'^(\s*{key}\s*=\s*)(.*?)(\s*(#.*)?$)'
            # Build the replacement line using \g<1> and \g<3> to avoid group reference issues
            if isinstance(value, str):
                replacement = rf'\g<1>"{value}"\g<3>'
            else:
                replacement = rf'\g<1>{value}\g<3>'
            # Use regex to replace the line
            new_config_content, count = re.subn(pattern, replacement, config_content, flags=re.MULTILINE)
            if count == 0:
                print(f"Warning: {key} not found in config.py. Adding it.")
                # If the key was not found, add it to the config file
                if isinstance(value, str):
                    new_line = f'{key} = "{value}"\n'
                else:
                    new_line = f"{key} = {value}\n"
                config_content += new_line
            else:
                print(f"Updated {key} in config.py")
                config_content = new_config_content

        # Write back the updated config content
        with open(config_file_path, 'w') as f:
            f.write(config_content)
        print("\nHyperparameters updated in config.py with the best-found parameters.")
    else:
        print(f"Error: {config_file_path} not found.")
