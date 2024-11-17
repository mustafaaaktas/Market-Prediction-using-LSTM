"""
Configuration file for setting project-wide constants and parameters.
This file should be used for defining variables that might need
adjustment or reference across multiple files.
"""

# --- Ticker and Date Configuration ---
TICKERS = ['MSFT']           # Default stock ticker to fetch data
START_DATE = '2019-10-01'    # Start date for fetching data
END_DATE = '2024-10-01'      # End date for fetching data

# --- Moving Average Configuration ---
MOVING_AVERAGE_PERIODS = [10, 20, 50]  # Periods for calculating moving averages

# --- LSTM Model Configuration ---
# The following hyperparameters will be updated by optuna_tuner.py
LOOK_BACK = 85            # Default number of time steps (look-back window) for LSTM model input
EPOCHS = 37               # Default number of epochs for training the LSTM model
BATCH_SIZE = 16           # Default batch size for training
LSTM_UNITS_LAYER_1 = 59  # Default number of units in the first LSTM layer
LSTM_UNITS_LAYER_2 = 111   # Default number of units in the second LSTM layer
DROPOUT_RATE = 0.18867207567360111        # Default dropout rate for regularization in LSTM layers (0 to disable)
LEARNING_RATE = 0.00995441871757892     # Default learning rate for the optimizer

# --- Plot Configuration ---
PLOT_WIDTH = 12           # Width of the plot
PLOT_HEIGHT = 6           # Height of the plot

# --- Other Parameters ---
RANDOM_SEED = 42          # Seed for reproducibility
LSTM_LAYERS = 1
OPTIMIZER = "Adam"
