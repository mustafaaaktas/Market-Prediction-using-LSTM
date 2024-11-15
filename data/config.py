"""
Configuration file for setting project-wide constants and parameters.
This file should be used for defining variables that might need
adjustment or reference across multiple files.
"""

# Ticker and Date Configuration
TICKERS = ['MSFT']  # Default stock ticker to fetch data
START_DATE = '2022-10-01'  # Start date for fetching data
END_DATE = '2023-10-01'    # End date for fetching data

# Moving Average Periods
MOVING_AVERAGE_PERIODS = [10, 20, 50]  # Periods for calculating moving averages

# Model Training Configuration
LOOK_BACK = 60   # Number of time steps (look-back window) for LSTM model input
EPOCHS = 10      # Number of epochs for training the LSTM model
BATCH_SIZE = 32  # Batch size for training

# Plot Configuration
PLOT_WIDTH = 12
PLOT_HEIGHT = 6

# Other Parameters
RANDOM_SEED = 42  # Seed for reproducibility
