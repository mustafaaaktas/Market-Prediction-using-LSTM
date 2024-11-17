"""
This module includes functionality for preparing data for training, building, and training LSTM models.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import hyperparameters from config.py
from data.config import (
    LOOK_BACK, EPOCHS, BATCH_SIZE,
    LSTM_UNITS_LAYER_1, LSTM_UNITS_LAYER_2,
    DROPOUT_RATE, LEARNING_RATE
)


def prepare_data_for_training(data, look_back=LOOK_BACK):
    """
    Prepares data for training, validation, and testing.

    Parameters:
        data (pd.DataFrame): Transformed stock data with an 'Adj Close' column.
        look_back (int): The number of previous time steps to use as input features.

    Returns:
        tuple: x_train, y_train, x_val, y_val, x_test, y_test, scaler
    """
    if 'Adj Close' not in data.columns:
        raise ValueError("Data must contain an 'Adj Close' column.")

    print(f"\nPreparing data with LOOK_BACK period of {look_back}...")

    # Extract the `Adj Close` column and reshape for scaling
    adj_close_data = data['Adj Close'].values.reshape(-1, 1)

    # Initialize the scaler and scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(adj_close_data)
    print("Data scaling completed.")

    # Prepare sequences for training and testing
    x_data, y_data = [], []

    for i in range(look_back, len(scaled_data)):
        x_data.append(scaled_data[i - look_back:i, 0])
        y_data.append(scaled_data[i, 0])

    # Convert to numpy arrays and reshape for LSTM input
    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))

    # Split data into training (70%), validation (15%), and testing (15%)
    total_data_len = len(x_data)
    train_size = int(total_data_len * 0.7)
    val_size = int(total_data_len * 0.85)

    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    x_val = x_data[train_size:val_size]
    y_val = y_data[train_size:val_size]
    x_test = x_data[val_size:]
    y_test = y_data[val_size:]

    # Debugging information
    print("\nData splitting completed:")
    print(f"Training data size: {x_train.shape[0]}")
    print(f"Validation data size: {x_val.shape[0]}")
    print(f"Testing data size: {x_test.shape[0]}")

    print("\nSample of scaled y_train (should be between 0 and 1):", y_train[:5])
    print("Sample of scaled y_val (should be between 0 and 1):", y_val[:5])
    print("Sample of scaled y_test (should be between 0 and 1):", y_test[:5])

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler


def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for stock price prediction using configuration parameters.

    Parameters:
        input_shape (tuple): Shape of the input data (time steps, features).

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    print(f"\nBuilding LSTM model with input shape {input_shape}...")
    model = Sequential()

    # Add a single LSTM layer with dropout
    model.add(LSTM(units=LSTM_UNITS_LAYER_1, input_shape=input_shape))
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))

    # Add a Dense output layer
    model.add(Dense(units=1))  # Predicting a single output

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')

    print("LSTM model compiled successfully.")
    return model


def train_lstm_model(model, x_train, y_train, x_val=None, y_val=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Trains the LSTM model.

    Parameters:
        model (tf.keras.Model): Compiled LSTM model.
        x_train (np.array): Training input data.
        y_train (np.array): Training target data.
        x_val (np.array): Validation input data.
        y_val (np.array): Validation target data.
        epochs (int): Number of epochs to train for.
        batch_size (int): Size of batches for training.

    Returns:
        History: Training history object for further analysis.
    """
    print(f"\nStarting training for {epochs} epochs with batch size {batch_size}...")

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    callbacks = [early_stopping, lr_scheduler]

    if x_val is not None and y_val is not None:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    print("Model training completed.")
    return history
