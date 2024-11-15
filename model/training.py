"""
training.py

This module includes functionality for preparing data for training, building, and training LSTM models.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input


def prepare_data_for_training(data, look_back=60):
    """
    Prepares data for training and testing.

    Parameters:
        data (pd.DataFrame): Transformed stock data with an 'Adj Close' column.
        look_back (int): The number of previous time steps to use as input features.

    Returns:
        tuple: x_train, y_train, x_test, y_test, scaler
    """
    if 'Adj Close' not in data.columns:
        raise ValueError("Data must contain an 'Adj Close' column.")

    # Extract the `Adj Close` column and reshape for scaling
    adj_close_data = data['Adj Close'].values.reshape(-1, 1)

    # Initialize the scaler and scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(adj_close_data)

    # Prepare sequences for training and testing
    x_train, y_train = [], []
    x_test, y_test = [], []

    training_data_len = int(len(scaled_data) * 0.8)  # 80% for training

    for i in range(look_back, training_data_len):
        x_train.append(scaled_data[i - look_back:i, 0])
        y_train.append(scaled_data[i, 0])

    for i in range(training_data_len, len(scaled_data)):
        x_test.append(scaled_data[i - look_back:i, 0])
        y_test.append(scaled_data[i, 0])

    # Convert to numpy arrays and reshape for LSTM input
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Debugging information
    print("\nSample of scaled y_train (should be between 0 and 1):", y_train[:5])
    print("Sample of scaled y_test (should be between 0 and 1):", y_test[:5])

    return x_train, y_train, x_test, y_test, scaler


def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for stock price prediction.

    Parameters:
        input_shape (tuple): Shape of the input data (time steps, features).

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    model = Sequential()

    # Explicitly define the input layer
    model.add(Input(shape=input_shape))

    # Add the first LSTM layer with return_sequences=True for stacking
    model.add(LSTM(units=50, return_sequences=True))

    # Add a second LSTM layer without returning sequences
    model.add(LSTM(units=50, return_sequences=False))

    # Add a Dense output layer
    model.add(Dense(units=1))  # Predicting a single output

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    print("LSTM model built and compiled successfully "
          "(using explicit Input layer).")
    return model


def train_lstm_model(model, x_train, y_train, epochs=10, batch_size=32):
    """
    Trains the LSTM model.

    Parameters:
        model (tf.keras.Model): Compiled LSTM model.
        x_train (np.array): Training input data.
        y_train (np.array): Training target data.
        epochs (int): Number of epochs to train for.
        batch_size (int): Size of batches for training.

    Returns:
        History: Training history object for further analysis.
    """
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    print("Model training completed.")
    return history
