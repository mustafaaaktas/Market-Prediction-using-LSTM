"""
Main script for fetching, transforming, analyzing, and visualizing stock data,
as well as training and evaluating the LSTM model.
"""
from matplotlib import pyplot as plt
import numpy as np

# Import configurations
from config import BATCH_SIZE, END_DATE, EPOCHS, LOOK_BACK, \
    MOVING_AVERAGE_PERIODS, PLOT_HEIGHT, PLOT_WIDTH, START_DATE, TICKERS

# Import functions from data_processing
from data_processing.data_fetch import fetch_stock_data
from data_processing.data_transform import transform_data

# Import functions from utils
from utils.stats_summary import (
    data_info, descriptive_statistics, calculate_moving_averages,
    calculate_daily_returns, average_closing_prices
)
from utils.visuals import (
    plot_price_trend, plot_moving_averages, plot_daily_returns,
    plot_daily_return_histograms, plot_correlation_heatmap,
    plot_volume_of_sales, plot_risk_vs_return
)

# Import functions from model
from model.training import (prepare_data_for_training,
                            build_lstm_model,
                            train_lstm_model)

# Fetch and transform data
print(f"Fetching data for tickers: {TICKERS} from {START_DATE} to {END_DATE}...")
raw_data = fetch_stock_data(TICKERS, START_DATE, END_DATE)
transformed_data = transform_data(raw_data)

# Display sample data
print("\nData Sample (Head):")
print(transformed_data.head())
print("\nData Sample (Tail):")
print(transformed_data.tail())

# Perform data summary and statistics
data_info(transformed_data)
descriptive_statistics(transformed_data)
calculate_moving_averages(transformed_data, MOVING_AVERAGE_PERIODS)
daily_returns = calculate_daily_returns(transformed_data)
average_closing_prices(transformed_data)

# # Visualization
# print("\nGenerating visualizations...")
# plot_price_trend(transformed_data)
# plot_moving_averages(transformed_data, MOVING_AVERAGE_PERIODS)
# plot_daily_returns(transformed_data)
# plot_daily_return_histograms(transformed_data)
# plot_correlation_heatmap(transformed_data)
# plot_volume_of_sales(transformed_data)
# plot_risk_vs_return(daily_returns)

# Prepare data for model training and evaluation
print("\nPreparing data for LSTM model training...")
x_train, y_train, x_test, y_test, scaler = prepare_data_for_training(transformed_data, LOOK_BACK)

# Display data shapes for verification
print("\nData Shapes:")
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Build and train the LSTM model
print("\nBuilding and training the LSTM model...")
input_shape = (x_train.shape[1], x_train.shape[2])
model = build_lstm_model(input_shape)
train_lstm_model(model, x_train, y_train, EPOCHS, BATCH_SIZE)
print("Model training completed.")

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(x_test)

# Inverse scale predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Display predictions and actual values
print("\nInverse Scaled Predictions (first 10 values):")
print(predictions[:10])
print("\nInverse Scaled Actual Values (first 10 values):")
print(y_test_scaled[:10])

# Evaluate the model
rmse = np.sqrt(np.mean((predictions - y_test_scaled) ** 2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")


# Visualize predictions
def plot_predictions(y_test, predictions):
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.title("LSTM Model - Actual vs Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


plot_predictions(y_test_scaled, predictions)

print("\nProcess completed.")
