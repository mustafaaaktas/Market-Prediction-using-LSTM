# Stock Market Prediction using LSTM and Optuna Hyperparameter Tuning

This project focuses on predicting stock prices using Long Short-Term Memory (LSTM) neural networks, a type of Recurrent Neural Network (RNN) particularly effective for time series forecasting. We employ **Optuna**, an automatic hyperparameter optimization framework, to fine-tune our LSTM model for optimal performance.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Fetching and Transformation](#data-fetching-and-transformation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [Detailed Explanation of Modules and Changes](#detailed-explanation-of-modules-and-changes)
- [Methodology and Understanding](#methodology-and-understanding)
- [Conclusion](#conclusion)

---

## Introduction

Predicting stock market prices is a challenging task due to the complex and volatile nature of financial markets. Machine learning models, especially LSTM networks, have shown promise in capturing temporal dependencies in time series data.

This project aims to:

- Fetch historical stock data for specified tickers.
- Perform data preprocessing and exploratory data analysis.
- Train an LSTM model to predict future stock prices.
- Utilize Optuna to optimize hyperparameters and improve model performance.
- Provide comprehensive documentation and code for reproducibility.

---

## Project Structure

Stock Market Prediction using LSTM/

├── data/<br>
│   ├── config.py               # Configuration file for parameters and hyperparameters  
│  
├── data_processing/  
│   ├── data_fetch.py           # Fetches historical stock data  
│   ├── data_transform.py       # Transforms raw data for analysis and modeling  
│  
├── model/  
│   ├── training.py             # Contains functions to build and train the LSTM model  
│  
├── utils/  
│   ├── stats_summary.py        # Generates statistical summaries and metrics  
│   ├── visuals.py              # Contains functions for data visualization (not shown)  
│  
├── main.py                     # Main script to run the entire pipeline  
├── OptunaTuner.py              # Hyperparameter tuning using Optuna  
├── README.md                   # Project documentation  
├── requirements.txt            # Python dependencies

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mustafaaaktas/Market-Prediction-using-LSTM

	2.	Create a virtual environment (optional but recommended): 
           python3 -m venv venv source venv/bin/activate
           On Windows use `venv\Scripts\activate`

	3.	Install the required packages: pip install -r requirements.txt

**Usage**

	1.	Configure the parameters: Open data/config.py and adjust the parameters as needed.

* TICKERS = ['MSFT']           # Stock tickers to fetch data for
* START_DATE = '2019-10-01'    # Start date for data fetching
* END_DATE = '2024-10-01'      # End date for data fetching


	2.	Run Hyperparameter Tuning (Optional but highly recommended): 
            Execute optuna_tuner.py to find the best hyperparameters.
            This will update config.py with the best-found hyperparameters.

* LOOK_BACK = 85
* EPOCHS = 37
* BATCH_SIZE = 16
* LSTM_UNITS_LAYER_1 = 59
* LSTM_UNITS_LAYER_2 = 111
* DROPOUT_RATE = 0.18867207567360111
* LEARNING_RATE = 0.00995441871757892


	3.	Run the main script:
            Execute main.py to run data fetching, preprocessing, model training, and evaluation.


**Data Fetching and Transformation**

Fetching Data (data_fetch.py)

	•	Function: fetch_stock_data(tickers, start, end)
	•	Description: Fetches historical stock data from Yahoo Finance 
                         for the specified tickers and date range.
	•	Usage: raw_data = fetch_stock_data(TICKERS, START_DATE, END_DATE)

Data Transformation (data_transform.py)

	•	Function: transform_data(data)
	•	Description: Transforms the raw data by resetting the index,
                         selecting relevant columns, converting dates,
                         and sorting chronologically.
	•	Usage: transformed_data = transform_data(raw_data)


Exploratory Data Analysis (EDA)

Statistical Summaries (stats_summary.py)

	•	Functions:
	•	data_info(data)
	•	descriptive_statistics(data)
	•	calculate_moving_averages(data, periods)
	•	calculate_daily_returns(data)
	•	average_closing_prices(data)
	•	Description: Generates various statistical summaries, including data info,
                     descriptive statistics, moving averages, daily returns, and
                     average closing prices.
	•	Usage: data_info(transformed_data),
                   descriptive_statistics(transformed_data),
                   calculate_moving_averages(transformed_data, MOVING_AVERAGE_PERIODS),
                   daily_returns = calculate_daily_returns(transformed_data), 
                   average_closing_prices(transformed_data)


Data Visualization (visuals.py)

	•	Note: The visuals.py module contains functions to visualize data trends,
                  moving averages, daily returns, and more. These visualizations
                  aid in understanding the data patterns and correlations.

Model Training and Evaluation

Data Preparation (training.py)

	•	Function: prepare_data_for_training(data, look_back)
	•	Description: Prepares the data for training by scaling, creating sequences,
                         and splitting into training, validation, and test sets.
	•	Usage: x_train, y_train, x_val, y_val, x_test, y_test,
               scaler = prepare_data_for_training(transformed_data,
               LOOK_BACK)

Building the LSTM Model (training.py)

	•	Function: build_lstm_model(input_shape)
	•	Description: Builds and compiles the LSTM model using the
                     specified hyperparameters from config.py.
	•	Usage: model = build_lstm_model(input_shape)

Training the Model (training.py)

	•	Function: train_lstm_model(model, x_train, y_train, x_val, y_val, epochs, batch_size)
	•	Description: Trains the LSTM model with early stopping and learning rate
                         reduction to prevent overfitting.
	•	Usage: train_lstm_model(model, x_train, y_train, x_val, y_val, EPOCHS, BATCH_SIZE)

Evaluating the Model

	•	Prediction: predictions = model.predict(x_test)
	•	Inverse Scaling: predictions = scaler.inverse_transform(predictions),
                             y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
	•	Performance Metrics: rmse = np.sqrt(np.mean((predictions - y_test_scaled) ** 2)),
                                 print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
	•	Visualization: def plot_predictions(y_test, predictions):
                              plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                              plt.plot(y_test, label='Actual Prices', color='blue')
                              plt.plot(predictions, label='Predicted Prices', color='red')
                              plt.title("LSTM Model - Actual vs Predicted Prices")
                              plt.xlabel("Time")
                              plt.ylabel("Price")
                              plt.legend()
                              plt.show()



**Hyperparameter Tuning with Optuna**

What is Optuna?

Optuna is an automatic hyperparameter optimization framework designed to find the best hyperparameters efficiently and effectively. It leverages techniques like pruning and early stopping to speed up the optimization process.



**Objective Function: <br>**
The objective(trial) function defines the model and the hyperparameters to tune. It returns the validation loss, which Optuna aims to minimize. <br>

Hyperparameters Tuned:
* LOOK_BACK: Number of time steps for the input sequence.
* LSTM_LAYERS: Number of LSTM layers (1 or 2).
* LSTM_UNITS_LAYER_1 and LSTM_UNITS_LAYER_2: Units in LSTM layers.
* DROPOUT_RATE: Dropout rate for regularization.
* BATCH_SIZE: Batch size for training.
* LEARNING_RATE: Learning rate for the optimizer.
* OPTIMIZER: Choice of optimizer (‘Adam’ or ‘RMSprop’).

* EarlyStopping: Stops training if the validation loss doesn’t improve.
* ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving.
* TFKerasPruningCallback: Allows Optuna to prune unpromising trials.



**Updating config.py:**

After tuning, the best hyperparameters are automatically updated in config.py for use in subsequent runs.

**Results**

After running the entire pipeline with the optimized hyperparameters, the model achieved: <br>
* Root Mean Squared Error (RMSE): Approximately 8.53, which is acceptable given the scale of stock prices.

**Observations:**
* The model captures the general trend of the stock prices.
* Early stopping and learning rate scheduling effectively prevent overfitting.
* The predictions are close to the actual prices with a minor underestimation bias.

**Acknowledgments**

This project was developed to demonstrate the application of LSTM networks in time series forecasting and the effectiveness of hyperparameter tuning using Optuna.

**Detailed Explanation of Modules and Changes**

_**main.py**_

Purpose: Serves as the main script to execute the data fetching, preprocessing,
                     model training, and evaluation pipeline.

* Key Functions and Steps: Data Fetching and Transformation: Fetches historical stock data using fetch_stock_data().
Transforms the data using transform_data().

* Exploratory Data Analysis: Generates data information and statistical summaries.
Calculates moving averages and daily returns.

* Data Preparation: Prepares data for training with prepare_data_for_training().

* Model Building and Training: Builds the LSTM model with build_lstm_model(). Trains the model using train_lstm_model().

* Evaluation: Makes predictions and evaluates the model performance. Visualizes the results.

* Changes Made: Integrated hyperparameter tuning results from config.py.
Enhanced logging and output messages for clarity.
Included visualization of actual vs. predicted prices.

**_OptunaTuner.py_**

Purpose: Automates hyperparameter tuning using Optuna.

* Key Components: Objective Function: Defines the model and returns validation loss.

* Hyperparameters: Includes a wider range and additional parameters like the optimizer choice.

* Callbacks: Incorporates early stopping and pruning to prevent overfitting and reduce computation time.

* Changes Made:
Added random seed setting for reproducibility.
Adjusted hyperparameter search space for more effective optimization.
Implemented Optuna’s pruning mechanism to skip unpromising trials.
Updated config.py automatically with the best-found hyperparameters.

**_training.py_**

Purpose: Contains functions to prepare data, build, and train the LSTM model.
* Key Functions:

  * prepare_data_for_training(): Scales data and splits it into training, validation, and test sets.
  
  * build_lstm_model(): Constructs the LSTM model based on hyperparameters.
  
  * train_lstm_model(): Trains the model with early stopping and learning rate reduction.

* Changes Made:
Simplified the LSTM model architecture for better generalization.
Added callbacks to prevent overfitting.
Ensured compatibility with hyperparameters from config.py.

_**data/config.py**_

Purpose: Stores configuration parameters and hyperparameters.

* Key Parameters: Stock tickers, date range, moving average periods.
Hyperparameters for the LSTM model, which are updated by OptunaTuner.py.
* Changes Made: Included additional parameters like LSTM_LAYERS and OPTIMIZER.
Ensured that hyperparameters are updated programmatically.

_**data_processing/data_fetch.py**_

Purpose: Fetches historical stock data from Yahoo Finance.
* Functionality: Downloads data for specified tickers and date range.
Handles exceptions and skips tickers with no data.

**_data_processing/data_transform.py_**

Purpose: Transforms raw data for analysis and modeling. 
* Functionality:
Selects relevant columns.
Converts date formats and sorts data chronologically.

**_utils/stats_summary.py_**

Purpose: Generates statistical summaries and metrics.
* Functions:

  * data_info(): Displays data information.

  * descriptive_statistics(): Computes descriptive stats for each company.
  
  * calculate_moving_averages(): Adds moving averages to the dataset.
  
  * calculate_daily_returns(): Calculates daily returns.
  
  * average_closing_prices(): Computes average closing prices.

**Methodology and Understanding**

* Why LSTM?

  * LSTMs are capable of learning long-term dependencies in sequential data, making them suitable for time series forecasting like stock prices.

* Why Hyperparameter Tuning with Optuna?

  * Finding the optimal hyperparameters manually is time-consuming and inefficient. Optuna automates this process using advanced search algorithms and pruning strategies, leading to better model performance.

* Training Approach:
  * Early Stopping: Prevents overfitting by stopping training when the validation loss stops improving.
  * Learning Rate Scheduling: Adjusts the learning rate during training to facilitate convergence.
  * Data Splitting: Ensures that the model is trained and validated on different subsets of data to generalize well.
* Changes Made:
  * 	Simplified the LSTM model architecture to prevent overfitting.
  * 	Adjusted hyperparameter ranges for more effective optimization.
  * 	Ensured proper data handling and preprocessing for time series data.

**Conclusion**

This project demonstrates the effective use of LSTM networks for stock price prediction and highlights the importance of hyperparameter tuning in machine learning models. By carefully preprocessing data, optimizing model parameters, and preventing overfitting, we achieved a model that generalizes well and provides valuable predictions.

Feel free to explore the code, experiment with different parameters, and extend the project to include more features or different models. Happy coding!

