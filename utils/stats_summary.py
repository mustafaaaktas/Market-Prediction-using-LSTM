"""
stats_summary.py

This module contains functions to generate statistical summaries and metrics for the dataset.
"""

import pandas as pd
from pandas.errors import PerformanceWarning


def data_info(data):
    """
    Prints information about the dataset, including column names, data types, and memory usage.
    """
    print("\nData Information:")
    print(data.info())


def descriptive_statistics(data):
    """
    Generate descriptive statistics for each company in the dataset.
    """

    # Ensure the DataFrame is ready for groupby operations
    if isinstance(data.index, pd.MultiIndex):
        print("Data has a multi-index.")
        if not data.index.is_lexsorted():
            print("Index is not lexsorted. "
                  "Sorting index for better performance...")
            data = data.sort_index()
        else:
            print("Index is lexsorted.")
    else:
        print("Data does not have a multi-index.")

    # Perform the groupby operation based on 'company_name'
    stats = data.groupby("company_name")["Adj Close"].describe()

    # Print descriptive statistics for each company
    print("\nDescriptive Statistics for Each Company:")
    print(stats)

    return stats


def calculate_moving_averages(data, periods):
    """
    Adds moving averages for specified periods to the dataset and prints the values.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data.
        periods (list): List of integers representing the moving average periods.

    Returns:
        pd.DataFrame: DataFrame with moving averages added.
    """
    for period in periods:
        col_name = f"MA_{period}"
        # Calculate the moving average for each company
        data[col_name] = data.groupby('company_name')['Adj Close'].transform(lambda x: x.rolling(window=period).mean())

        # Print the moving average value for each company
        print(f"\n{period}-Day Moving Averages:")
        for company in data['company_name'].unique():
            company_data = data[data['company_name'] == company]
            last_ma_value = company_data[col_name].iloc[-1]  # Get the last calculated value
            print(f"{company}: {last_ma_value:.2f}")

    return data


def calculate_daily_returns(data):
    """
    Calculates and adds daily returns to the dataset.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data.

    Returns:
        pd.DataFrame: DataFrame with daily returns added.
    """
    data['Daily_Return'] = data.groupby('company_name')['Adj Close'].pct_change()
    print("\nDaily Returns Per Company:")
    for company in data['company_name'].unique():
        print(f"{company}: "
              f"Last Daily Return = "
              f"{data[data['company_name'] == company]['Daily_Return'].iloc[-1]:.4f}")
    return data


def average_closing_prices(data):
    """
    Calculates and prints the average closing price for each company.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data.
    """
    averages = data.groupby('company_name')['Adj Close'].mean()
    print("\nAverage Closing Prices Per Company:")

    for company, avg in averages.items():
        # Ensure avg is a scalar and properly formatted
        if isinstance(avg, pd.Series):
            avg = avg.iloc[0]  # Extract the scalar value if avg is a Series
        print(f"{company}: {avg:.2f}")

