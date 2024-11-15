"""
data_transform.py

This module handles the transformation of raw stock data for analysis and modeling.
"""

import pandas as pd


def transform_data(data):
    """
    Transform raw stock data into a format suitable for analysis and modeling.

    Parameters:
        data (pd.DataFrame): Raw stock data.

    Returns:
        pd.DataFrame: Transformed stock data with necessary adjustments.
    """
    if data.empty:
        print("Data is empty. Cannot transform.")
        return pd.DataFrame()  # Return empty DataFrame if input data is empty

    # Reset index to ensure continuity
    data.reset_index(inplace=True)

    # Keep only the relevant columns
    data = data[['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume',
                 'company_name']]
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort by date to ensure chronological order
    data.sort_values('Date', inplace=True)

    print("Data transformed successfully.")
    return data
