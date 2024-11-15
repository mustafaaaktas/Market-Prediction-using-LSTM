"""
data_fetch.py

This module handles the retrieval of stock data using Yahoo Finance.
"""

import yfinance as yf
import pandas as pd


def fetch_stock_data(tickers, start, end):
    """
    Fetch historical stock data for given tickers within a date range.

    Parameters:
        tickers (list): List of stock tickers to fetch data for.
        start (str): Start date in the format 'YYYY-MM-DD'.
        end (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: Combined DataFrame with historical data for all tickers.
    """

    combined_data = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                print(f"Warning: No data found for {ticker}. Skipping.")
                continue

            # Add ticker name as a separate column
            data['company_name'] = ticker
            combined_data.append(data)
            print(f"Data fetched for ticker {ticker} "
                  f"from {start} to {end} successfully.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if combined_data:
        return pd.concat(combined_data, ignore_index=False) # Maintain datetime index
    else:
        print("No data fetched for any tickers.")
        return pd.DataFrame()  # Return empty DataFrame if no data was fetched
