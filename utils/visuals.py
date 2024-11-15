"""
visuals.py

This module contains functions for visualizing stock market data and analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_price_trend(data):
    """
    Plots the trend of the adjusted closing prices for each company.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data with 'Date' and 'Adj Close' columns.
    """
    plt.figure(figsize=(12, 6))
    for company in data['company_name'].unique():
        company_data = data[data['company_name'] == company]
        plt.plot(company_data['Date'], company_data['Adj Close'], label=company)
    plt.title("Stock Price Trend (Adj Close)")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Closing Price")
    plt.legend()
    plt.show()


def plot_moving_averages(data, periods):
    """
    Plots the moving averages for specified periods for each company.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data.
        periods (list): List of integers representing the moving average periods.
    """
    plt.figure(figsize=(12, 6))
    for company in data['company_name'].unique():
        company_data = data[data['company_name'] == company]
        plt.plot(company_data['Date'], company_data['Adj Close'], label=f"{company} Adj Close")
        for period in periods:
            col_name = f"MA_{period}"
            if col_name in company_data.columns:
                plt.plot(company_data['Date'], company_data[col_name], label=f"{company} {period}-Day MA")
    plt.title("Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_daily_returns(data):
    """
    Plots daily returns for each company.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data with 'Daily_Return' column.
    """
    plt.figure(figsize=(12, 6))
    for company in data['company_name'].unique():
        company_data = data[data['company_name'] == company]
        plt.plot(company_data['Date'], company_data['Daily_Return'], label=f"{company} Daily Returns")
    plt.title("Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.show()


def plot_daily_return_histograms(data, bins=50):
    """
    Plots histograms of daily returns for each company.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data with 'Daily_Return' column.
        bins (int): Number of bins for the histograms.
    """
    for company in data['company_name'].unique():
        company_data = data[data['company_name'] == company]
        plt.figure(figsize=(8, 5))
        sns.histplot(company_data['Daily_Return'].dropna(), bins=bins, kde=True)
        plt.title(f"{company} Daily Return Histogram")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.show()


def plot_correlation_heatmap(data):
    """
    Plots a heatmap of correlations between the adjusted closing prices of all companies.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data.
    """
    pivot_data = data.pivot(index='Date', columns='company_name', values='Adj Close')
    correlation_matrix = pivot_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap (Adj Close)")
    plt.show()


def plot_volume_of_sales(data):
    """
    Plots the volume of sales for each company over time.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data with 'Volume' column.
    """
    plt.figure(figsize=(12, 6))
    for company in data['company_name'].unique():
        company_data = data[data['company_name'] == company]
        plt.plot(company_data['Date'], company_data['Volume'], label=f"{company} Volume")
    plt.title("Volume of Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend()
    plt.show()


def plot_risk_vs_return(data):
    """
    Plots risk vs return for each company based on daily returns.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock data with 'Daily_Return' column.
    """
    summary = data.groupby('company_name')['Daily_Return'].agg(['mean', 'std']).reset_index()
    summary.columns = ['company_name', 'Return', 'Risk']
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=summary, x='Risk', y='Return', hue='company_name', s=100)
    plt.title("Risk vs Return")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return (Mean)")
    plt.legend(title="Company")
    plt.show()
