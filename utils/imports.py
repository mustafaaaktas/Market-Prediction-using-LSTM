"""
imports.py

This module centralizes all commonly used imports for the project.
By importing this module, all required libraries and dependencies are accessible.

Usage:
    Instead of importing individual libraries in multiple files,
    use `from imports import *` to include everything.
"""

# Core libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Financial data retrieval libraries
import yfinance as yf
from pandas_datareader import data as pdr

# Preprocessing and utilities
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Machine Learning libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Enable Yahoo Finance override for pandas_datareader
yf.pdr_override()
