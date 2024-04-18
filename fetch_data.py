import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Fetch historical stock data
def fetch_stock_data(symbol):
    # Adjusted to fetch 24 months of data
    stock_data = yf.download(symbol, period="24mo", interval="1d")
    stock_data = stock_data[['Close']]  # We'll use only the 'Close' prices
    return stock_data

# Preprocess data
def preprocess_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    
    # Create a dataset where X is the number of past days' stock prices
    # and y is the next day's stock price
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Example usage
if __name__ == "__main__":
    data = fetch_stock_data('AAPL')  # Example symbol, replace with your target
    X, y, scaler = preprocess_data(data)
