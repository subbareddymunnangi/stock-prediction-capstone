import yfinance as yf
import pandas as pd
import numpy as np

def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading {ticker} data...")
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    print(f"Downloaded {len(stock_data)} days of data")
    return stock_data

def create_features(df):
    df = df.copy()
    
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # Target: Next day's price
    df['Target'] = df['Close'].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    print(f"Created {len(df.columns)} features")
    return df