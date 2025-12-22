print("Testing imports...")
import pandas as pd
import numpy as np
import sklearn
import streamlit
import yfinance as yf

print("✅ All imports successful!")
print("\nDownloading sample data...")
data = yf.download("AAPL", start="2024-01-01", end="2024-01-31", progress=False)
print(f"✅ Downloaded {len(data)} days of data")
print("\n🎉 Everything is working!")
