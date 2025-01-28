import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Fetch Historical Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Plot Stock Data
def plot_stock_data(data, column):
    plt.figure(figsize=(10, 6))
    plt.plot(data[column], label=f"{column} Price")
    plt.title(f"{column} Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Step 3: Create Sequences for Training
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

# Step 4: Build and Train LSTM Model
def build_lstm_model(seq_length):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main Code
if __name__ == "__main__":
    # Fetch stock data
    ticker = 'NVDA'
    start_date = '2025-01-01'
    end_date = '2025-01-29'
    data = fetch_stock_data(ticker, start_date, end_date)

    # Plot closing price
    plot_stock_data(data, 'Close')

    # Preprocess data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])

    # Create sequences
    seq_length = 50
    x, y = create_sequences(scaled_data, seq_length)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # Build and train model
    model = build_lstm_model(seq_length)
    model.fit(x, y, batch_size=32, epochs=10)

    # Predict and visualize results
    predictions = model.predict(x)
    predictions = scaler.inverse_transform(predictions)

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label="Actual Price")
    plt.plot(range(seq_length, seq_length + len(predictions)), predictions, label="Predicted Price")
    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
