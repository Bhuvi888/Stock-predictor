import sys
import os

# Add the current directory to the Python path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from config import get_model_path, get_plot_path, model_exists, plot_exists, create_ticker_dirs

# --- 0. App Configuration & Title ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("LSTM Stock Price Predictor")
st.write("Enter a valid stock ticker from Yahoo Finance (e.g., 'RELIANCE.NS', 'AAPL', 'TSLA').")
st.write("If this is the first time for a stock, the model will be trained live, which may take a few minutes.")

# --- 1. LSTM Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# --- 2. Global Parameters ---
FEATURES = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA', 'EMA', 'RSI', 'Day_of_week', 'Month']
INPUT_SIZE = len(FEATURES)
SEQ_LENGTH = 45
HIDDEN_LAYER_SIZE = 200
EPOCHS = 75
BATCH_SIZE = 64

# --- 3. Cached Data Function ---
@st.cache_data(show_spinner=False)
def get_stock_data(ticker: str):
    data = yf.download(ticker, start="2015-01-01", end=datetime.now())
    if data.empty:
        return None

    data['SMA'] = data['Close'].rolling(window=14).mean().fillna(0)
    data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean().fillna(0)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].fillna(0)
    data['Day_of_week'] = data.index.dayofweek
    data['Month'] = data.index.month

    return data

# --- 4. Model Training Function ---
def train_model(ticker, placeholder):
    model_filename = get_model_path(ticker)
    plot_filename = get_plot_path(ticker)

    with st.spinner(f"Training model for {ticker}... This may take a moment."):
        data = get_stock_data(ticker)
        if data is None:
            placeholder.error(f"Could not download data for {ticker}. Please check the ticker symbol.")
            return None, None

        # Ensure ticker-specific directories exist
        create_ticker_dirs(ticker)

        feature_data = data[FEATURES].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data)
        close_scaler = MinMaxScaler()
        close_scaler.fit_transform(data[['Close']])

        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data[i:(i + seq_length)]
                y = data[i + seq_length, 0]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        x, y = create_sequences(scaled_data, SEQ_LENGTH)
        train_size = int(len(y) * 0.8)
        x_train = torch.from_numpy(x[0:train_size]).float()
        y_train = torch.from_numpy(y[0:train_size]).float().view(-1, 1)
        x_test = torch.from_numpy(x[train_size:len(x)]).float()
        y_test = torch.from_numpy(y[train_size:len(y)]).float().view(-1, 1)

        model = LSTMModel(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        for i in range(EPOCHS):
            for seq, labels in train_loader:
                optimizer.zero_grad()
                y_pred = model(seq)
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()
            if (i + 1) % 15 == 0:
                placeholder.text(f"Epoch {i+1}/{EPOCHS}... Loss: {single_loss.item():.8f}")

        # Plot
        model.eval()
        test_predictions = []
        with torch.no_grad():
            for seq in x_test:
                test_predictions.append(model(seq.unsqueeze(0)).item())

        y_pred_inv = close_scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
        y_test_inv = close_scaler.inverse_transform(y_test.numpy())

        plt.figure(figsize=(12, 6))
        plt.title(f'Price Prediction for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plot_index = data.index[train_size + SEQ_LENGTH:]
        plt.plot(plot_index, y_test_inv, label='Actual Price')
        plt.plot(plot_index, y_pred_inv, label='Predicted Price')
        plt.legend()
        plt.savefig(get_plot_path(ticker))

        torch.save(model.state_dict(), get_model_path(ticker))
        placeholder.success(f"Model for {ticker} trained and saved successfully!")
        return model, plot_filename

# --- 5. Prediction Function ---
def predict_price(ticker, model):
    data = get_stock_data(ticker)
    if data is None or len(data) < SEQ_LENGTH:
        return None

    scaler = MinMaxScaler()
    scaler.fit(data[FEATURES].values)
    close_scaler = MinMaxScaler()
    close_scaler.fit(data[['Close']].values)

    last_sequence = data[FEATURES].tail(SEQ_LENGTH).values
    scaled_last_sequence = scaler.transform(last_sequence)
    input_tensor = torch.from_numpy(scaled_last_sequence).float().unsqueeze(0)

    model.eval()
    with torch.no_grad():
        predicted_scaled_price = model(input_tensor).item()

    dummy_array = np.zeros((1, len(FEATURES)))
    dummy_array[0, 0] = predicted_scaled_price
    predicted_price = close_scaler.inverse_transform(dummy_array[:, 0].reshape(-1, 1))[0][0]

    return predicted_price

# --- 6. Streamlit UI ---
ticker_input = st.text_input("Enter Stock Ticker:", "TATAMOTORS.NS").upper()
predict_button = st.button("Predict Close Price")
results_placeholder = st.empty()

if predict_button and ticker_input:
    model_filename = get_model_path(ticker_input)
    plot_filename = get_plot_path(ticker_input)

    model = LSTMModel(INPUT_SIZE, HIDDEN_LAYER_SIZE)

    if model_exists(ticker_input):
        st.info(f"Found pre-trained model for {ticker_input}. Loading...")
        model.load_state_dict(torch.load(model_filename))
        plot_to_show = plot_filename if plot_exists(ticker_input) else None
    else:
        st.warning(f"No pre-trained model found for {ticker_input}.")
        model, plot_to_show = train_model(ticker_input, results_placeholder)

    if model:
        predicted_price = predict_price(ticker_input, model)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label=f"Predicted Close Price for {ticker_input}", value=f"{predicted_price:.2f}")
            st.info("This prediction is based on historical data and should not be considered financial advice.")
        with col2:
            if plot_to_show:
                st.image(plot_to_show, caption=f"Prediction vs. Actual Prices for {ticker_input}")
            else:
                st.write("No plot available.")
