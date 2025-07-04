import sys
import os
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from config import get_model_path, get_plot_path, model_exists, plot_exists, create_ticker_dirs

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- 0. App Configuration & Title ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- Header ---
with st.container():
    st.title("LSTM Stock Price Predictor")
    st.write("Select a stock from the list, or enter a custom ticker from Yahoo Finance (e.g., 'RELIANCE.NS', 'TATAMOTORS.NS').")
    st.write("The model will be trained live if a pre-trained version for the selected parameters isn't available. This might take a few minutes.")
    

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

# --- 2. Evaluation Metrics ---
def get_metrics(actual, predicted):
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
    return rmse, mape, smape

# --- 3. Main Content ---
with st.container():
    st.sidebar.header("Hyperparameter Tuning")
    use_default_hyperparameters = st.sidebar.checkbox("Use Default Hyperparameters", True)

    if use_default_hyperparameters:
        seq_length = 60
        hidden_layer_size = 200
        epochs = 75
        batch_size = 32
        learning_rate = 0.001
        st.sidebar.write("Using Default Hyperparameters:")
        st.sidebar.write(f"- Sequence Length: {seq_length}")
        st.sidebar.write(f"- Hidden Layer Size: {hidden_layer_size}")
        st.sidebar.write(f"- Epochs: {epochs}")
        st.sidebar.write(f"- Batch Size: {batch_size}")
        st.sidebar.write(f"- Learning Rate: {learning_rate}")
    else:
        st.sidebar.write("Select Custom Hyperparameters:")
        seq_length = st.sidebar.slider("Sequence Length", 10, 120, 60)
        hidden_layer_size = st.sidebar.slider("Hidden Layer Size", 50, 500, 200)
        epochs = st.sidebar.slider("Epochs", 25, 200, 75)
        batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)

    ticker_list = ["", "RELIANCE.NS", "TATAMOTORS.NS", "SBIN.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS"]
    selected_ticker = st.selectbox("Select Stock Ticker:", ticker_list)
    custom_ticker = st.text_input("Or Enter Custom Ticker:").upper()

    ticker = custom_ticker if custom_ticker else selected_ticker
    predict_button = st.button("Predict")

    if predict_button and ticker:
        with st.spinner(f"Running prediction for {ticker}... This may take a moment."):
            try:
                # --- 4. Setup and Configuration ---
                create_ticker_dirs(ticker)
                model_name = f"{ticker}_s{seq_length}_h{hidden_layer_size}_e{epochs}_b{batch_size}_lr{learning_rate}.pth"
                model_path = get_model_path(ticker, model_name)
                plot_name = model_name.replace('.pth', '.png')
                plot_path = get_plot_path(ticker, plot_name)

                # --- 5. Data Fetching and Preprocessing --
                @st.cache_data
                def load_data(ticker_symbol):
                    end_date = "2025-07-02"
                    start_date = end_date - timedelta(days=10 * 365) # 10 years of data
                    data = yf.download(ticker_symbol, start=start_date, end=end_date)
                    if data.empty:
                        return None
                    return data

                df = load_data(ticker)

                if df is None:
                    st.error(f"Could not download data for ticker '{ticker}'. Please check the ticker symbol.")
                else:
                    # Feature Engineering
                    df['SMA'] = df['Close'].rolling(window=20).mean()
                    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    df['Day_of_week'] = df.index.dayofweek
                    df['Month'] = df.index.month
                    df.dropna(inplace=True)

                    FEATURES = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA', 'EMA', 'RSI', 'Day_of_week', 'Month']
                    INPUT_SIZE = len(FEATURES)
                    
                    # --- 6. Train-Test Split ---
                    test_data_size = int(len(df) * 0.2)
                    train_data_df = df[:-test_data_size]
                    test_data_df = df[-test_data_size:]

                    # --- 7. Scaling ---
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    train_data_scaled = scaler.fit_transform(train_data_df[FEATURES].values)
                    test_data_scaled = scaler.transform(test_data_df[FEATURES].values)

                    close_price_scaler = MinMaxScaler(feature_range=(0, 1))
                    close_price_scaler.fit(train_data_df[['Close']])

                    def create_inout_sequences(input_data, seq_len):
                        inout_seq = []
                        L = len(input_data)
                        for i in range(L - seq_len):
                            train_seq = input_data[i:i + seq_len]
                            train_label = input_data[i + seq_len:i + seq_len + 1, 0]
                            inout_seq.append((train_seq, train_label))
                        return inout_seq

                    train_sequences = create_inout_sequences(train_data_scaled, seq_length)
                    test_sequences = create_inout_sequences(test_data_scaled, seq_length)

                    # --- 8. Model Training or Loading ---
                    model = LSTMModel(INPUT_SIZE, hidden_layer_size)

                    if not model_exists(model_path):
                        st.info(f"No pre-trained model found for {ticker}. Training a new model...")
                        train_data = TensorDataset(torch.FloatTensor([s[0] for s in train_sequences]), torch.FloatTensor([s[1] for s in train_sequences]))
                        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

                        loss_function = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i in range(epochs):
                            for seq, labels in train_loader:
                                optimizer.zero_grad()
                                y_pred = model(seq)
                                single_loss = loss_function(y_pred, labels)
                                single_loss.backward()
                                optimizer.step()
                            
                            progress_bar.progress((i + 1) / epochs)
                            status_text.text(f"Epoch {i+1}/{epochs} | Loss: {single_loss.item():.6f}")

                        torch.save(model.state_dict(), model_path)
                        st.success(f"Model trained and saved to {model_path}")
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        st.success(f"Loading pre-trained model for {ticker} from {model_path}")
                        model.load_state_dict(torch.load(model_path))

                    # --- 9. Prediction and Evaluation ---
                    model.eval()
                    
                    test_inputs = torch.FloatTensor([s[0] for s in test_sequences])
                    with torch.no_grad():
                        test_predictions_scaled = model(test_inputs).numpy()

                    actual_predictions = close_price_scaler.inverse_transform(test_predictions_scaled)
                    actuals = test_data_df['Close'].values[seq_length:]

                    rmse, mape, smape = get_metrics(actuals, actual_predictions.flatten())

                    full_scaled_data = scaler.transform(df[FEATURES].values)
                    last_sequence = torch.FloatTensor(full_scaled_data[-seq_length:]).view(1, seq_length, INPUT_SIZE)
                    with torch.no_grad():
                        next_day_prediction_scaled = model(last_sequence)
                    
                    next_day_prediction = close_price_scaler.inverse_transform(next_day_prediction_scaled.numpy())
                    
                    st.subheader(f"Predicted Close Price for next trading day:")
                    st.metric(label=f"{ticker}", value=f"₹{next_day_prediction[0][0]:.2f}")

                    st.subheader("Model Performance on Unseen Test Data")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"₹{rmse:.2f}")
                    col2.metric("MAPE", f"{mape:.2f}%")
                    col3.metric("SMAPE", f"{smape:.2f}%")

                    # --- 10. Plotting ---
                    plot_index = test_data_df.index[seq_length:]

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(plot_index, actuals, label='Actual Price', color='blue')
                    ax.plot(plot_index, actual_predictions, label='Predicted Price', color='red', linestyle='--')
                    ax.set_title(f'{ticker} Price Prediction on Test Set')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price (INR)')
                    ax.legend()
                    ax.grid(True)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    fig.savefig(plot_path)
                    st.info(f"Plot saved to {plot_path}")
                    plt.close(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("This could be due to an invalid ticker, no data available, or a model training issue. Please check the ticker and try again.")


# --- Footer ---
with st.container():
    st.write("---")
    st.header("Disclaimer")
    st.write("This is a tool for educational purposes and not financial advice. Always conduct your own thorough research before making any investment decisions.")
    st.write("Stock market predictions are inherently uncertain, and past performance is not indicative of future results.")
