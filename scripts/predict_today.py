import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

from config import get_model_path, create_ticker_dirs

# --- 1. Define LSTM Model (must be same as trained model) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=150, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# --- 2. Load Data and Preprocessing Setup ---
ticker = "TATAMOTORS.NS"
create_ticker_dirs(ticker)

# Download historical data to set up scalers and get recent data for prediction
# Fetch data up to yesterday to ensure we have enough history for indicators
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 5) # 5 years of data
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate features (must be consistent with training)
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

features = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA', 'EMA', 'RSI', 'Day_of_week', 'Month']

# Fit scalers on the entire historical data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data[features].values)

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(data[['Close']].values)

# --- 3. Prepare Input for Today's Prediction ---
seq_length = 45 # Must be same as training

# Get the last `seq_length` data points
last_sequence = data[features].tail(seq_length).values
scaled_last_sequence = scaler.transform(last_sequence)

# Convert to tensor
input_tensor = torch.from_numpy(scaled_last_sequence).float().unsqueeze(0) # Add batch dimension

# --- 4. Load Trained Model ---
input_size = len(features)
model = LSTMModel(input_size=input_size)
model.load_state_dict(torch.load(get_model_path(ticker)))
model.eval()

# --- 5. Make Prediction ---
with torch.no_grad():
    predicted_scaled_price = model(input_tensor).item()

# Create a dummy array for inverse transformation of the predicted close price
dummy_array = np.zeros((1, len(features)))
dummy_array[0, 0] = predicted_scaled_price # Place the predicted close price in the first column

predicted_price = close_scaler.inverse_transform(dummy_array[:, 0].reshape(-1, 1))[0][0]

print(f"Predicted closing price for today: {predicted_price:.2f}")