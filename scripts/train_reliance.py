import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

from config import get_model_path, get_plot_path, create_ticker_dirs

# --- 0. Hyperparameters & Settings ---
TICKER = "RELIANCE.NS" # Changed to Reliance
START_DATE = "2015-01-01"
END_DATE = datetime.now() # Fetch latest data

# Model Hyperparameters
SEQ_LENGTH = 45
HIDDEN_LAYER_SIZE = 150
EPOCHS = 75
BATCH_SIZE = 64
LEARNING_RATE = 0.001

create_ticker_dirs(TICKER)

# --- 1. Download Data ---
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

# --- Add Technical Indicators ---
data['SMA'] = data['Close'].rolling(window=14).mean().fillna(0)
data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean().fillna(0)

# Calculate RSI manually
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))
data['RSI'] = data['RSI'].fillna(0)

# --- Add Calendar Features ---
data['Day_of_week'] = data.index.dayofweek
data['Month'] = data.index.month


# --- 2. Preprocess Data ---
features = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA', 'EMA', 'RSI', 'Day_of_week', 'Month']
feature_data = data[features].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(feature_data)

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit_transform(data[['Close']])

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

x, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split into training and validation sets
train_size = int(len(y) * 0.8)
test_size = len(y) - train_size

x_train = torch.from_numpy(x[0:train_size]).float()
y_train = torch.from_numpy(y[0:train_size]).float().view(-1, 1)

x_test = torch.from_numpy(x[train_size:len(x)]).float()
y_test = torch.from_numpy(y[train_size:len(y)]).float().view(-1, 1)

# --- 3. Build LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=len(features), hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. Train the Model ---
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

print("Starting Reliance model training...")
for i in range(EPOCHS):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    if (i+1)%5 == 0:
        print(f'epoch: {i+1:3} loss: {single_loss.item():10.8f}')

print("Training complete.")

# --- 5. Evaluate the Model ---
model.eval()
test_predictions = []
with torch.no_grad():
    for seq in x_test:
        test_predictions.append(model(seq.unsqueeze(0)).item())

y_pred_inv = close_scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
y_test_inv = close_scaler.inverse_transform(y_test.numpy())

# --- 6. Print Evaluation Metrics ---
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)

print(f"Final Model RMSE: {rmse:.2f}")
print(f"Final Model MAE: {mae:.2f}")

# --- 7. Plot the results ---
plt.figure(figsize=(12,6))
plt.title('Reliance Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plot_index = data.index[train_size+SEQ_LENGTH:]
plt.plot(plot_index, y_test_inv, label='Actual Price')
plt.plot(plot_index, y_pred_inv, label='Predicted Price')
plt.legend()
plt.savefig(get_plot_path(TICKER))

# --- 8. Save the model ---
torch.save(model.state_dict(), get_model_path(TICKER))
print(f"Model saved as {get_model_path(TICKER)}")
print(f"Prediction plot saved as {get_plot_path(TICKER)}")
