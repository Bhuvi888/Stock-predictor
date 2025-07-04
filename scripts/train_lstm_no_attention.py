import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, LSTM, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from config import get_model_path, get_plot_path, create_ticker_dirs, PLOTS_DIR

def preprocess_data(data, seq_length):
    # Use more features
    features = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA', 'EMA', 'RSI', 'Day_of_week', 'Month']
    feature_data = data[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)

    # We need a separate scaler for the 'Close' price for inverse transforming
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit_transform(data[['Close']])

    def create_sequences(data, seq_length):
        xs = []
        ys = []
        for i in range(len(data)-seq_length):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length, 0] # We only want to predict the close price
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    x, y = create_sequences(scaled_data, seq_length)
    return x, y, scaler, close_scaler

def build_cnn_lstm_model(input_shape, hidden_layer_size, learning_rate):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(hidden_layer_size, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber(delta=1.0), metrics=['mae', 'mse'])
    return model

def train_and_save_model(ticker, seq_length=60, hidden_layer_size=96, epochs=75, batch_size=32, learning_rate=0.001):
    print(f"\n--- Processing {ticker} ---")
    create_ticker_dirs(ticker)

    # 1. Download Data
    start_date = "2015-01-01"
    end_date = datetime.now()
    data = yf.download(ticker, start=start_date, end=end_date)

    # Add Technical Indicators
    data['SMA'] = data['Close'].rolling(window=14).mean().fillna(0)
    data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean().fillna(0)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].fillna(0)

    # Add Calendar Features
    data['Day_of_week'] = data.index.dayofweek
    data['Month'] = data.index.month

    # 2. Preprocess Data
    x, y, scaler, close_scaler = preprocess_data(data, seq_length)

    # Split into training and validation sets
    train_size = int(len(y) * 0.8)
    val_size = int(len(y) * 0.1)

    x_train = x[0:train_size]
    y_train = y[0:train_size]

    x_val = x[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    x_test = x[train_size+val_size:len(x)]
    y_test = y[train_size+val_size:len(y)]

    # 3. Build CNN+LSTM Model
    model = build_cnn_lstm_model((x_train.shape[1], x_train.shape[2]), hidden_layer_size, learning_rate)

    # 4. Train the Model
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    model_filename_best = f"{ticker}_cnn_lstm_no_attention_best.keras"
    model_dir_best = get_model_path(ticker)
    os.makedirs(model_dir_best, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir_best, model_filename_best),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    print("Starting model training...")
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stop, checkpoint],
                        verbose=1)

    print("Training complete.")

    # 5. Evaluate the Model
    model = tf.keras.models.load_model(os.path.join(model_dir_best, model_filename_best))

    test_predictions = model.predict(x_test).flatten()

    y_pred_inv = close_scaler.inverse_transform(test_predictions.reshape(-1, 1))
    y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1))

    # 6. Print Evaluation Metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)

    print(f"Final Model RMSE: {rmse:.2f}")
    print(f"Final Model MAE: {mae:.2f}")

    # 7. Plot the results
    plt.figure(figsize=(12,6))
    plt.title(f'Stock Price Prediction for {ticker} (CNN-LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plot_index = data.index[seq_length + train_size + val_size:]
    plt.plot(plot_index, y_test_inv, label='Actual Price')
    plt.plot(plot_index, y_pred_inv, label='Predicted Price')
    plt.legend()
    plot_filename = f"{ticker}_cnn_lstm_no_attention.png"
    plot_path = os.path.join(PLOTS_DIR, ticker, plot_filename)
    plt.savefig(plot_path)
    plt.close() # Close the plot to free memory

    # 8. Save the model (final trained model, not necessarily the best)
    model_filename = f"{ticker}_cnn_lstm_no_attention.keras"
    model_dir = get_model_path(ticker)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path)
    print(f"Model saved as {model_path}")
    print(f"Prediction plot saved as {plot_path}")

    return model_path, close_scaler

def load_and_predict(ticker, model_path, seq_length, close_scaler, latest_data):
    model = tf.keras.models.load_model(model_path)

    # Preprocess the latest_data for prediction
    features = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA', 'EMA', 'RSI', 'Day_of_week', 'Month']
    feature_data = latest_data[features].values

    # Use the same scaler that was used for training
    # For prediction, we only need to scale the input data, not fit the scaler again
    # Assuming 'scaler' from training is available or re-initialized and fitted on historical data
    # For simplicity, let's assume 'latest_data' is already scaled or we have the original scaler
    # In a real app, you'd save/load the scaler or re-fit it on a larger dataset.
    # For this example, we'll create a temporary scaler for the input, but this is not ideal for production.
    # A better approach is to save the scaler used during training and load it here.
    temp_scaler = MinMaxScaler(feature_range=(0, 1))
    temp_scaler.fit(feature_data) # This is problematic if not the original scaler
    scaled_latest_data = temp_scaler.transform(feature_data)

    # Create sequence for prediction
    x_predict = np.array([scaled_latest_data[-seq_length:]])

    prediction = model.predict(x_predict).flatten()
    predicted_price = close_scaler.inverse_transform(prediction.reshape(-1, 1))
    return predicted_price[0][0]

if __name__ == "__main__":
    # Example Usage:
    TICKER_TO_TRAIN = "RELIANCE.NS"
    MODEL_PATH, CLOSE_SCALER = train_and_save_model(TICKER_TO_TRAIN)

    # Example of how you might use load_and_predict (requires fetching new data)
    # This part is illustrative and would need proper data handling in a real app
    print(f"\n--- Demonstrating Prediction for {TICKER_TO_TRAIN} ---")
    # Fetch some recent data to simulate new input for prediction
    # In a real app, this would be the latest available data
    recent_data = yf.download(TICKER_TO_TRAIN, period="60d") # Get enough data for a sequence

    # Add technical indicators and calendar features to recent_data
    recent_data['SMA'] = recent_data['Close'].rolling(window=14).mean().fillna(0)
    recent_data['EMA'] = recent_data['Close'].ewm(span=14, adjust=False).mean().fillna(0)
    delta = recent_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    recent_data['RSI'] = 100 - (100 / (1 + rs))
    recent_data['RSI'] = recent_data['RSI'].fillna(0)
    recent_data['Day_of_week'] = recent_data.index.dayofweek
    recent_data['Month'] = recent_data.index.month

    if len(recent_data) >= 60:
        predicted_value = load_and_predict(TICKER_TO_TRAIN, MODEL_PATH, 60, CLOSE_SCALER, recent_data)
        print(f"Predicted next day close price for {TICKER_TO_TRAIN}: {predicted_value:.2f}")
    else:
        print("Not enough recent data to make a prediction.")
