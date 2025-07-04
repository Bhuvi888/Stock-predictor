import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, LSTM, Dense, BatchNormalization, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

import os
from config import get_model_path, get_plot_path, create_ticker_dirs, PLOTS_DIR

# --- 0. Hyperparameters & Settings ---
TICKERS = ["RELIANCE.NS", "SBIN.NS", "TATAMOTORS.NS"]
START_DATE = "2015-01-01"
END_DATE = datetime.now() # Fetch latest data

# Model Hyperparameters
SEQ_LENGTH = 60
HIDDEN_LAYER_SIZE = 96  # Adjusted to match the notebook's LSTM layer
EPOCHS = 75
BATCH_SIZE = 32
LEARNING_RATE = 0.001

for TICKER in TICKERS:
    print(f"\n--- Processing {TICKER} ---")
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

    x, y = create_sequences(scaled_data, SEQ_LENGTH)

    # Split into training and validation sets
    train_size = int(len(y) * 0.8)
    val_size = int(len(y) * 0.1) # 10% for validation
    test_size = len(y) - train_size - val_size

    x_train = x[0:train_size]
    y_train = y[0:train_size]

    x_val = x[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    x_test = x[train_size+val_size:len(x)]
    y_test = y[train_size+val_size:len(y)]

    # --- 3. Build CNN+LSTM+Attention Model (Keras) ---
    class Attention(Layer):
        def __init__(self, **kwargs):
            super(Attention, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                     initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                     initializer="zeros", trainable=True)
            super().build(input_shape)

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)      # (batch_size, time_steps, 1)
            a = K.softmax(e, axis=1)                   # (batch_size, time_steps, 1)
            output = K.sum(x * a, axis=1)              # (batch_size, hidden_size)
            return output
        
        def get_config(self):
            base_config = super(Attention, self).get_config()
            return base_config

    def build_cnn_lstm_attention_model(input_shape):
        inputs = Input(shape=input_shape)

        x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Calculate the new sequence length after pooling operations
        # Original seq_len = 60
        # After first MaxPooling1D(pool_size=2): 60 / 2 = 30
        # After second MaxPooling1D(pool_size=2): 30 / 2 = 15
        # The LSTM input shape will be (None, 15, 128)
        # The Attention layer expects (time_steps, feature_dim)
        # So, step_dim for Attention will be 15

        x = LSTM(HIDDEN_LAYER_SIZE, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(x)
        x = Attention()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=Huber(delta=1.0), metrics=['mae', 'mse'])
        return model

    model = build_cnn_lstm_attention_model((x_train.shape[1], x_train.shape[2]))

    # --- 4. Train the Model ---
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    model_filename_best = f"{TICKER}_cnn_lstm_attention_best.keras"
    model_dir_best = get_model_path(TICKER)
    os.makedirs(model_dir_best, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir_best, model_filename_best),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    print("Starting model training...")
    history = model.fit(x_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stop, checkpoint],
                        verbose=1)

    print("Training complete.")

    # --- 5. Evaluate the Model ---
    # Load the best model weights for evaluation
    model = tf.keras.models.load_model(os.path.join(model_dir_best, model_filename_best), custom_objects={'Attention': Attention})

    test_predictions = model.predict(x_test).flatten()

    # Inverse transform the predictions
    y_pred_inv = close_scaler.inverse_transform(test_predictions.reshape(-1, 1))
    y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1))


    # --- 6. Print Evaluation Metrics ---
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)

    print(f"Final Model RMSE: {rmse:.2f}")
    print(f"Final Model MAE: {mae:.2f}")


    # --- 7. Plot the results ---
    plt.figure(figsize=(12,6))
    plt.title(f'Stock Price Prediction for {TICKER} (CNN-LSTM-Attention)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    # We need to adjust the index for plotting
    plot_index = data.index[SEQ_LENGTH + train_size + val_size:]
    plt.plot(plot_index, y_test_inv, label='Actual Price')
    plt.plot(plot_index, y_pred_inv, label='Predicted Price')
    plt.legend()
    plot_filename = f"{TICKER}_cnn_lstm_attention.png"
    plot_path = os.path.join(PLOTS_DIR, TICKER, plot_filename)
    plt.savefig(plot_path)
    plt.close() # Close the plot to free memory

    # --- 8. Save the model (final trained model, not necessarily the best) ---
    model_filename = f"{TICKER}_cnn_lstm_attention.keras"
    model_dir = get_model_path(TICKER)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path)
    print(f"Model saved as {model_path}")
    print(f"Prediction plot saved as {plot_path}")