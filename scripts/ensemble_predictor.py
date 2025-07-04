import sys
import os
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Evaluation Metrics ---
def get_metrics(actual, predicted):
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
    return rmse, mape, smape

# --- Main Ensemble Prediction Logic ---
def run_ensemble_prediction(ticker="RELIANCE.NS"):
    print(f"\n--- Running Ensemble Prediction for {ticker} ---")

    # --- Data Fetching and Preprocessing ---
    def load_data(ticker_symbol):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10 * 365) # 10 years of data
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        return data

    df = load_data(ticker)

    if df is None:
        print(f"Error: Could not download data for ticker '{ticker}'.")
        return
    
    # Flatten multi-level columns if yfinance returned them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Feature Engineering
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (100 + rs))
    df['Day_of_week'] = df.index.dayofweek
    df['Month'] = df.index.month

    # --- Add Lagged Features ---
    lookback_period = 5 # Use past 5 days' Close prices as features
    for i in range(1, lookback_period + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    df.dropna(inplace=True)

    FEATURES = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA', 'EMA', 'RSI', 'Day_of_week', 'Month']
    # Add lagged features to the FEATURES list
    for i in range(1, lookback_period + 1):
        FEATURES.append(f'Close_Lag_{i}')

    TARGET = 'Close'

    # Prepare data for traditional ML models (X, y)
    X = df[FEATURES].drop(columns=[TARGET])
    y = df[TARGET]

    # Ensure all column names are strings (redundant after flattening df.columns, but safe)
    X.columns = X.columns.astype(str)

    # --- Train-Test Split ---
    test_size = 0.2
    split_index = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # --- Scaling ---
    # Scale features (X)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale target (y) - important for some models, but for tree-based models, often not strictly necessary
    # However, for consistency and potential future use with other models, we'll scale it.
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # --- Model Training ---
    print("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train_scaled.flatten())

    print("Training GradientBoostingRegressor...")
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train_scaled, y_train_scaled.flatten())

    # --- Prediction on Test Set ---
    rf_predictions_scaled = rf_model.predict(X_test_scaled)
    gb_predictions_scaled = gb_model.predict(X_test_scaled)

    # Combine predictions (simple averaging)
    ensemble_predictions_scaled = (rf_predictions_scaled + gb_predictions_scaled) / 2

    # Inverse transform predictions to original scale
    ensemble_predictions = scaler_y.inverse_transform(ensemble_predictions_scaled.reshape(-1, 1)).flatten()
    actuals_test = y_test.values

    # --- Evaluate Ensemble Model ---
    rmse, mape, smape = get_metrics(actuals_test, ensemble_predictions)

    print(f"\nEnsemble Model Performance on Test Data for {ticker}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  SMAPE: {smape:.2f}%")

    # --- Predict Next Day's Price ---
    # To predict the next day, we need to construct the input features for that day.
    # This involves getting the last 'lookback_period' actual close prices
    # and the latest values for other features.

    # Create a dictionary to hold the next day's input features
    next_day_features_values = {}

    # Populate non-lagged features from the last row of df
    for feature in FEATURES:
        if not feature.startswith('Close_Lag_'):
            next_day_features_values[feature] = df[feature].iloc[-1]

    # Populate lagged features
    # Close_Lag_1 for the next day is the last actual Close price
    next_day_features_values['Close_Lag_1'] = df['Close'].iloc[-1]
    for i in range(2, lookback_period + 1):
        # Close_Lag_i for the next day is Close_Lag_{i-1} from the last row of df
        next_day_features_values[f'Close_Lag_{i}'] = df[f'Close_Lag_{i-1}'].iloc[-1]

    # Ensure the order of columns matches X.columns for scaling
    next_day_input_list = [next_day_features_values[col] for col in X.columns]
    next_day_input_df = pd.DataFrame([next_day_input_list], columns=X.columns)

    # Scale the next day's input features
    next_day_input_scaled = scaler_X.transform(next_day_input_df)

    rf_next_day_pred_scaled = rf_model.predict(next_day_input_scaled)
    gb_next_day_pred_scaled = gb_model.predict(next_day_input_scaled)

    ensemble_next_day_pred_scaled = (rf_next_day_pred_scaled + gb_next_day_pred_scaled) / 2
    next_day_prediction = scaler_y.inverse_transform(ensemble_next_day_pred_scaled.reshape(-1, 1)).flatten()[0]

    print(f"\nPredicted Close Price for next trading day ({ticker}): {next_day_prediction:.2f}")

    # --- Plotting ---
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stock-predictor", "plots", "generic")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{ticker}_ensemble_prediction_plot.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, actuals_test, label='Actual Price', color='blue')
    ax.plot(y_test.index, ensemble_predictions, label='Predicted Price', color='red', linestyle='--')
    ax.set_title(f'{ticker} Ensemble Model Prediction on Test Set')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (INR)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_ensemble_prediction(sys.argv[1].upper())
    else:
        run_ensemble_prediction()