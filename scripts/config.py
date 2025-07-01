import os

# Define the root directory of the stock-predictor application
# This path needs to be relative to where the training scripts are run from
# Assuming scripts are in genapp/scripts and stock-predictor is in genapp/stock-predictor
STOCK_PREDICTOR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stock-predictor'))

MODELS_DIR = os.path.join(STOCK_PREDICTOR_ROOT, "models")
PLOTS_DIR = os.path.join(STOCK_PREDICTOR_ROOT, "plots")

def get_model_path(ticker: str) -> str:
    """Returns the full path for a given model file within the stock-predictor structure."""
    return os.path.join(MODELS_DIR, ticker, f"{ticker}_lstm_final.pth")

def get_plot_path(ticker: str) -> str:
    """Returns the full path for a given plot file within the stock-predictor structure."""
    return os.path.join(PLOTS_DIR, ticker, f"{ticker}_prediction_plot.png")

def create_ticker_dirs(ticker: str):
    """Creates ticker-specific subdirectories within models and plots directories."""
    os.makedirs(os.path.join(MODELS_DIR, ticker), exist_ok=True)
    os.makedirs(os.path.join(PLOTS_DIR, ticker), exist_ok=True)
