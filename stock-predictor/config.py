import os

# --- Directory Paths ---
# Use os.path.join for creating platform-independent paths
# ROOT_DIR is the directory where app.py resides
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

MODELS_DIR = os.path.join(ROOT_DIR, "models")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

# --- Function to get model/plot paths ---
def get_model_path(ticker: str) -> str:
    """Returns the full path for a given model file."""
    return os.path.join(MODELS_DIR, ticker, f"{ticker}_lstm_final.pth")

def get_plot_path(ticker: str) -> str:
    """Returns the full path for a given plot file."""
    return os.path.join(PLOTS_DIR, ticker, f"{ticker}_prediction_plot.png")

def model_exists(ticker: str) -> bool:
    """Checks if a model file exists for a given ticker."""
    return os.path.exists(get_model_path(ticker))

def plot_exists(ticker: str) -> bool:
    """Checks if a plot file exists for a given ticker."""
    return os.path.exists(get_plot_path(ticker))

# --- Create base directories if they don't exist ---
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def create_ticker_dirs(ticker: str):
    """Creates ticker-specific subdirectories within models and plots directories."""
    os.makedirs(os.path.join(MODELS_DIR, ticker), exist_ok=True)
    os.makedirs(os.path.join(PLOTS_DIR, ticker), exist_ok=True)
