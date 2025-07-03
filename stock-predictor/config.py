import os

# --- Directory Paths ---
# Use os.path.join for creating platform-independent paths
# ROOT_DIR is the directory where app.py resides
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(ROOT_DIR, "..", "models")
PLOTS_DIR = os.path.join(ROOT_DIR, "..", "plots")

# --- Function to get model/plot paths ---
def get_model_path(ticker: str, model_name: str) -> str:
    """Returns the full path for a given model file."""
    return os.path.join(MODELS_DIR, ticker, model_name)

def get_plot_path(ticker: str, plot_name: str) -> str:
    """Returns the full path for a given plot file."""
    return os.path.join(PLOTS_DIR, ticker, plot_name)

def model_exists(model_path: str) -> bool:
    """Checks if a model file exists at the given path."""
    return os.path.exists(model_path)

def plot_exists(plot_path: str) -> bool:
    """Checks if a plot file exists at the given path."""
    return os.path.exists(plot_path)

# --- Create base directories if they don't exist ---
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def create_ticker_dirs(ticker: str):
    """Creates ticker-specific subdirectories within models and plots directories."""
    os.makedirs(os.path.join(MODELS_DIR, ticker), exist_ok=True)
    os.makedirs(os.path.join(PLOTS_DIR, ticker), exist_ok=True)
