import os
import subprocess
import re

# Define the hyperparameter search space
SEQ_LENGTHS = [30, 45, 60]
HIDDEN_LAYER_SIZES = [50, 100, 150]
EPOCHS_LIST = [50, 75]
BATCH_SIZES = [32, 64]
LEARNING_RATES = [0.001, 0.01]

script_path = os.path.join(os.path.dirname(__file__), 'train_lstm.py')

# Initialize best tracking variables
best_rmse = float('inf')
best_model_path = None
best_plot_path = None

print("Starting hyperparameter tuning...")

for seq_length in SEQ_LENGTHS:
    for hidden_size in HIDDEN_LAYER_SIZES:
        for epochs in EPOCHS_LIST:
            for batch_size in BATCH_SIZES:
                for lr in LEARNING_RATES:
                    print(f"\n--- Running with: SEQ_LENGTH={seq_length}, HIDDEN_SIZE={hidden_size}, EPOCHS={epochs}, BATCH_SIZE={batch_size}, LR={lr} ---")
                    
                    # Construct expected filenames for the current run
                    ticker = "TATAMOTORS.NS" # Assuming TICKER is constant as in train_lstm.py
                    model_filename = f"{ticker}_s{seq_length}_h{hidden_size}_e{epochs}_b{batch_size}_lr{lr}.pth"
                    plot_filename = f"{ticker}_s{seq_length}_h{hidden_size}_e{epochs}_b{batch_size}_lr{lr}.png"
                    
                    # Assuming models are saved in stock-predictor/models/TICKER/ and plots in stock-predictor/plots/TICKER/
                    # Need to derive the actual paths based on the project structure
                    # This assumes the script is run from the genapp directory
                    current_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stock-predictor', 'models', ticker)
                    current_plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stock-predictor', 'plots', ticker)

                    current_model_path = os.path.join(current_model_dir, model_filename)
                    current_plot_path = os.path.join(current_plot_dir, plot_filename)

                    command = [
                        "python",
                        script_path,
                        f"--seq_length={seq_length}",
                        f"--hidden_layer_size={hidden_size}",
                        f"--epochs={epochs}",
                        f"--batch_size={batch_size}",
                        f"--learning_rate={lr}"
                    ]
                    try:
                        result = subprocess.run(command, capture_output=True, text=True, check=True)
                        print(result.stdout)
                        if result.stderr:
                            print("Error Output:", result.stderr)
                        
                        # Parse RMSE from stdout
                        rmse_match = re.search(r"FINAL_RMSE:(\d+\.\d+)", result.stdout)
                        if rmse_match:
                            current_rmse = float(rmse_match.group(1))
                            print(f"  Current RMSE: {current_rmse:.4f}")

                            if current_rmse < best_rmse:
                                print(f"  *** New best RMSE found: {current_rmse:.4f} ***")
                                # Delete previous best model and plot if they exist
                                if best_model_path and os.path.exists(best_model_path):
                                    os.remove(best_model_path)
                                    print(f"  Deleted old best model: {best_model_path}")
                                if best_plot_path and os.path.exists(best_plot_path):
                                    os.remove(best_plot_path)
                                    print(f"  Deleted old best plot: {best_plot_path}")
                                
                                # Update best tracking variables
                                best_rmse = current_rmse
                                best_model_path = current_model_path
                                best_plot_path = current_plot_path
                            else:
                                print(f"  Current RMSE ({current_rmse:.4f}) is not better than best RMSE ({best_rmse:.4f}). Deleting files.")
                                # Delete current model and plot if not better
                                if os.path.exists(current_model_path):
                                    os.remove(current_model_path)
                                    print(f"  Deleted: {current_model_path}")
                                if os.path.exists(current_plot_path):
                                    os.remove(current_plot_path)
                                    print(f"  Deleted: {current_plot_path}")
                        else:
                            print("  Could not find RMSE in output. Keeping files for inspection.")

                    except subprocess.CalledProcessError as e:
                        print(f"Error running training script: {e}")
                        print(f"Stdout: {e.stdout}")
                        print(f"Stderr: {e.stderr}")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")

print("\nHyperparameter tuning complete.")
print(f"Final Best RMSE: {best_rmse:.4f}")
print(f"Best Model Path: {best_model_path}")
print(f"Best Plot Path: {best_plot_path}")