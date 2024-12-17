import os
import pickle
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Scripts.Load_raw_data.plot_raw_data import plot_raw_data

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

def plot_combined_data():
    """
    Load and plot combined raw data (original + synthetic).
    """
    # Load combined data
    combined_data_path = os.getenv("DIRECTORY_COMBINED_DATA")
    combined_file_names_path = os.getenv("DIRECTORY_COMBINED_FILE_NAMES")

    if not os.path.exists(combined_data_path):
        raise FileNotFoundError(f"Combined data file not found: {combined_data_path}")
    if not os.path.exists(combined_file_names_path):
        raise FileNotFoundError(f"Combined file names file not found: {combined_file_names_path}")

    with open(combined_data_path, "rb") as f:
        combined_dataFrames = pickle.load(f)

    with open(combined_file_names_path, "rb") as f:
        combined_file_names = pickle.load(f)

    # Plot combined data using the existing plot_raw_data function
    print("Plotting combined raw data (original + synthetic)...")
    plot_raw_data(combined_dataFrames, combined_file_names)

# Example usage
plot_combined_data()
