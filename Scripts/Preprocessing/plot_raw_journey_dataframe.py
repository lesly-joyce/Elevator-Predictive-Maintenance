import os
import sys
import pickle
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Scripts.Load_raw_data.plot_raw_data import plot_raw_data

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

def plot_extracted_journeys():
    """
    Load extracted journey data and plot each journey.
    """
    # Load extracted journey data
    journey_data_path = os.getenv("DIRECTORY_JOURNEYS_DATA")
    # Load file names for reference
    file_names_path = os.getenv("DIRECTORY_FILE_NAMES")

    if not os.path.exists(journey_data_path):
        raise FileNotFoundError(f"Journey data file not found: {journey_data_path}")
    if not os.path.exists(file_names_path):
        raise FileNotFoundError(f"File names data file not found: {file_names_path}")

    with open(journey_data_path, "rb") as f:
        raw_journey_dataFrames = pickle.load(f)

    with open(file_names_path, "rb") as f:
        raw_files_base = pickle.load(f)

    # Plot the extracted journey data
    print("Plotting extracted raw lift journey data...")
    plot_raw_data(raw_journey_dataFrames, raw_files_base)

# Example usage:
plot_extracted_journeys()
