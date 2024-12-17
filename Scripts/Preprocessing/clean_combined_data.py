import os
import pickle
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Scripts.Preprocessing.data_cleaning import apply_filter_to_dataset, normalize_datasets

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

def clean_combined_data():
    """
    Apply filters and normalization to the combined dataset.
    Save the cleaned data for further use.
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

    # Apply filters and normalization
    cutoff_frequency = float(os.getenv("CUTOFF_FREQUENCY_2", 10))
    sampling_rate = float(os.getenv("SAMPLING_RATE_2", 100))
    
    print("Cleaning combined data...")
    filtered_normalized_dataFrames = []
    for idx, df in enumerate(combined_dataFrames):
        print(f"Processing DataFrame {idx+1} of {len(combined_dataFrames)}")
        filtered_df = apply_filter_to_dataset([df], cutoff_frequency, sampling_rate)[0]
        normalized_df = normalize_datasets([filtered_df])[0]
        filtered_normalized_dataFrames.append(normalized_df)

    # Save the cleaned data
    cleaned_data_path = os.getenv("DIRECTORY_CLEANED_DATA")
    with open(cleaned_data_path, "wb") as f:
        pickle.dump(filtered_normalized_dataFrames, f)

    print(f"Cleaned data saved to: {cleaned_data_path}")

# Example usage
clean_combined_data()
