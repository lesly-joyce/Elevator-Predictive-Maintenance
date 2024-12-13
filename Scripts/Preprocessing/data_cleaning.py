import os
import pickle
from scipy.signal import butter, filtfilt

from dotenv import load_dotenv

# Dynamically load the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)


def low_pass_filter(data, cutoff_frequency, sampling_rate, order=5):
    """
    Applies a low-pass Butterworth filter to the data.
    :param data: Array-like, the data to be filtered.
    :param cutoff_frequency: Float, the cutoff frequency of the filter in Hz.
    :param sampling_rate: Float, the sampling rate of the data in Hz.
    :param order: Int, the order of the filter (default is 5).
    :return: Array-like, the filtered data.
    """
    nyquist_frequency = 0.5 * sampling_rate
    normal_cutt_off = cutoff_frequency / nyquist_frequency
    b, a = butter(order, normal_cutt_off, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def apply_filter_to_dataset(dataFrames, cutoff_frequency, sampling_rate):
    """
    Apply low-pass filter to a list of DataFrames.
    :param dataFrames: List of DataFrames.
    :param cutoff_frequency: Float, cutoff frequency in Hz.
    :param sampling_rate: Float, sampling rate in Hz.
    :return: List of filtered DataFrames.
    """
    filtered_dataFrames = []
    for df in dataFrames:
        for axis in ['x', 'y', 'z']:
            df[axis] = low_pass_filter(df[axis], cutoff_frequency, sampling_rate)
        filtered_dataFrames.append(df)
    return filtered_dataFrames

def normalize_datasets(dataFrames):
    """
    Normalize the filtered vibration data between -1 and 1.
    :param dataFrames: List of filtered DataFrames.
    :return: List of normalized DataFrames.
    """
    normalize_dataFrames = []
    for df in dataFrames:
        for axis in ['x', 'y', 'z']:
            min_val = df[axis].min()
            max_val = df[axis].max()
            df[axis] = 2 * (df[axis] - min_val) / (max_val - min_val) - 1
        normalize_dataFrames.append(df)
    return normalize_dataFrames

def preprocess_data():
    """
    Load raw data, apply filtering and normalization, and save processed data.
    """
    # Load raw data from pickle
    pickle_path = os.getenv("DIRECTORY_PICKLE")
    with open(pickle_path, "rb") as f:
        raw_dataFrames, raw_files = pickle.load(f)

    # Apply filtering
    filtered_dataFrames = apply_filter_to_dataset(raw_dataFrames, cutoff_frequency=0.3, sampling_rate=30)

    # Apply normalization
    normalized_dataFrames = normalize_datasets(filtered_dataFrames)

    # Save processed data to pickle
    processed_pickle_path = os.getenv("DIRECTORY_FILTERED_NORMALIZED_PICKLE")
    with open(processed_pickle_path, "wb") as f:
        pickle.dump(normalized_dataFrames, f)

    print("Data preprocessing complete. Processed data saved for further use.")
    
# Example usage:
preprocess_data()
