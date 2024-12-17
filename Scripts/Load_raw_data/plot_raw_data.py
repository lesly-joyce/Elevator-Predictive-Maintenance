import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

def load_pickled_data(data_pickle_path, file_names_pickle_path):
    """
    Load DataFrames and filenames from pickle files.
    :param data_pickle_path: Path to the data pickle file.
    :param file_names_pickle_path: Path to the file names pickle file.
    :return: Tuple of (dataFrames, file_names)
    """
    if not os.path.exists(data_pickle_path):
        raise FileNotFoundError(f"Data pickle file not found: {data_pickle_path}")
    
    if not os.path.exists(file_names_pickle_path):
        raise FileNotFoundError(f"File names pickle file not found: {file_names_pickle_path}")

    with open(data_pickle_path, "rb") as f:
        dataFrames = pickle.load(f)

    with open(file_names_pickle_path, "rb") as f:
        file_names = pickle.load(f)

    return dataFrames, file_names

def plot_raw_data(dataFrames, file_names):
    """
    Plot raw sensor data for visualization.
    :param dataFrames: List of pandas DataFrames containing the raw data.
    :param file_names: List of filenames corresponding to the DataFrames.
    """
    for i, (df, file_name) in enumerate(zip(dataFrames, file_names)):
        # Ensure 'timeStamp' column exists and is in datetime format
        if "timeStamp" not in df.columns:
            raise KeyError(f"'timeStamp' column not found in dataset {file_name}")
        
        # Ensure that 'timeStamp' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df["timeStamp"]):
            df["timeStamp"] = pd.to_datetime(df["timeStamp"])

        plt.figure(figsize=(14, 10))
        print(f"Dataset {i+1} - {file_name}")

        plt.subplot(3, 1, 1)
        plt.plot(df['timeStamp'], df['x'], color='r', label='Asse x')
        plt.xlabel('Timestamp')
        plt.ylabel('Sensor Readings')
        plt.title(f'Dataset {i+1} - {file_name} - Raw Sensor data Over Time')
        plt.xticks(rotation=45)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(df['timeStamp'], df['y'], color='g', label='Asse y')
        plt.xlabel('Timestamp')
        plt.ylabel('Sensor Readings')
        plt.title(f'Dataset {i+1} - {file_name} - Raw Sensor data Over Time')
        plt.xticks(rotation=45)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(df['timeStamp'], df['z'], color='b', label='Asse z')
        plt.xlabel('Timestamp')
        plt.ylabel('Sensor Readings')
        plt.title(f'Dataset {i+1} - {file_name} - Raw Sensor data Over Time')
        plt.xticks(rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()


# Load pickle paths from environment variables
data_pickle_path = os.getenv("DIRECTORY_PICKLE")
file_names_pickle_path = os.getenv("DIRECTORY_FILE_NAMES")

# Load the data from pickle files
dataFrames, file_names = load_pickled_data(data_pickle_path, file_names_pickle_path)

# Plot the raw data
plot_raw_data(dataFrames, file_names)
