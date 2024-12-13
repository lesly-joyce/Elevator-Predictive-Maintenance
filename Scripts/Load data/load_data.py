import pandas as pd 
import os
from utils import natural_sort_key



# Dynamically load the .env file
current_directory = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_directory, "../config.env")
load_dotenv(env_path)


def read_multiple_datasets(file_paths):
    """ 
    Reads multiple datasets from a list of file paths.
    :param file_paths: list of file paths to csv file.
    :returns: List of DataFrames containing the datasets
    """
    dataFrames = []
    for file_name in file_paths:
        df = pd.read_csv(file_name)

        # Check if 'timeStamp' is in the dataframe, and handle missing column case
        if 'timeStamp' in df.columns:
            df["timeStamp"] = pd.to_datetime(df["timeStamp"])  # Convert timestamp to datetime
        else:
            raise KeyError(f"Column 'timeStamp' not found in file {file_name}")


        #Add label based on filename
        if "OK" in file_name:
            df['label'] = 0  #well functioning
        elif "KO" in file_name: 
            df['label'] = 1    #faulty
        dataFrames.append(df)

    return dataFrames

def load_data():
    """
    Load raw datasets from the directory specified in the config file.
    :returns: List of DataFrames
    """
    directory_path = os.getenv("DIRECTORY_PATH")
    raw_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.csv')], key=natural_sort_key)
    raw_file_paths = [os.path.join(directory_path, f) for f in raw_files]
    raw_dataFrames = read_multiple_datasets(raw_file_paths)
    return raw_dataFrames, raw_files

# Example usage:
raw_dataFrames, raw_files = load_data()
print(raw_dataFrames, raw_files)
