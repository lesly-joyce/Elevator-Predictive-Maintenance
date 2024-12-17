import pandas as pd 
import os
import pickle
from utils import natural_sort_key
from dotenv import load_dotenv

# Dynamically load the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

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

    #Save the data to a pickle file for reuse
    directory_pickle = os.getenv("DIRECTORY_PICKLE")  #Load the path from the .env file
    file_names_path = os.getenv("DIRECTORY_FILE_NAMES")
    if not directory_pickle:
        raise ValueError("DIRECTORY_PICKLE environment variable is not set.")
    
    with open(directory_pickle, "wb") as f: 
        pickle.dump([df.copy() for df in raw_dataFrames],  f)

    with open(file_names_path, "wb") as f:
        pickle.dump(raw_files, f)
    
    print(f"Raw data and filenames saved: {directory_pickle}, {file_names_path}")
    return raw_dataFrames, raw_files

# Example usage:
raw_dataFrames, raw_files = load_data()
print(raw_dataFrames[0])
print(type(raw_dataFrames[0]))

