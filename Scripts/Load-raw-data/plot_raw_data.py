import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from dotenv import load_dotenv

# Dynamically load the .env file
load_dotenv("config.env")

def plot_raw_data():

    with open(os.getenv("DIRECTORY_PICKLE"), "rb") as f:
        dataFrames, file_names= pickle.load(f)

    """
    Plot raw sensor data for visualization.
    :param dataFrames: List of pandas DataFrames containing the raw data.
    :param file_names: List of filenames corresponding to the DataFrames.
    """
    for i, (df, file_name) in enumerate(zip(dataFrames,file_names)):

        # Ensure 'timeStamp' column exists and is in datetime format
        if "timeStamp" not in df.columns:
            raise KeyError(f"'timeStamp' column not found in dataset {file_name}")
        
        # Ensure that 'Timestamp' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df["timeStamp"]):
            df["timeStamp"] = pd.to_datetime(df["timeStamp"])
    
        #print(df['Timestamp_ms'] )

        plt.figure(figsize=(14, 10))
        # Print timestamps for debugging
        print(f"Dataset {i+1} - {file_name}")

        plt.subplot(3, 1, 1)
        plt.plot(df['timeStamp'], df['x'], color='r', label = 'Asse x')
        plt.xlabel('Timestamp')
        plt.ylabel('Sensor Readings')
        plt.title(f'Dataset {i+1} - {file_name} - Raw Sensor data Over Time')
        plt.xticks(rotation=45)
        plt.legend()
   
    
        plt.subplot(3, 1, 2)
        plt.plot(df['timeStamp'], df['y'], color='g', label = 'Asse y')
        plt.xlabel('Timestamp')
        plt.ylabel('Sensor Readings')
        plt.title(f'Dataset {i+1} - {file_name} - Raw Sensor data Over Time')
        plt.xticks(rotation=45)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(df['timeStamp'], df['z'], color='b', label = 'Asse z')
        plt.xlabel('Timestamp')
        plt.ylabel('Sensor Readings')
        plt.title(f'Dataset {i+1} - {file_name} - Raw Sensor data Over Time')
        plt.xticks(rotation=45)
        plt.legend()


        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()

plot_raw_data()