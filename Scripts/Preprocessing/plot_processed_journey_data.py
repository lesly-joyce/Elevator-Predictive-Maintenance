import pickle
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

def plot_filtered_normalized(dataFrames,file_names, journeys):
    """
    Plots filtered and normalized vibration data (X, Y, Z axes) for each dataset.
    Each dataset is plotted with its corresponding filename.
    """
    for i, (df, file_name, (start, start_timestamp, end, end_timestamp)) in enumerate(zip(dataFrames,file_names, journeys)):
        plt.figure(figsize=(14, 10))

        # Print timestamps for debugging
        print(f"Dataset {i+1} - {file_name}: {df.index}, Start Timestamp: {start_timestamp}, {df.index} End Timestamp: {end_timestamp}")
    
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['x'], label='Filtered and Normalized X', color='r')
        plt.xlabel('Index')
        plt.ylabel('Filtered Vibration X (Normalized)')
        plt.title(f'Dataset {i+1} - {file_name} - Filtered and Normalized Vibration X')
        plt.legend()
    
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['y'], label='Filtered and Normalized Y', color='g')
        plt.xlabel('Index')
        plt.ylabel('Filtered Vibration Y (Normalized)')
        plt.title(f'Dataset {i+1} - {file_name} - Filtered and Normalized Vibration Y')
        plt.legend()
    
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['z'], label='Filtered and Normalized Z', color='b')

        #Add vertical lines to indicate start and end of the lift journey
        if start is not None and end is not None:
            plt.axvline(start, color='m', linestyle='--', label='start of journey')
            plt.axvline(end, color='c', linestyle='--', label='End of journey')
        plt.xlabel('Index')
        plt.ylabel('Filtered Vibration Z (Normalized)')
        plt.title(f'Dataset {i+1} - {file_name} - Filtered and Normalized Vibration Z')
        plt.legend()
    
        plt.tight_layout()
        plt.show()

def visualize_processed_journeys():
    """
    Load processed data and detected journeys, then plot the data.
    """
    # Load preprocessed data
    processed_pickle_path = os.getenv("DIRECTORY_FILTERED_NORMALIZED_PICKLE")
    with open(processed_pickle_path, "rb") as f:
        normalized_dataFrames = pickle.load(f)

    # Load detected journeys
    journeys_pickle_path = os.getenv("DIRECTORY_JOURNEYS_PICKLE")
    with open(journeys_pickle_path, "rb") as f:
        journeys = pickle.load(f)
    
    # Load file names for reference
    file_names_path = os.getenv("DIRECTORY_FILE_NAMES")
    with open(file_names_path, "rb") as f:
        file_names = pickle.load(f)

    # Plot the data
    plot_filtered_normalized(normalized_dataFrames, file_names, journeys)

if __name__ == "__main__":
    visualize_processed_journeys()