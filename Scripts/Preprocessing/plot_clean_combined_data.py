import os
import pickle
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

def plot_all_filtered_normalized(dataFrames, file_names):
    """
    Plots filtered and normalized vibration data (X, Y, Z axes) for each dataset.
    Each dataset is plotted with its corresponding filename.
    """
    for i, (df, file_name) in enumerate(zip(dataFrames, file_names)):
        plt.figure(figsize=(14, 10))
    
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
        plt.xlabel('Index')
        plt.ylabel('Filtered Vibration Z (Normalized)')
        plt.title(f'Dataset {i+1} - {file_name} - Filtered and Normalized Vibration Z')
        plt.legend()
    
        plt.tight_layout()
        plt.show()

def visualize_clean_combined_data():
    """
    Load cleaned combined data and plot all datasets.
    """
    # Load cleaned data
    cleaned_data_path = os.getenv("DIRECTORY_CLEANED_DATA")
    combined_file_names_path = os.getenv("DIRECTORY_COMBINED_FILE_NAMES")

    if not cleaned_data_path or not combined_file_names_path:
        raise ValueError("DIRECTORY_CLEANED_DATA or DIRECTORY_COMBINED_FILE_NAMES environment variable is not set.")
    
    # Load cleaned combined data
    with open(cleaned_data_path, "rb") as f:
        cleaned_dataFrames = pickle.load(f)
    
    # Load file names
    with open(combined_file_names_path, "rb") as f:
        combined_file_names = pickle.load(f)

    print(f"Loaded {len(cleaned_dataFrames)} cleaned DataFrames for visualization.")

    # Plot all datasets
    plot_all_filtered_normalized(cleaned_dataFrames, combined_file_names)

# Example usage
if __name__ == "__main__":
    visualize_clean_combined_data()
