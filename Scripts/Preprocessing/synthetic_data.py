import os
import pickle
import random
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

def monte_carlo_generate_synthetic_data(raw_journey_dataFrames, num_rounds):
    """
    Generate synthetic data for each raw journey dataFrame using Kernel Density Estimation (KDE).

    :param raw_journey_dataFrames: List of DataFrames containing the raw journey data.
    :param num_rounds: Number of times to generate synthetic data for each journey.
    :return: List of synthetic DataFrames.
    """
    synthetic_dataFrames = []

    for _ in range(num_rounds):
        for df in raw_journey_dataFrames:
            num_samples = len(df)

            # Create an empty DataFrame to store synthetic data
            synthetic_df = pd.DataFrame()

            # Loop over each axis (x, y, z)
            for column in ['x', 'y', 'z']:
                # Estimate the Probability Density Function using Kernel Density Estimation (KDE)
                kde = gaussian_kde(df[column])
                
                # Generate synthetic data points using KDE sampling
                synthetic_data = kde.resample(num_samples).flatten()

                # Add the synthetic data to the new DataFrame
                synthetic_df[column] = synthetic_data

            # Copy the timestamp from the original data
            synthetic_df['timeStamp'] = df['timeStamp'].values

            # Copy the other columns
            columns_to_copy = ['entity_id', 'name', 'tripNum', 'tripDuration', 'tripDirection', 'label']
            for col in columns_to_copy:
                if col in df.columns:
                    synthetic_df[col] = df[col].values
                else:
                    synthetic_df[col] = None

            # Append the new synthetic DataFrame to the list
            synthetic_dataFrames.append(synthetic_df)

    return synthetic_dataFrames

def generate_and_combine_synthetic_data():
    """
    Generate synthetic data and combine it with original data, then save all data for further use.
    """
    # Load original journey data
    journey_data_path = os.getenv("DIRECTORY_JOURNEYS_DATA")
    with open(journey_data_path, "rb") as f:
        raw_journey_dataFrames = pickle.load(f)

    # Load the number of rounds from the .env file
    num_rounds = int(os.getenv("NUM_ROUNDS")) 

    # Generate synthetic data
    synthetic_dataFrames = monte_carlo_generate_synthetic_data(raw_journey_dataFrames, num_rounds=num_rounds)

    # Combine original and synthetic data
    all_dataFrames = raw_journey_dataFrames + synthetic_dataFrames
    random.shuffle(all_dataFrames)

    # Update file names to include synthetic data
    synthetic_file_names = [f"synthetic_data_{i}" for i in range(len(synthetic_dataFrames))]
    file_names_path = os.getenv("DIRECTORY_FILE_NAMES")
    with open(file_names_path, "rb") as f:
        raw_files_base = pickle.load(f)
    all_file_names = raw_files_base + synthetic_file_names

    # Save combined data and file names
    combined_data_path = os.getenv("DIRECTORY_COMBINED_DATA")
    with open(combined_data_path, "wb") as f:
        pickle.dump(all_dataFrames, f)

    combined_file_names_path = os.getenv("DIRECTORY_COMBINED_FILE_NAMES")
    with open(combined_file_names_path, "wb") as f:
        pickle.dump(all_file_names, f)

    print(f"Synthetic dataframes generated: {len(synthetic_dataFrames)}")
    print(f"Synthetic data generated and combined with original data.")
    print(f"Total DataFrames: {len(all_dataFrames)}")
    print(f"Combined data saved to: {combined_data_path}")
    print(f"File names saved to: {combined_file_names_path}")

# Example usage
generate_and_combine_synthetic_data()
