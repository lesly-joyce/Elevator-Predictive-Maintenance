import pickle
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

def calculate_derivative(dataFrames):
    """
    Calculate numerical derivatives for each axis (x, y, z) in the dataset.
    :param dataFrames: List of normalized DataFrames.
    :return: List of tuples containing derivatives (der_x, der_y, der_z).
    """
    derivatives = []
    for df in dataFrames:
        der_x = np.gradient(df['x'])
        der_y = np.gradient(df['y'])
        der_z = np.gradient(df['z'])
        derivatives.append((der_x, der_y, der_z))
    return derivatives

def find_lift_journey(dataFrames, derivatives, upperbound=0.005, lowerbound=-0.005, min_duration=320):
    """
    Detect lift journeys based on stability in the derivative.
    :param dataFrames: List of DataFrames.
    :param derivatives: List of tuples containing derivatives.
    :param upperbound: Upper bound for stable regions.
    :param lowerbound: Lower bound for stable regions.
    :param min_duration: Minimum duration of a journey in data points.
    :return: List of detected journeys.
    """
    journeys = []
    for df, (der_x, der_y, der_z) in zip(dataFrames, derivatives):
        count = 0
        current_start = 0
        start, start_timestamp, end, end_timestamp = None, None, None, None

        for index, value in enumerate(der_z):
            if lowerbound <= value <= upperbound:
                if current_start == 0:
                    current_start = index
                if count >= min_duration:
                    start = current_start
                    start_timestamp = df['timeStamp'][start]
                    end = index - 1
                    end_timestamp = df['timeStamp'][end]
                count += 1
            else:
                count = 0
                current_start = 0
        if start is None or end is None:
            print("No journey detected in this dataset. Adjust the thresholds or duration.")
        journeys.append((start, start_timestamp, end, end_timestamp))
    return journeys

def detect_lift_journeys():
    """
    Load filtered and normalized data, calculate derivatives, and detect lift journeys.
    Save detected journeys to a pickle file for reuse.
    """
    # Load preprocessed data
    processed_pickle_path = os.getenv("DIRECTORY_FILTERED_NORMALIZED_PICKLE")
    with open(processed_pickle_path, "rb") as f:
        normalized_dataFrames = pickle.load(f)

    # Calculate derivatives
    derivatives = calculate_derivative(normalized_dataFrames)

    # Detect lift journeys
    journeys = find_lift_journey(normalized_dataFrames, derivatives)

    # Save detected journeys to pickle
    journeys_pickle_path = os.getenv("DIRECTORY_JOURNEYS_PICKLE")
    with open(journeys_pickle_path, "wb") as f:
        pickle.dump(journeys, f)

    print("Lift journeys detected and saved for further use.", journeys)

# Example usage:
detect_lift_journeys()