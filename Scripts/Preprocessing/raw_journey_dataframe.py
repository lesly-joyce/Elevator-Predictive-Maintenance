import os
import pickle
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)


def extract_lift_journey_from_raw(raw_dataFrames, normalize_dataFrames, journeys):
    """
    Extract lift journeys from the original raw data using the previously found indices.
    """
    raw_journey_dataFrames = []

    for i, (raw_df, normalized_df, journey) in enumerate(zip(raw_dataFrames, normalize_dataFrames, journeys)):
        if journey is None or len(journey) != 4:
            print(f"Skipping dataset {i}: Invalid journey data {journey}")
            raw_journey_dataFrames.append(pd.DataFrame())
            continue

        start, _, end, _ = journey

        # Ensure start and end indices are within bounds
        if start < 0 or end >= len(normalized_df):
            print(f"Skipping dataset {i}: Invalid journey indices (start: {start}, end: {end})")
            raw_journey_dataFrames.append(pd.DataFrame())
            continue

        # Get timestamps from normalized DataFrame
        start_timestamp = normalized_df.iloc[start]['timeStamp']
        end_timestamp = normalized_df.iloc[end]['timeStamp']

        # Create mask for raw data based on these timestamps
        mask = (raw_df['timeStamp'] >= start_timestamp) & (raw_df['timeStamp'] <= end_timestamp)

        # Extract journey data and append
        journey_data = raw_df.loc[mask].copy()
        raw_journey_dataFrames.append(journey_data)
        print(f"Extracted journey DataFrame {i} - Rows: {journey_data.shape[0]}")

    return raw_journey_dataFrames


def extract_and_save_journeys():
    """
    Load raw data, normalized data, and detected journeys.
    Extract journey-specific raw data and save it to a pickle file.
    """
    # Load raw data
    raw_pickle_path = os.getenv("DIRECTORY_PICKLE")
    with open(raw_pickle_path, "rb") as f:
        raw_dataFrames = pickle.load(f)

    print(f"Type of raw_dataFrames: {type(raw_dataFrames)}")
    if len(raw_dataFrames) > 0:
        print(f"Type of first element in raw_dataFrames: {type(raw_dataFrames[0])}")

    # Load normalized data
    normalized_pickle_path = os.getenv("DIRECTORY_FILTERED_NORMALIZED_PICKLE")
    with open(normalized_pickle_path, "rb") as f:
        normalize_dataFrames = pickle.load(f)

    print(f"Type of normalize_dataFrames: {type(normalize_dataFrames)}")
    if len(normalize_dataFrames) > 0:
        print(f"Type of first element in normalize_dataFrames: {type(normalize_dataFrames[0])}")

    # Load detected journeys
    journeys_pickle_path = os.getenv("DIRECTORY_JOURNEYS_PICKLE")
    with open(journeys_pickle_path, "rb") as f:
        journeys = pickle.load(f)

    print(f"Type of journeys: {type(journeys)}")
    if len(journeys) > 0:
        print(f"First journey: {journeys[0]}")

    # Extract lift journeys
    raw_journey_dataFrames = extract_lift_journey_from_raw(raw_dataFrames, normalize_dataFrames, journeys)

    # Save extracted journeys to a pickle file
    journeys_data_path = os.getenv("DIRECTORY_JOURNEYS_DATA")
    with open(journeys_data_path, "wb") as f:
        pickle.dump(raw_journey_dataFrames, f)

    print(f"Lift journey-specific raw data saved to {journeys_data_path}")

# Example usage:
if __name__ == "__main__":
    extract_and_save_journeys()


