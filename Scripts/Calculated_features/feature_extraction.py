import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import os
from dotenv import load_dotenv

def extract_features(df, sampling_rate):
    """
    Extract time-domain and frequency-domain features from entire dataset.
    :param df: Dataframe containing the raw and filtered normalized vibration data from one lift journey.
    :param sampling_rate: The rate at which the data was sampled.
    :return: Dictionary containing extracted features.
    """
    features = {}
    axes = ['x', 'y', 'z']

    for axis in axes:
        col_name = f'{axis}'
        features.update(extracted_axis_features(df, col_name, axis, sampling_rate))

    return features


def extracted_axis_features(data, column, axis, sampling_rate):
    """
    Helper function to extract features for a given axis and data type.
    :param data: DataFrame or segment containing the data.
    :param column: Name of the column to process.
    :param axis: Current axis being processed (x, y, z).
    :param sampling_rate: Sampling rate of the data in Hz.
    :return: Dictionary of features for this specific axis and data type.
    """
    features = {}
    signal = data[column].fillna(data[column].median())  # Handle missing data

    # ==================== Time-Domain Features ====================
    features[f'{axis}_RMS'] = np.sqrt(np.mean(signal ** 2))
    features[f'{axis}_WL'] = np.sum(np.abs(np.diff(signal)))
    features[f'{axis}_ZC'] = ((signal[:-1] * signal[1:]) < 0).sum()  # Zero crossing
    features[f'{axis}_SSC'] = ((np.diff(signal[:-1]) * np.diff(signal[1:])) < 0).sum()  # Slope sign change
    features[f'{axis}_MAV'] = np.mean(np.abs(signal))
    features[f'{axis}_P2P'] = signal.max() - signal.min()  # Peak-to-peak
    features[f'{axis}_Variance'] = np.var(signal)
    features[f'{axis}_Skewness'] = skew(signal)
    features[f'{axis}_Kurtosis'] = kurtosis(signal)
    features[f'{axis}_Max'] = signal.max()
    features[f'{axis}_Min'] = signal.min()
    features[f'{axis}_25th_Percentile'] = np.percentile(signal, 25)
    features[f'{axis}_75th_Percentile'] = np.percentile(signal, 75)

    # ==================== Frequency-Domain Features ====================
    freq_component = np.fft.fft(signal)
    freq_magnitude = np.abs(freq_component)
    frequencies = np.fft.fftfreq(len(signal), d=1 / sampling_rate)
    psd = freq_magnitude ** 2
    psd_normalized = psd / np.sum(psd)

    features[f'{axis}_MNF'] = np.sum(frequencies * freq_magnitude) / np.sum(freq_magnitude)  # Mean frequency
    features[f'{axis}_MedianFrequency'] = np.median(frequencies)
    features[f'{axis}_SNR'] = 10 * np.log10(np.max(freq_magnitude) ** 2 / np.mean(freq_magnitude ** 2))
    features[f'{axis}_PSD'] = np.mean(freq_magnitude ** 2)
    features[f'{axis}_FrequencyCentroid'] = np.sum(frequencies * freq_magnitude) / np.sum(freq_magnitude)
    features[f'{axis}_RMSF'] = np.sqrt(np.mean(frequencies ** 2 * freq_magnitude))  # RMS frequency
    features[f'{axis}_MaxFrequency'] = frequencies[np.argmax(freq_magnitude)]
    features[f'{axis}_SpectralEntropy'] = entropy(psd_normalized)
    features[f'{axis}_SpectralSkewness'] = skew(freq_magnitude)
    features[f'{axis}_SpectralKurtosis'] = kurtosis(freq_magnitude)
    features[f'{axis}_DominantFrequency'] = frequencies[np.argmax(psd)]

    # Band power (0.5-5 Hz)
    band = (frequencies >= 0.5) & (frequencies <= 5)
    band_power = np.trapz(psd[band], frequencies[band])
    features[f'{axis}_BandPower'] = band_power

    return features


def save_extracted_features(data_frames, file_names, output_path, sampling_rate=100):
    """
    Extract features from a list of dataframes and save to a CSV file.
    :param data_frames: List of preprocessed dataframes.
    :param file_names: Corresponding list of file names.
    :param output_path: Path to save the extracted features CSV file.
    :param sampling_rate: Sampling rate for feature extraction.
    """
    feature_matrix = []
    
    for i, (df, file_name) in enumerate(zip(data_frames, file_names)):
        if isinstance(df, pd.DataFrame):
            features = extract_features(df, sampling_rate)
            label = df['label'].iloc[0]  # Extract label (same for all rows in journey)
            features['label'] = label
            feature_matrix.append(features)
            print(f"Features extracted from journey {i + 1} in file '{file_name}'")
        else:
            print(f"Error: DataFrame expected, but got {type(df)}")

    # Convert to DataFrame and save
    features_df = pd.DataFrame(feature_matrix)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    return features_df

if __name__ == "__main__":
    # Load environment variables
    dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
    load_dotenv(dotenv_path)

    # Example usage
    combined_data_path = os.getenv("DIRECTORY_CLEANED_DATA")
    OUTPUT_PATH = os.getenv("OUTPUT_FEATURES_PATH")

    # Load preprocessed data
    if combined_data_path and os.path.exists(combined_data_path):
        filtered_normalized_dataFrames = pd.read_pickle(combined_data_path)
        all_file_names = [f"file_{i}" for i in range(len(filtered_normalized_dataFrames))]

        # Extract and save features
        extracted_features_df = save_extracted_features(
            filtered_normalized_dataFrames, all_file_names, OUTPUT_PATH
        )
        print(extracted_features_df.head())
    else:
        print(f"Error: Data file not found at {combined_data_path}")
