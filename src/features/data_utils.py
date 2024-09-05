import pandas as pd
import h5py
import numpy as np


def load_csv_data(csv_file):
    """
    Load CSV data from a file.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(csv_file)
    return df


def filter_data(df, min_magnitude=3, max_distance=20):
    """
    Filter the DataFrame based on magnitude and source distance.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the earthquake data.
        min_magnitude (float): Minimum magnitude to filter events.
        max_distance (float): Maximum source distance to filter events.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df[
        (df.trace_category == "earthquake_local")
        & (df.source_distance_km <= max_distance)
        & (df.source_magnitude > min_magnitude)
    ]
    return filtered_df


def load_hdf5_data(file_name, ev_list):
    """
    Load waveform data from HDF5 file.

    Parameters:
        file_name (str): Path to the HDF5 file.
        ev_list (list): List of trace names.

    Returns:
        list: Loaded waveforms.
    """
    dtfl = h5py.File(file_name, "r")
    waveforms = []

    for evi in ev_list:
        dataset = dtfl.get("data/" + str(evi))
        data = np.array(dataset)
        waveforms.append(data)

    return np.array(waveforms)


def normalize_waveforms(waveforms):
    """
    Normalize waveform data.

    Parameters:
        waveforms (np.ndarray): Waveforms to be normalized.

    Returns:
        np.ndarray: Normalized waveforms.
    """
    return waveforms / np.max(np.abs(waveforms), axis=(1, 2), keepdims=True)


def normalize_waveforms_in_batches(waveforms, batch_size=1000):
    """
    Normalize waveform data in batches to avoid memory errors.

    Args:
        waveforms (np.ndarray): Original waveforms.
        batch_size (int): Size of each batch for normalization.

    Returns:
        np.ndarray: Normalized waveforms.
    """
    num_batches = waveforms.shape[0] // batch_size + 1  # Calculate number of batches
    normalized_waveforms = np.empty_like(
        waveforms
    )  # Create an empty array with the same shape

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, waveforms.shape[0])

        # Normalize the batch
        normalized_waveforms[start_idx:end_idx] = waveforms[start_idx:end_idx] / np.max(
            np.abs(waveforms[start_idx:end_idx]), axis=(1, 2), keepdims=True
        )

    return normalized_waveforms


def generate_normalized_noise_samples(num_samples, sample_length, num_channels):
    noise = np.random.normal(0, 1, (num_samples, sample_length, num_channels))
    noise = noise / np.max(np.abs(noise), axis=(1, 2), keepdims=True)
    return noise


def generate_normalized_noise_samples_in_batches(
    num_samples, sample_length, num_channels, batch_size=1000
):
    """
    Generate and normalize noise samples in batches to avoid memory errors.

    Args:
        num_samples (int): Total number of noise samples to generate.
        sample_length (int): Length of each sample (number of time steps).
        num_channels (int): Number of channels (e.g., 3 for seismic data).
        batch_size (int): Size of each batch for generation and normalization.

    Returns:
        np.ndarray: Generated and normalized noise samples.
    """
    # Preallocate the final array to hold all the noise samples
    noise_samples = np.empty(
        (num_samples, sample_length, num_channels), dtype=np.float32
    )

    # Calculate number of batches
    num_batches = num_samples // batch_size + int(num_samples % batch_size > 0)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        # Generate noise for this batch
        noise_batch = np.random.normal(
            0, 1, (end_idx - start_idx, sample_length, num_channels)
        )

        # Normalize the batch
        noise_batch = noise_batch / np.max(
            np.abs(noise_batch), axis=(1, 2), keepdims=True
        )

        # Store the normalized batch into the preallocated array
        noise_samples[start_idx:end_idx] = noise_batch

        print(
            f"Batch {i+1}/{num_batches} generated and normalized"
        )  # Optional: progress printout

    return noise_samples
