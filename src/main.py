from features.data_utils import (
    load_csv_data,
    filter_data,
    load_hdf5_data,
    normalize_waveforms,
    normalize_waveforms_in_batches,
    generate_normalized_noise_samples,
    generate_normalized_noise_samples_in_batches,
)

from models.model import SeismicEventCNN
from models.train_model import create_dataloaders, train_model
from visualization.visualize import plot_waveform_with_prediction

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# def main():
# Paths
# STEAD Training data

hdf5_file = "../data/chunk2.hdf5"
csv_file = "../data/chunk2.csv"


# Load CSV data
df = load_csv_data(csv_file)
df = df[:10000]
event_list = df["trace_name"].to_list()
# print(f"Total events in CSV: {len(df)}")

# Filter data
# df_filtered = filter_data(df)
# print(f"Total filtered events: {len(df_filtered)}")

# # Get the event list from the filtered data
# event_list = df_filtered["trace_name"].to_list()

# Load waveforms from HDF5
waveforms = load_hdf5_data(hdf5_file, event_list)
waveforms = normalize_waveforms_in_batches(waveforms, batch_size=10000)
waveform_labels = np.ones(len(waveforms))

num_noise_samples = int(len(waveforms) * 1)
sample_length = waveforms.shape[1]
num_channels = waveforms.shape[2]

noise_waveforms = generate_normalized_noise_samples_in_batches(
    num_noise_samples, sample_length, num_channels, batch_size=10000
)
noise_labels = np.zeros(num_noise_samples)

# Combine event and noise data
X = np.concatenate((waveforms, noise_waveforms), axis=0)
y = np.concatenate((waveform_labels, noise_labels), axis=0)

# Shuffle the dataset
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

train_loader, data_loader = create_dataloaders(X, y, batch_size=32, test_size=0.3)

detector = train_model(
    SeismicEventCNN(sample_length, num_channels),
    train_loader,
    data_loader,
    num_epochs=5,
    lr=0.001,
    device="cuda",
)

# Save the model
# torch.save(detector.state_dict(), "detector_model.pth")


# %%

client = Client("GFZ")

t0 = UTCDateTime("2007/01/01 02:00:00")
t1 = t0 + 24 * 60 * 60
stream = client.get_waveforms(
    network="CX", station="PB01", location="*", channel="HH?", starttime=t0, endtime=t1
)
inv = client.get_stations(
    network="CX", station="PB01", location="*", channel="HH?", starttime=t0, endtime=t1
)
# check what is in stream

stream[0].stats
stream[0].plot()


# trace.plot()
# Apply a bandpass filter between 1 and 45 Hz

# trace_filtered = trace.copy()  # Make a copy of the trace to avoid altering the original
trace = trace_filtered.filter("bandpass", freqmin=1.0, freqmax=45.0)

eq_data = pd.read_csv("../data/data.csv")

# from the catalog, let's create an origin time list of each of the events.
origin_times = []
for index, row in eq_data.iterrows():
    eq_time = UTCDateTime(
        year=int(row["Year"]),
        month=int(row["Month"]),
        day=int(row["Day"]),
        hour=int(row["Hour"]),
        minute=int(row["Minute"]),
        second=int(row["Second"]),
    )
    origin_times.append(eq_time)

# origin_times= origin_times[:10]
len(origin_times)

# Specify the number of seconds before the origin time you want to start your window
seconds_before = 5

# Sample rate of the data (assumed the same for all components)
sample_rate = stream[0].stats.sampling_rate  # get sampling rate from one of the traces
window_length_samples = 6000  # The length of the window in samples

# List to store extracted 3-channel data for each event
extracted_windows = []


# Loop through the origin times and extract data for each event
for origin_time in origin_times:
    t0_event = origin_time - seconds_before  # Start time for each event
    t1_event = (
        t0_event + (window_length_samples - 1) / sample_rate
    )  # End time to get 6000 samples

    # Extract Z, N, E components (HHZ, HHN, HHE)
    try:
        trace_Z = stream.select(channel="HHZ")[0].slice(t0_event, t1_event).data
        trace_N = stream.select(channel="HHN")[0].slice(t0_event, t1_event).data
        trace_E = stream.select(channel="HHE")[0].slice(t0_event, t1_event).data

        # Stack them to form a (3, window_length) array for this event
        three_channel_data = np.stack([trace_Z, trace_N, trace_E])

        # Normalize the data
        three_channel_data_normalized = three_channel_data / np.max(
            np.abs(three_channel_data), axis=1, keepdims=True
        )

        # Append the 3-channel data to the list
        extracted_windows.append(three_channel_data_normalized)

    except Exception as e:
        print(f"Error extracting data for origin time {origin_time}: {e}")

len(extracted_windows)

# Convert the list of windows to a numpy array of shape (num_events, 3, window_length)
extracted_windows = np.array(extracted_windows)

# Make sure the shape is correct
print(
    f"Extracted data shape: {extracted_windows.shape}"
)  # Should print (num_events, 3, 6000)

# Plot an example waveform
plt.plot(extracted_windows[0][0])  # Plot the Z-component of the first event
plt.title("Z-component of the first event")
plt.show()

# Now convert to a PyTorch tensor and feed into the model
input_tensor = torch.tensor(extracted_windows, dtype=torch.float32).to(
    "cuda"
)  # Change 'cpu' to 'cuda' if using GPU
input_tensor = input_tensor.permute(
    0, 1, 2
)  # Make sure the shape is (batch_size, channels, length)

# Make predictions using the model
with torch.no_grad():
    predictions = detector(input_tensor)

# Convert the predictions to numpy array for further processing
predictions_np = predictions.cpu().numpy()

# Print predictions
print(predictions_np)


# Set the threshold for detection
threshold = 0.7

# Count the number of predictions above the threshold
num_detected = np.sum(predictions_np >= threshold)
total_events = len(predictions_np)

# Print the number of detected events and the total events
print(
    f"Detected {num_detected} events out of {total_events} based on a threshold of {threshold:.2f}"
)


# Function to plot waveform along with the prediction probability
def plot_waveform_with_prediction(waveform, pred_prob, index):
    plt.figure(figsize=(10, 6))  # Adjust the figure size to be more visually balanced

    channels = waveform.shape[0]  # Assuming waveform is (channels, samples)

    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        plt.plot(waveform[i], label=f"Channel {i+1}")
        plt.axhline(y=0, color="k", linestyle="--")
        plt.title(f"Channel {i+1}", fontsize=10)
        plt.xlabel("Sample Index", fontsize=8)
        plt.ylabel("Amplitude", fontsize=8)

    # Add the prediction probability to the plot
    plt.suptitle(
        f"Waveform with Prediction - Index {index}\nPrediction Probability: {pred_prob[0]:.4f}",
        fontsize=14,
    )

    plt.tight_layout(pad=2.0)  # Adds padding between plots for better readability
    plt.show()


# Loop over a few examples to plot waveforms with predictions
for i in range(28):  # Adjust range as needed
    plot_waveform_with_prediction(extracted_windows[i], predictions_np[i], i)
len(extracted_windows)

# plt.plot(extracted_windows[1][0])


# %%
# create 5 windows of noise data and add to extracted_windows
num_noise_samples = 5
noise_windows = generate_normalized_noise_samples_in_batches(
    num_noise_samples, window_length_samples, num_channels, batch_size=1
)

noise_windows = noise_windows.transpose(0, 2, 1)  # Changes shape to (5, 3, 6000)

extracted_windows = np.concatenate((extracted_windows, noise_windows), axis=0)
test_tensor = torch.tensor(extracted_windows, dtype=torch.float32).to("cuda")

# Set the model to evaluation mode and make predictions
detector.eval()
with torch.no_grad():
    predictions = detector(test_tensor)

# Convert predictions to numpy for inspection
predictions_np = predictions.cpu().numpy()

# Print the predictions
print("Predictions for first 5 waveforms:")
for i, pred in enumerate(predictions_np):
    print(f"Waveform {i+1}: Probability of being an event = {pred[0]:.4f}")

for i in range(min(5, len(test_tensor))):
    plt.figure(figsize=(10, 4))
    for channel in range(test_tensor.shape[1]):  # Plot each channel
        plt.plot(test_tensor[i][channel].cpu().numpy(), label=f"Channel {channel+1}")
    plt.title(f"Waveform {i+1} with Prediction: {predictions_np[i][0]:.4f}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
# %%
