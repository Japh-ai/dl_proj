import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy


# File paths
file_name = "../../data/chunk2.hdf5"
csv_file = "../../data/chunk2.csv"

# Reading the csv file into a dataframe:
df = pd.read_csv(csv_file)
print(f"total events in csv file: {len(df)}")

# Filtering the dataframe
df = df[
    (df.trace_category == "earthquake_local")
    & (df.source_distance_km <= 20)
    & (df.source_magnitude > 3)
]
print(f"total events selected: {len(df)}")

# Making a list of trace names for the selected data
ev_list = df["trace_name"].to_list()

# Retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, "r")

# Extracting waveforms and labels
waveforms = []
labels = []

for evi in ev_list:
    dataset = dtfl.get("data/" + str(evi))
    data = np.array(dataset)
    waveforms.append(data)
    labels.append(1)  # Label for events


# Normalize the data
waveforms = waveforms / np.max(np.abs(waveforms), axis=(1, 2), keepdims=True)


# Convert to numpy arrays
waveforms = np.array(waveforms)
labels = np.array(labels)
plt.plot(waveforms[1])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()

# %%


# Function to generate and normalize noise data
def generate_normalized_noise_samples(num_samples, sample_length, num_channels):
    noise = np.random.normal(0, 1, (num_samples, sample_length, num_channels))
    noise = noise / np.max(np.abs(noise), axis=(1, 2), keepdims=True)
    return noise


num_noise_samples = len(waveforms)
sample_length = waveforms.shape[1]
num_channels = waveforms.shape[2]

noise_waveforms = generate_normalized_noise_samples(
    num_noise_samples, sample_length, num_channels
)
noise_labels = np.zeros(num_noise_samples)

plt.plot(noise_waveforms[0])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Noise Waveform")
plt.show()


# Combine event and noise data
X = np.concatenate((waveforms, noise_waveforms), axis=0)
y = np.concatenate((labels, noise_labels), axis=0)

# Shuffle the dataset
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

len(X)
plt.plot(X[4])
X[0]

##
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.7, random_state=41)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for training and validation sets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class SeismicEventCNN(nn.Module):
    def __init__(self, sample_length, num_channels):
        super(SeismicEventCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        # Calculate the correct dimension after convolutions and pooling
        conv_output_length = ((sample_length - 2) // 2 - 2) // 2
        self.fc1 = nn.Linear(64 * conv_output_length, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# Instantiate the model with the correct input dimensions
sample_length = waveforms.shape[1]
num_channels = waveforms.shape[2]
model = SeismicEventCNN(sample_length, num_channels)


# %% Train

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.permute(
            0, 2, 1
        )  # Change to (batch_size, num_channels, sample_length)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.round()
            correct += preds.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")


# %%visual inspection
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.permute(0, 2, 1)
        outputs = model(inputs)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# Convert predictions and labels to a binary format (0 or 1)
all_preds_binary = np.round(all_preds)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(
    range(len(all_preds)),
    all_preds,
    alpha=0.6,
    label="Predictions",
    marker="o",
    color="blue",
)
plt.scatter(
    range(len(all_labels)),
    all_labels,
    alpha=0.6,
    label="Ground Truth",
    marker="x",
    color="red",
)
plt.title("Model Predictions vs. Ground Truth")
plt.xlabel("Sample Index")
plt.ylabel("Prediction Value")
plt.legend()
plt.show()


# %%

model.eval()
all_preds = []
all_labels = []
all_waveforms = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.permute(0, 2, 1)
        outputs = model(inputs)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_waveforms.extend(inputs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_waveforms = np.array(all_waveforms)

model(inputs)
labels


def plot_waveform_with_predictions(waveform, pred, label, index):
    plt.figure(figsize=(12, 6))
    channels = waveform.shape[0]

    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        plt.plot(waveform[i], label="Waveform")
        plt.axhline(y=0, color="k", linestyle="--")
        plt.title(f"Channel {i+1}")
        if i == 0:
            plt.text(
                len(waveform[i]) // 2,
                np.max(waveform[i]) * 0.8,
                f"Prediction: {pred:.2f}",
                fontsize=12,
                color="blue",
            )
            plt.text(
                len(waveform[i]) // 2,
                np.max(waveform[i]) * 0.6,
                f"Label: {label}",
                fontsize=12,
                color="red",
            )
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.suptitle(f"Waveform with Predictions - Index {index}", y=1.02)
    plt.show()


# Plot a few examples
num_examples = 5
indices = np.random.choice(len(all_waveforms), num_examples, replace=False)

for idx in indices:
    plot_waveform_with_predictions(
        all_waveforms[idx], all_preds[idx][0], all_labels[idx][0], idx
    )


len(all_waveforms)


# %%

import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime


# it has a header
eq_data = pd.read_csv("../../data/data.csv")

# Assuming you have a DataFrame called 'eq_data' with your earthquake data
# eq_data should have columns 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', etc.
# Adjust column names as per your actual DataFrame structure

# Initialize the client
client = Client("GFZ")


# Define a function to download and plot waveform data for a given earthquake time
def download_and_plot_waveform(
    eq_time, station="AQU", network="CX", channel="HH?", duration=3600
):
    """
    Downloads and plots waveform data for a specific earthquake time.

    Parameters:
    eq_time: UTCDateTime object representing the earthquake time.
    station: The seismic station to download data from.
    network: The network code (e.g., "MN").
    channel: The channel pattern (e.g., "HH?").
    duration: Duration of the waveform to download (in seconds).
    """
    starttime = eq_time
    endtime = starttime + duration

    try:
        # Fetch waveform data
        stream = client.get_waveforms(
            network=network,
            station=station,
            location="*",
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )

        # Plot the waveform data
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(111)
        for i in range(len(stream)):
            ax.plot(stream[i].times(), stream[i].data, label=stream[i].stats.channel)
        ax.legend()
        plt.title(f"Waveform Data for Earthquake at {starttime}")
        plt.show()

    except Exception as e:
        print(f"Error fetching waveform for time {eq_time}: {e}")


# Example: Iterate over the earthquake data and download waveforms
for index, row in eq_data.iterrows():
    # Convert the row's date and time to UTCDateTime
    eq_time = UTCDateTime(
        year=int(row["Year"]),
        month=int(row["Month"]),
        day=int(row["Day"]),
        hour=int(row["Hour"]),
        minute=int(row["Minute"]),
        second=int(row["Second"]),
    )

    # Download and plot the waveform data for this earthquake
    download_and_plot_waveform(eq_time)


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
trace_filtered = trace.copy()  # Make a copy of the trace to avoid altering the original
trace = trace_filtered.filter("bandpass", freqmin=1.0, freqmax=45.0)


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
    predictions = model(input_tensor)

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
for i in range(23):  # Adjust range as needed
    plot_waveform_with_prediction(extracted_windows[i], predictions_np[i], i)
len(extracted_windows)
plt.plot(extracted_windows[1][0])

# %%
