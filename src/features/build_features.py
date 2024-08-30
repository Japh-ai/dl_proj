import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

# File paths
file_name = "../../data/raw/chunk2.hdf5"
csv_file = "../../data/raw/chunk2.csv"

# Reading the csv file into a dataframe:
df = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df)}')

# Filtering the dataframe
df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
print(f'total events selected: {len(df)}')

# Making a list of trace names for the selected data
ev_list = df['trace_name'].to_list()

# Retrieving selected waveforms from the hdf5 file: 
dtfl = h5py.File(file_name, 'r')

# Extracting waveforms and labels
waveforms = []
labels = []

for evi in ev_list:
    dataset = dtfl.get('data/' + str(evi))
    data = np.array(dataset)
    waveforms.append(data)
    labels.append(1)  # Label for events


# Normalize the data
waveforms = waveforms / np.max(np.abs(waveforms), axis=(1, 2), keepdims=True)


# Convert to numpy arrays
waveforms = np.array(waveforms)
labels = np.array(labels)
plt.plot(waveforms[0])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Waveform')
plt.show()

#%%

# Function to generate and normalize noise data
def generate_normalized_noise_samples(num_samples, sample_length, num_channels):
    noise = np.random.normal(0, 1, (num_samples, sample_length, num_channels))
    noise = noise / np.max(np.abs(noise), axis=(1, 2), keepdims=True)
    return noise

num_noise_samples = len(waveforms)
sample_length = waveforms.shape[1]
num_channels = waveforms.shape[2]

noise_waveforms = generate_normalized_noise_samples(num_noise_samples, sample_length, num_channels)
noise_labels = np.zeros(num_noise_samples)

plt.plot(noise_waveforms[0])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise Waveform')
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.6, random_state=42)

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


#%% Train
 
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.permute(0, 2, 1)  # Change to (batch_size, num_channels, sample_length)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
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
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')





#%%visual inspection
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

import matplotlib.pyplot as plt

# Convert predictions and labels to a binary format (0 or 1)
all_preds_binary = np.round(all_preds)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(all_preds)), all_preds, alpha=0.6, label='Predictions', marker='o', color='blue')
plt.scatter(range(len(all_labels)), all_labels, alpha=0.6, label='Ground Truth', marker='x', color='red')
plt.title('Model Predictions vs. Ground Truth')
plt.xlabel('Sample Index')
plt.ylabel('Prediction Value')
plt.legend()
plt.show()




#%%

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



import matplotlib.pyplot as plt

def plot_waveform_with_predictions(waveform, pred, label, index):
    plt.figure(figsize=(12, 6))
    channels = waveform.shape[0]
    
    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        plt.plot(waveform[i], label='Waveform')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title(f'Channel {i+1}')
        if i == 0:
            plt.text(len(waveform[i]) // 2, np.max(waveform[i]) * 0.8, f'Prediction: {pred:.2f}', fontsize=12, color='blue')
            plt.text(len(waveform[i]) // 2, np.max(waveform[i]) * 0.6, f'Label: {label}', fontsize=12, color='red')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.suptitle(f'Waveform with Predictions - Index {index}', y=1.02)
    plt.show()

# Plot a few examples
num_examples = 5
indices = np.random.choice(len(all_waveforms), num_examples, replace=False)

for idx in indices:
    plot_waveform_with_predictions(all_waveforms[idx], all_preds[idx][0], all_labels[idx][0], idx)


len(all_waveforms)




# %%
