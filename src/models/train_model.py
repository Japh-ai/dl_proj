import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from features.data_utils import (
    load_csv_data,
    filter_data,
    load_hdf5_data,
    normalize_waveforms,
)
from models.model import SeismicEventCNN


def create_dataloaders(X, y, batch_size=32, test_size=0.3):
    """
    Create DataLoader objects for training and validation sets.

    Parameters:
        X (np.ndarray): Input features.
        y (np.ndarray): Labels.
        batch_size (int): Batch size for DataLoader.
        test_size (float): Proportion of data for validation.

    Returns:
        DataLoader: Train and validation DataLoaders.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs=5, lr=0.001, device="cuda"):
    """
    Train the CNN model.

    Parameters:
        model (nn.Module): The PyTorch model.
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        device (str): Device ('cpu' or 'cuda').

    Returns:
        nn.Module: Trained model.
    """
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

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

        # Validation
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

    return model
