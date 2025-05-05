import os

from typing import Literal

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# A small simple autoencoder model
class AutoencoderFeatureExtractor(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoencoderFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def extract_features(self, x):
        with torch.no_grad():
            return self.encoder(x)


def train_autoencoder(
    option_type: Literal["European_Vanilla", "Worst_Off"],
    model,
    X_train,
    device,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
):
    """
    Training function for the autoencoder.

    Args:
        option_type (str): the type of option data
        model (AutoencoderFeatureExtractor object): the model that we want to train
        X_train (torch.tensor): training data
        device (torch.device): device (if gpu available it will be a cuda device, otherwise it will be cpu)
        epochs (int, optional): number of epochs we train our model. Defaults to 25.
        batch_size (int, optional): batch size of data being fed to the model. Defaults to 64.
        learning_rate (float, optional): learning rate for the optimizer, here is Adam. Defaults to 1e-5.

    Returns:
        AutoencoderFeatureExtractor object: _description_
    """
    model_path = f"results/{option_type}/autoencoder_model.pth"

    # Check if the model already exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Loading the model.")
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        return model

    print("Training the autoencoder...")

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    X_train = X_train.to(device)

    dataset = torch.utils.data.TensorDataset(X_train, X_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, inputs)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the trained model
    os.makedirs(f"results/{option_type}", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    return model


def autoencoder_feature_extraction(
    option: Literal["European_Vanilla", "Worst_Off"],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[pd.DataFrame]:
    """
    This function checks if there is an autoencoder available, if yes, it uses that to extract 40 important features.
    Otherwise, it makes one and trains it on training data, then it transforms both of the training and test date and returns them in a list.

    Parameters:
      - option (str): type of option data we are working with
      - X_train (pd.DataFrame): training data
      - X_test (pd.DataFrame): testing data
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = X_train.shape[1]
    encoding_dim = 40
    model = AutoencoderFeatureExtractor(input_dim, encoding_dim)
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    model = train_autoencoder(option, model, X_train_tensor, device)
    X_train_features = (
        model.extract_features(X_train_tensor.to(device)).cpu().detach().numpy()
    )
    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    X_test_features = (
        model.extract_features(X_test_tensor.to(device)).cpu().detach().numpy()
    )

    results = {"X_train_features": X_train_features, "X_test_features": X_test_features}
    return results
