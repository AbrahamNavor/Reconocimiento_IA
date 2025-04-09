import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class AutoencoderModel(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(AutoencoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()  # Usar sigmoid para valores escalados entre 0 y 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderTrainer:
    def __init__(self, dataframe, features, encoding_dim=32, learning_rate=0.001, epochs=50, batch_size=32):
        self.df = dataframe
        self.features = features
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scaled_features = None

    def preprocess(self):
        # Escalar las características
        X = self.df[self.features]
        self.scaled_features = self.scaler.fit_transform(X)
        self.scaled_features = torch.tensor(self.scaled_features, dtype=torch.float32)
        return self.scaled_features

    def train(self):
        scaled_features = self.preprocess()
        X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

        input_dim = X_train.shape[1]
        self.model = AutoencoderModel(input_dim, self.encoding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_loader = torch.utils.data.DataLoader(X_train, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(X_test, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    def get_encoded_data(self):
        self.model.eval()
        with torch.no_grad():
            encoded_data = self.model.encoder(self.scaled_features)
        return encoded_data.numpy()

    def reconstruct_data(self):
        self.model.eval()
        with torch.no_grad():
            reconstructed_data = self.model(self.scaled_features)
        return reconstructed_data.numpy()
