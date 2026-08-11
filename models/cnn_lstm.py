import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = x.flatten(1)

        return x


class CNNLSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        # CNN

        self.cnn = CNNFeatureExtractor()

        # LSTM

        self.lstm = nn.LSTM(

            input_size=128,

            hidden_size=64,

            num_layers=1,

            batch_first=True
        )

        # Prediction head

        self.regressor = nn.Sequential(

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        # Input shape:
        #
        # batch
        # sequence
        # channels
        # height
        # width
        #
        # Example:
        # [16, 5, 3, 128, 128]

        batch_size = x.size(0)

        sequence_length = x.size(1)

        channels = x.size(2)

        height = x.size(3)

        width = x.size(4)

        # Combine batch and sequence

        x = x.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        # Extract visual features

        features = self.cnn(x)

        # features:
        #
        # [batch * sequence, 128]

        # Restore sequence

        features = features.reshape(
            batch_size,
            sequence_length,
            128
        )

        # LSTM

        output, _ = self.lstm(
            features
        )

        # Use last timestep

        last_output = output[:, -1, :]

        # Predict dx, dy

        prediction = self.regressor(
            last_output
        )

        return prediction