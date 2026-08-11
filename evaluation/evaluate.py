import os
import sys
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from utils.dataset import RobotDataset
from models.cnn import CNNModel
from models.cnn_lstm import CNNLSTMModel

DEVICE = torch.device("cpu")

TEST_PATH = "dataset/data/test.npz"

CNN_PATH = "results/cnn_model.pth"

LSTM_PATH = "results/cnn_lstm_model.pth"

BATCH_SIZE = 32

test_dataset = RobotDataset(
    TEST_PATH
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Test samples:",
    len(test_dataset)
)

cnn = CNNModel()

cnn.load_state_dict(
    torch.load(
        CNN_PATH,
        map_location=DEVICE
    )
)

cnn.to(DEVICE)

cnn.eval()

lstm = CNNLSTMModel()

lstm.load_state_dict(
    torch.load(
        LSTM_PATH,
        map_location=DEVICE
    )
)

lstm.to(DEVICE)

lstm.eval()

criterion = nn.MSELoss(
    reduction="sum"
)

cnn_total_loss = 0.0

cnn_total_samples = 0


with torch.no_grad():

    for frames, targets in test_loader:

        frames = frames.to(DEVICE)

        targets = targets.to(DEVICE)

        # CNN sees only final frame

        inputs = frames[:, -1]

        predictions = cnn(
            inputs
        )

        loss = criterion(
            predictions,
            targets
        )

        cnn_total_loss += loss.item()

        cnn_total_samples += targets.size(0)


cnn_mse = (
    cnn_total_loss /
    cnn_total_samples
)

lstm_total_loss = 0.0

lstm_total_samples = 0


with torch.no_grad():

    for frames, targets in test_loader:

        frames = frames.to(DEVICE)

        targets = targets.to(DEVICE)

        # CNN-LSTM sees all 5 frames

        predictions = lstm(
            frames
        )

        loss = criterion(
            predictions,
            targets
        )

        lstm_total_loss += loss.item()

        lstm_total_samples += targets.size(0)


lstm_mse = (
    lstm_total_loss /
    lstm_total_samples
)

if cnn_mse < lstm_mse:

    winner = "CNN"

else:

    winner = "CNN-LSTM"


difference = abs(
    cnn_mse - lstm_mse
)

print("\n==========================================")
print("FINAL TEST-SET PERFORMANCE")
print("==========================================")

print(
    f"CNN Test MSE      : {cnn_mse:.10f}"
)

print(
    f"CNN-LSTM Test MSE : {lstm_mse:.10f}"
)

print(
    f"Absolute Difference: {difference:.10f}"
)

print(
    f"Lower-MSE Model   : {winner}"
)

os.makedirs(
    "results",
    exist_ok=True
)


csv_path = "results/model_comparison.csv"


with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Test_MSE"
    ])

    writer.writerow([
        "CNN",
        cnn_mse
    ])

    writer.writerow([
        "CNN-LSTM",
        lstm_mse
    ])


print(
    "\nResults saved to:",
    csv_path
)