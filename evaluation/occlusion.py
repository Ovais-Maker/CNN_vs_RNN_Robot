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

OCCLUSION_LEVELS = [
    0,
    20,
    40,
    60,
    80
]

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

def apply_temporal_occlusion(
    frames,
    percentage
):
    """
    Occlude the MOST RECENT frames.

    0%  -> no occlusion
    20% -> Frame 5
    40% -> Frames 4-5
    60% -> Frames 3-5
    80% -> Frames 2-5

    The same corrupted sequence is given to
    both CNN and CNN-LSTM.
    """

    frames = frames.clone()

    sequence_length = frames.size(1)

    num_occluded = int(
        sequence_length *
        percentage /
        100
    )

    if num_occluded > 0:

        start_index = (
            sequence_length -
            num_occluded
        )

        frames[
            :,
            start_index:,
            :,
            :,
            :
        ] = 0.0

    return frames

def evaluate(
    percentage
):

    cnn_total_loss = 0.0

    lstm_total_loss = 0.0

    total_samples = 0


    with torch.no_grad():

        for frames, targets in test_loader:

            frames = frames.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            corrupted = apply_temporal_occlusion(
                frames,
                percentage
            )

            cnn_input = corrupted[:, -1]

            cnn_prediction = cnn(
                cnn_input
            )

            cnn_loss = criterion(
                cnn_prediction,
                targets
            )

            cnn_total_loss += (
                cnn_loss.item()
            )

            lstm_prediction = lstm(
                corrupted
            )

            lstm_loss = criterion(
                lstm_prediction,
                targets
            )

            lstm_total_loss += (
                lstm_loss.item()
            )


            total_samples += (
                targets.size(0)
            )


    cnn_mse = (
        cnn_total_loss /
        total_samples
    )

    lstm_mse = (
        lstm_total_loss /
        total_samples
    )

    return cnn_mse, lstm_mse

results = []


print()
print("=" * 75)
print("CORRECTED TEMPORAL OCCLUSION EXPERIMENT")
print("=" * 75)


for percentage in OCCLUSION_LEVELS:

    print(
        f"\nTesting {percentage}% occlusion..."
    )


    cnn_mse, lstm_mse = evaluate(
        percentage
    )


    if cnn_mse < lstm_mse:

        winner = "CNN"

    elif lstm_mse < cnn_mse:

        winner = "CNN-LSTM"

    else:

        winner = "Tie"


    print(
        f"CNN Test MSE      : "
        f"{cnn_mse:.10f}"
    )

    print(
        f"CNN-LSTM Test MSE : "
        f"{lstm_mse:.10f}"
    )

    print(
        f"Better model      : "
        f"{winner}"
    )


    results.append(
        [
            percentage,
            cnn_mse,
            lstm_mse,
            winner
        ]
    )

os.makedirs(
    "results",
    exist_ok=True
)


csv_path = (
    "results/occlusion_results.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "Occlusion_Percentage",
            "CNN_MSE",
            "CNN_LSTM_MSE",
            "Better_Model"
        ]
    )

    writer.writerows(
        results
    )

print()
print("=" * 75)
print("OCCLUSION SUMMARY")
print("=" * 75)

print(
    f"{'Occlusion':<15}"
    f"{'CNN MSE':<20}"
    f"{'CNN-LSTM MSE':<20}"
    f"{'Winner'}"
)

print("-" * 75)


for row in results:

    print(
        f"{str(row[0]) + '%':<15}"
        f"{row[1]:<20.10f}"
        f"{row[2]:<20.10f}"
        f"{row[3]}"
    )


print()
print(
    "Results saved to:",
    csv_path
)

print(
    "\nCorrected experiment completed successfully!"
)