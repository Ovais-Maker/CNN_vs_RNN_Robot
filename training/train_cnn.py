import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow Python to find project folders
sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from utils.dataset import RobotDataset
from models.cnn import CNNModel

# CONFIGURATION

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

TRAIN_PATH = "dataset/data/train.npz"

VAL_PATH = "dataset/data/val.npz"

MODEL_PATH = "results/cnn_model.pth"

# DATASET

train_dataset = RobotDataset(
    TRAIN_PATH
)

val_dataset = RobotDataset(
    VAL_PATH
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)

# MODEL

model = CNNModel().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# TRAINING

best_val_loss = float("inf")

os.makedirs(
    "results",
    exist_ok=True
)


for epoch in range(EPOCHS):

    # TRAIN

    model.train()

    train_loss = 0.0

    for frames, targets in train_loader:

        frames = frames.to(DEVICE)

        targets = targets.to(DEVICE)

        # CNN receives only ONE frame.
        #
        # We use the final frame from
        # each 5-frame sequence.

        inputs = frames[:, -1]

        optimizer.zero_grad()

        predictions = model(
            inputs
        )

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for frames, targets in val_loader:

            frames = frames.to(DEVICE)

            targets = targets.to(DEVICE)

            inputs = frames[:, -1]

            predictions = model(
                inputs
            )

            loss = criterion(
                predictions,
                targets
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # PRINT RESULTS

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train MSE: {train_loss:.8f} "
        f"| Val MSE: {val_loss:.8f}"
    )

    # SAVE BEST MODEL

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "  -> Best CNN model saved."
        )


print("\nCNN training completed!")

print(
    "Best validation MSE:",
    f"{best_val_loss:.8f}"
)

print(
    "Model saved to:",
    MODEL_PATH
)