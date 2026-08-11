import numpy as np
import torch
from torch.utils.data import Dataset


class RobotDataset(Dataset):

    def __init__(self, file_path):

        data = np.load(file_path)

        self.X = data["X"]
        self.y = data["y"]

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        frames = self.X[index]

        target = self.y[index]

        # Convert image values:
        # 0-255 -> 0-1
        frames = torch.tensor(
            frames,
            dtype=torch.float32
        ) / 255.0

        # Convert:
        # Sequence, Height, Width, Channels
        #
        # to:
        # Sequence, Channels, Height, Width

        frames = frames.permute(
            0, 3, 1, 2
        )

        target = torch.tensor(
            target,
            dtype=torch.float32
        )

        return frames, target