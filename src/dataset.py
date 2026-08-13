import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


# =====================================================
# Processed Data Directory
# =====================================================

PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed"
)


# =====================================================
# ECG Dataset
# =====================================================

class ECGDataset(Dataset):

    VALID_SPLITS = [
        "train",
        "val",
        "test_normal",
        "test_abnormal"
    ]

    def __init__(self, split="train"):

        if split not in self.VALID_SPLITS:

            raise ValueError(

                f"Unknown split '{split}'. "

                f"Choose from {self.VALID_SPLITS}"

            )

        x_path = os.path.join(
            PROCESSED_DIR,
            f"{split}_x.npy"
        )

        y_path = os.path.join(
            PROCESSED_DIR,
            f"{split}_y.npy"
        )

        label_path = os.path.join(
            PROCESSED_DIR,
            f"{split}_labels.npy"
        )

        if not os.path.exists(x_path):
            raise FileNotFoundError(x_path)

        if not os.path.exists(y_path):
            raise FileNotFoundError(y_path)

        if not os.path.exists(label_path):
            raise FileNotFoundError(label_path)

        self.x = np.load(x_path)

        self.y = np.load(y_path)

        self.labels = np.load(label_path)

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        x = torch.tensor(
            self.x[idx],
            dtype=torch.float32
        ).unsqueeze(-1)

        y = torch.tensor(
            self.y[idx],
            dtype=torch.float32
        ).unsqueeze(-1)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return x, y, label


# =====================================================
# Dataloader
# =====================================================

def get_dataloader(
    split="train",
    batch_size=64,
    shuffle=True
):

    dataset = ECGDataset(split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    for split in [

        "train",

        "val",

        "test_normal",

        "test_abnormal"

    ]:

        loader = get_dataloader(
            split=split,
            batch_size=64,
            shuffle=False
        )

        x, y, labels = next(iter(loader))

        print("=" * 50)

        print(split.upper())

        print("Samples :", len(loader.dataset))

        print("Input Shape :", x.shape)

        print("Target Shape:", y.shape)

        print("Labels Shape:", labels.shape)

        print()