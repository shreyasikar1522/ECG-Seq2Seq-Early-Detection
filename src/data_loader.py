import os
import numpy as np
import pandas as pd

# Path to the raw dataset folder
RAW_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "raw"
)

SEQ_LEN = 140  # Each ECG beat has 140 timesteps


def load_single_file(path):
    """
    Reads a single ECG5000 file.

    Parameters
    ----------
    path : str
        Path to ECG5000_TRAIN.txt or ECG5000_TEST.txt

    Returns
    -------
    labels : numpy.ndarray
        Shape: (N,)
        ECG class labels (1-5)

    signals : numpy.ndarray
        Shape: (N, 140)
        ECG waveform values
    """

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    labels = df.iloc[:, 0].astype(int).values
    signals = df.iloc[:, 1:].astype(np.float32).values

    return labels, signals


def load_ecg5000():
    """
    Loads the complete ECG5000 dataset.

    Combines the TRAIN and TEST files into one dataset.
    We create our own train/validation/test split later.

    Returns
    -------
    labels : ndarray
        Shape (5000,)

    signals : ndarray
        Shape (5000, 140)
    """

    train_path = os.path.join(
        RAW_DIR,
        "ECG5000_TRAIN.txt"
    )

    test_path = os.path.join(
        RAW_DIR,
        "ECG5000_TEST.txt"
    )

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Could not find:\n{train_path}"
        )

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Could not find:\n{test_path}"
        )

    train_labels, train_signals = load_single_file(train_path)
    test_labels, test_signals = load_single_file(test_path)

    labels = np.concatenate(
        [train_labels, test_labels],
        axis=0
    )

    signals = np.concatenate(
        [train_signals, test_signals],
        axis=0
    )

    return labels, signals


if __name__ == "__main__":

    labels, signals = load_ecg5000()

    print("=" * 50)
    print("ECG5000 Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Signals Shape : {signals.shape}")
    print(f"Labels Shape  : {labels.shape}")

    print("\nSequence Length :", signals.shape[1])

    print("\nUnique Classes :", np.unique(labels))

    print("\nClass Distribution")

    unique, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")

    print("\nFirst ECG Signal")

    print(signals[0])

    print("\nFirst Label")

    print(labels[0])