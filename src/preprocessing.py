import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_loader import load_ecg5000


# -----------------------------
# Constants
# -----------------------------

NORMAL_LABEL = 1
INPUT_FRAC = 0.70

PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


# -----------------------------
# Split ECG into Input + Target
# -----------------------------

def split_input_target(signals, input_frac=INPUT_FRAC):

    seq_len = signals.shape[1]

    k = int(seq_len * input_frac)

    x = signals[:, :k]

    y = signals[:, k:]

    return x, y, k


# -----------------------------
# Main preprocessing pipeline
# -----------------------------

def preprocess_data(
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    input_frac=INPUT_FRAC
):

    labels, signals = load_ecg5000()

    x_all, y_all, k = split_input_target(
        signals,
        input_frac
    )

    # -----------------------------------
    # Separate Normal / Abnormal
    # -----------------------------------

    normal_mask = labels == NORMAL_LABEL
    abnormal_mask = labels != NORMAL_LABEL

    normal_indices = np.where(normal_mask)[0]
    abnormal_indices = np.where(abnormal_mask)[0]

    # -----------------------------------
    # Train / Validation / Normal Test
    # -----------------------------------

    train_idx, temp_idx = train_test_split(
        normal_indices,
        test_size=val_size + test_size,
        random_state=random_state,
        shuffle=True
    )

    relative_test_size = test_size / (val_size + test_size)

    val_idx, test_normal_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True
    )

    # -----------------------------------
    # Standardization
    # Fit ONLY on train-normal
    # -----------------------------------

    scaler = StandardScaler()

    scaler.fit(
        signals[train_idx].reshape(-1, 1)
    )

    def transform(arr):

        original_shape = arr.shape

        arr = scaler.transform(
            arr.reshape(-1, 1)
        )

        return arr.reshape(original_shape)

    # -----------------------------------
    # Train
    # -----------------------------------

    train_x = transform(
        x_all[train_idx]
    )

    train_y = transform(
        y_all[train_idx]
    )

    train_labels = labels[train_idx]

    # -----------------------------------
    # Validation
    # -----------------------------------

    val_x = transform(
        x_all[val_idx]
    )

    val_y = transform(
        y_all[val_idx]
    )

    val_labels = labels[val_idx]

    # -----------------------------------
    # Normal Test
    # -----------------------------------

    test_normal_x = transform(
        x_all[test_normal_idx]
    )

    test_normal_y = transform(
        y_all[test_normal_idx]
    )

    test_normal_labels = labels[
        test_normal_idx
    ]

    # -----------------------------------
    # Abnormal Test
    # -----------------------------------

    test_abnormal_x = transform(
        x_all[abnormal_indices]
    )

    test_abnormal_y = transform(
        y_all[abnormal_indices]
    )

    test_abnormal_labels = labels[
        abnormal_indices
    ]

    # -----------------------------------
    # Save Train
    # -----------------------------------

    np.save(
        os.path.join(PROCESSED_DIR, "train_x.npy"),
        train_x
    )

    np.save(
        os.path.join(PROCESSED_DIR, "train_y.npy"),
        train_y
    )

    np.save(
        os.path.join(PROCESSED_DIR, "train_labels.npy"),
        train_labels
    )

    # -----------------------------------
    # Save Validation
    # -----------------------------------

    np.save(
        os.path.join(PROCESSED_DIR, "val_x.npy"),
        val_x
    )

    np.save(
        os.path.join(PROCESSED_DIR, "val_y.npy"),
        val_y
    )

    np.save(
        os.path.join(PROCESSED_DIR, "val_labels.npy"),
        val_labels
    )

    # -----------------------------------
    # Save Normal Test
    # -----------------------------------

    np.save(
        os.path.join(PROCESSED_DIR, "test_normal_x.npy"),
        test_normal_x
    )

    np.save(
        os.path.join(PROCESSED_DIR, "test_normal_y.npy"),
        test_normal_y
    )

    np.save(
        os.path.join(PROCESSED_DIR, "test_normal_labels.npy"),
        test_normal_labels
    )

    # -----------------------------------
    # Save Abnormal Test
    # -----------------------------------

    np.save(
        os.path.join(PROCESSED_DIR, "test_abnormal_x.npy"),
        test_abnormal_x
    )

    np.save(
        os.path.join(PROCESSED_DIR, "test_abnormal_y.npy"),
        test_abnormal_y
    )

    np.save(
        os.path.join(PROCESSED_DIR, "test_abnormal_labels.npy"),
        test_abnormal_labels
    )

    # -----------------------------------
    # Save Scaler
    # -----------------------------------

    with open(
        os.path.join(PROCESSED_DIR, "scaler.pkl"),
        "wb"
    ) as f:

        pickle.dump(
            scaler,
            f
        )

    return {

        "train_x": train_x,
        "train_y": train_y,
        "train_labels": train_labels,

        "val_x": val_x,
        "val_y": val_y,
        "val_labels": val_labels,

        "test_normal_x": test_normal_x,
        "test_normal_y": test_normal_y,
        "test_normal_labels": test_normal_labels,

        "test_abnormal_x": test_abnormal_x,
        "test_abnormal_y": test_abnormal_y,
        "test_abnormal_labels": test_abnormal_labels,

        "k": k
    }


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":

    data = preprocess_data()

    print("=" * 50)
    print("Preprocessing Complete")
    print("=" * 50)

    print()

    print("Input Length :", data["k"])

    print()

    print("Train X :", data["train_x"].shape)
    print("Train Y :", data["train_y"].shape)

    print()

    print("Validation X :", data["val_x"].shape)
    print("Validation Y :", data["val_y"].shape)

    print()

    print("Normal Test X :", data["test_normal_x"].shape)
    print("Normal Test Y :", data["test_normal_y"].shape)

    print()

    print("Abnormal Test X :", data["test_abnormal_x"].shape)
    print("Abnormal Test Y :", data["test_abnormal_y"].shape)

    print()

    print("Files saved to")
    print(PROCESSED_DIR)