import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from .models.seq2seq import Seq2Seq
from .config import (
    ENCODER_TYPE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    TARGET_LENGTH
)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models_saved",
    "best_seq2seq_model.pth"
)

SCALER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "scaler.pkl"
)

# Threshold obtained from early_detection.py
THRESHOLD = 0.51025385
STEP_THRESHOLD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "step_thresholds.npy"
)

if os.path.exists(STEP_THRESHOLD_PATH):

    STEP_THRESHOLDS = np.load(STEP_THRESHOLD_PATH)

else:

    STEP_THRESHOLDS = None
INPUT_LENGTH = 98
TARGET_LENGTH = 42

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

model = Seq2Seq(
    encoder_type=ENCODER_TYPE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    target_len=TARGET_LENGTH,
    dropout=DROPOUT
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

criterion = nn.MSELoss(reduction="none")

# ----------------------------------------------------
# Load Scaler
# ----------------------------------------------------

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ----------------------------------------------------
# Prediction Function
# ----------------------------------------------------

def predict_single_ecg(ecg_signal):

    """
    Predict whether a single ECG beat is normal or abnormal
    using forecasting error.
    """

    ecg_signal = np.asarray(ecg_signal)

    if len(ecg_signal) != 140:
        raise ValueError(
            "ECG must contain exactly 140 samples."
        )

    # ------------------------------------------------
    # Normalize
    # ------------------------------------------------

    ecg_signal = scaler.transform(
        ecg_signal.reshape(-1, 1)
    ).reshape(-1)

    input_signal = ecg_signal[:INPUT_LENGTH]

    target_signal = ecg_signal[INPUT_LENGTH:]

    x = torch.tensor(
        input_signal,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    y = torch.tensor(
        target_signal,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    with torch.no_grad():

        prediction, attention = model(
            x,
            teacher_forcing_ratio=0
        )

        # ---------------------------------------------
        # Point-wise forecast error
        # ---------------------------------------------

        loss = criterion(
            prediction,
            y
        )

        # Shape -> (42,)
        step_errors = (
            loss
            .mean(dim=2)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # Overall forecast error
        forecast_error = float(step_errors.mean())

    # ------------------------------------------------
    # Classification
    # ------------------------------------------------

    prediction_label = (
        "Abnormal"
        if forecast_error > THRESHOLD
        else "Normal"
    )

    # ------------------------------------------------
    # Earliest divergence
    # ------------------------------------------------

    earliest_step = "No divergence"

    if STEP_THRESHOLDS is not None:

        for i, err in enumerate(step_errors):

            if err > STEP_THRESHOLDS[i]:

                earliest_step = i + 1

                break

    # ------------------------------------------------
    # Confidence
    # ------------------------------------------------

    if prediction_label == "Abnormal":

        confidence = min(
            100.0,
            (forecast_error / THRESHOLD) * 100
        )

    else:

        confidence = max(
            0.0,
            ( 1 - forecast_error / THRESHOLD) * 100
        )

    return {

        "Prediction": prediction_label,

        "Forecast Error": forecast_error,

        "Threshold": THRESHOLD,

        "Confidence": confidence,

        "Earliest Divergence Step": earliest_step,

        "Observed ECG": input_signal,

        "Step Errors": step_errors.tolist(),

        "Step Thresholds": (
            STEP_THRESHOLDS.tolist()
            if STEP_THRESHOLDS is not None
            else None
        ),

        "Forecast": prediction.squeeze().cpu().numpy(),

        "Ground Truth": target_signal,

        "Attention": attention.squeeze().cpu().numpy()

    }
# ----------------------------------------------------
# Test the prediction on one ECG
# ----------------------------------------------------

if __name__ == "__main__":

    from .data_loader import load_ecg5000

    labels, signals = load_ecg5000()

    # Change this index to test different ECGs
    # Test a normal ECG
    # sample_index = np.where(labels == 1)[0][0]

    # Test an abnormal ECG
    sample_index = np.where(labels != 1)[0][0]

    sample = signals[sample_index]

    actual_label = labels[sample_index]

    result = predict_single_ecg(sample)

    print("\n" + "=" * 60)
    print("Single ECG Prediction")
    print("=" * 60)

    print(f"Actual Label             : {actual_label}")

    if actual_label == 1:
        print("Ground Truth             : Normal")
    else:
        print("Ground Truth             : Abnormal")

    print(f"\nPrediction               : {result['Prediction']}")

    print(f"Forecast Error           : {result['Forecast Error']:.6f}")

    print(f"Threshold                : {result['Threshold']:.6f}")

    print(f"Confidence               : {result['Confidence']:.2f}%")

    print(f"Earliest Divergence Step : {result['Earliest Divergence Step']}")

    print("\nFirst 10 Forecast Step Errors")

    print("-" * 40)

    for i, err in enumerate(result["Step Errors"][:10]):

        print(f"Step {i+1:2d} : {err:.6f}")

    print("\nPrediction Complete.")