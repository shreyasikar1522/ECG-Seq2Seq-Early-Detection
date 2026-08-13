import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch

from .models.seq2seq import Seq2Seq
from .data_loader import load_ecg5000
from .config import (
    ENCODER_TYPE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    TARGET_LENGTH
)

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

RESULT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

INPUT_LENGTH = 98

# ---------------------------------------------------
# Load model
# ---------------------------------------------------

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

# ---------------------------------------------------
# Load scaler
# ---------------------------------------------------

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------------------
# Load ECG
# ---------------------------------------------------

labels, signals = load_ecg5000()

# Change this index whenever you want
sample_index = np.where(labels != 1)[0][0]

signal = signals[sample_index]

label = labels[sample_index]

signal = scaler.transform(
    signal.reshape(-1,1)
).reshape(-1)

input_signal = signal[:98]

target_signal = signal[98:]

x = torch.tensor(
    input_signal,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(-1).to(DEVICE)

# ---------------------------------------------------
# Predict
# ---------------------------------------------------

with torch.no_grad():

    prediction, attention = model(
        x,
        teacher_forcing_ratio=0
    )

prediction = prediction.squeeze().cpu().numpy()

attention = attention.squeeze().cpu().numpy()

# Shape

# (42,98)

print("Attention shape:", attention.shape)

# ---------------------------------------------------
# Heatmap
# ---------------------------------------------------

plt.figure(figsize=(10,6))

plt.imshow(

    attention,

    aspect="auto",

    origin="lower",

    cmap="viridis"

)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel(
    "Observed ECG Samples"
)

plt.ylabel(
    "Forecast Step"
)

plt.title(
    "Bahdanau Attention Heatmap"
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_DIR,

        "attention_heatmap.png"

    ),

    dpi=300

)

plt.show()

# ---------------------------------------------------
# Average attention
# ---------------------------------------------------

avg_attention = attention.mean(axis=0)

plt.figure(figsize=(12,4))

plt.plot(
    avg_attention,
    linewidth=2
)

plt.xlabel(
    "Input ECG Sample"
)

plt.ylabel(
    "Average Attention"
)

plt.title(
    "Average Attention over Forecast Horizon"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_DIR,

        "average_attention.png"

    ),

    dpi=300

)

plt.show()

# ---------------------------------------------------
# ECG + Attention
# ---------------------------------------------------

fig, ax1 = plt.subplots(figsize=(12,4))

ax1.plot(
    input_signal,
    label="Observed ECG",
    linewidth=2
)

ax1.set_xlabel("Sample")

ax1.set_ylabel("Normalized ECG")

ax2 = ax1.twinx()

ax2.plot(

    avg_attention,

    "--",

    linewidth=2,

    label="Average Attention"

)

ax2.set_ylabel("Attention")

fig.tight_layout()

plt.title(
    "Observed ECG with Average Attention"
)

plt.savefig(

    os.path.join(

        RESULT_DIR,

        "ecg_attention_overlay.png"

    ),

    dpi=300

)

plt.show()

print("\nSaved attention visualizations to:")
print(RESULT_DIR)
# ============================================================
# Complete Explainability Dashboard
# ============================================================

import torch.nn as nn

criterion = nn.MSELoss(reduction="none")

# Ground truth tensor
y = torch.tensor(
    target_signal,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(-1).to(DEVICE)

with torch.no_grad():

    pred_tensor, attention = model(
        x,
        teacher_forcing_ratio=0
    )

loss = criterion(
    pred_tensor,
    y
)

step_errors = (
    loss
    .mean(dim=2)
    .squeeze(0)
    .cpu()
    .numpy()
)

prediction = pred_tensor.squeeze().cpu().numpy()

attention = attention.squeeze().cpu().numpy()

avg_attention = attention.mean(axis=0)

# ----------------------------------------------------
# Earliest divergence
# ----------------------------------------------------

threshold_path = os.path.join(
    RESULT_DIR,
    "step_thresholds.npy"
)

step_thresholds = None

if os.path.exists(threshold_path):

    step_thresholds = np.load(threshold_path)

earliest = None

if step_thresholds is not None:

    for i, err in enumerate(step_errors):

        if err > step_thresholds[i]:

            earliest = i + 1
            break

# ----------------------------------------------------
# Figure
# ----------------------------------------------------

fig = plt.figure(
    figsize=(15,10)
)

# ----------------------------------------------------
# Input ECG
# ----------------------------------------------------

ax1 = plt.subplot(2,2,1)

ax1.plot(
    input_signal,
    linewidth=2
)

ax1.set_title(
    "Observed ECG"
)

ax1.set_xlabel(
    "Input Sample"
)

ax1.set_ylabel(
    "Normalized ECG"
)

ax1.grid(alpha=0.3)

# ----------------------------------------------------
# Forecast
# ----------------------------------------------------

ax2 = plt.subplot(2,2,2)

ax2.plot(
    target_signal,
    label="Ground Truth",
    linewidth=2
)

ax2.plot(
    prediction,
    "--",
    label="Forecast",
    linewidth=2
)

ax2.set_title(
    "Forecast vs Ground Truth"
)

ax2.set_xlabel(
    "Forecast Step"
)

ax2.legend()

ax2.grid(alpha=0.3)

# ----------------------------------------------------
# Attention
# ----------------------------------------------------

ax3 = plt.subplot(2,2,3)

im = ax3.imshow(

    attention,

    aspect="auto",

    origin="lower",

    cmap="viridis"

)

ax3.set_title(
    "Bahdanau Attention"
)

ax3.set_xlabel(
    "Observed Samples"
)

ax3.set_ylabel(
    "Forecast Step"
)

plt.colorbar(
    im,
    ax=ax3,
    fraction=0.046
)

# ----------------------------------------------------
# Forecast Error
# ----------------------------------------------------

ax4 = plt.subplot(2,2,4)

ax4.plot(
    step_errors,
    linewidth=2,
    label="Forecast Error"
)

if step_thresholds is not None:

    ax4.plot(
        step_thresholds,
        "--",
        linewidth=2,
        label="Threshold"
    )

if earliest is not None:

    ax4.axvline(
        earliest-1,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Earliest Divergence ({earliest})"
    )

ax4.set_title(
    "Forecast Error Evolution"
)

ax4.set_xlabel(
    "Forecast Step"
)

ax4.set_ylabel(
    "MSE"
)

ax4.legend()

ax4.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULT_DIR,
        "explainability_dashboard.png"
    ),

    dpi=300

)

plt.show()

print("\nDashboard saved to:")
print(
    os.path.join(
        RESULT_DIR,
        "explainability_dashboard.png"
    )
)