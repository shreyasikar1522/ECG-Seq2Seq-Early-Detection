import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from .dataset import ECGDataset
from .models.seq2seq import Seq2Seq

from .config import (
    ENCODER_TYPE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    TARGET_LENGTH
)
# =====================================================
# Configuration
# =====================================================

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

RESULT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)
# =====================================================
# Datasets
# =====================================================

normal_dataset = ECGDataset(
    split="test_normal"
)

abnormal_dataset = ECGDataset(
    split="test_abnormal"
)

normal_loader = DataLoader(
    normal_dataset,
    batch_size=64,
    shuffle=False
)

abnormal_loader = DataLoader(
    abnormal_dataset,
    batch_size=64,
    shuffle=False
)
# =====================================================
# Load Model
# =====================================================

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
# =====================================================
# Forecast Error Computation
# =====================================================

criterion = nn.MSELoss(reduction="none")


def compute_forecast_errors(loader):

    errors = []
    step_errors_all = []
    model.eval()

    with torch.no_grad():

        for x, y, _ in loader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            prediction,attention_maps = model(
                x,
                teacher_forcing_ratio=0
            )

            # MSE for every timestep
            loss = criterion(
                prediction,
                y
            )

            # Average over sequence and feature dimensions
            sample_errors = loss.mean(dim=(1,2))

            errors.extend(
                sample_errors.cpu().numpy()
            )

# ----------------------------------------
# Save step-wise forecast errors
# ----------------------------------------

            step_errors = (
                loss
                .mean(dim=2)
                .cpu()
                .numpy()
            )

            step_errors_all.append(step_errors)

    step_errors_all = np.concatenate(
    step_errors_all,
    axis=0
   )

    return (
         np.array(errors),
         step_errors_all
    )
# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Forecasting-Based Early Anomaly Detection")
    print("=" * 60)

    print()

    print("Device:", DEVICE)

    print()

    print("Normal ECG Samples   :", len(normal_dataset))
    print("Abnormal ECG Samples :", len(abnormal_dataset))

    print()

    print("Computing forecast errors...")

    normal_errors,normal_step_errors= compute_forecast_errors(
        normal_loader
    )

    abnormal_errors,abnormal_step_errors= compute_forecast_errors(
        abnormal_loader
    )

    print()

    print("Normal Mean Error   :", normal_errors.mean())
    print("Abnormal Mean Error :", abnormal_errors.mean())
# =====================================================
# Plot Error Distribution
# =====================================================

plt.figure(figsize=(10, 5))

plt.hist(
    normal_errors,
    bins=40,
    alpha=0.7,
    label="Normal ECG"
)

plt.hist(
    abnormal_errors,
    bins=40,
    alpha=0.7,
    label="Abnormal ECG"
)

plt.xlabel("Forecast MSE")

plt.ylabel("Number of ECG Beats")

plt.title("Forecast Error Distribution")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "forecast_error_distribution.png"
    ),
    dpi=300
)

plt.show()
# =====================================================
# Threshold Selection
# =====================================================

threshold = normal_errors.mean() + 3 * normal_errors.std()
step_mean = normal_step_errors.mean(axis=0)

step_std = normal_step_errors.std(axis=0)

step_thresholds = step_mean + 3 * step_std
print("\nThreshold :", threshold)
# =====================================================
# Classification
# =====================================================

normal_predictions = normal_errors > threshold

abnormal_predictions = abnormal_errors > threshold

y_true = np.concatenate([

    np.zeros(len(normal_errors)),

    np.ones(len(abnormal_errors))

])

y_pred = np.concatenate([

    normal_predictions,

    abnormal_predictions

])
# =====================================================
# Metrics
# =====================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

print("\nClassification Results")
print("=" * 40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Normal", "Abnormal"]

)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_DIR,

        "confusion_matrix.png"

    ),

    dpi=300

)

plt.show()
# =====================================================
# ROC Curve
# =====================================================

fpr, tpr, thresholds = roc_curve(

    y_true,

    np.concatenate(

        [

            normal_errors,

            abnormal_errors

        ]

    )

)

roc_auc = auc(

    fpr,

    tpr

)

print(f"\nAUC Score : {roc_auc:.4f}")
plt.figure(figsize=(6,6))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {roc_auc:.4f}"

)

plt.plot(

    [0,1],

    [0,1],

    "--",

    color="gray"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULT_DIR,

        "roc_curve.png"

    ),

    dpi=300

)

plt.show()
with open(

    os.path.join(

        RESULT_DIR,

        "anomaly_detection_report.txt"

    ),

    "w"

) as f:

    f.write("Forecasting Based Anomaly Detection\n")

    f.write("="*50 + "\n\n")

    f.write(f"Threshold : {threshold:.6f}\n")

    f.write(f"Accuracy : {accuracy:.6f}\n")

    f.write(f"Precision : {precision:.6f}\n")

    f.write(f"Recall : {recall:.6f}\n")

    f.write(f"F1 Score : {f1:.6f}\n")

    f.write(f"AUC : {roc_auc:.6f}\n")

    f.write("\n")

    f.write("Confusion Matrix\n")

    f.write(str(cm))
    np.save(

    os.path.join(

        RESULT_DIR,

        "normal_errors.npy"

    ),

    normal_errors

)

np.save(

    os.path.join(

        RESULT_DIR,

        "abnormal_errors.npy"

    ),

    abnormal_errors

)
np.save(

    os.path.join(
        RESULT_DIR,
        "step_thresholds.npy"
    ),

    step_thresholds

)