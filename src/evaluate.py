import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader

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

os.makedirs(RESULT_DIR, exist_ok=True)


# =====================================================
# Test Loader
# =====================================================

test_dataset = ECGDataset("test")

test_loader = DataLoader(
    test_dataset,
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
# Metrics
# =====================================================

criterion = nn.MSELoss()

total_mse = 0

all_predictions = []

all_targets = []

all_inputs = []

all_attention = []

# =====================================================
# Evaluation
# =====================================================

with torch.no_grad():

    for x, y, _ in test_loader:

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        prediction,attention = model(
            x,
            teacher_forcing_ratio=0
        )

        mse = criterion(
            prediction,
            y
        )

        total_mse += mse.item()

        all_predictions.append(
            prediction.cpu().numpy()
        )
        all_attention.append(

            attention.cpu().numpy()

        )
        all_targets.append(
            y.cpu().numpy()
        )

        all_inputs.append(
            x.cpu().numpy()
        )


# =====================================================
# Metrics
# =====================================================

predictions = np.concatenate(
    all_predictions,
    axis=0
)

attention_maps = np.concatenate(

    all_attention,

    axis=0

)

targets = np.concatenate(
    all_targets,
    axis=0
)

inputs = np.concatenate(
    all_inputs,
    axis=0
)

print("Prediction shape :", predictions.shape)
print("Target shape     :", targets.shape)

# Compute metrics over the whole dataset
mse = np.mean(
    (predictions - targets) ** 2
)

rmse = np.sqrt(mse)

mae = np.mean(
    np.abs(
        predictions - targets
    )
)
r2 = r2_score(
    targets.reshape(-1),
    predictions.reshape(-1)
)
print("=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.6f}")
metrics = {

    "MSE": float(mse),

    "RMSE": float(rmse),

    "MAE": float(mae),

    "R2": float(r2)

}

with open(

    os.path.join(
        RESULT_DIR,
        "metrics.json"
    ),

    "w"

) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("Metrics saved.")
# =====================================================
# Save Predictions
# =====================================================

pred_df = pd.DataFrame({

    "Actual":

        targets.reshape(-1),

    "Predicted":

        predictions.reshape(-1)

})

pred_df.to_csv(

    os.path.join(
        RESULT_DIR,
        "predictions.csv"
    ),

    index=False

)

print("\nPredictions saved.")


# =====================================================
# Save Random Forecast Examples
# =====================================================

np.random.seed(42)

num_examples = min(20, len(inputs))

random_indices = np.random.choice(
    len(inputs),
    num_examples,
    replace=False
)

for idx, sample in enumerate(random_indices):

    plt.figure(figsize=(10, 4))

    input_signal = inputs[sample].squeeze()

    target_signal = targets[sample].squeeze()

    pred_signal = predictions[sample].squeeze()

    x_input = np.arange(len(input_signal))

    x_target = np.arange(
        len(input_signal),
        len(input_signal) + len(target_signal)
    )

    plt.plot(
        x_input,
        input_signal,
        label="Input ECG",
        linewidth=2
    )

    plt.plot(
        x_target,
        target_signal,
        label="Actual Future",
        linewidth=2
    )

    plt.plot(
        x_target,
        pred_signal,
        "--",
        label="Predicted Future",
        linewidth=2
    )

    plt.title(f"Forecast Sample {idx+1}")

    plt.xlabel("Time Step")

    plt.ylabel("Normalized ECG")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"forecast_{idx+1}.png"
        ),
        dpi=300
    )

    plt.close()


print(f"\nSaved {num_examples} forecast plots to:")
print(RESULT_DIR)
for i in range(5):

    plt.figure(figsize=(10,4))

    ...

    plt.savefig(

        os.path.join(

            RESULT_DIR,

            f"forecast_{i+1}.png"

        )

    )

    plt.show()

print("\nEvaluation Complete.")
with open(

    os.path.join(
        RESULT_DIR,
        "evaluation_report.txt"
    ),

    "w"

) as f:

    f.write("ECG Forecasting Report\n")

    f.write("=" * 40 + "\n\n")

    f.write(f"MSE : {mse:.6f}\n")

    f.write(f"RMSE : {rmse:.6f}\n")

    f.write(f"MAE : {mae:.6f}\n")

    f.write(f"R2 : {r2:.6f}\n\n")

    f.write("Model\n")

    f.write(f"Encoder : {ENCODER_TYPE}\n")

    f.write(f"Hidden Size : {HIDDEN_SIZE}\n")

    f.write(f"Layers : {NUM_LAYERS}\n")

    f.write(f"Forecast Length : {TARGET_LENGTH}\n")