import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from .dataset import get_dataloader
from .models.seq2seq import Seq2Seq


from .config import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    ENCODER_TYPE,
    TARGET_LENGTH,
    PATIENCE,
    TEACHER_FORCING_RATIO,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MIN_LEARNING_RATE,
    MAX_GRAD_NORM
)

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models_saved"
)

RESULT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================================
# Reproducibility
# =====================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# =====================================================
# Device
# =====================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print("Using device:", DEVICE)


# =====================================================
# Data Loaders
# =====================================================

train_loader = get_dataloader(

    split="train",

    batch_size=BATCH_SIZE,

    shuffle=True

)

val_loader = get_dataloader(

    split="val",

    batch_size=BATCH_SIZE,

    shuffle=False

)


# =====================================================
# Model
# =====================================================

model = Seq2Seq(

    encoder_type=ENCODER_TYPE,

    hidden_size=HIDDEN_SIZE,

    num_layers=NUM_LAYERS,

    target_len=TARGET_LENGTH,

    dropout=DROPOUT

).to(DEVICE)


# =====================================================
# Loss
# =====================================================

criterion = nn.MSELoss()


# =====================================================
# Optimizer
# =====================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# =====================================================
# Scheduler
# =====================================================

scheduler = ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=SCHEDULER_FACTOR,

    patience=SCHEDULER_PATIENCE,

    min_lr=MIN_LEARNING_RATE


)


# =====================================================
# One Training Epoch
# =====================================================

def train_epoch():

    model.train()

    running_loss = 0

    progress = tqdm(

        train_loader,

        desc="Training",

        leave=False

    )

    for x, y, _ in progress:

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        optimizer.zero_grad()

        prediction,_ = model(

            x,

            target=y,

            teacher_forcing_ratio=TEACHER_FORCING_RATIO

        )

        loss = criterion(

            prediction,

            y

        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=MAX_GRAD_NORM

        )

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(

            loss=f"{loss.item():.5f}"

        )

    epoch_loss = running_loss / len(train_loader)

    return epoch_loss


# =====================================================
# Validation
# =====================================================

def validate():

    model.eval()

    running_loss = 0

    with torch.no_grad():

        for x, y, _ in val_loader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            prediction,_ = model(

                x,

                target=None,

                teacher_forcing_ratio=0

            )

            loss = criterion(

                prediction,

                y

            )

            running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader)

    return epoch_loss
# =====================================================
# Training Loop
# =====================================================

def train():

    set_seed()

    best_val_loss = float("inf")

    train_losses = []

    val_losses = []

    patience_counter = 0

    print("=" * 60)
    print("Training Started")
    print("=" * 60)

    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss = train_epoch()

        val_loss = validate()

        scheduler.step(val_loss)

        train_losses.append(train_loss)

        val_losses.append(val_loss)

        print(f"Train Loss : {train_loss:.6f}")
        print(f"Val Loss   : {val_loss:.6f}")

        # ---------------------------------------
        # Save Best Model
        # ---------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(

                model.state_dict(),

                os.path.join(

                    MODEL_DIR,

                    "best_seq2seq_model.pth"

                )

            )

            print("Best model saved.")

        else:

            patience_counter += 1

            print(f"No improvement ({patience_counter}/{PATIENCE})")

        # ---------------------------------------
        # Early Stopping
        # ---------------------------------------

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

    return train_losses, val_losses

# =====================================================
# Plot Loss Curves
# =====================================================

def plot_losses(train_losses, val_losses):

    plt.figure(figsize=(8,5))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("MSE Loss")

    plt.title("Training Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            RESULT_DIR,
            "loss_curve.png"
        )

    )

    plt.show()
# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    train_losses, val_losses = train()

    history = pd.DataFrame({

        "Epoch": range(1, len(train_losses) + 1),

        "Train Loss": train_losses,

        "Validation Loss": val_losses

    })

    history.to_csv(

        os.path.join(

            RESULT_DIR,

            "training_history.csv"

        ),

        index=False

    )

    print("Training history saved.")

    plot_losses(
        train_losses,
        val_losses
    )

    print("\nTraining Finished Successfully!")

    print("\nBest model saved at:")

    print(
        os.path.join(
            MODEL_DIR,
            "best_seq2seq_model.pth"
        )
    )
total_params = sum(

    p.numel()

    for p in model.parameters()

)

with open(

    os.path.join(
        RESULT_DIR,
        "model_summary.txt"
    ),

    "w"

) as f:

    f.write("ECG Forecasting Model\n\n")

    f.write(f"Encoder : {ENCODER_TYPE}\n")

    f.write(f"Hidden Size : {HIDDEN_SIZE}\n")

    f.write(f"Layers : {NUM_LAYERS}\n")

    f.write(f"Dropout : {DROPOUT}\n")

    f.write(f"Input Length : 98\n")

    f.write(f"Forecast Length : {TARGET_LENGTH}\n")

    f.write(f"Parameters : {total_params}\n")


config = {

    "Encoder": ENCODER_TYPE,

    "Hidden Size": HIDDEN_SIZE,

    "Layers": NUM_LAYERS,

    "Dropout": DROPOUT,

    "Learning Rate": LEARNING_RATE,

    "Batch Size": BATCH_SIZE,

    "Epochs": EPOCHS,

    "Teacher Forcing": TEACHER_FORCING_RATIO

}

with open(

    os.path.join(
        RESULT_DIR,
        "experiment_config.json"
    ),

    "w"

) as f:

    json.dump(
        config,
        f,
        indent=4
    )