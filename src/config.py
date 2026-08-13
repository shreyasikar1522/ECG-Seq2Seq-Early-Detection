"""
Configuration file for ECG Seq2Seq Forecasting Project
"""

# =====================================================
# Reproducibility
# =====================================================

SEED = 42

# =====================================================
# Data
# =====================================================

INPUT_LENGTH = 98
TARGET_LENGTH = 42

# =====================================================
# Model
# =====================================================

ENCODER_TYPE = "bilstm"

INPUT_SIZE = 1

OUTPUT_SIZE = 1

HIDDEN_SIZE = 32

NUM_LAYERS = 2

DROPOUT = 0.4

# =====================================================
# Training
# =====================================================

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 10

TEACHER_FORCING_RATIO = 0.7

# =====================================================
# Learning Rate Scheduler
# =====================================================

SCHEDULER_FACTOR = 0.2

SCHEDULER_PATIENCE = 2

MIN_LEARNING_RATE = 1e-6

# =====================================================
# Gradient Clipping
# =====================================================

MAX_GRAD_NORM = 1.0
