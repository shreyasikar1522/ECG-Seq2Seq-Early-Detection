# ❤️ ECG Seq2Seq Early Detection

### 🏥 Explainable Healthcare AI for Early ECG Anomaly Detection

An end-to-end **deep learning system for ECG forecasting and early cardiac anomaly detection** using a **Seq2Seq Encoder–Decoder architecture with BiLSTM, LSTM, and Bahdanau Attention**.

The system learns the dynamics of normal ECG signals, forecasts future ECG samples, and identifies abnormal heartbeats when the observed future signal significantly deviates from the model's forecast.

> ⚠️ **Research / educational project:** This system is intended for experimentation and demonstration and is **not a clinical diagnostic tool**.

---

## ✨ What Makes This Project Interesting?

Most ECG anomaly detection systems directly classify an ECG as normal or abnormal.

This project takes a different approach:

```text
                  ECG Signal
                      │
                      ▼
              ┌───────────────┐
              │  Input Context │
              │   98 samples   │
              └───────┬───────┘
                      │
                      ▼
          🧠 BiLSTM Encoder
                      │
                      ▼
           🔎 Bahdanau Attention
                      │
                      ▼
            🧠 LSTM Decoder
                      │
                      ▼
              🔮 Forecast
              42 future samples
                      │
                      ▼
          Compare with actual ECG
                      │
                      ▼
             📉 Forecast Error
                      │
              ┌───────┴───────┐
              │               │
           Normal          Abnormal
              │               │
              ▼               ▼
          🟢 Normal      🔴 Anomaly
                              │
                              ▼
                    🚩 Earliest Divergence
```

Instead of asking only:

> **"Is this ECG abnormal?"**

the system also asks:

> **"When does the abnormal ECG first start behaving differently from what the model expects?"**

---

# 🚀 Key Features

### 🧠 Deep Learning Forecasting

- Seq2Seq **Encoder–Decoder** architecture
- **Bidirectional LSTM Encoder**
- **LSTM Decoder**
- **Bahdanau Attention**
- Multi-step ECG forecasting
- Teacher forcing during training

### 🚨 Anomaly Detection

- Learns normal ECG dynamics
- Uses forecasting error for anomaly detection
- Global anomaly threshold
- Per-step forecasting errors
- Normal vs abnormal classification

### ⏱️ Early Detection

A key feature of the project is **Earliest Divergence Detection**.

Instead of waiting until the complete forecast horizon has passed, the system examines the error at each forecast step and identifies the first point where the ECG significantly diverges from expected behaviour.

```text
Forecast Step

1   2   3   4   5   6   7   8   9   ...

│   │   │   │   │   │   │   │
                    ↑
             Earliest Divergence
```

This provides an interpretable estimate of **how early abnormal behaviour becomes detectable**.

### 🔍 Explainable AI

The model provides:

- Bahdanau attention heatmaps
- Forecast error evolution
- Earliest divergence step
- Forecast vs ground-truth comparison
- Confidence estimation

### 🖥️ Interactive Application

Built an ECG-themed **Streamlit dashboard** featuring:

- Single ECG prediction
- ECG waveform visualization
- Forecast vs ground truth
- Attention heatmaps
- Forecast error evolution
- Earliest divergence visualization
- Model insights
- AI interpretation
- Clinical-style PDF report generation

---

# 🏗️ Model Architecture

```text
                    INPUT ECG
                   98 samples
                       │
                       ▼
              ┌─────────────────┐
              │  BiLSTM Encoder │
              └────────┬────────┘
                       │
               Hidden Representations
                       │
                       ▼
            ┌──────────────────────┐
            │  Bahdanau Attention  │
            └──────────┬───────────┘
                       │
                       ▼
               ┌──────────────┐
               │ LSTM Decoder │
               └──────┬───────┘
                      │
                      ▼
              42-step Forecast
                      │
                      ▼
             Compare with Actual
                      │
                      ▼
              Forecast Error
                      │
             ┌────────┴────────┐
             ▼                 ▼
          🟢 Normal         🔴 Abnormal
                               │
                               ▼
                     Earliest Divergence
```

---

# 📊 Data Processing

The project uses the **ECG5000 dataset**.

Each ECG heartbeat contains:

```text
140 samples
```

The signal is divided into:

| Component | Samples |
|---|---:|
| Input Context | 98 |
| Forecast Horizon | 42 |
| Total ECG | 140 |

The model receives:

```text
98 observed samples
```

and predicts:

```text
42 future samples
```

---

# 🔄 Preprocessing Pipeline

```text
ECG5000 Dataset
       │
       ▼
Load ECG Signals
       │
       ▼
Separate Normal / Abnormal
       │
       ▼
Normal Training Data
       │
       ▼
Train / Validation / Test
       │
       ▼
StandardScaler
       │
       ▼
98-sample Input + 42-sample Target
       │
       ▼
PyTorch Dataset / DataLoader
```

### Important preprocessing principle

The scaler is fitted **only on the normal training data** to avoid information leakage.

---

# 🧪 Training Strategy

The model is trained primarily on **normal ECG signals**.

The intuition is:

> Learn what normal ECG behaviour looks like → forecast what should happen next → identify abnormalities when the actual signal deviates substantially from the forecast.

This makes the system suitable for **forecasting-based anomaly detection**.

---

# 📉 Anomaly Detection

For each forecast step:

```text
Forecast Error = Difference between
                 predicted ECG and
                 actual ECG
```

The errors are evaluated against learned thresholds.

The overall forecast error determines the final classification:

```text
              Forecast Error
                    │
          ┌─────────┴─────────┐
          │                   │
       Below                Above
      Threshold             Threshold
          │                   │
          ▼                   ▼
      🟢 NORMAL            🔴 ABNORMAL
```

---

# ⏱️ Earliest Divergence Detection

For abnormal ECGs, the system evaluates the forecasting error **step-by-step**.

Example:

```text
Step       Error       Threshold
────────────────────────────────
1          0.12        0.40
2          0.18        0.42
3          0.21        0.43
4          0.29        0.45
5          0.61        0.47   ← 🚩 Divergence
6          0.72        0.49
7          0.91        0.51
```

The system reports:

```text
Earliest Divergence = Step 5
```

This provides more information than a simple binary anomaly label.

---

# 🔎 Explainability with Bahdanau Attention

The decoder uses **Bahdanau Attention** to determine which portions of the observed ECG are most relevant while forecasting each future step.

The application visualizes these attention weights as a heatmap.

```text
                Observed ECG Samples
        ───────────────────────────────►

Forecast
Step 1       ░░░▒▒▓▓████▓▒░░
Step 2       ░░▒▒▓██████▓▒░░
Step 3       ░▒▓████████▓▒░░
Step 4       ░░▒▓██████▓▒░░
  ⋮
Step 42      ░░░▒▒▓▓██▓▒░░░
```

Brighter regions represent stronger attention.

This provides insight into **which portions of the ECG influenced the forecast**.

---

# 🖥️ Streamlit Application

The project includes an interactive web application.

### Dashboard capabilities

❤️ ECG waveform visualization

🔮 Future ECG forecasting

🚨 Normal / Abnormal prediction

📊 Forecast error metrics

⏱️ Earliest divergence detection

🔥 Bahdanau attention heatmap

📉 Error evolution

🧠 Model interpretation

📄 Clinical-style PDF report

---

# 📄 Automated PDF Reports

The application can generate a downloadable report containing:

- Prediction summary
- Forecast error
- Confidence score
- Earliest divergence
- AI interpretation
- Forecast vs ground truth
- Forecast error evolution
- Bahdanau attention heatmap
- Model architecture information
- ECG5000 dataset information
- Date and time of analysis

---

# 🗂️ Project Structure

```text
ECG-Seq2Seq-Early-Detection/
│
├── app.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── early_detection.py
│   ├── predict_single.py
│   ├── visualize_attention.py
│   ├── report_generator.py
│   │
│   └── models/
│       └── seq2seq.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models_saved/
│   └── best_seq2seq_model.pth
│
├── results/
│   └── step_thresholds.npy
│
├── fonts/
│   └── OpenSans/
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Technologies Used

### Programming

- 🐍 Python

### Deep Learning

- 🔥 PyTorch
- LSTM
- BiLSTM
- Seq2Seq
- Encoder–Decoder Networks
- Bahdanau Attention

### Data & Scientific Computing

- NumPy
- Pandas
- Scikit-learn

### Visualization

- Plotly
- Matplotlib

### Application

- Streamlit

### Reporting

- ReportLab

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ECG-Seq2Seq-Early-Detection.git

cd ECG-Seq2Seq-Early-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install numpy pandas scikit-learn torch matplotlib plotly streamlit reportlab
```

---

# ▶️ Running the Application

From the project root:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually:

```text
http://localhost:8501
```

---

# 🧪 Running Individual Components

### Preprocessing

```bash
python -m src.preprocessing
```

### Single ECG Prediction

```bash
python -m src.predict_single
```

### Early Detection

```bash
python -m src.early_detection
```

### Attention Visualization

```bash
python -m src.visualize_attention
```

### Generate PDF Report

```bash
python -m src.report_generator
```

---

# 📈 End-to-End Pipeline

```text
              ECG5000 Dataset
                     │
                     ▼
              Preprocessing
                     │
                     ▼
             Normal ECG Training
                     │
                     ▼
              Seq2Seq Training
                     │
                     ▼
           BiLSTM Encoder
                     │
                     ▼
           Bahdanau Attention
                     │
                     ▼
             LSTM Decoder
                     │
                     ▼
           42-Step Forecast
                     │
                     ▼
             Forecast Error
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       Normal                Abnormal
                                │
                                ▼
                     Earliest Divergence
                                │
                                ▼
                       Explainability
                                │
                                ▼
                       Streamlit Dashboard
                                │
                                ▼
                         PDF Report
```

---

# 🎯 Project Objectives

- [x] ECG preprocessing pipeline
- [x] Normal-only training strategy
- [x] Seq2Seq encoder–decoder architecture
- [x] BiLSTM encoder
- [x] LSTM decoder
- [x] Bahdanau Attention
- [x] Multi-step ECG forecasting
- [x] Forecasting-based anomaly detection
- [x] Per-step error analysis
- [x] Earliest divergence detection
- [x] Single ECG prediction
- [x] Attention visualization
- [x] Interactive Streamlit dashboard
- [x] Automated PDF reporting

---

# 🔬 Research Direction

The current system establishes a forecasting-based framework for explainable ECG anomaly detection.

Potential future extensions include:

- 📊 Larger and more diverse ECG datasets
- 🧠 Transformer-based forecasting
- 🔀 Hybrid statistical + deep learning anomaly detection
- 📈 More robust uncertainty estimation
- 🫀 Patient-specific adaptation
- ⚡ Real-time ECG stream processing
- 🏥 Clinical validation on external datasets
- 📱 Deployment for real-time monitoring

---

# ⚠️ Disclaimer

This project is intended for **research, educational, and demonstration purposes only**.

It has not been clinically validated and should **not** be used for medical diagnosis, treatment decisions, or emergency healthcare.

---

# 👩‍💻 Author

**Shreyasi Kar**

🎓 M.Sc. Data Science — Chennai Mathematical Institute

Interested in:

- 🤖 Machine Learning
- 🧠 Deep Learning
- 📊 Statistical Learning
- 🏥 Healthcare AI
- 🔍 Explainable AI
- 📈 Time Series Analysis
