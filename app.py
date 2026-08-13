import streamlit as st
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from src.report_generator import generate_pdf_report
from src.predict_single import predict_single_ecg
from src.data_loader import load_ecg5000


# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="ECG AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================
# Custom CSS
# ======================================================

st.markdown(
"""
<style>

.stApp{

background-color:#07141F;

}

h1,h2,h3,h4{

color:white;

}

.metric-card{

background:#102332;

padding:20px;

border-radius:18px;

border:1px solid #2BC0E4;

box-shadow:0px 0px 12px rgba(0,255,200,0.15);

text-align:center;

}

.pred-normal{

background:#0E4732;

padding:25px;

border-radius:18px;

color:white;

font-size:28px;

font-weight:bold;

text-align:center;

border:2px solid #29D391;

}

.pred-abnormal{

background:#5C1D24;

padding:25px;

border-radius:18px;

color:white;

font-size:28px;

font-weight:bold;

text-align:center;

border:2px solid #FF6B6B;

}

.small{

font-size:18px;

color:#C8D6DF;

}

</style>
""",
unsafe_allow_html=True
)

# ======================================================
# Title
# ======================================================

st.markdown(
"""
# ❤️ ECG Forecasting & Early Anomaly Detection

### *Seq2Seq Forecasting • Bahdanau Attention • Explainable AI*
"""
)
st.success(
    "🟢 Model Loaded Successfully"
)
st.divider()

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("⚙ Controls")

mode = st.sidebar.radio(

"Choose ECG Source",

[
"Dataset Sample",

"Upload ECG"
]

)

labels, signals = load_ecg5000()

sample = None

true_label = None

# ------------------------------------------------------

if mode == "Dataset Sample":

    sample_index = st.sidebar.slider(

        "Dataset Sample",

        0,

        len(signals)-1,

        0

    )

    sample = signals[sample_index]

    true_label = labels[sample_index]

else:

    uploaded = st.sidebar.file_uploader(

        "Upload ECG (.csv)",

        type=["csv"]

    )

    if uploaded is not None:

        sample = np.loadtxt(

            uploaded,

            delimiter=","

        )

# ======================================================
# Prediction
# ======================================================

if sample is not None:

    result = predict_single_ecg(sample)

    observed = result["Observed ECG"]

    forecast = result["Forecast"]

    target = result["Ground Truth"]

    prediction = result["Prediction"]

    confidence = result["Confidence"]

    divergence = result["Earliest Divergence Step"]

    error = result["Forecast Error"]

    # ==================================================
    # ECG Plot
    # ==================================================

    st.subheader("📈 Observed ECG")

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            y=observed,

            mode="lines",

            line=dict(

                color="#00E5A8",

                width=3

            ),

            name="Observed ECG"

        )

    )

    fig.update_layout(

        template="plotly_dark",

        height=350,

        paper_bgcolor="#07141F",

        plot_bgcolor="#07141F",

        margin=dict(

            l=20,

            r=20,

            t=20,

            b=20

        )

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ==================================================
    # Prediction Cards
    # ==================================================

    left,right = st.columns([1,2])

    with left:

        if prediction=="Normal":

            st.markdown(

                f"""

<div class='pred-normal'>

🟢 NORMAL

</div>

""",

unsafe_allow_html=True

            )

        else:

            st.markdown(

                f"""

<div class='pred-abnormal'>

🔴 ABNORMAL

</div>

""",

unsafe_allow_html=True

            )

    with right:

        c1,c2,c3 = st.columns(3)

        c1.metric(

            "Forecast Error",

            f"{error:.4f}"

        )

        c2.metric(

            "Confidence",

            f"{confidence:.1f}%"

        )

        c3.metric(

            "Earliest Divergence",

            str(divergence)

        )

    # ==================================================
    # Forecast Plot
    # ==================================================

    st.subheader("🔮 Forecast vs Ground Truth")

    x = np.arange(98,140)

    fig2 = go.Figure()

    fig2.add_trace(

        go.Scatter(

            x=x,

            y=target,

            mode="lines",

            line=dict(

                color="#29D391",

                width=3

            ),

            name="Ground Truth"

        )

    )

    fig2.add_trace(

        go.Scatter(

            x=x,

            y=forecast,

            mode="lines",

            line=dict(

                color="#F4B942",

                width=3,

                dash="dash"

            ),

            name="Forecast"

        )

    )

    fig2.update_layout(

        template="plotly_dark",

        height=420,

        paper_bgcolor="#07141F",

        plot_bgcolor="#07141F"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    if true_label is not None:

        st.info(

            f"Ground Truth Label : {'Normal' if true_label==1 else 'Abnormal'}"

        )
# ==================================================
# Explainability Tabs
# ==================================================

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "🔥 Attention",
            "📉 Error Evolution",
            "📊 Model Insights"
        ]
    ) 
    with tab1:

        st.subheader("Bahdanau Attention Heatmap")

        attention = np.array(result["Attention"])

        fig, ax = plt.subplots(figsize=(10,5))

        im = ax.imshow(
            attention,
            aspect="auto",
            origin="lower",
            cmap="viridis"
        )

        ax.set_xlabel("Observed ECG Samples")

        ax.set_ylabel("Forecast Step")

        ax.set_title("Attention Matrix")

        plt.colorbar(
            im,
            ax=ax
        )

        st.pyplot(fig)

        st.caption(
            "Brighter regions indicate stronger attention to specific input samples during forecasting."
        )
    with tab2:

        st.subheader("Forecast Error Evolution")

        step_errors = np.array(
            result["Step Errors"]
        )

        thresholds = result["Step Thresholds"]

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                y=step_errors,

                mode="lines+markers",

                name="Forecast Error",

                line=dict(
                    color="#00E5A8",
                    width=3
                )

            )

        )

        if thresholds is not None:

            fig.add_trace(

                go.Scatter(

                    y=thresholds,

                    mode="lines",

                    name="Step Threshold",

                    line=dict(
                        color="#FF6B6B",
                        dash="dash"
                    )

                )

            )

        earliest = result["Earliest Divergence Step"]

        if isinstance(earliest, int):

            fig.add_vline(

                x=earliest-1,

                line_width=3,

                line_dash="dot",

                line_color="orange"

            )

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor="#07141F",

            plot_bgcolor="#07141F",

            height=450,

            xaxis_title="Forecast Step",

            yaxis_title="Prediction Error"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            f"Earliest Divergence : {earliest}"
        )
    with tab3:

        st.subheader("Model Summary")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Prediction",
                result["Prediction"]
            )

            st.metric(
                "Forecast Error",
                f"{result['Forecast Error']:.4f}"
            )

            st.metric(
                "Confidence",
                f"{result['Confidence']:.1f}%"
            )

        with c2:

            st.metric(
                "Forecast Horizon",
                "42"
            )

            st.metric(
                "Observed Length",
                "98"
            )

            st.metric(
                "Earliest Divergence",
                str(result["Earliest Divergence Step"])
            )

        st.markdown("---")

        st.markdown("### Clinical Interpretation")

        if result["Prediction"] == "Normal":

            st.success(
                """
    The forecast error remains within the expected range.
    The model predicts this heartbeat as **Normal**.
    No significant abnormal forecasting behaviour was detected.
    """
            )

        else:

            st.error(
                f"""
    The forecast error exceeded the anomaly threshold.

    Earliest divergence occurred at **Step {result['Earliest Divergence Step']}**.

    The model therefore classifies this heartbeat as **Abnormal**.
    """
            )
    # ==================================================
    # About the Model
    # ==================================================

    with st.expander("🧠 About the Model"):

        st.markdown("""
    ### Architecture

    - **Encoder:** Bidirectional LSTM
    - **Decoder:** LSTM
    - **Attention:** Bahdanau Attention
    - **Forecast Horizon:** 42 samples
    - **Observed Context:** 98 samples

    ---

    ### Pipeline

    Observed ECG

    ↓

    Forecast Future ECG

    ↓

    Forecast Error

    ↓

    Anomaly Detection

    ↓

    Explainability

    - Attention Heatmap
    - Earliest Divergence
    - Confidence Score

    ---

    ### Dataset

    ECG5000 Dataset

    Normal heartbeats are used for training.

    Abnormal beats are detected using forecasting error.
    """)

    # ==================================================
    # Export Results
    # ==================================================

    st.divider()

    st.subheader("📄 Export Results")

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pdf_path = os.path.join(
        "results",
        f"ECG_Report_{timestamp}.pdf"
    )

    generate_pdf_report(
        result,
        pdf_path
    )

    with open(pdf_path, "rb") as pdf_file:

        st.download_button(

            label="📥 Download Clinical PDF Report",

            data=pdf_file,

            file_name=f"ECG_Report_{timestamp}.pdf",

            mime="application/pdf"

        )

    # ==================================================
    # AI Interpretation
    # ==================================================

    st.divider()

    st.subheader("🩺 AI Interpretation")

    if result["Prediction"] == "Normal":

        interpretation = f"""
    The ECG has been classified as **Normal**.

    Forecast Error : **{result['Forecast Error']:.4f}**

    Confidence : **{result['Confidence']:.1f}%**

    The prediction error remained within the learned
    threshold.

    No significant abnormal forecasting behaviour
    was detected.

    Earliest Divergence :

    **{result['Earliest Divergence Step']}**
    """

        st.success(interpretation)

    else:

        interpretation = f"""
    The ECG has been classified as **Abnormal**.

    Forecast Error : **{result['Forecast Error']:.4f}**

    Confidence : **{result['Confidence']:.1f}%**

    The forecast error exceeded the anomaly threshold.

    Earliest Divergence :

    **Step {result['Earliest Divergence Step']}**

    The model detected abnormal prediction behaviour
    during forecasting.
    """

        st.error(interpretation)
