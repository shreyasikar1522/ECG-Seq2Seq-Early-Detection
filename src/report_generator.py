import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "fonts"
)

pdfmetrics.registerFont(

    TTFont(

        "OpenSans",

        os.path.join(
            FONT_DIR,
            "OpenSans-Regular.ttf"
        )

    )

)

pdfmetrics.registerFont(

    TTFont(

        "OpenSans-Bold",

        os.path.join(
            FONT_DIR,
            "OpenSans-Bold.ttf"
        )

    )

)

pdfmetrics.registerFont(

    TTFont(

        "OpenSans-SemiBold",

        os.path.join(
            FONT_DIR,
            "OpenSans-SemiBold.ttf"
        )

    )

)

from datetime import datetime
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)

# ============================================================
# Helper Functions
# ============================================================

def save_forecast_plot(result):
    """
    Creates Forecast vs Ground Truth plot.

    Returns temporary image path.
    """

    forecast = np.array(result["Forecast"])

    target = np.array(result["Ground Truth"])

    fig, ax = plt.subplots(figsize=(7,3))

    x = np.arange(len(target))

    ax.plot(
        x,
        target,
        linewidth=2,
        label="Ground Truth"
    )

    ax.plot(
        x,
        forecast,
        "--",
        linewidth=2,
        label="Forecast"
    )

    ax.set_title("Forecast vs Ground Truth")

    ax.set_xlabel("Forecast Step")

    ax.set_ylabel("Normalized ECG")

    ax.grid(alpha=0.3)

    ax.legend()

    temp = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    plt.tight_layout()

    plt.savefig(
        temp.name,
        dpi=300
    )

    plt.close()

    return temp.name


# ============================================================

def save_error_plot(result):
    """
    Creates Forecast Error Evolution plot.

    Returns temporary image path.
    """

    errors = np.array(
        result["Step Errors"]
    )

    thresholds = result["Step Thresholds"]

    fig, ax = plt.subplots(figsize=(7,3))

    ax.plot(
        errors,
        linewidth=2,
        label="Forecast Error"
    )

    if thresholds is not None:

        ax.plot(
            thresholds,
            "--",
            linewidth=2,
            label="Threshold"
        )

    earliest = result[
        "Earliest Divergence Step"
    ]

    if isinstance(earliest, int):

        ax.axvline(
            earliest-1,
            color="red",
            linestyle=":",
            linewidth=2
        )

    ax.set_title(
        "Forecast Error Evolution"
    )

    ax.set_xlabel(
        "Forecast Step"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    temp = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    plt.tight_layout()

    plt.savefig(
        temp.name,
        dpi=300
    )

    plt.close()

    return temp.name


# ============================================================

def save_attention_plot(result):
    """
    Creates attention heatmap.

    Returns temporary image path.
    """

    attention = np.array(
        result["Attention"]
    )

    fig, ax = plt.subplots(figsize=(7,4))

    im = ax.imshow(

        attention,

        aspect="auto",

        origin="lower",

        cmap="viridis"

    )

    ax.set_title(
        "Bahdanau Attention Heatmap"
    )

    ax.set_xlabel(
        "Observed ECG Samples"
    )

    ax.set_ylabel(
        "Forecast Step"
    )

    plt.colorbar(
        im,
        ax=ax
    )

    temp = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    plt.tight_layout()

    plt.savefig(
        temp.name,
        dpi=300
    )

    plt.close()

    return temp.name
 
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(
        7.5 * inch,
        0.5 * inch,
        f"Page {doc.page}"
    )
    canvas.restoreState()

# ============================================================
# PDF Generator
# ============================================================

def generate_pdf_report(
    result,
    output_path
):

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "HospitalTitle",
        parent=styles["Title"],
        fontName="OpenSans-Bold",
        fontSize=26,
        leading=32,
        textColor=HexColor("#0A5C7D"),
        alignment=TA_CENTER,
        spaceAfter=18
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        fontName="OpenSans",
        fontSize=12,
        leading=18,
        textColor=HexColor("#6B7280"),
        alignment=TA_CENTER,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="OpenSans-SemiBold",
        fontSize=17,
        leading=22,
        textColor=HexColor("#0B6E99"),
        spaceBefore=12,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(

        "Body",

        parent=styles["BodyText"],

        fontName="OpenSans",

        fontSize=11,

        leading=18,

        textColor=HexColor("#222222")

    )

    doc = SimpleDocTemplate(
        output_path
    )

    story = []

    # --------------------------------------------------------

    story.append(
    Paragraph(
            "❤️ ECG Forecasting & Explainable AI Report",
            title_style
        )
    )
    drawing = Drawing(450,35)

    drawing.add(

        Line(
            0,
            15,
            120,
            15
        )

    )

    drawing.add(

        Line(
            120,
            15,
            140,
            28
        )

    )

    drawing.add(

        Line(
            140,
            28,
            160,
            5
        )

    )

    drawing.add(

        Line(
            160,
            5,
            180,
            25
        )

    )

    drawing.add(

        Line(
            180,
            25,
            450,
            15
        )

    )

    story.append(
        drawing
    )

    story.append(
        Spacer(
            1,
            0.2*inch
        )
    )
    current_time = datetime.now().strftime(
        "%d %B %Y   %I:%M %p"
    )

    story.append(

        Paragraph(

            f"<b>Generated :</b> {current_time}",

            styles["BodyText"]

        )

    )

    story.append(
        Spacer(
            1,
            0.2*inch
        )
    )

    story.append(
        Paragraph(
            "Seq2Seq • Bahdanau Attention • ECG5000 Dataset",
             subtitle_style
        )
    )

    story.append(
       Spacer(1,0.15*inch)
    )

    story.append(
        Spacer(
            1,
            0.3*inch
        )
    )

    # ========================================================
    # Prediction Summary
    # ========================================================
    prediction = result["Prediction"]
    if prediction == "Normal":

        badge_color = "#1B9C85"

    else:

        badge_color = "#D64545"

    error = result["Forecast Error"]

    confidence = result["Confidence"]

    divergence = result[
        "Earliest Divergence Step"
    ]

    table_data = [

        ["Prediction", prediction],

        ["Forecast Error", f"{error:.4f}"],

        ["Confidence", f"{confidence:.2f}%"],

        ["Earliest Divergence", str(divergence)]

    ]

    table = Table(
        table_data,
        colWidths=[2.8*inch,2.8*inch]
    )

    table.setStyle(

        TableStyle([

            # -------------------------------------------------
            # Header
            # -------------------------------------------------

            ("BACKGROUND", (0,0), (-1,0), HexColor("#0B6E99")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "OpenSans-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 14),

            # -------------------------------------------------
            # Body
            # -------------------------------------------------

            ("BACKGROUND", (0,1), (-1,-1), HexColor("#F8FBFC")),
            ("TEXTCOLOR", (0,1), (-1,-1), HexColor("#1F2937")),
            ("FONTNAME", (0,1), (-1,-1), "OpenSans"),
            ("FONTSIZE", (0,1), (-1,-1), 12),

            # -------------------------------------------------
            # Alignment
            # -------------------------------------------------

            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

            # -------------------------------------------------
            # Padding
            # -------------------------------------------------

            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),

            # -------------------------------------------------
            # Borders
            # -------------------------------------------------

            ("GRID", (0,0), (-1,-1), 0.5, HexColor("#D0D7DE")),
            ("BOX", (0,0), (-1,-1), 1, HexColor("#0B6E99")),

            # -------------------------------------------------
            # Alternate Row Colors
            # -------------------------------------------------

            ("BACKGROUND", (0,1), (-1,1), HexColor("#F8FBFC")),
            ("BACKGROUND", (0,2), (-1,2), HexColor("#EEF7F9")),
            ("BACKGROUND", (0,3), (-1,3), HexColor("#F8FBFC")),
            ("BACKGROUND", (0,4), (-1,4), HexColor("#EEF7F9")),

        ])

    )

    story.append(table)

    story.append(
        Spacer(
            1,
            0.3*inch
        )
    )
    story.append(
        Paragraph(
            "Clinical Summary",
            heading_style
        )
    )

    summary = f"""

    <b>Prediction:</b> {prediction}<br/>

    <b>Confidence:</b> {confidence:.2f}%<br/>

    <b>Forecast Error:</b> {error:.4f}<br/>

    <b>Earliest Divergence:</b> {divergence}
    """

    story.append(
        Paragraph(
            summary,
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            0.2*inch
        )
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=styles["Heading1"],
        fontName="OpenSans-Bold",
        fontSize=22,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.white,
        backColor=HexColor(badge_color),
        spaceAfter=15
    )

    story.append(

        Paragraph(

            prediction.upper(),

            badge_style

        )

    )
    # ========================================================
    # AI Interpretation
    # ========================================================

    story.append(
        Paragraph(
            "AI Interpretation",
            heading_style
        )
    )

    if prediction == "Normal":

        interpretation = f"""
        <b>Prediction:</b> NORMAL<br/><br/>

        The forecasting error remained within the learned
        threshold.

        No significant abnormal forecasting behaviour was
        detected.

        <br/><br/>

        <b>Forecast Error:</b> {error:.4f}<br/>

        <b>Confidence:</b> {confidence:.2f}%<br/>

        <b>Earliest Divergence:</b> {divergence}
        """

    else:

        interpretation = f"""
        <b>Prediction:</b> ABNORMAL<br/><br/>

        The forecasting error exceeded the anomaly
        detection threshold.

        The model detected abnormal forecasting
        behaviour.

        <br/><br/>

        <b>Forecast Error:</b> {error:.4f}<br/>

        <b>Confidence:</b> {confidence:.2f}%<br/>

        <b>Earliest Divergence:</b> Step {divergence}
        """

    story.append(

        Paragraph(

            interpretation,

            normal_style

        )

    )

    story.append(
        Spacer(
            1,
            0.3*inch
        )
    )

    # ========================================================
    # Forecast Plot
    # ========================================================

    story.append(
        Paragraph(
            "Forecast vs Ground Truth",
            heading_style
        )
    )

    forecast_image = save_forecast_plot(
        result
    )

    story.append(

        Image(

            forecast_image,

            width=6.5*inch,

            height=3.2*inch

        )

    )

    story.append(
        Spacer(
            1,
            0.25*inch
        )
    )

    # ========================================================
    # Forecast Error Plot
    # ========================================================

    story.append(
        Paragraph(
            "Forecast Error Evolution",
            heading_style
        )
    )

    error_image = save_error_plot(
        result
    )

    story.append(

        Image(

            error_image,

            width=6.5*inch,

            height=3.2*inch

        )

    )

    story.append(
        Spacer(
            1,
            0.25*inch
        )
    )

    # ========================================================
    # Attention Heatmap
    # ========================================================

    story.append(
        Paragraph(
            "Bahdanau Attention Heatmap",
            heading_style
        )
    )

    attention_image = save_attention_plot(
        result
    )

    story.append(

        Image(

            attention_image,

            width=6.5*inch,

            height=3.6*inch

        )

    )

    story.append(
        Spacer(
            1,
            0.25*inch
        )
    )

    # ========================================================
    # Footer
    # ========================================================

    story.append(
        Paragraph(
            "Clinical Summary",
            heading_style
        )
    )

    summary = f"""

    <b>Prediction:</b> {prediction}<br/>

    <b>Confidence:</b> {confidence:.2f}%<br/>

    <b>Forecast Error:</b> {error:.4f}<br/>

    <b>Earliest Divergence:</b> {divergence}
    """

    story.append(
        Paragraph(
            summary,
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            0.2*inch
        )
    )

    # ========================================================
    # Build PDF
    # ========================================================

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    # ========================================================
    # Remove temporary images
    # ========================================================

    for image_path in [

        forecast_image,

        error_image,

        attention_image

    ]:

        if os.path.exists(image_path):

            os.remove(image_path)
# ============================================================
# Test Report Generator
# ============================================================

if __name__ == "__main__":

    from .data_loader import load_ecg5000
    from .predict_single import predict_single_ecg

    print("=" * 60)
    print("Generating ECG Report")
    print("=" * 60)

    # -----------------------------------------
    # Load ECG5000
    # -----------------------------------------

    labels, signals = load_ecg5000()

    # -----------------------------------------
    # Choose an ECG
    # -----------------------------------------

    # Normal ECG
    # sample_index = np.where(labels == 1)[0][0]

    # Abnormal ECG
    sample_index = 0

    signal = signals[sample_index]

    # -----------------------------------------
    # Run Prediction
    # -----------------------------------------

    result = predict_single_ecg(signal)

    # -----------------------------------------
    # Output PDF
    # -----------------------------------------

    output_file = "ECG_Report.pdf"

    generate_pdf_report(
        result,
        output_file
    )

    print()

    print("Prediction :", result["Prediction"])

    print(
        "Forecast Error :",
        round(
            result["Forecast Error"],
            4
        )
    )

    print(
        "Confidence :",
        round(
            result["Confidence"],
            2
        ),
        "%"
    )

    print(
        "Earliest Divergence :",
        result["Earliest Divergence Step"]
    )

    print()

    print(
        "PDF saved as:"
    )

    print(output_file)

    print()

    print("=" * 60)
    print("Done")
    print("=" * 60)