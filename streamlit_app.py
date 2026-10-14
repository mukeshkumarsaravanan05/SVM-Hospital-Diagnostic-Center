from __future__ import annotations
import io
import os
import tempfile
from datetime import datetime
from html import escape
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance

# Existing project pipeline — keep these modules and model workflow unchanged.
from ai_engine.preprocessing import preprocess_image
from ai_engine.prediction import CLASSES, predict_image
from ai_engine.image_validator import validate_xray
from ai_engine.image_quality import assess_image_quality
from ai_engine.gradcam import generate_gradcam


st.set_page_config(
    page_title="SVM Hospital Diagnostic Center | Research Demo",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- THEME -----------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0d1a2b;
        --panel: #172b42;
        --panel2: #1b344e;
        --line: #3d5b78;
        --text: #ffffff;
        --muted: #c6d5e5;
        --teal: #42ead8;
        --blue: #60a5fa;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 88% 8%, rgba(13, 148, 136, .23), transparent 34%),
            radial-gradient(ellipse at 15% 22%, rgba(37, 99, 235, .20), transparent 36%),
            var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #132943 0%, #0e1e32 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    h1, h2, h3, h4, p, label, li { color: var(--text); }
    .muted, [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
    .hero {
        padding: 30px 32px;
        border: 1px solid #294b60;
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(20, 38, 62, .98), rgba(7, 65, 70, .72));
        margin: 6px 0 18px 0;
    }
    .eyebrow {
        color: var(--teal);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero h1 { font-size: clamp(2rem, 4vw, 3rem); line-height: 1.12; margin: 0 0 12px 0; }
    .hero p { color: #d8e6f5; font-size: 1.04rem; line-height: 1.65; margin-bottom: 0; }
    .tag {
        display: inline-block; border: 1px solid #287d7b; border-radius: 30px;
        padding: 6px 12px; margin: 15px 6px 0 0; color: #8ff8ed; font-size: .82rem;
        background: rgba(6, 78, 79, .22);
    }
    .notice {
        padding: 15px 18px; border-left: 4px solid #f6bd60; border-radius: 0 12px 12px 0;
        background: rgba(120, 72, 20, .20); color: #ffe6b6; line-height: 1.65;
        margin-bottom: 22px;
    }
    .section-kicker {
        color: var(--teal); font-size: .78rem; font-weight: 800;
        letter-spacing: .14em; text-transform: uppercase; margin: 18px 0 5px;
    }
    .panel {
        padding: 18px 20px; border: 1px solid var(--line); border-radius: 16px;
        background: rgba(23, 43, 66, .96); margin-bottom: 14px;
    }
    div[data-testid="stMetric"] {
        background: #1b344e; border: 1px solid #3d5b78; border-radius: 14px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #e1edf8 !important; }
    div[data-testid="stMetricValue"] { color: #f4f8ff !important; }
    div[data-testid="stFileUploader"] {
        background: #1b344e; border: 1px dashed #58d6ce; border-radius: 14px; padding: 10px;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: #f3f6fb !important; color: #142238 !important;
        border-radius: 9px !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #f3f6fb !important; color: #142238 !important;
        border-radius: 9px !important;
    }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 10px; font-weight: 700; min-height: 42px;
    }
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #0f9f9a, #1676b8);
        color: white; border: 0;
    }
    div[data-testid="stMarkdownContainer"] p, div[data-testid="stCaptionContainer"] p { color: #dce8f5; }
    [data-testid="stWidgetLabel"] p, label { color: #f1f6fc !important; }
    hr { border-color: var(--line); }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- SIDEBAR ---------------------------------------
with st.sidebar:
    st.markdown("## 🩻 SVM Research")
    st.caption("Biomedical AI · Explainable imaging")
    page = st.radio(
        "Navigation",
        ["Overview", "Analyze X-ray", "Workflow & limitations"],
        index=1,
    )
    st.divider()
    st.markdown("### Project links")
    st.markdown("[GitHub repository ↗](https://github.com/mukeshkumarsaravanan05/SVM-Hospital-Diagnostic-Center)")
    st.markdown("[Live web demo ↗](https://svm-hospital-diagnostic-center.streamlit.app/)")
    st.caption("Built by Mukesh Kumar · Biomedical Engineering")

# ----------------------------- HERO ------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Biomedical AI · Research Demonstration</div>
      <h1>SVM Hospital Diagnostic Center</h1>
      <p>AI-assisted chest X-ray classification with image validation, quality checks,
      probability visualization, and Grad-CAM explainability — presented in a focused
      research dashboard.</p>
      <span class="tag">MobileNetV2</span><span class="tag">Explainable AI</span>
      <span class="tag">Image quality</span><span class="tag">PDF research report</span>
    </div>
    <div class="notice"><b>Research and educational use only.</b> This tool is not a
    medical device or standalone diagnostic system. Model probabilities are not
    calibrated clinical certainty. Do not upload identifiable patient images or personal
    health information to a public demo. Results must not replace assessment by a
    qualified clinician.</div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- PDF REPORT ------------------------------------
def make_pdf(
    original: Image.Image,
    heatmap: np.ndarray,
    predicted_class: str,
    confidence: float,
    probs: np.ndarray,
    validation_percent: float,
    valid: bool,
    quality: dict,
    patient_details: dict,
) -> bytes:
    """Create the requested three-page, research-only PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak,
    )

    navy = colors.HexColor("#103451")
    navy_dark = colors.HexColor("#0B2740")
    pale = colors.HexColor("#EEF5FA")
    pale2 = colors.HexColor("#F7FAFC")
    ink = colors.HexColor("#253746")
    muted = colors.HexColor("#5E7180")
    line = colors.HexColor("#CFDCE5")

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=13 * mm, bottomMargin=17 * mm,
        title="SVM Hospital Diagnostic Center - AI Chest X-ray Research Report",
        author="SVM Hospital Diagnostic Center",
    )
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                 fontSize=16, leading=19, textColor=colors.white, alignment=TA_LEFT, spaceAfter=2),
        "subtitle": ParagraphStyle("ReportSubtitle", parent=base["Normal"], fontSize=8.5, leading=11,
                                    textColor=colors.HexColor("#D9E7F1")),
        "section": ParagraphStyle("SectionBar", parent=base["Heading2"], fontName="Helvetica-Bold",
                                  fontSize=9.5, leading=12, textColor=colors.white, spaceBefore=1, spaceAfter=0),
        "body": ParagraphStyle("ReportBody", parent=base["BodyText"], fontSize=8.2, leading=11,
                                textColor=ink, spaceAfter=3),
        "small": ParagraphStyle("ReportSmall", parent=base["BodyText"], fontSize=7.2, leading=9,
                                 textColor=muted),
        "cell": ParagraphStyle("Cell", parent=base["BodyText"], fontSize=7.8, leading=9.5, textColor=ink),
        "cellbold": ParagraphStyle("CellBold", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.8,
                                    leading=9.5, textColor=ink),
        "center": ParagraphStyle("Centered", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.8,
                                 leading=9.5, alignment=TA_CENTER, textColor=navy),
        "big": ParagraphStyle("BigValue", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=15,
                              leading=18, alignment=TA_CENTER, textColor=navy),
    }

    def safe(value):
        return escape(str(value if value not in (None, "") else "Not provided")).replace("\n", "<br/>")

    def P(value, style="body"):
        return Paragraph(safe(value), styles[style])

    def section_bar(label):
        table = Table([[P(label.upper(), "section")]], colWidths=[180 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), navy),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def data_table(rows, widths=(48 * mm, 132 * mm), header=True):
        formatted = []
        for ri, row in enumerate(rows):
            formatted.append([P(cell, "cellbold" if (ri == 0 and header) or ci == 0 else "cell")
                              for ci, cell in enumerate(row)])
        table = Table(formatted, colWidths=list(widths), repeatRows=1 if header else 0, hAlign="LEFT")
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            commands += [("BACKGROUND", (0, 0), (-1, 0), pale)]
            if len(rows) > 1:
                commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale2]))
        else:
            commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, pale2]))
        table.setStyle(TableStyle(commands))
        return table

    def on_page(canvas, document):
        canvas.saveState()
        width, height = A4
        if document.page > 1:
            canvas.setFillColor(navy_dark)
            canvas.rect(0, height - 7 * mm, width, 7 * mm, fill=1, stroke=0)
        canvas.setStrokeColor(line)
        canvas.setLineWidth(0.5)
        canvas.line(15 * mm, 12 * mm, width - 15 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawString(15 * mm, 7.5 * mm, "CONFIDENTIAL • AI-ASSISTED DECISION SUPPORT • RESEARCH USE ONLY")
        canvas.drawRightString(width - 15 * mm, 7.5 * mm, f"Page {document.page} of 3")
        canvas.restoreState()

    generated = datetime.now()
    report_id = generated.strftime("%Y%m%d_%H%M%S")
    quality_score = quality.get("score", "Not available")
    resolution = quality.get("resolution", "Not available")
    brightness = quality.get("brightness", "Not available")
    contrast = quality.get("contrast", "Not available")
    sharpness = quality.get("sharpness", "Not available")
    risk = quality.get("risk_level", "Not available")
    review_reason = quality.get("review_reason", "Not available")
    score_text = f"{quality_score}/100" if str(quality_score) != "Not available" else "Not available"
    assessment_level = "MODERATE" if (not valid or (isinstance(quality_score, (int, float)) and quality_score < 85)) else "REVIEW REQUIRED"

    story = []
    header = Table([
        [P("SVM HOSPITAL DIAGNOSTIC CENTER", "title"), P(f"CONFIDENTIAL<br/>{generated.strftime('%d %b %Y')}", "subtitle")],
        [P("AI-Assisted Chest X-ray Diagnostic Report", "subtitle"), P("RESEARCH DEMONSTRATION", "subtitle")],
    ], colWidths=[132 * mm, 48 * mm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), navy_dark), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [header, Spacer(1, 3 * mm)]
    case_summary = " • ".join(x for x in [
        f"Case: {patient_details.get('case_id', '').strip()}" if patient_details.get('case_id', '').strip() else "",
        f"Label: {patient_details.get('patient_label', '').strip()}" if patient_details.get('patient_label', '').strip() else "",
        f"Age: {patient_details.get('age')}" if patient_details.get('age') else "",
        f"Sex: {patient_details.get('sex')}" if patient_details.get('sex') not in (None, "", "Not provided") else "",
        f"Exam date: {patient_details.get('exam_date')}" if patient_details.get('exam_date') else "",
    ] if x)
    meta = Table([
        [P("REPORT TYPE", "cellbold"), P("REPORT ID", "cellbold"), P("GENERATED", "cellbold")],
        [P("AI Chest X-ray Assessment", "cell"), P(report_id, "cell"), P(generated.strftime("%d-%m-%Y %H:%M:%S"), "cell")],
    ], colWidths=[60 * mm, 55 * mm, 65 * mm])
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale), ("GRID", (0, 0), (-1, -1), 0.4, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [meta]
    if case_summary:
        story += [Spacer(1, 1.5 * mm), P(case_summary, "small")]
    story += [Spacer(1, 2.5 * mm), section_bar("X-ray validation"), Spacer(1, 1.5 * mm)]
    story += [data_table([
        ["Validation item", "Result"],
        ["X-ray validator probability", f"{float(validation_percent):.2f}%"],
        ["Configured validation threshold", "See project configuration"],
        ["Validation result", "VALIDATOR PASSED" if valid else "VALIDATOR NOT PASSED"],
    ]), Spacer(1, 3 * mm), section_bar("Source radiograph & AI explanation"), Spacer(1, 1.5 * mm)]

    original_buffer = io.BytesIO()
    original.convert("RGB").save(original_buffer, format="JPEG", quality=92)
    original_buffer.seek(0)
    heatmap_buffer = io.BytesIO()
    Image.fromarray(np.asarray(heatmap).astype(np.uint8)).convert("RGB").save(heatmap_buffer, format="JPEG", quality=92)
    heatmap_buffer.seek(0)
    original_pdf_image = RLImage(original_buffer, width=79 * mm, height=58 * mm)
    heatmap_pdf_image = RLImage(heatmap_buffer, width=79 * mm, height=58 * mm)
    images = Table([
        [P("ORIGINAL CHEST X-RAY", "center"), P("GRAD-CAM AI ATTENTION MAP", "center")],
        [original_pdf_image, heatmap_pdf_image],
    ], colWidths=[90 * mm, 90 * mm])
    images.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale), ("GRID", (0, 0), (-1, -1), 0.4, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [images, Spacer(1, 2 * mm), section_bar("Image quality assessment"), Spacer(1, 1.5 * mm)]
    story += [data_table([
        ["Metric", "Result"], ["Resolution", resolution], ["Brightness", brightness],
        ["Contrast", contrast], ["Sharpness", sharpness], ["Overall quality score", score_text],
        ["Quality risk", risk], ["Review reason", review_reason],
    ]), PageBreak()]

    story += [section_bar("AI diagnostic summary"), Spacer(1, 2.5 * mm)]
    summary = Table([
        [P("AI PREDICTED CONDITION", "center"), P("MODEL CONFIDENCE", "center")],
        [P(str(predicted_class).upper(), "big"), P(f"{float(confidence):.2f}%", "big")],
    ], colWidths=[90 * mm, 90 * mm])
    summary.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, line), ("BACKGROUND", (0, 0), (-1, 0), pale),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [summary, Spacer(1, 3 * mm), section_bar("AI assessment status"), Spacer(1, 1.5 * mm)]
    status = Table([
        [P("ASSESSMENT LEVEL", "center"), P("CLINICAL REVIEW", "center")],
        [P(assessment_level, "big"), P("REQUIRED", "big")],
        [P("MODEL OUTPUT", "cellbold"), P("IMAGE QUALITY", "cellbold")],
        [P(f"{float(confidence):.2f}%", "cell"), P(score_text, "cell")],
    ], colWidths=[90 * mm, 90 * mm])
    status.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, line), ("BACKGROUND", (0, 0), (-1, 0), pale),
        ("BACKGROUND", (0, 2), (-1, 2), pale2), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [status, Spacer(1, 3 * mm), section_bar("AI probability distribution"), Spacer(1, 1.5 * mm)]
    probability_rows = [["Disease class", "Probability"]] + [
        [str(name), f"{float(value) * 100:.2f}%"] for name, value in zip(CLASSES, probs)
    ]
    story += [data_table(probability_rows, widths=(90 * mm, 90 * mm)), Spacer(1, 3 * mm),
              section_bar("Interpretation & limitations"), Spacer(1, 1.5 * mm)]
    story += [
        P("The predicted class and probability values are outputs of the configured AI pipeline. They should be interpreted alongside the source radiograph, image quality, clinical history, and a professional radiological assessment."),
        P("Grad-CAM is an interpretability visualization. It does not establish that the highlighted regions are clinically meaningful, and model probabilities are not calibrated clinical certainty."),
        Spacer(1, 1 * mm), section_bar("Clinical disclaimer"), Spacer(1, 1.5 * mm),
    ]
    disclaimer = Table([[P("For research and educational use only. This AI-generated assessment is not a standalone medical diagnosis and must not replace evaluation by a qualified clinician or radiologist. Final clinical decisions must be made by an appropriately qualified professional.")]], colWidths=[180 * mm])
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7E8")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#E7C77D")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [disclaimer, PageBreak()]

    story += [section_bar("AI system status"), Spacer(1, 2.5 * mm)]
    story += [data_table([
        ["System item", "Status"], ["Report generation", "Completed"],
        ["Image validation", "Passed" if valid else "Not passed"],
        ["Prediction output", "Recorded in this report"],
        ["Clinical validation", "Not established by this report"], ["Report identifier", report_id],
    ]), Spacer(1, 6 * mm), section_bar("Clinical review"), Spacer(1, 2 * mm)]
    signature = Table([
        [P("Reviewer name:", "cellbold"), ""],
        [P("Professional title / registration:", "cellbold"), ""],
        [P("Review notes:", "cellbold"), ""],
        [P("Signature:", "cellbold"), P("________________________________________", "cell")],
        [P("Date:", "cellbold"), P("________________________________________", "cell")],
    ], colWidths=[55 * mm, 125 * mm], rowHeights=[12 * mm, 12 * mm, 28 * mm, 14 * mm, 14 * mm])
    signature.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, line), ("BACKGROUND", (0, 0), (0, -1), pale),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [signature, Spacer(1, 8 * mm), P("This document records software outputs for the uploaded image only. It does not establish diagnostic accuracy, clinical validity, or patient outcome.", "small")]

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output.getvalue()


# ----------------------------- PAGE CONTENT ---------------------------------
if page == "Overview":
    st.markdown("## Research dashboard overview")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="panel"><h3>01 · Validate</h3><p>Check whether the uploaded image passes the configured X-ray validator.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="panel"><h3>02 · Analyze</h3><p>Run the existing MobileNetV2 classification and inspect the class probability distribution.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="panel"><h3>03 · Explain</h3><p>View the Grad-CAM visualization and download a PDF research report.</p></div>', unsafe_allow_html=True)
    st.info("Choose **Analyze X-ray** in the sidebar to enter examination details and run the pipeline.")
elif page == "Workflow & limitations":
    st.markdown("## Workflow & limitations")
    st.markdown("""
    1. **Input:** Upload an authorized chest X-ray image.
    2. **Validation:** The existing X-ray validator is applied using the project's configured threshold.
    3. **Image quality:** Resolution, brightness, contrast, sharpness, and quality-risk indicators are calculated.
    4. **Prediction:** The existing MobileNetV2 model returns class probabilities.
    5. **Explainability:** Grad-CAM generates a visual explanation aid.
    6. **Report:** Download a PDF containing examination details, outputs, probabilities, image, and Grad-CAM.

    **Limitations**
    - This is a research and educational demonstration, not a medical device.
    - Model probabilities are not calibrated clinical certainty and are not clinical accuracy metrics.
    - Validator and Grad-CAM outputs can be wrong or misleading.
    - Do not use this app to make or delay diagnosis or treatment.
    - Use fictional or de-identified details in a public demonstration.
    """)
else:
    st.markdown('<div class="section-kicker">Image analysis</div>', unsafe_allow_html=True)
    st.markdown("Enter optional, non-identifying examination details, then upload a chest X-ray.")
    with st.expander("Patient & examination details", expanded=True):
        st.caption("For public demos, use fictional or de-identified details only. Do not enter real patient identifiers or sensitive health information.")
        left, right = st.columns(2)
        with left:
            case_id = st.text_input("Patient / case ID", placeholder="e.g., CASE-001")
            patient_label = st.text_input("Patient name / anonymized label", placeholder="e.g., Anonymous case 001")
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=0, step=1)
        with right:
            sex = st.selectbox("Sex", ["Not provided", "Female", "Male", "Other", "Prefer not to say"])
            exam_date = st.date_input("Examination date", value=datetime.now().date())
            notes = st.text_area("Clinical indication / notes (optional)", placeholder="Brief context; avoid identifiers", height=80)

    patient_details = {
        "case_id": case_id.strip(),
        "patient_label": patient_label.strip(),
        "age": "" if age == 0 else str(age),
        "sex": sex,
        "exam_date": exam_date.strftime("%d/%m/%Y"),
        "notes": notes.strip(),
    }

    st.markdown("## Upload X-ray")
    uploaded = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="Upload only an image you are authorized to process. Avoid identifiable patient information.",
    )

    if uploaded is not None:
        temp_path = None
        try:
            raw = uploaded.getvalue()
            original = Image.open(io.BytesIO(raw)).convert("RGB")
            st.markdown("### X-ray preview")
            # Brighten only the on-screen preview. Prediction, Grad-CAM and PDF retain original pixels.
            preview_brightness = 1.25
            preview_image = ImageEnhance.Brightness(original).enhance(preview_brightness)
            preview_image = ImageEnhance.Contrast(preview_image).enhance(1.05)
            st.image(
                preview_image,
                caption="Brightness-enhanced preview (display only) — AI analysis and PDF use the original image",
                width=520,
            )

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                temp.write(raw)
                temp_path = temp.name

            with st.spinner("Checking image quality and running the AI pipeline..."):
                valid, validation_percent = validate_xray(temp_path)
                quality = assess_image_quality(temp_path)

            st.markdown('<div class="section-kicker">Image validation & quality</div>', unsafe_allow_html=True)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Validator output", f"{float(validation_percent):.2f}%")
            q2.metric("Quality score", f"{quality.get('score', 'N/A')}/100")
            q3.metric("Sharpness", str(quality.get("sharpness", "N/A")))
            q4.metric("Quality risk", str(quality.get("risk_level", "N/A")))

            with st.expander("View quality details"):
                st.write({
                    "Resolution": quality.get("resolution"),
                    "Brightness": quality.get("brightness"),
                    "Contrast": quality.get("contrast"),
                    "Review reason": quality.get("review_reason"),
                })

            if not valid:
                st.error(
                    f"The X-ray validator returned {float(validation_percent):.2f}% "
                    "against the configured threshold. Prediction is withheld."
                )
                st.info("The validator can make mistakes. Passing validation does not prove an image is a chest X-ray.")
            else:
                with st.spinner("Generating prediction and Grad-CAM..."):
                    _, input_tensor = preprocess_image(io.BytesIO(raw))
                    predicted_class, confidence, probs = predict_image(input_tensor)
                    heatmap = generate_gradcam(original, input_tensor)

                st.markdown('<div class="section-kicker">Model output</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted class", str(predicted_class))
                m2.metric("Model probability", f"{float(confidence):.2f}%")
                m3.metric("X-ray validator", f"{float(validation_percent):.2f}%")
                st.caption("These are model outputs, not a verified diagnosis or a measure of clinical accuracy.")

                left, right = st.columns(2)
                with left:
                    st.markdown("### Probability distribution")
                    prob_table = [
                        {"Class": str(name), "Probability (%)": round(float(p) * 100, 2)}
                        for name, p in zip(CLASSES, probs)
                    ]
                    st.dataframe(prob_table, use_container_width=True, hide_index=True)
                    chart_data = {
                        str(name): float(p) * 100 for name, p in zip(CLASSES, probs)
                    }
                    st.bar_chart(chart_data, y_label="Model probability (%)")
                with right:
                    st.markdown("### Grad-CAM visualization")
                    st.image(
                        heatmap,
                        caption="Grad-CAM explanation aid — not proof of clinically valid features.",
                        use_container_width=True,
                    )

                st.markdown("### Research report")
                st.caption("The PDF matches the reference layout: page 1 validation, source X-ray, Grad-CAM and quality; page 2 AI summary, probabilities and limitations; page 3 system status and clinical review.")
                pdf_bytes = make_pdf(
                    original=original,
                    heatmap=heatmap,
                    predicted_class=str(predicted_class),
                    confidence=float(confidence),
                    probs=np.asarray(probs),
                    validation_percent=float(validation_percent),
                    valid=bool(valid),
                    quality=quality,
                    patient_details=patient_details,
                )
                st.download_button(
                    "Download research report (PDF)",
                    data=pdf_bytes,
                    file_name=f"svm_xray_report_{exam_date.strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception:
            st.error("The image could not be processed. Check the model files and installed dependencies.")
            st.exception(__import__("sys").exc_info()[1])
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

st.divider()
st.caption("SVM Hospital Diagnostic Center • Biomedical Engineering research project by Mukesh Kumar")
