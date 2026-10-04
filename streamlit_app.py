from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from html import escape

import numpy as np
import streamlit as st
from PIL import Image

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
        --bg: #07111f;
        --panel: #101e30;
        --panel2: #14263b;
        --line: #2b4159;
        --text: #f2f7ff;
        --muted: #a9b8ca;
        --teal: #42ead8;
        --blue: #60a5fa;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 88% 8%, rgba(13, 148, 136, .16), transparent 34%),
            radial-gradient(ellipse at 15% 22%, rgba(37, 99, 235, .12), transparent 36%),
            var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1728 0%, #08111e 100%);
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
        background: rgba(16, 30, 48, .88); margin-bottom: 14px;
    }
    div[data-testid="stMetric"] {
        background: #142337; border: 1px solid #2b4159; border-radius: 14px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #c5d3e3 !important; }
    div[data-testid="stMetricValue"] { color: #f4f8ff !important; }
    div[data-testid="stFileUploader"] {
        background: #111f31; border: 1px dashed #36bdb6; border-radius: 14px; padding: 10px;
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
    """Build a structured, three-page research report matching the supplied reference."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, KeepTogether,
    )

    navy = colors.HexColor("#12334D")
    pale = colors.HexColor("#EAF2F8")
    stripe = colors.HexColor("#F4F8FB")
    border = colors.HexColor("#C9D7E2")
    ink = colors.HexColor("#253746")
    muted = colors.HexColor("#647789")
    amber = colors.HexColor("#FFF7E2")

    report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    exam_date = str(patient_details.get("exam_date") or datetime.now().strftime("%Y-%m-%d"))
    assessment_level = str(quality.get("risk_level", "REVIEW REQUIRED")).upper()
    if assessment_level in {"LOW", "LOW RISK"}:
        assessment_level = "LOW"
    else:
        assessment_level = "MODERATE" if "MODERATE" in assessment_level else assessment_level

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=17 * mm,
        title="AI-Assisted Chest X-ray Diagnostic Report",
        author="SVM Hospital Diagnostic Center",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=17, leading=20, alignment=TA_LEFT, textColor=colors.white,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontSize=9,
        leading=12, textColor=colors.HexColor("#DCE8F1"),
    ))
    styles.add(ParagraphStyle(
        name="SectionBar", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=9, leading=12, textColor=colors.white, backColor=navy,
        borderPadding=(5, 7, 5, 7), spaceBefore=2 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="SmallMuted", parent=styles["BodyText"], fontSize=7.5,
        leading=10, textColor=muted,
    ))
    styles.add(ParagraphStyle(
        name="Cell", parent=styles["BodyText"], fontSize=8, leading=10,
        textColor=ink,
    ))
    styles.add(ParagraphStyle(
        name="CellBold", parent=styles["Cell"], fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="CenterBig", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=16, leading=19, alignment=TA_CENTER, textColor=navy,
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer", parent=styles["BodyText"], fontSize=8,
        leading=11, textColor=ink, backColor=amber, borderPadding=7,
    ))

    def section(text):
        return Paragraph(text.upper(), styles["SectionBar"])

    def styled_table(rows, widths, header=True, font_size=8):
        table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.45, border),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, stripe]),
        ]
        if header:
            commands += [
                ("BACKGROUND", (0, 0), (-1, 0), pale),
                ("TEXTCOLOR", (0, 0), (-1, 0), navy),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        table.setStyle(TableStyle(commands))
        return table

    def footer(canvas, document):
        canvas.saveState()
        page_w, _ = A4
        canvas.setStrokeColor(border)
        canvas.setLineWidth(0.5)
        canvas.line(15 * mm, 12 * mm, page_w - 15 * mm, 12 * mm)
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(muted)
        canvas.drawString(15 * mm, 7.5 * mm, "CONFIDENTIAL • AI-ASSISTED DECISION SUPPORT • RESEARCH USE ONLY")
        canvas.drawRightString(page_w - 15 * mm, 7.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    def safe_text(value):
        return escape(str(value if value not in (None, "") else "Not provided")).replace("\n", "<br/>")

    story = []

    # PAGE 1 — Report metadata, validation, images and image quality.
    banner = Table([[
        [
            Paragraph("SVM HOSPITAL DIAGNOSTIC CENTER", styles["ReportTitle"]),
            Paragraph("AI-Assisted Chest X-ray Diagnostic Report", styles["ReportSubtitle"]),
        ],
        [
            Paragraph("<b>CONFIDENTIAL</b><br/>" + datetime.now().strftime("%d %b %Y"), styles["ReportSubtitle"]),
            Spacer(1, 4 * mm),
            Paragraph("<b>RESEARCH<br/>DEMONSTRATION</b>", styles["ReportSubtitle"]),
        ],
    ]], colWidths=[116 * mm, 49 * mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), navy),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [banner, Spacer(1, 4 * mm)]

    metadata = [
        ["REPORT TYPE", "REPORT ID", "GENERATED"],
        ["AI Chest X-ray Assessment", report_id, generated],
    ]
    story += [styled_table(metadata, [55 * mm, 50 * mm, 60 * mm]), section("X-ray validation")]
    validation_rows = [
        ["Validation item", "Result"],
        ["X-ray validator probability", f"{float(validation_percent):.2f}%"],
        ["Configured validation threshold", "Project-configured threshold (see application settings)"],
        ["Validation result", "VALIDATOR PASSED" if valid else "VALIDATOR DID NOT PASS"],
    ]
    story += [styled_table(validation_rows, [72 * mm, 93 * mm]), section("Source radiograph & AI explanation")]

    original_buffer = io.BytesIO()
    original.convert("RGB").save(original_buffer, format="JPEG", quality=90)
    original_buffer.seek(0)
    heatmap_buffer = io.BytesIO()
    Image.fromarray(np.asarray(heatmap).astype(np.uint8)).convert("RGB").save(
        heatmap_buffer, format="JPEG", quality=90
    )
    heatmap_buffer.seek(0)
    image_table = Table([
        [Paragraph("<b>ORIGINAL CHEST X-RAY</b>", styles["Cell"]),
         Paragraph("<b>GRAD-CAM AI ATTENTION MAP</b>", styles["Cell"])],
        [RLImage(original_buffer, width=75 * mm, height=59 * mm, kind="proportional"),
         RLImage(heatmap_buffer, width=75 * mm, height=59 * mm, kind="proportional")],
    ], colWidths=[82.5 * mm, 82.5 * mm], hAlign="LEFT")
    image_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.45, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [image_table, section("Image quality assessment")]
    quality_rows = [["Metric", "Result"]]
    for label, key in [
        ("Resolution", "resolution"), ("Brightness", "brightness"),
        ("Contrast", "contrast"), ("Sharpness", "sharpness"),
    ]:
        quality_rows.append([label, str(quality.get(key, "Not available"))])
    quality_rows += [
        ["Overall quality score", f"{quality.get('score', 'Not available')}/100"],
        ["Quality risk", str(quality.get("risk_level", "Not available")).upper()],
    ]
    story += [styled_table(quality_rows, [72 * mm, 93 * mm]), PageBreak()]

    # PAGE 2 — Diagnosis summary, probability distribution and limitations.
    story += [section("AI diagnostic summary")]
    summary = Table([
        [Paragraph("<b>AI PREDICTED CONDITION</b>", styles["Cell"]),
         Paragraph("<b>MODEL CONFIDENCE</b>", styles["Cell"])],
        [Paragraph(escape(str(predicted_class)).upper(), styles["CenterBig"]),
         Paragraph(f"{float(confidence):.2f}%", styles["CenterBig"])],
    ], colWidths=[82.5 * mm, 82.5 * mm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale),
        ("GRID", (0, 0), (-1, -1), 0.45, border),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [summary, section("AI assessment status")]
    status = Table([
        [Paragraph("<b>ASSESSMENT LEVEL</b>", styles["Cell"]),
         Paragraph("<b>CLINICAL REVIEW</b>", styles["Cell"])],
        [Paragraph(escape(assessment_level), styles["CenterBig"]),
         Paragraph("REQUIRED", styles["CenterBig"])],
        ["MODEL OUTPUT", "IMAGE QUALITY"],
        [f"{float(confidence):.2f}%", f"{quality.get('score', 'Not available')}/100"],
    ], colWidths=[82.5 * mm, 82.5 * mm])
    status.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale),
        ("BACKGROUND", (0, 2), (-1, 2), stripe),
        ("TEXTCOLOR", (0, 2), (-1, 2), navy),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.45, border),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [status, section("AI probability distribution")]
    prob_rows = [["Disease class", "Probability"]]
    for name, value in zip(CLASSES, probs):
        prob_rows.append([str(name), f"{float(value) * 100:.2f}%"])
    story += [styled_table(prob_rows, [82.5 * mm, 82.5 * mm]), section("Interpretation & limitations")]
    story += [
        Paragraph(
            "The predicted class and probability values are outputs of the configured AI pipeline. "
            "They should be interpreted alongside the source radiograph, image quality, clinical "
            "history, and a professional radiological assessment.", styles["BodyText"]),
        Spacer(1, 2 * mm),
        Paragraph(
            "Grad-CAM is an interpretability visualization. It does not establish that highlighted "
            "regions are clinically meaningful, and model probabilities are not calibrated clinical certainty.",
            styles["BodyText"]),
        section("Clinical disclaimer"),
        Paragraph(
            "For research and educational decision-support use only. This AI-generated assessment is "
            "not a standalone medical diagnosis and must not replace evaluation by a qualified clinician "
            "or radiologist. Final clinical decisions must be made by an appropriately qualified professional.",
            styles["Disclaimer"]),
        PageBreak(),
    ]

    # PAGE 3 — System status and space for human clinical review.
    story += [section("AI system status")]
    system_rows = [
        ["System item", "Status"],
        ["Report generation", "Completed"],
        ["Image validation", "Passed" if valid else "Did not pass"],
        ["Prediction output", "Recorded in this report"],
        ["Clinical validation", "Not established by this report"],
        ["Report identifier", report_id],
    ]
    story += [styled_table(system_rows, [72 * mm, 93 * mm]), section("Clinical review")]
    review_rows = [
        [Paragraph("<b>Reviewer name:</b>", styles["Cell"]), ""],
        [Paragraph("<b>Professional title / registration:</b>", styles["Cell"]), ""],
        [Paragraph("<b>Review notes:</b>", styles["Cell"]), ""],
        [Paragraph("<b>Signature:</b>", styles["Cell"]), "________________________________________"],
        [Paragraph("<b>Date:</b>", styles["Cell"]), "________________________________________"],
    ]
    review_table = Table(review_rows, colWidths=[52 * mm, 113 * mm],
                         rowHeights=[12 * mm, 14 * mm, 34 * mm, 13 * mm, 13 * mm])
    review_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), pale),
        ("GRID", (0, 0), (-1, -1), 0.45, border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [
        review_table, Spacer(1, 7 * mm),
        Paragraph(
            "This document records software outputs for the uploaded image only. It does not establish "
            "diagnostic accuracy, clinical validity, or patient outcome.", styles["SmallMuted"]),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
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
            st.image(original, caption="Uploaded X-ray image", width=460)

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
                st.caption("The PDF includes the entered examination details, analysis results, class probabilities, source image, and Grad-CAM visualization.")
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
