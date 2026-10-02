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
    """Create a research PDF with examination details, metrics, image, and Grad-CAM."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, KeepTogether,
    )

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SmallMuted", parent=styles["BodyText"], fontSize=8,
        leading=11, textColor=colors.HexColor("#526477"),
    ))
    story = [
        Paragraph("SVM Hospital Diagnostic Center", styles["Title"]),
        Paragraph("AI-Assisted Chest X-ray Analysis — Research Report", styles["Heading2"]),
        Paragraph(f"Report generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph(
            "<b>Research / educational use only.</b> This report is not a diagnosis, "
            "and model probabilities are not calibrated clinical certainty. Results must "
            "be reviewed by a qualified clinician or radiologist.",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("Patient & Examination Details", styles["Heading2"]),
    ]

    details_rows = [["Field", "Recorded value"]]
    for label, key in [
        ("Patient / case ID", "case_id"),
        ("Patient name / anonymized label", "patient_label"),
        ("Age", "age"),
        ("Sex", "sex"),
        ("Examination date", "exam_date"),
        ("Clinical indication / notes", "notes"),
    ]:
        value = str(patient_details.get(key, "") or "Not provided")
        value = escape(value).replace("\n", "<br/>")
        details_rows.append([label, Paragraph(value, styles["BodyText"])])
    details_table = Table(details_rows, colWidths=[62 * mm, 103 * mm], repeatRows=1)
    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FB")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [details_table, Spacer(1, 5 * mm), Paragraph("Analysis Results", styles["Heading2"])]

    rows = [
        ["Item", "Result"],
        ["Model-predicted class", predicted_class],
        ["Model probability for predicted class", f"{confidence:.2f}%"],
        ["X-ray validator output", f"{validation_percent:.2f}%"],
        ["Validator threshold passed", "Yes" if valid else "No"],
        ["Resolution", str(quality.get("resolution", "Not available"))],
        ["Brightness", str(quality.get("brightness", "Not available"))],
        ["Contrast", str(quality.get("contrast", "Not available"))],
        ["Sharpness", str(quality.get("sharpness", "Not available"))],
        ["Image-quality score", f"{quality.get('score', 'Not available')}/100"],
        ["Quality risk", str(quality.get("risk_level", "Not available"))],
        ["Review reason", str(quality.get("review_reason", "Not available"))],
    ]
    table = Table(rows, colWidths=[75 * mm, 90 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FB")]),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [table, Spacer(1, 5 * mm), Paragraph("Model Probability Distribution", styles["Heading2"])]

    prob_rows = [["Class", "Model probability"]]
    for name, value in zip(CLASSES, probs):
        prob_rows.append([str(name), f"{float(value) * 100:.2f}%"])
    prob_table = Table(prob_rows, colWidths=[100 * mm, 65 * mm], repeatRows=1)
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [prob_table, Spacer(1, 5 * mm), Paragraph("Source X-ray & Grad-CAM Visualization", styles["Heading2"])]

    original_buffer = io.BytesIO()
    original.convert("RGB").save(original_buffer, format="JPEG", quality=90)
    original_buffer.seek(0)
    heatmap_buffer = io.BytesIO()
    Image.fromarray(np.asarray(heatmap).astype(np.uint8)).convert("RGB").save(
        heatmap_buffer, format="JPEG", quality=90
    )
    heatmap_buffer.seek(0)
    image_table = Table(
        [[RLImage(original_buffer, width=77 * mm, height=62 * mm),
          RLImage(heatmap_buffer, width=77 * mm, height=62 * mm)]],
        colWidths=[82 * mm, 82 * mm],
    )
    image_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [
        image_table,
        Spacer(1, 3 * mm),
        Paragraph(
            "Grad-CAM is a model-explanation visualization, not proof that the model "
            "focused on clinically valid features. Validator scores and model probabilities "
            "are not substitutes for clinical assessment.",
            styles["SmallMuted"],
        ),
    ]
    doc.build(story)
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

