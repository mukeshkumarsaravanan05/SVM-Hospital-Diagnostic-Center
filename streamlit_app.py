from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

# Keep imports aligned with the existing project modules.
from ai_engine.preprocessing import preprocess_image
from ai_engine.prediction import CLASSES, predict_image
from ai_engine.image_validator import validate_xray
from ai_engine.image_quality import assess_image_quality
from ai_engine.gradcam import generate_gradcam

st.set_page_config(
    page_title="SVM Hospital Diagnostic Center | Research Demo",
    page_icon="🩻",
    layout="wide",
)

st.title("🩻 SVM Hospital Diagnostic Center")
st.caption("AI-assisted chest X-ray classification and explainability — research demonstration")

st.warning(
    "Research and educational use only. This tool is not a medical device or a standalone "
    "diagnostic system. Model probabilities are not calibrated clinical certainty. "
    "Do not upload identifiable patient images or personal health information to a public demo."
)

with st.expander("About this demo"):
    st.write(
        "This browser interface reuses the repository's MobileNetV2 prediction, X-ray "
        "validation, image-quality assessment, and Grad-CAM modules. Performance and "
        "clinical validity must be independently evaluated before any clinical use."
    )

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="Use only an image you are authorized to process. Avoid identifiable patient information.",
)

def make_pdf(
    original: Image.Image,
    heatmap: np.ndarray,
    predicted_class: str,
    confidence: float,
    probs: np.ndarray,
    validation_percent: float,
    valid: bool,
    quality: dict,
) -> bytes:
    """Create a simple downloadable research report without patient-identifying fields."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4,
        rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("SVM Hospital Diagnostic Center", styles["Title"]),
        Paragraph("AI-assisted chest X-ray research report", styles["Heading2"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 5 * mm),
        Paragraph(
            "<b>Research/decision-support only.</b> This output is not a diagnosis and must not "
            "replace assessment by a qualified clinician or radiologist.",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
    ]

    rows = [
        ["Item", "Result"],
        ["Model-predicted class", predicted_class],
        ["Model probability for predicted class", f"{confidence:.2f}%"],
        ["X-ray validator probability", f"{validation_percent:.2f}%"],
        ["Validator threshold passed", "Yes" if valid else "No"],
        ["Resolution", str(quality.get("resolution", "Not available"))],
        ["Brightness", str(quality.get("brightness", "Not available"))],
        ["Contrast", str(quality.get("contrast", "Not available"))],
        ["Sharpness", str(quality.get("sharpness", "Not available"))],
        ["Image-quality score", f"{quality.get('score', 'Not available')}/100"],
        ["Quality risk", str(quality.get("risk_level", "Not available"))],
    ]
    table = Table(rows, colWidths=[70 * mm, 95 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FB")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [table, Spacer(1, 5 * mm), Paragraph("Model probability distribution", styles["Heading2"])]
    prob_rows = [["Class", "Probability"]]
    for name, value in zip(CLASSES, probs):
        prob_rows.append([name, f"{float(value) * 100:.2f}%"])
    ptable = Table(prob_rows, colWidths=[100 * mm, 65 * mm], repeatRows=1)
    ptable.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [ptable, Spacer(1, 5 * mm), Paragraph("Source image and Grad-CAM visualization", styles["Heading2"])]

    original_buffer = io.BytesIO()
    original.convert("RGB").save(original_buffer, format="JPEG", quality=90)
    original_buffer.seek(0)
    heatmap_buffer = io.BytesIO()
    Image.fromarray(heatmap.astype(np.uint8)).save(heatmap_buffer, format="JPEG", quality=90)
    heatmap_buffer.seek(0)
    image_table = Table(
        [[RLImage(original_buffer, width=78 * mm, height=65 * mm),
          RLImage(heatmap_buffer, width=78 * mm, height=65 * mm)]],
        colWidths=[82 * mm, 82 * mm],
    )
    image_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [image_table, Spacer(1, 4 * mm),
              Paragraph("Grad-CAM is a model-explanation visualization and is not proof that the model focused on clinically valid features.", styles["BodyText"])]
    doc.build(story)
    return output.getvalue()


if uploaded is not None:
    try:
        raw = uploaded.getvalue()
        original = Image.open(io.BytesIO(raw)).convert("RGB")
        st.image(original, caption="Uploaded image", width=420)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp.write(raw)
            temp_path = temp.name

        try:
            with st.spinner("Checking image and running the AI pipeline..."):
                valid, validation_percent = validate_xray(temp_path)
                quality = assess_image_quality(temp_path)

                if not valid:
                    st.error(
                        f"The X-ray validator returned {validation_percent:.2f}% "
                        "against the configured 90% threshold. Prediction is withheld."
                    )
                    st.info("This validator is itself a model and can make mistakes. A passed result does not prove the image is a chest X-ray.")
                else:
                    _, input_tensor = preprocess_image(io.BytesIO(raw))
                    predicted_class, confidence, probs = predict_image(input_tensor)
                    heatmap = generate_gradcam(original, input_tensor)

                    st.subheader("AI output")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted class", predicted_class)
                    m2.metric("Model probability", f"{float(confidence):.2f}%")
                    m3.metric("X-ray validator", f"{float(validation_percent):.2f}%")

                    st.caption(
                        "These are model outputs, not a verified diagnosis or a measure of clinical accuracy."
                    )

                    left, right = st.columns(2)
                    with left:
                        st.markdown("#### Probability distribution")
                        prob_table = [
                            {"Class": name, "Probability (%)": round(float(p) * 100, 2)}
                            for name, p in zip(CLASSES, probs)
                        ]
                        st.dataframe(prob_table, use_container_width=True, hide_index=True)
                        st.bar_chart(
                            {name: float(p) * 100 for name, p in zip(CLASSES, probs)},
                            y_label="Model probability (%)",
                        )
                    with right:
                        st.markdown("#### Grad-CAM visualization")
                        st.image(heatmap, caption="Model attention visualization (Grad-CAM)", use_container_width=True)

                    st.markdown("#### Image-quality assessment")
                    q1, q2, q3 = st.columns(3)
                    q1.metric("Score", f"{quality.get('score', 'N/A')}/100")
                    q2.metric("Quality risk", str(quality.get("risk_level", "N/A")))
                    q3.metric("Sharpness", str(quality.get("sharpness", "N/A")))
                    st.write({
                        "Resolution": quality.get("resolution"),
                        "Brightness": quality.get("brightness"),
                        "Contrast": quality.get("contrast"),
                        "Review reason": quality.get("review_reason"),
                    })

                    pdf_bytes = make_pdf(
                        original, heatmap, predicted_class, float(confidence),
                        probs, float(validation_percent), bool(valid), quality,
                    )
                    st.download_button(
                        "Download research report (PDF)",
                        data=pdf_bytes,
                        file_name="svm_xray_research_report.pdf",
                        mime="application/pdf",
                    )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    except Exception as exc:
        st.error("The image could not be processed. Check the model files and installed dependencies.")
        st.exception(exc)

st.divider()
st.caption("SVM Hospital Diagnostic Center • Biomedical Engineering project by Mukesh Kumar")
