from __future__ import annotations
import io, os, tempfile, traceback
from datetime import datetime
import streamlit as st
from PIL import Image

st.set_page_config(page_title="SVM Hospital Diagnostic Center", page_icon="🩻", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(ellipse at 10% 0%,#142844 0,transparent 38%),radial-gradient(ellipse at 90% 8%,#0d3439 0,transparent 30%),#080d18;color:#e8eef8}
[data-testid="stHeader"]{background:rgba(8,13,24,.8)} [data-testid="stSidebar"]{background:#0d1728;border-right:1px solid #26364b}
.block-container{max-width:1400px;padding-top:1.5rem}.hero{border:1px solid #294c6c;border-radius:20px;padding:28px;background:linear-gradient(125deg,#142944,#0c192a 60%,#0b3037);margin-bottom:18px}
.eyebrow,.kicker{color:#69dfd2;text-transform:uppercase;letter-spacing:.13em;font-size:.76rem;font-weight:800}.hero h1{color:#f4f8ff;font-size:2.6rem;margin:.4rem 0}.hero p{color:#b6c7dd;line-height:1.6;max-width:900px}
.panel{background:#111e30;border:1px solid #293a50;border-radius:15px;padding:18px;height:100%}.panel h3{color:#eff6ff}.panel p{color:#aabbd0}.disclaimer{border-left:4px solid #e7b35c;background:#a06b161f;padding:14px 17px;border-radius:0 12px 12px 0;color:#e8d7b6}
.stButton button,.stDownloadButton button{border-radius:10px;background:#087f8c;color:white;font-weight:700} a{color:#75e6db!important} footer{visibility:hidden}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🩻 SVM Research")
    st.caption("Biomedical AI · Explainable imaging")
    page=st.radio("Navigation",["Overview","Analyze X-ray","Workflow & limitations"])
    st.divider()
    st.markdown("[GitHub repository ↗](https://github.com/mukeshkumarsaravanan05/SVM-Hospital-Diagnostic-Center)")
    st.caption("Built by Mukesh Kumar · Biomedical Engineering")

st.markdown("""
<div class="hero"><div class="eyebrow">Biomedical AI · Research demonstration</div>
<h1>SVM Hospital Diagnostic Center</h1>
<p>AI-assisted chest X-ray classification research interface with MobileNetV2, class-probability visualization, Grad-CAM explainability and research reporting.</p>
</div>
<div class="disclaimer"><b>Research and educational use only.</b> This is not a medical device or standalone diagnostic system. Model outputs are not a diagnosis or verified clinical accuracy. Never use them to make clinical decisions or delay professional care. Do not upload identifiable patient images to this public demo.</div>
""",unsafe_allow_html=True)

if page=="Overview":
    st.subheader("Project overview")
    c1,c2,c3=st.columns(3)
    with c1: st.markdown('<div class="panel"><div class="kicker">CLASSIFICATION</div><h3>Five classes</h3><p>COVID, Lung Opacity, Normal, Pneumonia and Tuberculosis model outputs.</p></div>',unsafe_allow_html=True)
    with c2: st.markdown('<div class="panel"><div class="kicker">EXPLAINABILITY</div><h3>Grad-CAM</h3><p>Visual explanation aid associated with the model output; not proof of clinical reasoning.</p></div>',unsafe_allow_html=True)
    with c3: st.markdown('<div class="panel"><div class="kicker">RESEARCH TOOLS</div><h3>Quality & report</h3><p>Image checks, model probabilities and a downloadable research report when dependencies are available.</p></div>',unsafe_allow_html=True)
    st.subheader("Analysis workflow")
    st.write("**01 Upload** → **02 Validate/check quality** → **03 Predict** → **04 Explain with Grad-CAM** → **05 Export report")
    st.info("This web interface loads the AI pipeline only when you open Analyze X-ray. This keeps the landing page responsive and makes model-loading errors visible rather than blocking the whole app.")
    st.markdown("**Source code:** [GitHub](https://github.com/mukeshkumarsaravanan05/SVM-Hospital-Diagnostic-Center)")

elif page=="Analyze X-ray":
    st.subheader("Analyze a chest X-ray")
    st.write("Upload only an image you are authorized to use. Remove patient names, IDs, dates of birth and other identifying information first.")
    uploaded=st.file_uploader("Upload image",type=["png","jpg","jpeg","bmp","webp"])
    if uploaded is not None:
        try:
            original=Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            st.image(original,caption="Uploaded image",width=420)
            if st.button("Run AI analysis",type="primary"):
                with st.spinner("Loading AI modules and analyzing image…"):
                    # Imports are intentionally delayed so a missing AI dependency cannot blank the landing page.
                    import numpy as np
                    from ai_engine.preprocessing import preprocess_image
                    from ai_engine.prediction import CLASSES, predict_image
                    from ai_engine.gradcam import generate_gradcam
                    raw=uploaded.getvalue()
                    _,tensor=preprocess_image(io.BytesIO(raw))
                    pred,confidence,probs=predict_image(tensor)
                    heatmap=generate_gradcam(original,tensor)
                st.success("Model inference completed. This is a research output, not a diagnosis.")
                a,b,c=st.columns(3)
                a.metric("Predicted class",str(pred)); b.metric("Model probability",f"{float(confidence):.2f}%"); c.metric("Classes",str(len(CLASSES)))
                left,right=st.columns(2)
                with left:
                    st.markdown("### Probability distribution")
                    st.dataframe([{"Class":str(n),"Probability (%)":round(float(p)*100,2)} for n,p in zip(CLASSES,probs)],use_container_width=True,hide_index=True)
                    st.bar_chart({str(n):float(p)*100 for n,p in zip(CLASSES,probs)})
                with right:
                    st.markdown("### Grad-CAM")
                    st.image(heatmap,caption="Explanation aid only; not proof of clinically valid features.",use_container_width=True)
                try:
                    from reportlab.lib import colors
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib.styles import getSampleStyleSheet
                    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
                    from reportlab.lib.units import mm
                    buf=io.BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=18*mm,rightMargin=18*mm)
                    styles=getSampleStyleSheet(); rows=[["Class","Probability"]]+[[str(n),f"{float(p)*100:.2f}%"] for n,p in zip(CLASSES,probs)]
                    tbl=Table(rows,repeatRows=1); tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#173B57")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),("PADDING",(0,0),(-1,-1),7)]))
                    doc.build([Paragraph("SVM Hospital Diagnostic Center",styles["Title"]),Paragraph("AI-assisted chest X-ray research output",styles["Heading2"]),Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M"),styles["Normal"]),Spacer(1,8*mm),Paragraph("Research/educational use only. Not a diagnosis; not for clinical decisions.",styles["BodyText"]),Spacer(1,5*mm),Paragraph(f"Predicted class: {pred} | Model probability: {float(confidence):.2f}%",styles["Heading3"]),tbl])
                    st.download_button("Download research PDF",data=buf.getvalue(),file_name="svm_xray_research_report.pdf",mime="application/pdf")
                except Exception as pdf_err:
                    st.warning(f"Prediction worked, but PDF export is unavailable: {pdf_err}")
        except Exception as e:
            st.error("AI analysis could not start. The website itself is running; check the specific error below.")
            st.code(f"{type(e).__name__}: {e}")
            st.caption("Confirm model file exists at models/disease_mobilenetv2_clean_v2.pth and that web dependencies match the project. Do not share patient data in logs.")

else:
    st.subheader("Workflow & limitations")
    st.markdown("**Pipeline:** Upload → preprocessing → MobileNetV2 prediction → class probabilities → Grad-CAM → research PDF.")
    st.markdown("- Model probabilities are not calibrated clinical certainty.\n- Grad-CAM does not establish that clinically meaningful features drove a prediction.\n- A successful run does not establish clinical performance.\n- Do not use outputs to make clinical decisions or delay professional care.")

st.divider()
st.caption("SVM Hospital Diagnostic Center · Biomedical Engineering project by Mukesh Kumar · Research demo only")
