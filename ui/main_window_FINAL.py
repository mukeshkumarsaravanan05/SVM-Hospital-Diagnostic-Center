import os
import re
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFileDialog, QLineEdit, QComboBox, QMessageBox,
    QScrollArea, QFrame, QProgressBar, QSizePolicy
)
from ai_engine.image_quality import assess_image_quality
from ai_engine.gradcam import generate_gradcam
from ai_engine.image_validator import validate_xray
from reports.report_generator import generate_pdf as create_pdf

CLASSES = ["COVID", "Lung Opacity", "Normal", "Pneumonia", "Tuberculosis"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.predicted_disease = None
        self.predicted_confidence = None
        self.predicted_probabilities = None
        self.quality_data = None
        self.heatmap_path = None
        self.xray_probability = None
        self.xray_valid = None
        self.setWindowTitle("SVM Hospital Diagnostic Center — AI Chest X-ray Disease Classification")
        self.resize(1400, 950)
        self.setMinimumSize(1100, 760)
        self.build_ui()
        self.apply_theme()

    def card(self, title):
        f = QFrame(); f.setObjectName("card")
        l = QVBoxLayout(f); l.setContentsMargins(18,16,18,18); l.setSpacing(12)
        h = QLabel(title); h.setObjectName("sectionTitle"); l.addWidget(h)
        return f, l

    def build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        header = QFrame(); header.setObjectName("header")
        hl = QVBoxLayout(header); hl.setContentsMargins(28,18,28,16); hl.setSpacing(4)
        t = QLabel("SVM HOSPITAL DIAGNOSTIC CENTER"); t.setObjectName("headerTitle"); t.setAlignment(Qt.AlignCenter)
        s = QLabel("AI-Assisted Chest X-ray Disease Prediction & Explainability"); s.setObjectName("headerSubtitle"); s.setAlignment(Qt.AlignCenter)
        hl.addWidget(t); hl.addWidget(s); outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        content = QWidget(); self.main_layout = QVBoxLayout(content); self.main_layout.setContentsMargins(24,22,24,22); self.main_layout.setSpacing(16); scroll.setWidget(content); outer.addWidget(scroll, 1)

        card, lay = self.card("01  •  PATIENT INFORMATION")
        grid = QGridLayout(); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(10)
        self.patient_id = QLineEdit(); self.patient_name = QLineEdit(); self.patient_age = QLineEdit(); self.patient_gender = QComboBox(); self.patient_gender.addItems(["Male","Female","Other"])
        self.patient_id.setPlaceholderText("Enter patient ID"); self.patient_name.setPlaceholderText("Enter patient name"); self.patient_age.setPlaceholderText("Age")
        fields = [("Patient ID",self.patient_id,0,0),("Patient Name",self.patient_name,0,2),("Age",self.patient_age,1,0),("Gender",self.patient_gender,1,2)]
        for name,w,r,c in fields:
            lab=QLabel(name); lab.setObjectName("fieldLabel"); grid.addWidget(lab,r,c); grid.addWidget(w,r,c+1); grid.setColumnStretch(c+1,1)
        lay.addLayout(grid); self.main_layout.addWidget(card)

        card, lay = self.card("02  •  IMAGE STATUS")
        g=QGridLayout(); g.setHorizontalSpacing(14)
        self.validation_label=self.status_box("X-ray Validation\nAwaiting image")
        self.quality_label=self.status_box("Image Quality\nAwaiting analysis")
        g.addWidget(self.validation_label,0,0); g.addWidget(self.quality_label,0,1); g.setColumnStretch(0,1); g.setColumnStretch(1,1)
        lay.addLayout(g); self.main_layout.addWidget(card)

        card, lay = self.card("03  •  RADIOGRAPH & AI EXPLANATION")
        g=QGridLayout(); g.setHorizontalSpacing(18)
        self.image_label=self.image_panel("Original Chest X-ray"); self.heatmap_label=self.image_panel("Grad-CAM AI Attention Map")
        g.addWidget(self.image_label,0,0); g.addWidget(self.heatmap_label,0,1); g.setColumnStretch(0,1); g.setColumnStretch(1,1)
        lay.addLayout(g); self.main_layout.addWidget(card)

        card, lay = self.card("04  •  AI PREDICTION SUMMARY")
        g=QGridLayout(); g.setHorizontalSpacing(14)
        self.prediction_value=QLabel("—"); self.confidence_value=QLabel("—"); self.assessment_value=QLabel("Awaiting prediction"); self.review_value=QLabel("Clinical/radiological review required")
        for w in (self.prediction_value,self.confidence_value,self.assessment_value,self.review_value): w.setAlignment(Qt.AlignCenter); w.setMinimumHeight(58)
        self.prediction_value.setObjectName("predictionValue"); self.confidence_value.setObjectName("confidenceValue"); self.assessment_value.setObjectName("assessmentValue"); self.review_value.setObjectName("reviewValue")
        for text,r,c in [("PREDICTED CLASS",0,0),("MODEL CONFIDENCE",0,1),("AI ASSESSMENT",2,0),("REVIEW STATUS",2,1)]:
            h=QLabel(text); h.setAlignment(Qt.AlignCenter); h.setObjectName("smallHeading"); g.addWidget(h,r,c)
        g.addWidget(self.prediction_value,1,0); g.addWidget(self.confidence_value,1,1); g.addWidget(self.assessment_value,3,0); g.addWidget(self.review_value,3,1); g.setColumnStretch(0,1); g.setColumnStretch(1,1)
        lay.addLayout(g); self.main_layout.addWidget(card)

        card, lay = self.card("05  •  DISEASE PROBABILITY DISTRIBUTION")
        self.probability_bars={}
        for disease in CLASSES:
            row=QHBoxLayout(); name=QLabel(disease); name.setMinimumWidth(125); name.setObjectName("probabilityName")
            bar=QProgressBar(); bar.setRange(0,10000); bar.setValue(0); bar.setTextVisible(False); bar.setFixedHeight(16)
            val=QLabel("0.00%"); val.setMinimumWidth(65); val.setAlignment(Qt.AlignRight|Qt.AlignVCenter); val.setObjectName("probabilityValue")
            row.addWidget(name); row.addWidget(bar,1); row.addWidget(val); lay.addLayout(row); self.probability_bars[disease]=(bar,val)
        self.main_layout.addWidget(card)

        notice=QLabel("Research / decision-support use only. AI output is not a standalone medical diagnosis and must be interpreted with the source radiograph, clinical history and qualified professional assessment.")
        notice.setWordWrap(True); notice.setObjectName("notice"); self.main_layout.addWidget(notice)

        actions=QFrame(); actions.setObjectName("actionBar"); al=QHBoxLayout(actions); al.setContentsMargins(18,12,18,12); al.setSpacing(12)
        self.btn_upload=QPushButton("UPLOAD CHEST X-RAY"); self.btn_predict=QPushButton("PREDICT DISEASE"); self.btn_report=QPushButton("GENERATE PDF REPORT")
        self.btn_upload.setObjectName("secondaryButton"); self.btn_predict.setObjectName("primaryButton"); self.btn_report.setObjectName("secondaryButton")
        for b in (self.btn_upload,self.btn_predict,self.btn_report): b.setMinimumHeight(48); b.setCursor(Qt.PointingHandCursor); al.addWidget(b)
        actions.setFixedHeight(70)
        actions.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
        )
        outer.addWidget(actions, 0)
        self.btn_upload.clicked.connect(self.upload_image); self.btn_predict.clicked.connect(self.predict_disease); self.btn_report.clicked.connect(self.generate_pdf)

    def status_box(self,text):
        w=QLabel(text); w.setAlignment(Qt.AlignCenter); w.setMinimumHeight(70); w.setWordWrap(True); w.setObjectName("statusBox"); return w

    def image_panel(self,title):
        w=QLabel("No image loaded"); w.setAlignment(Qt.AlignCenter); w.setMinimumSize(420,360); w.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding); w.setObjectName("imageDisplay"); w.setToolTip(title); return w

    def apply_theme(self):
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#F4F7FB;color:#203040;font-family:'Segoe UI';font-size:13px}
        QFrame#header{background:#0B3558;border:none}
        QLabel#headerTitle {
    color: #0B3558;font-size:27px;font-weight:700}
        QLabel#headerSubtitle {
    color: #1976D2;font-size:14px}
        QFrame#card{background:white;border:1px solid #D9E3EC;border-radius:12px}
        QLabel#sectionTitle{color:#0B3558;font-size:15px;font-weight:700}
        QLabel#fieldLabel{color:#334A5F;font-weight:600}
        QLineEdit,QComboBox{background:#FBFDFF;border:1px solid #C8D6E3;border-radius:6px;padding:9px 10px;min-height:20px}
        QLineEdit:focus,QComboBox:focus{border:1px solid #1976D2}
        QLabel#statusBox{background:#F6FAFD;border:1px solid #D5E4EF;border-radius:8px;color:#29465D;font-weight:600;padding:8px}
        QLabel#imageDisplay{background:#F9FBFD;border:2px solid #D5E0E8;border-radius:8px;color:#7A8996}
        QLabel#smallHeading{color:#708394;font-size:11px;font-weight:700;padding:4px}
        QLabel#predictionValue{color:#0B3558;font-size:25px;font-weight:800;background:#F3F8FC;border:1px solid #D5E5F0;border-radius:8px}
        QLabel#confidenceValue{color:#16724A;font-size:25px;font-weight:800;background:#F2FAF6;border:1px solid #CDE8DA;border-radius:8px}
        QLabel#assessmentValue,QLabel#reviewValue{background:#FFF8E8;border:1px solid #EAD9AA;border-radius:8px;color:#725A18;font-weight:700}
        QLabel#probabilityName{font-weight:600;color:#334A5F} QLabel#probabilityValue{font-weight:700;color:#203040}
        QProgressBar{background:#E9EFF4;border:none;border-radius:8px} QProgressBar::chunk{background:#1976D2;border-radius:8px}
        QLabel#notice{background:#FFF8E8;border:1px solid #E5D29A;border-radius:8px;color:#66521C;padding:12px}
        QFrame#actionBar{background:white;border-top:1px solid #D5E0E8}
        QPushButton{border-radius:7px;font-weight:700;padding:10px 18px}
        QPushButton#primaryButton{background:#1976D2;color:white;border:1px solid #1565C0} QPushButton#primaryButton:hover{background:#1565C0}
        QPushButton#secondaryButton{background:white;color:#145A8D;border:1px solid #AFC6D8} QPushButton#secondaryButton:hover{background:#F0F7FC}
        """)

    def set_image(self,label,pixmap):
        if not pixmap.isNull(): label.setPixmap(pixmap.scaled(label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.image_path and os.path.exists(self.image_path): self.set_image(self.image_label,QPixmap(self.image_path))

    def upload_image(self):
        path,_=QFileDialog.getOpenFileName(self,"Select Chest X-ray","","Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:return
        pix=QPixmap(path)
        if pix.isNull(): QMessageBox.warning(self,"Invalid Image","Unable to load the selected image."); return
        self.image_path=path; self.predicted_disease=None; self.predicted_confidence=None; self.predicted_probabilities=None; self.quality_data=None; self.heatmap_path=None; self.xray_probability=None; self.xray_valid=None
        self.set_image(self.image_label,pix); self.heatmap_label.clear(); self.heatmap_label.setText("Grad-CAM will appear after prediction")
        self.validation_label.setText("X-ray Validation\nAwaiting analysis"); self.quality_label.setText("Image Quality\nAwaiting analysis"); self.prediction_value.setText("—"); self.confidence_value.setText("—"); self.assessment_value.setText("Awaiting prediction"); self.review_value.setText("Clinical/radiological review required")
        for bar,val in self.probability_bars.values(): bar.setValue(0); val.setText("0.00%")

    def predict_disease(self):
        if not self.image_path: QMessageBox.warning(self,"No Image","Please upload a Chest X-ray image first."); return
        pid=self.patient_id.text().strip(); name=self.patient_name.text().strip(); age=self.patient_age.text().strip(); missing=[]
        if not pid: missing.append("Patient ID")
        if not name: missing.append("Patient Name")
        if not age: missing.append("Age")
        if missing: QMessageBox.warning(self,"Patient Information Required","Please enter:\n\n"+"\n".join("• "+x for x in missing)); return
        if not re.fullmatch(r"[A-Za-z]+(?:[ .'-][A-Za-z]+)*",name): QMessageBox.warning(self,"Invalid Patient Name","Use letters, spaces, apostrophes or hyphens only."); return
        try:
            av=int(age)
            if av<1 or av>120: raise ValueError
        except ValueError: QMessageBox.warning(self,"Invalid Age","Enter a valid age between 1 and 120."); return
        try: self.xray_valid,self.xray_probability=validate_xray(self.image_path)
        except Exception as e: QMessageBox.critical(self,"X-ray Validation Error",str(e)); return
        p=f" ({self.xray_probability:.2f}%)" if self.xray_probability is not None else ""
        self.validation_label.setText(f"X-ray Validation\n{'VALID CHEST X-RAY' if self.xray_valid else 'NOT A VALID CHEST X-RAY'}{p}")
        if not self.xray_valid: QMessageBox.warning(self,"Invalid Image","The uploaded image does not appear to be a valid chest X-ray."); return
        try: self.quality_data=assess_image_quality(self.image_path)
        except Exception as e: QMessageBox.critical(self,"Image Quality Error",str(e)); return
        q=float(self.quality_data.get("score",0)); risk=self.quality_data.get("risk_level","UNKNOWN"); reason=self.quality_data.get("review_reason","Not available")
        self.quality_label.setText(f"Image Quality\n{q:.0f}/100 • {risk}\n{reason}")
        if q<70: QMessageBox.warning(self,"Poor Image Quality",f"Image Quality Score: {q:.0f}/100\n\nPlease consider using a clearer chest X-ray."); return
        try:
            from ai_engine.preprocessing import preprocess_image
            from ai_engine.prediction import predict_image
            _,tensor=preprocess_image(self.image_path); disease,confidence,probabilities=predict_image(tensor)
        except Exception as e: QMessageBox.critical(self,"Prediction Error",str(e)); return
        self.predicted_disease=disease; self.predicted_confidence=float(confidence); self.predicted_probabilities=probabilities
        self.prediction_value.setText(disease); self.confidence_value.setText(f"{confidence:.2f}%")
        assessment="HIGH" if confidence>=90 and q>=80 else "MODERATE" if confidence>=70 and q>=70 else "LOW"
        self.assessment_value.setText(assessment); self.review_value.setText("CLINICAL REVIEW REQUIRED")
        for n,prob in zip(CLASSES,probabilities):
            pct=float(prob)*100; bar,val=self.probability_bars[n]; bar.setValue(int(round(pct*100))); val.setText(f"{pct:.2f}%")
        try:
            heatmap=generate_gradcam(self.image_path,tensor)
            if heatmap is None: raise RuntimeError("Grad-CAM returned no heatmap.")
            assets=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"generated_reports","assets"); os.makedirs(assets,exist_ok=True)
            self.heatmap_path=os.path.join(assets,"gradcam_latest.png")
            if not cv2.imwrite(self.heatmap_path,cv2.cvtColor(heatmap,cv2.COLOR_RGB2BGR)): raise RuntimeError("Unable to save Grad-CAM image.")
            h,w,ch=heatmap.shape; qi=QImage(heatmap.data,w,h,ch*w,QImage.Format_RGB888).copy(); self.set_image(self.heatmap_label,QPixmap.fromImage(qi))
        except Exception as e:
            self.heatmap_path=None; self.heatmap_label.clear(); self.heatmap_label.setText("Grad-CAM unavailable"); print("Grad-CAM Error:",e)
        print("="*60); print("AI DISEASE PREDICTION"); print("="*60); print("Disease    :",disease); print("Confidence :",confidence); print("Probabilities:",probabilities); print("X-ray Probability:",self.xray_probability); print("="*60)

    def generate_pdf(self):
        if self.predicted_disease is None: QMessageBox.warning(self,"No Prediction","Please upload an X-ray and predict the disease first."); return
        try:
            filename=create_pdf(patient_id=self.patient_id.text(),patient_name=self.patient_name.text(),age=self.patient_age.text(),gender=self.patient_gender.currentText(),disease=self.predicted_disease,confidence=self.predicted_confidence,probabilities=self.predicted_probabilities,quality=self.quality_data,image_path=self.image_path,heatmap_path=self.heatmap_path,xray_probability=self.xray_probability,xray_valid=self.xray_valid,xray_threshold=90.0)
        except Exception as e: QMessageBox.critical(self,"PDF Generation Error",f"Unable to generate the PDF report.\n\n{e}"); return
        QMessageBox.information(self,"PDF Generated",f"The diagnostic report was generated successfully.\n\nLocation:\n{filename}")

if __name__ == "__main__":
    app=QApplication([]); window=MainWindow(); window.show(); app.exec()



