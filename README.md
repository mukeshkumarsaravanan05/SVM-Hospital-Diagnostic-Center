# SVM Hospital Diagnostic Center
## AI-Assisted Chest X-ray Disease Prediction & Explainability

Research and educational AI application for chest X-ray disease classification using MobileNetV2, Grad-CAM, image-quality assessment, X-ray validation, probability analysis, and PDF reporting.

> Research / decision-support use only. This application is not a standalone medical diagnosis.

---
## Project Workflow

Chest X-ray Upload -> X-ray Validation -> Image Quality Assessment -> MobileNetV2 Disease Classification -> 5-Class Probability Distribution -> Grad-CAM Explainability -> AI Prediction Summary -> PDF Report Generation

---
## Disease Classes

The configured MobileNetV2 disease-classification pipeline contains five classes:

1. COVID
2. Lung Opacity
3. Normal
4. Pneumonia
5. Tuberculosis

The application displays the predicted class together with the probability distribution across all five classes.

---
## Model & Inference

The disease prediction pipeline uses MobileNetV2 with a five-class output layer.

**Model file:** `models/disease_mobilenetv2_clean_v2.pth`

**Input size:** 224 x 224 pixels

**Preprocessing:** Resize, ToTensor, and ImageNet normalization.

**Inference:** PyTorch model inference followed by Softmax probability calculation.

Core modules:
- `ai_engine/preprocessing.py`
- `ai_engine/prediction.py`
- `ai_engine/gradcam.py`

---
## Explainability & Reporting

### Grad-CAM

The application generates a Grad-CAM visualization from the disease-classification model to provide an interpretable visual representation of model-sensitive image regions.

### Image Quality Assessment

The application evaluates:
- Resolution
- Brightness
- Contrast
- Sharpness
- Overall quality score

### PDF Report

The application generates a structured AI-assisted PDF report containing patient information, X-ray validation, original radiograph, Grad-CAM visualization, image-quality assessment, predicted class, model confidence, probability distribution, and research/clinical disclaimer.

---
## Installation & Usage

### 1. Create Virtual Environment

```powershell
python -m venv .venv
```

### 2. Activate Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Verify Core Dependencies

```powershell
python -c "import PySide6, torch, torchvision, numpy, PIL, cv2, pytorch_grad_cam, reportlab, skimage, tensorflow; print('ALL FINAL DEPENDENCIES OK')"
```

### 5. Run the Application

```powershell
python -m ui.main_window_FINAL
```


---
## Technical Stack

- **Python** — Application development
- **PySide6** — Desktop GUI
- **PyTorch** — Deep-learning inference
- **Torchvision** — MobileNetV2 and image transforms
- **Pillow / OpenCV** — Image processing
- **Grad-CAM** — Model explainability
- **NumPy / scikit-image** — Image and numerical processing
- **ReportLab / fpdf2** — PDF report generation

---
## Project Purpose

This project demonstrates an end-to-end biomedical AI workflow combining chest X-ray image processing, deep-learning classification, explainable AI, image-quality assessment, and automated reporting.

The project is intended for research, educational, and portfolio purposes.

---
## Research & Clinical Disclaimer

This application is intended for research, educational, and AI decision-support purposes.

The predicted class and confidence values are outputs of the configured machine-learning pipeline. They should be interpreted together with the source radiograph, image quality, clinical history, and professional radiological assessment.

This application is **not a standalone medical diagnosis** and must not replace evaluation by a qualified clinician or radiologist.

Final clinical decisions must be made by an appropriately qualified healthcare professional.

---
## Author

**Mukesh Kumar**

Biomedical Engineering

EGS PILLAY Engineering College

---
## License

This project is intended for research and educational purposes.

