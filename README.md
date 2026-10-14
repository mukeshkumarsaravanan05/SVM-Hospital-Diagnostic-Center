# 🏥 SVM Hospital Diagnostic Center
### AI-Assisted Chest X-ray Classification & Explainability

An AI-powered biomedical research application for experimental chest X-ray classification, image validation, image-quality assessment, Grad-CAM visualization, and PDF report generation.

<p align="center">
  <a href="https://svm-hospital-diagnostic-center.streamlit.app/">
    <strong>🚀 Launch Live Web App</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/mukeshkumarsaravanan05/SVM-Hospital-Diagnostic-Center">
    <strong>💻 GitHub Repository</strong>
  </a>
</p>

> **Medical disclaimer:** This project is intended for research and educational demonstration only. It is not a medical device or a standalone diagnostic system. Predictions are experimental model outputs, not confirmed diagnoses. Always consult qualified healthcare professionals for clinical assessment.

---

## 📌 Project Overview

SVM Hospital Diagnostic Center is a biomedical AI project that combines deep learning and explainable AI techniques to demonstrate a chest X-ray analysis workflow.

The application provides a web interface where users can upload an X-ray image, review image validation and quality results, inspect predicted class probabilities, view a Grad-CAM visualization, and generate a PDF research report.

The project is designed to explore how AI-based image classification and visual explanations can support biomedical engineering research.

## ✨ Key Features

- **Chest X-ray classification:** Predicts one of five configured classes.
- **Image validation:** Checks whether an uploaded image passes the application's X-ray validation process.
- **Image-quality assessment:** Displays quality indicators to help identify images that may require review.
- **Explainable AI:** Generates a Grad-CAM visualization highlighting image regions that influence the model's output.
- **Probability distribution:** Displays the model's output probabilities for all five classes.
- **PDF report generation:** Produces a downloadable report containing analysis results and supporting visualizations.
- **Interactive web interface:** Provides an accessible dashboard through Streamlit.
- **Desktop application:** Includes a separate PySide6 desktop interface.

## 🩻 Supported Prediction Classes

1. COVID
2. Lung Opacity
3. Normal
4. Pneumonia
5. Tuberculosis

These are the classes configured in the application. Their inclusion does not establish that the model can reliably diagnose these conditions in clinical practice.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| Image classification model | MobileNetV2 |
| Explainable AI | Grad-CAM |
| Web application | Streamlit |
| Desktop application | PySide6 |
| Image processing | Pillow, OpenCV |
| Numerical processing | NumPy |
| Image-quality analysis | Image-processing utilities |
| Report generation | PDF reporting libraries |

## 🔬 AI Analysis Workflow

```text
Chest X-ray Upload
        |
        v
Image Validation
        |
        v
Image Quality Assessment
        |
        v
Image Preprocessing
(224 × 224, ImageNet normalization)
        |
        v
MobileNetV2 Inference
        |
        v
Five-Class Probability Distribution
        |
        +----------------------+
        |                      |
        v                      v
  Grad-CAM             Results Summary
        |                      |
        +-----------+----------+
                    |
                    v
             PDF Report
```

The model output and visual explanations should be interpreted cautiously. A Grad-CAM heatmap is a visualization of model behavior, not proof that the highlighted region represents a disease.

## 📊 Understanding the Results

The dashboard displays:

- Predicted class
- Model output probability
- Probability distribution across the five classes
- X-ray validation result
- Image-quality indicators
- Grad-CAM visualization
- Downloadable PDF report

**Important:** A high model probability is not equivalent to clinical accuracy, diagnostic certainty, or a calibrated estimate of disease risk. The displayed results must not be used alone to make medical decisions.

## 🚀 Live Demonstration

Try the deployed web application:

**[Open SVM Hospital Diagnostic Center](https://svm-hospital-diagnostic-center.streamlit.app/)**

The public demo is intended for educational exploration. Use only authorized, de-identified sample images and fictional examination details. Do not upload identifiable patient images or personal health information.

## 💻 Run the Project Locally

### Prerequisites

- Python installed on your computer
- Git
- The required project files and model assets

### 1. Clone the repository

```bash
git clone https://github.com/mukeshkumarsaravanan05/SVM-Hospital-Diagnostic-Center.git
cd SVM-Hospital-Diagnostic-Center
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

For the Streamlit web application:

```bash
pip install -r requirements.txt
```

For the desktop application, use the separate desktop dependency file:

```bash
pip install -r requirements-desktop.txt
```

Install the dependency set appropriate for the application you want to run. Do not combine the two files unnecessarily.

### 4. Launch the web application

```bash
streamlit run streamlit_app.py
```

Open the local address shown in the terminal, usually `http://localhost:8501`.

### 5. Launch the desktop application

```bash
python -m ui.main_window_FINAL
```

The desktop application has a separate interface from the Streamlit web application.

> **Setup note:** Successful execution requires the model files, application modules, and compatible dependencies to be present. Refer to the repository's requirements files and project documentation if setup fails.

## 📂 Project Components

The repository includes components for:

- `streamlit_app.py` — Streamlit web application entry point
- `ai_engine/` — AI inference, preprocessing, validation, image quality, and Grad-CAM modules
- `models/` — Model and image-validator assets
- `ui/` — Desktop application interface
- `reports/` — Report-related project files
- `requirements.txt` — Web application dependencies
- `requirements-desktop.txt` — Desktop application dependencies
- `packages.txt` — System packages used for web deployment

The exact contents may evolve as the project is maintained.

## ⚠️ Limitations & Responsible Use

- This project is a research and educational demonstration, not a clinically validated medical product.
- Model probabilities must not be interpreted as confirmed diagnoses.
- The application has not been established here as meeting clinical performance, regulatory, or medical-device requirements.
- Image validation and quality scores do not guarantee that an image is diagnostically suitable.
- Grad-CAM visualizations may be incomplete or misleading and do not establish causation.
- Performance may vary with image quality, acquisition equipment, patient populations, and data distribution.
- Do not use this application to delay or replace professional medical assessment.
- Use appropriately licensed, authorized, and de-identified data for demonstrations and testing.

## 🔮 Future Improvements

- Evaluate the model on an independent, appropriately curated test dataset.
- Report sensitivity, specificity, precision, recall, F1-score, and per-class confusion matrices.
- Assess probability calibration and generalization across datasets.
- Improve image-quality validation and explainability evaluation.
- Add automated software testing and deployment checks.
- Improve PDF report formatting and accessibility.
- Conduct appropriate privacy, security, and regulatory reviews before any proposed clinical use.

## 👨‍💻 Developer

**Mukesh Kumar S**  
Biomedical Engineering | AI in Healthcare | Deep Learning | Explainable AI

- **GitHub:** [mukeshkumarsaravanan05](https://github.com/mukeshkumarsaravanan05)
- **LinkedIn:** [Mukesh Kumar Saravanan](https://www.linkedin.com/in/mukesh-kumar-saravanan05/)
- **Live Project:** [SVM Hospital Diagnostic Center](https://svm-hospital-diagnostic-center.streamlit.app/)

## 📄 License

Check the repository for a license file before redistributing or reusing the code, model weights, or associated datasets. If no license is provided, do not assume that reuse is automatically permitted.

---

*Built as a biomedical engineering research and educational project exploring AI-assisted medical image analysis and explainability.*
