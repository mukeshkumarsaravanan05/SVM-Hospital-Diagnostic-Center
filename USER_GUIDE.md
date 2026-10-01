# User Guide — AI X-Ray Image Validator V1

## Start

Run:

.\run_app.ps1

Open:

http://localhost:8501/

## Supported Images

- PNG
- JPG
- JPEG

## Validation

Upload an image and click **Validate Image**.

The application displays:

- Prediction
- X-ray probability
- Locked threshold
- Model version
- Grad-CAM visualization

## Important

This is a research-stage modality-validation system.

It is not a medical diagnosis and its probability output must not be interpreted as calibrated clinical confidence.

## Invalid Inputs

Unsupported file types should be rejected.

Corrupted images should be rejected rather than producing a prediction.

## Grad-CAM

Grad-CAM is an AI interpretability visualization. It does not establish anatomical relevance or provide a medical diagnosis.
