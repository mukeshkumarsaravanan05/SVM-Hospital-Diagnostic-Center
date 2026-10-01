# Model Card — X-Ray Validator V1

## Model

EfficientNet-B0-based binary X-ray image validator.

## Intended Use

Research-stage validation of whether an uploaded image is consistent with an X-ray modality.

## Not Intended For

- Disease diagnosis
- Clinical decision making
- Autonomous clinical deployment
- Calibrated clinical probability estimation

## Input

224 × 224 RGB image.

## Output

X-ray probability and binary prediction using the locked threshold of 0.30.

## External Evaluation

- 500 NIH ChestX-ray14 X-ray images
- 500 DermaMNIST medical non-X-ray images
- Total: 1,000 images

## External Results

- ROC-AUC: 0.999732
- Accuracy at locked threshold: 50.0%
- Sensitivity: 100.0%
- Specificity: 0.0%
- Brier score: 0.376692
- ECE: 0.431463

## Known Limitations

The external evaluation demonstrated:

- threshold-transfer failure
- substantial probability miscalibration
- strong false-positive behavior on the evaluated non-X-ray medical domain

## Explainability

Grad-CAM is provided for interpretability.

Application consistency testing produced a zero difference for the tested V1 application path.

## Version

V1

## Status

FROZEN RESEARCH BASELINE
