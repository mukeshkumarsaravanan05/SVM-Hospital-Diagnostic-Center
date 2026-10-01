# Validation Report — X-Ray Validator V1

## Internal Validation

Internal test samples: 242

- Accuracy: 94.63%
- Precision: 97.81%
- Sensitivity: 95.21%
- Specificity: 92.59%
- F1-score: 96.50%
- ROC-AUC: 98.35%

Locked threshold: 0.30

## External Validation

External samples: 1,000

Composition:

- 500 NIH ChestX-ray14
- 500 DermaMNIST

Results:

- ROC-AUC: 0.999732
- Accuracy: 50.0%
- Sensitivity: 100.0%
- Specificity: 0.0%
- Brier score: 0.376692
- ECE: 0.431463

## Explainability

Historical research Grad-CAM consistency:

Maximum difference: 2.7e-7

Application Grad-CAM consistency:

Tested application path difference: 0.000000000000

## Application Testing

Total tests: 6

Passed: 6

Failed: 0

Overall: PASS

Detailed report:

reports/application_test_results_v1.json

## Final Status

The software application is functioning as a research-stage demonstration.

The external model evaluation does not establish clinical safety or deployment readiness.
