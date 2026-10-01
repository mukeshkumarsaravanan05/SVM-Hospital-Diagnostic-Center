"""
===========================================================
Prediction Module - V2
Project : SVM Hospital Diagnostic Center
Model   : MobileNetV2 V2
Input   : 224 x 224 + ImageNet normalization
===========================================================
"""

import os

import torch
import torch.nn as nn

from torchvision.models import mobilenet_v2


# ==========================================================
# CLASSES
# ==========================================================

CLASSES = [
    "COVID",
    "Lung Opacity",
    "Normal",
    "Pneumonia",
    "Tuberculosis"
]


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "disease_mobilenetv2_clean_v2.pth"
)


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    model = mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(CLASSES)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


model = load_model()


# ==========================================================
# PREDICTION
# ==========================================================

def predict_image(input_tensor):

    input_tensor = input_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    probs = probabilities.cpu().numpy()

    predicted_index = int(
        probs.argmax()
    )

    predicted_class = CLASSES[
        predicted_index
    ]

    confidence = (
        float(
            probs[predicted_index]
        ) * 100
    )

    return (
        predicted_class,
        confidence,
        probs
    )


# ==========================================================
# MODEL ACCESS
# ==========================================================

def get_model():

    return model