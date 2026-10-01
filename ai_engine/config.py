"""
===========================================================
Prediction Module
Project : SVM Hospital Diagnostic Center
Author  : Mukesh Kumar
===========================================================
"""

import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1

# ----------------------------------------------------------
# Disease Classes
# ----------------------------------------------------------
CLASSES = [
    "COVID",
    "Lung Opacity",
    "Normal",
    "Pneumonia",
    "Tuberculosis"
]

# ----------------------------------------------------------
# Model Path
# ----------------------------------------------------------
MODEL_PATH = os.path.join("models", "lung_model.pth")

# ----------------------------------------------------------
# Select Device
# ----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------
# Load AI Model
# ----------------------------------------------------------
def load_model():

    # Load EfficientNet-B1
    model = efficientnet_b1(weights=None)

    # Change output layer
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(in_features, 5)

    # Load trained weights
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    return model


# ----------------------------------------------------------
# Lazy Model Loading
# ----------------------------------------------------------

model = None


def get_model():

    global model

    if model is None:

        print()
        print("=" * 60)
        print("Loading Disease Classification Model...")
        print("=" * 60)

        model = load_model()

        print("Disease Classification Model Loaded")
        print("=" * 60)

    return model


# ----------------------------------------------------------
# Prediction Function
# ----------------------------------------------------------

def predict_image(input_tensor):

    model = get_model()

    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    probs = probabilities.cpu().numpy()

    predicted_index = probs.argmax()

    predicted_class = CLASSES[predicted_index]

    confidence = probs[predicted_index] * 100

    return predicted_class, confidence, probs