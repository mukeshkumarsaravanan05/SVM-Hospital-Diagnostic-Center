"""
===========================================================
X-Ray Image Validator
Project : SVM Hospital Diagnostic Center
===========================================================
"""

import os
import cv2
import numpy as np


# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================================
# MODEL PATH
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xray_validator.keras"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_SIZE = 224
THRESHOLD = 0.90


# ==========================================================
# LAZY MODEL LOADING
# ==========================================================

validator = None


def get_validator():

    global validator

    if validator is None:

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"X-ray validator model not found:\n{MODEL_PATH}"
            )

        print()
        print("=" * 60)
        print("Loading X-ray Validator Model...")
        print("=" * 60)

        # Import TensorFlow only when the validator is actually needed
        from tensorflow.keras.models import load_model

        validator = load_model(
            MODEL_PATH
        )

        print("X-ray Validator Model Loaded")
        print("=" * 60)

    return validator


# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )


    img = cv2.imread(
        image_path
    )


    if img is None:

        raise ValueError(
            f"Unable to read image:\n{image_path}"
        )


    # BGR → RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    # Resize
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )


    # Keep pixel values in 0–255.
    # EfficientNet preprocessing is already
    # handled inside the trained Keras model.

    img = img.astype(
        "float32"
    )


    # Add batch dimension

    img = np.expand_dims(
        img,
        axis=0
    )


    return img


# ==========================================================
# GET RAW X-RAY PROBABILITY
# ==========================================================

def _predict_probability(image_path):

    validator_model = get_validator()

    img = preprocess_image(
        image_path
    )


    prediction = validator_model.predict(
        img,
        verbose=0
    )


    probability = float(
        prediction[0][0]
    )


    return probability

# ==========================================================
# PREDICT X-RAY PROBABILITY
# ==========================================================

def _predict_probability(image_path):

    validator_model = get_validator()

    img = preprocess_image(
        image_path
    )

    prediction = validator_model.predict(
        img,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )

    return probability


# ==========================================================
# CHECK WHETHER IMAGE IS X-RAY
# ==========================================================

def is_xray(image_path):

    probability = _predict_probability(
        image_path
    )

    return probability >= THRESHOLD


# ==========================================================
# GET X-RAY PROBABILITY
# ==========================================================

def get_xray_probability(image_path):

    probability = _predict_probability(
        image_path
    )

    return probability * 100
# ==========================================================
# VALIDATE X-RAY AND GET PROBABILITY IN ONE INFERENCE
# ==========================================================

def validate_xray(image_path):

    probability = _predict_probability(
        image_path
    )

    probability_percent = (
        probability * 100
    )

    valid = (
        probability >= THRESHOLD
    )

    return valid, probability_percent
