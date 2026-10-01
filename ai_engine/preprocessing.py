"""
===========================================================
Image Preprocessing Module - V2
Project : SVM Hospital Diagnostic Center
Model   : MobileNetV2
Input   : 224 x 224
===========================================================
"""

from PIL import Image
from torchvision import transforms


# ==========================================================
# V2 MODEL PREPROCESSING
# ==========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])


# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image_file):

    original_image = Image.open(
        image_file
    ).convert("RGB")

    input_tensor = transform(
        original_image
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )

    return (
        original_image,
        input_tensor
    )