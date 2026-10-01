"""
===========================================================
Grad-CAM Module - V2
Project : SVM Hospital Diagnostic Center
Model   : MobileNetV2
Input   : 224 x 224
===========================================================
"""

import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from ai_engine.prediction import get_model


# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_SIZE = 224


# ==========================================================
# GENERATE GRAD-CAM
# ==========================================================

def generate_gradcam(image_source, input_tensor):

    """
    image_source:
        Image file path OR PIL Image

    input_tensor:
        Preprocessed tensor with shape
        (1, 3, 224, 224)

    Returns:
        Grad-CAM visualization
        numpy.ndarray
        Shape: (224, 224, 3)
    """

    # ------------------------------------------------------
    # Load V2 model
    # ------------------------------------------------------

    model = get_model()

    model.eval()

    # ------------------------------------------------------
    # Verify input
    # ------------------------------------------------------

    if input_tensor.ndim != 4:

        raise ValueError(
            f"Expected 4D tensor, got {input_tensor.shape}"
        )

    if input_tensor.shape[-2:] != (
        IMG_SIZE,
        IMG_SIZE
    ):

        raise ValueError(
            "Grad-CAM input size mismatch. "
            f"Expected ({IMG_SIZE},{IMG_SIZE}), "
            f"got {tuple(input_tensor.shape[-2:])}"
        )

    # ------------------------------------------------------
    # MobileNetV2 final convolutional layer
    # ------------------------------------------------------

    target_layers = [
        model.features[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    output = model(
        input_tensor
    )

    predicted_class = output.argmax(
        dim=1
    ).item()

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    # ------------------------------------------------------
    # Generate Grad-CAM
    # ------------------------------------------------------

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # ------------------------------------------------------
    # Read original image
    # ------------------------------------------------------

    if isinstance(
        image_source,
        Image.Image
    ):

        image = image_source.convert(
            "RGB"
        )

    else:

        image = Image.open(
            image_source
        ).convert(
            "RGB"
        )

    # ------------------------------------------------------
    # Resize visualization image
    # ------------------------------------------------------

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # ------------------------------------------------------
    # Convert to [0,1]
    # ------------------------------------------------------

    rgb_image = np.asarray(
        image
    ).astype(
        np.float32
    ) / 255.0

    # ------------------------------------------------------
    # Generate visualization
    # ------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization