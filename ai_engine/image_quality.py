import cv2
import numpy as np


def assess_image_quality(image_path):
    """
    Assess the quality of a Chest X-ray image.

    Returns:
        dict:
        {
            "resolution": "...",
            "brightness": "...",
            "contrast": "...",
            "sharpness": "...",
            "score": 92
        }
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to load image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # -----------------------------
    # Resolution
    # -----------------------------
    pixels = height * width

    if pixels >= 1024 * 1024:
        resolution = "Excellent"
        res_score = 25

    elif pixels >= 512 * 512:
        resolution = "Good"
        res_score = 20

    else:
        resolution = "Poor"
        res_score = 10

    # -----------------------------
    # Brightness
    # -----------------------------
    brightness = np.mean(gray)

    if 80 <= brightness <= 180:
        brightness_status = "Good"
        bright_score = 25

    elif 60 <= brightness <= 220:
        brightness_status = "Acceptable"
        bright_score = 18

    else:
        brightness_status = "Poor"
        bright_score = 8

    # -----------------------------
    # Contrast
    # -----------------------------
    contrast = np.std(gray)

    if contrast > 50:
        contrast_status = "Excellent"
        contrast_score = 25

    elif contrast > 30:
        contrast_status = "Good"
        contrast_score = 18

    else:
        contrast_status = "Poor"
        contrast_score = 8

    # -----------------------------
    # Sharpness
    # -----------------------------
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    if sharpness > 150:
        sharpness_status = "Excellent"
        sharp_score = 25

    elif sharpness > 80:
        sharpness_status = "Good"
        sharp_score = 18

    else:
        sharpness_status = "Blurry"
        sharp_score = 8

    # -----------------------------
    # Overall Score
    # -----------------------------
    total_score = (
        res_score
        + bright_score
        + contrast_score
        + sharp_score
    )

    # -----------------------------
    # Quality Risk Assessment
    # -----------------------------
    quality_issues = []

    if resolution == "Poor":
        quality_issues.append(
            "Low resolution"
        )

    if brightness_status == "Poor":
        quality_issues.append(
            "Poor brightness"
        )

    if contrast_status == "Poor":
        quality_issues.append(
            "Poor contrast"
        )

    if sharpness_status == "Blurry":
        quality_issues.append(
            "Blurry image"
        )

    if total_score >= 80:
        quality_risk = "LOW"

    elif total_score >= 70:
        quality_risk = "MODERATE"

    else:
        quality_risk = "HIGH"

    if quality_issues:
        review_reason = ", ".join(
            quality_issues
        )

    else:
        review_reason = (
            "No major image-quality issue detected"
        )

    return {
        "resolution": resolution,
        "brightness": brightness_status,
        "contrast": contrast_status,
        "sharpness": sharpness_status,
        "score": total_score,
        "risk_level": quality_risk,
        "review_reason": review_reason
    }
