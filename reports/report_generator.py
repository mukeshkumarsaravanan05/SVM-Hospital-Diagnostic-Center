"""
===========================================================
Professional Hospital Diagnostic Report Generator
Project : SVM Hospital Diagnostic Center
Author  : Mukesh Kumar
===========================================================
"""

from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "generated_reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# COLORS
# ==========================================================

NAVY = colors.HexColor("#0B2D4D")
BLUE = colors.HexColor("#1261A0")
LIGHT_BLUE = colors.HexColor("#EAF4FB")
PALE_BLUE = colors.HexColor("#F5FAFE")
BORDER = colors.HexColor("#C9D8E5")
TEXT = colors.HexColor("#253746")
MUTED = colors.HexColor("#617384")
WHITE = colors.white

GREEN = colors.HexColor("#177245")
AMBER = colors.HexColor("#B26A00")
RED = colors.HexColor("#B42318")


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def _safe(value):

    if value is None:
        return "Not provided"

    if str(value).strip() == "":
        return "Not provided"

    return str(value)


def _confidence(value):

    try:
        return float(value)

    except (TypeError, ValueError):

        return None


# ==========================================================
# SECTION HEADER
# ==========================================================

def _section(title, style):

    table = Table(
        [[
            Paragraph(
                title.upper(),
                style
            )
        ]],
        colWidths=[174 * mm],
        rowHeights=[8 * mm]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                NAVY
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                WHITE
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                4 * mm
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                2 * mm
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                2 * mm
            ),
        ])
    )

    return table


# ==========================================================
# INFORMATION TABLE
# ==========================================================

def _info_table(rows, body):

    data = []

    for label, value in rows:

        data.append([
            Paragraph(
                f"<b>{label}</b>",
                body
            ),
            Paragraph(
                _safe(value),
                body
            )
        ])

    table = Table(
        data,
        colWidths=[
            42 * mm,
            132 * mm
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                PALE_BLUE
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                BORDER
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.35,
                BORDER
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                2.2 * mm
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                2.2 * mm
            ),
        ])
    )

    return table


# ==========================================================
# IMAGE CARD
# ==========================================================

def _image_card(
    image_path,
    title,
    width=78 * mm,
    height=72 * mm,
):

    if not image_path:
        content = Paragraph(
            "Image not available",
            ParagraphStyle(
                "MissingImage",
                fontName="Helvetica",
                fontSize=8,
                textColor=MUTED,
                alignment=TA_CENTER
            )
        )

    else:

        path = Path(str(image_path))

        if path.exists():

            try:

                image = RLImage(
                    str(path)
                )

                image._restrictSize(
                    width,
                    height
                )

                content = image

            except Exception:

                content = Paragraph(
                    "Unable to load image",
                    ParagraphStyle(
                        "ImageError",
                        fontName="Helvetica",
                        fontSize=8,
                        textColor=MUTED,
                        alignment=TA_CENTER
                    )
                )

        else:

            content = Paragraph(
                "Image not available",
                ParagraphStyle(
                    "MissingImage2",
                    fontName="Helvetica",
                    fontSize=8,
                    textColor=MUTED,
                    alignment=TA_CENTER
                )
            )

    label_style = ParagraphStyle(
        "ImageTitle",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=NAVY,
        alignment=TA_CENTER,
        leading=11
    )

    table = Table(
        [
            [
                Paragraph(
                    title,
                    label_style
                )
            ],
            [
                content
            ]
        ],
        colWidths=[82 * mm],
        rowHeights=[8 * mm, 76 * mm]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                PALE_BLUE
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                BORDER
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                2 * mm
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                2 * mm
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                2 * mm
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                2 * mm
            ),
        ])
    )

    return table


# ==========================================================
# FOOTER
# ==========================================================

def _footer(canvas, doc):

    canvas.saveState()

    width, _ = A4

    canvas.setStrokeColor(
        BORDER
    )

    canvas.line(
        18 * mm,
        13 * mm,
        width - 18 * mm,
        13 * mm
    )

    canvas.setFont(
        "Helvetica",
        7.5
    )

    canvas.setFillColor(
        MUTED
    )

    canvas.drawString(
        18 * mm,
        8.5 * mm,
        "CONFIDENTIAL • AI-ASSISTED DECISION SUPPORT"
    )

    canvas.drawRightString(
        width - 18 * mm,
        8.5 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ==========================================================
# MAIN PDF FUNCTION
# ==========================================================

def generate_pdf(
    patient_id,
    patient_name,
    age,
    gender,
    disease,
    confidence,
    probabilities=None,
    quality=None,
    image_path=None,
    heatmap_path=None,
    xray_probability=None,
    xray_valid=None,
    xray_threshold=90.0,
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    now = datetime.now()

    timestamp = now.strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    file_timestamp = now.strftime(
        "%Y%m%d_%H%M%S"
    )

    name = _safe(
        patient_name
    )

    safe_name = "".join(
        "_"
        if c in '<>:"/\\|?*'
        else c
        for c in name
    ).strip()

    safe_name = (
        safe_name
        or "Patient"
    )

    filename = (
        REPORT_DIR
        / f"{safe_name}_{file_timestamp}.pdf"
    )


    # ======================================================
    # DOCUMENT
    # ======================================================

    doc = SimpleDocTemplate(
        str(filename),
        pagesize=A4,

        leftMargin=18 * mm,
        rightMargin=18 * mm,

        topMargin=15 * mm,
        bottomMargin=18 * mm,

        title="AI Chest X-ray Diagnostic Report",

        author="SVM Hospital Diagnostic Center"
    )


    # ======================================================
    # STYLES
    # ======================================================

    styles = getSampleStyleSheet()


    hospital = ParagraphStyle(
        "Hospital",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=WHITE
    )


    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor(
            "#DCEAF5"
        )
    )


    section = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=WHITE
    )


    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=8.8,
        leading=12,
        textColor=TEXT
    )


    small = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=7.8,
        leading=10,
        textColor=MUTED
    )


    center_label = ParagraphStyle(
        "CenterLabel",
        parent=small,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER
    )


    disease_style = ParagraphStyle(
        "Disease",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=NAVY,
        alignment=TA_CENTER
    )


    # ======================================================
    # STORY
    # ======================================================

    story = []


    # ======================================================
    # HEADER
    # ======================================================

    header = Table(
        [[
            [
                Paragraph(
                    "SVM HOSPITAL DIAGNOSTIC CENTER",
                    hospital
                ),

                Paragraph(
                    "AI-Assisted Chest X-ray Diagnostic Report",
                    subtitle
                ),
            ],

            Paragraph(
                "<b>CONFIDENTIAL</b><br/>"
                + now.strftime("%d %b %Y"),

                ParagraphStyle(
                    "HeaderRight",
                    parent=small,
                    fontName="Helvetica-Bold",
                    fontSize=8,
                    leading=12,
                    textColor=WHITE,
                    alignment=TA_CENTER
                )
            )
        ]],
        colWidths=[
            133 * mm,
            41 * mm
        ],
        rowHeights=[
            24 * mm
        ]
    )


    header.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                NAVY
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5 * mm
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4 * mm
            ),
        ])
    )


    story += [
        header,
        Spacer(1, 4 * mm)
    ]


    # ======================================================
    # REPORT META
    # ======================================================

    meta = Table(
        [[
            Paragraph(
                "<b>REPORT TYPE</b><br/>"
                "AI Chest X-ray Assessment",
                small
            ),

            Paragraph(
                f"<b>REPORT ID</b><br/>"
                f"{file_timestamp}",
                small
            ),

            Paragraph(
                f"<b>GENERATED</b><br/>"
                f"{timestamp}",
                small
            ),
        ]],
        colWidths=[
            58 * mm,
            58 * mm,
            58 * mm
        ]
    )


    meta.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                LIGHT_BLUE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.35,
                BORDER
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                2.5 * mm
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                2.5 * mm
            ),
        ])
    )


    story += [
        meta,
        Spacer(1, 4 * mm)
    ]


    # ======================================================
    # PATIENT INFORMATION
    # ======================================================

    story += [
        _section(
            "Patient Information",
            section
        ),

        Spacer(
            1,
            2 * mm
        ),

        _info_table(
            [
                ("Patient ID", patient_id),
                ("Patient Name", patient_name),
                ("Age", age),
                ("Gender", gender),
            ],
            body
        ),

        Spacer(
            1,
            4 * mm
        )
    ]
    # ======================================================
    # X-RAY VALIDATION
    # ======================================================

    story += [
        _section(
            "X-ray Validation",
            section
        ),

        Spacer(
            1,
            2 * mm
        )
    ]

    # Validation result

    if xray_valid is True:

        validation_result = "VALID CHEST X-RAY"

    elif xray_valid is False:

        validation_result = "NOT A VALID CHEST X-RAY"

    else:

        validation_result = "Not available"


    # Probability

    if xray_probability is None:

        probability_text = "Not available"

    else:

        probability_text = (
            f"{float(xray_probability):.2f}%"
        )


    validation_rows = [

        (
            "X-ray Probability",
            probability_text
        ),

        (
            "Validation Threshold",
            f"{float(xray_threshold):.0f}%"
        ),

        (
            "Validation Result",
            validation_result
        ),

    ]


    story += [

        _info_table(
            validation_rows,
            body
        ),

        Spacer(
            1,
            4 * mm
        )

    ]
    # ======================================================
    # SOURCE RADIOGRAPH
    # ======================================================

    story += [
        _section(
            "Source Radiograph & AI Explanation",
            section
        ),

        Spacer(
            1,
            2 * mm
        )
    ]


    original_card = _image_card(
        image_path,
        "ORIGINAL CHEST X-RAY"
    )


    heatmap_card = _image_card(
        heatmap_path,
        "GRAD-CAM AI ATTENTION MAP"
    )


    image_table = Table(
        [[
            original_card,
            heatmap_card
        ]],
        colWidths=[
            87 * mm,
            87 * mm
        ]
    )


    image_table.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                1 * mm
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                1 * mm
            ),
        ])
    )


    story += [
        image_table,
        Spacer(
            1,
            4 * mm
        )
    ]


    # ======================================================
    # IMAGE QUALITY
    # ======================================================

    story += [
        _section(
            "Image Quality Assessment",
            section
        ),

        Spacer(
            1,
            2 * mm
        )
    ]


    if isinstance(
        quality,
        dict
    ):

        qrows = [
            (
                "Resolution",
                quality.get(
                    "resolution",
                    "Not available"
                )
            ),

            (
                "Brightness",
                quality.get(
                    "brightness",
                    "Not available"
                )
            ),

            (
                "Contrast",
                quality.get(
                    "contrast",
                    "Not available"
                )
            ),

            (
                "Sharpness",
                quality.get(
                    "sharpness",
                    "Not available"
                )
            ),

            (
                "Overall Quality Score",
                quality.get(
                    "score",
                    "Not available"
                )
            ),
        ]

    else:

        qrows = [
            (
                "Assessment",
                "Quality metrics were not attached to this report."
            )
        ]


    story += [
        _info_table(
            qrows,
            body
        ),

        Spacer(
            1,
            4 * mm
        )
    ]


    # ======================================================
    # AI DIAGNOSTIC SUMMARY
    # ======================================================



    conf = _confidence(
        confidence
    )


    if conf is None:

        conf_text = _safe(
            confidence
        )

        conf_color = BLUE

    else:

        conf_text = (
            f"{conf:.2f}%"
        )

        if conf >= 90:

            conf_color = GREEN

        elif conf >= 70:

            conf_color = AMBER

        else:

            conf_color = BLUE


    result = Table(
        [[
            Paragraph(
                "AI PREDICTED CONDITION",
                center_label
            ),

            Paragraph(
                "MODEL CONFIDENCE",
                center_label
            )
        ], [
            Paragraph(
                _safe(disease),
                disease_style
            ),

            Paragraph(
                f"<font color='{conf_color.hexval()}'>"
                f"<b>{conf_text}</b>"
                f"</font>",

                ParagraphStyle(
                    "ConfBig",
                    parent=disease_style,
                    fontSize=17
                )
            )
        ]],

        colWidths=[
            87 * mm,
            87 * mm
        ],

        rowHeights=[
            8 * mm,
            17 * mm
        ]
    )


    result.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                PALE_BLUE
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, 1),
                WHITE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
        ])
    )


    story += [
    KeepTogether([
        _section(
            "AI Diagnostic Summary",
            section
        ),

        Spacer(
            1,
            2 * mm
        ),

        result,

        Spacer(
            1,
            4 * mm
        )
    ])
]


    # ======================================================
    # AI ASSESSMENT STATUS
    # ======================================================

    quality_score = 0

    if isinstance(quality, dict):

        try:
            quality_score = float(
                quality.get("score", 0)
            )
        except (TypeError, ValueError):
            quality_score = 0

    try:
        confidence_score = float(
            _confidence(confidence)
            or 0
        )
    except (TypeError, ValueError):
        confidence_score = 0

    if (
        confidence_score >= 90
        and
        quality_score >= 80
    ):

        assessment_status = "HIGH"

    elif (
        confidence_score >= 70
        and
        quality_score >= 70
    ):

        assessment_status = "MODERATE"

    else:

        assessment_status = "LOW"

    review_status = "REQUIRED"


    assessment_color = (
        GREEN
        if assessment_status == "HIGH"
        else AMBER
        if assessment_status == "MODERATE"
        else BLUE
    )


    assessment_table = Table(
        [
            [
                Paragraph(
                    "ASSESSMENT LEVEL",
                    center_label
                ),

                Paragraph(
                    "CLINICAL REVIEW",
                    center_label
                )
            ],
            [
                Paragraph(
                    f"<font color='{assessment_color.hexval()}'>"
                    f"<b>{assessment_status}</b>"
                    f"</font>",
                    ParagraphStyle(
                        "AssessmentBig",
                        parent=disease_style,
                        fontSize=15
                    )
                ),

                Paragraph(
                    f"<b>{review_status}</b>",
                    ParagraphStyle(
                        "ReviewBig",
                        parent=disease_style,
                        fontSize=15
                    )
                )
            ],
            [
                Paragraph(
                    "MODEL CONFIDENCE",
                    center_label
                ),

                Paragraph(
                    "IMAGE QUALITY",
                    center_label
                )
            ],
            [
                Paragraph(
                    f"<b>{confidence_score:.2f}%</b>",
                    small
                ),

                Paragraph(
                    f"<b>{quality_score:.0f}/100</b>",
                    small
                )
            ]
        ],
        colWidths=[
            87 * mm,
            87 * mm
        ],
        rowHeights=[
            8 * mm,
            15 * mm,
            8 * mm,
            10 * mm
        ]
    )


    assessment_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                PALE_BLUE
            ),

            (
                "BACKGROUND",
                (0, 2),
                (-1, 2),
                PALE_BLUE
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, 1),
                WHITE
            ),

            (
                "BACKGROUND",
                (0, 3),
                (-1, 3),
                WHITE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
        ])
    )


    story += [
        KeepTogether([
            _section(
                "AI Assessment Status",
                section
            ),

            Spacer(
                1,
                2 * mm
            ),

            assessment_table,

            Spacer(
                1,
                4 * mm
            )
        ])
    ]


    # ======================================================
    # AI PROBABILITY DISTRIBUTION
    # ======================================================

    if probabilities is not None:

        classes = [
            "COVID",
            "Lung Opacity",
            "Normal",
            "Pneumonia",
            "Tuberculosis"
        ]

        probability_rows = [
            [
                Paragraph(
                    "DISEASE CLASS",
                    center_label
                ),
                Paragraph(
                    "PROBABILITY",
                    center_label
                )
            ]
        ]

        try:

            for class_name, probability in zip(
                classes,
                probabilities
            ):

                probability_rows.append(
                    [
                        Paragraph(
                            _safe(class_name),
                            small
                        ),
                        Paragraph(
                            f"{float(probability) * 100:.2f}%",
                            small
                        )
                    ]
                )

            probability_table = Table(
                probability_rows,
                colWidths=[
                    87 * mm,
                    87 * mm
                ]
            )

            probability_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        PALE_BLUE
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        WHITE
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.8,
                        BORDER
                    ),
                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        BORDER
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        4
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        4
                    ),
                ])
            )

            story += [
                KeepTogether([
                    _section(
                        "AI Probability Distribution",
                        section
                    ),

                    Spacer(
                        1,
                        2 * mm
                    ),

                    probability_table,

                    Spacer(
                        1,
                        4 * mm
                    )
                ])
            ]

        except Exception as e:

            print(
                "Probability table error:",
                e
            )


    # ======================================================
    # CLINICAL INTERPRETATION
    # ======================================================

    story += [
        _section(
            "Clinical Interpretation Notice",
            section
        ),

        Spacer(
            1,
            2 * mm
        )
    ]


    notice = (
        "The reported condition and confidence are outputs of the "
        "configured AI prediction pipeline. They should be interpreted "
        "together with the source radiograph, image quality, clinical "
        "history and professional radiological assessment."
    )


    notice_table = Table(
        [[
            Paragraph(
                notice,
                body
            )
        ]],
        colWidths=[
            174 * mm
        ]
    )


    notice_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                PALE_BLUE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                BORDER
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                4 * mm
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4 * mm
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),
        ])
    )


    story += [
        notice_table,
        Spacer(
            1,
            4 * mm
        )
    ]


    # ======================================================
    # DISCLAIMER
    # ======================================================

    story += [
        _section(
            "Clinical Disclaimer",
            section
        ),

        Spacer(
            1,
            2 * mm
        )
    ]


    disclaimer = (
        "<b>For research and decision-support use.</b> "
        "This AI-generated assessment is not a standalone medical "
        "diagnosis and must not replace evaluation by a qualified "
        "clinician or radiologist. Final clinical decisions must "
        "be made by an appropriately qualified professional."
    )


    disclaimer_table = Table(
        [[
            Paragraph(
                disclaimer,
                body
            )
        ]],
        colWidths=[
            174 * mm
        ]
    )


    disclaimer_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#FFF8E8")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#E5C77A")
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                4 * mm
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4 * mm
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),
        ])
    )


    story += [
        disclaimer_table,
        Spacer(
            1,
            5 * mm
        )
    ]


    # ======================================================
    # SIGN-OFF
    # ======================================================

    signoff = Table(
        [[
            Paragraph(
                "<b>AI SYSTEM STATUS</b><br/>"
                "Report generated successfully",
                small
            ),

            Paragraph(
                "<b>CLINICAL REVIEW</b><br/>"
                "Name / Signature: ____________________",
                small
            ),
        ]],

        colWidths=[
            87 * mm,
            87 * mm
        ],

        rowHeights=[
            18 * mm
        ]
    )


    signoff.setStyle(
        TableStyle([
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                BORDER
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                3 * mm
            ),
        ])
    )


    story.append(
        signoff
    )


    # ======================================================
    # BUILD PDF
    # ======================================================

    doc.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer
    )


    return str(
        filename.resolve()
    )