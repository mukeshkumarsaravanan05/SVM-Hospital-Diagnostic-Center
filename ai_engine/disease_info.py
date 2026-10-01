"""
===========================================================
Disease Information Module
Project : SVM Hospital Diagnostic Center
===========================================================
"""

DISEASE_INFO = {

    "COVID": {
        "description": "COVID-19 is a viral respiratory disease caused by the SARS-CoV-2 virus.",
        "symptoms": [
            "Fever",
            "Dry cough",
            "Shortness of breath",
            "Loss of taste or smell",
            "Fatigue"
        ],
        "causes": [
            "SARS-CoV-2 viral infection",
            "Close contact with infected individuals"
        ],
        "precautions": [
            "Wear a mask",
            "Wash hands frequently",
            "Maintain physical distance",
            "Seek medical attention if symptoms worsen"
        ],
        "specialist": "Pulmonologist / Infectious Disease Specialist",
        "severity": "Moderate to Severe"
    },

    "Lung Opacity": {
        "description": "Lung opacity is an abnormal white area seen on a chest X-ray indicating inflammation, infection, or fluid accumulation.",
        "symptoms": [
            "Chest pain",
            "Persistent cough",
            "Difficulty breathing",
            "Fever"
        ],
        "causes": [
            "Pneumonia",
            "Pulmonary edema",
            "Lung inflammation"
        ],
        "precautions": [
            "Consult a pulmonologist",
            "Follow prescribed medications",
            "Avoid smoking"
        ],
        "specialist": "Pulmonologist",
        "severity": "Depends on the underlying condition"
    },

    "Normal": {
        "description": "The chest X-ray appears normal with no significant abnormalities detected by the AI model.",
        "symptoms": [
            "No abnormal symptoms detected from imaging"
        ],
        "causes": [
            "Healthy lungs"
        ],
        "precautions": [
            "Maintain a healthy lifestyle",
            "Avoid smoking",
            "Attend routine medical checkups if needed"
        ],
        "specialist": "General Physician",
        "severity": "None"
    },

    "Pneumonia": {
        "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs.",
        "symptoms": [
            "Fever",
            "Productive cough",
            "Chest pain",
            "Difficulty breathing",
            "Fatigue"
        ],
        "causes": [
            "Bacterial infection",
            "Viral infection",
            "Fungal infection"
        ],
        "precautions": [
            "Take prescribed antibiotics or antivirals",
            "Drink plenty of fluids",
            "Get adequate rest",
            "Seek medical care promptly"
        ],
        "specialist": "Pulmonologist",
        "severity": "Moderate to Severe"
    },

    "Tuberculosis": {
        "description": "Tuberculosis (TB) is a bacterial infection caused by Mycobacterium tuberculosis, primarily affecting the lungs.",
        "symptoms": [
            "Persistent cough",
            "Weight loss",
            "Night sweats",
            "Fever",
            "Blood in sputum"
        ],
        "causes": [
            "Mycobacterium tuberculosis infection"
        ],
        "precautions": [
            "Complete the full TB treatment course",
            "Cover mouth while coughing",
            "Improve ventilation",
            "Follow medical advice"
        ],
        "specialist": "Pulmonologist / Infectious Disease Specialist",
        "severity": "Serious but treatable"
    }
}


def get_disease_info(disease_name):
    """
    Returns the information for the predicted disease.
    """

    return DISEASE_INFO.get(
        disease_name,
        {
            "description": "Information not available.",
            "symptoms": [],
            "causes": [],
            "precautions": [],
            "specialist": "Unknown",
            "severity": "Unknown"
        }
    )