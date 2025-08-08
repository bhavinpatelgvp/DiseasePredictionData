"""
Configuration file for the Disease Prediction CRUD Application
"""

# Database configuration
DATABASE_PATH = "medical_records.db"

# Model configuration
MODEL_PATH = "disease_model.joblib"
SCALER_PATH = "scaler.joblib"

# Blood parameter normal ranges (for reference)
NORMAL_RANGES = {
    'hb': {'min': 12.0, 'max': 18.0, 'unit': 'g/dL'},
    'totalrbc': {'min': 4.0, 'max': 6.0, 'unit': 'million/μL'},
    'pcv': {'min': 35.0, 'max': 50.0, 'unit': '%'},
    'mcv': {'min': 80.0, 'max': 100.0, 'unit': 'fL'},
    'mch': {'min': 27.0, 'max': 33.0, 'unit': 'pg'},
    'mchc': {'min': 32.0, 'max': 36.0, 'unit': 'g/dL'},
    'rdw': {'min': 11.0, 'max': 15.0, 'unit': '%'},
    'wbc': {'min': 4.0, 'max': 11.0, 'unit': 'thousand/μL'},
    'neutrophi': {'min': 50.0, 'max': 70.0, 'unit': '%'},
    'lymph': {'min': 20.0, 'max': 40.0, 'unit': '%'},
    'eosin': {'min': 1.0, 'max': 4.0, 'unit': '%'},
    'mono': {'min': 2.0, 'max': 8.0, 'unit': '%'},
    'plt': {'min': 150.0, 'max': 450.0, 'unit': 'thousand/μL'},
    'mpv': {'min': 7.0, 'max': 11.0, 'unit': 'fL'}
}

# Disease categories
DISEASES = [
    'Healthy',
    'Anemia',
    'Leukemia', 
    'Thrombocytopenia',
    'Bacterial Infection',
    'Leukopenia'
]

# Blood groups
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Unknown']

# Application settings
APP_TITLE = "Disease Prediction CRUD Application"
APP_ICON = "🏥"
PAGE_LAYOUT = "wide"

# Export settings
EXPORT_DATE_FORMAT = "%Y%m%d_%H%M%S"
