"""Synthetic MEDS-format patient timeline for the eval example.

Returns a small DataFrame in the columns expected by `smb_utils.process_ehr_info`.
The values are fabricated and bear no relation to any real patient.
"""

import pandas as pd


def synthetic_patient_meds() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": ["patient_001"] * 12,
        "time": pd.to_datetime([
            "1985-03-15",
            "2024-01-05", "2024-01-05",
            "2024-01-10", "2024-01-10", "2024-01-10",
            "2024-01-15", "2024-01-15",
            "2024-01-20", "2024-01-20",
            "2024-01-25", "2024-01-25",
        ]),
        "table": [
            "person",
            "condition", "condition",
            "lab", "lab", "lab",
            "measurement", "measurement",
            "drug_exposure", "procedure",
            "lab", "lab",
        ],
        "code": [
            "Birth",
            "Essential Hypertension", "Type 2 Diabetes Mellitus",
            "Glucose", "HbA1c", "Creatinine",
            "Blood Pressure Systolic", "Blood Pressure Diastolic",
            "Metformin 500mg", "Annual Wellness Visit",
            "Glucose", "HbA1c",
        ],
        "numeric_value": [
            None,
            None, None,
            126.0, 7.2, 1.1,
            138.0, 88.0,
            None, None,
            118.0, 6.9,
        ],
        "text_value": [None] * 12,
        "unit": [
            None,
            None, None,
            "mg/dL", "%", "mg/dL",
            "mmHg", "mmHg",
            None, None,
            "mg/dL", "%",
        ],
    })
