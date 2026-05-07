"""End-to-end eval: synthetic MEDS DataFrame -> smb_utils -> model -> embedding."""

import sys
from pathlib import Path

import pandas as pd

# Make the repo root importable when running as `python examples/run_example.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import DEFAULT_MODEL_ID, embed_patient, load_model  # noqa: E402
from smb_utils import process_ehr_info  # noqa: E402

from synthetic_meds import synthetic_patient_meds  # noqa: E402


def main() -> None:
    df = synthetic_patient_meds()

    text = process_ehr_info(
        df,
        subject_id="patient_001",
        code_column="code",
        category_column="table",
        end_time=pd.Timestamp("2024-01-31"),
        time_bins=[(30, 15), (15, 7), (7, 0)],
    )

    print("=== formatted patient timeline ===")
    print(text)
    print("=== loading model ===")
    print(f"model_id: {DEFAULT_MODEL_ID}")
    model, tokenizer = load_model(DEFAULT_MODEL_ID)

    embedding = embed_patient(text, model, tokenizer)
    print(f"\nhidden_states[-1] shape: {tuple(embedding.shape)}")
    print(f"mean-pooled (first 8 dims): {embedding.mean(dim=1)[0, :8].float().tolist()}")


if __name__ == "__main__":
    main()
