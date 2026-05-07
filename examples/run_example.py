"""End-to-end smoke test: load the default checkpoint and embed the synthetic patient."""

import sys
from pathlib import Path

# Make the repo root importable when running as `python examples/run_example.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import DEFAULT_MODEL_ID, embed_patient, load_model  # noqa: E402


def main() -> None:
    text = (Path(__file__).parent / "synthetic_patient.txt").read_text()

    print(f"Loading {DEFAULT_MODEL_ID} ...")
    model, tokenizer = load_model(DEFAULT_MODEL_ID)

    embedding = embed_patient(text, model, tokenizer)
    print(f"Patient embedding shape: {tuple(embedding.shape)}")
    print(f"Mean-pooled (first 8 dims): {embedding.mean(dim=1)[0, :8].float().tolist()}")


if __name__ == "__main__":
    main()
