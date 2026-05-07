# SMB-Structure: Anonymous Submission

Code and weights for the NeurIPS 2026 submission "The Patient is not a
Moving Document: A World Model Training Paradigm for Longitudinal EHR."

## What's here

- `smb_utils/` — anonymized eval pipeline: turns a MEDS-format patient
  timeline (DataFrame) into the structured text the model consumes.
- `inference.py` — `load_model()` and `embed_patient()` helpers around
  the released checkpoints.
- `examples/run_example.py` — end-to-end eval on a synthetic patient:
  MEDS DataFrame → `smb_utils` → model → patient embedding.
- `model/` — model and tokenizer implementation.

## Quick start

```bash
pip install -r requirements.txt
python examples/run_example.py
```

This runs the smb-utils-driven eval against the default 1.7B Qwen3
checkpoint. Override the model with the `SMB_MODEL_ID` env var:

```bash
SMB_MODEL_ID=anon-9421/smb-structure-qwen3-8b-multi-objective \
    python examples/run_example.py
```

## Weights

All five released checkpoints are hosted on Hugging Face under the
anonymous `anon-9421` account:

| Model | Backbone | Objective | Link |
|---|---|---|---|
| `smb-structure-qwen3-1.7b-multi-objective` | Qwen3-1.7B | SFT + JEPA multi-objective | https://huggingface.co/anon-9421/smb-structure-qwen3-1.7b-multi-objective |
| `smb-structure-qwen3-8b-multi-objective` | Qwen3-8B | SFT + JEPA multi-objective | https://huggingface.co/anon-9421/smb-structure-qwen3-8b-multi-objective |
| `smb-structure-qwen3-8b-curriculum` | Qwen3-8B | SFT + JEPA curriculum | https://huggingface.co/anon-9421/smb-structure-qwen3-8b-curriculum |
| `smb-structure-llama3-8b-multi-objective` | Llama-3-8B | SFT + JEPA multi-objective | https://huggingface.co/anon-9421/smb-structure-llama3-8b-multi-objective |
| `smb-structure-llama3-8b-curriculum` | Llama-3-8B | SFT + JEPA curriculum | https://huggingface.co/anon-9421/smb-structure-llama3-8b-curriculum |

The 1.7B Qwen3 checkpoint is the default — smallest and fastest to load.
