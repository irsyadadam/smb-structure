# SMB-Structure: Anonymous Submission

Code and weights for the NeurIPS 2026 submission "The Patient is not a
Moving Document: A World Model Training Paradigm for Longitudinal EHR."

## What's here

- `inference.py` — load the model and run a forward pass
- `model/` — model and tokenizer implementation
- `examples/` — synthetic patient example demonstrating input format

## What's not here yet

To preserve anonymity and finalize documentation, the following will be
released upon acceptance:

- Training code (SFT + JEPA objectives)
- Preprocessing pipeline (MSK CCDE → token sequences)
- Evaluation cookbook (linear probe protocol, 515 endpoints)
- Pretrained checkpoints for additional model sizes

## Quick start

```bash
pip install -r requirements.txt
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

To run inference against a specific checkpoint:

```bash
python inference.py \
    --model_id anon-9421/smb-structure-qwen3-8b-multi-objective \
    --input examples/synthetic_patient.txt
```

The `--model_id` flag also accepts a local path to a downloaded snapshot.
