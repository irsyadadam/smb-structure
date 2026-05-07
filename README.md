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

Weights are hosted at: [anonymous HF link]

To run inference against a specific checkpoint:

```bash
python inference.py \
    --model_id <anonymous-hf-org>/<repo-name> \
    --input examples/synthetic_patient.txt
```

The `--model_id` flag also accepts a local path to a downloaded snapshot.
