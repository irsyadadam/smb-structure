"""Load an SMB-Structure checkpoint and run a single forward pass.

Reads a pre-formatted patient timeline (the output of the preprocessing
pipeline, which is not part of this anonymized release), tokenizes it,
runs a forward pass with hidden states enabled, and returns the final
hidden state as a patient embedding.
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = os.environ.get(
    "SMB_MODEL_ID", "<anonymous-hf-org>/smb-structure-qwen3-1.7b"
)


def load_model(model_id: str, device_map: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def embed_patient(text: str, model, tokenizer) -> torch.Tensor:
    """Return the last-layer hidden state for a pre-formatted timeline.

    Shape: (1, seq_len, hidden_size). Pool externally as needed (e.g.,
    mean over seq_len, or take the final non-pad position) to produce
    a fixed-size patient embedding.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help="HF repo id or local path. Override with SMB_MODEL_ID env var.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "examples" / "synthetic_patient.txt",
        help="Path to a pre-formatted patient timeline (plain text).",
    )
    args = parser.parse_args()

    text = args.input.read_text()
    model, tokenizer = load_model(args.model_id)
    embedding = embed_patient(text, model, tokenizer)

    print(f"input_tokens: {tokenizer(text, return_tensors='pt').input_ids.shape[1]}")
    print(f"hidden_states[-1] shape: {tuple(embedding.shape)}")
    print(f"mean-pooled embedding[:8]: {embedding.mean(dim=1)[0, :8].float().tolist()}")


if __name__ == "__main__":
    main()
