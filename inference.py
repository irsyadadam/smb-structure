"""Model loader and patient-embedding helpers.

The eval pipeline lives in `examples/run_example.py`, which uses
`smb_utils.process_ehr_info` to turn a MEDS DataFrame into the
structured text these helpers consume.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = os.environ.get(
    "SMB_MODEL_ID", "anon-9421/smb-structure-qwen3-1.7b-multi-objective"
)


def load_model(model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto"):
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
