"""SMB-Structure text-only language model wrapper.

Minimal inference wrapper around Qwen3 / Qwen2 / Llama / Phi backbones.
Patient-timeline text is fed in directly; hidden states are exposed via
the standard HF forward signature (output_hidden_states=True).

Usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "<anonymous-hf-org>/smb-structure-qwen3-1.7b",
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "<anonymous-hf-org>/smb-structure-qwen3-1.7b"
    )
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    PhiForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# Try to import Qwen3, fall back to AutoModelForCausalLM
try:
    from transformers import Qwen3ForCausalLM
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False
    Qwen3ForCausalLM = None


# =============================================================================
# LLM BACKEND MAPPING
# =============================================================================

def get_llm_class(model_type: str):
    """Get LLM class based on model type string."""
    model_type = model_type.lower()
    
    if "qwen3" in model_type:
        if HAS_QWEN3:
            return Qwen3ForCausalLM
        else:
            return AutoModelForCausalLM
    elif "qwen2" in model_type or "qwen" in model_type:
        return Qwen2ForCausalLM
    elif "llama-3" in model_type or "llama3" in model_type or "meta-llama-3" in model_type:
        return AutoModelForCausalLM  # Llama 3 uses AutoModelForCausalLM
    elif "llama" in model_type or "vicuna" in model_type:
        return LlamaForCausalLM
    elif "phi" in model_type:
        return PhiForCausalLM
    else:
        return AutoModelForCausalLM


# =============================================================================
# CHAT TEMPLATES
# =============================================================================

@dataclass
class Qwen3Template:
    """ChatML template for Qwen3 models."""
    system_prompt: str = "You are a helpful assistant."
    
    def format_chat(self, prompt: str, system: str = None) -> str:
        sys = system or self.system_prompt
        return (
            f"<|im_start|>system\n{sys}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    @property
    def stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|endoftext|>"]


@dataclass
class Llama3Template:
    """Template for Llama 3 models."""
    system_prompt: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    
    def format_chat(self, prompt: str, system: str = None) -> str:
        sys = system or self.system_prompt
        return f"{sys} USER: {prompt} ASSISTANT:"
    
    @property
    def stop_tokens(self) -> List[str]:
        return ["<|end_of_text|>", "<|eot_id|>"]

@dataclass
class Qwen2Template:
    """Template for Qwen2 base models."""
    system_prompt: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    
    def format_chat(self, prompt: str, system: str = None) -> str:
        sys = system or self.system_prompt
        return f"{sys} USER: {prompt} ASSISTANT:"
    
    @property
    def stop_tokens(self) -> List[str]:
        return ["<|endoftext|>", "<|im_end|>"]


@dataclass
class LlamaTemplate:
    """Template for Llama/Vicuna models."""
    system_prompt: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    
    def format_chat(self, prompt: str, system: str = None) -> str:
        sys = system or self.system_prompt
        return f"{sys} USER: {prompt} ASSISTANT:"
    
    @property
    def stop_tokens(self) -> List[str]:
        return ["</s>"]


@dataclass  
class PhiTemplate:
    """Template for Phi models."""
    system_prompt: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    
    def format_chat(self, prompt: str, system: str = None) -> str:
        sys = system or self.system_prompt
        return f"{sys} USER: {prompt} ASSISTANT:"
    
    @property
    def stop_tokens(self) -> List[str]:
        return ["<|endoftext|>"]


def get_template(model_type: str):
    """Get chat template based on model type."""
    model_type = model_type.lower()
    
    if "qwen3" in model_type:
        return Qwen3Template()
    elif "qwen2" in model_type or "qwen" in model_type:
        return Qwen2Template()
    elif "llama-3" in model_type or "llama3" in model_type or "meta-llama-3" in model_type:
        return Llama3Template()
    elif "llama" in model_type or "vicuna" in model_type:
        return LlamaTemplate()
    elif "phi" in model_type:
        return PhiTemplate()
    else:
        return Qwen3Template()  # Default


# =============================================================================
# CONFIGURATION
# =============================================================================

class SMBUnstructuredConfig(PretrainedConfig):
    """Configuration for SMB Unstructured text-only model."""
    
    model_type = "smb_unstructured"
    
    def __init__(
        self,
        llm_model_name_or_path: str = "",
        tokenizer_name_or_path: str = None,
        text_config: dict = None,
        hidden_size: int = 2048,
        vocab_size: int = 32000,
        pad_token: str = None,
        pad_token_id: int = None,
        tokenizer_padding_side: str = "right",
        tokenizer_model_max_length: int = 2048,
        use_cache: bool = True,
        **kwargs
    ):
        self.llm_model_name_or_path = llm_model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or llm_model_name_or_path
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.tokenizer_padding_side = tokenizer_padding_side
        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.use_cache = use_cache
        
        # Load text config
        if text_config is not None:
            if isinstance(text_config, dict):
                self.text_config = AutoConfig.for_model(**text_config)
            else:
                self.text_config = text_config
        else:
            self.text_config = None
        
        # Extract hidden_size and vocab_size from text_config
        if self.text_config is not None:
            self.hidden_size = getattr(self.text_config, "hidden_size", hidden_size)
            self.vocab_size = getattr(self.text_config, "vocab_size", vocab_size)
        
        super().__init__(**kwargs)


# =============================================================================
# MAIN MODEL
# =============================================================================

class SMBUnstructuredPreTrainedModel(PreTrainedModel):
    """Base class for SMB Unstructured models."""
    
    config_class = SMBUnstructuredConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class SMBUnstructuredForCausalLM(SMBUnstructuredPreTrainedModel, GenerationMixin):
    """
    SMB Unstructured text-only language model.
    
    A minimal wrapper around the base LLM for inference.
    """
    
    def __init__(self, config: SMBUnstructuredConfig):
        super().__init__(config)
        
        # Detect LLM type from text_config
        if config.text_config is not None:
            llm_type = getattr(config.text_config, "model_type", "qwen3")
        else:
            llm_type = "qwen3"
        
        # Initialize language model
        llm_class = get_llm_class(llm_type)
        
        if llm_class == AutoModelForCausalLM:
            self.language_model = llm_class.from_config(config.text_config)
        else:
            self.language_model = llm_class(config.text_config)
        
        # Get chat template
        self.template = get_template(llm_type)
        self._llm_type = llm_type
        
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self, 
        new_num_tokens: Optional[int] = None, 
        pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass - direct passthrough to language model."""
        
        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache if use_cache is not None else self.config.use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate text - direct passthrough to language model."""
        
        # Handle inputs_embeds vs input_ids
        if "inputs_embeds" not in kwargs and inputs is not None:
            kwargs["input_ids"] = inputs
        
        return self.language_model.generate(**kwargs)

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        inputs_embeds=None, 
        **kwargs
    ):
        """Prepare inputs for generation."""
        return self.language_model.prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            **kwargs
        )

    def chat(
        self,
        prompt: str,
        tokenizer,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = None,
        **kwargs,
    ) -> str:
        """
        Chat interface for text generation.
        
        Args:
            prompt: User input prompt.
            tokenizer: Tokenizer instance.
            system_prompt: Optional system prompt override.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling parameter (default 0.9).
            top_k: Top-k sampling parameter (default 50).
            do_sample: Whether to sample (auto-detected from temperature).
            **kwargs: Additional generation arguments.
            
        Returns:
            Generated text response.
        """
        # Format prompt with template
        formatted_prompt = self.template.format_chat(prompt, system_prompt)
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        input_length = input_ids.shape[1]
        
        # Build stop token IDs
        eos_token_ids = []
        if tokenizer.eos_token_id is not None:
            eos_token_ids.append(tokenizer.eos_token_id)
        
        for token in self.template.stop_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id and token_id not in eos_token_ids:
                eos_token_ids.append(token_id)
        
        # Determine sampling strategy
        if do_sample is None:
            do_sample = temperature > 0
        
        # Build generation config
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": eos_token_ids if len(eos_token_ids) > 1 else (eos_token_ids[0] if eos_token_ids else None),
            "use_cache": True,
        }
        
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs["do_sample"] = False
        
        # Add any additional kwargs (but don't override what we set)
        for k, v in kwargs.items():
            if k not in gen_kwargs:
                gen_kwargs[k] = v
        
        # Generate
        with torch.inference_mode():
            output_ids = self.language_model.generate(**gen_kwargs)
        
        # Decode only new tokens
        generated_ids = output_ids[:, input_length:]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = response.strip()
        
        # Clean up stop tokens from response
        for token in self.template.stop_tokens:
            if response.endswith(token):
                response = response[:-len(token)].strip()
        
        return response

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype


# =============================================================================
# REGISTER WITH AUTO CLASSES
# =============================================================================

AutoConfig.register("smb_unstructured", SMBUnstructuredConfig)
AutoModelForCausalLM.register(SMBUnstructuredConfig, SMBUnstructuredForCausalLM)