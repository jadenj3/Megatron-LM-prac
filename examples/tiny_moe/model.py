"""A tiny decoder-only MoE, assembled directly from Megatron Core."""

import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

VOCAB_SIZE = 128
SEQUENCE_LENGTH = 64


def build_model() -> GPTModel:
    """Call after initializing Megatron's process groups and RNG tracker."""
    config = TransformerConfig(
        # Each block contains causal self-attention followed by an MoE.
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        # All four experts live on one GPU. Use ordinary sequential expert MLPs.
        expert_model_parallel_size=1,
        moe_token_dispatcher_type="allgather",
        moe_grouped_gemm=False,
        add_bias_linear=False,  # Required by this checkout's MoE implementation.
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
    )
    # A specification chooses the existing Core classes used inside each block.
    layer_spec = get_gpt_layer_local_spec(num_experts=config.num_moe_experts)
    return GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        position_embedding_type="learned_absolute",
        share_embeddings_and_output_weights=False,
        parallel_output=False,
    )
