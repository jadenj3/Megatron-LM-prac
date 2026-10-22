# Tiny MoE training exercise

A small, randomly initialized decoder-only transformer using **Megatron Core**
directly. Start with `model.py` for the architecture, then read `train.py` for
the data and training loop. This is an example configuration of Core's existing
`GPTModel`, so it belongs under `examples/` rather than adding a library model
under `megatron/core/models/`.

The model has two layers, hidden size 128, four attention heads, and four
experts per layer. Each expert is a 128 → 256 → 128 GELU MLP; the router selects
two experts per token. It uses a 128-token vocabulary, sequences of length 64,
learned position embeddings, and approximately 0.7 million total parameters.
All experts live on one GPU. A small auxiliary loss encourages balanced routing.

## Run

Use a Linux environment with an NVIDIA GPU, CUDA-enabled PyTorch, and this
checkout's Megatron Core dependencies installed. A Mac's MPS backend cannot run
this training path. From the repository root:

```bash
torchrun --standalone --nproc_per_node=1 -m examples.tiny_moe.train
```

For a short smoke run:

```bash
torchrun --standalone --nproc_per_node=1 -m examples.tiny_moe.train --steps 5
```

No pretrained weights, tokenizer, dataset files, or online downloads are used by
the example. The default synthetic data starts at a random token and counts
up modulo 128. For example, input `[126, 127, 0, 1]` has labels `[127, 0, 1, 2]`.
The causal mask hides future input tokens. This task should produce a falling
language-model loss, starting near `ln(128) ≈ 4.85`.

Add `--random-data` to use fresh independent random tokens instead. That mode
checks execution; its expected next-token loss cannot improve below `ln(128)`.

## Follow one training step

1. Generate a batch of input tokens and shifted labels.
2. Clear gradients with `optimizer.zero_grad()`.
3. Core's `get_forward_backward_func()` schedule calls `forward_step`, averages
   the per-token losses, and runs backward. It also handles MoE auxiliary-loss
   gradient scaling.
4. Clip gradients and update parameters with ordinary PyTorch AdamW.

The printed `lm_loss` excludes the router auxiliary loss, which still contributes
gradients during backward. `grad_norm` is the norm before clipping.

This deliberately uses float32, one microbatch per update, and one process.
There is no data-parallel wrapper, checkpointing, or validation loop. Increasing
the number of processes requires adding gradient synchronization and data
partitioning; the script rejects that configuration. For now, edit `model.py` to
explore architecture settings and use `--steps`, `--batch-size`, and `--lr` to
adjust training.
