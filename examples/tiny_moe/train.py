"""Single-GPU training exercise. Run from the repo root with torchrun -m."""

import argparse
import os

import torch


def synthetic_batches(batch_size, seq_length, vocab_size, device, random_data=False):
    """Yield input/next-token pairs without a tokenizer or dataset download."""
    positions = torch.arange(seq_length, device=device).expand(batch_size, -1)
    # Megatron's boolean attention mask uses True for positions to hide.
    mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=device).triu(1)
    mask = mask[None, None, :, :]  # [1, 1, sequence, sequence]
    while True:
        if random_data:
            sequence = torch.randint(vocab_size, (batch_size, seq_length + 1), device=device)
        else:
            # Random starting tokens, followed by a learnable counting pattern.
            start = torch.randint(vocab_size, (batch_size, 1), device=device)
            sequence = (start + torch.arange(seq_length + 1, device=device)) % vocab_size
        tokens = sequence[:, :-1].contiguous()
        labels = sequence[:, 1:].contiguous()
        yield tokens, positions, mask, labels


def forward_step(data_iterator, model):
    """Core's schedule calls this, then calls the returned loss function."""
    tokens, positions, mask, labels = next(data_iterator)
    # Supplying labels makes GPTModel return one cross-entropy loss per token.
    per_token_loss = model(tokens, positions, mask, labels=labels)

    def loss_func(losses):
        loss = losses.float().mean()
        return loss, {"lm_loss": loss.detach()}

    return per_token_loss, loss_func


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--random-data", action="store_true", help="Use unpredictable random tokens.")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("steps, batch-size, and lr must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("this example supports one process; use --nproc_per_node=1")
    if not torch.cuda.is_available():
        parser.error("Megatron Core training requires an NVIDIA CUDA GPU")
    if "RANK" not in os.environ:
        parser.error("launch with torchrun --standalone --nproc_per_node=1 -m examples.tiny_moe.train")

    # Keep --help and data inspection usable without importing the full Core stack.
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker

    from .model import SEQUENCE_LENGTH, VOCAB_SIZE, build_model

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    try:
        # Core's modules consult these groups even when every group has size one.
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        torch.manual_seed(args.seed)
        model_parallel_cuda_manual_seed(args.seed)
        model = build_model().cuda()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        data = synthetic_batches(
            args.batch_size, SEQUENCE_LENGTH, VOCAB_SIZE, "cuda", args.random_data
        )
        forward_backward = get_forward_backward_func()
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            clear_aux_losses_tracker()
            # Core runs forward_step, the loss function, and backward. It also
            # scales the router's auxiliary-loss gradients for this microbatch.
            metrics = forward_backward(
                forward_step_func=forward_step,
                data_iterator=data,
                model=model,
                num_microbatches=1,
                seq_length=SEQUENCE_LENGTH,
                micro_batch_size=args.batch_size,
                forward_only=False,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step == 1 or step % 10 == 0 or step == args.steps:
                loss = metrics[0]["lm_loss"].item()
                print(f"step {step:3d} | lm_loss {loss:.4f} | grad_norm {grad_norm.item():.4f}")
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
