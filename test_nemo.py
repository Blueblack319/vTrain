# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import torch
from functools import partial
from contextlib import nullcontext
import inspect
import os
import nemo
from nemo.collections import llm

from nemo.collections.llm.gpt.model.deepseek import DeepSeekV2LiteConfig


def create_example_batch(
    micro_batch_size=2, seq_length=5, vocab_size=10000, device="cuda"
):
    """
    Creates a toy batch for Megatron-LM-style training.

    Args:
        micro_batch_size (int): Number of samples in the micro-batch
        seq_length (int): Sequence length
        vocab_size (int): Vocab size for random token generation

    Returns:
        tokens (torch.LongTensor): [micro_batch_size, seq_length]
        labels (torch.LongTensor): [micro_batch_size, seq_length]
        loss_mask (torch.FloatTensor): [micro_batch_size, seq_length]
        attention_mask (torch.BoolTensor): [micro_batch_size, 1, seq_length, seq_length]
        position_ids (torch.LongTensor): [micro_batch_size, seq_length]
    """

    # 1) TOKENS: random integers in [0, vocab_size).
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(micro_batch_size, seq_length),
        dtype=torch.long,
        device=device,
    )

    # 2) LABELS: toy example shifting the tokens for next-token prediction.
    #    We'll shift each row left by 1; the last token is random:
    labels = tokens.clone()
    if seq_length > 1:
        labels[:, :-1] = tokens[:, 1:]
    labels[:, -1] = torch.randint(
        low=0,
        high=vocab_size,
        size=(micro_batch_size,),
        dtype=torch.long,
        device=device,
    )

    # 3) LOSS MASK: mark everything as 1.0, except we might zero-out some padding.
    loss_mask = torch.ones(
        (micro_batch_size, seq_length), dtype=torch.float16, device=device
    )
    # For example, pretend the last token of sample #2 is padding:
    # if micro_batch_size > 1:
    #     loss_mask[1, -1] = 0.0

    # 4) ATTENTION MASK: shape [batch_size, 1, seq_length, seq_length].
    #    We create a causal (lower-triangular) mask, so token j only attends
    #    to tokens [0..j].
    # attention_mask = torch.zeros(
    #     (micro_batch_size, 1, seq_length, seq_length), dtype=torch.bool
    # )
    # for i in range(micro_batch_size):
    #     for j in range(seq_length):
    #         for k in range(seq_length):
    #             # True where k <= j (no future tokens)
    #             attention_mask[i, 0, j, k] = k <= j
    attention_mask = (
        torch.tril(
            torch.ones((micro_batch_size, seq_length, seq_length), device=device)
        )
        .view(micro_batch_size, 1, seq_length, seq_length)
        .half()
    )

    # 5) POSITION IDS: typical range [0..seq_length-1].
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # Expand to [batch_size, seq_length]
    position_ids = (
        position_ids.unsqueeze(0).expand(micro_batch_size, seq_length).clone()
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


if __name__ == "__main__":
    seq_length = 256
    global_batch_size = 1
    vocab_size = 50257

    tokens, labels, loss_mask, attention_mask, position_ids = create_example_batch(
        micro_batch_size=global_batch_size, vocab_size=vocab_size, device="cuda"
    )

    ## setup the dummy dataset
    data = llm.MockDataModule(
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=global_batch_size,
    )

    ## initialize a small GPT model
    deepseek_config = DeepSeekV2LiteConfig(
        num_layers=27,
        # hidden_size=2048,
        # ffn_hidden_size=10944,
        hidden_size=512,
        ffn_hidden_size=512,
        num_attention_heads=16,
        num_moe_experts=64,
        moe_ffn_hidden_size=1408,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
    )
    model = llm.DeepSeekModel(deepseek_config, tokenizer=data.tokenizer)

    for _ in range(10):
        _ = model(tokens, position_ids, attention_mask)
        torch.cuda.synchronize()
