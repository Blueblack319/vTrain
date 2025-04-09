from test_layer import MCoreGPTModel
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
import os
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
# Configure the CUDA caching allocator to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.float16)

    hidden_size = 1024
    num_attention_heads = 16
    batch_size = 16
    seq_length = 2048
    vocab_size = 50257
    world_size = 1
    max_position_length = seq_length
    max_sequence_length = seq_length
    config = TransformerConfig(
        num_layers=4,
        hidden_size=hidden_size,  # <- Should match your hidden_states dimension (1024)
        num_attention_heads=num_attention_heads,  # 1024 / 16 = head_dim 64 (a common size)
        use_cpu_initialization=False,
    )
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)

    # Create a model
    model = (
        MCoreGPTModel(config, vocab_size, world_size, max_position_length).cuda().half()
    )

    # [x] Test for Transformer Layer
    # hidden_states = torch.randn(
    #     seq_length, batch_size, hidden_size, dtype=torch.float16, device="cuda"
    # ).contiguous()
    # attention_mask = torch.ones(
    #     1, 1, seq_length, seq_length, dtype=torch.float16, device="cuda"
    # ).contiguous()
    attention_mask = (
        torch.tril(torch.ones((batch_size, seq_length, seq_length), device="cuda"))
        .view(batch_size, 1, seq_length, seq_length)
        .half()
    )

    # [x] Test for Embedding
    input_ids = torch.randint(
        0, vocab_size, (batch_size, max_position_length), device="cuda"
    ).long()
    position_ids = (
        (torch.arange(seq_length, dtype=torch.long, device=input_ids.device))
        .unsqueeze(0)
        .expand_as(input_ids)
    )

    for _ in range(10):
        _ = model(input_ids, position_ids, attention_mask)
        torch.cuda.synchronize()
