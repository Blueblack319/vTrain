from test_layer import MCoreGPTModel
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
import os
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

# vTrain
from src.model.fused_adam import FusedAdam as Adam
from vtrain_profiler import init_trace, timestamp, finish_trace
from pathlib import Path

# PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
# Configure the CUDA caching allocator to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def modify_functions(module):
    # assign pre-/post-hooks for forward and backward functions
    def forward_with_info(self, *args, **kwargs):
        timestamp(f"forward start {self.name}")
        ret = self.forward_(*args, **kwargs)
        timestamp(f"forward end {self.name}")

        name = self.name

        def backward_pre_hook(self, *args):
            timestamp(f"backward start {name}")

        # register backward prehook to the corresponding backward function
        if isinstance(ret, tuple):
            ret[0].grad_fn.register_prehook(backward_pre_hook)
        else:
            ret.grad_fn.register_prehook(backward_pre_hook)
        return ret

    def backward_hook(self, *args):
        timestamp(f"backward end {self.name}")

    module.forward_ = module.forward
    module.forward = forward_with_info.__get__(module, module.__class__)
    if all(p.requires_grad for p in module.parameters()):
        module.register_full_backward_hook(backward_hook)

    return module


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

    # [ ] Modify functions
    # for name, module in model.named_children():
    #     if name == "transformer":
    #         for name, layer in module.layers.named_children():
    #             # print(name)
    #             # print(layer)
    #             layer.name = name
    #             modify_functions(layer)
    #     else:
    #         # print(name)
    #         # print(module)
    #         module.name = name
    #         modify_functions(module)

    _criterion = torch.nn.CrossEntropyLoss().cuda()

    def submodule_backward_hook(module, grad_input, grad_output):
        # This code runs as part of the backward pass. We can add a record_function scope here:
        with record_function(f"Backward_{module._get_name()}"):
            # We do not change anything with gradients here; we're just labeling.
            pass
        return grad_input  # must return the same or updated grad_input

    model.register_full_backward_hook(submodule_backward_hook)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./test/profiler_logs/profiler_test_model"
        ),
    ) as prof:
        for batch_idx in range(5):
            # _ = model(
            #     tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            # )
            logit = model(input_ids, position_ids, attention_mask)
            logit = logit.view(-1, vocab_size)
            labels = torch.randint(
                0, vocab_size, (batch_size, seq_length), device="cuda"
            ).view(-1)
            print(logit.shape)
            loss = _criterion(
                logit,
                labels,
            )
            loss.backward()
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total"))

    # init_trace()
    # _ = model(input_ids, position_ids, attention_mask)
    # torch.cuda.synchronize()
    # traces = finish_trace().strip().split("\n")

    # log_filename = Path(f"test/logs/trace/trace_last")
    # log_filename.parent.mkdir(parents=True, exist_ok=True)
    # with open(log_filename, "w") as f:
    #     f.write("\n".join(traces))

    # for _ in range(10):
    #     _ = model(input_ids, position_ids, attention_mask)
    #     torch.cuda.synchronize()
