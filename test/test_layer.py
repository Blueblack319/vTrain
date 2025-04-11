import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Union
from megatron.core.transformer.identity_op import IdentityOp, IdentityFuncOp
import time

###
import os
from megatron.core.transformer.mlp import MLP
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType

from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from vtrain_profiler import init_trace, timestamp, finish_trace
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core import parallel_state
from torch.profiler import profile, record_function, ProfilerActivity

###

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
# Configure the CUDA caching allocator to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class TransformerLayer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        hidden_size=1024,
        num_attention_heads=16,
        tp_size=1,
        layer_number=1,
    ):
        super().__init__()

        self.layer_number = layer_number

        self.hidden_size = hidden_size
        self.tp_size = tp_size

        # Input Layernorm (Identity)
        self.input_layernorm = IdentityOp()

        # Decouple
        ###
        # Self Attention
        self.transformer_config = config
        self.core_attention = TEDotProductAttention(
            self.transformer_config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        # Linear Projection
        # Utils.initialize_model_parallel(1, 1)
        # torch.manual_seed(42)
        # model_parallel_cuda_manual_seed(42)

        self.linear_proj = TERowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=True,
            config=self.transformer_config,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name=None,
        )
        # Linear QKV Projection
        self.linear_qkv = TELayerNormColumnParallelLinear(
            input_size=hidden_size,  # hidden_size
            output_size=hidden_size * 3,  # Q + K + V projection
            config=self.transformer_config,
            init_method=self.transformer_config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name=None,
        )

        self.q_layernorm = IdentityOp()
        self.k_layernorm = IdentityOp()

        # Bias Dropout Add (self-attention)
        # self.self_attn_bda = get_bias_dropout_add(training=True, fused=True)
        self.self_attn_bda = get_bias_dropout_add

        # MLP Block
        self.pre_mlp_layernorm = IdentityOp()

        self.mlp = MLP(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.mlp.submodules,
        )
        self.mlp.cuda()

        # Bias Dropout Add (MLP)
        # self.mlp_bda = get_bias_dropout_add(training=True, fused=True)
        self.mlp_bda = get_bias_dropout_add
        ###

    def forward(self, hidden_states, attention_mask=None):
        # Residual connection.
        residual = hidden_states
        num_heads = self.core_attention.config.num_attention_heads
        head_dim = self.hidden_size // num_heads
        seq_length, batch_size, _ = hidden_states.shape

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        ###
        # Linear projection for QKV
        qkv_output, _ = self.linear_qkv(hidden_states)

        # Reshape QKV tensors properly for attention
        qkv_output = qkv_output.view(seq_length, batch_size, num_heads, 3 * head_dim)
        # Split QKV
        q, k, v = torch.chunk(qkv_output, 3, dim=-1)

        # Additional LayerNorm on Q and K if necessary
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        # Self-attention
        core_attention_output = self.core_attention(
            q,
            k,
            v,
            attention_mask,
            attn_mask_type=AttnMaskType.causal,
            attention_bias=None,
            packed_seq_params=None,
        )

        # Apply linear projection after attention
        attention_output_with_bias = self.linear_proj(core_attention_output)

        # Bias Dropout Add for attention
        hidden_states = self.self_attn_bda(
            self.training, self.transformer_config.bias_dropout_fusion
        )(attention_output_with_bias, residual, self.transformer_config.hidden_dropout)
        ###

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        output = self.mlp_bda(
            self.training, self.transformer_config.bias_dropout_fusion
        )(mlp_output_with_bias, residual, self.transformer_config.hidden_dropout)

        return output


class MCoreTransformerBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__()
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process

        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    config=config,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    tp_size=1,
                    layer_number=layer_number + 1,
                )
                for layer_number in range(config.num_layers)
            ]
        )

        # self.final_layernorm = torch.nn.LayerNorm(normalized_shape=config.hidden_size)
        self.final_layernorm = TENorm(
            config=config, hidden_size=config.hidden_size, eps=1e-5
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor]):
        for l_no, layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push("Transformer Layer")
            with record_function(f"YJH_PROFILE_Transformer_Layer_{l_no}"):
                hidden_states = layer(hidden_states, attention_mask)
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Final LayerNorm")
        with record_function(f"YJH_PROFILE_Final_LayerNorm"):
            output = self.final_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()
        return output

    # def __init__(
    #     self,
    #     num_layers,
    #     hidden_size,
    #     world_size,
    #     vocab_size=50257,
    #     num_attention_heads=16,
    #     embedding_dropout_prob=0.1,
    #     attention_dropout_prob=0.1,
    #     output_dropout_prob=0.1,
    #     max_sequence_length=1024,
    #     checkpoint_activations=False,
    #     checkpoint_num_layers=1,
    # ):


def backward_hook(module, grad_input, grad_output):
    with record_function(f"Backward_{module.__class__.__name__}"):
        # The hook does not define how backward is computed—
        # it simply runs *after* this module’s backward pass has computed grad_input/grad_output.
        pass
    return None  # you typically return None if you're not altering grads


class MCoreGPTModel(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int = 50257,
        world_size: int = 1,
        max_sequence_length=2048,
        pre_process: bool = True,
        post_process: bool = True,
        post_layer_norm: bool = True,
    ):
        super(MCoreGPTModel, self).__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.max_sequence_length = max_sequence_length
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.pre_process = pre_process
        self.post_process = post_process
        self.post_layer_norm = post_layer_norm

        # embeddings
        self.embeddings = LanguageModelEmbedding(
            config=config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type="learned_absolute",
            scatter_to_sequence_parallel=False,
        )

        # transformer
        self.transformer = MCoreTransformerBlock(
            config=config, post_layer_norm=True, pre_process=True, post_process=True
        )

        # output layer
        self.output_layer = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=vocab_size // world_size,
            init_method=config.init_method,
            bias=True,
            config=config,
            skip_bias_add=False,
        )
        # Register the hook on the modules you want to profile
        self.register_full_backward_hook(backward_hook)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        seq_length = self.max_sequence_length

        if attention_mask is None:
            attention_mask = (
                torch.tril(
                    torch.ones(
                        (input_ids.shape[0], seq_length, seq_length),
                        device=input_ids.device,
                    )
                )
                .view(input_ids.shape[0], 1, seq_length, seq_length)
                .half()
            )

        # Embeddings.
        torch.cuda.nvtx.range_push("Embedding")
        with record_function("YJH_PROFILE_Embedding"):
            hidden_states = self.embeddings(input_ids, position_ids)
        torch.cuda.nvtx.range_pop()

        # Transformer.
        hidden_states = self.transformer(hidden_states, attention_mask)

        if not self.post_process:
            return hidden_states

        # Logits
        torch.cuda.nvtx.range_push("Output Layer")
        with record_function("YJH_PROFILE_Output_Layer"):
            logits, _ = self.output_layer(hidden_states)
        torch.cuda.nvtx.range_pop()
        return logits


# Quick usage example (test)
if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    hidden_size = 1024
    num_attention_heads = 16
    batch_size = 16
    seq_length = 2048
    vocab_size = 50257
    max_position_length = seq_length
    max_sequence_length = seq_length
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,  # <- Should match your hidden_states dimension (1024)
        num_attention_heads=num_attention_heads,  # 1024 / 16 = head_dim 64 (a common size)
        use_cpu_initialization=False,
    )
    world_size = 1
    # parallel_state.initialize_model_parallel(world_size)
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)

    embedding = (
        LanguageModelEmbedding(
            config=config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type="learned_absolute",
            scatter_to_sequence_parallel=False,
        )
        .cuda()
        .half()
    )
    transformer = (
        MCoreTransformerBlock(
            config=config, post_layer_norm=True, pre_process=True, post_process=True
        )
        .cuda()
        .half()
    )
    output_layer = (
        ColumnParallelLinear(
            input_size=hidden_size,
            output_size=vocab_size // world_size,
            init_method=config.init_method,
            bias=True,
            config=config,
            skip_bias_add=False,
        )
        .cuda()
        .half()
    )

    model = (
        MCoreGPTModel(config, vocab_size, world_size, max_position_length).cuda().half()
    )

    # [ ] integration
    # for name, layer in transformer.named_children():
    #     print(name)

    # [x] Input for Transformer Layer
    hidden_states = torch.randn(
        seq_length, batch_size, hidden_size, dtype=torch.float16, device="cuda"
    ).contiguous()
    # attention_mask = torch.ones(
    #     1, 1, seq_length, seq_length, dtype=torch.float16, device="cuda"
    # ).contiguous()
    attention_mask = (
        torch.tril(torch.ones((batch_size, seq_length, seq_length), device="cuda"))
        .view(batch_size, 1, seq_length, seq_length)
        .half()
    )

    # [x] Input for Embedding
    input_ids = torch.randint(
        0, vocab_size, (batch_size, max_position_length), device="cuda"
    ).long()
    position_ids = (
        (torch.arange(seq_length, dtype=torch.long, device=input_ids.device))
        .unsqueeze(0)
        .expand_as(input_ids)
    )

    ### Nsys
    # Embedding
    # for i in range(5):
    #     output = embedding(input_ids, position_ids)
    #     print(output)
    #     torch.cuda.synchronize()

    # Transformer Layer
    # for i in range(5):
    #     output = transformer(hidden_states, attention_mask)
    #     torch.cuda.synchronize()

    # Output Layer
    # for i in range(5):
    #     logits, _ = output_layer(hidden_states, weight=None, runtime_gather_output=None)
    #     torch.cuda.synchronize()

    # Put things all together
    # for i in range(5):
    #     hidden_states = embedding(input_ids, position_ids)
    #     output = transformer(hidden_states, attention_mask)
    #     logits, _ = output_layer(hidden_states, weight=None, runtime_gather_output=None)
    #     print(output)
    #     torch.cuda.synchronize()

    # Model
    for i in range(5):
        output = model(input_ids, position_ids, attention_mask)
        torch.cuda.synchronize()
    ###

    ### CUPTI
    # Embedding
    # init_trace()
    # for i in range(5):
    #     timestamp(f"iter {i}")
    #     output = embedding(input_ids, position_ids)
    #     torch.cuda.synchronize()
    # timestamp("done")

    # Transformer layer
    # init_trace()
    # for i in range(5):
    #     timestamp(f"iter {i}")
    #     output = layer(hidden_states, attention_mask)
    #     torch.cuda.synchronize()
    # timestamp("done")

    # Output layer
    # init_trace()
    # for i in range(5):
    #     timestamp(f"iter {i}")
    #     output = output_layer(hidden_states, weight=None, runtime_gather_output=None)
    #     torch.cuda.synchronize()
    # timestamp("done")

    ###

    ### Collect trace
    # trace = finish_trace().strip().split("\n")
    # trace.sort(key=lambda l: int(l.split(",")[0]))
    # Utils.destroy_model_parallel()

    # trace_filtered = []
    # for line in trace:
    #     trace_type = line.split(",")[2]

    #     # Print CUDA-related traces only
    #     if trace_type in ["RUNTIME", "DRIVER"]:
    #         continue
    #     trace_filtered.append(line)

    # with open("embedding.csv", "w") as f:
    #     f.write("\n".join(trace_filtered))
    ###
