import torch
import torch.nn as nn
from torch.nn import LayerNorm, Dropout, Embedding, ModuleList, Identity

# from .gpt_modeling import ShardedGptTransformer
# from .gpt_modeling import ShardedGptEmbeddings
# from .gpt_modeling import ShardedGptLogit

# from apex.normalization import FusedLayerNorm as LayerNorm
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.transformer import GPTTransformer
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state

from mcore_gpt_modeling import CustomGPTTransformer


class MCoreGptModel(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        world_size,
        vocab_size=50257,
        num_attention_heads=16,
        embedding_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        max_sequence_length=1024,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
    ):

        super(MCoreGptModel, self).__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.max_sequence_length = max_sequence_length

        parallel_state.initialize_model_parallel(world_size)

        self.config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=hidden_size * 4,
            seq_length=max_sequence_length,
            vocab_size=vocab_size,
            hidden_dropout=output_dropout_prob,
            attention_dropout=attention_dropout_prob,
            embedding_dropout=embedding_dropout_prob,
            use_cpu_initialization=False,
            fp16=True,
        )

        # Embeddings
        self.embedding = LanguageModelEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
            embedding_dropout=embedding_dropout_prob,
            position_embedding_type="learned_absolute",
        )

        # Transformer
        # self.transformer = CustomGPTTransformer(self.config)
        transformer_layers = ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    self_attention=SelfAttention(
                        hidden_size=hidden_size,
                        num_attention_heads=num_attention_heads,
                        attention_dropout=attention_dropout_p,
                        core_attention=TEDotProductAttention(
                            hidden_size=hidden_size,
                            num_attention_heads=num_attention_heads,
                            attention_dropout=attention_dropout_p,
                            attention_type="flash",  # or 'fused', 'unfused'
                        ),
                        linear_proj=TERowParallelLinear(
                            hidden_size, hidden_size, bias=True
                        ),
                        linear_qkv=TEColumnParallelLinear(
                            hidden_size, hidden_size * 3, bias=True
                        ),
                    ),
                    mlp=MLP(
                        hidden_size=hidden_size,
                        ffn_hidden_size=ffn_hidden_size,
                        hidden_dropout=hidden_dropout_p,
                        linear_fc1=ColumnParallelLinear(
                            hidden_size, ffn_hidden_size, bias=True
                        ),
                        linear_fc2=RowParallelLinear(
                            ffn_hidden_size, hidden_size, bias=True
                        ),
                    ),
                    input_layernorm=Identity(),
                    pre_cross_attn_layernorm=Identity(),
                    cross_attention=Identity(),
                    cross_attn_bda=Identity(),
                    pre_mlp_layernorm=Identity(),
                )
                for _ in range(num_layers)
            ]
        )

        # Final LayerNorm
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-5)

        # Logits(Output layer)
        self.logit = ColumnParallelLinear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), device=input_ids.device
            ).unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.tril(
                torch.ones(
                    (input_ids.shape[0], 1, input_ids.size(1), input_ids.size(1)),
                    device=input_ids.device,
                )
            ).bool()

        # Embeddings
        embedding = self.embedding(input_ids, position_ids)

        # Transformer
        transformer_output = self.transformer(embedding, attention_mask)
        transformer_output = self.layernorm(transformer_output)

        # Logits
        logits = self.logit(transformer_output)

        return logits


# Usage example (quick test)
if __name__ == "__main__":
    model = MCoreGptModel(num_layers=2, hidden_size=512, world_size=1)
    input_ids = torch.randint(0, 50257, (2, 128)).cuda()
    model.cuda()
    logits = model(input_ids)
    print(logits.shape)  # Should output [batch_size, seq_length, vocab_size]
