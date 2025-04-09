# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.initialize import initialize_megatron
from megatron.core.transformer.transformer_layer import TransformerLayer

# vTrain
from src.model.fused_adam import FusedAdam as Adam
from vtrain_profiler import init_trace, timestamp, finish_trace
from pathlib import Path


#################################################################
# From Megatron-LM
#################################################################

stimer = StragglerDetector()


def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print("saving allocated state during OOM")
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(
                    f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}",
                    "wb",
                ),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0("building GPT model ...")
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                print(f"\n\nget_gpt_decoder_block_spec() used\n\n")
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config,
                    use_transformer_engine=use_te,
                    normalization=args.normalization,
                )
            else:
                # Define the decoder layer spec
                if use_te:
                    print(f"\n\nget_gpt_layer_with_transformer_engine_spec() used\n\n")
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    print(f"\n\nget_gpt_layer_local_spec() used\n\n")
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization,
                    )
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=use_te
            )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if (
                    "preserve_high_precision_init_val"
                    in inspect.signature(fp8_model_init).parameters
                ):
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                )

        with build_model_context(**build_model_context_args):
            model = (
                GPTModel(
                    config=config,
                    transformer_layer_spec=transformer_layer_spec,
                    vocab_size=args.padded_vocab_size,
                    max_sequence_length=args.max_position_embeddings,
                    pre_process=pre_process,
                    post_process=post_process,
                    fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                    parallel_output=True,
                    share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                    position_embedding_type=args.position_embedding_type,
                    rotary_percent=args.rotary_percent,
                    rotary_base=args.rotary_base,
                    rope_scaling=args.use_rope_scaling,
                    mtp_block_spec=mtp_block_spec,
                )
                .cuda()
                .half()
            )

    return model, config


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat(
        [torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)]
    )

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {"lm loss": (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator
        )
    timers("batch-generator").stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


#################################################################


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

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
    )
    model, config = model_provider(post_process=True, pre_process=True)
    args = get_args()

    seq_length = args.seq_length
    max_position_length = seq_length
    batch_size = 16
    hidden_size = config.hidden_size
    vocab_size = args.padded_vocab_size
    micro_batch_size = args.micro_batch_size

    # tokens, labels, loss_mask, attention_mask, position_ids = create_example_batch(
    #     micro_batch_size=micro_batch_size, vocab_size=vocab_size, device="cuda"
    # )
    # [x] Input for Transformer Layer
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

    # [x] Input for Embedding
    input_ids = torch.randint(
        0, vocab_size, (batch_size, max_position_length), device="cuda"
    ).long()
    position_ids = (
        (torch.arange(seq_length, dtype=torch.long, device=input_ids.device))
        .unsqueeze(0)
        .expand_as(input_ids)
    )

    # [ ] Modify functions
    for name, module in model.named_children():
        if name == "decoder":
            for t_idx, transformer in module.layers.named_children():
                if isinstance(transformer, TransformerLayer):
                    transformer.name = f"transformer_{t_idx}"
                    modify_functions(transformer)
        else:
            module.name = name
            modify_functions(module)
    ###

    # [ ] Collect traces
    _criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        _criterion = _criterion.cuda()
        # [ ] Half?

    def criterion(outputs, labels):
        timestamp("forward start loss")
        loss = _criterion(outputs, labels)
        timestamp("forward end loss")
        return loss

    ###

    for batch_idx in range(5):
        # _ = model(
        #     tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
        # )
        init_trace()
        _ = model(input_ids, position_ids, attention_mask)
        traces = finish_trace().strip().split("\n")

        log_filename = Path(f"test/logs/trace/trace_{batch_idx}")
        log_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(log_filename, "w") as f:
            f.write("\n".join(traces))

    # pretrain(
    #     train_valid_test_datasets_provider,
    #     model_provider,
    #     ModelType.encoder_or_decoder,
    #     forward_step,
    #     args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    # )
