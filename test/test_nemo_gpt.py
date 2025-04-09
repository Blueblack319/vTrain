import torch
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.callbacks import NsysCallback
from megatron.core.optimizer import OptimizerConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.profilers import PyTorchProfiler

import time
from typing import Optional

# vTrain
from vtrain_profiler import init_trace, timestamp, finish_trace
import logging
from pathlib import Path

# PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

logger = logging.getLogger()


class IterationDurationCallback(Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0  # <--- Add this
    ):
        self.batch_start_time = time.time()

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,  # <--- Add this
    ):
        duration = time.time() - self.batch_start_time
        pl_module.log(
            "iteration_duration", duration, on_step=True, on_epoch=False, prog_bar=True
        )


class GPTModelCallback(Callback):
    def __init__(self):
        super().__init__()
        self.traces = None

    # YJH: Mimic modify_functions()
    def on_train_start(self, trainer, pl_module) -> None:
        print("Modifying model starts...")

        # assign pre-/post-hooks for forward and backward functions
        def forward_with_info(self, *args, **kwargs):
            torch.cuda.synchronize()
            timestamp(f"forward start {self.name}")
            ret = self.forward_(*args, **kwargs)
            torch.cuda.synchronize()
            timestamp(f"forward end {self.name}")

            name = self.name

            def backward_pre_hook(self, *args):
                torch.cuda.synchronize()
                timestamp(f"backward start {name}")
                print(f"backward start {name}")

            def backward_hook(self, *args):
                torch.cuda.synchronize()
                timestamp(f"backward end {name}")
                print(f"backward end {name}")

            # register backward prehook to the corresponding backward function
            if isinstance(ret, tuple):
                ret[0].grad_fn.register_prehook(backward_pre_hook)
                ret[0].grad_fn.register_hook(backward_hook)
            else:
                ret.grad_fn.register_prehook(backward_pre_hook)
                ret.grad_fn.register_hook(backward_hook)
            return ret

        def backward_hook(self, *args):
            torch.cuda.synchronize()
            timestamp(f"backward end {self.name}")
            print(f"backward end {self.name}")

        for name, module in pl_module.module.module.module.named_children():
            if name == "decoder":
                for name, module in module.layers.named_children():
                    module.name = "transformer"
                    module.forward_ = module.forward
                    module.forward = forward_with_info.__get__(module, module.__class__)
                    # if all(p.requires_grad for p in module.parameters()):
                    #     module.register_full_backward_hook(backward_hook)
            else:
                module.name = name
                module.forward_ = module.forward
                module.forward = forward_with_info.__get__(module, module.__class__)
                # if all(p.requires_grad for p in module.parameters()):
                #     module.register_full_backward_hook(backward_hook)
        print("Modifying model ends...")

    # YJH: Profiling kernels
    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int
    ):
        # print("Starting trace...", flush=True)
        init_trace()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        # print("Finishing trace...", flush=True)
        self.traces = finish_trace().strip().split("\n")

        # log_filename = Path(f"logs/trace/trace_{batch_idx}")
        # log_filename.parent.mkdir(parents=True, exist_ok=True)
        # with open(log_filename, "w") as f:
        #     f.write("\n".join(gpt_callback.traces))


"""
[ ] micro_batch_size in NeMo?
[x] ffn_hidden_size == hidden_size in vTrain
"""

if __name__ == "__main__":
    seq_length = 2048
    global_batch_size = 16
    log_base_dir = "logs/trace"
    log_filename = Path(log_base_dir) / f"trace_test"

    ## setup the dummy dataset
    data = llm.MockDataModule(
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=global_batch_size,
    )

    ## initialize a small GPT model
    gpt_config = llm.GPTConfig(
        num_layers=5,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        enable_cuda_graph=False,
    )
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=6e-4,
        bf16=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    # [ ] For vTrain
    gpt_callback = GPTModelCallback()

    # [ ] For PyTorch Profiler
    profiler = PyTorchProfiler(
        schedule=torch.profiler.schedule(skip_first=2, wait=1, warmup=1, active=10),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "tb_logs/profiler_test_nemo_gpt"
        ),
        # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes=True,
        # profile_memory=True,
        # with_stack=True,
    )
    trainer = nl.Trainer(
        profiler=profiler,
        devices=1,  # you can change the number of devices to suit your setup
        max_steps=9,
        accelerator="gpu",
        strategy=strategy,
        enable_checkpointing=False,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[
            IterationDurationCallback(),
            # NsysCallback(start_step=2, end_step=100, ranks=[0], gen_shape=False),
            gpt_callback,
        ],
    )

    # YJH: Loggers
    tensorboard = TensorBoardLogger(
        save_dir="tb_logs",
        name="test_nemo_gpt",
    )
    nemo_logger = nl.NeMoLogger(
        log_dir="logs",  ## logs and checkpoints will be written here
        name="test_nemo_gpt",
        tensorboard=tensorboard,
        # wandb=wandb_logger,
        update_logger_directory=True,
    )

    # with profile(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/filename")
    # ) as p:

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=opt,
    )
    # logger.info(f"number of traces collected: {len(gpt_callback.traces)}")

    log_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(log_filename, "w") as f:
        f.write("\n".join(gpt_callback.traces))
