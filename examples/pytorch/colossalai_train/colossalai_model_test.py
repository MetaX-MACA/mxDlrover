import argparse
import functools
import random
from contextlib import nullcontext
from functools import partial
from time import (time, sleep)

import colossalai
import numpy as np
import torch
from colossalai.booster import Booster
from colossalai.booster.plugin import (GeminiPlugin, HybridParallelPlugin,
                                       LowLevelZeroPlugin)
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.legacy.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from datasets import load_dataset
from torch.optim import AdamW
from transformers import LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# from arsenal import (enable_MLP_fusion, replace_flash_attn, replace_rms_norm,
#                      replace_sdp_attn)
from data_utils import RandomDataset
from model_utils import format_numel_str, get_model_numel
from dlrover.trainer.torch.flash_checkpoint.colossalai_hybrid import (load_checkpoint, save_checkpoint)
# from ckpt_io import (load_checkpoint, save_checkpoint)
# ==============================
# Constants
# ==============================

def tokenize_batch_for_pretrain(batch, tokenizer: LlamaTokenizer, max_length: int):
    texts = [sample["text"] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


MODEL_CONFIGS = {
    "7b": LlamaConfig(max_position_embeddings=4096),
    "13b": LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=4096,
    ),
    "70b": LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        max_position_embeddings=4096,
        num_key_value_heads=8,
    ),
}

class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.duration: float = 0.0

    def start(self) -> None:
        self.start_time = time()

    def end(self) -> None:
        assert self.start_time is not None
        self.duration += time() - self.start_time
        self.start_time = None

    def reset(self) -> None:
        self.duration = 0.0

def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Random seed")
    parser.add_argument("-c", "--model_config", type=str, default="7b", help="Model configuration")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["gemini", "gemini_auto", "fsdp", "fsdp_cpu", "3d", "3d_cpu", "zero2", "zero2_cpu", "zero1"],
        default="gemini",
        help="Choose which plugin to use",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-a", "--grad_accum", type=int, default=1, help="Gradient accumulation step: Number of forward-backward for each optimizer update.")
    parser.add_argument("-s", "--num_steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("-i", "--ignore_steps", type=int, default=2, help="Number of steps to ignore")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--grad_checkpoint_ratio", type=float, default=0.5, help="Gradient checkpointing ratio")
    parser.add_argument("-l", "--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "-w", "--warmup_ratio", type=float, default=0.8, help="warm up ratio of non-model data. Only for gemini-auto"
    )
    parser.add_argument("-m", "--memory_limit", type=int, help="Gemini memory limit in mb")
    parser.add_argument("--use_hybrid_adam", action="store_true", help="Use hybrid adam as the optimizer")
    parser.add_argument("--use_fused_rms_norm", action="store_true", help="Use FusedRMSNorm implemented by Apex")
    parser.add_argument("--flash_attn", type=str, choices=["normal", "packed", "none"], default="normal", help="Use flash attention 2 in its normal usage or qkv packed usage.")
    parser.add_argument("--use_sdp_attn", action="store_true", help="Use Scaled Dot Product Attention in Pytorch")
    parser.add_argument("--use_fused_mlp", action="store_true", help="Use nvprimes_nvfuser backend to compile mlp forward in torch 2.0")
    parser.add_argument("--shard_param_frac", type=float, default=1.0, help="Shard param fraction. Only for gemini")
    parser.add_argument("--offload_optim_frac", type=float, default=0.0, help="Offload optim fraction. Only for gemini")
    parser.add_argument("--offload_param_frac", type=float, default=0.0, help="Offload param fraction. Only for gemini")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size of pipeline parallel")
    parser.add_argument("--zero", type=int, default=0, help="Zero Stage when hybrid plugin is enabled")
    parser.add_argument("--use_torch_profiler", action="store_true", help="Whether to enable torch profiler")
    parser.add_argument("--torch_profiler_path", type=str, default="./logs", help="Path to the directory where pytorch profiler is saved.")
    parser.add_argument("--empty_cache", action="store_true", help="Whether to empty cache after each iteration.")
    parser.add_argument("--include_optimizer_time", action="store_true", help="Whether to include optimizer time in the performance report.")
    parser.add_argument("--disable_internal_sync", action="store_true", help="Whether to disable internal torch.cuda.synchronize() in the EvalHook.")
    parser.add_argument("--display_loss", action="store_true", help="Whether to display loss in each iteration.")
    parser.add_argument("--dataset", default=None, help="Dataset path")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weigth_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--extra_dp_size", type=int, default=1, help="Extra data parallel size")
    parser.add_argument("--save_interval", type=int, default=100, help="Save interval")
    parser.add_argument("--save_dir", type=str, default="checkpoint_dir", help="Checkpoint directory")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint")
    args = parser.parse_args()

    #args.dataset = "/software/home/meli/downloads/RedPajama-Data-1T-Sample"

    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    logger = get_dist_logger()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision="bf16",
            shard_param_frac=args.shard_param_frac,
            offload_optim_frac=args.offload_optim_frac,
            offload_param_frac=args.offload_param_frac,
            enable_fused_normalization=args.use_fused_rms_norm,
            enable_gradient_accumulation=(args.grad_accum > 1),
            tp_size=args.tp,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            placement_policy="auto", 
            precision="bf16", 
            warmup_non_model_data_ratio=args.warmup_ratio, 
            enable_fused_normalization=args.use_fused_rms_norm,
            enable_gradient_accumulation=(args.grad_accum > 1),
            tp_size=args.tp,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "3d":
        plugin = HybridParallelPlugin(
            extra_dp_size=args.extra_dp_size,
            tp_size=args.tp,
            pp_size=args.pp,
            pp_style="interleaved",
            num_model_chunks=2,
            zero_stage=args.zero,
            enable_fused_normalization=args.use_fused_rms_norm,
            microbatch_size=args.mbs,
            precision="bf16",
            max_norm=args.grad_clip,
        )
    elif args.plugin == "3d_cpu":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            pp_style="1f1b",
            num_model_chunks=1,
            zero_stage=args.zero,
            cpu_offload=True,
            enable_fused_normalization=args.use_fused_rms_norm,
            microbatch_size=args.mbs,
            precision="bf16",
            max_norm=args.grad_clip,
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2, precision="bf16", initial_scale=2**16, max_norm=1.0, reduce_bucket_size_in_m=64
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2, precision="bf16", initial_scale=2**16, cpu_offload=True, max_norm=1.0, reduce_bucket_size_in_m=64
        )
    elif args.plugin == "zero1":
        plugin = LowLevelZeroPlugin(
            stage=1, precision="bf16", initial_scale=2**16, max_norm=1.0
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Dataset and Dataloader
    # ==============================
    if isinstance(plugin, GeminiPlugin):
        dp_size = plugin.zero_size
    elif isinstance(plugin, HybridParallelPlugin):
        dp_size = plugin.dp_size
    else:
        dp_size = coordinator.world_size

    config = MODEL_CONFIGS[args.model_config]

    if args.dataset == "random":
        dataset = RandomDataset(
            num_samples=args.batch_size, max_length=args.max_length, vocab_size=config.vocab_size
        )
        dataloader = plugin.prepare_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
    else:
        if coordinator.is_master():
            logger.info(f"Loading dataset {args.dataset}")
        dataset = load_dataset(args.dataset, split="train")
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
        tokenizer.pad_token = tokenizer.unk_token
        dataloader = plugin.prepare_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=partial(tokenize_batch_for_pretrain, tokenizer=tokenizer, max_length=args.max_length)
        )

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    init_ctx = (
        LazyInitContext(default_device=get_current_device())
        if isinstance(plugin, (GeminiPlugin, HybridParallelPlugin))
        else nullcontext()
    )

    coordinator.print_on_master(f"Creating Model...")
    with init_ctx:
        model = LlamaForCausalLM(config)

    # Set Optimizations
    assert 0 <= args.grad_checkpoint_ratio <= 1
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        gpc.grad_checkpoint_ratio = args.grad_checkpoint_ratio

    if args.use_fused_rms_norm and not isinstance(plugin, HybridParallelPlugin) and not isinstance(plugin, GeminiPlugin): 
        # Shardformer/Gemini will handle FusedRMSNorm itself
        replace_rms_norm(model)

    # if args.flash_attn != "none":
    #     replace_flash_attn(model, mode=args.flash_attn)
    # elif args.use_sdp_attn:
    #     replace_sdp_attn(model)

    if args.use_fused_mlp:
        enable_MLP_fusion(model)


    model_numel = get_model_numel(model)
    coordinator.print_on_master(f"Finish Creating Model, Model params: {format_numel_str(model_numel)}, Plugin: {args.plugin}, Size of DP-PP-TP: {dp_size}-{args.pp}-{args.tp}")
    coordinator.print_on_master(f"Batch Size per Device: {args.batch_size}, Gradient Accumulation Step: {args.grad_accum}, Global Batch Size: {args.batch_size * args.grad_accum * dp_size}")
    if isinstance(plugin, HybridParallelPlugin):
        coordinator.print_on_master(f"Pipeline Micro Batch Size: {args.mbs}, Zero Stage: {args.zero}")

    coordinator.print_on_master(f"Use Gradient Checkpoint: {args.grad_checkpoint}, "
                                f"Flash Attention Mode: {args.flash_attn}, "
                                f"Use SDP Attention: {args.use_sdp_attn}, "
                                f"Use Hybrid Adam: {args.use_hybrid_adam}, "
                                f"Use Fused RMS Norm: {args.use_fused_rms_norm}, "
                                f"Use Fused MLP: {args.use_fused_mlp}")

    if args.plugin in ["gemini", "gemini_auto"] or args.use_hybrid_adam:
        optimizer = HybridAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weigth_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weigth_decay)
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=args.num_steps * len(dataloader), warmup_steps=args.warmup_steps, eta_min=0.1 * args.lr
    )

    torch.set_default_dtype(torch.bfloat16)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
    )
    torch.set_default_dtype(torch.float)
    torch.cuda.empty_cache()

    # ==============================
    # Start Training Loop
    # =============================

    timer = Timer()
    start_step = 0

    if args.load_checkpoint is not None:
        timer.start()
        if "modeling" in args.load_checkpoint:
            coordinator.print_on_master(f"Continued pretrain from checkpoint {args.load_checkpoint}")
            booster.load_model(model, args.load_checkpoint)
        else:
            coordinator.print_on_master(f"Load model checkpoint from {args.load_checkpoint}")
            start_epoch, start_step, sampler_start_idx = load_checkpoint(
                load_dir=args.load_checkpoint,
                booster=booster,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            coordinator.print_on_master(
                f"Loaded checkpoint {args.load_checkpoint} at epoch {start_epoch} step {start_step}"
            )
            coordinator.print_on_master(f"Loaded sample at index {sampler_start_idx}")
            # dataloader.sampler.set_start_index(start_index=sampler_start_idx)
        timer.end()
        coordinator.print_on_master(f"Load time {timer.duration}")
    optimizer.zero_grad()


    if args.empty_cache:
        torch.cuda.empty_cache()
    step = start_step
    i = start_step
    epoch = 1
    coordinator.print_on_master("\nStart saving model checkpoint with running states")
    timer.start()
    save_checkpoint(
        save_dir=args.save_dir,
        booster=booster,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=1,
        step=i + 1,
        batch_size=args.batch_size,
        coordinator=coordinator,
    )
    timer.end()
    coordinator.print_on_master(
        f"Saved checkpoint at epoch {epoch} step {step + 1} at folder {args.save_dir} with {timer.duration}s"
    )

    optimizer.zero_grad()


if __name__ == "__main__":
    main()
