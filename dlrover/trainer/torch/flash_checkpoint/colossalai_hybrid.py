# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any, Dict, Tuple, Union
import sys

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.constants import CheckpointConstant
from dlrover.python.common.singleton import Singleton
from dlrover.python.common.storage import PosixDiskStorage
from .checkpointer import StorageType
from .colossalai_engine import ColossalAICheckpointEngine

try:
    from colossalai.booster import Booster
    from colossalai.cluster import DistCoordinator
    from colossalai.checkpoint_io import CheckpointIO, HybridParallelCheckpointIO
    from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroCheckpointIO
except ImportError:
    logger.warning("Please check the colossalai.checkpoint_io exists.")

torch_native_save = torch.save
torch_native_load = torch.load


class ColossalAICheckpointer(Singleton):
    def __init__(
        self, checkpoint_dir, storage=None, comm_backend="", pp_rank=0, pp_size=1
    ):
        self.state_dict = {}
        self.paths = {}
        self.checkpoint_dir = checkpoint_dir
        self.storage = PosixDiskStorage() if not storage else storage
        self.engine = ColossalAICheckpointEngine(
            checkpoint_dir=checkpoint_dir,
            storage=self.storage,
            comm_backend=comm_backend,
            pp_rank=pp_rank,
            pp_size=pp_size,
        )

    def save(self, state_dict, path: str):
        if not isinstance(path, str):
            torch_native_save(state_dict, path)
            return
        name = os.path.basename(path)
        self.state_dict[name] = state_dict
        self.paths[name] = path

    def load(self, path: str, **kwargs):
        def load_func(path):
            return torch_native_load(path, map_location="cpu")

        if not isinstance(path, str):
            return torch_native_load(path)

        name = os.path.basename(path)
        state_dict = self.engine.load(resume_path=path)
        if name in state_dict:
            return state_dict[name]
        else:
            return self.storage.read_state_dict(path, load_func)


def save_json(data: Dict[str, Any], file_path: Union[str, os.PathLike]) -> None:
    """
    Save as JSON format
    """
    with open(file=file_path, mode="w", encoding="utf-8") as fp:
        json.dump(data, fp=fp, ensure_ascii=False, indent=4)


def load_json(file_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load file in JSON format
    """
    with open(file=file_path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_checkpoint(
    save_dir: Union[str, os.PathLike],
    booster: Booster,
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    storage_type=StorageType.DISK,
) -> None:
    """
    Save model checkpoint, optimizer, LR scheduler and intermedidate running states.
    """

    if not isinstance(booster.checkpoint_io, HybridParallelCheckpointIO) and \
        not isinstance(booster.checkpoint_io, LowLevelZeroCheckpointIO):
        logger.warning("Only support HybridParallelCheckpointIO & LowLevelZeroCheckpointIO checkpoint")
        sys.exit()

    root_dir = save_dir
    save_dir = os.path.join(save_dir, f"step-{step}")
    os.makedirs(os.path.join(save_dir, "modeling"), exist_ok=True)
    if isinstance(booster.checkpoint_io, HybridParallelCheckpointIO):
        saver = ColossalAICheckpointer.singleton_instance(
            root_dir,
            pp_rank=booster.checkpoint_io.pp_rank,
            pp_size=booster.checkpoint_io.pp_size,
        )
    else:
        saver = ColossalAICheckpointer.singleton_instance(
            root_dir,
            pp_rank=0,
            pp_size=1
        )

    try:
        torch.save = saver.save
        booster.save_model(model, os.path.join(save_dir, "modeling"), shard=True)

        booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
        booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
        if storage_type == StorageType.MEMORY:
            saver.engine.save_to_memory(step, saver.state_dict, saver.paths)
        else:
            saver.engine.save_to_storage(step, saver.state_dict, saver.paths)
    finally:
        torch.save = torch_native_save
    saver.state_dict.clear()

    running_states = {
        "epoch": epoch,
        "step": step,
        "sample_start_index": step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))


def load_checkpoint(
    load_dir: Union[str, os.PathLike],
    booster: Booster,
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
) -> Tuple[int, int, int]:
    """
    Load model checkpoint, optimizer, LR scheduler and intermedidate running states.
    """

    if not isinstance(booster.checkpoint_io, HybridParallelCheckpointIO) and \
        not isinstance(booster.checkpoint_io, LowLevelZeroCheckpointIO):
        logger.warning("Only support HybridParallelCheckpointIO & LowLevelZeroCheckpointIO checkpoint")
        sys.exit()

    if isinstance(booster.checkpoint_io, HybridParallelCheckpointIO):
        checkpointer = ColossalAICheckpointer.singleton_instance(
            load_dir,
            pp_rank=booster.checkpoint_io.pp_rank,
            pp_size=booster.checkpoint_io.pp_size,
        )
    else:
        checkpointer = ColossalAICheckpointer.singleton_instance(
            load_dir,
            pp_rank=0,
            pp_size=1
        )

    dlrover_tracer_file = os.path.join(
        checkpointer.checkpoint_dir, CheckpointConstant.TRACER_FILE_NAME
    )
    step = checkpointer.storage.read(dlrover_tracer_file)
    load_dir = os.path.join(load_dir, f"step-{step}")

    # Update booster params states.
    try:
        torch.load = checkpointer.load
        booster.load_model(model=model, checkpoint=os.path.join(load_dir, "modeling"))
        booster.load_optimizer(
            optimizer=optimizer, checkpoint=os.path.join(load_dir, "optimizer")
        )
        booster.load_lr_scheduler(
            lr_scheduler=lr_scheduler, checkpoint=os.path.join(load_dir, "lr_scheduler")
        )
    finally:
        torch.load = torch_native_load

    running_states = load_json(file_path=os.path.join(load_dir, "running_states.json"))
    return (
        running_states["epoch"],
        running_states["step"],
        running_states["sample_start_index"],
    )
