# 2024-Modified by MetaX Integrated Circuits (Shanghai)Co., Ltd.All Rights Reserved.
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

import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    CheckpointConfig,
    CheckpointEvent,
    CheckpointEventType,
    ColossalAICheckpointSaver,
)

from .engine import CheckpointEngine, timer


class ColossalAICheckpointEngine(CheckpointEngine):
    """
    The checkpoint engine synchronously writes the state dict of
    Megatron-LM model and optimizer into
    the shared memory and notify the agent in main process to
    asynchronously save the state dict from the shared memory into
    the storage.
    """

    def __init__(
        self,
        checkpoint_dir,
        storage,
        comm_backend="",
        pp_rank=0,
        pp_size=1,
    ):
        # only tp=0, dp=0 will save shard
        self._pp_world_size = pp_size
        self._pp_rank = pp_rank
        device_number = dist.get_world_size() // env_utils.get_local_world_size()
        global_shard_num = self.get_global_shard_num()
        self._local_shard_num = max(1, global_shard_num // device_number)
        super().__init__(
            checkpoint_dir,
            storage,
            comm_backend,
            replica_count=0,
            local_shard_id=self._pp_rank % self._local_shard_num,
        )

    def get_saving_ranks(self):
        """
        Get the ranks which need to save the sharding state dict into
        the memory.
        """
        world_size = dist.get_world_size()
        itererval = world_size // self._pp_world_size
        save_ranks = []
        for i in range(world_size):
            if i % itererval == 0:
                save_ranks.append(i)
        return save_ranks

    @timer
    def save_to_memory(self, step, state_dict, paths):
        """
        Synchronously Saves the state dict into the shared memory with the main
        process. If the agent in the main process is saving the shared memory
        into the storage, the method will skip to write the shared memory.
        Only local rank 0 save the state dict into the memory because the
        state dict is replicated across all ranks.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the key is a category in
                ["model_states", "optim_states"] of the state dict and
                the value is the path of storage to save.
        """
        conf = CheckpointConfig(step=step, paths=paths)
        return self.save_state_dict_to_memory(state_dict, conf)

    @timer
    def save_to_storage(self, step, state_dict, paths):
        """
        Asynchonously saves the state dict into the storage. It synchonously
        saves the state dict into the shared memory and put the path
        into a shared queue. The agent in the main process waits for the queue
        for save the state dict in the shared memory into the storage.
        Only rank 0 saves the state dict into the storage.

        Args:
            step (int): the iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            paths (dict): the key is a category in
                ["model_states", "optim_states"] of the state dict and
                the value is the path of storage to save.
        """
        succeed = True
        if step > self._cached_step:
            succeed = self.save_to_memory(step, state_dict, paths)

        if dist.is_initialized():
            dist.barrier()

        # Only local rank 0 to notify the saving event to the agent.
        if len(state_dict) == 0:
            return
        if succeed and self._local_rank == 0:
            event = CheckpointEvent(type=CheckpointEventType.SAVE, step=step)
            self._event_queue.put(event)

    def get_local_shard_num(self):
        global_shard_num = self.get_global_shard_num()
        return min(self._local_shard_num, global_shard_num)

    def get_global_shard_num(self):
        return self._pp_world_size

    def get_saver_class(self):
        return ColossalAICheckpointSaver

    def load(self, resume_path=""):
        """
        The method firstly try to load the state dict from the shared memory.
        If there is no state dict in the shared memory, the method will
        load the state dict from the storage.

        Returns:
            A dict.
        """
        state_dict = self.get_state_dict_from_memory()
        if state_dict:
            return state_dict
        return {}
