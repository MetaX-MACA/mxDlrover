# 2024-Modified by MetaX Integrated Circuits (Shanghai)Co., Ltd.All Rights Reserved.
# Copyright 2022 The DLRover Authors. All rights reserved.
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
import time

from dlrover.python.common.singleton import Singleton
from dlrover.python.scheduler.kubernetes import k8sClient
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.constants import NodeType

class DragonflyV2TopoManager(Singleton):
    def __init__(self, namespace, jobname):
        """
        DragonflyV2TopoManager manager dragonfly topo, uses to assign pod or get topo info

        Args:
            namespace: The name of the Kubernetes namespace where DLRover
                pods will be created.
        """

        self._dragonfly_enable = False
        self._dragonfly_per_group_num = 0
        self._dragonfly_topo_key= ""
        self._job_name = jobname

        self._k8s_client = k8sClient.singleton_instance(namespace)
        self._job = self._retry_to_get_job()
        if not self._job:
            raise ValueError(f"Cannot get the training job {self._job_name}.")

        self._init_dragonfly_parm()

    def _retry_to_get_job(self):
        for _ in range(3):
            job = self._k8s_client.get_custom_resource(
                name=self._job_name,
                group="elastic.iml.github.io",
                version="v1alpha1",
                plural="elasticjobs",
            )
            if job:
                return job
            else:
                time.sleep(5)
        return None

    def _init_dragonfly_parm(self):
        worker_spec = self._job["spec"]["replicaSpecs"][NodeType.WORKER]
        self._config_worker_num = worker_spec.get("replicas", 0)
        worker_metadata = worker_spec["template"].get("metadata", None)
        if worker_metadata != None:
            gpu_group_size = int(
                worker_metadata["annotations"].get("metax-tech.com/gpu-group-size", "0")
            )
            gpu_per_node :int = 0
            for container in worker_spec["template"]["spec"]["containers"]:
                if None == container.get("resources"):
                    continue
                gpu_per_node = int(container["resources"]["limits"].get(
                    "metax-tech.com/gpu", "0"
                ))
                if gpu_per_node != 0:
                    break
            if 0 != gpu_group_size and 0 != gpu_per_node:
                self._dragonfly_per_group_num = gpu_group_size // gpu_per_node
                self._dragonfly_enable = True

            logger.info(
                "dragonfly 2.0 node_group_size %d, gpu_group_size %d, gpu_per_node %d",
                self._dragonfly_per_group_num,
                gpu_group_size,
                gpu_per_node,
            )

    def dragonfly_enable(self):
        """
            return whether dragonfly is enabled
        """

        return self._dragonfly_enable

    def get_groups(self, node_id):
        """Get all node_ids of the group according to the node_id

        Args:
            node_id: int like 0,1,2"

        return:
            groups: like [0,1,2,3], [0,1], [4,5]
        """

        start_index = (node_id // self._dragonfly_per_group_num) * self._dragonfly_per_group_num
        return [i for i in range(start_index, start_index + self._dragonfly_per_group_num)]
