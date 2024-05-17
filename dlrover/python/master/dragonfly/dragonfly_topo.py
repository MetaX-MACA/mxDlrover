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

from dlrover.python.common.singleton import Singleton
from dlrover.python.scheduler.kubernetes import k8sClient
from dlrover.python.common.log import default_logger as logger

from typing import Dict, List

from kubernetes import client
from kubernetes.client import V1Affinity
from distutils.util import strtobool

class DragonflyTopoManager(Singleton):
    def __init__(self, namespace):
        """
        DragonflyTopoManager manager dragonfly topo, uses to assign pod or get topo info

        Args:
            namespace: The name of the Kubernetes namespace where DLRover
                pods will be created.
        """

        self._dragonfly_enable = False
        self._dragonfly_per_group_num = 1
        self._dragonfly_topo_key= ""

        self._k8s_client = k8sClient.singleton_instance(namespace)
        try:
            configmap = self._k8s_client.get_configmap("dragonfly-config")
            logger.info("configmap Dragonfly Data:")
            for key, value in configmap.data.items():
                logger.info(f"{key}:{value}")

            self._init_dragonfly_parm(configmap)
        except client.ApiException as e:
            logger.error("Failed to get configmap, reason: {e}\n")

    def _init_dragonfly_parm(self, configmap):
        self._dragonfly_enable = strtobool(configmap.data.get("dragonfly_enable", "False"))
        self._dragonfly_per_group_num = int(configmap.data.get("dragonfly_per_group_num", "1"))
        self._dragonfly_topo_key = configmap.data.get("dragonfly_topo_key", "")
    
    def dragonfly_enable(self):
        """
            return whether dragonfly is enabled
        """

        return self._dragonfly_enable
    
    def generate_affinity(self, job_name, rank_id):
        """The affinity property is generated for the Pod 
            so that the Pod can be scheduled to the correct node group 

        Args:
            job_name: str like job-name
            rank_id: int like 0,1,2"
        """
        
        affinity = client.V1Affinity(
            pod_affinity=client.V1PodAffinity(
                required_during_scheduling_ignored_during_execution=[client.V1PodAffinityTerm(
                    label_selector=client.V1LabelSelector(
                        match_expressions=[client.V1LabelSelectorRequirement(
                            key=self._generate_label_key(),
                            operator="In",
                            values=[self._generate_label_value(job_name, rank_id)],
                        )],
                    ),
                    topology_key=self._dragonfly_topo_key
                )]
            ),
            pod_anti_affinity=client.V1PodAntiAffinity(
                required_during_scheduling_ignored_during_execution=[client.V1PodAffinityTerm(
                    label_selector=client.V1LabelSelector(
                        match_expressions=[client.V1LabelSelectorRequirement(
                            key=self._generate_label_key(),
                            operator="NotIn",
                            values=[self._generate_label_value(job_name, rank_id)],
                        )],
                    ),
                    topology_key=self._dragonfly_topo_key,
                )]
            )
        )

        return affinity
    
    def generate_label(self, job_name, rank_id):
        """The label is generated for the Pod 
            so that the Pod can be scheduled to the correct node group 

        Args:
            job_name: str like job-name
            rank_id: int like 0,1,2"
        """
        
        labels = {
            self._generate_label_key(): self._generate_label_value(job_name, rank_id)
        }
        
        return labels
    
    def get_groups(self, node_id):
        """Get all node_ids of the group according to the node_id

        Args:
            node_id: int like 0,1,2"

        return:
            groups: like [0,1,2,3], [0,1], [4,5]
        """

        start_index = (node_id // self._dragonfly_per_group_num) * self._dragonfly_per_group_num
        return [i for i in range(start_index, start_index + self._dragonfly_per_group_num)]

    def _generate_label_key(self):
        return "dlrover.dragonfly"
    
    def _generate_label_value(self, job_name, rank_id):
        return job_name + "-group-index-" + str(rank_id // self._dragonfly_per_group_num)

