# Copyright 2024 The DLRover Authors. All rights reserved.
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

from dlrover.python.scheduler.kubernetes import k8sClient
from dlrover.python.common.log import default_logger as logger

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class NodeTopologyMeta(object):
    node_id: int = 0
    node_rank: int = 0
    process_num: int = 0
    node_ip: str = ""
    asw: str = ""
    psw: str = ""


class TopologyQuerier(metaclass=ABCMeta):
    @abstractmethod
    def query(self, node_ip) -> Tuple[str, str]:
        """Query the asw and psw id of a node by the IP."""
        pass


class TopologySorter(metaclass=ABCMeta):
    @abstractmethod
    def sort(
        self, nodes: Dict[int, NodeTopologyMeta]
    ) -> Dict[int, NodeTopologyMeta]:
        """Query the asw and psw id of a node by the IP."""
        pass


class DefaultTopologyQuerier(TopologyQuerier):
    def query(self, node_ip) -> Tuple[str, str]:
        return "", ""


class FileTopologyQuerier(TopologyQuerier):
    def __init__(self, filename):
        self.node_sw_config = {}
        self._read_configfile(filename)

    def _read_configfile(self, filename):
        try:
            logger.info("topology-configfile Data:")
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        asw, psw, ip = parts
                        self.node_sw_config[ip] = (asw, psw)
                        logger.info(f"ASW: {asw}, PSW: {psw}, IP Address: {ip}")
        except FileNotFoundError as e:
            logger.error("Failed to get topology_configfile, reason: {e}\n")

    def query(self, node_ip) -> Tuple[str, str]:
        try:
            return self.node_sw_config[node_ip]
        except KeyError:
            return "", ""


class ConfigmapTopologyQuerier(TopologyQuerier):
    def __init__(self, namespace):
        self.node_sw_config = {}
        self._read_configmap(namespace)

    def _read_configmap(self, namespace):
        self._k8s_client = k8sClient.singleton_instance(namespace)
        try:
            configmap = self._k8s_client.get_configmap("node-topology-config")
            logger.info("configmap node-topology-config Data:")
            lines = configmap.data['topology_config'].splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) == 3:
                    asw, psw, ip = parts
                    self.node_sw_config[ip] = (asw, psw)
                    logger.info(f"ASW: {asw}, PSW: {psw}, IP Address: {ip}")
        except client.ApiException as e:
            logger.error("Failed to get topology-configmap, reason: {e}\n")

    def query(self, node_ip) -> Tuple[str, str]:
        try:
            return self.node_sw_config[node_ip]
        except KeyError:
            return "", ""


class DpTopologySorter(TopologySorter):
    """
    The sorter places the nodes under an asw (access switch) together in
    the list of nodes. In allreduce communication, the communication packets
    between nodes with continuous ranks under an asw will not pass the psw.

    """

    def sort(
        self, nodes: Dict[int, NodeTopologyMeta]
    ) -> Dict[int, NodeTopologyMeta]:
        asw_nodes: Dict[str, List[NodeTopologyMeta]] = {}
        rank0_node = next(iter(nodes.values()))
        rank0_asw = rank0_node.asw
        for _, meta in nodes.items():
            asw_nodes.setdefault(meta.asw, [])
            asw_nodes[meta.asw].append(meta)

        sorted_nodes: Dict[int, NodeTopologyMeta] = OrderedDict()
        asw0_nodes = asw_nodes.pop(rank0_asw, [])
        for node_meta in asw0_nodes:
            sorted_nodes[node_meta.node_rank] = node_meta

        for node_metas in asw_nodes.values():
            for node_meta in node_metas:
                sorted_nodes[node_meta.node_rank] = node_meta
        return sorted_nodes
