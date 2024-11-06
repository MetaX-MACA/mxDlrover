# 2024-Modified by MetaX Integrated Circuits (Shanghai)Co., Ltd.All Rights Reserved.
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

import unittest
import os
from typing import Dict

from dlrover.python.master.elastic_training.net_topology import (
    DefaultTopologyQuerier,
    FileTopologyQuerier,
    DpTopologySorter,
    NodeTopologyMeta,
)


class NetTopologyTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_dp_topology_sorter(self):
        sw_querier = DefaultTopologyQuerier()
        sorter = DpTopologySorter()
        nodes: Dict[int, NodeTopologyMeta] = {}
        node_num = 10
        for i in range(node_num):
            node_ip = f"192.168.0.{i}"
            asw, psw = sw_querier.query(node_ip)
            node = NodeTopologyMeta(
                node_id=i,
                node_rank=i,
                process_num=8,
                node_ip=node_ip,
                asw=asw,
                psw=psw,
            )
            nodes[i] = node
        sorted_nodes = sorter.sort(nodes)
        node_ranks = list(sorted_nodes.keys())
        self.assertListEqual(node_ranks, list(range(node_num)))

        for node in nodes.values():
            asw_index = node.node_id % 3
            node.asw = f"asw-{asw_index}"

        sorted_nodes = sorter.sort(nodes)
        node_ranks = list(sorted_nodes.keys())
        expected_ranks = [0, 3, 6, 9, 1, 4, 7, 2, 5, 8]
        self.assertListEqual(node_ranks, expected_ranks)

        # sw_file_querier = FileTopologyQuerier('data/topo_config.txt')
        if ("dlrover/python/tests" in os.getcwd()):
            sw_file_querier = FileTopologyQuerier('data/topo_config.txt')
        else:
            sw_file_querier = FileTopologyQuerier(
                'dlrover/python/tests/data/topo_config.txt'
            )

        for node in nodes.values():
            asw, psw = sw_file_querier.query(node.node_ip)
            node.asw = asw
            node.psw = psw

        sorted_nodes = sorter.sort(nodes)
        node_ranks = list(sorted_nodes.keys())
        expected_ranks = [0, 4, 8, 1, 5, 9, 2, 6, 3, 7]
        self.assertListEqual(node_ranks, expected_ranks)
