# limitations under the License.

import os
import unittest

from dlrover.python.master.dragonfly.dragonfly_topo_v2 import DragonflyV2TopoManager
from dlrover.python.tests.test_utils import mock_k8s_client, get_dragonfly_job

class DragonflyV2TopoManagerUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["POD_IP"] = "127.0.0.1"
        self.k8s_client = mock_k8s_client()

    def test_check_dragonfly_configuration(self):
        stub = self.k8s_client.get_custom_resource 
        self.k8s_client.get_custom_resource = get_dragonfly_job
        topo_manager = DragonflyV2TopoManager.singleton_instance("elasticjob-sample", "default")
        self.assertEqual(
            topo_manager.dragonfly_enable(),
            True,
        )
        self.assertTrue(topo_manager.get_groups(1) == [0, 1])
        self.k8s_client.get_custom_resource = stub