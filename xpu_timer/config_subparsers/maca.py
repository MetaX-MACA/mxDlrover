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

import textwrap

from . import BaseBuildRender


class BuildRender(BaseBuildRender):
    @classmethod
    def add_arguments(cls, parser):
        BaseBuildRender.add_arguments(parser)

    def __post_init__(self):
        if self.args.sdk_path is None:
            self.args.sdk_path = "/opt/maca"

        delete_packate = ",".join(f"//xpu_timer/{non_target}/..." for non_target in self.args.non_target)
        self.bazelrc_config.append(f"build --deleted_packages={delete_packate}")

    def rend_config_bzl(self):
        maca_config = textwrap.dedent(
            """
        XPU_TIMER_CONFIG = struct(
            linkopt = [
                "-L{cuda_path}/lib",
                "-lruntime_cu",
            ],
            copt = [
                "-DXPU_MACA",
            ],
            deps = ["@maca//:maca_headers"],
            py_bin = [
                "//xpu_timer/maca:libparse_params.so",
                "//xpu_timer/maca:intercepted.sym.default",
            ],
            gen_symbol = ["//xpu_timer/maca:gen_nvidia_symbols.py"],
            timer_deps = ["//xpu_timer/maca:maca_timer"],
            hook_deps = ["//xpu_timer/maca:maca_hook"],
        )

        """
        )

        self.xpu_timer_config.append(maca_config.format(cuda_path=self.sdk_path))
        return "\n".join(self.xpu_timer_config)

    def rend_bazelrc(self):
        self.bazelrc_config.append(f"build --action_env=MACA_HOME=/opt/maca --action_env=CUDA_HOME=/opt/maca/tools/cu-bridge")
        return "\n".join(self.bazelrc_config)

    def setup_files(self):
        with open("WORKSPACE.template") as f:
            workspace = f.read()

        deps = textwrap.dedent(
            """
            load("//third_party/maca:maca_workspace.bzl", "maca_workspace")
            maca_workspace()
            """
        )
        return workspace + deps

    def setup_platform_version(self):
        version = None
        path = f"{self.sdk_path}/Version.txt"
        pattern = "Version:"
        with open(path) as f:
            for line in f:
                if line.startswith(pattern):
                    version = line.split(pattern)[-1]
                    break
        if version is None:
            raise ValueError("Cannot found version")

        version = version.split(".")
        major = version[0]
        minor = version[1]
        return f"maca{major}{minor}", "MACA"
