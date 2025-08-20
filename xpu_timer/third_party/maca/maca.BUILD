package(
    default_visibility=["//visibility:public"]
)

cc_library(
    name = "maca_headers",
    hdrs = glob(["**/*.h", "**/*.hpp"]),
    includes = ["include",
        "tools/cu-bridge/include",
        "include/mxc",
        "include/mcc",
        "include/mcr",
        "include/mcblas",
        "include/common",
        "include/mcsparse",
        "include/mcsolver",
    ],
    linkopts = ["-lruntime_cu", "-L/opt/maca/lib"],
)

cc_library(
	name = "flash_attn_headers",
    hdrs = ["csrc/flash_attn/src/flash.h"],
    includes = ["csrc/flash_attn/src/"],
    deps = [
        "@torch//:torch_headers",
        "@maca//:maca_headers",
    ]
)
