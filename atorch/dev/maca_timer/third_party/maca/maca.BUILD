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
    ],
    linkopts = ["-lruntime_cu"],
)
