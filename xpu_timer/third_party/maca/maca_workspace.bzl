load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def maca_workspace():
    native.new_local_repository(
        name = "maca",
        path = "/opt/maca",
        build_file = "//third_party/maca:maca.BUILD",
    )

    http_archive(
        name = "flash_attn_maca",
        build_file = "//third_party/maca:flash_attn.BUILD",
        strip_prefix = "flash-attention-fa2_pack_glm_mask",
        urls = ["https://github.com/intelligent-machine-learning/flash-attention/archive/fa2_pack_glm_mask.tar.gz"],
        sha256 = "1e2ab9fb7198c57f7f332e0b30f988bb47fb66220af56de6411d8744805e2e2b",
    )
