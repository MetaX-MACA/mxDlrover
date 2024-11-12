def maca_workspace():
    native.new_local_repository(
        name = "maca",
        path = "/opt/maca",
        build_file = "//third_party/maca:maca.BUILD",
    )

