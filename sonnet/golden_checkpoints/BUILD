package(default_visibility = ["//sonnet:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

load("//sonnet/src:build_defs.bzl", "snt_py_test")

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":goldens",
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "goldens",
    srcs = ["goldens.py"],
    deps = [
        # numpy dep,
        # six dep,
        "//sonnet",
        # tensorflow dep,
    ],
)

snt_py_test(
    name = "goldens_test",
    srcs = ["goldens_test.py"],
    data = glob(["*/**"]),
    deps = [
        ":goldens",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # six dep,
        "//sonnet",
        "//sonnet/src:test_utils",
        # tensorflow dep,
        # wrapt dep,
    ],
)
