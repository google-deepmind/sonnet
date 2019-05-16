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
        # absl:app
        # absl/flags
        # absl/logging
        # tensorflow
    ],
)

py_library(
    name = "goldens",
    srcs = ["goldens.py"],
    deps = [
        # numpy
        # six
        "//sonnet",
        # tensorflow
    ],
)

snt_py_test(
    name = "goldens_test",
    srcs = ["goldens_test.py"],
    data = glob(["*/**"]),
    deps = [
        ":goldens",
        # absl/logging
        # absl/testing:absltest
        # absl/testing:parameterized
        # six
        "//sonnet",
        "//sonnet/src:test_utils",
        # tensorflow
        # wrapt
    ],
)
