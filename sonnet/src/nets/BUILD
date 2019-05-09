package(default_visibility = ["//sonnet:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

load("//sonnet/src:build_defs.bzl", "snt_py_test")

py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:linear",
        # tensorflow dep,
    ],
)

snt_py_test(
    name = "mlp_test",
    srcs = ["mlp_test.py"],
    deps = [
        ":mlp",
        # absl/testing:parameterized dep,
        "//sonnet/src:test_utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "cifar10_convnet",
    srcs = ["cifar10_convnet.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:batch_norm",
        "//sonnet/src:conv",
        "//sonnet/src:linear",
        "//sonnet/src:moving_averages",
        # tensorflow dep,
    ],
)

snt_py_test(
    name = "cifar10_convnet_test",
    srcs = ["cifar10_convnet_test.py"],
    deps = [
        ":cifar10_convnet",
        # absl/testing:parameterized dep,
        "//sonnet/src:test_utils",
        # tensorflow dep,
    ],
)
