package(default_visibility = ["//sonnet:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

snt_py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:linear",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "mlp_test",
    srcs = ["mlp_test.py"],
    deps = [
        ":mlp",
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "cifar10_convnet",
    srcs = ["cifar10_convnet.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:batch_norm",
        "//sonnet/src:conv",
        "//sonnet/src:linear",
        "//sonnet/src:moving_averages",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "cifar10_convnet_test",
    timeout = "long",
    srcs = ["cifar10_convnet_test.py"],
    deps = [
        ":cifar10_convnet",
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "vqvae",
    srcs = ["vqvae.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:initializers",
        "//sonnet/src:moving_averages",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "vqvae_test",
    srcs = ["vqvae_test.py"],
    deps = [
        ":vqvae",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)
