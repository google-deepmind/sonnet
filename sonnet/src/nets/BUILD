load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

package(default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

snt_py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:initializers",
        "//sonnet/src:linear",
        # pip: tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
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
        "//sonnet/src:initializers",
        "//sonnet/src:linear",
        "//sonnet/src:types",
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
        "//sonnet/src:types",
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
        # pip: tree
    ],
)

snt_py_library(
    name = "resnet",
    srcs = ["resnet.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:batch_norm",
        "//sonnet/src:conv",
        "//sonnet/src:initializers",
        "//sonnet/src:linear",
        "//sonnet/src:pad",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "resnet_test",
    srcs = ["resnet_test.py"],
    deps = [
        ":resnet",
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)
