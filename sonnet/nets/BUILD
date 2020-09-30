load("//sonnet/src:build_defs.bzl", "snt_py_library")

package(default_visibility = ["//sonnet:__pkg__"])

licenses(["notice"])  # Apache 2.0 License

snt_py_library(
    name = "nets",
    srcs = [
        "__init__.py",
        "resnet.py",
    ],
    deps = [
        "//sonnet/src/nets:cifar10_convnet",
        "//sonnet/src/nets:mlp",
        "//sonnet/src/nets:resnet",
        "//sonnet/src/nets:vqvae",
    ],
)
