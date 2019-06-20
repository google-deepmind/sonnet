package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

load("//sonnet/src:build_defs.bzl", "snt_py_library")

snt_py_library(
    name = "sonnet",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":distribute",
        ":initializers",
        ":nets",
        ":optimizers",
        ":pad",
        ":regularizers",
        "//sonnet/src:axis_norm",
        "//sonnet/src:base",
        "//sonnet/src:batch_apply",
        "//sonnet/src:batch_norm",
        "//sonnet/src:bias",
        "//sonnet/src:conv",
        "//sonnet/src:conv_transpose",
        "//sonnet/src:deferred",
        "//sonnet/src:dropout",
        "//sonnet/src:embed",
        "//sonnet/src:group_norm",
        "//sonnet/src:linear",
        "//sonnet/src:metrics",
        "//sonnet/src:once",
        "//sonnet/src:recurrent",
        "//sonnet/src:reshape",
        "//sonnet/src:sequential",
    ],
)

snt_py_library(
    name = "distribute",
    srcs = ["distribute.py"],
    deps = [
        "//sonnet/src:replicator",
    ],
)

snt_py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    deps = [
        "//sonnet/src:initializers",
    ],
)

snt_py_library(
    name = "nets",
    srcs = ["nets.py"],
    deps = [
        "//sonnet/src/nets:cifar10_convnet",
        "//sonnet/src/nets:mlp",
    ],
)

snt_py_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    deps = [
        "//sonnet/src:adam",
        "//sonnet/src:momentum",
        "//sonnet/src:rmsprop",
        "//sonnet/src:sgd",
    ],
)

snt_py_library(
    name = "pad",
    srcs = ["pad.py"],
    deps = [
        "//sonnet/src:pad",
    ],
)

snt_py_library(
    name = "regularizers",
    srcs = ["regularizers.py"],
    deps = [
        "//sonnet/src:regularizers",
    ],
)
