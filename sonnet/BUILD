package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

py_library(
    name = "sonnet",
    srcs = ["__init__.py"],
    deps = [
        ":initializers",
        ":nets",
        ":optimizers",
        ":pad",
        "//sonnet/src:axis_norm",
        "//sonnet/src:base",
        "//sonnet/src:batch_norm",
        "//sonnet/src:conv",
        "//sonnet/src:conv_transpose",
        "//sonnet/src:deferred",
        "//sonnet/src:dropout",
        "//sonnet/src:group_norm",
        "//sonnet/src:linear",
        "//sonnet/src:metrics",
        "//sonnet/src:once",
        "//sonnet/src:recurrent",
        "//sonnet/src:reshape",
        "//sonnet/src:sequential",
    ],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//sonnet/src:initializers",
    ],
)

py_library(
    name = "nets",
    srcs = ["nets.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//sonnet/src/nets:cifar10_convnet",
        "//sonnet/src/nets:mlp",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//sonnet/src:adam",
        "//sonnet/src:momentum",
        "//sonnet/src:rmsprop",
        "//sonnet/src:sgd",
    ],
)

py_library(
    name = "pad",
    srcs = ["pad.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//sonnet/src:pad",
    ],
)
