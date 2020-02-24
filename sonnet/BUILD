load("//sonnet/src:build_defs.bzl", "snt_py_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

snt_py_library(
    name = "sonnet",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":distribute",
        ":initializers",
        ":mixed_precision",
        ":optimizers",
        ":pad",
        ":regularizers",
        "//sonnet/nets",
        "//sonnet/src:axis_norm",
        "//sonnet/src:base",
        "//sonnet/src:batch_apply",
        "//sonnet/src:batch_norm",
        "//sonnet/src:bias",
        "//sonnet/src:build",
        "//sonnet/src:conv",
        "//sonnet/src:conv_transpose",
        "//sonnet/src:custom_getter",
        "//sonnet/src:deferred",
        "//sonnet/src:depthwise_conv",
        "//sonnet/src:dropout",
        "//sonnet/src:embed",
        "//sonnet/src:group_norm",
        "//sonnet/src:leaky_clip_by_value",
        "//sonnet/src:linear",
        "//sonnet/src:metrics",
        "//sonnet/src:moving_averages",
        "//sonnet/src:once",
        "//sonnet/src:recurrent",
        "//sonnet/src:reshape",
        "//sonnet/src:scale_gradient",
        "//sonnet/src:sequential",
        "//sonnet/src:utils",
    ],
)

snt_py_library(
    name = "distribute",
    srcs = ["distribute.py"],
    deps = [
        "//sonnet/src/distribute:batch_norm",
        "//sonnet/src/distribute:replicator",
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
    name = "mixed_precision",
    srcs = ["mixed_precision.py"],
    deps = [
        "//sonnet/src:mixed_precision",
    ],
)

snt_py_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    deps = [
        "//sonnet/src/optimizers:adam",
        "//sonnet/src/optimizers:momentum",
        "//sonnet/src/optimizers:rmsprop",
        "//sonnet/src/optimizers:sgd",
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
