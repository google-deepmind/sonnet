package(default_visibility = ["//sonnet:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

snt_py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        ":once",
        ":utils",
        # six
        # tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    deps = [
        ":base",
        ":test_utils",
        # tensorflow
        # wrapt
    ],
)

snt_py_library(
    name = "reshape",
    srcs = ["reshape.py"],
    deps = [
        ":base",
        # numpy
        # tensorflow
    ],
)

snt_py_test(
    name = "reshape_test",
    srcs = ["reshape_test.py"],
    deps = [
        ":reshape",
        ":test_utils",
        # absl/testing:parameterized
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "conv",
    srcs = ["conv.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":pad",
        ":utils",
        # numpy
        # six
        # tensorflow
    ],
)

snt_py_test(
    name = "conv_test",
    srcs = ["conv_test.py"],
    deps = [
        ":conv",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":utils",
        # numpy
        # six
        # tensorflow
    ],
)

snt_py_test(
    name = "depthwise_conv_test",
    srcs = ["depthwise_conv_test.py"],
    deps = [
        ":depthwise_conv",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "conv_transpose",
    srcs = ["conv_transpose.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":utils",
        # numpy
        # tensorflow
    ],
)

snt_py_test(
    name = "conv_transpose_test",
    srcs = ["conv_transpose_test.py"],
    deps = [
        ":conv_transpose",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "deferred",
    srcs = ["deferred.py"],
    deps = [
        ":base",
    ],
)

snt_py_test(
    name = "deferred_test",
    srcs = ["deferred_test.py"],
    deps = [
        ":base",
        ":deferred",
        ":test_utils",
        # tensorflow
    ],
)

snt_py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    deps = [
        # numpy
        # six
        # tensorflow
    ],
)

snt_py_test(
    name = "initializers_test",
    srcs = ["initializers_test.py"],
    deps = [
        ":initializers",
        ":test_utils",
        # absl/testing:parameterized
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "linear",
    srcs = ["linear.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "linear_test",
    srcs = ["linear_test.py"],
    gpu = False,  # TODO(tomhennigan) Flip me!
    deps = [
        ":linear",
        ":test_utils",
        # absl/testing:parameterized
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "batch_norm",
    srcs = ["batch_norm.py"],
    deps = [
        ":base",
        ":initializers",
        ":moving_averages",
        ":once",
        ":utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "batch_norm_test",
    srcs = ["batch_norm_test.py"],
    deps = [
        "test_utils",
        ":batch_norm",
        ":initializers",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "pad",
    srcs = ["pad.py"],
    deps = [
        ":utils",
    ],
)

snt_py_test(
    name = "pad_test",
    srcs = ["pad_test.py"],
    deps = [
        ":pad",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        ":base",
        # six
    ],
)

snt_py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    py3_tags = ["noasan"],
    deps = [
        ":metrics",
        # absl/testing:absltest
    ],
)

snt_py_library(
    name = "once",
    srcs = ["once.py"],
    deps = [":utils"],
)

snt_py_test(
    name = "once_test",
    srcs = ["once_test.py"],
    py3_tags = ["noasan"],
    deps = [
        ":once",
        # absl/testing:absltest
        # absl/testing:parameterized
    ],
)

snt_py_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    deps = [
        ":base",
        ":conv",
        ":initializers",
        ":linear",
        ":once",
        ":utils",
        # six
        # tensorflow
    ],
)

snt_py_test(
    name = "recurrent_test",
    srcs = ["recurrent_test.py"],
    deps = [
        ":initializers",
        ":recurrent",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "regularizers",
    srcs = ["regularizers.py"],
    deps = [
        ":utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "regularizers_test",
    srcs = ["regularizers_test.py"],
    deps = [
        ":regularizers",
        ":test_utils",
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "sequential",
    srcs = ["sequential.py"],
    deps = [
        ":base",
    ],
)

snt_py_test(
    name = "sequential_test",
    srcs = ["sequential_test.py"],
    deps = [
        ":sequential",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "adam",
    srcs = ["adam.py"],
    deps = [
        ":base",
        ":once",
        ":optimizer_utils",
        ":utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "adam_test",
    srcs = ["adam_test.py"],
    deps = [
        ":adam",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "momentum",
    srcs = ["momentum.py"],
    deps = [
        ":base",
        ":once",
        ":optimizer_utils",
        ":utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "momentum_test",
    srcs = ["momentum_test.py"],
    deps = [
        ":momentum",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    deps = [
        ":base",
        ":once",
        ":optimizer_utils",
        ":utils",
        # six
        # tensorflow
    ],
)

snt_py_test(
    name = "rmsprop_test",
    srcs = ["rmsprop_test.py"],
    deps = [
        ":rmsprop",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "sgd",
    srcs = ["sgd.py"],
    deps = [
        ":base",
        ":optimizer_utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "sgd_test",
    srcs = ["sgd_test.py"],
    deps = [
        ":sgd",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
)

snt_py_library(
    name = "axis_norm",
    srcs = ["axis_norm.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":utils",
        # six
        # tensorflow
    ],
)

snt_py_test(
    name = "axis_norm_test",
    srcs = ["axis_norm_test.py"],
    deps = [
        ":axis_norm",
        ":initializers",
        ":test_utils",
        # absl/testing:parameterized
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "group_norm",
    srcs = ["group_norm.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":utils",
        # six
        # tensorflow
    ],
)

snt_py_test(
    name = "group_norm_test",
    srcs = ["group_norm_test.py"],
    deps = [
        ":group_norm",
        ":initializers",
        ":test_utils",
        # absl/testing:parameterized
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "moving_averages",
    srcs = ["moving_averages.py"],
    deps = [
        ":metrics",
        ":once",
        # tensorflow
    ],
)

snt_py_test(
    name = "moving_averages_test",
    srcs = ["moving_averages_test.py"],
    deps = [
        ":moving_averages",
        ":test_utils",
        # absl/testing:parameterized
        # tensorflow
    ],
)

snt_py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":initializers",
        # six
        # tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":initializers",
        ":test_utils",
        ":utils",
        # absl/testing:parameterized
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "dropout",
    srcs = ["dropout.py"],
    deps = [
        ":base",
        # tensorflow
    ],
)

snt_py_test(
    name = "dropout_test",
    srcs = ["dropout_test.py"],
    deps = [
        ":dropout",
        ":test_utils",
        # absl/testing:parameterized
        # numpy
        # tensorflow
    ],
)

snt_py_library(
    name = "bias",
    srcs = ["bias.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "bias_test",
    srcs = ["bias_test.py"],
    deps = [
        ":bias",
        ":test_utils",
        # tensorflow
    ],
)

snt_py_library(
    name = "replicator",
    srcs = ["replicator.py"],
    deps = [
        # tensorflow
    ],
)

snt_py_test(
    name = "replicator_test",
    srcs = ["replicator_test.py"],
    deps = [
        ":replicator",
        ":test_utils",
        # tensorflow
    ],
)

snt_py_library(
    name = "embed",
    srcs = ["embed.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        # tensorflow
    ],
)

snt_py_test(
    name = "embed_test",
    srcs = ["embed_test.py"],
    deps = [
        ":embed",
        ":initializers",
        ":test_utils",
        # tensorflow
    ],
)
