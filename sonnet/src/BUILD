load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

package(default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

snt_py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        ":once",
        ":types",
        ":utils",
        # pip: six
        # pip: tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    deps = [
        ":base",
        ":test_utils",
        # pip: numpy
        # pip: tensorflow
        # pip: wrapt
    ],
)

snt_py_library(
    name = "build",
    srcs = ["build.py"],
    deps = [
        # pip: tensorflow
        # pip: tree
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "build_test",
    srcs = ["build_test.py"],
    deps = [
        ":build",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_library(
    name = "reshape",
    srcs = ["reshape.py"],
    deps = [
        ":base",
        ":once",
        ":types",
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "reshape_test",
    srcs = ["reshape_test.py"],
    deps = [
        ":reshape",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "leaky_clip_by_value",
    srcs = ["leaky_clip_by_value.py"],
    deps = [
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "leaky_clip_by_value_test",
    srcs = ["leaky_clip_by_value_test.py"],
    deps = [
        ":leaky_clip_by_value",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "scale_gradient",
    srcs = ["scale_gradient.py"],
    deps = [
        ":types",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "scale_gradient_test",
    srcs = ["scale_gradient_test.py"],
    deps = [
        ":scale_gradient",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
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
        # pip: numpy
        # pip: six
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "conv_test",
    srcs = ["conv_test.py"],
    deps = [
        ":conv",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
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
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "depthwise_conv_test",
    srcs = ["depthwise_conv_test.py"],
    deps = [
        ":depthwise_conv",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "conv_transpose",
    srcs = ["conv_transpose.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":types",
        ":utils",
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "conv_transpose_test",
    srcs = ["conv_transpose_test.py"],
    deps = [
        ":conv_transpose",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "custom_getter",
    srcs = ["custom_getter.py"],
    deps = [
        ":base",
        # pip: contextlib2
        # pip: tensorflow
        # pip: tree
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "custom_getter_test",
    srcs = ["custom_getter_test.py"],
    deps = [
        ":base",
        ":custom_getter",
        ":test_utils",
        # pip: tensorflow
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
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    deps = [
        ":types",
        # pip: numpy
        # pip: six
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "initializers_test",
    srcs = ["initializers_test.py"],
    deps = [
        ":initializers",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: tensorflow
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
        # pip: tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "linear_test",
    srcs = ["linear_test.py"],
    deps = [
        ":linear",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "batch_norm",
    srcs = ["batch_norm.py"],
    deps = [
        ":base",
        ":initializers",
        ":metrics",
        ":moving_averages",
        ":once",
        ":types",
        ":utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "batch_norm_test",
    srcs = ["batch_norm_test.py"],
    deps = [
        ":batch_norm",
        ":initializers",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
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
        # pip: absl/testing:parameterized
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        ":base",
        ":once",
        # pip: six
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
        ":test_utils",
        # pip: tensorflow
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
    deps = [
        ":once",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
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
        ":types",
        ":utils",
        # pip: six
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "recurrent_test",
    srcs = ["recurrent_test.py"],
    deps = [
        ":initializers",
        ":recurrent",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_library(
    name = "regularizers",
    srcs = ["regularizers.py"],
    deps = [
        ":types",
        # pip: six
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "regularizers_test",
    srcs = ["regularizers_test.py"],
    deps = [
        ":regularizers",
        ":test_utils",
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "sequential",
    srcs = ["sequential.py"],
    deps = [":base"],
)

snt_py_test(
    name = "sequential_test",
    srcs = ["sequential_test.py"],
    deps = [
        ":sequential",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "axis_norm",
    srcs = ["axis_norm.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":types",
        ":utils",
        # pip: six
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "axis_norm_test",
    srcs = ["axis_norm_test.py"],
    deps = [
        ":axis_norm",
        ":initializers",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "group_norm",
    srcs = ["group_norm.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":types",
        ":utils",
        # pip: six
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "group_norm_test",
    srcs = ["group_norm_test.py"],
    deps = [
        ":group_norm",
        ":initializers",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "moving_averages",
    srcs = ["moving_averages.py"],
    deps = [
        ":metrics",
        ":once",
        ":types",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "moving_averages_test",
    srcs = ["moving_averages_test.py"],
    deps = [
        ":moving_averages",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":initializers",
        # pip: absl/logging
        # pip: six
        # pip: tabulate
        # pip: tensorflow
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
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        # pip: absl/testing:parameterized
        # pip: tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_library(
    name = "dropout",
    srcs = ["dropout.py"],
    deps = [
        ":base",
        ":types",
        ":utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "dropout_test",
    srcs = ["dropout_test.py"],
    deps = [
        ":dropout",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "bias",
    srcs = ["bias.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":types",
        ":utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "bias_test",
    srcs = ["bias_test.py"],
    deps = [
        ":bias",
        ":test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "embed",
    srcs = ["embed.py"],
    deps = [
        ":base",
        ":initializers",
        ":types",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "embed_test",
    srcs = ["embed_test.py"],
    deps = [
        ":embed",
        ":initializers",
        ":test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "batch_apply",
    srcs = ["batch_apply.py"],
    deps = [
        ":base",
        # pip: numpy
        # pip: tensorflow
        # pip: tree
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "batch_apply_test",
    srcs = ["batch_apply_test.py"],
    deps = [
        ":base",
        ":batch_apply",
        ":test_utils",
        # pip: tensorflow
        # pip: wrapt
    ],
)

snt_py_library(
    name = "mixed_precision",
    srcs = ["mixed_precision.py"],
    deps = [
        ":custom_getter",
        ":utils",
        # pip: contextlib2
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "mixed_precision_test",
    srcs = ["mixed_precision_test.py"],
    deps = [
        ":base",
        ":mixed_precision",
        ":test_utils",
        # pip: absl/testing:parameterized
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_library(
    name = "parallel_linear",
    srcs = ["parallel_linear.py"],
    deps = [
        ":base",
        ":initializers",
        ":once",
        ":utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "parallel_linear_test",
    srcs = ["parallel_linear_test.py"],
    deps = [
        ":base",
        ":linear",
        ":parallel_linear",
        ":test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
    deps = [
        # pip: numpy
        # pip: tensorflow
    ],
)
