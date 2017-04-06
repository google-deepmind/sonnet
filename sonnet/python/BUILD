# Description:
#   Sonnet: DeepMind abstractions for Neural Networks in TensorFlow.

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "base",
    srcs = [
        "modules/base.py",
    ],
    deps = [
        ":util",
    ],
)

py_library(
    name = "basic",
    srcs = ["modules/basic.py"],
    deps = [
        ":base",
        ":nest",
        ":util",
    ],
)

py_library(
    name = "modules",
    srcs = [
        "__init__.py",
        "modules/__init__.py",
        "modules/attention.py",
        "modules/basic_rnn.py",
        "modules/batch_norm.py",
        "modules/block_matrix.py",
        "modules/clip_gradient.py",
        "modules/conv.py",
        "modules/embed.py",
        "modules/experimental.py",
        "modules/gated_rnn.py",
        "modules/nets/__init__.py",
        "modules/nets/alexnet.py",
        "modules/nets/convnet.py",
        "modules/nets/dilation.py",
        "modules/nets/mlp.py",
        "modules/pondering_rnn.py",
        "modules/rnn_core.py",
        "modules/scale_gradient.py",
        "modules/sequential.py",
        "modules/spatial_transformer.py",
    ],
    deps = [
        ":base",
        ":basic",
        ":util",
    ],
)

py_library(
    name = "util",
    srcs = ["modules/util.py"],
    deps = [
    ],
)

py_library(
    name = "nest",
    srcs = ["ops/nest.py"],
    deps = [
    ],
)

py_library(
    name = "initializers",
    srcs = ["ops/initializers.py"],
    deps = [
    ],
)

py_library(
    name = "resampler",
    srcs = [
        "ops/resampler.py",
    ],
    data = [":ops/_resampler.so"],
)

py_library(
    name = "ops",
    srcs = [
        "__init__.py",
        "ops/__init__.py",
    ],
    deps = [
        ":initializers",
        ":nest",
        ":resampler",
    ],
)

cc_binary(
    name = "ops/_resampler.so",
    linkopts = ["-lm"],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//sonnet/cc:resampler_op_lib",
        "//sonnet/cc/kernels:resampler_op",
    ],
)

module_tests = [
    ("alexnet_test", "nets/", "large"),
    ("attention_test", "", "small"),
    ("base_test", "", "small"),
    ("basic_test", "", "small"),
    ("basic_rnn_test", "", "medium"),
    ("batch_norm_test", "", "small"),
    ("block_matrix_test", "", "small"),
    ("clip_gradient_test", "", "small"),
    ("convnet_test", "nets/", "small"),
    ("conv_test", "", "large"),
    ("dilation_test", "nets/", "small"),
    ("embed_test", "", "small"),
    ("experimental_test", "", "small"),
    ("gated_rnn_test", "", "medium"),
    ("mlp_test", "nets/", "small"),
    ("pondering_rnn_test", "", "small"),
    ("rnn_core_test", "", "small"),
    ("scale_gradient_test", "", "small"),
    ("sequential_test", "", "small"),
    ("spatial_transformer_test", "", "small"),
    ("util_test", "", "small"),
]

[py_test(
    name = test_name,
    size = test_size,
    srcs = ["modules/%s%s.py" % (test_subdir, test_name)],
    deps = [
        "//sonnet",
        # nose_parameterized dep
        # mock dep
    ],
) for test_name, test_subdir, test_size in module_tests]

py_test(
    name = "conv_gpu_test",
    size = "small",
    srcs = ["modules/conv_gpu_test.py"],
    tags = [
        "nomsan",  # Relies on the precompiled library libcuda, which is incompatible with MSan.
        "notap",
        "requires-gpu-sm35",
    ],
    deps = [
        # nose_parameterized dep
        "//sonnet",
    ],
)

py_test(
    name = "nest_test",
    size = "small",
    srcs = ["ops/nest_test.py"],
    deps = [
        ":nest",
        # nose_parameterized dep
    ],
)

py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["ops/initializers_test.py"],
    data = ["//sonnet/python/ops/testdata:restore_initializer_test_checkpoint"],
    deps = [
        ":initializers",
        ":modules",
        # numpy dep
    ],
)

py_test(
    name = "resampler_test",
    size = "small",
    srcs = ["ops/resampler_test.py"],
    tags = [
        "nomsan",  # Relies on the precompiled library libcuda, which is incompatible with MSan.
        "requires-gpu-sm35",
    ],
    deps = [
        # nose_parameterized dep
        ":resampler",
        "//sonnet",
    ],
)
