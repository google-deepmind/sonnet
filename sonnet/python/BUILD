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
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        # tensorflow dep,
    ],
)

py_library(
    name = "basic",
    srcs = ["modules/basic.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":nest",
        ":util",
        # tensorflow dep,
    ],
)

py_library(
    name = "custom_getters",
    srcs = [
        "custom_getters/__init__.py",
        "custom_getters/context.py",
        "custom_getters/non_trainable.py",
        "custom_getters/restore_initializer.py",
        "custom_getters/stop_gradient.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
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
        "modules/layer_norm.py",
        "modules/nets/__init__.py",
        "modules/nets/alexnet.py",
        "modules/nets/convnet.py",
        "modules/nets/dilation.py",
        "modules/nets/mlp.py",
        "modules/pondering_rnn.py",
        "modules/residual.py",
        "modules/rnn_core.py",
        "modules/scale_gradient.py",
        "modules/sequential.py",
        "modules/spatial_transformer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":basic",
        ":util",
        # tensorflow dep,
    ],
)

py_library(
    name = "util",
    srcs = ["modules/util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "nest",
    srcs = ["ops/nest.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "initializers",
    srcs = ["ops/initializers.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "ops",
    srcs = [
        "__init__.py",
        "ops/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":initializers",
        ":nest",
        # tensorflow dep,
    ],
)

module_tests = [
    ("alexnet_test", "nets/", "large"),
    ("attention_test", "", "small"),
    ("base_test", "", "small"),
    ("basic_test", "", "small"),
    ("basic_rnn_test", "", "medium"),
    ("batch_norm_test", "", "small"),
    ("layer_norm_test", "", "small"),
    ("block_matrix_test", "", "small"),
    ("clip_gradient_test", "", "small"),
    ("convnet_test", "nets/", "small"),
    ("conv_test", "", "large"),
    ("dilation_test", "nets/", "small"),
    ("embed_test", "", "small"),
    ("gated_rnn_test", "", "medium"),
    ("mlp_test", "nets/", "small"),
    ("pondering_rnn_test", "", "small"),
    ("rnn_core_test", "", "small"),
    ("residual_test", "", "small"),
    ("scale_gradient_test", "", "small"),
    ("sequential_test", "", "small"),
    ("spatial_transformer_test", "", "small"),
    ("util_test", "", "small"),
]

[py_test(
    name = test_name,
    size = test_size,
    srcs = ["modules/%s%s.py" % (test_subdir, test_name)],
    srcs_version = "PY2AND3",
    deps = [
        # mox dep,
        # numpy dep,
        "//sonnet",
        "//sonnet/testing/parameterized",
        # tensorflow dep,
    ],
) for test_name, test_subdir, test_size in module_tests]

py_test(
    name = "conv_gpu_test",
    size = "small",
    srcs = ["modules/conv_gpu_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "nomsan",  # Relies on the precompiled library libcuda, which is incompatible with MSan.
        "notap",
        "requires-gpu-sm35",
    ],
    deps = [
        "//sonnet",
        "//sonnet/testing/parameterized",
    ],
)

py_test(
    name = "nest_test",
    size = "small",
    srcs = ["ops/nest_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nest",
        "//sonnet/testing/parameterized",
    ],
)

py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["ops/initializers_test.py"],
    data = ["//sonnet/python/ops/testdata:restore_initializer_test_checkpoint"],
    srcs_version = "PY2AND3",
    deps = [
        ":initializers",
        # numpy dep,
        "//sonnet",
        # tensorflow dep,
    ],
)

custom_getters_tests = [
    (
        "context_test",
        "small",
        [],
    ),
    (
        "non_trainable_test",
        "small",
        [],
    ),
    (
        "restore_initializer_test",
        "small",
        [],
    ),
    (
        "stop_gradient_test",
        "small",
        [],
    ),
]

[py_test(
    name = test_name,
    size = test_size,
    srcs = ["custom_getters/%s.py" % test_name],
    tags = test_tags,
    deps = [
        # numpy dep,
        "//sonnet",
        "//sonnet/testing/parameterized",
        # tensorflow dep,
    ],
) for test_name, test_size, test_tags in custom_getters_tests]
