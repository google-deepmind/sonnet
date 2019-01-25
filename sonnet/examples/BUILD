# Description:
#   Examples for the Sonnet library.

# Placeholder for internal Python version compatibility macro.

licenses(["notice"])  # Apache 2.0 License

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

py_library(
    name = "examples",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":brnn_ptb",
        ":mnist_mlp",
        ":module_with_build_args_lib",
        ":rmc_nth_farthest",
        ":rnn_shakespeare",
    ],
)

py_binary(
    name = "module_with_build_args",
    srcs = ["module_with_build_args.py"],
    default_python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":module_with_build_args_lib"],
)

py_library(
    name = "module_with_build_args_lib",
    srcs = ["module_with_build_args.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//sonnet",
        # tensorflow dep,
    ],
)

py_binary(
    name = "mnist_mlp",
    srcs = [
        "dataset_mnist_cifar10.py",
        "mnist_mlp.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        "//sonnet",
        # tensorflow dep,
    ],
)

py_binary(
    name = "rnn_shakespeare",
    srcs = [
        "dataset_shakespeare.py",
        "rnn_shakespeare.py",
    ],
    data = [
        "data/ts.test.txt",
        "data/ts.train.txt",
        "data/ts.valid.txt",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow IO dep,
        # numpy dep,
        "//sonnet",
        # tensorflow dep,
    ],
)

py_binary(
    name = "brnn_ptb",
    srcs = [
        "brnn_ptb.py",
        "ptb_reader.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # six dep,
        "//sonnet",
        "//sonnet/python:custom_getters",
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_binary(
    name = "rmc_learn_to_execute",
    srcs = [
        "learn_to_execute.py",
        "rmc_learn_to_execute.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # absl/flags dep,
        # numpy dep,
        # scipy dep,
        # six dep,
        "//sonnet",
        # tensorflow dep,
    ],
)

py_binary(
    name = "rmc_nth_farthest",
    srcs = [
        "dataset_nth_farthest.py",
        "rmc_nth_farthest.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # absl/flags dep,
        # numpy dep,
        # scipy dep,
        # six dep,
        "//sonnet",
        # tensorflow dep,
    ],
)

py_test(
    name = "rnn_shakespeare_test",
    size = "large",
    srcs = ["rnn_shakespeare_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "nomsan",  # takes too long with MSAN
        "notsan",  # takes too long with TSAN
    ],
    deps = [
        ":rnn_shakespeare",
        # tensorflow dep,
    ],
)

py_test(
    name = "brnn_ptb_test",
    size = "large",
    srcs = ["brnn_ptb_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":brnn_ptb",
        # tensorflow dep,
    ],
)

py_test(
    name = "rmc_nth_farthest_test",
    size = "large",
    srcs = ["rmc_nth_farthest_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":rmc_nth_farthest",
        "//sonnet",
        # tensorflow dep,
    ],
)

py_test(
    name = "rmc_learn_to_execute_test",
    size = "large",
    srcs = [
        "rmc_learn_to_execute_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":rmc_learn_to_execute",
        "//sonnet",
        # tensorflow dep,
    ],
)
