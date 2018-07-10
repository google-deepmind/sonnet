# Description:
#   Examples for the Sonnet library.

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

# Placeholder for internal Python version compatibility macro.

py_library(
    name = "examples",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":brnn_ptb",
        ":module_with_build_args",
        ":rmc_nth_farthest",
        ":rnn_shakespeare",
    ],
)

py_binary(
    name = "module_with_build_args",
    srcs = [
        "module_with_build_args.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
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
    srcs = [
        "rnn_shakespeare_test.py",
    ],
    srcs_version = "PY2AND3",
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
    srcs = [
        "rmc_nth_farthest_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":rmc_nth_farthest",
        "//sonnet",
        # tensorflow dep,
    ],
)
