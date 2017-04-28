# Description:
#   Examples for the Sonnet library.

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "examples",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":module_with_build_args",
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
        # Tensorflow IO dep,
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
    ],
)
