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
    deps = [
        ":rnn_shakespeare",
    ],
)
