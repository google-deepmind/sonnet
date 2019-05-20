package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

py_binary(
    name = "simple_mnist",
    srcs = ["simple_mnist.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        # absl:app
        "//sonnet",
        # tensorflow
        # tensorflow_datasets
    ],
)
