# Description:
#   Utilities for Sonnet.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

py_binary(
    name = "migrate_checkpoint",
    srcs = ["migrate_checkpoint.py"],
    default_python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//sonnet",  # build_cleaner:keep
        # tensorflow dep,
    ],
)
