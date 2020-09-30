# Description:
#   Differentiable Neural Computer

load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

package(default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

snt_py_library(
    name = "control",
    srcs = ["control.py"],
    deps = [
        "//sonnet/src:linear",
        "//sonnet/src:recurrent",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "control_test",
    srcs = ["control_test.py"],
    main = "control_test.py",
    deps = [
        ":control",
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: six
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_library(
    name = "read",
    srcs = ["read.py"],
    deps = [
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "read_test",
    srcs = ["read_test.py"],
    main = "read_test.py",
    deps = [
        ":read",
        # pip: numpy
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # pip: numpy
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    main = "util_test.py",
    deps = [
        ":util",
        # pip: absl/testing:parameterized
        # pip: numpy
        "//sonnet/src:linear",
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_library(
    name = "write",
    srcs = ["write.py"],
    deps = [
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "write_test",
    srcs = ["write_test.py"],
    main = "write_test.py",
    deps = [
        ":write",
        # pip: numpy
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)
