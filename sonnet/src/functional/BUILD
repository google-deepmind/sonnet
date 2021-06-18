load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

package(default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"])

licenses(["notice"])

snt_py_library(
    name = "haiku",
    srcs = ["haiku.py"],
    deps = [
        ":utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "jax",
    srcs = ["jax.py"],
    deps = [
        ":utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    deps = [
        ":haiku",
        "//sonnet/src:base",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//sonnet/src:utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "haiku_test",
    srcs = ["haiku_test.py"],
    deps = [
        ":haiku",
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "jax_test",
    srcs = ["jax_test.py"],
    deps = [
        ":jax",
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "optimizers_test",
    srcs = ["optimizers_test.py"],
    deps = [
        ":optimizers",
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)
