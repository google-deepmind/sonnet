load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

package(default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

snt_py_library(
    name = "optimizer_tests",
    testonly = 1,
    srcs = ["optimizer_tests.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_library(
    name = "adam",
    srcs = ["adam.py"],
    deps = [
        ":optimizer_utils",
        "//sonnet/src:base",
        "//sonnet/src:once",
        "//sonnet/src:types",
        "//sonnet/src:utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "adam_test",
    srcs = ["adam_test.py"],
    shard_count = 10,
    deps = [
        ":adam",
        ":optimizer_tests",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "momentum",
    srcs = ["momentum.py"],
    deps = [
        ":optimizer_utils",
        "//sonnet/src:base",
        "//sonnet/src:once",
        "//sonnet/src:types",
        "//sonnet/src:utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "momentum_test",
    srcs = ["momentum_test.py"],
    shard_count = 10,
    deps = [
        ":momentum",
        ":optimizer_tests",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    deps = [
        ":optimizer_utils",
        # pip: six
        "//sonnet/src:base",
        "//sonnet/src:once",
        "//sonnet/src:types",
        "//sonnet/src:utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "rmsprop_test",
    srcs = ["rmsprop_test.py"],
    shard_count = 10,
    deps = [
        ":optimizer_tests",
        ":rmsprop",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "sgd",
    srcs = ["sgd.py"],
    deps = [
        ":optimizer_utils",
        "//sonnet/src:base",
        "//sonnet/src:types",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "sgd_test",
    srcs = ["sgd_test.py"],
    shard_count = 10,
    deps = [
        ":optimizer_tests",
        ":sgd",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    deps = [
        "//sonnet/src:types",
        "//sonnet/src/distribute:replicator",
        # pip: tensorflow
    ],
)
