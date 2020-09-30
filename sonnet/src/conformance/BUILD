load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

package(
    default_testonly = True,
    default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0 License

snt_py_library(
    name = "goldens",
    srcs = ["goldens.py"],
    deps = [
        # pip: absl/testing:parameterized
        # pip: numpy
        # pip: six
        "//sonnet",
        # pip: tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        # pip: six
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "checkpoint_test",
    timeout = "long",
    srcs = ["checkpoint_test.py"],
    data = [":checkpoints"],
    shard_count = 10,
    deps = [
        ":goldens",
        # pip: absl/logging
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        "//sonnet/src/distribute:replicator",
        "//sonnet/src/distribute:replicator_test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "distribute_test",
    srcs = ["distribute_test.py"],
    shard_count = 10,
    deps = [
        ":descriptors",
        ":goldens",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        "//sonnet/src/distribute:replicator_test_utils",
        # pip: tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "doctest_test",
    srcs = ["doctest_test.py"],
    deps = [
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "descriptors",
    srcs = ["descriptors.py"],
    deps = [
        "//sonnet",
        # pip: tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)

snt_py_test(
    name = "descriptors_test",
    srcs = ["descriptors_test.py"],
    deps = [
        ":descriptors",
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "function_test",
    timeout = "long",
    srcs = ["function_test.py"],
    shard_count = 10,
    deps = [
        ":descriptors",
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "goldens_test",
    timeout = "long",
    srcs = ["goldens_test.py"],
    shard_count = 10,
    deps = [
        ":goldens",
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "pickle_test",
    timeout = "long",
    srcs = ["pickle_test.py"],
    shard_count = 10,
    deps = [
        ":goldens",
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "saved_model_test",
    timeout = "long",
    srcs = ["saved_model_test.py"],
    shard_count = 10,
    deps = [
        ":goldens",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "xla_test",
    timeout = "long",
    srcs = ["xla_test.py"],
    shard_count = 10,
    deps = [
        ":goldens",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
        # tf: compiler/jit:xla_cpu_jit
        # tf: compiler/jit:xla_gpu_jit
    ],
)

snt_py_test(
    name = "keras_test",
    timeout = "long",
    srcs = ["keras_test.py"],
    shard_count = 10,
    deps = [
        ":descriptors",
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "build_test",
    timeout = "long",
    srcs = ["build_test.py"],
    shard_count = 10,
    deps = [
        ":descriptors",
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "copy_test",
    timeout = "long",
    srcs = ["copy_test.py"],
    shard_count = 10,
    deps = [
        ":goldens",
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # pip: tree
    ],
)

snt_py_test(
    name = "tensorflow1_test",
    timeout = "long",
    srcs = ["tensorflow1_test.py"],
    deps = [
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "optimizer_test",
    timeout = "long",
    srcs = ["optimizer_test.py"],
    deps = [
        ":descriptors",
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)
