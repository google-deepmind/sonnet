package(
    default_testonly = True,
    default_visibility = ["//sonnet:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

snt_py_library(
    name = "goldens",
    srcs = ["goldens.py"],
    deps = [
        # pip: numpy
        # pip: six
        "//sonnet",
        # pip: tensorflow
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
    srcs = ["checkpoint_test.py"],
    data = [":checkpoints"],
    deps = [
        ":goldens",
        # pip: absl/logging
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:replicator",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "distribute_test",
    srcs = ["distribute_test.py"],
    gpu = False,
    deps = [
        ":goldens",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
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

snt_py_test(
    name = "goldens_test",
    srcs = ["goldens_test.py"],
    deps = [
        ":goldens",
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "pickle_test",
    srcs = ["pickle_test.py"],
    deps = [
        ":goldens",
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    deps = [
        ":goldens",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "xla_test",
    srcs = ["xla_test.py"],
    deps = [
        ":goldens",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
        # tf: compiler/jit:xla_cpu_jit
        # tf: compiler/jit:xla_gpu_jit
    ],
)
