package(default_visibility = ["//sonnet:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

snt_py_library(
    name = "link_tf_api",
    srcs = ["link_tf_api.py"],
    deps = [
        # pip: docutils
        # pip: six
        # pip: tensorflow
    ],
)

# TODO(b/135667439) Python 3 blocker: docutils internally is PY2 only.
snt_py_test(
    name = "link_tf_api_test",
    srcs = ["link_tf_api_test.py"],
    gpu = False,
    python_version = "PY2",
    tpu = False,
    deps = [
        ":link_tf_api",
        # pip: absl/testing:absltest
        # pip: tensorflow
    ],
)
