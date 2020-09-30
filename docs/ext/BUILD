load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

package(default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

snt_py_library(
    name = "link_tf_api",
    srcs = ["link_tf_api.py"],
    deps = [
        # pip: docutils
        # pip: six
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "link_tf_api_test",
    srcs = ["link_tf_api_test.py"],
    gpu = False,
    tpu = False,
    deps = [
        ":link_tf_api",
        # pip: absl/testing:absltest
        # pip: tensorflow
    ],
)
