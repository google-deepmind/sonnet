# Description:
#   Custom ops for use in sonnet.

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "all_kernels",
    deps = [
        ":resampler_op",
    ],
)

cuda_deps = ["@org_tensorflow//tensorflow/core:gpu_lib"]

cc_library(
    name = "resampler_op",
    srcs = ["resampler_op.cc"],
    hdrs = ["resampler_op.h"],
    linkstatic = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//third_party/eigen3",
        "@protobuf//:protobuf_headers",
    ],
    alwayslink = 1,
)
