licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ops",
    deps = [
        ":resampler_op_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "resampler_op_lib",
    srcs = ["ops/resampler.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//third_party/eigen3",
        "@protobuf//:protobuf_headers",
    ],
    alwayslink = 1,
)
