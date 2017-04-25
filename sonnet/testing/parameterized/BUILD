package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 License

py_library(
    name = "parameterized",
    srcs = [
        "__init__.py",
        "parameterized.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "parameterized_test",
    size = "small",
    srcs = ["parameterized_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":parameterized",
    ],
)
