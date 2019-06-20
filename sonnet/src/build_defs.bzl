"""Sonnet specific build rules."""

def snt_py_library(name, **kwargs):
    """Proxy for py_library.

    Internally we override this to enable type checking via PyType (more
    information at https://github.com/google/pytype).

    Args:
        name: library name.
        **kwargs: keyword args passed straight to py_library.
    """
    native.py_library(name = name, **kwargs)

def snt_py_test(
        name,
        deps = [],
        tags = [],
        py3_tags = [],
        python_version = "PY3",
        main = None,
        gpu = True,
        tpu = True,
        **kwargs):
    """Runs a py_test.

    Args:
        name: test target name to generate suffixed with `test`.
        deps: additional dependencies for the test targets.
        tags: tags to be assigned to the different test targets.
        py3_tags: python3 specific tags to be appended to tags.
        python_version: ignored in OSS build, Sonnet always runs with Python 3.
        main: main script to be run for the test.
        gpu: Whether the test can be run on GPU. Note ignored by test.
        tpu: Whether the test can be run on TPU. Note ignored by test.
        **kwargs: extra keyword arguments to the test.
    """
    if main == None:
        main = name + ".py"

    native.py_test(
        name = name,
        deps = deps,
        tags = tags + py3_tags,
        main = main,
        python_version = "PY3",
        **kwargs
    )
