workspace(name = "sonnet")


local_repository(
  name = "org_tensorflow",
  path = "tensorflow",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
name = "io_bazel_rules_closure",
sha256 = "bc41b80486413aaa551860fc37471dbc0666e1dbb5236fb6177cb83b0c105846",
strip_prefix = "rules_closure-dec425a4ff3faf09a56c85d082e4eed05d8ce38f",
urls = [
"http://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dec425a4ff3faf09a56c85d082e4eed05d8ce38f.tar.gz",  # 2017-06-02
"https://github.com/bazelbuild/rules_closure/archive/dec425a4ff3faf09a56c85d082e4eed05d8ce38f.tar.gz",
],
)

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")
