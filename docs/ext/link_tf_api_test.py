# Copyright 2019 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for ``:tf:`` Sphinx role."""

from absl.testing import absltest
from docs.ext import link_tf_api

DOC_BASE_URL = "https://www.tensorflow.org/versions/r2.0/api_docs/python/tf"


class LinkTfApiTest(absltest.TestCase):

  def test_non_existent(self):
    self.assertIsNone(link_tf_api.tf_doc_url("tomhennigan"))
    self.assertIsNone(link_tf_api.tf_doc_url("autograph.1"))

  def test_link_to_top_level(self):
    self.assertEqual(
        link_tf_api.tf_doc_url("function"), DOC_BASE_URL + "/function")
    self.assertEqual(link_tf_api.tf_doc_url("Module"), DOC_BASE_URL + "/Module")

  def test_link_to_nested_package(self):
    self.assertEqual(
        link_tf_api.tf_doc_url("autograph.to_code"),
        DOC_BASE_URL + "/autograph/to_code")

  def test_link_to_method_of_exported_class(self):
    self.assertEqual(
        link_tf_api.tf_doc_url("TensorArray.read"),
        DOC_BASE_URL + "/TensorArray#read")

  def test_link_to_non_existent_method_of_exported_class(self):
    self.assertIsNone(link_tf_api.tf_doc_url("TensorArray.tomhennigan"))


if __name__ == "__main__":
  absltest.main()
