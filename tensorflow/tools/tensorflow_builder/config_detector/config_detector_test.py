"""Tests for tensorflow_builder config_detector."""

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


_MODULE_PATH = pathlib.Path(__file__).resolve().parent / "config_detector.py"
_CUDA_MODULE_NAME = (
    "tensorflow.tools.tensorflow_builder.config_detector.data."
    "cuda_compute_capability"
)


def _load_config_detector_module():
  absl_module = types.ModuleType("absl")
  app_module = types.ModuleType("absl.app")
  app_module.run = lambda *_args, **_kwargs: None
  flags_module = types.ModuleType("absl.flags")
  flags_module.FLAGS = types.SimpleNamespace(debug=False)
  flags_module.DEFINE_boolean = lambda *_args, **_kwargs: None
  flags_module.DEFINE_string = lambda *_args, **_kwargs: None

  cuda_module = types.ModuleType(_CUDA_MODULE_NAME)
  cuda_module.retrieve_from_golden = mock.MagicMock(return_value={})
  cuda_module.retrieve_from_web = mock.MagicMock(return_value={})

  module_names = [
      "absl",
      "absl.app",
      "absl.flags",
      "tensorflow",
      "tensorflow.tools",
      "tensorflow.tools.tensorflow_builder",
      "tensorflow.tools.tensorflow_builder.config_detector",
      "tensorflow.tools.tensorflow_builder.config_detector.data",
  ]
  inserted_modules = {}
  for name in module_names:
    if name not in sys.modules:
      inserted_modules[name] = types.ModuleType(name)
      sys.modules[name] = inserted_modules[name]

  sys.modules["absl"] = absl_module
  sys.modules["absl.app"] = app_module
  sys.modules["absl.flags"] = flags_module
  sys.modules[_CUDA_MODULE_NAME] = cuda_module

  spec = importlib.util.spec_from_file_location(
      "config_detector_under_test", _MODULE_PATH
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  return module, cuda_module, inserted_modules


class ConfigDetectorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config_detector, self.cuda_module, self.inserted_modules = (
        _load_config_detector_module()
    )
    self.config_detector.PLATFORM = "linux"
    self.config_detector.GPU_TYPE = None

  def tearDown(self):
    for name in list(self.inserted_modules.keys()):
      sys.modules.pop(name, None)
    sys.modules.pop(_CUDA_MODULE_NAME, None)
    super().tearDown()

  def test_get_platform_linux(self):
    with mock.patch.object(
        self.config_detector, "run_shell_cmd", return_value=("linux\n", "")
    ):
      self.assertEqual(self.config_detector.get_platform(), "linux")

  def test_get_platform_unsupported_raises(self):
    with mock.patch.object(
        self.config_detector, "run_shell_cmd", return_value=("darwin\n", "")
    ):
      with self.assertRaises(SystemExit):
        self.config_detector.get_platform()

  def test_get_cuda_version_all_parses_all_detected_versions(self):
    out = b"/usr/local/cuda-12.2\n/usr/local/cuda-11.8"
    with mock.patch.object(
        self.config_detector, "run_shell_cmd", return_value=(out, "")
    ):
      self.assertEqual(
          self.config_detector.get_cuda_version_all(), ["12.2", "11.8"]
      )

  def test_get_cuda_compute_capability_returns_none_for_unknown_gpu(self):
    self.config_detector.GPU_TYPE = "unknown"

    result = self.config_detector.get_cuda_compute_capability()

    self.assertIsNone(result)
    self.cuda_module.retrieve_from_golden.assert_not_called()

  def test_get_cuda_compute_capability_uses_requested_source(self):
    self.config_detector.GPU_TYPE = b"Tesla K80"
    self.cuda_module.retrieve_from_web.return_value = {b"Tesla K80": ["3.7"]}
    self.cuda_module.retrieve_from_golden.return_value = {b"Tesla K80": ["3.5"]}

    from_web = self.config_detector.get_cuda_compute_capability(
        source_from_url=True
    )
    from_golden = self.config_detector.get_cuda_compute_capability(
        source_from_url=False
    )

    self.assertEqual(from_web, ["3.7"])
    self.assertEqual(from_golden, ["3.5"])

  def test_save_to_file_appends_json_extension(self):
    json_data = {"Platform": "linux"}
    with tempfile.TemporaryDirectory() as tmp_dir:
      self.config_detector.PATH_TO_DIR = tmp_dir

      self.config_detector.save_to_file(json_data, "configs")

      output_file = pathlib.Path(tmp_dir) / "configs.json"
      self.assertTrue(output_file.exists())


if __name__ == "__main__":
  unittest.main()
