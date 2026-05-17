# Description:
#   TensorFlow builder (TensorFlow on Demand project).

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_binary(
    name = "config_detector",
    srcs = ["config_detector.py"],
    data = [
        "//tensorflow/tools/tensorflow_builder/config_detector/data/golden:cuda_cc_golden",
    ],
    strict_deps = True,
    deps = [
        ":cuda_compute_capability_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "cuda_compute_capability",
    srcs = ["data/cuda_compute_capability.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@six_archive//:six",
    ],
)

py_library(
    name = "cuda_compute_capability_lib",
    srcs = ["data/cuda_compute_capability.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@six_archive//:six",
    ],
)

py_test(
    name = "config_detector_test",
    srcs = ["config_detector_test.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl:flags",
        "@local_xla//xla/tsl/platform:test",
    ],
)
