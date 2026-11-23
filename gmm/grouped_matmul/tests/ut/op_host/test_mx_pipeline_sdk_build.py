# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Configure the actual host CMake file with SDK versions, without requiring an NPU."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HOST = Path(__file__).resolve().parents[3] / "op_host"


class SdkBuildGateTest(unittest.TestCase):
    def test_original_operator_abi(self):
        root = HOST.parent
        for rel in [
            "op_host/grouped_matmul_mx_prefill_def.cpp",
            "op_kernel/grouped_matmul_mx_prefill.cpp",
        ]:
            self.assertFalse((root / rel).exists())
        for rel in [
            "op_api/grouped_matmul.cpp",
            "op_host/CMakeLists.txt",
            "op_host/grouped_matmul_infershape.cpp",
            "op_host/op_tiling/grouped_matmul_tiling.cpp",
        ]:
            self.assertNotIn(
                "GroupedMatmulMxPrefill", (root / rel).read_text(encoding="utf-8")
            )
        pipeline = (
            root
            / "op_kernel/arch35/quant_adaptive_sliding_window_templates/gmm_weight_quant_tensor_api_mx_kernel.h"
        )
        self.assertNotIn("StaticInputs", pipeline.read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("cmake"), "Requires CMake")
    def test_sdk_range(self):
        for version, expected in [
            ("9.1.0", "0"),
            ("9.2.0", "1"),
            ("9.3.0", "1"),
            ("9.4.0", "1"),
            ("9.4.1", "1"),
            ("10.0.0", "1"),
            ("", "0"),
        ]:
            for platform in ["ascend950", "ascend350", "none"]:
                self._check_configuration(
                    version, expected if platform == "ascend950" else "0", platform
                )

    def _check_configuration(self, version, expected, platform):
        for shared in [False, True]:
            with (
                self.subTest(version=version, platform=platform, shared_target=shared),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                project = (
                    '''cmake_minimum_required(VERSION 3.18)
project(mx_sdk_gate CXX)
set(BUILD_OPEN_PROJECT FALSE)
set(BUILD_OPS_RTY_KERNEL FALSE)
set(ASCEND_COMPUTE_UNIT "'''
                    + platform
                    + '''")
set(OPHOST_NAME mock)
function(get_cann_package_version out)
  set(${out} "'''
                    + version
                    + """" PARENT_SCOPE)
endfunction()
macro(add_ops_compile_options)
endmacro()
macro(add_modules_sources)
  if(NOT TARGET mock_tiling_obj)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp" "")
add_library(mock_tiling_obj OBJECT "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
  endif()
endmacro()
"""
                )
                if shared:
                    project += """file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/root_dummy.cpp" "")
add_library(mock_tiling_obj OBJECT "${CMAKE_CURRENT_BINARY_DIR}/root_dummy.cpp")
"""
                project += (
                    '''add_subdirectory("'''
                    + str(HOST)
                    + '''" gmm_host)
get_source_file_property(defs "'''
                    + str(
                        HOST
                        / "op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp"
                    )
                    + """"
  TARGET_DIRECTORY mock_tiling_obj COMPILE_DEFINITIONS)
if(NOT "${defs}" STREQUAL "GROUPED_MATMUL_ENABLE_MX_PIPELINE="""
                    + expected
                    + """")
  message(FATAL_ERROR "Unexpected gate: ${defs}")
endif()
"""
                )
                (root / "CMakeLists.txt").write_text(project)
                subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(root / "build")],
                    check=True,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
