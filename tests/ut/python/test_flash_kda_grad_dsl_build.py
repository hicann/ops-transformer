# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise category discovery and the Native-only Flash KDA Grad build entry."""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "open_project,selected,registered,enabled",
    [(True, True, True, True), (False, True, True, False), (True, False, False, False)],
)
def test_flash_kda_grad_category_discovery(
    tmp_path, open_project, selected, registered, enabled
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.20)
project(flash_kda_grad_dsl_discovery NONE)
include("{ROOT}/cmake/func.cmake")
set(BUILD_OPEN_PROJECT {"TRUE" if open_project else "FALSE"})
set(ASCEND_OP_NAME {"flash_kda_grad" if selected else "not_selected"})
function(enable_cannbotdsl_kernel name source_dir)
    file(WRITE "${{CMAKE_BINARY_DIR}}/dsl-call.txt" "${{name}}\n${{source_dir}}\n")
endfunction()
add_subdirectory("{ROOT}/attention" attention)
file(WRITE "${{CMAKE_BINARY_DIR}}/compiled.txt" "${{COMPILED_OPS}}\n${{COMPILED_OP_DIRS}}\n")
"""
    )
    build = tmp_path / "build"
    result = subprocess.run(
        ["cmake", "-S", str(project), "-B", str(build)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (build / "compiled.txt").read_text().splitlines()
    if registered:
        assert lines == [
            "flash_kda_grad",
            str(ROOT / "attention/flash_kda_grad"),
        ]
    else:
        assert not any(lines)
    call = build / "dsl-call.txt"
    assert call.exists() == enabled
    if enabled:
        assert call.read_text().splitlines() == [
            "flash_kda_grad",
            str(ROOT / "attention/flash_kda_grad/op_dsl"),
        ]
