# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# engram_hash is an arch35 (ascend950) operator. The AddOpTestCase routing in
# cmake/ut.cmake skips op_kernel-level test cases when ARCH_DIRECTORY contains arch35, so the
# real kernel-run cases live under op_kernel/arch35/. Recurse into any arch sub-directory.
file(GLOB CURRENT_SOURCE_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_SOURCE_DIRS})
    # file(GLOB ... LIST_DIRECTORIES true) 返回绝对路径, 不能再拼 CURRENT_SOURCE_DIR 前缀。
    if(EXISTS "${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
