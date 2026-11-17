# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


# The operator registers its kernel/test objects through AddAicpuOpTestCase.
if(NOT TARGET quant_flash_attn_metadata_cases_obj)
    return()
endif()
target_include_directories(quant_flash_attn_metadata_cases_obj PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base")
find_library(QFA_GTEST_MAIN NAMES gtest_main HINTS "${GTEST_INCLUDE}/../lib" REQUIRED)
find_library(QFA_AICPU_CONTEXT NAMES aicpu_context_host
    HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64" REQUIRED)
find_library(QFA_AICPU_NODEDEF NAMES aicpu_nodedef_host
    HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64" REQUIRED)
find_library(QFA_AICPU_PROTOBUF NAMES ascend_protobuf
    HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64" REQUIRED)
add_executable(quant_flash_attn_metadata_ut $<TARGET_OBJECTS:quant_flash_attn_metadata_cases_obj>)
target_link_libraries(quant_flash_attn_metadata_ut PRIVATE
    intf_llt_pub_asan_cxx17 ${QFA_GTEST_MAIN} gtest
    ${QFA_AICPU_CONTEXT} ${QFA_AICPU_NODEDEF} ${QFA_AICPU_PROTOBUF} c_sec dl pthread)
add_test(NAME quant_flash_attn_metadata_ut COMMAND quant_flash_attn_metadata_ut)
