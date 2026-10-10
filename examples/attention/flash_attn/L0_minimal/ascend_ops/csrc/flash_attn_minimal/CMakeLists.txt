# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "BUILD_TORCH_OPS ON in flash_attn_minimal")

set(flash_attn_minimal_NPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_interface.cpp
)

set_source_files_properties(
    ${flash_attn_minimal_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--npu-arch=dav-3510 -xasc "
)

add_library(flash_attn_minimal_objects OBJECT ${flash_attn_minimal_NPU_SOURCES})

target_compile_options(flash_attn_minimal_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(flash_attn_minimal_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()
