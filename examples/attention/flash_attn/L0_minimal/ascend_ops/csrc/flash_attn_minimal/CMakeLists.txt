# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# ----------------------------------------------------------------------------

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
