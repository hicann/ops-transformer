# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(QUANT_FLASH_ATTN_METADATA_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for quant_flash_attn_metadata")

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(OP_NAME ${PARENT_DIR} NAME)

add_modules_sources(OPTYPE ${OP_NAME} ACLNNTYPE aclnn)

# Metadata and attention use the same platform workspace/L2 scheduling rules.
foreach(opapi_target cust_opapi op_host_aclnn)
    if(TARGET ${opapi_target})
        target_link_libraries(${opapi_target} PRIVATE tiling_api)
    endif()
endforeach()
