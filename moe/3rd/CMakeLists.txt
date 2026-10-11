# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    file(GLOB MOE_ACLNN_3RD_SRCS
        # moe_gather_v2
        ${CMAKE_CURRENT_SOURCE_DIR}/moe_gather_v2/op_host/op_api/moe_gather_v2.cpp
        # moe_inplace_index_add_with_sorted
        ${CMAKE_CURRENT_SOURCE_DIR}/moe_inplace_index_add_with_sorted/op_host/op_api/moe_inplace_index_add.cpp
        # moe_masked_scatter
        ${CMAKE_CURRENT_SOURCE_DIR}/moe_masked_scatter/op_host/op_api/moe_masked_scatter.cpp
    )

    if(MOE_ACLNN_3RD_SRCS)
        add_opapi_modules()
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${MOE_ACLNN_3RD_SRCS})
    endif()

    file(GLOB MOE_TILING_3RD_SRCS
        # moe_inplace_index_add_with_sorted
        ${CMAKE_CURRENT_SOURCE_DIR}/moe_inplace_index_add_with_sorted/op_host/moe_inplace_index_add_with_sorted_tiling.cpp
    )

    if(MOE_TILING_3RD_SRCS)
        add_tiling_modules()
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${MOE_TILING_3RD_SRCS})
    endif()
endif()
