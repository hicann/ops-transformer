# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        recurrent_gated_delta_rule_def.cpp
    )
endif()

if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME RecurrentGatedDeltaRule
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
    )
else()
    add_ops_compile_options(
        OP_NAME RecurrentGatedDeltaRule
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
    )
endif()

add_modules_sources_with_soc(
    # 该变量用于判断op_api目录是否从op_host移出
    OP_API_INDEPENDENT ON
    # op_api新的地址
    OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OPTYPE recurrent_gated_delta_rule ACLNNTYPE aclnn_exclude
)
