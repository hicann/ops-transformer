# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        moe_distribute_combine_add_rms_norm_def.cpp
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE ON
        OPTYPE moe_distribute_combine_add_rms_norm moe_distribute_combine_add_rms_norm ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(moe_distribute_combine_add_rms_norm_depends mc2/common mc2/3rd mc2/moe_distribute_dispatch_v2 mc2/moe_distribute_combine_v2 mc2/moe_distribute_dispatch_v3 mc2/moe_distribute_combine_v3 PARENT_SCOPE)
    set(moe_distribute_combine_add_rms_norm_a3_depends mc2/common mc2/3rd mc2/moe_distribute_dispatch_v2 mc2/moe_distribute_combine_v2 mc2/moe_distribute_dispatch_v3 mc2/moe_distribute_combine_v3 PARENT_SCOPE)
    set(moe_distribute_combine_add_rms_norm_apt_depends mc2/common mc2/3rd mc2/moe_distribute_dispatch_v2 mc2/moe_distribute_combine_v2 mc2/moe_distribute_dispatch_v3 mc2/moe_distribute_combine_v3 PARENT_SCOPE)
    set(moe_distribute_combine_add_rms_norm_a6_depends mc2/common mc2/3rd mc2/moe_distribute_dispatch_v2 mc2/moe_distribute_combine_v2 mc2/moe_distribute_dispatch_v3 mc2/moe_distribute_combine_v3 PARENT_SCOPE)
    install(
        DIRECTORY
        ${OPS_ADV_DIR}/mc2/3rd/rms_norm
        ${OPS_ADV_DIR}/mc2/3rd/norm_common
        DESTINATION
        ${IMPL_INSTALL_DIR}/ascendc/3rd
    )

    # --cce-auto-sync=off：指定CCE编译器是否自动执行线程间或模块间的同步操作
    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend910_93")
        add_ops_compile_options(
            OP_NAME MoeDistributeCombineAddRmsNorm
            OPTIONS --cce-auto-sync=off
        )
    endif()
else() # 回黄host
    add_mc2_modules_sources(OPTYPE moe_distribute_combine_add_rms_norm moe_distribute_combine_add_rms_norm ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
