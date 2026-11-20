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
        distribute_barrier_extend_def.cpp
    )
    add_modules_sources_with_soc(
        OP_MC2_ENABLE ON
        OPTYPE distribute_barrier_extend ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(distribute_barrier_extend_depends mc2/common mc2/3rd mc2/distribute_barrier PARENT_SCOPE)
    set(distribute_barrier_extend_apt_depends mc2/common mc2/3rd mc2/distribute_barrier PARENT_SCOPE)

    # --cce-auto-sync=off：指定CCE编译器是否自动执行线程间或模块间的同步操作
    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend950" OR
       "${CONDITION_UNIT}" STREQUAL "ascend960dt" OR
       "${CONDITION_UNIT}" STREQUAL "ascend960pr")
        add_ops_compile_options(
            OP_NAME DistributeBarrierExtend
            OPTIONS --cce-auto-sync=off
        )
    endif()
else() # 回黄host
    add_mc2_modules_sources(OPTYPE distribute_barrier_extend ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
