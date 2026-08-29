# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    # Core OpDef registration
    target_sources(op_host_aclnnInner PRIVATE
        all_gather_matmul_v3_def.cpp
    )

    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE ON
        OPTYPE all_gather_matmul_v3 all_gather_matmul_v3 ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(all_gather_matmul_v3_depends mc2/common mc2/3rd PARENT_SCOPE)
    set(all_gather_matmul_v3_apt_depends mc2/common mc2/3rd PARENT_SCOPE)
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/ops-tensor/include/tensor_api/impl/c_api
                         ${CMAKE_SOURCE_DIR}/third_party/ops-tensor/include/tensor_api/include/c_api)

    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME AllGatherMatmulV3
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS --cce-auto-sync=off
                    -mllvm
                    -cce-aicore-dcci-before-kernel-end=false
                    -I${CMAKE_SOURCE_DIR}/mc2/common/op_kernel
                    -I${CMAKE_SOURCE_DIR}/mc2/common/op_kernel/apace/tests/st/.third_party/ops-tensor/include
        )
    endif()
    target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_SOURCE_DIR}/mc2/common/op_kernel
        ${CMAKE_SOURCE_DIR}/mc2
    )
else()
    add_mc2_modules_sources(OPTYPE all_gather_matmul_v3 all_gather_matmul_v3 ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
