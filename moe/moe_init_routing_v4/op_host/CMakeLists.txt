# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "=== Debug: start ops.moe.moe_init_routing_v4.op_host.CMakeLists.txt ")

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if (BUILD_OPEN_PROJECT)
    message(STATUS "=== Debug: BUILD is BUILD_OPEN_PROJECT")
    set(moe_init_routing_v4_depends moe/moe_init_routing_v3
        CACHE INTERNAL "Dependencies for moe_init_routing_v4")
    set(moe_init_routing_v4_apt_depends moe/moe_init_routing_v3
        CACHE INTERNAL "Dependencies for moe_init_routing_v4")
    target_sources(op_host_aclnnExc PRIVATE
        moe_init_routing_v4_def.cpp
    )
    if(CONDITION_UNIT STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME MoeInitRoutingV4
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
                    -mllvm 
                    -cce-aicore-dcci-before-kernel-end=false
        )
    else()
        add_ops_compile_options(
            OP_NAME MoeInitRoutingV4
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
        )
    endif()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    message(STATUS "=== Debug: BUILD is NOT BUILD_OPS_RTY_KERNEL")
    add_modules_sources(OPTYPE moe_init_routing_v4 ACLNNTYPE aclnn_exclude)
endif()
