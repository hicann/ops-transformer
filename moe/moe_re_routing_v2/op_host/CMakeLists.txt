# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "=== Debug: start ops.moe.moe_re_routing_v2.op_host.CMakeLists.txt")

if (BUILD_OPEN_PROJECT)
    message(STATUS "=== Debug: BUILD is BUILD_OPEN_PROJECT")
    set(moe_re_routing_v2_depends moe/moe_re_routing
        CACHE INTERNAL "Dependencies for moe_re_routing_v2")
    set(moe_re_routing_v2_apt_depends moe/moe_re_routing
        CACHE INTERNAL "Dependencies for moe_re_routing_v2_apt")
    target_sources(op_host_aclnnInner PRIVATE
        moe_re_routing_v2_def.cpp
    )
    add_ops_compile_options(
        OP_NAME MoeReRoutingV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -mllvm 
                -cce-aicore-dcci-before-kernel-end=false
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    message(STATUS "=== Debug: BUILD is NOT BUILD_OPS_RTY_KERNEL")
    add_modules_sources(OPTYPE moe_re_routing_v2 ACLNNTYPE aclnn)
endif()
