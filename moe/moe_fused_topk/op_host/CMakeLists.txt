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
    target_sources(op_host_aclnnExc PRIVATE
    moe_fused_topk_def.cpp
    )
    add_ops_compile_options(
            OP_NAME MoeFusedTopk
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE moe_fused_topk ACLNNTYPE aclnn_exclude)

    # InferDataType 按规范放在 op_graph/ 下，标准构建宏不扫描 op_graph/*_graph_infer*.cpp，
    # 此处显式并入 infer obj（共享的 ophost_xxx_infer_obj），确保注册进入 liboptiling 与 cust_proto。
    if(TARGET ${OPHOST_NAME}_infer_obj)
        target_sources(${OPHOST_NAME}_infer_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../op_graph/moe_fused_topk_graph_infer.cpp)
    endif()
endif()
