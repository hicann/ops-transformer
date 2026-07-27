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
    quant_grouped_matmul_dequant_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
        OP_NAME QuantGroupedMatmulDequant
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
    )
else()
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE quant_grouped_matmul_dequant ACLNNTYPE aclnn_exclude)
endif()
    # add_opapi_modules()
    # target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${gmm\quant_grouped_matmul_dequant\op_api})
