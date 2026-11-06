# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        text_gated_residual_def.cpp
    )
endif()

add_modules_sources(OPTYPE text_gated_residual ACLNNTYPE aclnn_exclude)

set(HC_COMPILE_OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
if (DEFINED CATLASS AND EXISTS "${CATLASS}/include/catlass/catlass.hpp")
    list(APPEND HC_COMPILE_OPTIONS "-I${CATLASS}/include")
endif()
add_ops_compile_options(
    OP_NAME TextGatedResidual
    OPTIONS ${HC_COMPILE_OPTIONS}
)
