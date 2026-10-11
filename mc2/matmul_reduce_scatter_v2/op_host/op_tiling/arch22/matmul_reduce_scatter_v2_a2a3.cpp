/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file matmul_reduce_scatter_v2_a2a3.cpp
 * \brief
 */
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "matmul_reduce_scatter_v2_aiv_mode_smallm_tiling.h"

namespace optiling {
constexpr uint32_t ATTR_COMMMODE = 10;

ge::graphStatus MatmulReduceScatterTilingV2Func(gert::TilingContext* context)
{
    OP_LOGI("MatmulReduceScatterTilingV2", "Start to do tiling in MatmulReduceScatterTilingV2Func A2/A3");
    auto attrs = context->GetAttrs();
    auto commModePtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMMMODE));
    OP_TILING_CHECK(commModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "commMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::strcmp(commModePtr, "aiv") != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "commMode", commModePtr,
                                                          "The value of commMode must be aiv."),
                    return ge::GRAPH_FAILED);
    return MatmulReduceScatterTilingV2AivModeFunc(context);
}
} // namespace optiling
