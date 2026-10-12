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
 * \file nsa_compress_attention_tiling.cpp
 * \brief
 */

#include "nsa_compress_attention_tiling.h"
#include <queue>
#include <cmath>
#include <cfloat>
#include "log/log.h"
#include "err/ops_err.h"
#include <register/op_impl_registry.h>
#include "op_host/data_copy_transpose_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "nsa_compress_attention_tiling_common.h"
using namespace Ops::Transformer::OpTiling;
using namespace ge;
using namespace AscendC;

namespace optiling {

constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr size_t VALUE_INPUT_INDEX = 2;
constexpr size_t INPUTLAYOUT_ATTRS_INDEX = 2;
constexpr size_t DIM_3 = 3;

static ge::graphStatus CheckParams(const gert::TilingContext* context)
{
    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE_INPUT_INDEX) != nullptr && context->GetAttrs() != nullptr) {
        auto& queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
        auto& keyShape = context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
        auto& valueShape = context->GetInputShape(VALUE_INPUT_INDEX)->GetStorageShape();
        const char* inputLayout = context->GetAttrs()->GetAttrPointer<char>(INPUTLAYOUT_ATTRS_INDEX);
        if (strlen(inputLayout) == DIM_3 && strcmp(inputLayout, "TND") == 0) {
            // q_D != k_D
            // q shape: N2, T, G, D1
            // k shape: T, N2, D2
            OP_CHECK_IF((queryShape.GetDim(3) != keyShape.GetDim(2)),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query or key shape is invalid"),
                        return ge::GRAPH_FAILED);
            // kD < vD
            OP_CHECK_IF((keyShape.GetDim(2) < valueShape.GetDim(2)),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "key or value shape is invalid"),
                        return ge::GRAPH_FAILED);
            return ge::SUCCESS;
        } else {
            OP_LOGW(context, "Got invalid input_layout[%s]. Only support TND now.", inputLayout);
            return ge::GRAPH_FAILED;
        }
        return ge::SUCCESS;
    }
    OP_LOGW(context, "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingNsaCompressAttention(gert::TilingContext* context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto resultCode = TilingRegistry::GetInstance().DoTilingImpl(context);
    return resultCode;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForNsaCompressAttention(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<NsaCompressAttentionCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NsaCompressAttention)
    .Tiling(TilingNsaCompressAttention)
    .TilingInputsDataDependency({4, 5, 6})
    .TilingParse<NsaCompressAttentionCompileInfo>(TilingPrepareForNsaCompressAttention); // 向框架注册入口函数

} // namespace optiling
