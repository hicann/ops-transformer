/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_grad_tiling.cpp
 * \brief
 */
#include <string>
#include <set>
#include "quant_flash_attn_grad_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;
using namespace std;

namespace optiling {

constexpr int64_t QUERY_IDX = 0;
constexpr int64_t KEY_IDX = 1;
constexpr int64_t VALUE_IDX = 2;
constexpr int64_t DO_IDX = 3;
constexpr int64_t ATTN_OUT_IDX = 4;
constexpr int64_t Q_DESCALE_IDX = 5;
constexpr int64_t K_DESCALE_IDX = 6;
constexpr int64_t V_DESCALE_IDX = 7;
constexpr int64_t DO_DESCALE_IDX = 8;
constexpr int64_t P_SCALE_IDX = 9;
constexpr int64_t DS_SCALE_IDX = 10;
constexpr int64_t SOFTMAX_LSE = 11;
constexpr int64_t CU_SEQLENS_Q = 12;
constexpr int64_t CU_SEQLENS_KV = 13;
constexpr int64_t SEQUSED_Q = 14;
constexpr int64_t SEQUSED_KV = 15;
constexpr int64_t SINKS = 16;
constexpr int64_t METADATA = 18;
constexpr int64_t DQ_IDX = 0;
constexpr int64_t DK_IDX = 1;
constexpr int64_t DV_IDX = 2;
constexpr int64_t DSINK_IDX = 3;
constexpr int64_t QUANT_MODE_IDX = 0;
constexpr int64_t MASK_MODE = 2;
constexpr int64_t WIN_LEFT = 3;
constexpr int64_t WIN_RIGHT = 4;
constexpr int64_t MAX_SEQLEN_Q = 5;
constexpr int64_t MAX_SEQLEN_KV = 6;
constexpr int64_t WINDOW = 4;

static ge::graphStatus ParseAttrs(gert::TilingContext *context, const string &opName)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *maxSeqlenQ = attrs->GetAttrPointer<int64_t>(MAX_SEQLEN_Q);
    OP_CHECK_IF(maxSeqlenQ != nullptr && *maxSeqlenQ != -1, OP_LOGE(opName, "maxSeqlenQ not support."),
                return ge ::GRAPH_FAILED);
    const int64_t *maxSeqlenKV = attrs->GetAttrPointer<int64_t>(MAX_SEQLEN_KV);
    OP_CHECK_IF(maxSeqlenKV != nullptr && *maxSeqlenKV != -1, OP_LOGE(opName, "maxSeqlenKV not support."),
                return ge::GRAPH_FAILED);
    const int64_t *maskMode = attrs->GetAttrPointer<int64_t>(MASK_MODE);
    OP_CHECK_IF(maskMode != nullptr && *maskMode != 0, OP_LOGE(opName, "maskMode must be 0."), return ge::GRAPH_FAILED);
    const int64_t *winRight = attrs->GetAttrPointer<int64_t>(WIN_RIGHT);
    OP_CHECK_IF(winRight != nullptr && *winRight != -1, OP_LOGE(opName, "winRight not support."),
                return ge::GRAPH_FAILED);
    const int64_t *winLeft = attrs->GetAttrPointer<int64_t>(WIN_LEFT);
    OP_CHECK_IF(winLeft != nullptr && *winLeft != -1, OP_LOGE(opName, "winLeft not support."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateRequiredInputs(gert::TilingContext *context, const string &opName)
{
    auto queryDesc = context->GetInputDesc(QUERY_IDX);
    auto keyDesc = context->GetInputDesc(KEY_IDX);
    auto valueDesc = context->GetInputDesc(VALUE_IDX);
    auto doDesc = context->GetInputDesc(DO_IDX);
    auto attnOutDesc = context->GetInputDesc(ATTN_OUT_IDX);
    auto qDescaleDesc = context->GetInputDesc(Q_DESCALE_IDX);
    auto kDescaleDesc = context->GetInputDesc(K_DESCALE_IDX);
    auto vDescaleDesc = context->GetInputDesc(V_DESCALE_IDX);
    auto doDescaleDesc = context->GetInputDesc(DO_DESCALE_IDX);
    auto pScaleDesc = context->GetInputDesc(P_SCALE_IDX);
    auto dsScaleDesc = context->GetInputDesc(DS_SCALE_IDX);
    auto softmaxLseDesc = context->GetInputDesc(SOFTMAX_LSE);
    auto cuSeqlensQDesc = context->GetOptionalInputDesc(CU_SEQLENS_Q);
    auto cuSeqlensKVDesc = context->GetOptionalInputDesc(CU_SEQLENS_KV);
    auto sequsedQDesc = context->GetOptionalInputDesc(SEQUSED_Q);
    auto sequsedKVDesc = context->GetOptionalInputDesc(SEQUSED_KV);
    auto sinksDesc = context->GetOptionalInputDesc(SINKS);
    auto metadataDesc = context->GetOptionalInputDesc(METADATA);

    auto dqDesc = context->GetOutputDesc(DQ_IDX);
    auto dkDesc = context->GetOutputDesc(DK_IDX);
    auto dvDesc = context->GetOutputDesc(DV_IDX);
    auto dsinkDesc = context->GetOutputDesc(DSINK_IDX);

    OP_CHECK_IF(queryDesc == nullptr, OP_LOGE(opName, "query must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDesc == nullptr, OP_LOGE(opName, "key must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueDesc == nullptr, OP_LOGE(opName, "value must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(doDesc == nullptr, OP_LOGE(opName, "dout must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(attnOutDesc == nullptr, OP_LOGE(opName, "attnOut must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qDescaleDesc == nullptr, OP_LOGE(opName, "qDescale must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDescaleDesc == nullptr, OP_LOGE(opName, "kDescale must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vDescaleDesc == nullptr, OP_LOGE(opName, "vDescale must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(doDescaleDesc == nullptr, OP_LOGE(opName, "doDescale must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(pScaleDesc == nullptr, OP_LOGE(opName, "pScale must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dsScaleDesc == nullptr, OP_LOGE(opName, "dsScale must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxLseDesc == nullptr, OP_LOGE(opName, "softmaxLse must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuSeqlensQDesc != nullptr, OP_LOGE(opName, "cuSeqlensQ not support."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuSeqlensKVDesc != nullptr, OP_LOGE(opName, "cuSeqlensKV not support."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sequsedQDesc != nullptr, OP_LOGE(opName, "sequsedQ not support."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sequsedKVDesc != nullptr, OP_LOGE(opName, "sequsedKV not support."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinksDesc != nullptr, OP_LOGE(opName, "sinks not support."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataDesc == nullptr, OP_LOGE(opName, "metadata must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dqDesc == nullptr, OP_LOGE(opName, "dq must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dkDesc == nullptr, OP_LOGE(opName, "dk must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dvDesc == nullptr, OP_LOGE(opName, "dv must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dsinkDesc == nullptr, OP_LOGE(opName, "dSink must be provided."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingQuantFlashAttnGrad(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    auto opName = context->GetNodeName();
    if (ParseAttrs(context, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateRequiredInputs(context, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingParseForQuantFlashAttnGrad([[maybe_unused]] gert::TilingParseContext *context)
{
    OP_CHECK_IF(
        context == nullptr,
        OP_LOGE(context, "The op [QuantFlashAttentionScoreGrad] received bad params, the reason is: [context is null]"),
        return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfoPtr == nullptr,
        OP_LOGE(context,
                "The op [QuantFlashAttentionScoreGrad] received bad params, the reason is: [platformInfoPtr is null]"),
        return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<QuantFlashAttnGradCompileInfo>();
    OP_CHECK_IF(
        compileInfoPtr == nullptr,
        OP_LOGE(context,
                "The op [QuantFlashAttentionScoreGrad] received bad params, the reason is: [compileInfoPtr is null]"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantFlashAttnGrad)
    .Tiling(TilingQuantFlashAttnGrad)
    .TilingParse<QuantFlashAttnGradCompileInfo>(TilingParseForQuantFlashAttnGrad); // 向框架注册入口函数

} // namespace optiling
