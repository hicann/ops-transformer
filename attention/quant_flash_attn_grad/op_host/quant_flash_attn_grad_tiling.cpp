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
constexpr int64_t LAYOUT_Q = 7;
constexpr int64_t LAYOUT_KV = 8;
constexpr int64_t MAIN_INPUT_COUNT = 5;
constexpr int64_t SUPPORTED_HEAD_DIM = 128;

static bool IsTnd(gert::TilingContext *context)
{
    auto layout = context->GetAttrs()->GetAttrPointer<char>(LAYOUT_Q);
    return layout != nullptr && std::string(layout) == "TND";
}

static ge::graphStatus ValidateTnd(gert::TilingContext *context, const string &opName)
{
    const auto &queryShape = context->GetInputShape(QUERY_IDX)->GetStorageShape();
    const auto &keyShape = context->GetInputShape(KEY_IDX)->GetStorageShape();
    int64_t prefixSize = 0;
    for (int64_t idx = 12; idx <= 13; ++idx) {
        auto desc = context->GetOptionalInputDesc(idx);
        auto shape = context->GetOptionalInputShape(idx);
        OP_CHECK_IF(desc == nullptr || shape == nullptr || desc->GetDataType() != ge::DT_INT32 ||
                        shape->GetStorageShape().GetDimNum() != 1 || shape->GetStorageShape().GetDim(0) < 2,
                    OP_LOGE(opName, "TND requires int32 cu_seqlens [B+1]."), return ge::GRAPH_FAILED);
        auto count = shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF(idx == 13 && count != prefixSize, OP_LOGE(opName, "TND cu_seqlens batch counts must match."),
                    return ge::GRAPH_FAILED);
        prefixSize = count;
    }
    OP_CHECK_IF(prefixSize - 1 > queryShape.GetDim(0) || prefixSize - 1 > keyShape.GetDim(0),
                OP_LOGE(opName, "Positive TND lengths require Tq/Tkv >= B."), return ge::GRAPH_FAILED);
    auto lse = context->GetInputShape(SOFTMAX_LSE);
    OP_CHECK_IF(lse == nullptr || lse->GetStorageShape().GetDimNum() != 2 ||
                    lse->GetStorageShape().GetDim(0) != queryShape.GetDim(1) ||
                    lse->GetStorageShape().GetDim(1) != queryShape.GetDim(0),
                OP_LOGE(opName, "TND LSE must have shape [N,Tq]."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseAttrs(gert::TilingContext *context, const string &opName)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *quantMode = attrs->GetAttrPointer<int64_t>(QUANT_MODE_IDX);
    OP_CHECK_IF(quantMode == nullptr, OP_LOGE(opName, "quant_mode must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*quantMode != 0, OP_LOGE(opName, "quant_mode only supports 0, got %ld.", *quantMode),
                return ge::GRAPH_FAILED);
    const int64_t *maxSeqlenQ = attrs->GetAttrPointer<int64_t>(MAX_SEQLEN_Q);
    OP_CHECK_IF(maxSeqlenQ != nullptr && (IsTnd(context) ? (*maxSeqlenQ != -1 && *maxSeqlenQ <= 0) : *maxSeqlenQ != -1),
                OP_LOGE(opName, "maxSeqlenQ not support."), return ge ::GRAPH_FAILED);
    const int64_t *maxSeqlenKV = attrs->GetAttrPointer<int64_t>(MAX_SEQLEN_KV);
    OP_CHECK_IF(
        maxSeqlenKV != nullptr && (IsTnd(context) ? (*maxSeqlenKV != -1 && *maxSeqlenKV <= 0) : *maxSeqlenKV != -1),
        OP_LOGE(opName, "maxSeqlenKV not support."), return ge::GRAPH_FAILED);
    const int64_t *maskMode = attrs->GetAttrPointer<int64_t>(MASK_MODE);
    int64_t maskModeValue = (maskMode != nullptr) ? *maskMode : 0;
    OP_CHECK_IF(maskModeValue != 0 && maskModeValue != 3 && maskModeValue != 4,
                OP_LOGE(opName, "maskMode only support 0, 3, 4, now is %ld.", maskModeValue), return ge::GRAPH_FAILED);
    const int64_t *winLeft = attrs->GetAttrPointer<int64_t>(WIN_LEFT);
    const int64_t *winRight = attrs->GetAttrPointer<int64_t>(WIN_RIGHT);
    int64_t winLeftValue = (winLeft != nullptr) ? *winLeft : -1;
    int64_t winRightValue = (winRight != nullptr) ? *winRight : -1;
    if (maskModeValue == 4) {
        // -1表示该方向不限窗, host后续按+inf处理; 其余负值对应的负token场景暂不支持
        OP_CHECK_IF(winLeftValue < -1 || winRightValue < -1,
                    OP_LOGE(opName,
                            "maskMode 4 requires winLeft >= 0 and winRight >= 0, or -1 for unlimited, "
                            "now is [%ld, %ld].",
                            winLeftValue, winRightValue),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(winLeftValue != -1 || winRightValue != -1,
                    OP_LOGE(opName, "winLeft/winRight only support in maskMode 4, now is [%ld, %ld].", winLeftValue,
                            winRightValue),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateMainInputShapes(gert::TilingContext *context, const string &opName)
{
    const auto *attrs = context->GetAttrs();
    const char *layoutQ = attrs->GetAttrPointer<char>(LAYOUT_Q);
    const char *layoutKV = attrs->GetAttrPointer<char>(LAYOUT_KV);
    if (layoutQ == nullptr && layoutKV == nullptr) {
        layoutQ = "BSND";
    } else {
        OP_CHECK_IF(layoutQ == nullptr || layoutKV == nullptr || std::string(layoutQ) != layoutKV,
                    OP_LOGE(opName, "layout_q and layout_kv must be the same."), return ge::GRAPH_FAILED);
    }
    const std::string layout(layoutQ);
    OP_CHECK_IF(layout != "BSND" && layout != "BNSD" && layout != "TND",
                OP_LOGE(opName, "Unsupported layout %s; expected BSND, BNSD or TND.", layoutQ),
                return ge::GRAPH_FAILED);

    const size_t expectedRank = layout == "TND" ? 3 : 4;
    const size_t headAxis = layout == "BSND" ? 2 : 1;
    const char *inputNames[MAIN_INPUT_COUNT] = {"q", "k", "v", "dout", "attn_out"};
    const gert::StorageShape *inputShapes[MAIN_INPUT_COUNT] = {};
    for (int64_t idx = 0; idx < MAIN_INPUT_COUNT; ++idx) {
        inputShapes[idx] = context->GetInputShape(idx);
        OP_CHECK_IF(inputShapes[idx] == nullptr, OP_LOGE(opName, "%s requires a shape.", inputNames[idx]),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputShapes[idx]->GetStorageShape().GetDimNum() != expectedRank,
                    OP_LOGE(opName, "%s must have rank %zu for layout %s.", inputNames[idx], expectedRank, layoutQ),
                    return ge::GRAPH_FAILED);
    }

    const int64_t n1 = inputShapes[QUERY_IDX]->GetStorageShape().GetDim(headAxis);
    const int64_t n2 = inputShapes[KEY_IDX]->GetStorageShape().GetDim(headAxis);
    OP_CHECK_IF(n2 <= 0, OP_LOGE(opName, "N2 (key head count) must be greater than 0, got %ld.", n2),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(n1 != n2, OP_LOGE(opName, "N1 must equal N2, got N1=%ld and N2=%ld.", n1, n2), return ge::GRAPH_FAILED);
    for (int64_t idx = 0; idx < MAIN_INPUT_COUNT; ++idx) {
        const auto &shape = inputShapes[idx]->GetStorageShape();
        for (size_t axis = 0; axis < expectedRank; ++axis) {
            OP_CHECK_IF(shape.GetDim(axis) <= 0,
                        OP_LOGE(opName, "%s dimension %zu must be positive, got %ld.", inputNames[idx], axis,
                                shape.GetDim(axis)),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(shape.GetDim(expectedRank - 1) != SUPPORTED_HEAD_DIM,
                    OP_LOGE(opName, "quant_mode=0 requires %s D=128, got %ld.", inputNames[idx],
                            shape.GetDim(expectedRank - 1)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateSequsedInput(gert::TilingContext *context, const string &opName, int64_t inputIdx,
                                            int64_t batchSize)
{
    auto desc = context->GetOptionalInputDesc(inputIdx);
    if (desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto shape = context->GetOptionalInputShape(inputIdx);
    OP_CHECK_IF(shape == nullptr, OP_LOGE(opName, "seqused input %ld requires a shape.", inputIdx),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE(opName, "seqused input %ld must have int32 dtype.", inputIdx), return ge::GRAPH_FAILED);
    const auto &storageShape = shape->GetStorageShape();
    OP_CHECK_IF(storageShape.GetDimNum() != 1, OP_LOGE(opName, "seqused input %ld must have rank 1.", inputIdx),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(storageShape.GetDim(0) != batchSize,
                OP_LOGE(opName, "seqused input %ld must contain one length per batch (%ld).", inputIdx, batchSize),
                return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(queryDesc->GetDataType() != ge::DT_HIFLOAT8, OP_LOGE(opName, "query must have HIFLOAT8 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDesc->GetDataType() != ge::DT_HIFLOAT8, OP_LOGE(opName, "key must have HIFLOAT8 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueDesc->GetDataType() != ge::DT_HIFLOAT8, OP_LOGE(opName, "value must have HIFLOAT8 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(doDesc->GetDataType() != ge::DT_HIFLOAT8, OP_LOGE(opName, "dout must have HIFLOAT8 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxLseDesc->GetDataType() != ge::DT_FLOAT, OP_LOGE(opName, "softmax_lse must have FLOAT32 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attnOutDesc->GetDataType() != ge::DT_BF16, OP_LOGE(opName, "attn_out must have BF16 dtype."),
                return ge::GRAPH_FAILED);
    if (ValidateMainInputShapes(context, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(!IsTnd(context) && cuSeqlensQDesc != nullptr, OP_LOGE(opName, "cuSeqlensQ not support."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsTnd(context) && cuSeqlensKVDesc != nullptr, OP_LOGE(opName, "cuSeqlensKV not support."),
                return ge::GRAPH_FAILED);
    if (IsTnd(context) && ValidateTnd(context, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // sequsedQ/sequsedKV describe the effective prefix of each padded or packed batch segment.
    // 具体的 varlen 分核由 quant_flash_attn_metadata 算子在 device 侧计算后经 metadata 下发。
    if (sequsedQDesc != nullptr || sequsedKVDesc != nullptr) {
        auto queryShape = context->GetInputShape(QUERY_IDX);
        OP_CHECK_IF(queryShape == nullptr || queryShape->GetStorageShape().GetDimNum() != (IsTnd(context) ? 3 : 4),
                    OP_LOGE(opName, "seqused requires query rank matching TND or padded layout."),
                    return ge::GRAPH_FAILED);
        const int64_t batchSize = IsTnd(context) ?
                                      context->GetOptionalInputShape(CU_SEQLENS_Q)->GetStorageShape().GetDim(0) - 1 :
                                      queryShape->GetStorageShape().GetDim(0);
        if (ValidateSequsedInput(context, opName, SEQUSED_Q, batchSize) != ge::GRAPH_SUCCESS ||
            ValidateSequsedInput(context, opName, SEQUSED_KV, batchSize) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_CHECK_IF(sinksDesc != nullptr, OP_LOGE(opName, "sinks not support."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataDesc == nullptr, OP_LOGE(opName, "metadata must be provided."), return ge::GRAPH_FAILED);
    auto metadataShape = context->GetOptionalInputShape(METADATA);
    OP_CHECK_IF(metadataShape == nullptr, OP_LOGE(opName, "metadata requires a shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataDesc->GetDataType() != ge::DT_INT32, OP_LOGE(opName, "metadata must have int32 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataShape->GetStorageShape().GetDimNum() != 2, OP_LOGE(opName, "metadata must have rank 2."),
                return ge::GRAPH_FAILED);
    // TND/sparse seqused 预留 17+10B（TND line swizzle）；dense padded seqused 为 17+3B。
    int64_t requiredMetadataRowSize = 16;
    if (IsTnd(context)) {
        auto cuSeqlensQ = context->GetOptionalInputShape(CU_SEQLENS_Q);
        OP_CHECK_IF(cuSeqlensQ == nullptr, OP_LOGE(opName, "TND requires cu_seqlens_q."), return ge::GRAPH_FAILED);
        requiredMetadataRowSize = 17 + 10 * (cuSeqlensQ->GetStorageShape().GetDim(0) - 1);
    } else if (sequsedQDesc != nullptr || sequsedKVDesc != nullptr) {
        const int64_t batchSize = context->GetInputShape(QUERY_IDX)->GetStorageShape().GetDim(0);
        const auto maskMode = context->GetAttrs()->GetAttrPointer<int64_t>(MASK_MODE);
        const bool isSparse = maskMode != nullptr && (*maskMode == 3 || *maskMode == 4);
        // Header, batch prefix (B + 1), and two dense or eight sparse B-arrays.
        requiredMetadataRowSize = 17 + (isSparse ? 10 : 3) * batchSize;
    }
    OP_CHECK_IF(metadataShape->GetStorageShape().GetDim(0) != 2 ||
                    metadataShape->GetStorageShape().GetDim(1) < requiredMetadataRowSize,
                OP_LOGE(opName, "metadata must have shape [2, row size >= %ld].", requiredMetadataRowSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dqDesc == nullptr, OP_LOGE(opName, "dq must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dkDesc == nullptr, OP_LOGE(opName, "dk must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dvDesc == nullptr, OP_LOGE(opName, "dv must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dsinkDesc == nullptr, OP_LOGE(opName, "dSink must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dqDesc->GetDataType() != ge::DT_BF16, OP_LOGE(opName, "dq must have BF16 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dkDesc->GetDataType() != ge::DT_BF16, OP_LOGE(opName, "dk must have BF16 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dvDesc->GetDataType() != ge::DT_BF16, OP_LOGE(opName, "dv must have BF16 dtype."),
                return ge::GRAPH_FAILED);
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
