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
 * \file dense_lightning_indexer_softmax_lse_v2_tiling.cpp
 * \brief
 */

#include "dense_lightning_indexer_softmax_lse_v2_tiling.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

ge::graphStatus DenseLISoftmaxLseV2CompileInfo::ParamCheck(
    gert::TilingContext *context, int64_t layout, int64_t maskMode, int64_t cmpRatio, const std::string &layoutQStr,
    const std::string &layoutKStr, int64_t bSize, int64_t s1Size, int64_t s2Size, int64_t n1Size, int64_t n2Size,
    int64_t dSize, int64_t keyB, int64_t keyD, int64_t weightB, int64_t weightS1, int64_t weightN1)
{
    OP_CHECK_IF(layoutQStr != layoutKStr,
                OP_LOGE(context->GetNodeName(), "layoutQ and layoutK must be the same, but got %s and %s.",
                        layoutQStr.c_str(), layoutKStr.c_str()),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(maskMode != 0 && maskMode != 3,
                OP_LOGE(context->GetNodeName(), "mask_mode only supports 0 or 3, but got %ld.", maskMode),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(cmpRatio < 1 || cmpRatio > 128,
                OP_LOGE(context->GetNodeName(), "cmp_ratio must be in [1, 128], but got %ld.", cmpRatio),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(dSize != 128, OP_LOGE(context->GetNodeName(), "head_dim must be 128, but got %ld.", dSize),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(keyD != dSize,
                OP_LOGE(context->GetNodeName(),
                        "keyIndex D dim must be equal to queryIndex D dim, "
                        "but got qD=%ld kD=%ld.",
                        dSize, keyD),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(n1Size < 1 || n1Size > 128,
                OP_LOGE(context->GetNodeName(), "num_heads_q must be in [1, 128], but got %ld.", n1Size),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(n2Size != 1, OP_LOGE(context->GetNodeName(), "num_heads_k must be 1, but got %ld.", n2Size),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(bSize <= 0, OP_LOGE(context->GetNodeName(), "batch_size must be positive, but got %ld.", bSize),
                return ge::GRAPH_PARAM_INVALID);

    if (layout == 0) {
        OP_CHECK_IF(keyB != bSize,
                    OP_LOGE(context->GetNodeName(),
                            "keyIndex B dim must be equal to queryIndex B dim, "
                            "but got qB=%ld kB=%ld.",
                            bSize, keyB),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(s1Size <= 0, OP_LOGE(context->GetNodeName(), "S1 must be positive for BSND, but got %ld.", s1Size),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(s2Size < 0,
                    OP_LOGE(context->GetNodeName(), "S2 must be non-negative for BSND, but got %ld.", s2Size),
                    return ge::GRAPH_PARAM_INVALID);
    } else {
        OP_CHECK_IF(s1Size <= 0, OP_LOGE(context->GetNodeName(), "T1 must be positive for TND, but got %ld.", s1Size),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(s2Size < 0,
                    OP_LOGE(context->GetNodeName(), "T2 must be non-negative for TND, but got %ld.", s2Size),
                    return ge::GRAPH_PARAM_INVALID);
    }

    if (layout == 1) {
        auto cuSeqLensQShape = context->GetOptionalInputShape(3);
        OP_CHECK_IF(cuSeqLensQShape == nullptr,
                    OP_LOGE(context->GetNodeName(), "cu_seqlens_q must be provided for TND layout."),
                    return ge::GRAPH_PARAM_INVALID);
        auto cuSeqLensKShape = context->GetOptionalInputShape(4);
        OP_CHECK_IF(cuSeqLensKShape == nullptr,
                    OP_LOGE(context->GetNodeName(), "cu_seqlens_k must be provided for TND layout."),
                    return ge::GRAPH_PARAM_INVALID);
    }

    if (maskMode == 3 && cmpRatio > 1) {
        auto cmpResidualKShape = context->GetOptionalInputShape(7);
        OP_CHECK_IF(cmpResidualKShape == nullptr,
                    OP_LOGE(context->GetNodeName(), "cmp_residual_k must be provided when "
                                                    "mask_mode=3 and cmp_ratio>1."),
                    return ge::GRAPH_PARAM_INVALID);
    }

    auto cuSeqLensQShape = context->GetOptionalInputShape(3);
    if (cuSeqLensQShape != nullptr) {
        OP_CHECK_IF(cuSeqLensQShape->GetStorageShape().GetDim(0) != bSize + 1,
                    OP_LOGE(context->GetNodeName(),
                            "cu_seqlens_q length must be batch_size + 1, "
                            "but got %ld and batch_size=%ld.",
                            cuSeqLensQShape->GetStorageShape().GetDim(0), bSize),
                    return ge::GRAPH_PARAM_INVALID);
    }

    auto cuSeqLensKShape = context->GetOptionalInputShape(4);
    if (cuSeqLensKShape != nullptr) {
        OP_CHECK_IF(cuSeqLensKShape->GetStorageShape().GetDim(0) != bSize + 1,
                    OP_LOGE(context->GetNodeName(),
                            "cu_seqlens_k length must be batch_size + 1, "
                            "but got %ld and batch_size=%ld.",
                            cuSeqLensKShape->GetStorageShape().GetDim(0), bSize),
                    return ge::GRAPH_PARAM_INVALID);
    }

    auto seqUsedQShape = context->GetOptionalInputShape(5);
    if (seqUsedQShape != nullptr) {
        OP_CHECK_IF(seqUsedQShape->GetStorageShape().GetDim(0) != bSize,
                    OP_LOGE(context->GetNodeName(),
                            "seqused_q length must be batch_size, "
                            "but got %ld and batch_size=%ld.",
                            seqUsedQShape->GetStorageShape().GetDim(0), bSize),
                    return ge::GRAPH_PARAM_INVALID);
    }

    auto seqUsedKShape = context->GetOptionalInputShape(6);
    if (seqUsedKShape != nullptr) {
        OP_CHECK_IF(seqUsedKShape->GetStorageShape().GetDim(0) != bSize,
                    OP_LOGE(context->GetNodeName(),
                            "seqused_k length must be batch_size, "
                            "but got %ld and batch_size=%ld.",
                            seqUsedKShape->GetStorageShape().GetDim(0), bSize),
                    return ge::GRAPH_PARAM_INVALID);
    }

    auto cmpResidualKShape = context->GetOptionalInputShape(7);
    if (cmpResidualKShape != nullptr) {
        OP_CHECK_IF(cmpResidualKShape->GetStorageShape().GetDim(0) != bSize,
                    OP_LOGE(context->GetNodeName(),
                            "cmp_residual_k length must be batch_size, "
                            "but got %ld and batch_size=%ld.",
                            cmpResidualKShape->GetStorageShape().GetDim(0), bSize),
                    return ge::GRAPH_PARAM_INVALID);
    }

    if (layout == 0) {
        OP_CHECK_IF(weightB != bSize,
                    OP_LOGE(context->GetNodeName(), "weight B dim must be %ld, but got %ld.", bSize, weightB),
                    return ge::GRAPH_PARAM_INVALID);
    }
    OP_CHECK_IF(weightS1 != s1Size,
                OP_LOGE(context->GetNodeName(), "weight S1 dim must be %ld, but got %ld.", s1Size, weightS1),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(weightN1 != n1Size,
                OP_LOGE(context->GetNodeName(), "weight N1 dim must be %ld, but got %ld.", n1Size, weightN1),
                return ge::GRAPH_PARAM_INVALID);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DenseLightningIndexerSoftmaxLseV2TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // Extract layout attr
    auto attrs = context->GetAttrs();
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t layout = 0; // 0=BSND, 1=TND
    std::string layoutQStr = "BSND";
    std::string layoutKStr = "BSND";
    if (attrs != nullptr) {
        auto maskModePtr = attrs->GetAttrPointer<int64_t>(2);
        auto cmpRatioPtr = attrs->GetAttrPointer<int64_t>(3);
        if (maskModePtr != nullptr) {
            maskMode = *maskModePtr;
        }
        if (cmpRatioPtr != nullptr) {
            cmpRatio = *cmpRatioPtr;
        }
        // Attr indices: 0=layoutQ(String), 1=layoutK(String), 2=maskMode(Int), 3=cmpRatio(Int)
        auto layoutQPtr = attrs->GetAttrPointer<char>(0);
        if (layoutQPtr != nullptr) {
            layoutQStr = std::string(layoutQPtr);
            if (layoutQStr == "TND") {
                layout = 1;
            }
        }
        auto layoutKPtr = attrs->GetAttrPointer<char>(1);
        if (layoutKPtr != nullptr) {
            layoutKStr = std::string(layoutKPtr);
        }
    }

    // Extract runtime dimensions from input shapes.
    auto queryShape = context->GetInputShape(0);
    auto keyShape = context->GetInputShape(1);
    auto weightShape = context->GetInputShape(2);
    OP_CHECK_IF(queryShape == nullptr || keyShape == nullptr || weightShape == nullptr,
                OP_LOGE(context->GetNodeName(), "query_index, key_index or weight shape is null."),
                return ge::GRAPH_PARAM_INVALID);
    int64_t bSize;
    int64_t s1Size;
    int64_t n1Size;
    int64_t n2Size;
    int64_t dSize;
    int64_t s2Size;
    int64_t keyB;
    int64_t keyD;
    int64_t weightB;
    int64_t weightS1;
    int64_t weightN1;

    if (layout == 1) {
        // TND: query [T1, N1, D], key [T2, N2, D], weight [T1, N1]
        s1Size = queryShape->GetStorageShape().GetDim(0);
        n1Size = queryShape->GetStorageShape().GetDim(1);
        dSize = queryShape->GetStorageShape().GetDim(2);
        s2Size = keyShape->GetStorageShape().GetDim(0);
        n2Size = keyShape->GetStorageShape().GetDim(1);
        keyD = keyShape->GetStorageShape().GetDim(2);
        keyB = 1;
        weightS1 = weightShape->GetStorageShape().GetDim(0);
        weightN1 = weightShape->GetStorageShape().GetDim(1);
        weightB = 1;
        // B from cu_seq_lens_q input (index 3, shape [B+1])
        auto cuSeqLensQShape = context->GetOptionalInputShape(3);
        if (cuSeqLensQShape != nullptr) {
            bSize = cuSeqLensQShape->GetStorageShape().GetDim(0) - 1;
        } else {
            bSize = 1;
        }
    } else {
        // BSND: query [B, S1, N1, D], key [B, S2, N2, D], weight [B, S1, N1]
        bSize = queryShape->GetStorageShape().GetDim(0);
        s1Size = queryShape->GetStorageShape().GetDim(1);
        n1Size = queryShape->GetStorageShape().GetDim(2);
        dSize = queryShape->GetStorageShape().GetDim(3);
        s2Size = keyShape->GetStorageShape().GetDim(1);
        n2Size = keyShape->GetStorageShape().GetDim(2);
        keyB = keyShape->GetStorageShape().GetDim(0);
        keyD = keyShape->GetStorageShape().GetDim(3);
        weightB = weightShape->GetStorageShape().GetDim(0);
        weightS1 = weightShape->GetStorageShape().GetDim(1);
        weightN1 = weightShape->GetStorageShape().GetDim(2);
    }

    // Unified parameter validation
    auto ret = DenseLISoftmaxLseV2CompileInfo::ParamCheck(context, layout, maskMode, cmpRatio, layoutQStr, layoutKStr,
                                                          bSize, s1Size, s2Size, n1Size, n2Size, dSize, keyB, keyD,
                                                          weightB, weightS1, weightN1);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "DenseLISoftmaxLseV2CompileInfo::ParamCheck failed."),
                return ge::GRAPH_PARAM_INVALID);

    // DenseLILseV2Tiling comes from the pypto kernel (.py dataclass) via force-include.
    DenseLILseV2Tiling *tilingData = context->GetTilingData<DenseLILseV2Tiling>();
    tilingData->b = bSize;
    tilingData->s1 = s1Size;
    tilingData->s2 = s2Size;
    tilingData->n1 = n1Size;
    tilingData->d = dSize;

    // Determine if S2 exceeds reduce_sum_vec UB capacity.
    constexpr int64_t TKV = 128;
    constexpr int64_t TKV_HALF = TKV / 2;
    constexpr int64_t KV_CACHE_NUM = 46832;
    int64_t s2Tiles = (s2Size + TKV - 1) / TKV;
    int64_t maxUbS2Tiles = KV_CACHE_NUM / TKV_HALF;
    int64_t isLongS2 = (s2Tiles > maxUbS2Tiles) ? 1 : 0;

    tilingData->cmp_ratio = cmpRatio;

    // Check if optional inputs seq_used_q (index 5) and seq_used_k (index 6) are provided
    int64_t hasSeqUsedQ = (context->GetOptionalInputShape(5) != nullptr) ? 1 : 0;
    int64_t hasSeqUsedK = (context->GetOptionalInputShape(6) != nullptr) ? 1 : 0;

    // TilingKey: bit0=is_long_s2, bit1=has_seq_used_q, bit2=has_seq_used_k,
    //            bits3-4=mask_mode, bit5=layout
    uint64_t tilingKey = static_cast<uint64_t>(isLongS2) | (static_cast<uint64_t>(hasSeqUsedQ) << 1) |
                         (static_cast<uint64_t>(hasSeqUsedK) << 2) | (static_cast<uint64_t>(maskMode) << 3) |
                         (static_cast<uint64_t>(layout) << 5);
    context->SetTilingKey(tilingKey);

    // Workspace: system reserve + workspace_max [1,72] FP32 (288 bytes) + workspace_sum [1,72] FP32 (288 bytes)
    //            + workspace [36, S2] FP32 (for long S2 reduce_sum spill).
    constexpr size_t WS_MAX_BYTES = 288;
    constexpr int64_t WS_MAX_CORES = 36;
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += WS_MAX_BYTES * 2;
    if (isLongS2) {
        workspaceSize += static_cast<size_t>(WS_MAX_CORES) * static_cast<size_t>(s2Size) * sizeof(float);
    }
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "DenseLightningIndexerSoftmaxLseV2 tiling completed: blockDim=%u, aivNum=%u, aicNum=%u, "
            "layout=%ld, B=%ld, S1=%ld, S2=%ld, N1=%ld, N2=%ld, D=%ld, isLongS2=%ld, "
            "maskMode=%ld, cmpRatio=%ld, hasSeqUsedQ=%ld, hasSeqUsedK=%ld, workspaceSize=%zu.",
            blockDim, aivNum, aicNum, layout, bSize, s1Size, s2Size, n1Size, n2Size, dSize, isLongS2, maskMode,
            cmpRatio, hasSeqUsedQ, hasSeqUsedK, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDenseLightningIndexerSoftmaxLseV2(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DenseLightningIndexerSoftmaxLseV2)
    .Tiling(DenseLightningIndexerSoftmaxLseV2TilingFunc)
    .TilingParse<DenseLISoftmaxLseV2CompileInfo>(TilingParseForDenseLightningIndexerSoftmaxLseV2);

} // namespace optiling
