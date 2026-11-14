/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../mixed_quant_sparse_flash_mla_tiling.h"
#include "../../op_kernel/arch22/mixed_quant_sparse_flash_mla_tiling_data_arch22.h"
#include "../../op_kernel/arch22/mixed_quant_sparse_flash_mla_template_tiling_key_arch22.h"
#include <algorithm>
#include <string>
#include <vector>

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus MixedQuantSparseFlashMlaTiling::DoInt8CsaTiling(MQSMLATilingInfo* info)
{
    OP_CHECK_IF(info->batchConsistency, OP_LOGE(info->opName, "INT8 CSA does not support batch-consistency mode"),
                return ge::GRAPH_FAILED);
    // Reuse the arch22 CSA tile sizes and bounded QK/PV/merge workspaces.
    if (DoTurboQuantTiling(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto* data = context_->GetTilingData<MixedQuantSparseFlashMlaTqTilingData>();
    OP_CHECK_IF(data == nullptr, OP_LOGE(info->opName, "tiling data is nullptr"), return ge::GRAPH_FAILED);
    // Use the mode-1 execution template after validating quant_mode=1 and INT8 inputs.
    data->tqBaseParams.kvQuantMode = QUANT_CONTIGUOUS;
    auto& p = data->int8Params;
    p.batch = info->bSize;
    p.heads = info->gSize;
    p.qSeq = info->s1Size;
    p.qTokens = info->qLayout == MQSMLALayout::TND ? info->qTSize : static_cast<uint64_t>(info->bSize) * info->s1Size;
    p.oriSeq = info->s2Size;
    p.cmpSeq = info->cmpS2Size;
    p.oriBlockSize = info->oriBlockSize;
    p.cmpBlockSize = info->cmpBlockSize;
    p.oriTableSize = info->oriMaxBlockNumPerBatch;
    p.cmpTableSize = info->cmpMaxBlockNumPerBatch;
    p.oriBlocks = info->oriKvStorageShape.GetDim(0);
    p.cmpBlocks = info->cmpKvStorageShape.GetDim(0);
    p.oriStride = info->oriKvStride;
    p.cmpStride = info->cmpKvStride;
    p.topk = info->cmpSparseBlockCount;
    p.cmpRatio = info->cmpRatio;
    p.oriMask = info->oriMaskMode;
    p.cmpMask = info->cmpMaskMode;
    p.winLeft = info->oriWinLeft;
    p.winRight = info->oriWinRight;
    p.softmaxScale = info->softmaxScale;
    p.returnLse = info->returnSoftmaxLse;
    context_->SetTilingKey(GET_TPL_TILING_KEY(0U, static_cast<uint32_t>(info->qLayout),
                                              static_cast<uint32_t>(info->kvLayout), CSA_TEMPLATE, 0U, QUANT_CONTIGUOUS,
                                              DTYPE_INT8, 0U, 0U, 0U, TOPK_INDEX));
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MixedQuantSparseFlashMlaTiling::DoTurboQuantTiling(MQSMLATilingInfo* tilingInfo)
{
    constexpr uint32_t S2_BASE_SIZE = 512;
    constexpr uint32_t BYTE_BLOCK = 32;
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
    constexpr uint32_t VEC1_RES_ELEM_SIZE = 2;
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;
    constexpr uint32_t VEC2_RES_ELEM_SIZE = 4;
    constexpr uint32_t PRELOAD_NUM = 2;
    constexpr uint32_t MERGE_CACHE_GM_BUF_NUM = 3;
    constexpr uint32_t MERGE_CACHE_ELEM_SIZE = 2;
    auto align = [](uint32_t value, uint32_t alignment) { return (value + alignment - 1) / alignment * alignment; };

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));

    const uint32_t mBaseSize = tilingInfo->gSize;
    const uint32_t s2BaseSizeAlign = align(S2_BASE_SIZE, BYTE_BLOCK);
    // The kernel splits the full TND M dimension into tiles no larger than mBaseSize.
    const uint32_t cubeMSize = mBaseSize;
    const uint32_t mmResUbSize = s2BaseSizeAlign * align(cubeMSize, 16U);
    const uint32_t bmm2ResUbSize = align(tilingInfo->qkHeadDim, BYTE_BLOCK) * align(cubeMSize, 16U);

    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += static_cast<size_t>(PRELOAD_NUM) * mmResUbSize * (MM1_RES_ELEM_SIZE + VEC1_RES_ELEM_SIZE) * aicNum;
    workspaceSize +=
        static_cast<size_t>(PRELOAD_NUM) * bmm2ResUbSize * (MM2_RES_ELEM_SIZE + VEC2_RES_ELEM_SIZE) * aicNum;
    workspaceSize += static_cast<size_t>(MERGE_CACHE_GM_BUF_NUM) * S2_BASE_SIZE * tilingInfo->qkHeadDim *
                     MERGE_CACHE_ELEM_SIZE * aicNum;
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        const std::string errorReason =
            "Failed to obtain workspace_sizes from GE context: GetWorkspaceSizes(1) returned nullptr.";
        OP_LOGE_LIBOPAPI_REPORT(tilingInfo->opName, "%s", errorReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", std::vector<const char*>({"opName", "reason"}),
                                  std::vector<const char*>({tilingInfo->opName, errorReason.c_str()}));
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = workspaceSize;

    auto* tqTilingData = context_->GetTilingData<MixedQuantSparseFlashMlaTqTilingData>();
    if (tqTilingData == nullptr) {
        const std::string errorReason = "Failed to obtain tiling data from GE context: GetTilingData returned nullptr.";
        OP_LOGE_LIBOPAPI_REPORT(tilingInfo->opName, "%s", errorReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", std::vector<const char*>({"opName", "reason"}),
                                  std::vector<const char*>({tilingInfo->opName, errorReason.c_str()}));
        return ge::GRAPH_FAILED;
    }
    tqTilingData->tqBaseParams.batchSize = tilingInfo->bSize;
    tqTilingData->tqBaseParams.qSeqSize = tilingInfo->s1Size;
    tqTilingData->tqBaseParams.kvSeqSize = tilingInfo->s2Size;
    tqTilingData->tqBaseParams.paBlockSize = tilingInfo->oriBlockSize;
    tqTilingData->tqBaseParams.oriBlockSize = tilingInfo->oriBlockSize;
    tqTilingData->tqBaseParams.cmpBlockSize = tilingInfo->cmpBlockSize;
    tqTilingData->tqBaseParams.oriMaxBlockNumPerBatch = tilingInfo->oriMaxBlockNumPerBatch;
    tqTilingData->tqBaseParams.nNumOfQInOneGroup = tilingInfo->gSize;
    tqTilingData->tqBaseParams.actualLenDimsQ = tilingInfo->actualLenDimsQ;
    tqTilingData->tqBaseParams.actualLenDimsKV = tilingInfo->actualLenDimsKV;
    tqTilingData->tqBaseParams.softmaxScale = tilingInfo->softmaxScale;
    tqTilingData->tqBaseParams.outputLayout = static_cast<uint32_t>(tilingInfo->outLayout);
    tqTilingData->tqBaseParams.oriMaskMode = tilingInfo->oriMaskMode;
    tqTilingData->tqBaseParams.oriKvStride0 = tilingInfo->oriKvStride;
    tqTilingData->tqBaseParams.oriWinLeft = tilingInfo->oriWinLeft;
    tqTilingData->tqBaseParams.oriWinRight = tilingInfo->oriWinRight;
    tqTilingData->tqBaseParams.sparseBlockSize = tilingInfo->sparseBlockSize;
    tqTilingData->tqBaseParams.usedCoreNum = aicNum;
    tqTilingData->tqBaseParams.mmResUbSize = mmResUbSize;
    tqTilingData->tqBaseParams.bmm2ResUbSize = bmm2ResUbSize;
    tqTilingData->tqBaseParams.mBaseSize = mBaseSize;
    tqTilingData->tqBaseParams.s2BaseSize = S2_BASE_SIZE;
    tqTilingData->tqBaseParams.returnSoftmaxLse = tilingInfo->returnSoftmaxLse;
    tqTilingData->tqBaseParams.kvQuantMode = static_cast<uint32_t>(tilingInfo->quantMode);
    tqTilingData->tqCmpParams.cmpMaxBlockNumPerBatch = tilingInfo->cmpMaxBlockNumPerBatch;
    tqTilingData->tqCmpParams.sparseBlockCount = tilingInfo->cmpSparseBlockCount;
    tqTilingData->tqCmpParams.cmpRatio = tilingInfo->cmpRatio;
    tqTilingData->tqCmpParams.cmpMaskMode = tilingInfo->cmpMaskMode;
    tqTilingData->tqCmpParams.cmpKvStride0 = tilingInfo->cmpKvStride;

    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(0U, static_cast<uint32_t>(tilingInfo->qLayout), static_cast<uint32_t>(tilingInfo->kvLayout),
                           CSA_TEMPLATE, 0U, static_cast<uint32_t>(tilingInfo->quantMode), DTYPE_FP8_E4M3FN, 0U, 0U, 0U,
                           static_cast<uint32_t>(tilingInfo->topkValueMode));
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
