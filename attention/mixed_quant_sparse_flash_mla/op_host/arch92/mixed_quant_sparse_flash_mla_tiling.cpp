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
#include "../../op_kernel/arch92/mixed_quant_sparse_flash_mla_template_tiling_key_arch92.h"
#include <algorithm>

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus MixedQuantSparseFlashMlaTiling::DoA6CsaTiling(MQSMLATilingInfo* tilingInfo)
{
    if (tilingInfo->opParamInfo.cmpKv.tensor == nullptr) {
        OP_CHECK_IF(tilingInfo->opParamInfo.cmpSparseIndices.tensor != nullptr,
                    OP_LOGE("MixedQuantSparseFlashMla", "cmpSparseIndices must be empty when cmpKv is not provided."),
                    return ge::GRAPH_FAILED);
        perfMode_ = QSMLATemplateMode::SWA_TEMPLATE_MODE;
    } else if (tilingInfo->opParamInfo.cmpSparseIndices.tensor != nullptr) {
        perfMode_ = QSMLATemplateMode::CSA_TEMPLATE_MODE;
    } else {
        perfMode_ = QSMLATemplateMode::HCA_TEMPLATE_MODE;
    }
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOGI(tilingInfo->opName, "MQSMLA block dim: %u aiv Num: %u aic Num: %u.", blockDim, aivNum, aicNum);

    // -------------set workspacesize-----------------
    constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
    constexpr uint32_t M_BASE_SIZE = 64;   // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 128; // S2轴基本块大小
    constexpr uint32_t D_SIZE = 512;
    constexpr uint32_t VEC_RES_ELEM_SIZE = 2; // 2: fp16/bf16
    constexpr uint32_t TOPK_MAX_SIZE = 2048;  // TopK选取个数
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    bool isSplitG = tilingInfo->gSize > M_BASE_SIZE; // gSize超过64时采用Split-G
    if (isSplitG) {
        workspaceSize += (S2_BASE_SIZE * D_SIZE * VEC_RES_ELEM_SIZE * TRIPLE_BUFFER_NUM * (aicNum >> 1));
    } else {
        workspaceSize += (S2_BASE_SIZE * D_SIZE * VEC_RES_ELEM_SIZE * TRIPLE_BUFFER_NUM * aicNum);
    }
    constexpr uint32_t UB_SIZE = 376 * 1024; // UB大小共384KB,预留8k

    uint32_t totalBS1 =
        (tilingInfo->qLayout == MQSMLALayout::TND) ? tilingInfo->s1Size : (tilingInfo->bSize * tilingInfo->s1Size);
    workspaceSize += (totalBS1 * tilingInfo->cmpSparseBlockCount * sizeof(int64_t));
    size_t* workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.baseParams.set_batchSize(tilingInfo->bSize);
    tilingData_.baseParams.set_kvSeqSize(tilingInfo->s2Size);
    tilingData_.baseParams.set_qSeqSize(tilingInfo->s1Size);
    tilingData_.baseParams.set_cmpKvSeqSize(tilingInfo->cmpS2Size);
    tilingData_.baseParams.set_cmpSparseBlockCount(tilingInfo->cmpSparseBlockCount);
    tilingData_.baseParams.set_nNumOfQInOneGroup(tilingInfo->gSize);
    tilingData_.baseParams.set_paOriBlockSize(tilingInfo->oriBlockSize);
    tilingData_.baseParams.set_paCmpBlockSize(tilingInfo->cmpBlockSize);
    tilingData_.baseParams.set_oriMaxBlockNumPerBatch(tilingInfo->oriMaxBlockNumPerBatch);
    tilingData_.baseParams.set_cmpMaxBlockNumPerBatch(tilingInfo->cmpMaxBlockNumPerBatch);

    tilingData_.baseParams.set_tileSize(tilingInfo->tileSize);
    tilingData_.baseParams.set_ropeHeadDim(tilingInfo->ropeHeadDim);
    tilingData_.baseParams.set_softmaxScale(tilingInfo->softmaxScale);
    tilingData_.baseParams.set_oriKvStride(tilingInfo->oriKvStride);
    tilingData_.baseParams.set_cmpKvStride(tilingInfo->cmpKvStride);
    tilingData_.baseParams.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.baseParams.set_oriMaskMode(tilingInfo->oriMaskMode);
    tilingData_.baseParams.set_cmpMaskMode(tilingInfo->cmpMaskMode);
    tilingData_.baseParams.set_oriWinLeft(tilingInfo->oriWinLeft);
    tilingData_.baseParams.set_oriWinRight(tilingInfo->oriWinRight);
    tilingData_.baseParams.set_sparseBlockSize(tilingInfo->sparseBlockSize);
    tilingData_.baseParams.set_dSize(tilingInfo->dSize);
    tilingData_.baseParams.set_dSizeVInput(tilingInfo->dSizeVInput);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t qType = static_cast<uint32_t>(tilingInfo->qType);
    uint32_t oriKvType = static_cast<uint32_t>(tilingInfo->oriKvType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t qLayout = static_cast<uint32_t>(tilingInfo->qLayout);
    uint32_t inputKvLayout = static_cast<uint32_t>(tilingInfo->kvLayout);
    uint32_t blocksizeFlag = static_cast<uint32_t>((tilingInfo->cmpBlockSize & (tilingInfo->cmpBlockSize - 1)) ==
                                                   0); // blockSize2是否为2的幂次
    uint32_t vectorizeUbSize = tilingInfo->cmpMaxBlockNumPerBatch * sizeof(int32_t) +
                               tilingInfo->cmpSparseBlockCount * sizeof(int32_t) +
                               tilingInfo->cmpSparseBlockCount * sizeof(int64_t); // 物理地址计算向量化所需ub大小
    uint32_t vectorizeFlag =
        static_cast<uint32_t>((perfMode_ == QSMLATemplateMode::CSA_TEMPLATE_MODE) && (vectorizeUbSize <= UB_SIZE) &&
                              blocksizeFlag && tilingInfo->topkValueMode == 1); // 是否满足向量化条件
    uint32_t tilingKey = GET_TPL_TILING_KEY(0U, qLayout, inputKvLayout, static_cast<uint32_t>(perfMode_),
                                            static_cast<uint32_t>(tilingInfo->gSize > 64),
                                            ((oriKvType == ge::DT_FLOAT8_E4M3FN) ? DTYPE_FP8_E4M3FN : DTYPE_HIF8),
                                            vectorizeFlag, static_cast<uint32_t>(tilingInfo->topkValueMode));
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
