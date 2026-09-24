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
#include "../../op_kernel/arch35/mixed_quant_sparse_flash_mla_template_tiling_key_arch35.h"
#include <algorithm>

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus MixedQuantSparseFlashMlaTiling::DoCsaTiling(MQSMLATilingInfo *tilingInfo)
{
    if (tilingInfo->opParamInfo.cmpKv.tensor == nullptr) {
        OP_CHECK_IF(
            tilingInfo->opParamInfo.cmpSparseIndices.tensor != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("MixedQuantSparseFlashMla", "cmp_sparse_indices",
                                                     "Cmp_sparse_indices must be empty when cmpKv is not provided"),
            return ge::GRAPH_FAILED);
        if (tilingInfo->opParamInfo.oriSparseIndices.tensor == nullptr) {
            perfMode_ = QSMLATemplateMode::SWA_TEMPLATE_MODE;
        } else {
            perfMode_ = QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE;
        }
    } else if (tilingInfo->opParamInfo.cmpSparseIndices.tensor != nullptr) {
        if (tilingInfo->opParamInfo.oriSparseIndices.tensor == nullptr) {
            perfMode_ = QSMLATemplateMode::CSA_TEMPLATE_MODE;
        } else {
            perfMode_ = QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE;
        }
    } else {
        perfMode_ = QSMLATemplateMode::HCA_TEMPLATE_MODE;
    }
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOGI(tilingInfo->opName, "QSMLA block dim: %u aiv Num: %u aic Num: %u.", blockDim, aivNum, aicNum);

    // -------------set workspacesize-----------------
    constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
    constexpr uint32_t S2_BASE_SIZE = 128; // S2轴基本块大小
    constexpr uint32_t D_SIZE = 512;
    constexpr uint32_t VEC_RES_ELEM_SIZE = 2; // 2: fp16/bf16字节数
    constexpr uint32_t TOPK_MAX_SIZE = 2048;  // TopK选取个数
    constexpr uint32_t UB_SIZE = 184 * 1024;
    constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128;
    constexpr int64_t QUANT_CONTIGUOUS_MODE = 1;
    constexpr uint32_t MAX_S2_SPLIT_NUM = 2;      // 每核最多S2切分次数
    constexpr uint32_t FLOAT_ELEM_SIZE = 4;       // sizeof(float)
    constexpr uint32_t FD_BLOCK_ELEM = 8;         // FD广播份数
    constexpr uint32_t FD_MAX_SUM_REGION_NUM = 2; // max和sum两个区域
    constexpr uint32_t BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM = 33;
    uint32_t alignedOriSparseBlockCount = (tilingInfo->oriSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                          SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    uint32_t alignedCmpSparseBlockCount = (tilingInfo->cmpSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                          SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    uint64_t oriUbSize = static_cast<uint64_t>(tilingInfo->oriMaxBlockNumPerBatch) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedOriSparseBlockCount) * (sizeof(int32_t) + sizeof(int64_t));
    uint64_t cmpUbSize = static_cast<uint64_t>(tilingInfo->cmpMaxBlockNumPerBatch) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedCmpSparseBlockCount) * (sizeof(int32_t) + sizeof(int64_t));
    bool oriBlockSizePowerOfTwo =
        tilingInfo->oriBlockSize > 0 && (tilingInfo->oriBlockSize & (tilingInfo->oriBlockSize - 1)) == 0;
    bool cmpBlockSizePowerOfTwo =
        tilingInfo->cmpBlockSize > 0 && (tilingInfo->cmpBlockSize & (tilingInfo->cmpBlockSize - 1)) == 0;
    bool blockSizeSupported = (perfMode_ == QSMLATemplateMode::CSA_TEMPLATE_MODE && cmpBlockSizePowerOfTwo) ||
                              (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE && oriBlockSizePowerOfTwo) ||
                              (perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE && oriBlockSizePowerOfTwo &&
                               cmpBlockSizePowerOfTwo);
    uint64_t vectorizeUbSize =
        (perfMode_ == QSMLATemplateMode::CSA_TEMPLATE_MODE) ?
            cmpUbSize :
            ((perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) ? oriUbSize : std::max(oriUbSize, cmpUbSize));
    uint32_t vectorizeFlag = static_cast<uint32_t>((perfMode_ == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                                                    perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                                                    perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) &&
                                                   tilingInfo->quantMode == QUANT_CONTIGUOUS_MODE &&
                                                   tilingInfo->kvLayout == MQSMLALayout::PA_BBND &&
                                                   blockSizeSupported && vectorizeUbSize <= UB_SIZE);

    size_t workspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    bool isSplitG = tilingInfo->gSize > 64; // gSize超过64时采用Split-G
    workspaceSize += static_cast<size_t>(S2_BASE_SIZE) * D_SIZE * VEC_RES_ELEM_SIZE * TRIPLE_BUFFER_NUM *
                     (isSplitG ? (aicNum >> 1) : aicNum);
    if (vectorizeFlag != 0) {
        uint64_t totalBS1 = (tilingInfo->qLayout == MQSMLALayout::TND) ?
                                tilingInfo->s1Size :
                                static_cast<uint64_t>(tilingInfo->bSize) * tilingInfo->s1Size;
        if (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
            perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            workspaceSize += totalBS1 * alignedOriSparseBlockCount * sizeof(int64_t);
        }
        if (perfMode_ == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
            perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            workspaceSize += totalBS1 * alignedCmpSparseBlockCount * sizeof(int64_t);
        }
    }
    uint32_t fdStagingMSize = tilingInfo->gSize;
    uint32_t fdStagingSlotNum = isSplitG ? (aicNum >> 1) : aicNum;
    if (tilingInfo->batchConsistency) {
        size_t combineElemSize = static_cast<size_t>(fdStagingMSize) * D_SIZE +
                                 static_cast<size_t>(FD_MAX_SUM_REGION_NUM) * fdStagingMSize * FD_BLOCK_ELEM;
        workspaceSize += 2ULL * fdStagingSlotNum * combineElemSize * FLOAT_ELEM_SIZE;
        workspaceSize +=
            static_cast<size_t>(aicNum) * BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * combineElemSize * FLOAT_ELEM_SIZE;
    } else {
        // 末尾的2对应每个split分别暂存max和sum。
        size_t s2SplitStagingPerSlot =
            static_cast<size_t>(fdStagingMSize) * D_SIZE * FLOAT_ELEM_SIZE * MAX_S2_SPLIT_NUM +
            static_cast<size_t>(fdStagingMSize) * FD_BLOCK_ELEM * FLOAT_ELEM_SIZE * MAX_S2_SPLIT_NUM *
                FD_MAX_SUM_REGION_NUM;
        workspaceSize += s2SplitStagingPerSlot * fdStagingSlotNum;
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.baseParams.set_batchSize(tilingInfo->bSize);
    tilingData_.baseParams.set_kvSeqSize(tilingInfo->s2Size);
    tilingData_.baseParams.set_cmpKvSeqSize(tilingInfo->cmpS2Size);
    tilingData_.baseParams.set_qSeqSize(tilingInfo->s1Size);
    tilingData_.baseParams.set_oriSparseBlockCount(tilingInfo->oriSparseBlockCount);
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
    tilingData_.baseParams.set_returnSoftmaxLse(tilingInfo->returnSoftmaxLse);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t qType = static_cast<uint32_t>(tilingInfo->qType);
    uint32_t oriKvType = static_cast<uint32_t>(tilingInfo->oriKvType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t qLayout = static_cast<uint32_t>(tilingInfo->qLayout);
    uint32_t inputKvLayout = static_cast<uint32_t>(tilingInfo->kvLayout);
    // maskmode为4+3，无topk len输入且不输出lse时, 走HIGH_PERF高性能模板
    bool highPerf = (tilingInfo->oriMaskMode == 4 && tilingInfo->cmpMaskMode == 3) &&
                    tilingInfo->opParamInfo.oriTopkLength.tensor == nullptr &&
                    tilingInfo->opParamInfo.cmpTopkLength.tensor == nullptr && !tilingInfo->returnSoftmaxLse;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        0U, qLayout, inputKvLayout, static_cast<uint32_t>(perfMode_), static_cast<uint32_t>(isSplitG),
        static_cast<uint32_t>(tilingInfo->quantMode),
        ((oriKvType == ge::DT_FLOAT8_E4M3FN) ? DTYPE_FP8_E4M3FN : DTYPE_HIF8),
        static_cast<uint32_t>(tilingInfo->batchConsistency), vectorizeFlag, static_cast<uint32_t>(highPerf));
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
