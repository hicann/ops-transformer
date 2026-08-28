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
 * \file bsa_select_block_mask_tiling_base.cpp
 * \brief
 */
#include "bsa_select_block_mask_tiling_base.h"
#include <tiling/tiling_api.h>
using namespace ge;
using namespace AscendC;
namespace optiling {

bool BSASelectBlockMaskTilingBase::IsA5Platform() const
{
    return socVersion == platform_ascendc::SocVersion::ASCEND950;
}

uint32_t BSASelectBlockMaskTilingBase::GetAivRatio() const
{
    if (aicNum == 0 || aivNum == 0 || aicNum > aivNum) {
        return AIC_AIV_RATIO;
    }
    return std::max(1U, aivNum / aicNum);
}

uint32_t BSASelectBlockMaskTilingBase::CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum,
                                                        uint32_t aivCoreNum) const
{
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    uint32_t ratio = aivCoreNum / aicCoreNum;
    return (sliceNum + ratio - 1) / ratio;
}

uint32_t BSASelectBlockMaskTilingBase::CalcLaunchAivNum(uint32_t activeCoreNum) const
{
    uint64_t launchAivNum = static_cast<uint64_t>(activeCoreNum) * GetAivRatio();
    return static_cast<uint32_t>(std::min<uint64_t>(launchAivNum, aivNum));
}

ge::graphStatus BSASelectBlockMaskTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    size_t idx = 0;
    auto qLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto kvLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto sparsityPtr = attrs->GetAttrPointer<float>(idx++);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparsityPtr);

    auto queryShape = context_->GetInputShape(QUERY_INPUT_INDEX);
    auto keyShape = context_->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);

    auto maskShape = context_->GetOutputShape(MASK_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskShape);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    return ge::GRAPH_SUCCESS;
}

bool BSASelectBlockMaskTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto qLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto kvLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto sparsityPtr = attrs->GetAttrPointer<float>(idx++);

    queryLayout = qLayoutPtr;
    kvLayout = kvLayoutPtr;
    if (scaleValuePtr != nullptr && *scaleValuePtr != 0.0f) {
        scaleValue = *scaleValuePtr;
    }
    sparsity = *sparsityPtr;

    OP_CHECK_IF(sparsity <= SPARSITY_MIN || sparsity >= SPARSITY_MAX,
                OP_LOGE(opName, "sparsity[%f] out of range (0, 1).", sparsity), return false);

    sparsityMode = static_cast<uint8_t>(sparsity > 0.5f ? SparsityMode::BOTTOM_K : SparsityMode::TOP_K);

    auto blockShapeTensor = context_->GetOptionalInputTensor(BLOCK_SHAPE_INPUT_INDEX);
    if (blockShapeTensor != nullptr) {
        auto &blockShapeShape = blockShapeTensor->GetShape().GetStorageShape();
        if (blockShapeShape.GetDimNum() != 0) {
            const int64_t *blockShapeData = blockShapeTensor->GetData<int64_t>();
            if (blockShapeData != nullptr) {
                blockShapeX = static_cast<uint64_t>(blockShapeData[0]);
                blockShapeY = static_cast<uint64_t>(blockShapeData[1]);
            }
        }
    }

    OP_CHECK_IF(blockShapeX % BLOCK_SHAPE_ALIGN != 0 || blockShapeY % BLOCK_SHAPE_ALIGN != 0,
                OP_LOGE(opName, "blockShapeX[%lu] and blockShapeY[%lu] must be multiples of %u.", blockShapeX,
                        blockShapeY, BLOCK_SHAPE_ALIGN),
                return false);

    auto actualBlockLenQTensor = context_->GetOptionalInputTensor(ACTUAL_BLOCK_LEN_Q_INPUT_INDEX);
    useActualBlockLenQ =
        (actualBlockLenQTensor != nullptr && actualBlockLenQTensor->GetShape().GetStorageShape().GetDimNum() != 0) ? 1 :
                                                                                                                     0;

    auto actualBlockLenKVTensor = context_->GetOptionalInputTensor(ACTUAL_BLOCK_LEN_KV_INPUT_INDEX);
    useActualBlockLenK =
        (actualBlockLenKVTensor != nullptr && actualBlockLenKVTensor->GetShape().GetStorageShape().GetDimNum() != 0) ?
            1 :
            0;

    auto postBlockShapeTensor = context_->GetOptionalInputTensor(POST_BLOCK_SHAPE_INPUT_INDEX);
    OP_LOGD(opName, "postBlockShapeTensor ptr=%p", postBlockShapeTensor);
    if (postBlockShapeTensor != nullptr && postBlockShapeTensor->GetShape().GetStorageShape().GetDimNum() != 0) {
        const int64_t *postBlockShapeData = postBlockShapeTensor->GetData<int64_t>();
        OP_LOGD(opName, "postBlockShapeData ptr=%p", postBlockShapeData);
        if (postBlockShapeData != nullptr) {
            postBlockShapeX = static_cast<uint16_t>(postBlockShapeData[0]);
            postBlockShapeY = static_cast<uint16_t>(postBlockShapeData[1]);
            usePostBlockShape = 1;
            OP_LOGI(opName, "post_block_shape=[%u, %u] usePostBlockShape=%u", postBlockShapeX, postBlockShapeY,
                    usePostBlockShape);
            OP_CHECK_IF(postBlockShapeX == 0 || postBlockShapeY == 0,
                        OP_LOGE(opName, "postBlockShapeX[%u] and postBlockShapeY[%u] must be positive.",
                                postBlockShapeX, postBlockShapeY),
                        return false);
            OP_CHECK_IF(postBlockShapeX % BLOCK_SHAPE_ALIGN != 0 || postBlockShapeY % BLOCK_SHAPE_ALIGN != 0,
                        OP_LOGE(opName, "postBlockShapeX[%u] and postBlockShapeY[%u] must be multiples of %u.",
                                postBlockShapeX, postBlockShapeY, BLOCK_SHAPE_ALIGN),
                        return false);
        }
    }

    auto actualSeqLenQTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_LENS_Q_INPUT_INDEX);
    useActualSeqLenQ =
        (actualSeqLenQTensor != nullptr && actualSeqLenQTensor->GetShape().GetStorageShape().GetDimNum() != 0) ? 1 : 0;
    auto actualSeqLenKTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_LENS_KV_INPUT_INDEX);
    useActualSeqLenK =
        (actualSeqLenKTensor != nullptr && actualSeqLenKTensor->GetShape().GetStorageShape().GetDimNum() != 0) ? 1 : 0;

    OP_LOGD(opName,
            "attrs: queryLayout[%s] kvLayout[%s] blockShapeX[%lu] blockShapeY[%lu] "
            "sparsity[%f] sparsityMode[%d] scaleValue[%f].",
            queryLayout, kvLayout, blockShapeX, blockShapeY, sparsity, sparsityMode, scaleValue);
    return true;
}

bool BSASelectBlockMaskTilingBase::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INPUT_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();

    OP_CHECK_IF(queryDtype != keyDtype, OP_LOGE(opName, "query and key dtype must be same."), return false);
    OP_CHECK_IF(queryDtype != ge::DT_FLOAT16 && queryDtype != ge::DT_BF16,
                OP_LOGE(opName, "query dtype must be FP16 or BF16."), return false);

    return true;
}

bool BSASelectBlockMaskTilingBase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();

    size_t qDimNum = queryShape.GetDimNum();
    size_t kvDimNum = keyShape.GetDimNum();
    size_t qLayoutLen = strlen(queryLayout);
    size_t kvLayoutLen = strlen(kvLayout);

    OP_CHECK_IF(qDimNum != qLayoutLen || kvDimNum != kvLayoutLen,
                OP_LOGE(opName, "Invalid layout: query dim[%zu] vs layout len[%zu], key dim[%zu] vs layout len[%zu].",
                        qDimNum, qLayoutLen, kvDimNum, kvLayoutLen),
                return false);

    if (qLayoutLen == 4UL && queryLayout[0] == 'B' && queryLayout[1] == 'N') {
        batchSize = queryShape.GetDim(0);
        numHeads = queryShape.GetDim(1);
        maxQSeqlen = queryShape.GetDim(2);
        dSize = queryShape.GetDim(3);
        tilingKeyLayoutQ = LayoutType::LAYOUT_BNSD;
    } else if (qLayoutLen == 3UL && queryLayout[0] == 'T' && queryLayout[1] == 'N') {
        // TND 排布：[TotalSeq, Heads, Dim]；Batch 从 actualSeqLens 推导
        numHeads = queryShape.GetDim(1);
        dSize = queryShape.GetDim(2);
        batchSize = 1;
        maxQSeqlen = 0;
        tilingKeyLayoutQ = LayoutType::LAYOUT_TND;

        auto actualSeqLensQTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_LENS_Q_INPUT_INDEX);
        if (actualSeqLensQTensor != nullptr && actualSeqLensQTensor->GetShape().GetStorageShape().GetDimNum() != 0) {
            batchSize = actualSeqLensQTensor->GetShape().GetStorageShape().GetDim(0);
            auto seqLensQ = actualSeqLensQTensor->GetData<int64_t>();
            if (batchSize > 0) {
                maxQSeqlen = static_cast<uint32_t>(seqLensQ[0]);
                for (uint32_t i = 1; i < batchSize; i++) {
                    maxQSeqlen = std::max(maxQSeqlen, static_cast<uint32_t>(seqLensQ[i]));
                }
            }
        }
    } else {
        OP_LOGE(opName, "Unsupported query layout[%s].", queryLayout);
        return false;
    }

    if (kvLayoutLen == 4UL && kvLayout[0] == 'B' && kvLayout[1] == 'N') {
        maxKvSeqlen = keyShape.GetDim(2);
        tilingKeyLayoutKV = LayoutType::LAYOUT_BNSD;
    } else if (kvLayoutLen == 3UL && kvLayout[0] == 'T' && kvLayout[1] == 'N') {
        tilingKeyLayoutKV = LayoutType::LAYOUT_TND;
        maxKvSeqlen = 0;
        auto actualSeqLensKVTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_LENS_KV_INPUT_INDEX);
        if (actualSeqLensKVTensor != nullptr && actualSeqLensKVTensor->GetShape().GetStorageShape().GetDimNum() != 0) {
            uint32_t kvBatch = actualSeqLensKVTensor->GetShape().GetStorageShape().GetDim(0);
            auto seqLensKV = actualSeqLensKVTensor->GetData<int64_t>();
            if (kvBatch > 0) {
                maxKvSeqlen = static_cast<uint32_t>(seqLensKV[0]);
                for (uint32_t i = 1; i < kvBatch; i++) {
                    maxKvSeqlen = std::max(maxKvSeqlen, static_cast<uint32_t>(seqLensKV[i]));
                }
            }
        } else {
            maxKvSeqlen = keyShape.GetDim(0);
        }
    } else {
        OP_LOGE(opName, "Unsupported kv layout[%s].", kvLayout);
        return false;
    }

    OP_CHECK_IF(dSize != D_SIZE_128, OP_LOGE(opName, "headDim must be 128, got[%u].", dSize), return false);
    OP_CHECK_IF(numHeads < HEAD_NUM_MIN, OP_LOGE(opName, "numHeads[%u] must be greater than or equal to 1.", numHeads),
                return false);

    xBlocks = BSACeilDivision(maxQSeqlen, static_cast<uint32_t>(blockShapeX));
    yBlocks = BSACeilDivision(maxKvSeqlen, static_cast<uint32_t>(blockShapeY));

    if (usePostBlockShape) {
        postXBlocks = BSACeilDivision(xBlocks, static_cast<uint32_t>(postBlockShapeX));
        postYBlocks = BSACeilDivision(yBlocks, static_cast<uint32_t>(postBlockShapeY));
    } else {
        postXBlocks = xBlocks;
        postYBlocks = yBlocks;
    }

    uint64_t totalXY = static_cast<uint64_t>(postXBlocks) * postYBlocks;
    bool isTnd = tilingKeyLayoutQ == LayoutType::LAYOUT_TND || tilingKeyLayoutKV == LayoutType::LAYOUT_TND;
    OP_CHECK_IF(
        totalXY == 0 || (totalXY == 1 && !isTnd),
        OP_LOGE(opName,
                "postXBlocks[%u] * postYBlocks[%u] must be positive, and greater than 1 for BNSD TopK selection.",
                postXBlocks, postYBlocks),
        return false);

    if (sparsityMode == static_cast<uint8_t>(SparsityMode::TOP_K)) {
        topKValue = static_cast<uint64_t>(std::round(sparsity * totalXY));
    } else {
        topKValue = static_cast<uint64_t>(std::round((1.0f - sparsity) * totalXY));
    }
    topKValue = std::max(topKValue, static_cast<uint64_t>(1));

    OP_LOGD(opName,
            "layout: batchSize[%u] numHeads[%u] maxQSeqlen[%u] maxKvSeqlen[%u] "
            "xBlocks[%u] yBlocks[%u] topKValue[%lu].",
            batchSize, numHeads, maxQSeqlen, maxKvSeqlen, xBlocks, yBlocks, topKValue);
    return true;
}

bool BSASelectBlockMaskTilingBase::CrossShapeVerify()
{
    auto &queryShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();

    if (tilingKeyLayoutQ == LayoutType::LAYOUT_BNSD) {
        OP_CHECK_IF(keyShape.GetDim(0) != static_cast<int64_t>(batchSize),
                    OP_LOGE(opName, "B mismatch between query[%u] and key[%ld].", batchSize, keyShape.GetDim(0)),
                    return false);
        OP_CHECK_IF(
            keyShape.GetDim(1) != static_cast<int64_t>(numHeads),
            OP_LOGE(opName, "N mismatch (GQA not supported): query N[%u] vs key N[%ld].", numHeads, keyShape.GetDim(1)),
            return false);
    }

    return true;
}

void BSASelectBlockMaskTilingBase::CalcMultiCoreParams()
{
    uint32_t coreNum = aicNum;

    // xBlocks 分核（用于 Matmul/Softmax 阶段）
    uint32_t activeCoreNum = std::min(xBlocks, coreNum);
    uint32_t baseRows = xBlocks / activeCoreNum;
    uint32_t extraCores = xBlocks % activeCoreNum;
    uint32_t rowsPerCore = baseRows + (extraCores > 0 ? 1 : 0);

    // yBlocks 分核（用于 Key 池化阶段）
    // uint32_t activeYVecCoreNum = std::min(yBlocks, coreNum);

    uint32_t activeYVecCoreNum = std::min(yBlocks, CalcLaunchAivNum(activeCoreNum));
    uint32_t baseYRows = yBlocks / activeYVecCoreNum;
    uint32_t extraYCores = yBlocks % activeYVecCoreNum;
    uint32_t yBlocksPerCore = baseYRows + (extraYCores > 0 ? 1 : 0);

    multiCoreParams_->set_coreNum(coreNum);
    multiCoreParams_->set_activeCoreNum(activeCoreNum);
    multiCoreParams_->set_activeYCoreNum(activeYVecCoreNum);
    multiCoreParams_->set_rowsPerCore(rowsPerCore);
    multiCoreParams_->set_extraCores(extraCores);
    multiCoreParams_->set_totalRows(xBlocks);
    multiCoreParams_->set_yBlocksPerCore(yBlocksPerCore);
    multiCoreParams_->set_extraYCores(extraYCores);
}

void BSASelectBlockMaskTilingBase::CalcOutputParams()
{
    uint64_t qCmpSize = static_cast<uint64_t>(xBlocks) * dSize * (sizeof(float) / 2);
    uint64_t kCmpSize = static_cast<uint64_t>(yBlocks) * dSize * (sizeof(float) / 2);
    uint64_t attnScoreFp16Size = static_cast<uint64_t>(xBlocks) * yBlocks * (sizeof(float) / 2);

    uint64_t pooledScoreSize = usePostBlockShape ? static_cast<uint64_t>(batchSize) * numHeads * postXBlocks *
                                                       postYBlocks * (sizeof(float) / 2) :
                                                   0;

    uint64_t scoreFp32Size = Q_CHUNK_SIZE * aicNum * yBlocks * sizeof(float);

    uint64_t sortLen =
        usePostBlockShape ? static_cast<uint64_t>(postXBlocks) * postYBlocks : static_cast<uint64_t>(xBlocks) * yBlocks;
    uint64_t tileLen = std::min(static_cast<uint64_t>(7168), std::max(static_cast<uint64_t>(128), sortLen));
    uint64_t totalTileNum = (sortLen + tileLen - 1) / tileLen;
    uint32_t launchAivNum = IsA5Platform() ? CalcLaunchAivNum(multiCoreParams_->get_activeCoreNum()) : aivNum;
    uint64_t topkHeaderSize =
        static_cast<uint64_t>(RADIX_NUM_BINS * launchAivNum + launchAivNum + launchAivNum) * sizeof(int32_t);
    uint64_t topkTileDataSize = totalTileNum * (1 + RADIX_NUM_BINS) * sizeof(int32_t);
    uint64_t topkWorkspaceSize = topkHeaderSize + topkTileDataSize;

    uint64_t totalWorkspaceSize = qCmpSize + kCmpSize + attnScoreFp16Size + scoreFp32Size + topkWorkspaceSize;
    uint64_t pooledScoreOffset = 0;
    if (usePostBlockShape) {
        pooledScoreOffset = attnScoreFp16Size;
        totalWorkspaceSize += pooledScoreSize;
    }

    outputParams_->set_qCmpSize(qCmpSize);
    outputParams_->set_kCmpSize(kCmpSize);
    outputParams_->set_attnScoreSize(attnScoreFp16Size);
    outputParams_->set_softmaxTmpSize(scoreFp32Size);
    outputParams_->set_topkWorkspaceSize(topkWorkspaceSize);
    outputParams_->set_pooledScoreSize(pooledScoreSize);
    outputParams_->set_pooledScoreOffset(pooledScoreOffset);
    outputParams_->set_totalWorkspaceSize(totalWorkspaceSize);
}

ge::graphStatus BSASelectBlockMaskTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();

    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "invalid context."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
                OP_LOGE(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CrossShapeVerify(), OP_LOGE(opName, "cross shape verify failed."), return ge::GRAPH_FAILED);

    tilingData = context_->GetTilingData<BSASelectBlockMaskTilingData>();
    baseParams_ = &tilingData->baseParams;
    multiCoreParams_ = &tilingData->multiCoreParams;
    outputParams_ = &tilingData->outputParams;

    baseParams_->set_batchSize(batchSize);
    baseParams_->set_numHeads(numHeads);
    baseParams_->set_maxQSeqlen(maxQSeqlen);
    baseParams_->set_maxKvSeqlen(maxKvSeqlen);
    baseParams_->set_dSize(dSize);
    baseParams_->set_blockShapeX(blockShapeX);
    baseParams_->set_blockShapeY(blockShapeY);
    baseParams_->set_xBlocks(xBlocks);
    baseParams_->set_yBlocks(yBlocks);
    baseParams_->set_scaleValue(scaleValue);
    baseParams_->set_sparsity(sparsity);
    baseParams_->set_topKValue(topKValue);
    baseParams_->set_sparsityMode(sparsityMode);
    baseParams_->set_queryLayout(static_cast<uint8_t>(tilingKeyLayoutQ));
    baseParams_->set_kvLayout(static_cast<uint8_t>(tilingKeyLayoutKV));
    baseParams_->set_useActualBlockLenQ(useActualBlockLenQ);
    baseParams_->set_useActualBlockLenK(useActualBlockLenK);
    baseParams_->set_useActualSeqLenQ(useActualSeqLenQ);
    baseParams_->set_useActualSeqLenK(useActualSeqLenK);
    baseParams_->set_qChunkSize(Q_CHUNK_SIZE);
    baseParams_->set_kChunkSize(K_CHUNK_SIZE);
    baseParams_->set_usePostBlockShape(usePostBlockShape);
    baseParams_->set_postBlockShapeX(postBlockShapeX);
    baseParams_->set_postBlockShapeY(postBlockShapeY);
    baseParams_->set_postXBlocks(postXBlocks);
    baseParams_->set_postYBlocks(postYBlocks);

    OP_LOGD(opName,
            "INPUTPARAM batchSize:[%u], numHeads:[%u], maxQSeqlen:[%u], maxKvSeqlen:[%u], "
            "xBlocks:[%u], yBlocks:[%u], topKValue:[%lu], sparsity:[%f].",
            batchSize, numHeads, maxQSeqlen, maxKvSeqlen, xBlocks, yBlocks, topKValue, sparsity);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSASelectBlockMaskTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const BSASelectBlockMaskCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(opName, "compileInfoPtr is null."), return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        ubSize = compileInfoPtr->ubSize;
        l1Size = compileInfoPtr->l1Size;
        l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
        socVersion = compileInfoPtr->socVersion;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        socVersion = ascendcPlatform.GetSocVersion();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    }
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OP_LOGE(opName, "invalid platform core num, aivNum[%u] aicNum[%u].", aivNum, aicNum),
                return ge::GRAPH_FAILED);
    OP_LOGD(opName,
            "platform info: socVersion(%d) aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) "
            "l0cSize(%lu) l2CacheSize(%lu).",
            static_cast<int32_t>(socVersion), aivNum, aicNum, ubSize, l1Size, l0cSize, l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSASelectBlockMaskTilingBase::DoOpTiling()
{
    CalcMultiCoreParams();
    uint32_t launchAivNum = CalcLaunchAivNum(multiCoreParams_->get_activeCoreNum());
    uint32_t blockDim =
        IsA5Platform() ? CalcTschBlockDim(launchAivNum, aicNum, aivNum) : multiCoreParams_->get_activeCoreNum();
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(SCHEDULE_MODE_ALL_CORE);

    CalcOutputParams();

    OP_LOGD(opName,
            "DoOpTiling: socVersion[%d] blockDim[%u] activeCoreNum[%u] activeYVecCoreNum[%u] "
            "launchAivNum[%u] rowsPerCore[%u] yBlocksPerCore[%u] xBlocks[%u] yBlocks[%u].",
            static_cast<int32_t>(socVersion), blockDim, multiCoreParams_->get_activeCoreNum(),
            multiCoreParams_->get_activeYCoreNum(), launchAivNum, multiCoreParams_->get_rowsPerCore(),
            multiCoreParams_->get_yBlocksPerCore(), multiCoreParams_->get_totalRows(), yBlocks);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSASelectBlockMaskTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    workspaces[0] = static_cast<size_t>(outputParams_->get_totalWorkspaceSize()) + WORK_SPACE_RESERVE_SIZE;
    OP_LOGD(opName, "workspace size:[%ld], totalWorkspaceSize:[%lu].", workspaces[0],
            outputParams_->get_totalWorkspaceSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t BSASelectBlockMaskTilingBase::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint8_t>(tilingKeyLayoutQ), static_cast<uint8_t>(tilingKeyLayoutKV));
}

} // namespace optiling
