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
 * \file lightning_indexer_grad_tiling.cpp
 * \brief
 */
#include "../op_kernel/lightning_indexer_grad_tiling.h"
#include "lightning_indexer_grad_tiling_data.h"
#include "../op_kernel/lightning_indexer_grad_template_tiling_key.h"

#include <algorithm>
#include <vector>

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {

constexpr size_t GM_ALIGN = 512;
constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
constexpr uint32_t LAYOUT_BSND = 0;
constexpr uint32_t LAYOUT_TND = 1;
constexpr uint64_t MAX_GROUPNUM = 64;
constexpr uint64_t MAX_HEADIM = 128;
constexpr uint64_t MAX_TOPK = 2048;
constexpr uint64_t LIMIT_HEADNUMK = 1;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t S2_LOAD_ALIGN = 512;

uint64_t GetS1Load(uint64_t sparseMode, uint64_t actualSeqQ, uint64_t actualSeqK, uint64_t s1Idx, uint64_t topK)
{
    if (sparseMode == 0) {
        return topK;
    }
    int64_t realS2 =
        static_cast<int64_t>(actualSeqK) - static_cast<int64_t>(actualSeqQ) + static_cast<int64_t>(s1Idx) + 1;
    if (realS2 <= 0) {
        return 1;
    }
    uint64_t clippedS2 = std::min(static_cast<uint64_t>(realS2), topK);
    uint64_t alignedS2 = (clippedS2 + S2_LOAD_ALIGN - 1) / S2_LOAD_ALIGN * S2_LOAD_ALIGN;
    return std::min(alignedS2, topK);
}

void SplitContinuousS1ByLoad(const std::vector<uint64_t> &s1Load, uint32_t coreNum, uint64_t *coreStartIdx)
{
    uint64_t totalS1 = s1Load.size();
    std::fill(coreStartIdx, coreStartIdx + LIG_MAX_CORE_NUM + 1, totalS1);
    coreStartIdx[0] = 0;
    if (totalS1 == 0 || coreNum == 0) {
        return;
    }

    std::vector<uint64_t> prefixLoad(totalS1 + 1, 0);
    for (uint64_t i = 0; i < totalS1; i++) {
        prefixLoad[i + 1] = prefixLoad[i] + s1Load[i];
    }

    uint64_t totalLoad = prefixLoad[totalS1];
    for (uint32_t coreIdx = 1; coreIdx < coreNum; coreIdx++) {
        uint64_t minEnd = coreStartIdx[coreIdx - 1] + 1;
        uint64_t maxEnd = totalS1 - (coreNum - coreIdx);
        uint64_t targetLoad = (totalLoad * coreIdx + coreNum / 2) / coreNum;
        auto beginIt = prefixLoad.begin() + minEnd;
        auto endIt = prefixLoad.begin() + maxEnd + 1;
        auto upperIt = std::lower_bound(beginIt, endIt, targetLoad);
        uint64_t upper = std::min(static_cast<uint64_t>(upperIt - prefixLoad.begin()), maxEnd);
        uint64_t lower = upper > minEnd ? upper - 1 : upper;
        uint64_t lowerDiff =
            prefixLoad[lower] > targetLoad ? prefixLoad[lower] - targetLoad : targetLoad - prefixLoad[lower];
        uint64_t upperDiff =
            prefixLoad[upper] > targetLoad ? prefixLoad[upper] - targetLoad : targetLoad - prefixLoad[upper];
        coreStartIdx[coreIdx] = lowerDiff <= upperDiff ? lower : upper;
    }
    coreStartIdx[coreNum] = totalS1;
}

ge::graphStatus LightningIndexerGradTiling::DoTiling()
{
    opName_ = context_->GetNodeName();
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_CHECK_IF(blockDim <= 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "blockDim must be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockDim > LIG_MAX_CORE_NUM,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "blockDim must not exceed %u, but got %u.",
                                            LIG_MAX_CORE_NUM, blockDim),
                return ge::GRAPH_FAILED);

    uint64_t workspaceOffset = ascendcPlatform.GetLibApiWorkSpaceSize();
    ;
    size_t *workSpaces = context_->GetWorkspaceSizes(1);

    // input tensor
    opParamInfo.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo.query.shape = context_->GetInputShape(QUERY_INDEX);
    queryDataType = opParamInfo.query.desc->GetDataType();

    opParamInfo.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo.key.shape = context_->GetInputShape(KEY_INDEX);

    opParamInfo.dy.desc = context_->GetInputDesc(DY_INDEX);
    opParamInfo.dy.shape = context_->GetInputShape(DY_INDEX);

    opParamInfo.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INDEX);
    opParamInfo.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INDEX);

    opParamInfo.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);

    // input attr
    auto attrs = context_->GetAttrs();
    opParamInfo.headNum = *attrs->GetInt(ATTR_HEADNUM_INDEX);
    opParamInfo.layout = attrs->GetStr(ATTR_LAYOUT_INDEX);
    opParamInfo.sparseMode = *attrs->GetInt(ATTR_SPARSEMODE_INDEX);
    opParamInfo.preTokens = *attrs->GetInt(ATTR_PRETOKENS_INDEX);
    opParamInfo.nextTokens = *attrs->GetInt(ATTR_NEXTTOKENS_INDEX);
    bool deterministic = *attrs->GetBool(ATTR_DETERMINSTIC_INDEX) || (context_->GetDeterministic() == 1);
    opParamInfo.deterministic = deterministic;

    uint32_t dyShapeDim = opParamInfo.dy.shape->GetStorageShape().GetDimNum();
    uint32_t dataType = static_cast<uint32_t>(queryDataType);
    uint32_t inputLayout = -1;

    if (std::string(opParamInfo.layout) == "BSND") {
        batch = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
        seqlenQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
        headNumQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        headDim = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_FOUR));
        seqlenK = static_cast<uint32_t>(opParamInfo.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
        headNumK = static_cast<uint32_t>(opParamInfo.key.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        topK = static_cast<uint32_t>(opParamInfo.dy.shape->GetStorageShape().GetDim(dyShapeDim - 1));
        OP_CHECK_IF((batch <= 0) || (seqlenQ <= 0) || (seqlenK <= 0) || (headNumQ <= 0) || (headNumK <= 0) ||
                        (topK <= 0) || (topK > MAX_TOPK),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        context_->GetNodeName(),
                        "invalid shape: batch, seqlenQ, seqlenK, headNumQ, headNumK and topK must be > 0, "
                        "and topK must be <= %lu, but got "
                        "batch=%u, seqlenQ=%u, seqlenK=%u, headNumQ=%u, headNumK=%u, topK=%u",
                        MAX_TOPK, batch, seqlenQ, seqlenK, headNumQ, headNumK, topK),
                    return ge::GRAPH_FAILED);
        groupNum = headNumQ / headNumK;
        dkSize = batch * seqlenK * headNumK * headDim;
        inputLayout = LAYOUT_BSND;
    } else if (std::string(opParamInfo.layout) == "TND") {
        opParamInfo.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
        opParamInfo.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
        opParamInfo.actualSeqLengthsK.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
        opParamInfo.actualSeqLengthsK.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
        OP_CHECK_IF(opParamInfo.actualSeqLengthsQ.tensor == nullptr || opParamInfo.actualSeqLengthsK.tensor == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        context_->GetNodeName(),
                        "TND layout requires actual_seq_lengths_query and actual_seq_lengths_key to be provided, "
                        "but got null."),
                    return ge::GRAPH_FAILED);

        batch = static_cast<uint32_t>(opParamInfo.actualSeqLengthsQ.tensor->GetShapeSize());
        seqlenQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
        headNumQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
        headDim = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        seqlenK = static_cast<uint32_t>(opParamInfo.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
        headNumK = static_cast<uint32_t>(opParamInfo.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
        topK = static_cast<uint32_t>(opParamInfo.dy.shape->GetStorageShape().GetDim(dyShapeDim - 1));
        OP_CHECK_IF((batch <= 0) || (seqlenQ <= 0) || (seqlenK <= 0) || (headNumQ <= 0) || (headNumK <= 0) ||
                        (topK <= 0) || (topK > MAX_TOPK),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        context_->GetNodeName(),
                        "invalid shape: batch, seqlenQ, seqlenK, headNumQ, headNumK and topK must be > 0, "
                        "and topK must be <= %lu, but got "
                        "batch=%u, seqlenQ=%u, seqlenK=%u, headNumQ=%u, headNumK=%u, topK=%u",
                        MAX_TOPK, batch, seqlenQ, seqlenK, headNumQ, headNumK, topK),
                    return ge::GRAPH_FAILED);
        groupNum = headNumQ / headNumK;
        dkSize = seqlenK * headNumK * headDim;
        inputLayout = LAYOUT_TND;
    } else {
        OP_LOGE(context_, "only support layout is BSND and TND, but got %s.\n", opParamInfo.layout);
        return ge::GRAPH_FAILED;
    }

    uint32_t dkCoreSize = seqlenK * headDim;
    // check headDim, groupNum, headNumK
    OP_CHECK_IF(
        (headDim != MAX_HEADIM) || (groupNum <= 0) || (groupNum > MAX_GROUPNUM) || (headNumK != LIMIT_HEADNUMK),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                    "only support headDim is %lu, groupNum is in range [1, %lu], headNumK is %lu, "
                                    "but current headDim is %lu, groupNum is %lu, headNumK is %lu",
                                    MAX_HEADIM, MAX_GROUPNUM, LIMIT_HEADNUMK, static_cast<uint64_t>(headDim),
                                    static_cast<uint64_t>(groupNum), static_cast<uint64_t>(headNumK)),
        return ge::GRAPH_FAILED);

    // check sparseMode
    OP_CHECK_IF((opParamInfo.sparseMode != 0) && (opParamInfo.sparseMode != 3),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "sparseMode only support value is 0 or 3"),
                return ge::GRAPH_FAILED);

    uint32_t usedAicNum = blockDim;
    std::vector<uint64_t> s1Load;
    if (!opParamInfo.deterministic) {
        // The non-deterministic kernel ends with SyncAll, so all used cores
        // must be launched together in batch schedule mode.
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "failed to set batch schedule mode."),
                    return ge::GRAPH_FAILED);

        // Build one global B*S1/T1 work domain. Each S1 is weighted by the real
        // amount of S2 work and every core receives one contiguous S1 interval.
        if (inputLayout == LAYOUT_BSND) {
            s1Load.reserve(static_cast<uint64_t>(batch) * seqlenQ);
            for (uint32_t bIdx = 0; bIdx < batch; bIdx++) {
                for (uint32_t s1Idx = 0; s1Idx < seqlenQ; s1Idx++) {
                    s1Load.push_back(GetS1Load(opParamInfo.sparseMode, seqlenQ, seqlenK, s1Idx, topK));
                }
            }
        } else {
            // The aclnn interface supplies actual sequence lengths as device tensors.
            // Tiling runs on the host, so GetData() cannot be dereferenced here. Build
            // a data-independent proxy from T1/T2 while keeping the split domain equal
            // to the real global T1. The kernel still reads the cumulative lengths from
            // GM to map every global T1 position to its exact batch and local S1 index.
            s1Load.reserve(seqlenQ);
            for (uint32_t bIdx = 0; bIdx < batch; bIdx++) {
                uint64_t actualQ = seqlenQ / batch + (bIdx < seqlenQ % batch ? 1 : 0);
                uint64_t actualK = seqlenK / batch + (bIdx < seqlenK % batch ? 1 : 0);
                for (uint64_t s1Idx = 0; s1Idx < actualQ; s1Idx++) {
                    s1Load.push_back(GetS1Load(opParamInfo.sparseMode, actualQ, actualK, s1Idx, topK));
                }
            }
        }
        usedAicNum = std::max(1U, std::min(blockDim, static_cast<uint32_t>(s1Load.size())));
    }
    context_->SetBlockDim(usedAicNum);
    tilingData_->set_totalS1(s1Load.size());
    SplitContinuousS1ByLoad(s1Load, usedAicNum, tilingData_->get_coreS1StartIdxPtr());

    // set tilingData
    tilingData_->set_batch(batch);
    tilingData_->set_seqlenQ(seqlenQ);
    tilingData_->set_headNumQ(headNumQ);
    tilingData_->set_headDim(headDim);
    tilingData_->set_seqlenK(seqlenK);
    tilingData_->set_headNumK(headNumK);
    tilingData_->set_groupNum(groupNum);
    tilingData_->set_topK(topK);
    tilingData_->set_usedCoreNum(usedAicNum * 2);
    tilingData_->set_dkSize(dkSize);
    tilingData_->set_dkCoreSize(dkCoreSize);
    tilingData_->set_sparseMode(static_cast<uint64_t>(opParamInfo.sparseMode));
    tilingData_->set_deterministic(static_cast<uint64_t>(opParamInfo.deterministic));

    // set workspace
    tilingData_->set_dkWorkSpaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + dkSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingData_->set_dkCoreWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * dkCoreSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    uint64_t keyGatherWorkspaceSize = MAX_HEADIM * MAX_TOPK * sizeof(uint16_t) * DOUBLE_BUFFER;
    tilingData_->set_keyGatherWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * keyGatherWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    uint64_t reluInWorkspaceSize = MAX_GROUPNUM * MAX_TOPK * sizeof(float) * DOUBLE_BUFFER;
    tilingData_->set_reluInWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * reluInWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    uint64_t reluGradWorkspaceSize = MAX_GROUPNUM * MAX_TOPK * sizeof(uint16_t) * DOUBLE_BUFFER;
    tilingData_->set_reluGradWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * reluGradWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    uint64_t scatterAddWorkspaceSize = MAX_HEADIM * MAX_TOPK * sizeof(float) * DOUBLE_BUFFER;
    tilingData_->set_scatterAddWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * scatterAddWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    workspaceOffset += WORKSPACE_RSV_BYTE;
    workSpaces[0] = (workspaceOffset - 0);

    // -------------set tilingkey-----------------
    uint32_t tilingKey = GET_TPL_TILING_KEY(dataType, inputLayout);
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForLightningIndexerGrad(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus TilingForLightningIndexerGrad(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("LightningIndexerGrad", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    LightningIndexerGradTiling LIGTiling(context);
    return LIGTiling.DoTiling();
}

IMPL_OP_OPTILING(LightningIndexerGrad)
    .Tiling(TilingForLightningIndexerGrad)
    .TilingParse<LIGCompileInfo>(TilingPrepareForLightningIndexerGrad);

} // namespace optiling
