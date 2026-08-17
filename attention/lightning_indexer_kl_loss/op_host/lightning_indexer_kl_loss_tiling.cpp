/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/lightning_indexer_kl_loss_tiling_data.h"
#include "../op_kernel/lightning_indexer_kl_loss_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

struct LightningIndexerKLLossCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, int64_t &coreNum)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t ComputeTileLength(int64_t K, int64_t ubSize)
{
    // UB 划分为 7 块连续 buffer:
    // ub_target_score_in: tileLen * K * dtypeSize
    // ub_index_probs_in:  tileLen * K * dtypeSize
    // ub_reduce_sum:      tileLen * 8 * dtypeSize  (8 元素对齐)
    // ub_log_P:           tileLen * K * dtypeSize
    // ub_log_Y:           tileLen * K * dtypeSize
    // ub_out:             8 * dtypeSize
    // tmp_ub:             tileLen * K * dtypeSize
    // 总元素数 = tileLen * (5K + 8) + 8
    // 总内存占用 <= ubSize
    int64_t totalElems = ubSize / sizeof(float);
    int64_t maxTileLen = (totalElems - 8) / (5 * K + 8);
    return maxTileLen;
}

// 先定义一个向上取整函数
int Ceil(int a, int b) { return (a + b - 1) / b; }

ge::graphStatus GetWorkspaceSize(gert::TilingContext *context, bool deterministic, bool isHalf, int64_t coreNum)
{
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (deterministic || isHalf) {
        currentWorkspace[0] = static_cast<size_t>(coreNum) * sizeof(float) * 8 + sysWorkspaceSize;
    } else {
        currentWorkspace[0] = sysWorkspaceSize;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDim(gert::TilingContext *context)
{
    const auto targetScoreShape = context->GetInputShape(0);
    const auto indexProbsShape = context->GetInputShape(1);
    uint64_t dimNum = targetScoreShape->GetOriginShape().GetDimNum();
    if (dimNum != 2 && dimNum != 3) {
        std::string dimNumStr = std::to_string(dimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "target_score", dimNumStr.c_str(), "2D or 3D");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_NULL_WITH_CONTEXT(context, indexProbsShape);
    auto targetScoreOrigin = targetScoreShape->GetOriginShape();
    auto indexProbsOrigin = indexProbsShape->GetOriginShape();
    std::string targetShapeStr = std::to_string(targetScoreOrigin.GetDim(0));
    std::string indexShapeStr = std::to_string(indexProbsOrigin.GetDim(0));

    for (uint64_t i = 1; i < dimNum; i++) {
        targetShapeStr += ", " + std::to_string(targetScoreOrigin.GetDim(i));
        indexShapeStr += ", " + std::to_string(indexProbsOrigin.GetDim(i));
    }
    if (targetShapeStr != indexShapeStr) {
        std::string reasonMsg =
            "The shape of target_score " + targetShapeStr + " does not match the shape of index_probs " + indexShapeStr;
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "target_score and index_probs",
            ("target_score: " + targetShapeStr + ", index_probs: " + indexShapeStr).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t K = targetScoreOrigin.GetDim(dimNum - 1);
    if (K > 8192) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "target_score", targetShapeStr.c_str(),
                                              "The last dim of input target_score should be less than 8192");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LightningIndexerKLLossTilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    const auto targetScoreShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetScoreShape);
    OP_CHECK_IF(CheckInputDim(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "LightningIndexerKLLoss CheckInputDim FAILED."), return ge::GRAPH_FAILED);

    auto inputShape = targetScoreShape->GetOriginShape();
    int64_t dimNum = static_cast<int64_t>(inputShape.GetDimNum());
    int64_t K = inputShape.GetDim(dimNum - 1);
    int64_t M = 1;
    for (int64_t i = 0; i < dimNum - 1; i++) {
        M *= inputShape.GetDim(i);
    }

    auto attrs = context->GetAttrs();
    float eps = *attrs->GetFloat(0);
    std::string weightTypeStr = std::string(attrs->GetStr(1));
    uint64_t weightType = (weightTypeStr == "probs") ? 1 : 0;
    bool deterministic = context->GetDeterministic() == 1;
    // UB 统一按 fp32 划分
    int64_t KAligned = (K + 7) / 8 * 8;
    int64_t tileLength = ComputeTileLength(KAligned, ubSize);
    tileLength = std::min(tileLength, M);

    // 计算 tile 块总数
    int64_t totalTileNum = (M + tileLength - 1) / tileLength;
    // round-robin 分核
    int64_t formerNum = totalTileNum % coreNum;
    int64_t formerTileNum = (totalTileNum + coreNum - 1) / coreNum;
    int64_t tailTileNum = totalTileNum / coreNum;

    LightningIndexerKLLossTilingData *tiling = context->GetTilingData<LightningIndexerKLLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LightningIndexerKLLossTilingData), 0, sizeof(LightningIndexerKLLossTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalLength = M;
    tiling->K = K;
    tiling->KAligned = KAligned;
    tiling->formerNum = formerNum;
    tiling->formerTileNum = formerTileNum;
    tiling->tailTileNum = tailTileNum;
    tiling->tileLength = tileLength;
    tiling->eps = eps;
    tiling->coreNum = coreNum;
    context->SetBlockDim(coreNum);

    // TilingKey: [bit 0: deterministic] | [bits 1-2: DataType]
    //   DataType 0: FLOAT16, 1: FLOAT32, 2: BFLOAT16, 3: FLOAT16_PRECISION
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint8_t dataTypeVal = (dataType == ge::DT_FLOAT16) ? 0 :
                          (dataType == ge::DT_BF16)    ? 2 :
                          (dataType == ge::DT_FLOAT)   ? 1 :
                                                         0;
    ASCENDC_TPL_SEL_PARAM(context, deterministic, dataTypeVal, weightType);
    OP_CHECK_IF(GetWorkspaceSize(context, deterministic, dataTypeVal != 1, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLightningIndexerKLLoss([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LightningIndexerKLLoss)
    .Tiling(LightningIndexerKLLossTilingFunc)
    .TilingParse<LightningIndexerKLLossCompileInfo>(TilingParseForLightningIndexerKLLoss);
} // namespace optiling
