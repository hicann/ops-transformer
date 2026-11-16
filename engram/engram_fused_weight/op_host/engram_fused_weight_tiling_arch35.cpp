/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_fused_weight_tiling_arch35.cpp
 * \brief EngramFusedWeight tiling 实现 (arch35 / ascend950)
 *
 * 输入范围约束:
 *   1. weight_hidden / weight_embed 均为 2D [hcMult, hiddenSize] 且 shape 相同;
 *   2. dtype 支持 BF16 / FLOAT16 (两个输入 dtype 一致), 输出固定 FP32;
 *   3. 总元素数满足 numElements % 256 == 0 (numElements == 0 允许, 直接返回空结果);
 *   4. 无规模上限: blockSize 随总量自适应。
 */

#include <algorithm>
#include <sstream>
#include "engram_fused_weight_tiling_arch35.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

ge::graphStatus EngramFusedWeightTilingArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<EngramFusedWeightCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFusedWeightTilingArch35::ValidateShapes(const gert::Shape& whStorage,
                                                              const gert::Shape& weStorage,
                                                              const gert::Shape& outStorage)
{
    const char* nodeName = context_->GetNodeName();

    OP_CHECK_IF(
        whStorage.GetDimNum() != INPUT_DIM_NUM,
        OP_LOGE(nodeName, "weight_hidden must be 2D [hcMult, hiddenSize], got dimNum=%zu", whStorage.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        weStorage.GetDimNum() != INPUT_DIM_NUM,
        OP_LOGE(nodeName, "weight_embed must be 2D [hcMult, hiddenSize], got dimNum=%zu", weStorage.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        weStorage.GetDim(0) != whStorage.GetDim(0) || weStorage.GetDim(1) != whStorage.GetDim(1),
        OP_LOGE(nodeName, "weight_hidden and weight_embed must have the same shape, got [%ld, %ld] and [%ld, %ld]",
                whStorage.GetDim(0), whStorage.GetDim(1), weStorage.GetDim(0), weStorage.GetDim(1)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outStorage.GetDimNum() != INPUT_DIM_NUM || outStorage.GetDim(0) != whStorage.GetDim(0) ||
                    outStorage.GetDim(1) != whStorage.GetDim(1),
                OP_LOGE(nodeName, "weight_fused must have the same shape as weight_hidden [%ld, %ld]",
                        whStorage.GetDim(0), whStorage.GetDim(1)),
                return ge::GRAPH_FAILED);

    totalElements_ = whStorage.GetShapeSize();
    // 总元素数须被 256 整除; 0 元素 (空输入) 允许, kernel 侧直接返回。
    OP_CHECK_IF(
        totalElements_ != 0 && totalElements_ % TOTAL_ELEM_ALIGN != 0,
        OP_LOGE(nodeName, "the total number of input elements must be divisible by 256, got %ld", totalElements_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFusedWeightTilingArch35::ValidateDtypes()
{
    const char* nodeName = context_->GetNodeName();

    auto whDesc = context_->GetInputDesc(WEIGHT_HIDDEN_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, whDesc);
    OP_CHECK_IF(whDesc->GetDataType() != ge::DT_BF16 && whDesc->GetDataType() != ge::DT_FLOAT16,
                OP_LOGE(nodeName, "weight_hidden dtype only support BFLOAT16 or FLOAT16, got %d",
                        static_cast<int>(whDesc->GetDataType())),
                return ge::GRAPH_FAILED);

    auto weDesc = context_->GetInputDesc(WEIGHT_EMBED_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weDesc);
    OP_CHECK_IF(weDesc->GetDataType() != ge::DT_BF16 && weDesc->GetDataType() != ge::DT_FLOAT16,
                OP_LOGE(nodeName, "weight_embed dtype only support BFLOAT16 or FLOAT16, got %d",
                        static_cast<int>(weDesc->GetDataType())),
                return ge::GRAPH_FAILED);

    // kernel 侧两个输入共用同一模板类型 T0, dtype 必须一致。
    OP_CHECK_IF(whDesc->GetDataType() != weDesc->GetDataType(),
                OP_LOGE(nodeName, "weight_hidden and weight_embed must have the same dtype"), return ge::GRAPH_FAILED);

    auto outDesc = context_->GetOutputDesc(WEIGHT_FUSED_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);
    OP_CHECK_IF(
        outDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(nodeName, "weight_fused dtype only support FLOAT, got %d", static_cast<int>(outDesc->GetDataType())),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFusedWeightTilingArch35::GetShapeAndDtypeInfo()
{
    auto whShape = context_->GetInputShape(WEIGHT_HIDDEN_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, whShape);
    auto weShape = context_->GetInputShape(WEIGHT_EMBED_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weShape);
    auto outShape = context_->GetOutputShape(WEIGHT_FUSED_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outShape);

    OP_CHECK_IF(ValidateShapes(whShape->GetStorageShape(), weShape->GetStorageShape(), outShape->GetStorageShape()) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ValidateShapes failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtypes() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "ValidateDtypes failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFusedWeightTilingArch35::DoOpTiling()
{
    // 空输入: blockDim 固定 1, 全部因子置 0, kernel 侧按 perCoreElem == 0 直接返回。
    if (totalElements_ == 0) {
        usedCoreNums_ = 1;
        tilingData_.set_totalElements(0);
        tilingData_.set_blockSize(0);
        tilingData_.set_perCoreElem(0);
        tilingData_.set_blockFactor(0);
        tilingData_.set_tailBlockFactor(0);
        tilingData_.set_perCoreTailElem(0);
        tilingData_.set_tailCoreTailElem(0);
        tilingData_.set_usedCoreNums(usedCoreNums_);
        return ge::GRAPH_SUCCESS;
    }

    int64_t blockSize =
        static_cast<int64_t>((ubSize_ - UB_RESERVED_BYTE) / UB_BYTES_PER_ELEMENT); // 单次 UB 能加载的元素数量
    int64_t perCoreElem = static_cast<int64_t>((totalElements_ + coreNum_ - 1) / coreNum_); // 头核要处理的元素数量

    if (totalElements_ <= TOTAL_ELEM_SMALL) {
        perCoreElem = std::max(perCoreElem, BLOCK_SIZE_SMALL);
    } else if (totalElements_ <= TOTAL_ELEM_MEDIUM) {
        perCoreElem = std::max(perCoreElem, BLOCK_SIZE_MEDIUM);
    } else if (totalElements_ <= TOTAL_ELEM_LARGE) {
        perCoreElem = std::max(perCoreElem, BLOCK_SIZE_LARGE);
    } else {
        perCoreElem = std::max(perCoreElem, BLOCK_SIZE_MAX);
    }

    usedCoreNums_ = (totalElements_ + perCoreElem - 1) / perCoreElem;
    OP_CHECK_IF(usedCoreNums_ <= 0,
                OP_LOGE(context_->GetNodeName(), "usedCoreNums %ld must be positive", usedCoreNums_),
                return ge::GRAPH_FAILED);

    int64_t blockFactor = (perCoreElem + blockSize - 1) / blockSize;           // 头核的 UB 循环次数
    int64_t tailCoreElem = totalElements_ - (usedCoreNums_ - 1) * perCoreElem; // 尾核要处理的元素数量
    int64_t tailBlockFactor = static_cast<int64_t>((tailCoreElem + blockSize - 1) / blockSize); // 尾核的 UB 循环次数

    int64_t perCoreTailElem = perCoreElem - (blockFactor - 1) * blockSize; // 头核最后一次 UB 循环加载的元素数量
    int64_t tailCoreTailElem = tailCoreElem - (tailBlockFactor - 1) * blockSize; // 尾核最后一次 UB 循环加载的元素数量

    tilingData_.set_totalElements(totalElements_);
    tilingData_.set_blockSize(blockSize);
    tilingData_.set_perCoreElem(perCoreElem);
    tilingData_.set_blockFactor(blockFactor);
    tilingData_.set_tailBlockFactor(tailBlockFactor);
    tilingData_.set_perCoreTailElem(perCoreTailElem);
    tilingData_.set_tailCoreTailElem(tailCoreTailElem);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFusedWeightTilingArch35::PostTiling()
{
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNums_));
    context_->SetTilingKey(0);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "get workspaces nullptr"),
                return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);

    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata nullptr"),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

void EngramFusedWeightTilingArch35::DumpTilingInfo()
{
    std::ostringstream info;
    info << "totalElements: " << tilingData_.get_totalElements();
    info << ", blockSize: " << tilingData_.get_blockSize();
    info << ", perCoreElem: " << tilingData_.get_perCoreElem();
    info << ", blockFactor: " << tilingData_.get_blockFactor();
    info << ", tailBlockFactor: " << tilingData_.get_tailBlockFactor();
    info << ", perCoreTailElem: " << tilingData_.get_perCoreTailElem();
    info << ", tailCoreTailElem: " << tilingData_.get_tailCoreTailElem();
    info << ", usedCoreNums: " << tilingData_.get_usedCoreNums();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus EngramFusedWeightTilingArch35::RunTiling()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("engram_fused_weight_tiling", "failed to get tiling context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAndDtypeInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "GetShapeAndDtypeInfo failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "DoOpTiling failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(PostTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "PostTiling failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForEngramFusedWeight(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("EngramFusedWeight", "Tiling context is null"),
                return ge::GRAPH_FAILED);

    EngramFusedWeightTilingArch35 tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForEngramFusedWeight(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFusedWeight)
    .Tiling(TilingForEngramFusedWeight)
    .TilingParse<EngramFusedWeightCompileInfo>(TilingPrepareForEngramFusedWeight);

} // namespace optiling
