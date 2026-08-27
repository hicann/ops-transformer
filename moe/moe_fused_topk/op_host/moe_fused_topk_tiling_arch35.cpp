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
 * \file moe_fused_topk_tiling_arch35.cpp
 * \brief Standalone Ascend950 tiling implementation for MoeFusedTopk.
 */

#include "moe_fused_topk_tiling_arch35.h"

#include <algorithm>
#include <vector>

#include "err/ops_err.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "platform/platform_info.h"
#include "securec.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"

namespace {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t ADD_NUM_INPUT_INDEX = 1;
constexpr int32_t MAPPING_TABLE_INPUT_INDEX = 3;

constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;

constexpr uint32_t ATTR_GROUP_NUM_INDEX = 0;
constexpr uint32_t ATTR_GROUP_TOPK_INDEX = 1;
constexpr uint32_t ATTR_TOP_N_INDEX = 2;
constexpr uint32_t ATTR_TOP_K_INDEX = 3;
constexpr uint32_t ATTR_ACTIVATE_TYPE_INDEX = 4;
constexpr uint32_t ATTR_IS_NORM_INDEX = 5;
constexpr uint32_t ATTR_SCALE_INDEX = 6;
constexpr uint32_t ATTR_ENABLE_EXPERT_MAPPING_INDEX = 7;

constexpr uint32_t RESERVED_UB = 16 * 1024;
constexpr uint32_t SORT_UNIT = 32;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

constexpr bool TOPK_IS_REUSE_SOURCE = false;
constexpr bool TOPK_IS_INIT_INDEX = false;
constexpr bool TOPK_IS_LARGEST = true;
constexpr uint32_t FP32_DTYPE_SIZE = 4U;
constexpr uint32_t INT32_DTYPE_SIZE = 4U;
constexpr uint32_t FP32_BLOCK_ALIGN_NUM = 8U;
constexpr uint32_t FP16_BLOCK_ALIGN_NUM = 16U;
} // namespace

namespace optiling {
using Ops::Base::CeilAlign;

class MoeFusedTopkTilingArch35 {
public:
    explicit MoeFusedTopkTilingArch35(gert::TilingContext *context)
        : context_(context)
    {}

    ge::graphStatus Run();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    void GetTilingKey();
    void GetUsedCore();
    ge::graphStatus GetTopKTiling();
    void GetTmpBuffSize();
    ge::graphStatus SplitUb();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint();

    gert::TilingContext *context_ = nullptr;
    uint32_t firstDimSize_ = 0;
    uint32_t secondDimSize_ = 0;
    uint32_t addNumDimSize_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t groupTopk_ = 0;
    uint32_t topN_ = 0;
    uint32_t topK_ = 0;
    uint32_t activateType_ = 0;
    uint32_t isNorm_ = 0;
    float scale_ = 1.0F;
    bool enableExpertMapping_ = false;
    uint32_t groupEles_ = 0;
    uint32_t expertNum_ = 0;
    uint32_t tableDim_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t batchPerCore_ = 1;
    uint32_t tailBatch_ = 0;
    uint32_t ubFactorElement_ = 0;
    uint32_t topkMaxValue_ = 0;
    uint32_t topkMinValue_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t workspacePerCore_ = 0;
    MoeFusedTopkArch35TilingData tilingData_{};
};

ge::graphStatus MoeFusedTopkTilingArch35::GetPlatformInfo()
{
    auto compileInfo = static_cast<const MoeFusedTopkCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    OP_CHECK_IF(compileInfo->ubSize <= RESERVED_UB,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "MoeFusedTopk GetHardwareInfo Failed, ubSize: %lu",
                                            compileInfo->ubSize),
                return ge::GRAPH_FAILED);

    coreNum_ = compileInfo->coreNum;
    ubSize_ = (compileInfo->ubSize - RESERVED_UB) / BYTE_BLOCK * BYTE_BLOCK;
    OP_CHECK_IF(coreNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "MoeFusedTopk GetHardwareInfo Failed, vectorCoreNum: %u", coreNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "MoeFusedTopk GetHardwareInfo Failed, ubSize: %lu",
                                            ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingArch35::GetShapeAttrsInfo()
{
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    auto addNumShapePtr = context_->GetInputShape(ADD_NUM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, addNumShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto addNumShape = addNumShapePtr->GetStorageShape();
    firstDimSize_ = static_cast<uint32_t>(xShape.GetDim(DIM_INDEX0));
    secondDimSize_ = static_cast<uint32_t>(xShape.GetDim(DIM_INDEX1));
    addNumDimSize_ = static_cast<uint32_t>(addNumShape.GetDim(DIM_INDEX0));

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    groupNum_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_GROUP_NUM_INDEX));
    groupTopk_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_GROUP_TOPK_INDEX));
    topN_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_TOP_N_INDEX));
    topK_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_TOP_K_INDEX));
    activateType_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_ACTIVATE_TYPE_INDEX));
    isNorm_ = static_cast<uint32_t>(*(attrs->GetAttrPointer<bool>(ATTR_IS_NORM_INDEX)));
    scale_ = *(attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX));
    enableExpertMapping_ = *(attrs->GetAttrPointer<bool>(ATTR_ENABLE_EXPERT_MAPPING_INDEX));
    groupEles_ = groupNum_ == 0 ? secondDimSize_ : secondDimSize_ / groupNum_;

    if (enableExpertMapping_) {
        auto mappingTableShapePtr = context_->GetInputShape(MAPPING_TABLE_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, mappingTableShapePtr);
        auto mappingTableShape = mappingTableShapePtr->GetStorageShape();
        expertNum_ = static_cast<uint32_t>(mappingTableShape.GetDim(DIM_INDEX0));
        tableDim_ = static_cast<uint32_t>(mappingTableShape.GetDim(DIM_INDEX1));
    }
    return ge::GRAPH_SUCCESS;
}

void MoeFusedTopkTilingArch35::GetTilingKey()
{
    tilingKey_ = enableExpertMapping_ ? 1U : 0U;
}

void MoeFusedTopkTilingArch35::GetUsedCore()
{
    if (firstDimSize_ <= coreNum_) {
        batchPerCore_ = 1;
        usedCoreNum_ = firstDimSize_;
        tailBatch_ = 0;
        return;
    }
    batchPerCore_ = firstDimSize_ / coreNum_;
    tailBatch_ = firstDimSize_ % coreNum_;
    usedCoreNum_ = coreNum_;
}

ge::graphStatus MoeFusedTopkTilingArch35::GetTopKTiling()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    int32_t topkInner = static_cast<int32_t>(CeilAlign(groupEles_, SORT_UNIT));

    OP_CHECK_IF(
        !AscendC::TopKTilingFunc(ascendcPlatform, topkInner, groupNum_, topN_, FP32_DTYPE_SIZE, TOPK_IS_INIT_INDEX,
                                 AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST, tilingData_.topkTilingData),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "TopKTilingFunc Failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, topkInner, groupNum_, TOPK_IS_REUSE_SOURCE,
                                               TOPK_IS_INIT_INDEX, AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST,
                                               FP32_DTYPE_SIZE, topkMaxValue_, topkMinValue_),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetTopKMaxMinTmpSize Failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MoeFusedTopkTilingArch35::GetTmpBuffSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    std::vector<int64_t> sigmoidVec = {static_cast<int64_t>(secondDimSize_)};
    ge::Shape sigmoidShape(sigmoidVec);
    AscendC::GetSigmoidMaxMinTmpSize(sigmoidShape, FP32_DTYPE_SIZE, false, maxValue, minValue);
    topkMinValue_ = std::max(topkMinValue_, minValue);
    topkMaxValue_ = std::max(topkMaxValue_, maxValue);

    uint32_t groupSortTmpSize =
        AscendC::GetSortTmpSize(ascendcPlatform, CeilAlign(groupEles_, SORT_UNIT), FP32_DTYPE_SIZE);
    topkMinValue_ = std::max(topkMinValue_, groupSortTmpSize);
    topkMaxValue_ = std::max(topkMaxValue_, groupSortTmpSize);

    std::vector<int64_t> srcBroadCastVec = {static_cast<int64_t>(groupNum_), 1};
    std::vector<int64_t> dstBroadCastVec = {static_cast<int64_t>(groupNum_), static_cast<int64_t>(groupEles_)};
    ge::Shape srcBroadCastShape(srcBroadCastVec);
    ge::Shape dstBroadCastShape(dstBroadCastVec);
    AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcBroadCastShape, dstBroadCastShape, FP32_DTYPE_SIZE, false,
                                       maxValue, minValue);
    topkMinValue_ = std::max(topkMinValue_, minValue);
    topkMaxValue_ = std::max(topkMaxValue_, maxValue);

    uint32_t secondDimSizeAlignSortCount = CeilAlign(secondDimSize_, SORT_UNIT);
    uint32_t expertSortTmpSize = AscendC::GetSortTmpSize(ascendcPlatform, secondDimSizeAlignSortCount, FP32_DTYPE_SIZE);
    topkMinValue_ = std::max(topkMinValue_, expertSortTmpSize);
    topkMaxValue_ = std::max(topkMaxValue_, expertSortTmpSize);

    constexpr uint32_t VECTOR_FP32_ELEMS = 256U / FP32_DTYPE_SIZE;
    uint32_t sumRepeatCount = (topN_ + VECTOR_FP32_ELEMS - 1) / VECTOR_FP32_ELEMS;
    uint32_t sumTmpSize = CeilAlign(sumRepeatCount, SORT_UNIT) * groupNum_ * FP32_DTYPE_SIZE;
    topkMaxValue_ = std::max(topkMaxValue_, sumTmpSize);

    topkMinValue_ = std::max(topkMinValue_, secondDimSizeAlignSortCount * FP32_DTYPE_SIZE);
    topkMaxValue_ = std::max(topkMaxValue_, topkMinValue_);
}

ge::graphStatus MoeFusedTopkTilingArch35::SplitUb()
{
    uint64_t needUbSize = 0;
    uint64_t tilingDataSize = CeilAlign(tilingData_.GetDataSize(), static_cast<size_t>(BYTE_BLOCK));
    ubFactorElement_ = static_cast<uint32_t>((ubSize_ - tilingDataSize) / BYTE_BLOCK);

    needUbSize += tilingDataSize;
    needUbSize += CeilAlign(groupEles_, SORT_UNIT) * groupNum_ * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(secondDimSize_, FP16_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(topK_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(topK_, SORT_UNIT) * INT32_DTYPE_SIZE;
    needUbSize += CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(groupEles_, SORT_UNIT) * groupNum_ * sizeof(int64_t);
    needUbSize += CeilAlign(secondDimSize_, SORT_UNIT) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(secondDimSize_, SORT_UNIT) * INT32_DTYPE_SIZE;
    needUbSize += CeilAlign(expertNum_ * INT32_DTYPE_SIZE, BYTE_BLOCK);
    needUbSize += topkMaxValue_;

    OP_CHECK_IF(needUbSize > ubSize_,
                OPS_REPORT_VECTOR_INNER_ERR(
                    context_->GetNodeName(),
                    "This case need minimum UB size is %lu, which is out of total UB size: %lu.", needUbSize, ubSize_),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "Ascend950 case need minimum UB size is %lu.", needUbSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingArch35::SetKernelTiling()
{
    workspacePerCore_ = secondDimSize_ * FP32_DTYPE_SIZE;
    size_t usedWorkspaceSize = SYS_WORKSPACE_SIZE + usedCoreNum_ * workspacePerCore_;

    tilingData_.set_firstDimSize(firstDimSize_);
    tilingData_.set_secondDimSize(secondDimSize_);
    tilingData_.set_addNumDimSize(addNumDimSize_);
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_groupTopk(groupTopk_);
    tilingData_.set_topN(topN_);
    tilingData_.set_topK(topK_);
    tilingData_.set_activateType(activateType_);
    tilingData_.set_isNorm(isNorm_);
    tilingData_.set_scale(scale_);
    tilingData_.set_groupEles(groupEles_);
    tilingData_.set_blockNum(usedCoreNum_);
    tilingData_.set_ubFactorElement(ubFactorElement_);
    tilingData_.set_batchPerCore(batchPerCore_);
    tilingData_.set_tailBatch(tailBatch_);
    tilingData_.set_expertNum(expertNum_);
    tilingData_.set_tableDim(tableDim_);
    tilingData_.set_topkMaxValue(topkMaxValue_);
    tilingData_.set_topkMinValue(topkMinValue_);
    tilingData_.set_workspacePerCore(workspacePerCore_);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData->GetData());
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingData_.GetDataSize(),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "Raw tiling data capacity %zu is smaller than required %zu.",
                                            rawTilingData->GetCapacity(), tilingData_.GetDataSize()),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);
    workspaces[0] = usedWorkspaceSize;
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void MoeFusedTopkTilingArch35::TilingDataPrint()
{
    OP_LOGD(context_->GetNodeName(), "arch35 tilingKey:           %lu", tilingKey_);
    OP_LOGD(context_->GetNodeName(), "arch35 usedCoreNum:         %u", usedCoreNum_);
    OP_LOGD(context_->GetNodeName(), "arch35 firstDimSize:        %u", tilingData_.get_firstDimSize());
    OP_LOGD(context_->GetNodeName(), "arch35 secondDimSize:       %u", tilingData_.get_secondDimSize());
    OP_LOGD(context_->GetNodeName(), "arch35 groupNum:            %u", tilingData_.get_groupNum());
    OP_LOGD(context_->GetNodeName(), "arch35 groupEles:           %u", tilingData_.get_groupEles());
    OP_LOGD(context_->GetNodeName(), "arch35 topkMaxValue:        %u", tilingData_.get_topkMaxValue());
    OP_LOGD(context_->GetNodeName(), "arch35 topkMinValue:        %u", tilingData_.get_topkMinValue());
    OP_LOGD(context_->GetNodeName(), "arch35 workspacePerCore:    %lu", tilingData_.get_workspacePerCore());
}

ge::graphStatus MoeFusedTopkTilingArch35::Run()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_);
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfo failed."),
                return ge::GRAPH_FAILED);

    GetTilingKey();
    GetUsedCore();
    OP_CHECK_IF(GetTopKTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetTopKTiling failed."),
                return ge::GRAPH_FAILED);
    GetTmpBuffSize();
    OP_CHECK_IF(SplitUb() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SplitUb failed."),
                return ge::GRAPH_FAILED);
    return SetKernelTiling();
}

ge::graphStatus TilingMoeFusedTopkArch35(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    MoeFusedTopkTilingArch35 tiling(context);
    return tiling.Run();
}

class MoeFusedTopkTilingArch35Base : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeFusedTopkTilingArch35Base(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override
    {
        return context_ != nullptr && Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return context_ == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return context_ == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        return TilingMoeFusedTopkArch35(context_);
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return context_ == nullptr ? 0U : context_->GetTilingKey();
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
};

REGISTER_OPS_TILING_TEMPLATE(MoeFusedTopk, MoeFusedTopkTilingArch35Base, 1000);
} // namespace optiling
