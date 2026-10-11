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
 * \file moe_masked_scatter_tiling.cpp
 * \brief
 */
#include "moe_masked_scatter_tiling.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling {
constexpr int64_t OUTPUT_Y_IDX = 0;
constexpr int64_t INPUT_X_IDX = 0;
constexpr int64_t INPUT_MASK_IDX = 1;
constexpr int64_t INPUT_UPDATES_IDX = 2;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t MIN_DATA_SIZE = 1024;
constexpr int64_t DCACHE_SIZE = 32768;
static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8,
                                                     ge::DT_INT8,  ge::DT_INT16,   ge::DT_INT32,  ge::DT_INT64,
                                                     ge::DT_BOOL,  ge::DT_BF16};

inline static bool IsSupportDtype(const std::set<ge::DataType>& supportDtype, const ge::DataType dtype)
{
    return (supportDtype.count(dtype) != 0);
}

bool MoeMaskedScatterTiling::IsCapable()
{
    return true;
}

ge::graphStatus MoeMaskedScatterTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize_ = ubSizePlatform - DCACHE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeMaskedScatterTiling::CheckDataType()
{
    auto inputXPtr = context_->GetRequiredInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    dType_ = inputXPtr->GetDataType();
    auto inputMaskPtr = context_->GetRequiredInputDesc(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputMaskPtr);
    auto maskDtype = inputMaskPtr->GetDataType();
    OP_CHECK_IF(maskDtype != ge::DT_BOOL,
                OP_LOGE(context_->GetNodeName(), "The dtype of mask only support bool currently, please check."),
                return ge::GRAPH_FAILED);
    auto inputUpdatesPtr = context_->GetRequiredInputDesc(INPUT_UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputUpdatesPtr);
    auto updatesDtype = inputUpdatesPtr->GetDataType();
    auto outputYPtr = context_->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(dType_ != updatesDtype || dType_ != yDtype,
                OP_LOGE(context_->GetNodeName(), "The x, updates and y must have the same dtype, please check."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !IsSupportDtype(SUPPORT_DTYPE, dType_),
        OP_LOGE(context_->GetNodeName(),
                "The dtype only support float32, float16, double, uint8, int8, int16, int32, int64, bool, bfloat16 \
currently, please check."),
        return ge::GRAPH_FAILED);
    dataTypeSize_ = ge::GetSizeByDataType(dType_);
    OP_CHECK_IF(dataTypeSize_ == -1, OP_LOGE(context_->GetNodeName(), "Get the size of dtype failed, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeMaskedScatterTiling::CheckOutputShape()
{
    auto xShapePtr = context_->GetRequiredInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto updatesShapePtr = context_->GetRequiredInputShape(INPUT_UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatesShapePtr);
    auto updatesShape = updatesShapePtr->GetStorageShape();
    updateEleNums_ = updatesShape.GetShapeSize();
    auto maskShapePtr = context_->GetRequiredInputShape(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskShapePtr);
    auto maskShape = maskShapePtr->GetStorageShape();
    auto yShapePtr = context_->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape != maskShape || xShape != yShape,
                OP_LOGE(context_->GetNodeName(), "The x, mask and y must have the same shape, please check."),
                return ge::GRAPH_FAILED);
    inputShape_ = xShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeMaskedScatterTiling::GetShapeAttrsInfo()
{
    if (CheckDataType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return CheckOutputShape();
}

ge::graphStatus MoeMaskedScatterTiling::DoOpTiling()
{
    xEleNums_ = inputShape_.GetShapeSize();
    isInt64Mask_ = xEleNums_ > INT32_MAX;
    int64_t prefixSumDtypeSize =
        isInt64Mask_ ? static_cast<int64_t>(sizeof(int64_t)) : static_cast<int64_t>(sizeof(int32_t));
    int64_t maxUbAvailable = ubSize_ / (DOUBLE_BUFFER + DIGIT_TWO * prefixSumDtypeSize);
    normalCoreData_ = std::max(Ops::Base::CeilDiv(xEleNums_, totalCoreNum_), MIN_DATA_SIZE);
    usedCoreNum_ = Ops::Base::CeilDiv(xEleNums_, normalCoreData_);
    tailCoreData_ = xEleNums_ - (usedCoreNum_ - 1) * normalCoreData_;
    int64_t vfLen = Ops::Base::GetVRegSize(context_) / prefixSumDtypeSize;
    ubFactor_ = maxUbAvailable / vfLen * vfLen;
    blockFactor_ = Ops::Base::CeilDiv(normalCoreData_, ubFactor_);
    tailBlockFactor_ = Ops::Base::CeilDiv(tailCoreData_, ubFactor_);
    tailUbFactor_ = normalCoreData_ % ubFactor_;
    tailUbFactor_ = tailUbFactor_ == 0 ? ubFactor_ : tailUbFactor_;
    tailCoreTailUbFactor_ = tailCoreData_ % ubFactor_;
    tailCoreTailUbFactor_ = tailCoreTailUbFactor_ == 0 ? ubFactor_ : tailCoreTailUbFactor_;
    if (dataTypeSize_ != 1 && dataTypeSize_ != DIGIT_TWO) {
        return ge::GRAPH_SUCCESS;
    }
    // 执行scatter操作时，如果位宽是1或者2，先scatter到ub，再搬出，此时需要冲切UB
    int64_t oneBlockSize = Ops::Base::GetUbBlockSize(context_);
    prefixSumUbSize_ = Ops::Base::CeilAlign(totalCoreNum_ * prefixSumDtypeSize, oneBlockSize);
    bufferSize_ = (ubSize_ - prefixSumUbSize_) / (DOUBLE_BUFFER + DOUBLE_BUFFER * dataTypeSize_ + dataTypeSize_);
    bufferSize_ = Ops::Base::FloorAlign(bufferSize_, oneBlockSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeMaskedScatterTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeMaskedScatterTiling::GetTilingKey() const
{
    // tilingKey由两位组成，第一位表示x的位宽，第二位表示mask的shapeSize大于/小于INT32_MAX, 0表示小于等于，1表示大于
    return isInt64Mask_ ? dataTypeSize_ * DIGIT_TEN + 1 : dataTypeSize_ * DIGIT_TEN;
}

ge::graphStatus MoeMaskedScatterTiling::GetWorkspaceSize()
{
    int64_t maskDataTypeSize =
        isInt64Mask_ ? static_cast<int64_t>(sizeof(int64_t)) : static_cast<int64_t>(sizeof(int32_t));
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE + usedCoreNum_ * maskDataTypeSize + xEleNums_ * maskDataTypeSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeMaskedScatterTiling::PostTiling()
{
    tilingData_.set_normalCoreData(normalCoreData_);
    tilingData_.set_tailCoreData(tailCoreData_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_tailBlockFactor(tailBlockFactor_);
    tilingData_.set_tailUbFactor(tailUbFactor_);
    tilingData_.set_tailCoreTailUbFactor(tailCoreTailUbFactor_);
    tilingData_.set_updateEleNums(updateEleNums_);
    tilingData_.set_bufferSize(bufferSize_);
    tilingData_.set_prefixSumUbSize(prefixSumUbSize_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeMaskedScatterTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "usedCoreNum: " << usedCoreNum_;
    info << ", normalCoreData: " << normalCoreData_;
    info << ", tailCoreData: " << tailCoreData_;
    info << ", ubFactor: " << ubFactor_;
    info << ", blockFactor: " << blockFactor_;
    info << ", tailBlockFactor: " << tailBlockFactor_;
    info << ", tailUbFactor:" << tailUbFactor_;
    info << ", tailCoreTailUbFactor: " << tailCoreTailUbFactor_;
    info << ", updateEleNums: " << updateEleNums_;
    info << ", bufferSize: " << bufferSize_;
    info << ", prefixSumUbSize: " << prefixSumUbSize_;
    info << ", tilingKey: " << GetTilingKey();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

static ge::graphStatus Tiling4MoeMaskedScatter(gert::TilingContext* context)
{
    MoeMaskedScatterTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4MoeMaskedScatter(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeMaskedScatter)
    .Tiling(Tiling4MoeMaskedScatter)
    .TilingParse<MoeMaskedScatterCompileInfo>(TilingPrepare4MoeMaskedScatter);
} // namespace optiling
