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
 * \file grouped_no_quant_matmul_tiling.cpp
 * \brief
 */
#include "grouped_no_quant_matmul_tiling.h"
#include "../../../op_kernel/arch35/non_quant/grouped_matmul_tiling_key.h"
#include "gmm/common/op_host/log_format_util.h"

enum class GmmTrans {
    NoTrans = 0,
    ATrans = 1,
    BTrans = 2,
    ABTrans = 3
};
namespace optiling {
bool GroupedNoQuantMatmulTiling::SetTiling(gert::TilingContext* context)
{
    auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is nullptr."), return false);
    usedCoreNum_ = compileInfoPtr->aicNum;
    OP_CHECK_IF(!Init(context), OP_LOGE(context->GetNodeName(), "Init failed"), return false);
    OP_CHECK_IF((compileInfoPtr->npuArch == NpuArch::DAV_9201 || compileInfoPtr->npuArch == NpuArch::DAV_9202) &&
                    activeType_ != ACT_TYPE_NONE,
                OP_LOGE(context->GetNodeName(),
                        "In no quant groupedMatmul, activation is not supported on ASCEND960DT/ASCEND960PR."),
                return false);
    if (activeType_ != ACT_TYPE_NONE) {
        auto yDesc = context->GetOutputDesc(INDEX_Y);
        OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context->GetNodeName(), "yDesc is nullptr."), return false);
        const ge::DataType yDtype = yDesc->GetDataType();
        // Keep the same dtype contract as the aclnn layer: activation is only fused for FP16/BF16 output.
        OP_CHECK_IF(yDtype != ge::DT_FLOAT16 && yDtype != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                        "In no quant groupedMatmul, activation is only supported with FLOAT16 or BF16 output"),
                    return false);
    }
    OP_CHECK_IF(activeType_ == ACT_TYPE_GELU_TANH &&
                    (compileInfoPtr->aicNum == 0U || compileInfoPtr->aivNum < compileInfoPtr->aicNum),
                OP_LOGE(context->GetNodeName(), "GMM GELU requires at least one AIV per AIC."), return false);
    const bool enableSplitM =
        compileInfoPtr->aicNum != 0U &&
        static_cast<uint64_t>(compileInfoPtr->aivNum) == static_cast<uint64_t>(compileInfoPtr->aicNum) * NUM_TWO;
    splitM_ = enableSplitM ? 1U : 0U;
    OP_CHECK_IF(!CalMatMulTiling(context, compileInfoPtr),
                OP_LOGE(context->GetNodeName(), "Unable to calculate matmul-tiling"), return false);
    SetGMMTiling();
    SetMatMulTiling();
    SetTilingKey(context);
    OP_CHECK_IF(!SaveTilingData(context, tilingData_), OP_LOGE(context->GetNodeName(), "Unable to set custom param"),
                return false);
    PrintTilingResult(context);
    return true;
}

uint64_t GroupedNoQuantMatmulTiling::GetBiasReserveSize() const
{
    return hasBias_ ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0UL;
}

bool GroupedNoQuantMatmulTiling::CalBaseMMTiling(const gert::TilingContext* context,
                                                 const GMMCompileInfo* compileInfoPtr)
{
    // according to the double buffer enabled L0B, compute baseK
    baseK_ = (compileInfoPtr->l0BSize / DB_SIZE) / (baseN_ * GetSizeByDataType(xDType_));
    baseK_ = baseK_ & ~ALIGN_DOWN_16; // 16 bytes down-align
    OP_CHECK_IF(baseK_ == 0, OP_LOGE(context->GetNodeName(), "baseK_ cannot be 0."), return false);
    // according to the double buffer enabled L0A/L0C, compute baseM(cube)
    uint32_t maxBaseM = static_cast<uint32_t>(compileInfoPtr->l0CSize / (baseN_ * FP32_DTYPE_SIZE));
    baseM_ = std::min<uint32_t>((compileInfoPtr->l0ASize / DB_SIZE) / (baseK_ * GetSizeByDataType(xDType_)), maxBaseM);
    if (baseM_ > BASE_M_DEFAULT) {
        baseM_ = BASE_M_DEFAULT;
    }
    OP_CHECK_IF(baseM_ == 0, OP_LOGE(context->GetNodeName(), "baseM_ cannot be 0."), return false);
    return CalL1Tiling(context, compileInfoPtr);
}

bool GroupedNoQuantMatmulTiling::CalL1Tiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr)
{
    uint64_t reserveBTSize = hasBias_ ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0UL;
    uint64_t totalL1Size = hasBias_ ? compileInfoPtr->l1Size - reserveBTSize : compileInfoPtr->l1Size;
    uint64_t l1ASize = baseM_ > baseN_ ? PARTA_L1_SIZE : totalL1Size - PARTA_L1_SIZE;
    uint64_t l1BSize = totalL1Size - l1ASize;
    stepKa_ = l1ASize / NUM_TWO / baseM_ / baseK_ / GetSizeByDataType(xDType_);      // 2: half of l1
    stepKb_ = l1BSize / NUM_TWO / baseN_ / baseK_ / GetSizeByDataType(weightDtype_); // 2: half of l1

    OP_CHECK_IF(stepKa_ == 0 || stepKb_ == 0, OP_LOGE(context->GetNodeName(), "stepka or stepkb cannot be 0"),
                return false);
    if (stepKa_ >= stepKb_) {
        stepKa_ = stepKa_ / stepKb_ * stepKb_;
    } else {
        stepKb_ = stepKb_ / stepKa_ * stepKa_;
    }
    depthA1_ = stepKa_ * DB_SIZE;
    depthB1_ = stepKb_ * DB_SIZE;
    return true;
}

/** @brief only support split M :s-s-s and s-m-s
 */
void GroupedNoQuantMatmulTiling::SetDisableL2Cache(const gert::TilingContext* context,
                                                   const GMMCompileInfo* compileInfoPtr)
{
    uint64_t innerB = transposeWeight_ ? k_ : n_;
    uint64_t dataTypeSize = GetSizeByDataType(xDType_);
    bool flag = transposeWeight_ ? (baseK_ * stepKb_ * dataTypeSize % ALIGN_128 == 0) :
                                   (baseN_ * dataTypeSize % ALIGN_128 == 0);
    uint64_t totalSize = m_ * k_ * dataTypeSize + groupNum_ * k_ * n_ * dataTypeSize + m_ * n_ * dataTypeSize;
    OP_LOGD(context->GetNodeName(), "Input and Output TotalSize: %lu, l2Size: %lu", totalSize, compileInfoPtr->l2Size);
    if (totalSize < compileInfoPtr->l2Size) {
        weightNoL2Cache_ = false;
        return;
    }
    weightNoL2Cache_ = ((innerB * dataTypeSize % ALIGN_128 == 0) && flag || weightNzFlag_) && weightNoL2Cache_;
}

bool GroupedNoQuantMatmulTiling::Init(const gert::TilingContext* context)
{
    OP_CHECK_IF(!GetAttrs(context), OP_LOGE(context->GetNodeName(), "Unable to calculate matmul-tiling"), return false);

    auto xTensor = context->GetDynamicInputTensor(INDEX_X, 0);
    OP_CHECK_IF(xTensor == nullptr, OP_LOGE(context->GetNodeName(), "xTensor is nullptr."), return false);
    gert::Shape xShape = xTensor->GetStorageShape();
    xDimNum_ = static_cast<uint32_t>(xShape.GetDimNum());

    auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, 0);
    OP_CHECK_IF(wTensor == nullptr, OP_LOGE(context->GetNodeName(), "wTensor is nullptr."), return false);
    gert::Shape wShape = wTensor->GetOriginShape();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());

    OP_CHECK_IF(
        wDimNum < MIN_DIM || xDimNum_ < MIN_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "x, weight",
            Ops::Transformer::Gmm::FormatString("%u, %u", xDimNum_, wDimNum).c_str(),
            Ops::Transformer::Gmm::FormatString("The shape dims of %s must be at least %d", "x, weight", 2).c_str()),
        return false);
    xKDim_ = transposeX_ ? 0U : xDimNum_ - 1U;
    weightNDim_ = transposeWeight_ ? wDimNum - DIM_TWO : wDimNum - DIM_ONE;

    auto biasPtr = context->GetDynamicInputTensor(INDEX_BIAS, 0); // 0: obtain the first tensor of the tensorList
    hasBias_ = !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);

    isSingleWeight_ = (context->GetDynamicInputTensor(INDEX_WEIGHT, 1) == nullptr);
    isSingleX_ = (context->GetDynamicInputTensor(INDEX_X, 1) == nullptr);
    isSingleY_ = (splitItem_ == 2 || splitItem_ == 3); // 2&3: output tensor is single
    // Under SPLIT_M, a single weight requires a one-element weight list holding a 3-D tensor; a 2-D weight
    // is handled as the multi-tensor case (the E=1 special case of WeightNz single x/single y).
    // A legal single weight for SPLIT_K is 2-D and is not subject to this constraint.
    if (groupType_ == SPLIT_M) {
        isSingleWeight_ = (isSingleWeight_ && wDimNum == DIM_THREE);
    }

    if (weightNzFlag_) {
        uint64_t c0 = BLOCK_SIZE / std::max(1, GetSizeByDataType(weightDtype_));
        if (wDimNum > NZ_DIM_NUM) {
            weightNDim_ = transposeWeight_ ? wDimNum - DIM_THREE : wDimNum - DIM_FOUR;
            nzFactor_ = transposeWeight_ ? BASIC_BLOCK_SIZE_16 : static_cast<int64_t>(c0);
        }
        OP_CHECK_IF(!CheckWeightNZShape(context, static_cast<int64_t>(c0)),
                    OP_LOGE(context->GetNodeName(), "The shape of nz weight is invalid."), return false);
    }

    if (!HandleGroupTypeDispatch(context, xShape, wShape, wDimNum)) {
        return false;
    }
    return CheckGroupListLen(context);
}

/**
 * @brief Validates the groupList length consistency (common to all split scenarios).
 * groupNum_ is determined by the shape parsing of each split path (a single 3-D weight takes E; a multi-tensor
 * weight list takes the tensor count; a single 2-D weight takes 1), so whenever groupList is present, its
 * length must equal groupNum_.
 * Otherwise the kernel would compute with groupNum_ and silently ignore the extra groups in groupList.
 */
bool GroupedNoQuantMatmulTiling::CheckGroupListLen(const gert::TilingContext* context) const
{
    auto groupListTensor = context->GetDynamicInputTensor(INDEX_GROUPLIST, 0);
    if (groupListTensor == nullptr) {
        return true; // Whether groupList is mandatory is decided by each split path; this check only validates length
                     // consistency.
    }
    int64_t groupListLen = groupListTensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(
        groupListLen != static_cast<int64_t>(groupNum_),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "groupList length",
            Ops::Transformer::Gmm::FormatString("%ld", groupListLen).c_str(),
            Ops::Transformer::Gmm::FormatString("The value of groupList length must be equal to groupNum %u", groupNum_)
                .c_str()),
        return false);
    return true;
}

bool GroupedNoQuantMatmulTiling::HandleGroupTypeDispatch(const gert::TilingContext* context, const gert::Shape& xShape,
                                                         const gert::Shape& wShape, uint32_t wDimNum)
{
    if (groupType_ == SPLIT_K) {
        return GMMGetTensorShapeSplitK(context, xShape, wShape);
    }
    if (groupType_ == SPLIT_M) {
        return GMMGetTensorShapeSplitM(context, xShape, wShape);
    }
    if (groupType_ == NO_SPLIT) {              // not split any axis
        if (isSingleWeight_ && wDimNum > 2U) { // 2: dim of splited weight tensor
            return SeparatedXSingleWeight(context, wShape);
        }
        return SeparatedXSeparatedWeight(context);
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
        context->GetNodeName(), "groupType", Ops::Transformer::Gmm::FormatString("%ld", groupType_).c_str(),
        Ops::Transformer::Gmm::FormatString("The value of %s must be in %s", "groupType", "{-1, 0, 2}").c_str());
    return false;
}

bool GroupedNoQuantMatmulTiling::CheckWeightNZShape(const gert::TilingContext* context, int64_t numInOneBlk) const
{
    OP_CHECK_IF(numInOneBlk <= 0,
                OP_LOGE(context->GetNodeName(), "the value of numInOneBlk is invalid, %ld", numInOneBlk), return false);
    uint32_t i = 0;
    while (true) {
        auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, i++);
        if (wTensor == nullptr) {
            break;
        }
        auto wShape = wTensor->GetOriginShape();
        size_t kValue = wShape.GetDim(wShape.GetDimNum() - (transposeWeight_ ? 1 : 2));
        size_t nValue = wShape.GetDim(wShape.GetDimNum() - (transposeWeight_ ? 2 : 1));
        OP_CHECK_IF((kValue % numInOneBlk != 0 || nValue % numInOneBlk != 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "weight", Ops::Base::ToString(wShape).c_str(),
                        Ops::Transformer::Gmm::FormatString(
                            "When the format of %s is %s, %s of %s must be exactly divisible by %s", "weight",
                            "FRACTAL_NZ", "n-axis and k-axis", "weight", "32B")
                            .c_str()),
                    return false);
    }
    return true;
}

bool GroupedNoQuantMatmulTiling::GetAttrs(const gert::TilingContext* context)
{
    auto attr = context->GetAttrs();
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "attr is nullptr."),
                return false); // check attr is not null
    const bool* transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_IDX_TRANS_W);
    const bool* transposeXPtr = attr->GetAttrPointer<bool>(ATTR_IDX_TRANS_X);
    const int64_t* groupTypePtr = attr->GetAttrPointer<int64_t>(ATTR_IDX_GROUPTYPE);
    const int64_t* splitItemPtr = attr->GetAttrPointer<int64_t>(ATTR_IDX_SPLIT_ITEM);
    const int64_t* groupListTypePtr = attr->GetAttrPointer<int64_t>(ATTR_IDX_GROUP_LIST_TYPE);
    const int64_t* activeTypePtr = attr->GetAttrPointer<int64_t>(ATTR_IDX_ACT_TYPE);
    transposeWeight_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    transposeX_ = transposeXPtr != nullptr ? *transposeXPtr : false;
    groupType_ = groupTypePtr != nullptr ? *groupTypePtr : NO_SPLIT;
    splitItem_ = splitItemPtr != nullptr ? *splitItemPtr : 0;
    groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;
    activeType_ = activeTypePtr != nullptr ? *activeTypePtr : ACT_TYPE_NONE;
    OP_CHECK_IF(activeType_ != ACT_TYPE_NONE && activeType_ != ACT_TYPE_RELU && activeType_ != ACT_TYPE_GELU_TANH,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "actType", Ops::Transformer::Gmm::FormatString("%ld", activeType_).c_str(),
                    Ops::Transformer::Gmm::FormatString("In %s case, the value of %s must be in %s", "no-quant",
                                                        "actType", "{0, 1, 2}")
                        .c_str()),
                return false);

    auto xDesc = context->GetDynamicInputDesc(INDEX_X, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "xDesc is nullptr."), return false);
    xDType_ = xDesc->GetDataType();

    auto w0Desc = context->GetDynamicInputDesc(INDEX_WEIGHT, 0);
    OP_CHECK_IF(w0Desc == nullptr, OP_LOGE(context->GetNodeName(), "w0Desc is nullptr."), return false);
    weightDtype_ = w0Desc->GetDataType();
    auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
    weightNzFlag_ = wFormat0 == ge::FORMAT_FRACTAL_NZ;
    if (!CheckNoQuantGroupList(context)) {
        return false;
    }
    OP_CHECK_IF(activeType_ != ACT_TYPE_NONE && groupType_ == SPLIT_K,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "groupType", Ops::Transformer::Gmm::FormatString("%ld", groupType_).c_str(),
                    Ops::Transformer::Gmm::FormatString("In %s case with %s enabled, the value of %s must be in %s",
                                                        "no-quant", "activation", "groupType", "{-1, 0}")
                        .c_str()),
                return false);
    return true;
}

bool GroupedNoQuantMatmulTiling::CheckNoQuantGroupList(const gert::TilingContext* context) const
{
    OP_CHECK_IF(
        groupListType_ != GroupedMatmul::GROUPLIST_TYPE_CUMSUM &&
            groupListType_ != GroupedMatmul::GROUPLIST_TYPE_COUNT &&
            groupListType_ != GroupedMatmul::GROUPLIST_TYPE_SPARSE_M,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "groupListType", Ops::Transformer::Gmm::FormatString("%u", groupListType_).c_str(),
            Ops::Transformer::Gmm::FormatString("In %s case, the value of %s must be in %s", "no-quant",
                                                "groupListType", "{0, 1, 2}")
                .c_str()),
        return false);
    // Sparse groupList ([E, 2], non-zero groups front-loaded) only splits the M axis.
    OP_CHECK_IF(
        groupListType_ == GroupedMatmul::GROUPLIST_TYPE_SPARSE_M && groupType_ != SPLIT_M,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "groupListType", Ops::Transformer::Gmm::FormatString("%u", groupListType_).c_str(),
            Ops::Transformer::Gmm::FormatString("In %s case, %s %s only supports %s %s", "no-quant", "groupListType",
                                                "2(sparse)", "groupType", "0(split-M)")
                .c_str()),
        return false);
    auto groupListTensor = context->GetDynamicInputTensor(INDEX_GROUPLIST, 0);
    if (groupListTensor == nullptr) {
        return true; // The optional groupList presence is enforced later by the split paths and the kernel.
    }
    gert::Shape groupListShape = groupListTensor->GetStorageShape();
    size_t groupListDimNum = groupListShape.GetDimNum();
    // groupListType 0(cumsum)/1(count): groupList is 1D; groupListType 2(sparse): groupList is 2D [E, 2].
    if (groupListType_ == GroupedMatmul::GROUPLIST_TYPE_SPARSE_M) {
        OP_CHECK_IF(
            groupListDimNum != DIM_TWO,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "groupList",
                                         Ops::Transformer::Gmm::FormatString("%zu", groupListDimNum).c_str(), "2"),
            return false);
        OP_CHECK_IF(
            groupListShape.GetDim(1) != DIM_TWO,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), "groupList", Ops::Base::ToString(groupListShape).c_str(),
                Ops::Transformer::Gmm::FormatString("The %s of %s must be equal to %s when %s is %s", "2nd dimension",
                                                    "groupList", "2", "groupListType", "2(sparse)")
                    .c_str()),
            return false);
    } else {
        OP_CHECK_IF(
            groupListDimNum != DIM_ONE,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "groupList",
                                         Ops::Transformer::Gmm::FormatString("%zu", groupListDimNum).c_str(), "1"),
            return false);
    }
    return true;
}

bool GroupedNoQuantMatmulTiling::CalMatMulTiling(const gert::TilingContext* context,
                                                 const GMMCompileInfo* compileInfoPtr)
{
    if (groupNum_ < 1U || groupNum_ > MAX_TENSOR) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "groupNum", Ops::Transformer::Gmm::FormatString("%u", groupNum_).c_str(),
            Ops::Transformer::Gmm::FormatString("The value of %s must be within the range [%u, %u]", "groupNum", 1U,
                                                MAX_TENSOR)
                .c_str());
        return false;
    }
    OP_CHECK_IF(
        kZero_,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "K-axis of x", Ops::Transformer::Gmm::FormatString("%s", "0").c_str(),
            Ops::Transformer::Gmm::FormatString("The value of %s must be a positive number", "K-axis of x").c_str()),
        return false);
    OP_CHECK_IF(!CalBaseMMTiling(context, compileInfoPtr),
                OP_LOGE(context->GetNodeName(), "Unable to calculate BaseMMTiling."), return false);
    if (groupNum_ == 1U) {
        CalSplitKBasicBlockTiling(compileInfoPtr);
        SetDisableL2Cache(context, compileInfoPtr);
        SetL0cDbTiling(compileInfoPtr); // m/n fixed per group in the no-split case; the host decides L0C double
                                        // buffering directly.
        return true;
    }
    if (groupType_ == SPLIT_M || groupType_ == NO_SPLIT) {
        SetDisableL2Cache(context, compileInfoPtr);
        SetL0cDbTiling(compileInfoPtr);
        return true;
    } else if (groupType_ == SPLIT_K) {
        CalSplitKBasicBlockTiling(compileInfoPtr);
        SetL0cDbTiling(
            compileInfoPtr); // m/n fixed per group in the K-split case; the host decides L0C double buffering directly.
        return true;
    }
    return false;
}

void GroupedNoQuantMatmulTiling::SetGMMTiling()
{
    tilingData_.gmmNoQuantParam.groupNum = groupNum_;
    tilingData_.gmmNoQuantParam.hasBias = static_cast<uint32_t>(hasBias_);
    tilingData_.gmmNoQuantParam.groupType = groupType_;
    tilingData_.gmmNoQuantParam.groupListType = groupListType_;
    tilingData_.gmmNoQuantParam.singleWeight = static_cast<uint32_t>(isSingleWeight_);
    tilingData_.gmmNoQuantParam.singleX = static_cast<uint32_t>(isSingleX_);
    tilingData_.gmmNoQuantParam.singleY = static_cast<uint32_t>(isSingleY_);
    tilingData_.gmmNoQuantParam.coreNum = usedCoreNum_;
    tilingData_.gmmNoQuantParam.mTailCnt = static_cast<uint32_t>(mTailCnt_);
    tilingData_.gmmNoQuantParam.nTailCnt = static_cast<uint32_t>(nTailCnt_);
    tilingData_.gmmNoQuantParam.weightNoL2Cache = weightNoL2Cache_;
    tilingData_.gmmNoQuantParam.activeType = static_cast<uint32_t>(activeType_);
    tilingData_.gmmNoQuantParam.splitM = static_cast<uint32_t>(splitM_);
}

void GroupedNoQuantMatmulTiling::SetMatMulTiling()
{
    FillMmTilingData(tilingData_.mmTilingData);
}

void GroupedNoQuantMatmulTiling::SetTilingKey(gert::TilingContext* context)
{
    tilingKeyBuilder_.gmmTrans = static_cast<uint8_t>(transposeX_) | (static_cast<uint8_t>(transposeWeight_) << 1);
    // NO_QUANT_ACT reuses the aclnn actType value: 0=NONE, 1=RELU, 2=GELU_TANH.
    tilingKeyBuilder_.act = static_cast<uint8_t>(activeType_);
    context->SetTilingKey(tilingKeyBuilder_.GenTilingKey());
}

bool GroupedNoQuantMatmulTiling::GMMGetTensorShapeSplitM(const gert::TilingContext* context, const gert::Shape xShape,
                                                         const gert::Shape wShape)
{
    if (!isSingleX_ && !isSingleWeight_ && isSingleY_) { // split M, m-m-s
        return SeparatedXSeparatedWeight(context);
    }
    if (isSingleX_ && !isSingleWeight_ && !isSingleY_) { // splitM, s-m-m
        return SplitMSingleXSeparatedWeight(context, xShape);
    }
    if (isSingleX_ && !isSingleWeight_ && isSingleY_) { // split M, s-m-s
        return SplitMSingleXSeparatedWeight(context, xShape);
    }
    if (!isSingleX_ && !isSingleWeight_ && !isSingleY_) { // split M, m-m-m
        return SeparatedXSeparatedWeight(context);
    }
    if (!isSingleX_ && isSingleWeight_) { // split M, m-s-m/m-s-s
        return SeparatedXSingleWeight(context, wShape);
    }
    if (isSingleX_ && isSingleWeight_ && isSingleY_) { // split M, s-s-s
        return SplitMSingleXSingleWeightSingleY(xShape, wShape);
    }
    OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
        context->GetNodeName(), "x, weight, out",
        Ops::Transformer::Gmm::FormatString("%s, %s, %s", isSingleX_ ? "single" : "multi",
                                            isSingleWeight_ ? "single" : "multi", isSingleY_ ? "single" : "multi")
            .c_str(),
        Ops::Transformer::Gmm::FormatString(
            "Under SPLIT_M scene, x, weight, out only support 7 combinations: "
            "multi-multi-multi, single-single-single, multi-multi-single, single-multi-multi, "
            "single-multi-single, multi-single-multi, multi-single-single, "
            "but current x is %s tensor, weight is %s tensor, out is %s tensor, "
            "this combination is not supported",
            isSingleX_ ? "single" : "multi", isSingleWeight_ ? "single" : "multi", isSingleY_ ? "single" : "multi")
            .c_str());
    return false;
}

bool GroupedNoQuantMatmulTiling::GMMGetTensorShapeSplitK(const gert::TilingContext* context, const gert::Shape xShape,
                                                         const gert::Shape wShape)
{
    if (isSingleX_ && !isSingleWeight_ && !isSingleY_) { // splitK, s-m-m
        return SplitKSingleXSeparatedWeight(context, xShape);
    }
    if (!isSingleX_ && isSingleWeight_) { // splitK, m-s-m/m-s-s
        return SeparatedXSingleWeight(context, wShape);
    }
    if (isSingleX_ && isSingleWeight_ && isSingleY_) { // splitK, s-s-s
        return SplitKSingleXSingleWeightSingleY(context, xShape, wShape);
    }
    OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
        context->GetNodeName(), "x, weight, out",
        Ops::Transformer::Gmm::FormatString("%s, %s, %s", isSingleX_ ? "single" : "multi",
                                            isSingleWeight_ ? "single" : "multi", isSingleY_ ? "single" : "multi")
            .c_str(),
        Ops::Transformer::Gmm::FormatString(
            "Under SPLIT_K scene, x, weight, out only support 4 combinations: "
            "single-multi-multi, multi-single-multi, multi-single-single, single-single-single, "
            "but current x is %s tensor, weight is %s tensor, out is %s tensor, "
            "this combination is not supported",
            isSingleX_ ? "single" : "multi", isSingleWeight_ ? "single" : "multi", isSingleY_ ? "single" : "multi")
            .c_str());
    return false;
}

void GroupedNoQuantMatmulTiling::PrintTilingResult(const gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(),
            "GMM Tiling result: groupNum: %u, singleX: %u, singleWeight: %u, singleY: %u,"
            "groupType: %d, groupListType: %u, hasBias: %u, mTailCnt: %u, nTailCnt: %u, weightNoL2Cache: %u,"
            "activeType: %u, splitM: %u",
            tilingData_.gmmNoQuantParam.groupNum, tilingData_.gmmNoQuantParam.singleX,
            tilingData_.gmmNoQuantParam.singleWeight, tilingData_.gmmNoQuantParam.singleY,
            tilingData_.gmmNoQuantParam.groupType, tilingData_.gmmNoQuantParam.groupListType,
            tilingData_.gmmNoQuantParam.hasBias, tilingData_.gmmNoQuantParam.mTailCnt,
            tilingData_.gmmNoQuantParam.nTailCnt, tilingData_.gmmNoQuantParam.weightNoL2Cache,
            tilingData_.gmmNoQuantParam.activeType, tilingData_.gmmNoQuantParam.splitM);

    OP_LOGD(context->GetNodeName(),
            "GMM MatMul Tiling result: M: %d, N: %d, Ka: %d, mL1: %d, nL1: %d, kL1: %d, baseM: %d, baseN: %d,"
            " baseK: %d, stepKa: %d, stepKb: %d, dbL0C: %d",
            tilingData_.mmTilingData.M, tilingData_.mmTilingData.N, tilingData_.mmTilingData.Ka,
            tilingData_.mmTilingData.mL1, tilingData_.mmTilingData.nL1, tilingData_.mmTilingData.kL1,
            tilingData_.mmTilingData.baseM, tilingData_.mmTilingData.baseN, tilingData_.mmTilingData.baseK,
            tilingData_.mmTilingData.stepKa, tilingData_.mmTilingData.stepKb, tilingData_.mmTilingData.dbL0C);
}

uint64_t TilingKeyBuilder::GenTilingKey() const
{
    uint64_t transInfo = static_cast<uint64_t>(this->gmmTrans);
    bool atrans_ = (transInfo == static_cast<uint64_t>(GmmTrans::ATrans)) ||
                   (transInfo == static_cast<uint64_t>(GmmTrans::ABTrans));
    bool btrans_ = (transInfo == static_cast<uint64_t>(GmmTrans::BTrans)) ||
                   (transInfo == static_cast<uint64_t>(GmmTrans::ABTrans));
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(btrans_), static_cast<uint64_t>(atrans_),
                              static_cast<uint64_t>(this->act));
}
} // namespace optiling
