/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_tiling.cpp
 * \brief
 */
#include <cmath>
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "moe_gating_top_k_tiling.h"

namespace optiling {
const static int64_t GROUP_SELECT_MODE_MAX = 0;
const static int64_t GROUP_SELECT_MODE_SUM = 1;
const static int64_t RENORM_NO = 0;
const static int64_t RENORM_L1 = 1;
const static int64_t NORM_TYPE_SOFTMAX = 0;
const static int64_t NORM_TYPE_SIGMOID = 1;
const static int64_t OUT_FLAG_FALSE = 0;
const static int64_t OUT_FLAG_TRUE = 1;
const static size_t X_INPUT_DIMS = 2;
const static size_t BIAS_INPUT_DIMS = 1;
const static size_t Y_OUTPUT_DIMS = 2;
const static size_t EXPERT_IDX_OUTPUY_DIMS = 2;
const static size_t OUT_OUTPUT_DIMS = 2;
const static int64_t MAX_EXPERT_COUNT = 2048;
const static int64_t MAX_K_FOR_HASH = 64;

const static int64_t X_INPUT_INDEX = 0;
const static int64_t BIAS_INPUT_INDEX = 1;
const static int64_t INPUT_IDS_INPUT_INDEX = 2;
const static int64_t TID_TO_EID_INPUT_INDEX = 3;
const static int64_t Y_OUTPUT_INDEX = 0;
const static int64_t EXPERT_IDX_OUTPUT_INDEX = 1;
const static int64_t OUT_OUTPUT_INDEX = 2;
const static int64_t K_ATTR_INDEX = 0;
const static int64_t K_GROUP_ATTR_INDEX = 1;
const static int64_t GROUP_COUNT_ATTR_INDEX = 2;
const static int64_t GROUP_SELECT_MODE_ATTR_INDEX = 3;
const static int64_t RENORM_ATTR_INDEX = 4;
const static int64_t NORM_TYPE_ATTR_INDEX = 5;
const static int64_t OUT_FLAG_ATTR_INDEX = 6;
const static int64_t ROUTED_SCALING_FACTOR_ATTR_INDEX = 7;
const static int64_t EPS_ATTR_INDEX = 8;
const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216; // 预留16M空间
const static uint32_t DATATYPESIZE_FLOAT = 4;
const static bool IS_LARGEST = true;
const static bool IS_INITINDEX = false;
const static bool IS_REUSESOURCE = false;
const static uint64_t WITH_GROUP_CONDITION = 1;
const static uint64_t WITHOUT_GROUP_CONDITION = 2;
const static uint64_t MAX_IN_GROUP_CONDITION = 3;
constexpr int32_t ROW_COUNT_PER_TASK = 1;

const static uint64_t TILING_KEY_EXPERTNUM_GROUPNUM_ALIGN_HIGH_PERF = 0;
const static uint64_t TILING_KEY_WITHOUT_GROUP = 1;
const static uint64_t TILING_KEY_GENERALIZED = 2;
const static uint64_t TILING_KEY_HASH_WITHOUT_GROUP_INT32_INT64 = 3;
const static uint64_t TILING_KEY_HASH_WITHOUT_GROUP_INT32_INT32 = 4;
const static uint64_t TILING_KEY_HASH_WITHOUT_GROUP_INT64_INT64 = 5;
const static uint64_t TILING_KEY_HASH_WITHOUT_GROUP_INT64_INT32 = 6;

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(4))); // 4 for four
}

class MoeGatingTopKTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeGatingTopKTilingBase(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeGatingTopKTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    void SplitRows();
    void CalTmpBufUbSize();

    const gert::Shape *xShape_ = nullptr;
    const gert::Shape *biasShape_ = nullptr;
    const gert::Shape *inputIdsShape_ = nullptr;
    const gert::Shape *tid2eidShape_ = nullptr;
    const gert::Shape *yShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outShape_ = nullptr;
    ge::DataType inputIdsDtype_;
    ge::DataType tid2eidDtype_;

    int64_t rows_ = 0;
    int64_t expertCount_ = 0;
    int64_t addBias_ = 0;
    int64_t hashFlag_ = 0;

    int64_t k_ = 0;
    int64_t kGroup_ = 0;
    int64_t groupCount_ = 0;
    int64_t perGroupExpertCount_ = 0;
    int64_t groupSelectMode_ = GROUP_SELECT_MODE_MAX;
    int64_t renorm_ = RENORM_NO;
    int64_t normType_ = NORM_TYPE_SOFTMAX;
    int64_t outFlag_ = OUT_FLAG_FALSE;
    float routedScalingFactor_ = 1.0;
    float eps_ = 1e-20f;

    int64_t inputDtypeSize_;
    const char *opName_ = "";
    MoeGatingTopKTilingData moeGatingTopKTilingData_;
};

ge::graphStatus MoeGatingTopKTilingBase::CheckInputShape()
{
    size_t xDimNum = xShape_->GetDimNum();
    OP_CHECK_IF(xDimNum != X_INPUT_DIMS,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(xDimNum),
                                             std::to_string(X_INPUT_DIMS)),
                return ge::GRAPH_FAILED);

    // 通过输入获取rows 和 expertCount
    rows_ = xShape_->GetDim(0);
    expertCount_ = xShape_->GetDim(1);
    moeGatingTopKTilingData_.set_rowCount(rows_);
    moeGatingTopKTilingData_.set_expertCount(expertCount_);
    if (biasShape_ != nullptr) {
        addBias_ = 1;
        size_t biasDimNum = biasShape_->GetDimNum();
        OP_CHECK_IF(biasDimNum != BIAS_INPUT_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "bias", std::to_string(biasDimNum),
                                                 std::to_string(BIAS_INPUT_DIMS)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(biasShape_->GetDim(0) != expertCount_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "bias dim[0]",
                                              std::to_string(biasShape_->GetDim(0)), std::to_string(expertCount_)),
                    return ge::GRAPH_FAILED);
    }
    moeGatingTopKTilingData_.set_addBias(addBias_);

    if (inputIdsShape_ != nullptr && tid2eidShape_ != nullptr) {
        hashFlag_ = 1;
    }
    moeGatingTopKTilingData_.set_hashFlag(hashFlag_);

    if (hashFlag_ == 1) {
        OP_CHECK_IF(k_ > MAX_K_FOR_HASH,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_),
                                              "less than or equal to 64 for hash mode"),
                    return ge::GRAPH_FAILED);
        if (tid2eidShape_->GetDim(1) != k_) {
            std::string reasonMsg = "The 1st axis of input tid2eid must be equal to the value of attribute k, "
                                    "which is " +
                                    std::to_string(k_);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "tid2eid",
                                                  Ops::Base::ToString(*tid2eidShape_), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        if (inputIdsShape_->GetDim(0) != rows_) {
            std::string reasonMsg = "The 0th axis of input input_ids must be equal to the 0th axis of input x " +
                                    Ops::Base::ToString(*xShape_);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input_ids",
                                                  Ops::Base::ToString(*inputIdsShape_), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    OP_CHECK_IF(
        k_ > expertCount_,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_), "less than or equal to expert num"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingBase::CheckAttr()
{
    OP_CHECK_IF(expertCount_ > MAX_EXPERT_COUNT,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_count", std::to_string(expertCount_),
                                          ("less than or equal to " + std::to_string(MAX_EXPERT_COUNT))),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(k_ <= 0, OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_), "greater than 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        kGroup_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k_group", std::to_string(kGroup_), "greater than 0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(kGroup_ > groupCount_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k_group", std::to_string(kGroup_),
                                          "less than or equal to group_count"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(groupCount_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_count", std::to_string(groupCount_),
                                          "greater than 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(normType_ != NORM_TYPE_SOFTMAX && normType_ != NORM_TYPE_SIGMOID,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "norm_type", std::to_string(normType_), "0 or 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(groupSelectMode_ != GROUP_SELECT_MODE_SUM && groupSelectMode_ != GROUP_SELECT_MODE_MAX,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_select_mode",
                                          std::to_string(groupSelectMode_), "0 or 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(renorm_ != RENORM_NO && renorm_ != RENORM_L1,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "renorm", std::to_string(renorm_), "0 or 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        expertCount_ % groupCount_ != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "expert_count", std::to_string(expertCount_),
                                              "expert_count must be divisible by group_count"),
        return ge::GRAPH_FAILED);
    perGroupExpertCount_ = expertCount_ / groupCount_;

    bool isSimplifiedPath = (kGroup_ == groupCount_ || groupCount_ == expertCount_);
    OP_CHECK_IF(hashFlag_ == 1 && !isSimplifiedPath,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "hash mode", "hash mode only supports simplified path",
                    "k_group must equal group_count or group_count must equal expert_count"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(perGroupExpertCount_ < 1,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_expert_count",
                                          std::to_string(perGroupExpertCount_), "greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupSelectMode_ == GROUP_SELECT_MODE_SUM && perGroupExpertCount_ < 2,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "group_expert_count",
                                                      std::to_string(perGroupExpertCount_),
                                                      "group expert count should be greater than 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(k_ > kGroup_ * perGroupExpertCount_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_),
                                          "less than or equal to k_group * group_expert_count"),
                return ge::GRAPH_FAILED);
    int64_t groupExpertCountAlign = Ops::Base::CeilAlign(perGroupExpertCount_, 32L);
    if (groupCount_ != 1 && groupCount_ != expertCount_ && kGroup_ != groupCount_) {
        // 分组场景下才需要校验对齐后的数量
        OP_CHECK_IF(groupCount_ * groupExpertCountAlign > MAX_EXPERT_COUNT,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_count * group_expert_count_align",
                                              std::to_string(groupCount_ * groupExpertCountAlign),
                                              ("less than or equal to " + std::to_string(MAX_EXPERT_COUNT))),
                    return ge::GRAPH_FAILED);
    }

    moeGatingTopKTilingData_.set_perGroupExpertCount(perGroupExpertCount_);
    moeGatingTopKTilingData_.set_perGroupExpertCountAlign(groupExpertCountAlign);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    // 获取输入shape信息
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    xShape_ = &xShapePtr->GetStorageShape();

    auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INPUT_INDEX);
    biasShape_ = biasShapePtr == nullptr ? nullptr : &biasShapePtr->GetStorageShape();

    auto inputIdsShapePtr = context_->GetOptionalInputShape(INPUT_IDS_INPUT_INDEX);
    inputIdsShape_ = inputIdsShapePtr == nullptr ? nullptr : &inputIdsShapePtr->GetStorageShape();
    auto tid2eidShapePtr = context_->GetOptionalInputShape(TID_TO_EID_INPUT_INDEX);
    tid2eidShape_ = tid2eidShapePtr == nullptr ? nullptr : &tid2eidShapePtr->GetStorageShape();

    // 获取输出shape
    auto yShapePtr = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    yShape_ = &yShapePtr->GetStorageShape();
    auto expertIdxPtr = context_->GetOutputShape(EXPERT_IDX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxPtr);
    expertIdxShape_ = &expertIdxPtr->GetStorageShape();
    auto outPtr = context_->GetOutputShape(OUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outPtr);
    outShape_ = &outPtr->GetStorageShape();

    auto x = context_->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    auto xDtype = x->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                                  "float32, half, bf16"),
        return ge::GRAPH_FAILED);

    if (biasShapePtr != nullptr) {
        auto biasDtype = context_->GetOptionalInputDesc(BIAS_INPUT_INDEX)->GetDataType();
        OP_CHECK_IF((biasDtype != xDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "bias", Ops::Base::ToString(biasDtype).c_str(),
                                              Ops::Base::ToString(xDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (inputIdsShapePtr != nullptr) {
        inputIdsDtype_ = context_->GetOptionalInputDesc(INPUT_IDS_INPUT_INDEX)->GetDataType();
        OP_CHECK_IF((inputIdsDtype_ != ge::DataType::DT_INT32 && inputIdsDtype_ != ge::DataType::DT_INT64),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "input_ids",
                                              Ops::Base::ToString(inputIdsDtype_).c_str(), "int32 or int64"),
                    return ge::GRAPH_FAILED);
    }
    if (tid2eidShapePtr != nullptr) {
        tid2eidDtype_ = context_->GetOptionalInputDesc(TID_TO_EID_INPUT_INDEX)->GetDataType();
        OP_CHECK_IF((tid2eidDtype_ != ge::DataType::DT_INT32 && tid2eidDtype_ != ge::DataType::DT_INT64),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "tid2eid",
                                              Ops::Base::ToString(tid2eidDtype_).c_str(), "int32 or int64"),
                    return ge::GRAPH_FAILED);
    }

    auto yDesc = context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF((yDtype != xDtype),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", Ops::Base::ToString(yDtype).c_str(),
                                          Ops::Base::ToString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto expertIdDesc = context_->GetOutputDesc(EXPERT_IDX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdDesc);
    auto expertIdDtype = expertIdDesc->GetDataType();
    OP_CHECK_IF((expertIdDtype != ge::DataType::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expert_idx",
                                          Ops::Base::ToString(expertIdDtype).c_str(), "int32"),
                return ge::GRAPH_FAILED);

    auto normOutDesc = context_->GetOutputDesc(OUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, normOutDesc);
    auto normOutDtype = normOutDesc->GetDataType();
    OP_CHECK_IF(
        (normOutDtype != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "out", Ops::Base::ToString(normOutDtype).c_str(), "float"),
        return ge::GRAPH_FAILED);

    // 获取属性
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *kPtr = attrs->GetAttrPointer<int64_t>(K_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kPtr);
    k_ = *kPtr;
    moeGatingTopKTilingData_.set_k(k_);
    OP_LOGI(context_, "Attr k is: %ld ", k_);

    const int64_t *kGroupPtr = attrs->GetAttrPointer<int64_t>(K_GROUP_ATTR_INDEX);
    if (kGroupPtr != nullptr) {
        kGroup_ = *kGroupPtr;
        moeGatingTopKTilingData_.set_kGroup(kGroup_);
    }
    OP_LOGI(context_, "Attr k_group is: %ld ", kGroup_);

    const int64_t *groupCountPtr = attrs->GetAttrPointer<int64_t>(GROUP_COUNT_ATTR_INDEX);
    if (groupCountPtr != nullptr) {
        groupCount_ = *groupCountPtr;
        moeGatingTopKTilingData_.set_groupCount(groupCount_);
    }
    OP_LOGI(context_, "Attr group_count is: %ld ", groupCount_);

    const int64_t *groupSelectModePtr = attrs->GetAttrPointer<int64_t>(GROUP_SELECT_MODE_ATTR_INDEX);
    if (groupSelectModePtr != nullptr) {
        groupSelectMode_ = *groupSelectModePtr;
        moeGatingTopKTilingData_.set_groupSelectMode(groupSelectMode_);
    }
    OP_LOGI(context_, "Attr group_select_mode is: %ld ", groupSelectMode_);

    const int64_t *renormPtr = attrs->GetAttrPointer<int64_t>(RENORM_ATTR_INDEX);
    if (renormPtr != nullptr) {
        renorm_ = *renormPtr;
        moeGatingTopKTilingData_.set_renorm(renorm_);
    }
    OP_LOGI(context_, "Attr renorm is: %ld ", renorm_);

    const int64_t *normTypePtr = attrs->GetAttrPointer<int64_t>(NORM_TYPE_ATTR_INDEX);
    if (normTypePtr != nullptr) {
        normType_ = *normTypePtr;
        moeGatingTopKTilingData_.set_normType(normType_);
    }
    OP_LOGI(context_, "Attr norm_type is: %ld ", normType_);

    const bool *outFlagPtr = attrs->GetAttrPointer<bool>(OUT_FLAG_ATTR_INDEX);
    if (outFlagPtr != nullptr) {
        outFlag_ = (*outFlagPtr) ? 1 : 0;
        moeGatingTopKTilingData_.set_outFlag(outFlag_);
    }
    OP_LOGI(context_, "Attr out_flag is: %ld ", outFlag_);

    const float *routedScalingFactorPtr = attrs->GetAttrPointer<float>(ROUTED_SCALING_FACTOR_ATTR_INDEX);
    if (routedScalingFactorPtr != nullptr) {
        routedScalingFactor_ = *routedScalingFactorPtr;
        moeGatingTopKTilingData_.set_routedScalingFactor(routedScalingFactor_);
    }
    OP_LOGI(context_, "Attr routed_scaling_factor is: %f ", routedScalingFactor_);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
        moeGatingTopKTilingData_.set_eps(eps_);
    }
    OP_LOGI(context_, "Attr eps is: %f ", eps_);

    inputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platform_info"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingBase::CheckOutShape()
{
    OP_CHECK_IF((yShape_->GetDimNum() != xShape_->GetDimNum()),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y", std::to_string(yShape_->GetDimNum()),
                                             std::to_string(xShape_->GetDimNum())),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((expertIdxShape_->GetDimNum() != xShape_->GetDimNum()),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expert_idx",
                                             std::to_string(expertIdxShape_->GetDimNum()),
                                             std::to_string(xShape_->GetDimNum())),
                return ge::GRAPH_FAILED);
    if (outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDimNum() != xShape_->GetDimNum()),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "out", std::to_string(outShape_->GetDimNum()),
                                                 std::to_string(xShape_->GetDimNum())),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((yShape_->GetDim(0) != xShape_->GetDim(0)),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "y dim[0]", std::to_string(yShape_->GetDim(0)),
                                          std::to_string(xShape_->GetDim(0))),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (expertIdxShape_->GetDim(0) != xShape_->GetDim(0)),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_idx dim[0]",
                                  std::to_string(expertIdxShape_->GetDim(0)), std::to_string(xShape_->GetDim(0))),
        return ge::GRAPH_FAILED);
    if (outFlag_ && outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDim(0) != xShape_->GetDim(0)),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "out dim[0]",
                                              std::to_string(outShape_->GetDim(0)), std::to_string(xShape_->GetDim(0))),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((yShape_->GetDim(1) != k_),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "y dim[1]", std::to_string(yShape_->GetDim(1)),
                                          std::to_string(k_)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((expertIdxShape_->GetDim(1) != k_),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_idx dim[1]",
                                          std::to_string(expertIdxShape_->GetDim(1)), std::to_string(k_)),
                return ge::GRAPH_FAILED);
    if (outFlag_ && outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDim(1) != xShape_->GetDim(1)),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "out dim[1]",
                                              std::to_string(outShape_->GetDim(1)), std::to_string(xShape_->GetDim(1))),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKTilingBase::SplitRows()
{
    int64_t perCoreRows = Ops::Base::CeilDiv(rows_, static_cast<int64_t>(aicoreParams_.numBlocks));
    int64_t needCoreNum = Ops::Base::CeilDiv(rows_, perCoreRows);
    // perCoreRows cannot be 0
    int64_t lastCoreRows = rows_ % perCoreRows == 0 ? perCoreRows : rows_ % perCoreRows;
    moeGatingTopKTilingData_.set_needCoreNum(needCoreNum);
    moeGatingTopKTilingData_.set_perCoreRowCount(perCoreRows);
    moeGatingTopKTilingData_.set_lastCoreRowCount(lastCoreRows);
    int64_t vmsCount = CeilLog4(Ops::Base::CeilDiv(kGroup_, 4L));
    OP_LOGI(context_, "vms count is: %ld", vmsCount);
    moeGatingTopKTilingData_.set_vmsCount(vmsCount); // 需要归并的轮数
}

void MoeGatingTopKTilingBase::CalTmpBufUbSize()
{
    std::vector<int64_t> shape_vec = {expertCount_};
    ge::Shape shape(shape_vec);
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSigmoidMaxMinTmpSize(shape, sizeof(float), false, maxValue, minValue);

    int64_t indexTmpBuf = (expertCount_ + 31) / 32 * 32 * static_cast<int64_t>(sizeof(float));
    moeGatingTopKTilingData_.set_calTmpBufUbSize(std::max(indexTmpBuf, static_cast<int64_t>(minValue)));
}

ge::graphStatus MoeGatingTopKTilingBase::DoOpTiling()
{
    auto ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    CalTmpBufUbSize();
    SplitRows();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingBase::PostTiling()
{
    context_->SetBlockDim(moeGatingTopKTilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    moeGatingTopKTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeGatingTopKTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKTilingBase::GetTilingKey() const
{
    // DeepSeekV3排序对齐高性能场景
    if (expertCount_ == 256 && groupCount_ == 8 && kGroup_ == 4 && k_ <= 32 && addBias_ &&
        groupSelectMode_ == GROUP_SELECT_MODE_SUM && renorm_ == RENORM_NO && normType_ == NORM_TYPE_SIGMOID &&
        !outFlag_) {
        // DeepSeekV3排序对齐高性能场景
        return TILING_KEY_EXPERTNUM_GROUPNUM_ALIGN_HIGH_PERF;
    } else if (groupCount_ == 1 || groupCount_ == expertCount_ || kGroup_ == groupCount_) {
        /**
         * 不分组场景：
         * 1. 分组数为 1
         * 2. 分组数等于专家数（每个组只有一个专家）
         * 3. 选择所有组
         */
        if (hashFlag_ == 0) {
            return TILING_KEY_WITHOUT_GROUP;
        } else if (inputIdsDtype_ == ge::DataType::DT_INT32 && tid2eidDtype_ == ge::DataType::DT_INT64) {
            return TILING_KEY_HASH_WITHOUT_GROUP_INT32_INT64;
        } else if (inputIdsDtype_ == ge::DataType::DT_INT32 && tid2eidDtype_ == ge::DataType::DT_INT32) {
            return TILING_KEY_HASH_WITHOUT_GROUP_INT32_INT32;
        } else if (inputIdsDtype_ == ge::DataType::DT_INT64 && tid2eidDtype_ == ge::DataType::DT_INT64) {
            return TILING_KEY_HASH_WITHOUT_GROUP_INT64_INT64;
        } else {
            return TILING_KEY_HASH_WITHOUT_GROUP_INT64_INT32;
        }
    } else {
        return TILING_KEY_GENERALIZED;
    }
}

void MoeGatingTopKTilingBase::Reset()
{
    opName_ = nullptr;
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopK, MoeGatingTopKTilingBase, 2000);
} // namespace optiling
