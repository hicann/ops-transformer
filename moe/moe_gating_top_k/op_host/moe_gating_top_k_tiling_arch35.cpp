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
 * \file moe_gating_top_k_tiling_arch35.cpp
 * \brief
 */

#include "log/log.h"
#include "moe_gating_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

#include <string>

namespace optiling {
const static uint64_t MOE_GATING_TOP_K_REGBASE_TILING_KEY = 10000;
const static uint64_t MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_TILING_KEY = 10005;
const static uint64_t MOE_GATING_TOP_K_E_K_FULLLOAD_REGBASE_TILING_KEY = 10010;
const static int64_t TILINGKEY_INT32_INT64_SUFFIX = 1;
const static int64_t TILINGKEY_INT32_INT32_SUFFIX = 2;
const static int64_t TILINGKEY_INT64_INT64_SUFFIX = 3;
const static int64_t TILINGKEY_INT64_INT32_SUFFIX = 4;

const static int64_t GROUP_SELECT_MODE_MAX = 0;
const static int64_t GROUP_SELECT_MODE_SUM = 1;
const static int64_t RENORM_NO = 0;
const static int64_t RENORM_L1 = 1;
const static int64_t NORM_TYPE_SOFTMAX = 0;
const static int64_t NORM_TYPE_SIGMOID = 1;
const static int64_t NORM_TYPE_SOFTPLUS = 2;
const static int64_t OUT_FLAG_FALSE = 0;
const static int64_t OUT_FLAG_TRUE = 1;
const static int64_t EK_FULLLOAD_EXPERT_COUNT = 256;
const static int64_t EK_FULLLOAD_GROUP_COUNT = 8;
const static int64_t EK_FULLLOAD_K_GROUP = 4;
const static int64_t EK_FULLLOAD_MAX_K = 32;
const static int64_t WITHOUT_GROUP_K = 1;
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
const static int64_t MRGSORT_SIZE = 4;
const static int64_t NORM_TYPE_ATTR_INDEX = 5;
const static int64_t OUT_FLAG_ATTR_INDEX = 6;
const static int64_t ROUTED_SCALING_FACTOR_ATTR_INDEX = 7;
const static int64_t EPS_ATTR_INDEX = 8;
const static int64_t DEFAULT_WORKSPACE_SIZE = static_cast<int64_t>(16 * 1024 * 1024); // 预留16M空间
const static int64_t DEFAULT_BATCH_ROWS = 4;
const static int64_t MAX_BATCH_ROWS = 128;
const static int64_t MAX_SINGLE_EXPERT_BATCH_ROWS = 4096;

class MoeGatingTopKTilingRegbase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeGatingTopKTilingRegbase(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeGatingTopKTilingRegbase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
            return false;
        }
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
    ge::graphStatus CheckAttrBasic();
    ge::graphStatus CheckAttrExpert();
    ge::graphStatus CheckAttrMode();
    ge::graphStatus CheckOutShape();
    ge::graphStatus GetInputOutputShape();
    ge::graphStatus CheckDtypes();
    ge::graphStatus GetAttrs();
    ge::graphStatus GetBasicAttrs();
    ge::graphStatus GetModeAttrs();
    ge::graphStatus GetOtherAttrs();
    ge::graphStatus CheckInputShape4Hash();
    ge::graphStatus CheckDtype4Hash();
    void CalTmpBufUbSize();
    void SplitRows();
    void Tiling4GatherOutComputeSplitK();

    const gert::Shape *xShape_ = nullptr;
    const gert::Shape *biasShape_ = nullptr;
    const gert::Shape *inputIdsShape_ = nullptr;
    const gert::Shape *tid2eidShape_ = nullptr;
    const gert::Shape *yShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outShape_ = nullptr;

    ge::DataType inputIdsDtype_;
    ge::DataType tid2eidDtype_;

    int64_t rows_;
    int64_t expertCount_;
    int64_t addBias_ = 0;
    bool hashFlag_ = false;

    int64_t k_;
    int64_t kGroup_ = 1;
    int64_t groupCount_ = 1;
    int64_t groupSelectMode_ = GROUP_SELECT_MODE_MAX;
    int64_t renorm_ = RENORM_NO;
    int64_t normType_ = NORM_TYPE_SOFTMAX;
    int64_t outFlag_ = OUT_FLAG_FALSE;
    float routedScalingFactor_ = 1.0;
    float eps_ = 1e-20f;

    int64_t inputDtypeSize_;
    const char *opName_ = "";
    MoeGatingTopKRegbaseTilingData moeGatingTopKTilingData_;
    platform_ascendc::SocVersion socVersion;
};

ge::graphStatus MoeGatingTopKTilingRegbase::CheckInputShape4Hash()
{
    OP_CHECK_IF(k_ > MAX_K_FOR_HASH,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_),
                                          "less than or equal to 64 for hash mode"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tid2eidShape_->GetDim(1) != k_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "tid2eid dim[1]",
                                          std::to_string(tid2eidShape_->GetDim(1)), std::to_string(k_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputIdsShape_->GetDim(0) != rows_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "inputIds dim[0]",
                                          std::to_string(inputIdsShape_->GetDim(0)), std::to_string(rows_)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckInputShape()
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
    OP_CHECK_IF(expertCount_ > MAX_EXPERT_COUNT,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_count", std::to_string(expertCount_),
                                          ("less than or equal to " + std::to_string(MAX_EXPERT_COUNT))),
                return ge::GRAPH_FAILED);

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

    OP_CHECK_IF(
        k_ > expertCount_,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_), "less than or equal to expert num"),
        return ge::GRAPH_FAILED);

    if (hashFlag_ && CheckInputShape4Hash() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    moeGatingTopKTilingData_.set_hashFlag(hashFlag_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckAttrBasic()
{
    OP_CHECK_IF(k_ <= 0, OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_), "greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        kGroup_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k_group", std::to_string(kGroup_), "greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupCount_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_count", std::to_string(groupCount_),
                                          "greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expertCount_ % groupCount_ != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "expert_count", std::to_string(expertCount_),
                                              "expert_count must be divisible by group_count"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(kGroup_ > groupCount_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k_group", std::to_string(kGroup_),
                                          "less than or equal to group_count"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupCount_ == expertCount_ && kGroup_ < k_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k_group * group_expert_count",
                                          std::to_string(kGroup_), "greater than or equal to k"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckAttrExpert()
{
    bool isSimplifiedPath = (kGroup_ == groupCount_ || groupCount_ == expertCount_);
    OP_CHECK_IF(hashFlag_ && !isSimplifiedPath,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "hash mode", "hash mode only supports simplified path",
                    "k_group must equal group_count or group_count must equal expert_count"),
                return ge::GRAPH_FAILED);

    if (isSimplifiedPath) {
        kGroup_ = 1;
        groupCount_ = 1;
    }
    moeGatingTopKTilingData_.set_kGroup(kGroup_);
    moeGatingTopKTilingData_.set_groupCount(groupCount_);
    int64_t groupExpertCount = expertCount_ / groupCount_;
    int64_t groupExpertCountAlign = Ops::Base::CeilAlign(groupExpertCount, 32L);
    moeGatingTopKTilingData_.set_perGroupExpertCount(expertCount_ / groupCount_);
    moeGatingTopKTilingData_.set_perGroupExpertCountAlign(groupExpertCountAlign);

    OP_CHECK_IF(groupCount_ * groupExpertCountAlign > MAX_EXPERT_COUNT,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_count * group_expert_count_align",
                                          std::to_string(groupCount_ * groupExpertCountAlign),
                                          ("less than or equal to " + std::to_string(MAX_EXPERT_COUNT))),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kGroup_ * groupExpertCount < k_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k_group * group_expert_count",
                                          std::to_string(kGroup_ * groupExpertCount), "greater than or equal to k"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(groupExpertCount < 1,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_expert_count",
                                          std::to_string(groupExpertCount), "greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckAttrMode()
{
    OP_CHECK_IF(groupSelectMode_ != GROUP_SELECT_MODE_SUM && groupSelectMode_ != GROUP_SELECT_MODE_MAX,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_select_mode",
                                          std::to_string(groupSelectMode_), "0 or 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupSelectMode_ == GROUP_SELECT_MODE_SUM && (expertCount_ / groupCount_) < 2,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "group_expert_count",
                                                      std::to_string(expertCount_ / groupCount_),
                                                      "group expert count should be greater than 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(renorm_ != RENORM_NO && renorm_ != RENORM_L1,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "renorm", std::to_string(renorm_), "0 or 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(normType_ != NORM_TYPE_SOFTMAX && normType_ != NORM_TYPE_SIGMOID && normType_ != NORM_TYPE_SOFTPLUS,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "norm_type", std::to_string(normType_), "0, 1 or 2"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckAttr()
{
    OP_CHECK_IF(CheckAttrBasic() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "CheckAttrBasic", "failed",
                                                      "CheckAttrBasic failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttrExpert() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "CheckAttrExpert", "failed",
                                                      "CheckAttrExpert failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttrMode() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "CheckAttrMode", "failed",
                                                      "CheckAttrMode failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetInputOutputShape()
{
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    xShape_ = &xShapePtr->GetStorageShape();
    auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INPUT_INDEX);
    biasShape_ = biasShapePtr == nullptr ? nullptr : &biasShapePtr->GetStorageShape();

    auto inputIdsShapePtr = context_->GetOptionalInputShape(INPUT_IDS_INPUT_INDEX);
    inputIdsShape_ = inputIdsShapePtr == nullptr ? nullptr : &inputIdsShapePtr->GetStorageShape();
    auto tid2eidShapePtr = context_->GetOptionalInputShape(TID_TO_EID_INPUT_INDEX);
    tid2eidShape_ = tid2eidShapePtr == nullptr ? nullptr : &tid2eidShapePtr->GetStorageShape();
    if (inputIdsShape_ != nullptr && tid2eidShape_ != nullptr) {
        hashFlag_ = true;
    }

    auto yShapePtr = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    yShape_ = &yShapePtr->GetStorageShape();
    auto expertIdxPtr = context_->GetOutputShape(EXPERT_IDX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxPtr);
    expertIdxShape_ = &expertIdxPtr->GetStorageShape();
    auto outPtr = context_->GetOutputShape(OUT_OUTPUT_INDEX);
    if (outPtr != nullptr) {
        outShape_ = &outPtr->GetStorageShape();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckDtype4Hash()
{
    auto inputIdsDesc = context_->GetOptionalInputDesc(INPUT_IDS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputIdsDesc);
    inputIdsDtype_ = inputIdsDesc->GetDataType();
    OP_CHECK_IF((inputIdsDtype_ != ge::DataType::DT_INT32 && inputIdsDtype_ != ge::DataType::DT_INT64),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "inputIds",
                                          Ops::Base::ToString(inputIdsDtype_).c_str(), "int32, int64"),
                return ge::GRAPH_FAILED);

    auto tid2eidDesc = context_->GetOptionalInputDesc(TID_TO_EID_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tid2eidDesc);
    tid2eidDtype_ = tid2eidDesc->GetDataType();
    OP_CHECK_IF((tid2eidDtype_ != ge::DataType::DT_INT32 && tid2eidDtype_ != ge::DataType::DT_INT64),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "tid2eid",
                                          Ops::Base::ToString(tid2eidDtype_).c_str(), "int32, int64"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckDtypes()
{
    auto x = context_->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    auto xDtype = x->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                                  "float32, half, bf16"),
        return ge::GRAPH_FAILED);

    if (biasShape_ != nullptr) {
        auto biasDtype = context_->GetOptionalInputDesc(BIAS_INPUT_INDEX)->GetDataType();
        OP_CHECK_IF((biasDtype != xDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "bias", Ops::Base::ToString(biasDtype).c_str(),
                                              Ops::Base::ToString(xDtype).c_str()),
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

    auto outDesc = context_->GetOutputDesc(OUT_OUTPUT_INDEX);
    if (outFlag_ && outDesc != nullptr) {
        auto outDtype = outDesc->GetDataType();
        OP_CHECK_IF((outDtype != ge::DataType::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "outDtype",
                                              Ops::Base::ToString(outDtype).c_str(), "float32"),
                    return ge::GRAPH_FAILED);
    }

    if (hashFlag_ && CheckDtype4Hash() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    inputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetBasicAttrs()
{
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
    }
    OP_LOGI(context_, "Attr k_group is: %ld ", kGroup_);

    const int64_t *groupCountPtr = attrs->GetAttrPointer<int64_t>(GROUP_COUNT_ATTR_INDEX);
    if (groupCountPtr != nullptr) {
        groupCount_ = *groupCountPtr;
    }
    OP_LOGI(context_, "Attr group_count is: %ld ", groupCount_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetModeAttrs()
{
    auto attrs = context_->GetAttrs();

    const int64_t *groupSelectModePtr = attrs->GetAttrPointer<int64_t>(GROUP_SELECT_MODE_ATTR_INDEX);
    if (groupSelectModePtr != nullptr) {
        groupSelectMode_ = *groupSelectModePtr;
    }
    moeGatingTopKTilingData_.set_groupSelectMode(groupSelectMode_);
    OP_LOGI(context_, "Attr group_select_mode is: %ld ", groupSelectMode_);

    const int64_t *renormPtr = attrs->GetAttrPointer<int64_t>(RENORM_ATTR_INDEX);
    if (renormPtr != nullptr) {
        renorm_ = *renormPtr;
    }
    moeGatingTopKTilingData_.set_renorm(renorm_);
    OP_LOGI(context_, "Attr renorm is: %ld ", renorm_);

    const int64_t *normTypePtr = attrs->GetAttrPointer<int64_t>(NORM_TYPE_ATTR_INDEX);
    if (normTypePtr != nullptr) {
        normType_ = *normTypePtr;
    }
    moeGatingTopKTilingData_.set_normType(normType_);
    OP_LOGI(context_, "Attr norm_type is: %ld ", normType_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetOtherAttrs()
{
    auto attrs = context_->GetAttrs();

    const bool *outFlagPtr = attrs->GetAttrPointer<bool>(OUT_FLAG_ATTR_INDEX);
    if (outFlagPtr != nullptr) {
        outFlag_ = (*outFlagPtr) ? 1 : 0;
    }
    moeGatingTopKTilingData_.set_outFlag(outFlag_);
    OP_LOGI(context_, "Attr out_flag is: %ld ", outFlag_);

    const float *routedScalingFactorPtr = attrs->GetAttrPointer<float>(ROUTED_SCALING_FACTOR_ATTR_INDEX);
    if (routedScalingFactorPtr != nullptr) {
        routedScalingFactor_ = *routedScalingFactorPtr;
    }
    moeGatingTopKTilingData_.set_routedScalingFactor(routedScalingFactor_);
    OP_LOGI(context_, "Attr routed_scaling_factor is: %f ", routedScalingFactor_);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
    }
    moeGatingTopKTilingData_.set_eps(eps_);
    OP_LOGI(context_, "Attr eps is: %f ", eps_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetAttrs()
{
    auto ret = GetBasicAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetModeAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetOtherAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();

    auto ret = GetInputOutputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckDtypes();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platform_info"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckOutShape()
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

void MoeGatingTopKTilingRegbase::CalTmpBufUbSize()
{
    std::vector<int64_t> shape_vec = {groupCount_ * moeGatingTopKTilingData_.get_perGroupExpertCountAlign()};
    ge::Shape softmaxShape(shape_vec);

    uint32_t softmaxTmpSize = AscendC::GetSoftMaxMaxTmpSize(softmaxShape, sizeof(float), true);
    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), softmaxTmpSize, moeGatingTopKTilingData_.softmaxTilingData);
}

void MoeGatingTopKTilingRegbase::SplitRows()
{
    int64_t perCoreRows = Ops::Base::CeilDiv(rows_, static_cast<int64_t>(aicoreParams_.numBlocks));
    int64_t needCoreNum = Ops::Base::CeilDiv(rows_, perCoreRows);
    if (perCoreRows == 0) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "perCoreRows", std::to_string(perCoreRows),
                                  "greater than 0");
        return;
    }
    int64_t lastCoreRows = rows_ % perCoreRows == 0 ? perCoreRows : rows_ % perCoreRows;
    moeGatingTopKTilingData_.set_needCoreNum(needCoreNum);
    moeGatingTopKTilingData_.set_perCoreRowCount(perCoreRows);
    moeGatingTopKTilingData_.set_lastCoreRowCount(lastCoreRows);

    int64_t vmsCount = 0;
    if (kGroup_ > MRGSORT_SIZE) {
        int64_t index = MRGSORT_SIZE;
        while (index < kGroup_) {
            index = index * MRGSORT_SIZE;
            vmsCount++;
        }
    }
    moeGatingTopKTilingData_.set_vmsCount(vmsCount);

    int64_t expertCountAlign = Ops::Base::CeilAlign(expertCount_, 32L);
    int64_t batchRows = DEFAULT_BATCH_ROWS;
    if (expertCount_ == 1) {
        // expertCount==1: no alignment, per-row UB = (xIn + yOut + expertIdx + out) * depth(2)
        // all queues use double buffer
        int64_t perRowUbSize = (inputDtypeSize_ + inputDtypeSize_ + sizeof(int32_t) + sizeof(float)) * 2;
        int64_t maxBatchByUb = static_cast<int64_t>(aicoreParams_.ubSize) / perRowUbSize;
        // reserve 20% UB for pipe/event overhead
        maxBatchByUb = maxBatchByUb * 4 / 5;
        batchRows = std::min(std::max(maxBatchByUb, static_cast<int64_t>(1)), MAX_SINGLE_EXPERT_BATCH_ROWS);
    } else {
        bool isSimplifiedPath = (kGroup_ == groupCount_ || groupCount_ == expertCount_);
        if (isSimplifiedPath && k_ == WITHOUT_GROUP_K && !hashFlag_) {
            // without_group path: batchRows scales xIn/yOut/expertIdx, fixed buffers occupy UB too
            // all queues use double buffer
            int64_t dtypeRatio = sizeof(float) / inputDtypeSize_;
            int64_t blockElements = 32 / inputDtypeSize_;
            int64_t kStride = Ops::Base::CeilAlign(k_, blockElements);
            // fixed: outOutQueue + biasBuf + xNormBuf + xNormWithBiasBuf + sortedBuf + calcTmpBuf + indexBuf
            int64_t fixedUb = expertCountAlign * sizeof(float) * (16 + dtypeRatio);
            // per-row: xInQueue + yOutQueue + expertIdxOutQueue (all double buffered)
            int64_t perRowUb =
                (expertCountAlign * sizeof(float) * dtypeRatio + kStride * sizeof(float) + kStride * sizeof(int32_t)) *
                2;
            int64_t availableUb = static_cast<int64_t>(aicoreParams_.ubSize) - fixedUb;
            if (perRowUb > 0 && availableUb > 0) {
                int64_t maxBatchByUb = availableUb / perRowUb;
                batchRows = std::min(std::max(maxBatchByUb, static_cast<int64_t>(1)), MAX_BATCH_ROWS);
            }
        }
    }
    moeGatingTopKTilingData_.set_batchRows(batchRows);
}

ge::graphStatus MoeGatingTopKTilingRegbase::DoOpTiling()
{
    auto ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    CalTmpBufUbSize();
    SplitRows();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::PostTiling()
{
    context_->SetBlockDim(moeGatingTopKTilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    moeGatingTopKTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeGatingTopKTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKTilingRegbase::GetTilingKey() const
{
    if (!hashFlag_) {
        if (expertCount_ == EK_FULLLOAD_EXPERT_COUNT && groupCount_ == EK_FULLLOAD_GROUP_COUNT &&
            kGroup_ == EK_FULLLOAD_K_GROUP && k_ <= EK_FULLLOAD_MAX_K && addBias_ &&
            groupSelectMode_ == GROUP_SELECT_MODE_SUM && renorm_ == RENORM_NO && normType_ == NORM_TYPE_SIGMOID &&
            outFlag_ == OUT_FLAG_FALSE) {
            return MOE_GATING_TOP_K_E_K_FULLLOAD_REGBASE_TILING_KEY;
        }
        bool isSimplifiedPath = (kGroup_ == groupCount_ || groupCount_ == expertCount_);
        if (isSimplifiedPath && k_ == WITHOUT_GROUP_K) {
            return MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_TILING_KEY;
        }
        return MOE_GATING_TOP_K_REGBASE_TILING_KEY;
    }

    bool isInputIdsInt32 = (inputIdsDtype_ == ge::DataType::DT_INT32);
    bool isTid2eidInt32 = (tid2eidDtype_ == ge::DataType::DT_INT32);

    if (isInputIdsInt32 && !isTid2eidInt32) {
        return MOE_GATING_TOP_K_REGBASE_TILING_KEY + TILINGKEY_INT32_INT64_SUFFIX;
    } else if (isInputIdsInt32 && isTid2eidInt32) {
        return MOE_GATING_TOP_K_REGBASE_TILING_KEY + TILINGKEY_INT32_INT32_SUFFIX;
    } else if (!isInputIdsInt32 && !isTid2eidInt32) {
        return MOE_GATING_TOP_K_REGBASE_TILING_KEY + TILINGKEY_INT64_INT64_SUFFIX;
    } else {
        return MOE_GATING_TOP_K_REGBASE_TILING_KEY + TILINGKEY_INT64_INT32_SUFFIX;
    }
}

void MoeGatingTopKTilingRegbase::Reset()
{
    opName_ = nullptr;
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopK, MoeGatingTopKTilingRegbase, 1000);
} // namespace optiling
