/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_tiling_ar35.cpp
 * \brief
 */

#include "log/log.h"
#include "moe_gating_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
const static uint64_t MOE_GATING_TOP_K_REGBASE_TILING_KEY = 10000;

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

const static int64_t X_INPUT_INDEX = 0;
const static int64_t BIAS_INPUT_INDEX = 1;
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


class MoeGatingTopKTilingRegbase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeGatingTopKTilingRegbase(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
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
        if (socVersion != platform_ascendc::SocVersion::ASCEND910_95) {
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
    ge::graphStatus CheckOutShape();
    void SplitRows();
    void Tiling4GatherOutComputeSplitK();

    const gert::Shape *xShape_ = nullptr;
    const gert::Shape *biasShape_ = nullptr;
    const gert::Shape *yShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outShape_ = nullptr;

    int64_t rows_;
    int64_t expertCount_;
    int64_t addBias_ = 0;

    int64_t k_;
    int64_t kGroup_ = 1;
    int64_t groupCount_ = 1;
    int64_t groupSelectMode_ = GROUP_SELECT_MODE_MAX;
    int64_t renorm_ = RENORM_NO;
    int64_t normType_ = NORM_TYPE_SOFTMAX;
    int64_t outFlag_ = OUT_FLAG_FALSE;
    float routedScalingFactor_ = 1.0;
    float eps_ = 1e-20;

    int64_t inputDtypeSize_;
    const char *opName_ = "";
    MoeGatingTopKRegbaseTilingData moeGatingTopKTilingData_;
    platform_ascendc::SocVersion socVersion;
};

ge::graphStatus MoeGatingTopKTilingRegbase::CheckInputShape()
{
    size_t xDimNnum = xShape_->GetDimNum();
    OP_CHECK_IF(xDimNnum != X_INPUT_DIMS,
                OP_LOGE(context_, "The number of x dim is: %zu, but should be %zu.", xDimNnum, X_INPUT_DIMS),
                return ge::GRAPH_FAILED);

    // 通过输入获取rows 和 expertCount
    rows_ = xShape_->GetDim(0);
    expertCount_ = xShape_->GetDim(1);
    moeGatingTopKTilingData_.set_rowCount(rows_);
    moeGatingTopKTilingData_.set_expertCount(expertCount_);
    OP_CHECK_IF(
        expertCount_ > MAX_EXPERT_COUNT,
        OP_LOGE(context_, "expert count is: %ld, but should not greater than %ld.", expertCount_, MAX_EXPERT_COUNT),
        return ge::GRAPH_FAILED);

    if (biasShape_ != nullptr) {
        addBias_ = 1;
        size_t biasDimNnum = biasShape_->GetDimNum();
        OP_CHECK_IF(biasDimNnum != BIAS_INPUT_DIMS,
                    OP_LOGE(context_, "The number of bias dim is: %zu, but should be %zu.", biasDimNnum, BIAS_INPUT_DIMS),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(biasShape_->GetDim(0) != expertCount_,
                    OP_LOGE(context_, "The first dim of bias is: %zu, but should be expert num: %ld.",
                         biasShape_->GetDim(0), expertCount_),
                    return ge::GRAPH_FAILED);
    }
    moeGatingTopKTilingData_.set_addBias(addBias_);

    OP_CHECK_IF(k_ > expertCount_,
                OP_LOGE(context_, "k is: %ld, expert num is: %ld, k cannot be greater than expert num.", k_, expertCount_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckAttr()
{
    OP_CHECK_IF(k_ <= 0, OP_LOGE(context_, "k is: %ld, but should be greater than 0.", k_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kGroup_ <= 0, OP_LOGE(context_, "k_group is: %ld, but should be greater than 0.", kGroup_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupCount_ <= 0, OP_LOGE(context_, "group_count is: %ld, but should be greater than 0.", groupCount_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertCount_ % groupCount_ != 0,
                OP_LOGE(context_, "expert num : %ld is not divisible by group_count: %ld", expertCount_, groupCount_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kGroup_ > groupCount_,
                OP_LOGE(context_, "k_group is: %ld, but should not greater than group_count: %ld", kGroup_, groupCount_),
                return ge::GRAPH_FAILED);

    int64_t groupExpertCount = expertCount_ / groupCount_;
    int64_t groupExpertCountAlign = Ops::Base::CeilAlign(groupExpertCount, 32l);
    moeGatingTopKTilingData_.set_perGroupExpertCount(expertCount_ / groupCount_);
    moeGatingTopKTilingData_.set_perGroupExpertCountAlign(groupExpertCountAlign);
    OP_CHECK_IF(groupCount_ * groupExpertCountAlign > MAX_EXPERT_COUNT,
                OP_LOGE(context_, "group count * group expert count align is: %ld, but should not greater than %zu.",
                     groupCount_ * groupExpertCountAlign, MAX_EXPERT_COUNT),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kGroup_ * groupExpertCount < k_,
                OP_LOGE(context_, "k_group * group expert count is: %ld, but it must be greater than or equal to k: %ld.",
                     kGroup_ * groupExpertCount, k_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(groupExpertCount < 2,
                OP_LOGE(context_, "per group expert count is: %ld, but should not less than 2.", groupExpertCount),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupSelectMode_ != GROUP_SELECT_MODE_SUM,
                OP_LOGE(context_, "group_select_mode is: %ld, but currently only supports %ld.", groupSelectMode_,
                     GROUP_SELECT_MODE_SUM),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(renorm_ != RENORM_NO,
                OP_LOGE(context_, "renorm is: %ld, but currently only support %ld.", renorm_, RENORM_NO),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(normType_ != NORM_TYPE_SIGMOID,
                OP_LOGE(context_, "norm_type is: %ld, but currently only support %ld.", normType_, NORM_TYPE_SIGMOID),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(outFlag_ != 0, OP_LOGE(context_, "out_flag is: True, but currently only support False."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    // 获取输入shape信息
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    xShape_ = &xShapePtr->GetStorageShape();
    auto biasShapePtr = context_->GetInputShape(BIAS_INPUT_INDEX);
    biasShape_ = biasShapePtr == nullptr ? nullptr : &biasShapePtr->GetStorageShape();

    // 获取输出shape
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

    auto x = context_->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    auto xDtype = x->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16),
        OP_LOGE(context_, "x dtype %s error, only supports float32, half, bf16. please check.",
             ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto bias = context_->GetOptionalInputShape(BIAS_INPUT_INDEX);
    if (bias != nullptr) {
        auto biasDtype = context_->GetOptionalInputDesc(BIAS_INPUT_INDEX)->GetDataType();
        OP_CHECK_IF((biasDtype != xDtype),
                    OP_LOGE(context_, "bias dtype %s not equal x dtype %s, please check.",
                         ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                         ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    auto yDesc = context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF((yDtype != xDtype),
                OP_LOGE(context_, "y out dtype %s must be the same with x dtype %s.",
                     ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                     ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto expertIdDesc = context_->GetOutputDesc(EXPERT_IDX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdDesc);
    auto expertIdDtype = expertIdDesc->GetDataType();
    OP_CHECK_IF((expertIdDtype != ge::DataType::DT_INT32),
                OP_LOGE(context_, "expertId out dtype %s error, only supports int32. please check.",
                     ge::TypeUtils::DataTypeToSerialString(expertIdDtype).c_str()),
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
    }
    moeGatingTopKTilingData_.set_kGroup(kGroup_);
    OP_LOGI(context_, "Attr k_group is: %ld ", kGroup_);

    const int64_t *groupCountPtr = attrs->GetAttrPointer<int64_t>(GROUP_COUNT_ATTR_INDEX);
    if (groupCountPtr != nullptr) {
        groupCount_ = *groupCountPtr;
    }
    moeGatingTopKTilingData_.set_groupCount(groupCount_);
    OP_LOGI(context_, "Attr group_count is: %ld ", groupCount_);

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

    auto outDesc = context_->GetOutputDesc(OUT_OUTPUT_INDEX);
    if (outFlag_ && outDesc != nullptr) {
        auto outDtype = outDesc->GetDataType();
        OP_CHECK_IF((outDtype != ge::DataType::DT_FLOAT),
                    OP_LOGE(context_, "norm out dtype %s error, only supports float32. please check.",
                         ge::TypeUtils::DataTypeToSerialString(outDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    inputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
    socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKTilingRegbase::CheckOutShape()
{
    OP_CHECK_IF((yShape_->GetDimNum() != xShape_->GetDimNum()),
                OP_LOGE(context_, "y out shape num %ld and x shape num %ld not equal, please check.", yShape_->GetDimNum(),
                     xShape_->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((expertIdxShape_->GetDimNum() != xShape_->GetDimNum()),
                OP_LOGE(context_, "expertId out shape num %ld and x shape num %ld not equal, please check.",
                     expertIdxShape_->GetDimNum(), xShape_->GetDimNum()),
                return ge::GRAPH_FAILED);
    if (outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDimNum() != xShape_->GetDimNum()),
                    OP_LOGE(context_, "norm out shape num %ld and x shape num %ld not equal, please check.",
                         outShape_->GetDimNum(), xShape_->GetDimNum()),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((yShape_->GetDim(0) != xShape_->GetDim(0)),
                OP_LOGE(context_, "y out dim[0] %ld not euqal x dim[0] %ld, please check.", yShape_->GetDim(0),
                     xShape_->GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((expertIdxShape_->GetDim(0) != xShape_->GetDim(0)),
                OP_LOGE(context_, "expertId out dim[0] %ld not euqal x dim[0] %ld, please check.",
                     expertIdxShape_->GetDim(0), xShape_->GetDim(0)),
                return ge::GRAPH_FAILED);
    if (outFlag_ && outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDim(0) != xShape_->GetDim(0)),
                    OP_LOGE(context_, "norm out dim[0] %ld and x dim[0] %ld not equal, please check.",
                         outShape_->GetDim(0), outShape_->GetDim(0)),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((yShape_->GetDim(1) != k_),
                OP_LOGE(context_, "y dim[1] %ld not euqal k %ld, please check.", yShape_->GetDim(1), k_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((expertIdxShape_->GetDim(1) != k_),
                OP_LOGE(context_, "expertId dim[1] %ld not euqal k %ld, please check.", expertIdxShape_->GetDim(1), k_),
                return ge::GRAPH_FAILED);
    if (outFlag_ && outShape_ != nullptr) {
        OP_CHECK_IF((outShape_->GetDim(1) != xShape_->GetDim(1)),
                    OP_LOGE(context_, "normOut dim[1] %ld and x dim[1] %ld not equal, please check.", outShape_->GetDim(1),
                         xShape_->GetDim(1)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKTilingRegbase::SplitRows()
{
    int64_t perCoreRows = Ops::Base::CeilDiv(rows_, static_cast<int64_t>(aicoreParams_.blockDim));
    int64_t needCoreNum = Ops::Base::CeilDiv(rows_, perCoreRows);
    if (perCoreRows == 0) {
        OP_LOGE(context_, "perCoreRows can't be 0.");
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
    return MOE_GATING_TOP_K_REGBASE_TILING_KEY;
}

void MoeGatingTopKTilingRegbase::Reset()
{
    opName_ = nullptr;
    return;
}

REGISTER_TILING_TEMPLATE("MoeGatingTopK", MoeGatingTopKTilingRegbase, 1000);
} // namespace optiling
