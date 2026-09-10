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
 * \file grouped_mat_mul_allto_allv_tiling.cpp
 */

#include "grouped_mat_mul_allto_allv_tiling.h"
#include "op_mc2.h"
#include "mc2_log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_comm_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "graph/utils/type_utils.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace optiling {

const std::vector<uint32_t> GroupedMatmulAllToAllvTiling::GMM_X_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GroupedMatmulAllToAllvTiling::GMM_WEIGHT_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GroupedMatmulAllToAllvTiling::GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<int64_t> GroupedMatmulAllToAllvTiling::A5_SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64};
const std::set<int64_t> GroupedMatmulAllToAllvTiling::A3_SUPPORT_RANK_SIZE{8, 16, 32, 64, 128};

ge::graphStatus GroupedMatmulAllToAllvTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    localParams_.opName = opName_;
    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulAllToAllvTiling::IsCapable()
{
    return true;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorNotSupport()
{
    auto sendCountsTensorShape = context_->GetOptionalInputShape(SEND_COUNTS_TENSOR_OPTIONAL_INDEX);
    auto recvCountsTensorShape = context_->GetOptionalInputShape(RECV_COUNTS_TENSOR_OPTIONAL_INDEX);
    OP_TILING_CHECK(
        sendCountsTensorShape != nullptr || recvCountsTensorShape != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sendCountsTensor/recvCountsTensor", "not nullptr",
                                              "The values of sendCountsTensor/recvCountsTensor must be nullptr."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorSupport()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorMM()
{
    auto mmXTensorShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    auto mmWeightTensorShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    auto mmYShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);

    bool isMmXNull = (mmXTensorShape == nullptr);
    bool isMmWeightNull = (mmWeightTensorShape == nullptr);
    bool isMmYNull = (mmYShape == nullptr);
    if (!isMmYNull) {
        isMmYNull = mmYShape->GetStorageShape().GetDimNum() == 0;
    }
    bool allNull = isMmXNull && isMmWeightNull && isMmYNull;
    bool allNotNull = !isMmXNull && !isMmWeightNull && !isMmYNull;
    OP_TILING_CHECK(
        !allNull && !allNotNull,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "mmXTensor/mmWeightTensor/mmYTensor", "inconsistent state",
                                              "all must exist or not exist at same time"),
        return ge::GRAPH_FAILED);
    if (!isMmXNull) {
        localParams_.hasSharedMm = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsGmm()
{
    auto gmmXDesc = context_->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmX"), return ge::GRAPH_FAILED);
    localParams_.gmmXDtype = gmmXDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_X_DTYPE_LIST, localParams_.gmmXDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmX", Ops::Base::ToString(localParams_.gmmXDtype).c_str(),
                                              "DT_FLOAT16 or DT_BF16"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeight"),
                    return ge::GRAPH_FAILED);
    auto gmmWeightDesc = context_->GetInputDesc(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeight"),
                    return ge::GRAPH_FAILED);
    localParams_.gmmWeightDtype = gmmWeightDesc->GetDataType();
    OP_TILING_CHECK(
        !IsContains(GMM_WEIGHT_DTYPE_LIST, localParams_.gmmWeightDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmWeight", Ops::Base::ToString(localParams_.gmmWeightDtype).c_str(),
                                  "DT_FLOAT16 or DT_BF16"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        localParams_.gmmXDtype != localParams_.gmmWeightDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "gmmX", Ops::Base::ToString(localParams_.gmmXDtype).c_str(),
                                              "The dtype of gmmX must be the same as that of gmmWeight"),
        return ge::GRAPH_FAILED);

    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "y"), return ge::GRAPH_FAILED);
    localParams_.yDtype = yDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_Y_DTYPE_LIST, localParams_.yDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "y", Ops::Base::ToString(localParams_.yDtype).c_str(),
                                              "DT_FLOAT16 or DT_BF16"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        localParams_.gmmXDtype != localParams_.yDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "gmmX", Ops::Base::ToString(localParams_.gmmXDtype).c_str(),
                                              "The dtype of gmmX must be the same as that of y"),
        return ge::GRAPH_FAILED);
    localParams_.gmmYDtype = localParams_.yDtype;

    const gert::StorageShape *gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape *yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmX"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "y"), return ge::GRAPH_FAILED);

    auto status = CheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(yStorageShape, DIM_TWO, "yShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.A = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.ep = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.gmmWeightDim1 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.gmmWeightDim2 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);
    localParams_.BsK = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.N1 = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    auto mmXDesc = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX);
    auto mmWeightDesc = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmX"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeight"), return ge::GRAPH_FAILED);
    localParams_.mmXDtype = mmXDesc->GetDataType();
    localParams_.mmWeightDtype = mmWeightDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_X_DTYPE_LIST, localParams_.mmXDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmX", Ops::Base::ToString(localParams_.mmXDtype).c_str(),
                                              "DT_FLOAT16 or DT_BF16"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContains(GMM_WEIGHT_DTYPE_LIST, localParams_.mmWeightDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmWeight", Ops::Base::ToString(localParams_.mmWeightDtype).c_str(),
                                  "DT_FLOAT16 or DT_BF16"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        localParams_.gmmXDtype != localParams_.mmXDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "mmX", Ops::Base::ToString(localParams_.mmXDtype).c_str(),
                                              "The dtype of mmX must be the same as that of gmmX"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmXDtype != localParams_.mmWeightDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "mmWeight",
                                                          Ops::Base::ToString(localParams_.mmWeightDtype).c_str(),
                                                          "The dtype of mmWeight must be the same as that of gmmX"),
                    return ge::GRAPH_FAILED);
    auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmYDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmY"), return ge::GRAPH_FAILED);
    localParams_.mmYDtype = mmYDesc->GetDataType();

    const gert::StorageShape *mmXStorageShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    const gert::StorageShape *mmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmX"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeight"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmYStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmY"), return ge::GRAPH_FAILED);

    auto status = CheckShapeDimensions(mmXStorageShape, DIM_TWO, "mmXShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(mmWeightStorageShape, DIM_TWO, "mmWeightShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(mmYStorageShape, DIM_TWO, "mmYShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.Bs = mmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H2 = mmXStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.mmWeightDim0 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.mmWeightDim1 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmYDim0 = mmYStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(
        localParams_.Bs != mmYDim0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "mmX/mmY", std::to_string(localParams_.Bs).c_str(),
                                                 "Shape dim 0 of mmX must be equal to shape dim 0 of mmY."),
        return ge::GRAPH_FAILED);
    localParams_.N2 = mmYStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);

    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(transGmmWeightPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "transGmmWeight"),
                    return ge::GRAPH_FAILED);
    localParams_.isGmmWeightTrans = *transGmmWeightPtr;

    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(transMmWeightPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "transMmWeight"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *transMmWeightPtr == true && !localParams_.hasSharedMm,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "transMmWeight", std::to_string(*transMmWeightPtr).c_str(),
                                              "should not be true when mmX is null"),
        return ge::GRAPH_FAILED);
    localParams_.isMmWeightTrans = *transMmWeightPtr;

    localParams_.gmmXScaleDtype = localParams_.gmmXDtype;
    localParams_.gmmWeightScaleDtype = localParams_.gmmXDtype;
    localParams_.mmXScaleDtype = localParams_.gmmXDtype;
    localParams_.mmWeightScaleDtype = localParams_.gmmXDtype;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckAndSetLocalParams()
{
    auto status = CheckAndSetLocalParamsGmm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetLocalParamsMm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetLocalParamsAttr();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckParamsRelationGmm()
{
    return CheckParamsRelationGmmTransShape();
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckParamsRelationMm()
{
    return CheckParamsRelationMmTransShape();
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckParamsAttrEpAndSetLocalParams()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "group"), return ge::GRAPH_FAILED);

    int64_t rankDim = 0;
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    if (*epWorldSizePtr == RANK_DEFAULT_NUM) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, rankDim), OP_LOGE(opName_, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *epWorldSizePtr;
    }

    std::string supportRankSizeRange;
    const std::set<int64_t> &supportRankSize =
        (npuArch_ == NpuArch::DAV_3510) ? A5_SUPPORT_RANK_SIZE : A3_SUPPORT_RANK_SIZE;
    for (const auto &v : supportRankSize) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    OP_TILING_CHECK(
        supportRankSize.find(rankDim) == supportRankSize.end(),
        OP_LOGE_FOR_INVALID_VALUE(opName_, "rankSize", std::to_string(rankDim).c_str(), supportRankSizeRange.c_str()),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    bool isEpNumMatch = (localParams_.ep > 0) && (localParams_.ep < 33);
    if (!isEpNumMatch) {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "ep", std::to_string(localParams_.ep).c_str(),
                                  std::to_string(MAX_EXPERT_NUM_PER_RANK).c_str());
        return ge::GRAPH_FAILED;
    }
    uint64_t expertNum = localParams_.ep * rankDim;
    OP_TILING_CHECK(expertNum > MAX_EXPERT_NUM,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "expertNum", std::to_string(expertNum).c_str(),
                                                          "exceeds max expert num"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::GetPlatformInfo()
{
    auto status = Mc2Tiling::Mc2GroupedMatmul::QuantGroupedMatmulAllToAllvTilingCommon::GetPlatformInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Failed to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    std::string socVersion;
    (void)platformInfo->GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    // AIV has its own MTE tiling and GetPlatformInfo override. Reject 910B in
    // this legacy candidate before GetTilingKey, whose failure is only a key.
    OP_TILING_CHECK(socVersion == "Ascend910B",
                    OP_LOGE(opName_, "Ascend 910B only supports commMode 'aiv'; AICPU tiling is not supported."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint32_t GroupedMatmulAllToAllvTiling::GetCommModeIndex() const
{
    return ATTR_COMM_MODE;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::GetAndConvertCommMode(gert::TilingContext *context,
                                                                    uint8_t &commMode) const
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    const char *commModeStr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE);
    OP_TILING_CHECK(commModeStr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "comm_mode"),
                    return ge::GRAPH_FAILED);
    const size_t maxLength = 7UL;
    if (npuArch_ == NpuArch::DAV_3510) {
        if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
            commMode = Mc2Comm::COMM_MODE_AICPU;
        } else if (strncmp(commModeStr, "ccu", maxLength) == 0) {
            commMode = Mc2Comm::COMM_MODE_CCU;
        } else {
            OP_LOGD(context->GetNodeName(), "Currently, commMode only support 'ccu', 'ai_cpu', but got %s.",
                    commModeStr);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
            commMode = Mc2Comm::COMM_MODE_AICPU;
        } else {
            OP_LOGD(context->GetNodeName(), "Currently, commMode only support 'ai_cpu', but got %s.", commModeStr);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    uint8_t commMode = 0;
    if (GetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans,
                                                  localParams_.isMmWeightTrans, commMode);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, commMode, tilingKey);
    return tilingKey;
}

static ge::graphStatus GroupedMatMulAlltoAllvTilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct GroupedMatMulAlltoAllvCompileInfo {};
static ge::graphStatus TilingParseForGroupedMatMulAlltoAllv(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatMulAlltoAllv)
    .Tiling(GroupedMatMulAlltoAllvTilingFunc)
    .TilingParse<GroupedMatMulAlltoAllvCompileInfo>(TilingParseForGroupedMatMulAlltoAllv);
} // namespace optiling
