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
 * \file grouped_mat_mul_allto_allv_mte_tiling.cpp
 */

#include "grouped_mat_mul_allto_allv_mte_tiling.h"
#include "../../../op_kernel/arch22/grouped_mat_mul_allto_allv_mte_tiling.h"
#include "op_mc2.h"
#include "mc2_log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_comm_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "graph/utils/type_utils.h"
#include <cstring>

using namespace Mc2Log;
using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace optiling {

namespace {
constexpr uint8_t COMM_MODE_AIV = 2U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint32_t OP_TYPE_MULTI_PUT = 18U;
constexpr uint64_t A2_EXPERT_OVERLAP_MAX_RANK_SIZE = 8UL;
const std::string A3_MTE_HCCL_ALG_CONFIG = "AlltoAll=level0:fullmesh;level1:pairwise";
const std::string A2_MTE_HCCL_ALG_CONFIG = "MultiPut=level0:fullmesh";
} // namespace

const std::vector<uint32_t> GroupedMatmulAllToAllvMteTiling::GMM_X_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GroupedMatmulAllToAllvMteTiling::GMM_WEIGHT_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GroupedMatmulAllToAllvMteTiling::GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<int64_t> GroupedMatmulAllToAllvMteTiling::A5_SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64};
const std::set<int64_t> GroupedMatmulAllToAllvMteTiling::A2_SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64, 128};
const std::set<int64_t> GroupedMatmulAllToAllvMteTiling::A3_SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64, 128};

ge::graphStatus GroupedMatmulAllToAllvMteTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    localParams_.opName = opName_;
    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulAllToAllvMteTiling::IsCapable()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const char *commMode = attrs->GetAttrPointer<char>(ATTR_COMM_MODE);
    return commMode != nullptr && strncmp(commMode, "aiv", 7UL) == 0;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::DoOpTiling()
{
    // SyncAll is used by the MIX AIC/AIV path.  Match the arch22
    // AllToAllMatmul/MatmulAlltoAll scheduling contract so every participating
    // core is launched as one batch.
    constexpr uint32_t batchMode = 1U;
    MC2_CHECK_LOG_RET(opName_, context_->SetScheduleMode(batchMode));
    MC2_CHECK_LOG_RET(opName_, CheckAndSetInputOutputInfo());
    MC2_CHECK_LOG_RET(opName_, SetTilingCommonInfo());
    MC2_CHECK_LOG_RET(opName_, SetMteWorkspaceInfo());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::SetMteWorkspaceInfo()
{
    const uint64_t dtypeSize = mc2tiling::GetDataTypeSize(opName_, localParams_.gmmYDtype);
    const uint64_t rowBytes = localParams_.N1 * dtypeSize;
    OP_TILING_CHECK(rowBytes > MC2KernelTemplate::Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES,
                    OP_LOGE(opName_,
                            "MTE peer copy requires one complete output row in payload UB: row=%lu bytes, "
                            "capacity=%lu bytes.",
                            rowBytes, MC2KernelTemplate::Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES),
                    return ge::GRAPH_FAILED);
    const uint64_t gmmOutputWorkspaceSize = mc2tiling::AlignUp(
        localParams_.A * localParams_.N1 * dtypeSize, MC2KernelTemplate::Gmma2avMteTiling::WORKSPACE_ALIGNMENT);
    const uint64_t countStorageBytes =
        mc2tiling::AlignUp(localParams_.epWorldSize * localParams_.ep * sizeof(int32_t),
                           MC2KernelTemplate::Gmma2avMteTiling::COUNT_TABLE_ALIGNMENT_BYTES);
    const uint64_t cumsumWorkspaceSize =
        mc2tiling::AlignUp(countStorageBytes, MC2KernelTemplate::Gmma2avMteTiling::WORKSPACE_ALIGNMENT);
    mteWorkspaceSize_ = static_cast<uint64_t>(libApiWorkSpaceSize_) + gmmOutputWorkspaceSize + cumsumWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckOpInputSingleParamsTensorNotSupport()
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

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckOpInputSingleParamsTensorSupport()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckOpInputSingleParamsTensorMM()
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

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckAndSetLocalParamsGmm()
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

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckAndSetLocalParamsMm()
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
    OP_TILING_CHECK(!IsContains(GMM_Y_DTYPE_LIST, localParams_.mmYDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmY", Ops::Base::ToString(localParams_.mmYDtype).c_str(),
                                              "DT_FLOAT16 or DT_BF16"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        localParams_.mmXDtype != localParams_.mmYDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "mmY", Ops::Base::ToString(localParams_.mmYDtype).c_str(),
                                              "The dtype of mmY must be the same as that of mmX"),
        return ge::GRAPH_FAILED);

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

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckAndSetLocalParamsAttr()
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

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckAndSetLocalParams()
{
    auto status = CheckAndSetLocalParamsGmm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetLocalParamsMm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetLocalParamsAttr();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckFormat();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckFormat()
{
    auto gmmXDesc = context_->GetInputDesc(GMM_X_INDEX);
    auto gmmWeightDesc = context_->GetInputDesc(GMM_WEIGHT_INDEX);
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(gmmXDesc == nullptr || gmmWeightDesc == nullptr || yDesc == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmX/gmmWeight/y desc"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName_, "gmmX",
                                               Ops::Base::ToString(gmmXDesc->GetStorageFormat()).c_str(), "FORMAT_ND"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        gmmWeightDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(opName_, "gmmWeight", Ops::Base::ToString(gmmWeightDesc->GetStorageFormat()).c_str(),
                                   "FORMAT_ND"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        yDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(opName_, "y", Ops::Base::ToString(yDesc->GetStorageFormat()).c_str(), "FORMAT_ND"),
        return ge::GRAPH_FAILED);
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }

    auto mmXDesc = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX);
    auto mmWeightDesc = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX);
    auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXDesc == nullptr || mmWeightDesc == nullptr || mmYDesc == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "mmX/mmWeight/mmY desc"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmXDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName_, "mmX", Ops::Base::ToString(mmXDesc->GetStorageFormat()).c_str(),
                                               "FORMAT_ND"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        mmWeightDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(opName_, "mmWeight", Ops::Base::ToString(mmWeightDesc->GetStorageFormat()).c_str(),
                                   "FORMAT_ND"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmYDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName_, "mmY", Ops::Base::ToString(mmYDesc->GetStorageFormat()).c_str(),
                                               "FORMAT_ND"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckParamsRelationGmm()
{
    return CheckParamsRelationGmmTransShape();
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckParamsRelationMm()
{
    return CheckParamsRelationMmTransShape();
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckParamsAttrEpAndSetLocalParams()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "group"), return ge::GRAPH_FAILED);

    int64_t rankDim = 0;
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    int64_t communicatorRankDim = 0;
    OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, communicatorRankDim),
                    OP_LOGE(opName_, "GetRankSize failed."), return ge::GRAPH_FAILED);
    rankDim = (*epWorldSizePtr == RANK_DEFAULT_NUM) ? communicatorRankDim : *epWorldSizePtr;
    OP_TILING_CHECK(
        rankDim != communicatorRankDim,
        OP_LOGE(opName_, "epWorldSize=%ld does not match communicator rank size=%ld.", rankDim, communicatorRankDim),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankDim <= 0 || static_cast<uint64_t>(rankDim) > MC2KernelTemplate::Gmma2avMteTiling::MAX_RANK_SIZE,
                    OP_LOGE(opName_, "rankSize=%ld must be in [1, %u].", rankDim,
                            MC2KernelTemplate::Gmma2avMteTiling::MAX_RANK_SIZE),
                    return ge::GRAPH_FAILED);

    std::string supportRankSizeRange;
    const std::set<int64_t> &supportRankSize = (npuArch_ == Ops::Base::DAV_3510) ?
                                                   A5_SUPPORT_RANK_SIZE :
                                                   (isA3_ ? A3_SUPPORT_RANK_SIZE : A2_SUPPORT_RANK_SIZE);
    for (const auto &v : supportRankSize) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    OP_TILING_CHECK(
        supportRankSize.find(rankDim) == supportRankSize.end(),
        OP_LOGE_FOR_INVALID_VALUE(opName_, "rankSize", std::to_string(rankDim).c_str(), supportRankSizeRange.c_str()),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    OP_TILING_CHECK(localParams_.ep == 0 || localParams_.ep > MAX_GLOBAL_EXPERT_NUM,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "ep", std::to_string(localParams_.ep).c_str(), "(0, 1024]"),
                    return ge::GRAPH_FAILED);
    const uint64_t rankSize = static_cast<uint64_t>(rankDim);
    OP_TILING_CHECK(
        localParams_.ep > MC2KernelTemplate::Gmma2avMteTiling::MAX_COUNT_NUM / rankSize,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "expertNum", std::to_string(localParams_.ep * rankSize).c_str(),
                                              "rankSize * expertNumPerRank must be in [1, 1024]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckAndSetSendRecvCountsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const uint64_t expertNum = localParams_.ep * localParams_.epWorldSize;
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr || recvCountsPtr == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "sendCounts/recvCounts"), return ge::GRAPH_FAILED);
    const uint64_t sendCountsSize = sendCountsPtr->GetSize();
    const uint64_t recvCountsSize = recvCountsPtr->GetSize();
    OP_TILING_CHECK(sendCountsSize != expertNum || recvCountsSize != expertNum,
                    OP_LOGE(opName_, "sendCounts and recvCounts must both contain %lu elements, got %lu and %lu.",
                            expertNum, sendCountsSize, recvCountsSize),
                    return ge::GRAPH_FAILED);

    const int64_t *sendCounts = static_cast<const int64_t *>(sendCountsPtr->GetData());
    const int64_t *recvCounts = static_cast<const int64_t *>(recvCountsPtr->GetData());
    OP_TILING_CHECK(sendCounts == nullptr || recvCounts == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "sendCounts/recvCounts data"), return ge::GRAPH_FAILED);
    uint64_t sendCountsSum = 0UL;
    uint64_t recvCountsSum = 0UL;
    for (uint64_t i = 0UL; i < expertNum; ++i) {
        OP_TILING_CHECK(sendCounts[i] < 0 || sendCounts[i] > static_cast<int64_t>(localParams_.A),
                        OP_LOGE(opName_, "sendCounts[%lu]=%ld is outside [0, %lu].", i, sendCounts[i], localParams_.A),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            recvCounts[i] < 0 || recvCounts[i] > static_cast<int64_t>(localParams_.BsK),
            OP_LOGE(opName_, "recvCounts[%lu]=%ld is outside [0, %lu].", i, recvCounts[i], localParams_.BsK),
            return ge::GRAPH_FAILED);
        sendCounts_[i] = static_cast<int32_t>(sendCounts[i]);
        recvCounts_[i] = static_cast<int32_t>(recvCounts[i]);
        sendCountsSum += static_cast<uint64_t>(sendCounts[i]);
        recvCountsSum += static_cast<uint64_t>(recvCounts[i]);
    }
    OP_TILING_CHECK(sendCountsSum != localParams_.A,
                    OP_LOGE(opName_, "sendCounts sum must equal A=%lu, got %lu.", localParams_.A, sendCountsSum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsSum != localParams_.BsK,
                    OP_LOGE(opName_, "recvCounts sum must equal BSK=%lu, got %lu.", localParams_.BsK, recvCountsSum),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckTopK(uint64_t topK)
{
    // Only A3 AIV extends top-k. AICPU and other SoCs retain the common limits.
    if (!isA3_ || !IsCapable()) {
        return QuantGroupedMatmulAllToAllvTilingCommon::CheckTopK(topK);
    }
    constexpr uint64_t maxAivTopK = 16UL;
    OP_TILING_CHECK(
        topK < MIN_K_VALUE || topK > maxAivTopK,
        OP_LOGE_FOR_INVALID_VALUE(opName_, "K", std::to_string(topK),
                                  "[" + std::to_string(MIN_K_VALUE) + ", " + std::to_string(maxAivTopK) + "]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckLocalParams()
{
    MC2_CHECK_LOG_RET(opName_,
                      Mc2Tiling::Mc2GroupedMatmul::QuantGroupedMatmulAllToAllvTilingCommon::CheckLocalParams());
    OP_TILING_CHECK(localParams_.A == 0UL || localParams_.A > MAX_TOKEN_NUM,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "A", std::to_string(localParams_.A).c_str(), "[1, 5000000]"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.BsK == 0UL || localParams_.BsK > MAX_TOKEN_NUM,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "BSK", std::to_string(localParams_.BsK).c_str(), "[1, 5000000]"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::GetPlatformInfo()
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
    isA3_ = socVersion == "Ascend910_93";
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::PostTiling()
{
    context_->SetBlockDim(localParams_.aicCoreNum);

    MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData outData{};
    auto &taskTiling = outData.taskTilingInfo;
    taskTiling.BSK = localParams_.BsK;
    taskTiling.BS = localParams_.Bs;
    taskTiling.H1 = localParams_.H1;
    taskTiling.H2 = localParams_.H2;
    taskTiling.A = localParams_.A;
    taskTiling.N1 = localParams_.N1;
    taskTiling.N2 = localParams_.N2;
    taskTiling.epWorldSize = localParams_.epWorldSize;
    taskTiling.e = localParams_.ep;
    taskTiling.aivCoreNum = localParams_.aivCoreNum;
    taskTiling.aicCoreNum = localParams_.aicCoreNum;
    const uint64_t countNum = localParams_.epWorldSize * localParams_.ep;
    for (uint64_t i = 0UL; i < countNum; ++i) {
        taskTiling.sendCnt[i] = sendCounts_[i];
        taskTiling.recvCnt[i] = recvCounts_[i];
    }
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "group"), return ge::GRAPH_FAILED);
    outData.isA3 = static_cast<uint32_t>(isA3_);
    const uint32_t opType = isA3_ ? OP_TYPE_ALL_TO_ALL : OP_TYPE_MULTI_PUT;
    const std::string &algConfig = isA3_ ? A3_MTE_HCCL_ALG_CONFIG : A2_MTE_HCCL_ALG_CONFIG;
    Mc2CcTilingConfig mc2InitConfig(group, opType, algConfig);
    OP_TILING_CHECK(mc2InitConfig.SetCommEngine(Mc2Comm::ENGINE_MTE) != 0,
                    OP_LOGE(opName_, "Failed to select the MTE communication engine."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2InitConfig.GetTiling(outData.mc2InitTiling) != 0,
                    OP_LOGE(opName_, "Failed to generate the MTE Mc2InitTiling header."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2InitConfig.GetTiling(outData.mc2CcTiling) != 0,
                    OP_LOGE(opName_, "Failed to generate the MTE Mc2CcTiling descriptor."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2tiling::GetCclBufferSize(group, &outData.commBufferSize, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Failed to get CCL buffer size for MTE communication."), return ge::GRAPH_FAILED);

    const uint64_t countMatrixBytes =
        mc2tiling::AlignUp(localParams_.epWorldSize * localParams_.ep * sizeof(int32_t),
                           MC2KernelTemplate::Gmma2avMteTiling::COUNT_TABLE_ALIGNMENT_BYTES);
    const uint64_t sendCountStagingBytes = MC2KernelTemplate::Gmma2avMteTiling::SEND_COUNT_STAGING_FROM_TAIL -
                                           MC2KernelTemplate::Gmma2avMteTiling::SYNC_REGION_FROM_TAIL;
    OP_TILING_CHECK(countMatrixBytes > sendCountStagingBytes,
                    OP_LOGE(opName_, "MTE send-count staging requires %lu bytes, but its CCL region has %lu bytes.",
                            countMatrixBytes, sendCountStagingBytes),
                    return ge::GRAPH_FAILED);
    const uint64_t syncBytes = MC2KernelTemplate::Gmma2avMteTiling::FIXED_SYNC_BYTES;
    OP_TILING_CHECK(syncBytes > MC2KernelTemplate::Gmma2avMteTiling::CCL_TAIL_SAFETY_BYTES,
                    OP_LOGE(opName_, "MTE fixed sync layout requires %lu bytes, but the CCL tail region has %lu bytes.",
                            syncBytes, MC2KernelTemplate::Gmma2avMteTiling::CCL_TAIL_SAFETY_BYTES),
                    return ge::GRAPH_FAILED);
    const uint64_t dtypeSize = mc2tiling::GetDataTypeSize(opName_, localParams_.gmmYDtype);
    const uint64_t sendBytes = localParams_.A * localParams_.N1 * dtypeSize;
    // Peers pull the published GMM result directly into their output tensor.
    // Only sendBytes occupies the window payload; the count table is already
    // contained in CONTROL_REGION_BYTES alongside the synchronization region.
    const uint64_t dataAndControlBytes = sendBytes + MC2KernelTemplate::Gmma2avMteTiling::CONTROL_REGION_BYTES;
    const uint64_t requiredCclBufferSize =
        dataAndControlBytes > MC2KernelTemplate::Gmma2avMteTiling::MIN_CCL_BUFFER_BYTES ?
            dataAndControlBytes :
            MC2KernelTemplate::Gmma2avMteTiling::MIN_CCL_BUFFER_BYTES;
    OP_TILING_CHECK(outData.commBufferSize < requiredCclBufferSize,
                    OP_LOGE(opName_,
                            "CCL buffer is too small for MTE communication: actual=%lu bytes, required=%lu bytes "
                            "(send=%lu bytes, control including count matrix=%lu bytes, "
                            "arch22 minimum=%lu bytes).",
                            outData.commBufferSize, requiredCclBufferSize, sendBytes,
                            MC2KernelTemplate::Gmma2avMteTiling::CONTROL_REGION_BYTES,
                            MC2KernelTemplate::Gmma2avMteTiling::MIN_CCL_BUFFER_BYTES),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(outData.commBufferSize < MC2KernelTemplate::Gmma2avMteTiling::SYNC_REGION_FROM_TAIL,
                    OP_LOGE(opName_, "CCL buffer=%lu bytes is smaller than the MTE sync-tail offset=%lu bytes.",
                            outData.commBufferSize, MC2KernelTemplate::Gmma2avMteTiling::SYNC_REGION_FROM_TAIL),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(outData.commBufferSize % MC2KernelTemplate::Gmma2avMteTiling::SYNC_SLOT_BYTES != 0UL,
                    OP_LOGE(opName_, "CCL buffer=%lu bytes does not preserve %lu-byte sync-slot alignment.",
                            outData.commBufferSize, MC2KernelTemplate::Gmma2avMteTiling::SYNC_SLOT_BYTES),
                    return ge::GRAPH_FAILED);

    MC2_CHECK_LOG_RET(opName_, CheckExpertPipeline());
    MC2_CHECK_LOG_RET(opName_, SetExpertChunkRows(outData.expertChunkRows));

    outData.cumsumWorkspaceOffset = mc2tiling::AlignUp(localParams_.A * localParams_.N1 * dtypeSize,
                                                       MC2KernelTemplate::Gmma2avMteTiling::WORKSPACE_ALIGNMENT);

    outData.cocTiling.m0 = MC2KernelTemplate::Gmma2avMteTiling::L1_TILE_M;
    outData.cocTiling.k0 = MC2KernelTemplate::Gmma2avMteTiling::L1_TILE_K;
    outData.cocTiling.n0 = MC2KernelTemplate::Gmma2avMteTiling::L1_TILE_N;
    outData.cocTiling.ubMoveNum = static_cast<int32_t>(localParams_.N1);
    outData.cocTiling.swizzlCount = MC2KernelTemplate::Gmma2avMteTiling::SWIZZLE_COUNT;
    outData.cocTiling.swizzlDirect = MC2KernelTemplate::Gmma2avMteTiling::SWIZZLE_DIRECTION;

    auto *outTilingData = context_->GetTilingData<MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK(outTilingData == nullptr, OP_LOGE(opName_, "Failed to get MTE tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        tilingBufCap < sizeof(outData),
        OP_LOGE(opName_, "TilingBuffer too small, capacity = %zu, need = %zu.", tilingBufCap, sizeof(outData)),
        return ge::GRAPH_FAILED);
    static_assert(sizeof(outData) % sizeof(uint64_t) == 0U, "MTE tiling data must be 8-byte aligned");
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &outData, sizeof(outData));
    OP_TILING_CHECK(ret != EOK, OP_LOGE(opName_, "postTiling: memcpy_s failed with ret=%d.", ret),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(sizeof(outData));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::CheckExpertPipeline() const
{
    OP_TILING_CHECK(!isA3_ && localParams_.epWorldSize > A2_EXPERT_OVERLAP_MAX_RANK_SIZE,
                    OP_LOGE(opName_, "Expert-overlap mode on Ascend 910B only supports W=2, 4 or 8, got W=%lu.",
                            localParams_.epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        localParams_.epWorldSize > MC2KernelTemplate::Gmma2avMteTiling::MAX_RANK_SIZE ||
            localParams_.ep > MC2KernelTemplate::Gmma2avMteTiling::MAX_COUNT_NUM / localParams_.epWorldSize,
        OP_LOGE(opName_,
                "Expert-overlap sync layout capacity exceeded: W=%lu, E=%lu, maxW=%u, "
                "max(W*E)=%u.",
                localParams_.epWorldSize, localParams_.ep, MC2KernelTemplate::Gmma2avMteTiling::MAX_RANK_SIZE,
                MC2KernelTemplate::Gmma2avMteTiling::MAX_COUNT_NUM),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MC2KernelTemplate::Gmma2avMteTiling::FIXED_SYNC_BYTES >
                        MC2KernelTemplate::Gmma2avMteTiling::SYNC_REGION_FROM_TAIL,
                    OP_LOGE(opName_, "Expert-overlap fixed sync layout does not fit its reserved CCL region."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::SetExpertChunkRows(uint32_t &expertChunkRows) const
{
    using namespace MC2KernelTemplate;
    expertChunkRows = GMMA2AV_MTE_SINGLE_CHUNK;
    if (!isA3_ || localParams_.ep < 2UL || localParams_.A <= GMMA2AV_MTE_DENSE_EXPERT_CHUNK_ROWS) {
        return ge::GRAPH_SUCCESS;
    }

    // Weight each expert's destination fanout by its local GMM rows. Counts
    // already reside in host tiling attributes; no environment or collective
    // is needed. Sparse fanout uses one chunk to avoid repeated pipeline drains.
    uint64_t weightedPeerRows = 0UL;
    for (uint64_t expert = 0UL; expert < localParams_.ep; ++expert) {
        uint64_t rows = 0UL;
        uint64_t peers = 0UL;
        for (uint64_t rank = 0UL; rank < localParams_.epWorldSize; ++rank) {
            const int32_t count = sendCounts_[rank * localParams_.ep + expert];
            if (count > 0) {
                rows += static_cast<uint64_t>(count);
                ++peers;
            }
        }
        weightedPeerRows += rows * peers;
    }
    // A3 baseline heuristic: use 1536 rows when the weighted fanout exceeds
    // half the communication domain. Rank-local chunk boundaries may differ;
    // the kernel exchanges persistent expert-granular readiness, not chunk IDs.
    if (weightedPeerRows * 2UL > localParams_.A * localParams_.epWorldSize) {
        expertChunkRows = GMMA2AV_MTE_DENSE_EXPERT_CHUNK_ROWS;
    }
    OP_LOGD(opName_, "MTE overlap chunk rows=%u (0=single), weighted peer rows=%lu, A=%lu, W=%lu.", expertChunkRows,
            weightedPeerRows, localParams_.A, localParams_.epWorldSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Failed to get workspace."), return ge::GRAPH_FAILED);
    workspaces[0] = mteWorkspaceSize_;
    OP_LOGD(opName_, "MTE workspace size=%lu", mteWorkspaceSize_);
    return ge::GRAPH_SUCCESS;
}

uint32_t GroupedMatmulAllToAllvMteTiling::GetCommModeIndex() const
{
    return ATTR_COMM_MODE;
}

ge::graphStatus GroupedMatmulAllToAllvMteTiling::GetAndConvertCommMode(gert::TilingContext *context,
                                                                       uint8_t &commMode) const
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    const char *commModeStr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE);
    OP_TILING_CHECK(commModeStr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "comm_mode"),
                    return ge::GRAPH_FAILED);
    constexpr size_t maxLength = 7UL;
    OP_TILING_CHECK(
        npuArch_ == Ops::Base::DAV_3510,
        OP_LOGE(context->GetNodeName(), "GroupedMatMulAlltoAllv AIV-driven MTE mode only supports Atlas A2/A3."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(strncmp(commModeStr, "aiv", maxLength) != 0,
                    OP_LOGE(context->GetNodeName(), "This tiling only supports commMode 'aiv', got %s.", commModeStr),
                    return ge::GRAPH_FAILED);
    commMode = COMM_MODE_AIV;
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulAllToAllvMteTiling::GetTilingKey() const
{
    uint8_t commMode = 0;
    if (GetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // The common quant tiling header declares COMM_MODE as a one-bit field. The AIV-driven MTE implementation
    // additionally uses value 2, so encode its device tiling-key layout locally: bool/bool/bool/uint2.
    constexpr uint64_t commModeShift = 3UL;
    constexpr uint64_t gmmWeightTransShift = 1UL;
    constexpr uint64_t sharedMmWeightTransShift = 2UL;
    const uint64_t tilingKey = static_cast<uint64_t>(localParams_.hasSharedMm) |
                               (static_cast<uint64_t>(localParams_.isGmmWeightTrans) << gmmWeightTransShift) |
                               (static_cast<uint64_t>(localParams_.isMmWeightTrans) << sharedMmWeightTransShift) |
                               (static_cast<uint64_t>(commMode) << commModeShift);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, commMode, tilingKey);
    return tilingKey;
}

} // namespace optiling
