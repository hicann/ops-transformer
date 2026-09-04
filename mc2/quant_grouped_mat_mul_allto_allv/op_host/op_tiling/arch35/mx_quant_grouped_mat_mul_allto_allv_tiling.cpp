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
 * \file mx_quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "mx_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../quant_grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include "mc2_comm_utils.h"
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace Mc2Tiling {

const std::vector<uint32_t> MX_QUANT_GMM_X_DTYPE_LIST = {
    ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT4_E2M1,
};
const std::vector<uint32_t> MX_QUANT_GMM_WEIGHT_DTYPE_LIST = {
    ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT4_E2M1,
};
const std::vector<uint32_t> MX_QUANT_GMM_X_SCALE_DTYPE_LIST = {
    ge::DT_FLOAT8_E8M0,
};
const std::vector<uint32_t> MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST = {
    ge::DT_FLOAT8_E8M0,
};
const std::vector<uint32_t> MX_QUANT_GMM_Y_DTYPE_LIST = {
    ge::DT_FLOAT16,
    ge::DT_BF16,
};

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::MxCheckShapeDimensions(const gert::StorageShape *shape,
                                                                            uint64_t dims, const char *shapeName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        (dimNum != dims),
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, shapeName, std::to_string(dimNum) + "D", std::to_string(dims) + "D"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool MxQuantGroupedMatmulAllToAllvTiling::IsCapable()
{
    QuantModePair mode = GetQuantMode(context_, opName_);
    OP_TILING_CHECK(mode == QUANT_PAIR_ERROR,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "quantMode", "QUANT_PAIR_ERROR", "valid quant mode"),
                    return false);
    OP_LOGD(opName_, "QuantMode=%d, expected MX mode=%d", mode, QUANT_PAIR_MX);
    if (mode == QUANT_PAIR_MX) {
        OP_LOGD(opName_, "MxQuantGroupedMatmulAllToAllvTiling MX mode capable.");
        return true;
    }
    OP_LOGD(opName_, "Skip MxQuantGroupedMatmulAllToAllvTiling, mode mismatch.");
    return false;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsGmm()
{
    localParams_.gmmXDtype = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    localParams_.gmmWeightDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_X_DTYPE_LIST, localParams_.gmmXDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmX", Ops::Base::ToString(localParams_.gmmXDtype).c_str(),
                                              "(DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.gmmWeightDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmWeight", Ops::Base::ToString(localParams_.gmmWeightDtype).c_str(),
                                  "(DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1)"),
        return ge::GRAPH_FAILED);
    localParams_.yDtype = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_Y_DTYPE_LIST, localParams_.yDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "y", Ops::Base::ToString(localParams_.yDtype).c_str(),
                                              "(DT_FLOAT16, DT_BF16)"),
                    return ge::GRAPH_FAILED);
    localParams_.gmmYDtype = localParams_.yDtype;

    const gert::StorageShape *gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmX"), return ge::GRAPH_FAILED);

    const gert::StorageShape *gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeight"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "y"), return ge::GRAPH_FAILED);

    auto status = MxCheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = MxCheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = MxCheckShapeDimensions(yStorageShape, DIM_TWO, "yShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    localParams_.A = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.ep = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.BsK = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.N1 = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.gmmWeightDim1 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.gmmWeightDim2 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXDtype = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightDtype = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_X_DTYPE_LIST, localParams_.mmXDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmX", Ops::Base::ToString(localParams_.mmXDtype).c_str(),
                                              "(DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.mmWeightDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmWeight", Ops::Base::ToString(localParams_.mmWeightDtype).c_str(),
                                  "(DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1)"),
        return ge::GRAPH_FAILED);
    localParams_.mmYDtype = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_Y_DTYPE_LIST, localParams_.mmYDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmY", Ops::Base::ToString(localParams_.mmYDtype).c_str(),
                                              "(DT_FLOAT16, DT_BF16)"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmXStorageShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmX"), return ge::GRAPH_FAILED);

    const gert::StorageShape *mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmWeightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeight"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmYStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmY"), return ge::GRAPH_FAILED);

    auto status = MxCheckShapeDimensions(mmXStorageShape, DIM_TWO, "mmXShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = MxCheckShapeDimensions(mmWeightStorageShape, DIM_TWO, "mmWeightShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = MxCheckShapeDimensions(mmYStorageShape, DIM_TWO, "mmYShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    localParams_.Bs = mmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H2 = mmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    uint64_t mmYDim0 = mmYStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(localParams_.Bs != mmYDim0,
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmX, mmY",
                                              (std::string("[") + std::to_string(localParams_.Bs) + ", ...], [" +
                                               std::to_string(mmYDim0) + ", ...]")
                                                  .c_str(),
                                              "mmX DIM0 and mmY DIM0 should be equal"),
                    return ge::GRAPH_FAILED);

    localParams_.N2 = mmYStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.mmWeightDim0 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.mmWeightDim1 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsRelationGmm()
{
    auto gmmXScaleDesc = context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX);
    auto gmmWeightScaleDesc = context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gmmXScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeightScale"),
                    return ge::GRAPH_FAILED);

    localParams_.gmmXScaleDtype = gmmXScaleDesc->GetDataType();
    localParams_.gmmWeightScaleDtype = gmmWeightScaleDesc->GetDataType();
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.gmmXScaleDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmXScale", Ops::Base::ToString(localParams_.gmmXScaleDtype).c_str(),
                                  "(DT_FLOAT8_E8M0)"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.gmmWeightScaleDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmWeightScale",
                                  Ops::Base::ToString(localParams_.gmmWeightScaleDtype).c_str(), "(DT_FLOAT8_E8M0)"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *gmmXScaleStorageShape = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX);
    const gert::StorageShape *gmmWeightScaleStorageShape = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gmmXScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeightScale"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.gmmXQuantMode != QUANT_MX,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "gmmXQuantMode", std::to_string(localParams_.gmmXQuantMode),
                                              std::to_string(QUANT_MX)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        localParams_.gmmWeightQuantMode != QUANT_MX,
        OP_LOGE_WITH_INVALID_ATTR(opName_, "gmmWeightQuantMode", std::to_string(localParams_.gmmWeightQuantMode),
                                  std::to_string(QUANT_MX)),
        return ge::GRAPH_FAILED);
    ge::graphStatus status = MxCheckShapeDimensions(gmmXScaleStorageShape, DIM_THREE, "gmmXScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = MxCheckShapeDimensions(gmmWeightScaleStorageShape, DIM_FOUR, "gmmWeightScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.gmmQuantSuit = QUANT_PAIR_MX;
    if (!localParams_.isGmmWeightTrans) {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmX, gmmWeight",
                                                  (std::string("gmmX.dim1=") + std::to_string(localParams_.H1) +
                                                   ", gmmWeight.dim1=" + std::to_string(localParams_.gmmWeightDim1))
                                                      .c_str(),
                                                  "gmmX dim1 should equal gmmWeight dim1 in Non-Transposed scenario"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim2,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "y, gmmWeight",
                                                  (std::string("y.dim1=") + std::to_string(localParams_.N1) +
                                                   ", gmmWeight.dim2=" + std::to_string(localParams_.gmmWeightDim2))
                                                      .c_str(),
                                                  "y dim1 should equal gmmWeight dim2 in Non-Transposed scenario"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim2,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmX, gmmWeight",
                                                  (std::string("gmmX.dim1=") + std::to_string(localParams_.H1) +
                                                   ", gmmWeight.dim2=" + std::to_string(localParams_.gmmWeightDim2))
                                                      .c_str(),
                                                  "gmmX dim1 should equal gmmWeight dim2 in Transposed scenario"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "y, gmmWeight",
                                                  (std::string("y.dim1=") + std::to_string(localParams_.N1) +
                                                   ", gmmWeight.dim1=" + std::to_string(localParams_.gmmWeightDim1))
                                                      .c_str(),
                                                  "y dim1 should equal gmmWeight dim1 in Transposed scenario"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsRelationMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    auto mmXScaleDesc = context_->GetOptionalInputDesc(MM_X_SCALE_OPTIONAL_INDEX);
    auto mmWeightScaleDesc = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmXScale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeightScale"),
                    return ge::GRAPH_FAILED);

    localParams_.mmXScaleDtype = mmXScaleDesc->GetDataType();
    localParams_.mmWeightScaleDtype = mmWeightScaleDesc->GetDataType();
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.mmXScaleDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmXScale", Ops::Base::ToString(localParams_.mmXScaleDtype).c_str(),
                                  "(DT_FLOAT8_E8M0)"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.mmWeightScaleDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmWeightScale",
                                  Ops::Base::ToString(localParams_.mmWeightScaleDtype).c_str(), "(DT_FLOAT8_E8M0)"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *mmXScaleStorageShape = context_->GetOptionalInputShape(MM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightScaleStorageShape =
        context_->GetOptionalInputShape(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeightScale"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.mmXQuantMode != QUANT_MX,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "mmXQuantMode", std::to_string(localParams_.mmXQuantMode),
                                              std::to_string(QUANT_MX)),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.mmWeightQuantMode != QUANT_MX,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "mmWeightQuantMode",
                                              std::to_string(localParams_.mmWeightQuantMode), std::to_string(QUANT_MX)),
                    return ge::GRAPH_FAILED);
    ge::graphStatus status = MxCheckShapeDimensions(mmXScaleStorageShape, DIM_THREE, "mmXScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = MxCheckShapeDimensions(mmWeightScaleStorageShape, DIM_THREE, "mmWeightScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.mmQuantSuit = QUANT_PAIR_MX;
    if (!localParams_.isMmWeightTrans) {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim0,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmX, mmWeight",
                                                  (std::string("mmX.dim1=") + std::to_string(localParams_.H2) +
                                                   ", mmWeight.dim0=" + std::to_string(localParams_.mmWeightDim0))
                                                      .c_str(),
                                                  "mmX dim1 should equal mmWeight dim0 in Non-Transposed scenario"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmY, mmWeight",
                                                  (std::string("mmY.dim1=") + std::to_string(localParams_.N2) +
                                                   ", mmWeight.dim1=" + std::to_string(localParams_.mmWeightDim1))
                                                      .c_str(),
                                                  "mmY dim1 should equal mmWeight dim1 in Non-Transposed scenario"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmX, mmWeight",
                                                  (std::string("mmX.dim1=") + std::to_string(localParams_.H2) +
                                                   ", mmWeight.dim1=" + std::to_string(localParams_.mmWeightDim1))
                                                      .c_str(),
                                                  "mmX dim1 should equal mmWeight dim1 in Transposed scenario"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim0,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmY, mmWeight",
                                                  (std::string("mmY.dim1=") + std::to_string(localParams_.N2) +
                                                   ", mmWeight.dim0=" + std::to_string(localParams_.mmWeightDim0))
                                                      .c_str(),
                                                  "mmY dim1 should equal mmWeight dim0 in Transposed scenario"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsAttrEpAndSetLocalParams()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "group", "null", "not null"),
                    return ge::GRAPH_FAILED);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "epWorldSize", "null", "not null"),
                    return ge::GRAPH_FAILED);

    int64_t rankDim = 0;
    std::string supportRankSizeRange;
    for (const auto &v : SUPPORT_RANK_SIZE) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    if (*epWorldSizePtr == RANK_DEFAULT_NUM) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, rankDim), OP_LOGE(opName_, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *epWorldSizePtr;
    }
    OP_TILING_CHECK(
        SUPPORT_RANK_SIZE.find(rankDim) == SUPPORT_RANK_SIZE.end(),
        OP_LOGE_WITH_INVALID_ATTR(opName_, "epWorldSize", std::to_string(rankDim), supportRankSizeRange.c_str()),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    bool isEpNumMatch = (localParams_.ep > 0) && (localParams_.ep < 33);
    if (!isEpNumMatch) {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "ep", std::to_string(localParams_.ep), "in range (0, 32]");
        return ge::GRAPH_FAILED;
    } else {
        uint64_t expertNum = localParams_.ep * rankDim;
        OP_TILING_CHECK(expertNum > MAX_EXPERT_NUM,
                        OP_LOGE_FOR_INVALID_VALUE(opName_, "expertNum", std::to_string(expertNum),
                                                  "<= " + std::to_string(MAX_EXPERT_NUM)),
                        return ge::GRAPH_FAILED);
    }

    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_OPTIONAL_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "groupSize", "null", "not null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*groupSizePtr < 0,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "groupSize", std::to_string(*groupSizePtr), ">= 0"),
                    return ge::GRAPH_FAILED);

    localParams_.groupSize = *groupSizePtr;
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;

    OP_TILING_CHECK(((groupSizeM != MX_GROUP_SIZE_M) && (groupSizeM != 0)) ||
                        ((groupSizeN != MX_GROUP_SIZE_N) && (groupSizeN != 0)) ||
                        ((groupSizeK != MX_GROUP_SIZE_K) && (groupSizeK != 0)),
                    OP_LOGE_WITH_INVALID_ATTR(
                        opName_, "groupSize",
                        (std::string("groupSizeM=") + std::to_string(groupSizeM) +
                         ", groupSizeN=" + std::to_string(groupSizeN) + ", groupSizeK=" + std::to_string(groupSizeK))
                            .c_str(),
                        "groupSizeM=0/1, groupSizeN=0/1, groupSizeK=0/32"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckMxQuantDtypeConstraints()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(
        localParams_.gmmXDtype != localParams_.mmXDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "mmX", Ops::Base::ToString(localParams_.mmXDtype).c_str(),
                                              "The dtype of mmX must be the same as that of gmmX"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.gmmWeightDtype != localParams_.mmWeightDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "mmWeight", Ops::Base::ToString(localParams_.mmWeightDtype).c_str(),
                        "The dtype of mmWeight must be the same as that of gmmWeight"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        localParams_.yDtype != localParams_.mmYDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "mmY", Ops::Base::ToString(localParams_.mmYDtype).c_str(),
                                              "The dtype of mmY must be the same as that of y"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckMxQuantGmmScaleShapes()
{
    bool TransGmmWeightFlag = false;
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const bool *isTransGmmWeight = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    if (isTransGmmWeight) {
        TransGmmWeightFlag = *isTransGmmWeight;
    }
    const gert::StorageShape *gmmXScaleShape = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX);
    const gert::StorageShape *gmmWeightScaleShape = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK((gmmXScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((gmmWeightScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeightScale"),
                    return ge::GRAPH_FAILED);

    uint64_t gmmXScaleDim0 = gmmXScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t gmmXScaleDim1 = gmmXScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gmmXScaleDim2 = gmmXScaleShape->GetStorageShape().GetDim(DIM_TWO);
    uint64_t gmmWeightScaleDim0 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t gmmWeightScaleDim1 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gmmWeightScaleDim2 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_TWO);
    uint64_t gmmWeightScaleDim3 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_THREE);

    uint64_t gmmxDivH1 = (localParams_.H1 + MX_SCALE_GROUP - 1) / MX_SCALE_GROUP;
    OP_TILING_CHECK(
        (localParams_.A != gmmXScaleDim0) || (gmmxDivH1 != gmmXScaleDim1) || (gmmXScaleDim2 != EVEN_ALIGN),
        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmXScale",
                                  (std::string("[") + std::to_string(gmmXScaleDim0) + "," +
                                   std::to_string(gmmXScaleDim1) + "," + std::to_string(gmmXScaleDim2) + "]")
                                      .c_str(),
                                  (std::string("[") + std::to_string(localParams_.A) + "," + std::to_string(gmmxDivH1) +
                                   "," + std::to_string(EVEN_ALIGN) + "]")
                                      .c_str()),
        return ge::GRAPH_FAILED);

    if (localParams_.isGmmWeightTrans) { // Transposed Scenario
        OP_TILING_CHECK(
            (gmmWeightScaleDim0 != localParams_.ep) || (gmmWeightScaleDim1 != localParams_.N1) ||
                (gmmWeightScaleDim2 != gmmxDivH1) || (gmmWeightScaleDim3 != EVEN_ALIGN),
            OP_LOGE_FOR_INVALID_SHAPE(
                opName_, "gmmWeightScale",
                (std::string("[") + std::to_string(gmmWeightScaleDim0) + "," + std::to_string(gmmWeightScaleDim1) +
                 "," + std::to_string(gmmWeightScaleDim2) + "," + std::to_string(gmmWeightScaleDim3) + "]")
                    .c_str(),
                (std::string("[") + std::to_string(localParams_.ep) + "," + std::to_string(localParams_.N1) + "," +
                 std::to_string(gmmxDivH1) + "," + std::to_string(EVEN_ALIGN) + "]")
                    .c_str()),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            (gmmWeightScaleDim0 != localParams_.ep) || (gmmWeightScaleDim1 != gmmxDivH1) ||
                (gmmWeightScaleDim2 != localParams_.N1) || (gmmWeightScaleDim3 != EVEN_ALIGN),
            OP_LOGE_FOR_INVALID_SHAPE(
                opName_, "gmmWeightScale",
                (std::string("[") + std::to_string(gmmWeightScaleDim0) + "," + std::to_string(gmmWeightScaleDim1) +
                 "," + std::to_string(gmmWeightScaleDim2) + "," + std::to_string(gmmWeightScaleDim3) + "]")
                    .c_str(),
                (std::string("[") + std::to_string(localParams_.ep) + "," + std::to_string(gmmxDivH1) + "," +
                 std::to_string(localParams_.N1) + "," + std::to_string(EVEN_ALIGN) + "]")
                    .c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckMxQuantMmScaleShapes()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    bool TransmmWeightFlag = false;
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const bool *isTransmmWeight = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    if (isTransmmWeight) {
        TransmmWeightFlag = *isTransmmWeight;
    }
    const gert::StorageShape *mmXScaleShape = context_->GetOptionalInputShape(MM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightScaleShape = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK((mmXScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "mmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmWeightScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeightScale"),
                    return ge::GRAPH_FAILED);

    uint64_t mmXScaleDim0 = mmXScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t mmXScaleDim1 = mmXScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmXScaleDim2 = mmXScaleShape->GetStorageShape().GetDim(DIM_TWO);
    uint64_t mmWeightScaleDim0 = mmWeightScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t mmWeightScaleDim1 = mmWeightScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmWeightScaleDim2 = mmWeightScaleShape->GetStorageShape().GetDim(DIM_TWO);

    uint64_t mmxDivH2 = (localParams_.H2 + MX_SCALE_GROUP - 1) / MX_SCALE_GROUP;
    OP_TILING_CHECK((localParams_.Bs != mmXScaleDim0) || (mmxDivH2 != mmXScaleDim1) || (mmXScaleDim2 != EVEN_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmXScale",
                                              (std::string("[") + std::to_string(mmXScaleDim0) + "," +
                                               std::to_string(mmXScaleDim1) + "," + std::to_string(mmXScaleDim2) + "]")
                                                  .c_str(),
                                              (std::string("[") + std::to_string(localParams_.Bs) + "," +
                                               std::to_string(mmxDivH2) + "," + std::to_string(EVEN_ALIGN) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    if (localParams_.isMmWeightTrans) { // Transposed Scenario
        OP_TILING_CHECK((mmWeightScaleDim0 != localParams_.N2) || (mmWeightScaleDim1 != mmxDivH2) ||
                            (mmWeightScaleDim2 != EVEN_ALIGN),
                        OP_LOGE_FOR_INVALID_SHAPE(
                            opName_, "mmWeightScale",
                            (std::string("[") + std::to_string(mmWeightScaleDim0) + "," +
                             std::to_string(mmWeightScaleDim1) + "," + std::to_string(mmWeightScaleDim2) + "]")
                                .c_str(),
                            (std::string("[") + std::to_string(localParams_.N2) + "," + std::to_string(mmxDivH2) + "," +
                             std::to_string(EVEN_ALIGN) + "]")
                                .c_str()),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((mmWeightScaleDim0 != mmxDivH2) || (mmWeightScaleDim1 != localParams_.N2) ||
                            (mmWeightScaleDim2 != EVEN_ALIGN),
                        OP_LOGE_FOR_INVALID_SHAPE(
                            opName_, "mmWeightScale",
                            (std::string("[") + std::to_string(mmWeightScaleDim0) + "," +
                             std::to_string(mmWeightScaleDim1) + "," + std::to_string(mmWeightScaleDim2) + "]")
                                .c_str(),
                            (std::string("[") + std::to_string(mmxDivH2) + "," + std::to_string(localParams_.N2) + "," +
                             std::to_string(EVEN_ALIGN) + "]")
                                .c_str()),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckMxfp4SpecificConstraints()
{
    if (localParams_.gmmXDtype != ge::DT_FLOAT4_E2M1 && localParams_.gmmWeightDtype != ge::DT_FLOAT4_E2M1) {
        return ge::GRAPH_SUCCESS;
    }
    // In MXFP4 mode，the data types of x and weight must be consistent
    OP_TILING_CHECK(
        localParams_.gmmXDtype != localParams_.gmmWeightDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "gmmX, gmmWeight",
            (Ops::Base::ToString(localParams_.gmmXDtype) + ", " + Ops::Base::ToString(localParams_.gmmWeightDtype))
                .c_str(),
            "When the quantization mode is MXFP4, the dtypes of gmmX and gmmWeight must be the same"),
        return ge::GRAPH_FAILED);
    OP_LOGD(opName_, "MXFP4 mode enabled: gmmXDtype=%s", Ops::Base::ToString(localParams_.gmmXDtype).c_str());
    // non-transposed: weight shape is(e,H1,N1)，N must be even
    if (!localParams_.isGmmWeightTrans) {
        OP_TILING_CHECK(localParams_.N1 % 2 != 0,
                        OP_LOGE_FOR_INVALID_VALUE(opName_, "N1", std::to_string(localParams_.N1),
                                                  "must be even in mxfp4 non-transposed weight scenario"),
                        return ge::GRAPH_FAILED);
    }
    // H1 must be even and not equal to 2
    OP_TILING_CHECK((localParams_.H1 % 2 != 0) || (localParams_.H1 == 2),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "H1", std::to_string(localParams_.H1),
                                              "In mxfp4 scenario, H1 must be even and not equal to 2"),
                    return ge::GRAPH_FAILED);
    // exist mm
    if (localParams_.hasSharedMm) {
        // non-transposed: weight shape is(H2,N2)
        if (!localParams_.isMmWeightTrans) {
            OP_TILING_CHECK(localParams_.N2 % 2 != 0,
                            OP_LOGE_FOR_INVALID_VALUE(opName_, "N2", std::to_string(localParams_.N2),
                                                      "must be even in mxfp4 non-transposed weight scenario"),
                            return ge::GRAPH_FAILED);
        }
        // H2 must be even and not equal to 2
        OP_TILING_CHECK((localParams_.H2 % 2 != 0) || (localParams_.H2 == 2),
                        OP_LOGE_FOR_INVALID_VALUE(opName_, "H2", std::to_string(localParams_.H2),
                                                  "In mxfp4 scenario, H2 must be even and not equal to 2"),
                        return ge::GRAPH_FAILED);
    }
    OP_LOGD(opName_, "MXFP4 specific constraints check passed.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetInputOutputInfo()
{
    auto status = CheckOpInputSingleParamsTensor();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckParamsRelationAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckMxQuantDtypeConstraints();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckMxQuantGmmScaleShapes();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckMxQuantMmScaleShapes();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckMxfp4SpecificConstraints();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t MxQuantGroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    uint8_t commMode = 0;
    if (QuantGroupedMatmulAllToAllvTilingCommon::QuantGetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans,
                                                  localParams_.isMmWeightTrans, commMode);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, commMode, tilingKey);
    return tilingKey;
}

// 注册tiling类
REGISTER_OPS_TILING_TEMPLATE(QuantGroupedMatMulAlltoAllv, MxQuantGroupedMatmulAllToAllvTiling, 1);

} // namespace Mc2Tiling
