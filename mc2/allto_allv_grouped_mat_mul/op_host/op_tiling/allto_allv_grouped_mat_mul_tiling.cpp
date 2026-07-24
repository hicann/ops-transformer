/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_tiling.cc
 * \brief
 */

#include "allto_allv_grouped_mat_mul_tiling.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include "../../../allto_allv_quant_grouped_mat_mul/op_host/op_tiling/allto_allv_quant_grouped_mat_mul_tiling_base.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling_key.h"
#include "mc2_comm_utils.h"

namespace optiling {
constexpr uint32_t SINGLE_GROUP_NUM = 1;
constexpr uint32_t GMM_ACT_TYPE_NONE = 0;

constexpr uint32_t GMM_ARRAY_MAX_NUM = 128U;
constexpr uint32_t DEFAULT_MERGED_EXPERT_NUM = 4U;
constexpr uint64_t PER_RANK_TOTAL_MN_THRESHOLD = 20UL * 1024UL * 1024UL;
constexpr uint32_t SMALL_EXPERT_THRESHOLD = 4U;
constexpr uint32_t ORIGINAL_LOOP_THRESHOLD = 4U;
constexpr uint32_t MIN_LOOP_COUNT = 2U;

static uint32_t CalcExpertNum(uint64_t e, uint64_t epWorldSize, uint64_t bsk, uint64_t n1,
                              uint32_t packFactor = 1U)
{
    if (e == 0 || e == 1) {
        return 1U;
    }
    if (e <= SMALL_EXPERT_THRESHOLD) {
        return 1U;
    }
    uint32_t originalLoopCount = static_cast<uint32_t>((e + DEFAULT_MERGED_EXPERT_NUM - 1U)
        / DEFAULT_MERGED_EXPERT_NUM);
    if (originalLoopCount <= ORIGINAL_LOOP_THRESHOLD) {
        return DEFAULT_MERGED_EXPERT_NUM;
    }

    uint64_t perRankTokens = (epWorldSize == 0) ? bsk : bsk / epWorldSize;
    uint64_t perRankTotalMN = (packFactor == 0) ? perRankTokens * n1
        : perRankTokens * n1 / packFactor;

    uint32_t expertNum = DEFAULT_MERGED_EXPERT_NUM;
    if (perRankTotalMN < PER_RANK_TOTAL_MN_THRESHOLD) {
        uint32_t upperByArray = (epWorldSize == 0) ? GMM_ARRAY_MAX_NUM
            : static_cast<uint32_t>(GMM_ARRAY_MAX_NUM / epWorldSize);
        uint32_t upperByLoop = static_cast<uint32_t>(e / MIN_LOOP_COUNT);
        uint32_t upperBound = std::min({static_cast<uint32_t>(e), upperByArray, upperByLoop});
        expertNum = std::max(upperBound, 1U);
    }
    return expertNum;
}

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
constexpr uint32_t OUTPUT_Y_INDEX = 0U;
constexpr uint32_t NUM_ZERO = 0;
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t E_MAX_VALUE_NON_QUANT = 48;
constexpr uint32_t MAX_BSK = 52428800;
constexpr uint32_t MAX_SHAPE_SIZE = 65536;
static const size_t ATTR_COMM_MODE_INDEX = 7;

static inline bool IsShapePresent(const gert::StorageShape *shape)
{
    if (shape == nullptr) {
        return false;
    }
    const auto &storageShape = shape->GetStorageShape();
    return storageShape.GetDimNum() > 0;
}

ge::graphStatus AlltoAllvGmmTiling::GetContextAttr(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK((groupEpPtr == nullptr) || (epWorldSizePtr == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "group or epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*epWorldSizePtr == 0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epWorldSize",
                                              std::to_string(*epWorldSizePtr).c_str(), "positive value"),
                    return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    std::vector<int64_t> validEpWorldSizeValues;
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        validEpWorldSizeValues = {2, 4, 8, 16, 32, 64, 128, 256};
    } else {
        validEpWorldSizeValues = {8, 16, 32, 64, 128};
    }
    OP_TILING_CHECK(std::find(validEpWorldSizeValues.begin(), validEpWorldSizeValues.end(), *epWorldSizePtr) ==
                        validEpWorldSizeValues.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epWorldSize",
                                              std::to_string(*epWorldSizePtr).c_str(), "valid set for this platform"),
                    return ge::GRAPH_FAILED);

    auto mmXStorageShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX);
    hasSharedExpertFlag_ = IsShapePresent(mmXStorageShape);

    auto permuteOutFlagPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX);
    OP_TILING_CHECK(permuteOutFlagPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "permuteOutFlag"),
                    return ge::GRAPH_FAILED);
    tilingData->isPermuteOut = *permuteOutFlagPtr;

    groupPtr_ = groupEpPtr;
    group_ = groupEpPtr;
    epWorldSizePtr_ = epWorldSizePtr;

    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(transGmmWeightPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "transGmmWeight"),
                    return ge::GRAPH_FAILED);
    transGmmWeight_ = *transGmmWeightPtr;

    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(transMmWeightPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "transMmWeight"),
                    return ge::GRAPH_FAILED);
    transMmWeight_ = hasSharedExpertFlag_ ? *transMmWeightPtr : false;

    OP_LOGI(context_->GetNodeName(), "EpGroup is %s, epWorldSize is %lld.", groupPtr_, *epWorldSizePtr_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::GetShapeAndFormat(const gert::TilingContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMKN(const gert::TilingContext *context)
{
    (void)context;
    OP_TILING_CHECK(mmDataTypeSize == 0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "matmul dtype size",
                                              std::to_string(mmDataTypeSize).c_str(), "positive size"),
                    return ge::GRAPH_FAILED);
    uint32_t numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize;
    OP_TILING_CHECK(numInOneBlk == 0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "numInOneBlk",
                                              std::to_string(numInOneBlk).c_str(), "positive value"),
                    return ge::GRAPH_FAILED);
    int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
    OP_TILING_CHECK(
        maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
        OP_LOGE_FOR_INVALID_VALUE(
            context_->GetNodeName(), "Gmm M, N, K",
            (std::string(std::to_string(maxM_)) + ", " + std::to_string(maxN_) + ", " + std::to_string(maxK_)).c_str(),
            "within 32B-aligned max range"),
        return ge::GRAPH_FAILED);
    if (IsShapePresent(context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        OP_TILING_CHECK(maxMForMM_ > maxMKN || maxNForMM_ > maxMKN || maxKForMM_ > maxMKN,
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "Mm M, N, K",
                                                  (std::string(std::to_string(maxMForMM_)) + ", " +
                                                   std::to_string(maxNForMM_) + ", " + std::to_string(maxKForMM_))
                                                      .c_str(),
                                                  "within 32B-aligned max range"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckSendRecvDataVolumn(const gert::TilingContext *context) const
{
    uint64_t eExpert = tilingData->taskTilingInfo.e;
    uint64_t epWorldSize = tilingData->taskTilingInfo.epWorldSize;

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts or recvCounts"),
                    return ge::GRAPH_FAILED);

    const int64_t *sendCountsLocal = static_cast<const int64_t *>(sendCountsPtr->GetData());
    const int64_t *recvCountsLocal = static_cast<const int64_t *>(recvCountsPtr->GetData());
    uint64_t recvSum = 0U;
    uint64_t sendSum = 0U;
    uint64_t H1 = tilingData->taskTilingInfo.H1;
    uint64_t bsk = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    uint64_t a = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (NeedToCheckCounts()) {
        for (uint64_t i = 1U; i <= epWorldSize; i++) {
            recvSum = 0U;
            sendSum = 0U;
            for (uint64_t j = (i - 1U) * eExpert; j <= i * eExpert - 1U; j++) {
                OP_TILING_CHECK(
                    (sendCountsLocal[j] < NUM_ZERO) || (sendCountsLocal[j] > static_cast<int64_t>(bsk)),
                    OP_LOGE_FOR_INVALID_VALUE(
                        context_->GetNodeName(), "sendCounts",
                        (std::string("sendCounts[") + std::to_string(j) + "]=" + std::to_string(sendCountsLocal[j]))
                            .c_str(),
                        (std::string("in range [0, ") + std::to_string(bsk) + "]").c_str()),
                    return ge::GRAPH_FAILED);
                OP_TILING_CHECK(
                    (recvCountsLocal[j] < NUM_ZERO) || (recvCountsLocal[j] > static_cast<int64_t>(a)),
                    OP_LOGE_FOR_INVALID_VALUE(
                        context_->GetNodeName(), "recvCounts",
                        (std::string("recvCounts[") + std::to_string(j) + "]=" + std::to_string(recvCountsLocal[j]))
                            .c_str(),
                        (std::string("in range [0, ") + std::to_string(a) + "]").c_str()),
                    return ge::GRAPH_FAILED);
                recvSum += static_cast<uint64_t>(recvCountsLocal[j]) * H1 * 2U;
                sendSum += static_cast<uint64_t>(sendCountsLocal[j]) * H1 * 2U;
            }
        }
    }

    uint64_t sendCountsSum = std::accumulate(sendCountsLocal, sendCountsLocal + sendCountsSize, 0ULL);
    OP_TILING_CHECK(sendCountsSum != bsk,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sendCounts",
                                              (std::string("sum=") + std::to_string(sendCountsSum)).c_str(),
                                              (std::string("BSK=") + std::to_string(bsk)).c_str()),
                    return ge::GRAPH_FAILED);
    uint64_t recvCountsSum = std::accumulate(recvCountsLocal, recvCountsLocal + recvCountsSize, 0ULL);
    OP_TILING_CHECK(recvCountsSum != a,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "recvCounts",
                                              (std::string("sum=") + std::to_string(recvCountsSum)).c_str(),
                                              (std::string("A=") + std::to_string(a)).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeSize(const gert::TilingContext *context) const
{
    OP_TILING_CHECK((context->GetInputShape(GMM_X_INDEX) == nullptr) ||
                        (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmX or gmmWeight"), return ge::GRAPH_FAILED);

    uint64_t BSK = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    if (BSK <= NUM_ZERO || BSK >= MAX_BSK) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "BSK", std::to_string(BSK).c_str(),
                                  (std::string("[1, ") + std::to_string(MAX_BSK) + ")").c_str());
        return ge::GRAPH_FAILED;
    }

    uint64_t H1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    if (H1 <= NUM_ZERO || H1 >= MAX_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "H1", std::to_string(H1).c_str(),
                                  (std::string("[1, ") + std::to_string(MAX_SHAPE_SIZE) + ")").c_str());
        return ge::GRAPH_FAILED;
    }

    uint64_t N1 = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
    if (N1 <= NUM_ZERO || N1 >= MAX_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "N1", std::to_string(N1).c_str(),
                                  (std::string("[1, ") + std::to_string(MAX_SHAPE_SIZE) + ")").c_str());
        return ge::GRAPH_FAILED;
    }

    if (IsShapePresent(context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        uint64_t BS = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape().GetDim(0);
        if (BS <= NUM_ZERO || BS >= MAX_BSK) {
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "BS", std::to_string(BS).c_str(),
                                      (std::string("[1, ") + std::to_string(MAX_BSK) + ")").c_str());
            return ge::GRAPH_FAILED;
        }
        uint64_t H2 = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape().GetDim(1);
        if (H2 <= NUM_ZERO || H2 > H2_MAX_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "H2", std::to_string(H2).c_str(),
                                      (std::string("[1, ") + std::to_string(H2_MAX_VALUE) + "]").c_str());
            return ge::GRAPH_FAILED;
        }
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        uint64_t N2 = context->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape().GetDim(n2DimIndex);
        if (N2 <= NUM_ZERO || N2 >= MAX_SHAPE_SIZE) {
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "N2", std::to_string(N2).c_str(),
                                      (std::string("[1, ") + std::to_string(MAX_SHAPE_SIZE) + ")").c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeSize(const gert::TilingContext *context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts"),
                    return ge::GRAPH_FAILED);

    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts"),
                    return ge::GRAPH_FAILED);

    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    if (sendCountsSize != recvCountsSize) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sendCounts",
                                  (std::string("sendCountsSize=") + std::to_string(sendCountsSize) +
                                   ", recvCountsSize=" + std::to_string(recvCountsSize))
                                      .c_str(),
                                  "recvCountsSize");
        return ge::GRAPH_FAILED;
    }

    if (sendCountsSize > MC2KernelTemplate::MAX_EXPERT_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "sendCounts", std::to_string(sendCountsSize).c_str(),
                                  (std::string("<=") + std::to_string(MC2KernelTemplate::MAX_EXPERT_SIZE)).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeRelation(const gert::TilingContext *context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "epWorldSize"),
                    return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts"),
                    return ge::GRAPH_FAILED);

    const int64_t *sendCountsAttr = static_cast<const int64_t *>(sendCountsPtr->GetData());
    OP_TILING_CHECK(sendCountsAttr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts data"),
                    return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t eExpert = sendCountsSize / (*epWorldSizePtr);

    if (sendCountsSize != e_ * (*epWorldSizePtr)) {
        OP_LOGE_FOR_INVALID_VALUE(
            context_->GetNodeName(), "sendCounts", (std::string("size=") + std::to_string(sendCountsSize)).c_str(),
            (std::string("e*epWorldSize=") + std::to_string(e_) + "*" + std::to_string(*epWorldSizePtr)).c_str());
        return ge::GRAPH_FAILED;
    }

    if (eExpert <= NUM_ZERO || eExpert > E_MAX_VALUE_NON_QUANT) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "eExpert", std::to_string(eExpert).c_str(),
                                  (std::string("[1, ") + std::to_string(E_MAX_VALUE_NON_QUANT) + "]").c_str());
        return ge::GRAPH_FAILED;
    }

    for (uint64_t i = 0; i < sendCountsSize; i++) {
        if (sendCountsAttr[i] < NUM_ZERO) {
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "sendCounts",
                (std::string("sendCounts[") + std::to_string(i) + "]=" + std::to_string(sendCountsAttr[i])).c_str(),
                "non-negative");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeRelation(const gert::TilingContext *context) const
{
    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    uint64_t H1 = gmmXShape.GetDim(1);
    uint64_t h1DimIndex = transGmmWeight_ ? DIM_TWO : DIM_ONE;
    uint64_t H2 = gmmWeightShape.GetDim(h1DimIndex);
    if (H1 != H2) {
        OP_LOGE_FOR_INVALID_VALUE(
            context_->GetNodeName(), "gmmX and gmmWeight",
            (std::string("H1=") + std::to_string(H1) + ", gmmWeight_H1=" + std::to_string(H2)).c_str(),
            "H1 should be equal");
        return ge::GRAPH_FAILED;
    }

    // Check mmX/mmWeight/mmY consistency
    auto mmXStorageShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX);
    auto mmWeightStorageShape = context->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX);
    auto outputMmYStorageShape = context->GetOutputShape(OUTPUT_MM_Y_INDEX);
    // If mmY is present, mmX and mmWeight must also be present
    if (IsShapePresent(outputMmYStorageShape)) {
        OP_TILING_CHECK(!IsShapePresent(mmXStorageShape), OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmX"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!IsShapePresent(mmWeightStorageShape),
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmWeight"), return ge::GRAPH_FAILED);
    }

    // Check gmmY output: N1 dimension should match gmmWeight
    OP_TILING_CHECK(context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmY"), return ge::GRAPH_FAILED);
    auto gmmYShape = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape();
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    uint64_t N1 = gmmWeightShape.GetDim(n1DimIndex);
    uint64_t gmmYDim1 = gmmYShape.GetDim(DIM_ONE);
    if (gmmYDim1 != N1) {
        OP_LOGE_FOR_INVALID_VALUE(
            context_->GetNodeName(), "gmmY and gmmWeight",
            (std::string("gmmY_N1=") + std::to_string(gmmYDim1) + ", gmmWeight_N1=" + std::to_string(N1)).c_str(),
            "N1 should be equal");
        return ge::GRAPH_FAILED;
    }

    // Check permuteOut output when permuteOutFlag is true
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    auto permuteOutFlagPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX);
    if (permuteOutFlagPtr != nullptr && *permuteOutFlagPtr) {
        auto permuteOutStorageShape = context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX);
        OP_TILING_CHECK(!IsShapePresent(permuteOutStorageShape),
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "permuteOut"), return ge::GRAPH_FAILED);
        auto permuteOutShape = permuteOutStorageShape->GetStorageShape();
        uint64_t permuteOutA = permuteOutShape.GetDim(DIM_ZERO);
        uint64_t gmmYA = gmmYShape.GetDim(DIM_ZERO);
        if (permuteOutA != gmmYA) {
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "permuteOut and gmmY",
                (std::string("A=") + std::to_string(permuteOutA) + ", gmmY_A=" + std::to_string(gmmYA)).c_str(),
                "A should be equal");
            return ge::GRAPH_FAILED;
        }
        uint64_t permuteOutH1 = permuteOutShape.GetDim(DIM_ONE);
        if (permuteOutH1 != H1) {
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "permuteOut and gmmX",
                (std::string("H1=") + std::to_string(permuteOutH1) + ", gmmX_H1=" + std::to_string(H1)).c_str(),
                "H1 should be equal");
            return ge::GRAPH_FAILED;
        }
    } else if (permuteOutFlagPtr != nullptr && !(*permuteOutFlagPtr)) {
        auto permuteOutStorageShape = context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX);
        OP_TILING_CHECK(IsShapePresent(permuteOutStorageShape),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "permuteOut", "present",
                                                              "The value of permuteOut must be nullptr."),
                        return ge::GRAPH_FAILED);
    }

    if (hasSharedExpertFlag_) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        auto mmWeightShape = context->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t BS = mmXShape.GetDim(DIM_ZERO);
        uint64_t H = mmXShape.GetDim(1);
        uint64_t h2DimIndex = transMmWeight_ ? DIM_ONE : DIM_ZERO;
        uint64_t H2MM = mmWeightShape.GetDim(h2DimIndex);
        if (H != H2MM) {
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "mmX and mmWeight",
                (std::string("H2=") + std::to_string(H) + ", mmWeight_H2=" + std::to_string(H2MM)).c_str(),
                "H2 should be equal");
            return ge::GRAPH_FAILED;
        }

        uint64_t BSK = gmmXShape.GetDim(0);
        if (BSK % BS != 0) {
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "BSK and BS",
                (std::string("BSK=") + std::to_string(BSK) + ", BS=" + std::to_string(BS)).c_str(),
                "BSK should be divisible by BS");
            return ge::GRAPH_FAILED;
        }
        uint64_t K = BSK / BS;
        if (K < K_MIN_VALUE || K > K_MAX_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "K", std::to_string(K).c_str(),
                (std::string("[") + std::to_string(K_MIN_VALUE) + ", " + std::to_string(K_MAX_VALUE) + "]").c_str());
            return ge::GRAPH_FAILED;
        }

        // Check mmY output when mmX is present
        OP_TILING_CHECK(!IsShapePresent(context->GetOutputShape(OUTPUT_MM_Y_INDEX)),
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmY"), return ge::GRAPH_FAILED);
        auto mmYShape = context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape();
        uint64_t mmYBS = mmYShape.GetDim(DIM_ZERO);
        if (mmYBS != BS) {
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "mmY and mmX",
                (std::string("BS=") + std::to_string(mmYBS) + ", mmX_BS=" + std::to_string(BS)).c_str(),
                "BS should be equal");
            return ge::GRAPH_FAILED;
        }
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        uint64_t N2 = mmWeightShape.GetDim(n2DimIndex);
        uint64_t mmYDim1 = mmYShape.GetDim(DIM_ONE);
        if (mmYDim1 != N2) {
            OP_LOGE_FOR_INVALID_VALUE(
                context_->GetNodeName(), "mmY and mmWeight",
                (std::string("N2=") + std::to_string(mmYDim1) + ", mmWeight_N2=" + std::to_string(N2)).c_str(),
                "N2 should be equal");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeDims(const gert::TilingContext *context)
{
    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    OP_TILING_CHECK(gmmXShape.GetDimNum() != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "gmmX",
                                                 (std::to_string(gmmXShape.GetDimNum()) + "D").c_str(), "2D"),
                    return ge::GRAPH_FAILED);
    uint64_t gmmXDim0 = gmmXShape.GetDim(0);
    OP_TILING_CHECK(gmmXDim0 > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "gmmX", std::to_string(gmmXDim0).c_str(),
                                              "less than 2147483647"),
                    return ge::GRAPH_FAILED);
    maxM_ = static_cast<int32_t>(gmmXDim0);
    uint64_t gmmXDim1 = gmmXShape.GetDim(1);
    maxK_ = static_cast<int32_t>(gmmXDim1);

    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    OP_TILING_CHECK(gmmWeightShape.GetDimNum() != DIM_THREE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "gmmWeight",
                                                 (std::to_string(gmmWeightShape.GetDimNum()) + "D").c_str(), "3D"),
                    return ge::GRAPH_FAILED);
    e_ = gmmWeightShape.GetDim(0);
    uint64_t h1DimIndex = transGmmWeight_ ? DIM_TWO : DIM_ONE;
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    maxK_ = static_cast<int32_t>(gmmWeightShape.GetDim(h1DimIndex));
    maxN_ = static_cast<int32_t>(gmmWeightShape.GetDim(n1DimIndex));

    mmXDataType_ = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    gmmXDataType_ = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    gmmWeightDataType_ = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    mmDataTypeSize = GetSizeByDataType(mmXDataType_);

    if (IsShapePresent(context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        OP_TILING_CHECK(mmXShape.GetDimNum() != DIM_TWO,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "mmX",
                                                     (std::to_string(mmXShape.GetDimNum()) + "D").c_str(), "2D"),
                        return ge::GRAPH_FAILED);
        uint64_t mmXDim0 = mmXShape.GetDim(0);
        uint64_t mmXDim1 = mmXShape.GetDim(1);
        OP_TILING_CHECK(mmXDim0 > static_cast<uint64_t>(INT32_MAX),
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmX", std::to_string(mmXDim0).c_str(),
                                                  "less than 2147483647"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mmXDim1 > static_cast<uint64_t>(INT32_MAX),
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmX", std::to_string(mmXDim1).c_str(),
                                                  "less than 2147483647"),
                        return ge::GRAPH_FAILED);
        maxMForMM_ = static_cast<int32_t>(mmXDim0);
        maxKForMM_ = static_cast<int32_t>(mmXDim1);
        mmXDataType_ = context->GetOptionalInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
        mmWeightDataType_ = context->GetOptionalInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();

        auto mmWeightShape = context->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        OP_TILING_CHECK(mmWeightShape.GetDimNum() != DIM_TWO,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "mmWeight",
                                                     (std::to_string(mmWeightShape.GetDimNum()) + "D").c_str(), "2D"),
                        return ge::GRAPH_FAILED);
        uint64_t kDimIndex = transMmWeight_ ? DIM_ONE : DIM_ZERO;
        uint64_t nDimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        uint64_t mmKDim = mmWeightShape.GetDim(kDimIndex);
        uint64_t mmNDim = mmWeightShape.GetDim(nDimIndex);
        OP_TILING_CHECK(mmKDim > static_cast<uint64_t>(INT32_MAX),
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmWeight", std::to_string(mmKDim).c_str(),
                                                  "less than 2147483647"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mmNDim > static_cast<uint64_t>(INT32_MAX),
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "mmWeight", std::to_string(mmNDim).c_str(),
                                                  "less than 2147483647"),
                        return ge::GRAPH_FAILED);
        maxKForMM_ = static_cast<int32_t>(mmKDim);
        maxNForMM_ = static_cast<int32_t>(mmNDim);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckDType(const gert::TilingContext *context) const
{
    ge::DataType gmmXDtype = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    ge::DataType gmmWeightDtype = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    if (gmmXDtype != gmmWeightDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "gmmX and gmmWeight",
            (Ops::Base::ToString(gmmXDtype) + " and " + Ops::Base::ToString(gmmWeightDtype)).c_str(),
            "The dtypes of gmmX and gmmWeight must be the same");
        return ge::GRAPH_FAILED;
    }

    if (gmmXDtype != ge::DT_FLOAT16 && gmmXDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmX", Ops::Base::ToString(gmmXDtype).c_str(),
                                  "FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }

    if (IsShapePresent(context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        ge::DataType mmXDtype = context->GetOptionalInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
        ge::DataType mmWeightDtype = context->GetOptionalInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();
        if (mmXDtype != mmWeightDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "mmX and mmWeight",
                (Ops::Base::ToString(mmXDtype) + " and " + Ops::Base::ToString(mmWeightDtype)).c_str(),
                "The dtypes of mmX and mmWeight must be the same");
            return ge::GRAPH_FAILED;
        }

        if (mmXDtype != ge::DT_FLOAT16 && mmXDtype != ge::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "mmX", Ops::Base::ToString(mmXDtype).c_str(),
                                      "FLOAT16 or BF16");
            return ge::GRAPH_FAILED;
        }

        if (gmmXDtype != mmXDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "gmmX and mmX",
                (Ops::Base::ToString(gmmXDtype) + " and " + Ops::Base::ToString(mmXDtype)).c_str(),
                "The dtypes of gmmX and mmX must be the same");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMmShapeDims(const gert::TilingContext *context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::GetAndConvertCommMode(gert::TilingContext *context, uint8_t &commMode) const
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    const char *commModeStr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE_INDEX);
    OP_TILING_CHECK(commModeStr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "comm_mode"),
                    return ge::GRAPH_FAILED);
    const size_t maxLength = 7UL;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
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

ge::graphStatus AlltoAllvGmmTiling::SetHcclTiling(const gert::TilingContext *context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);

    auto tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);

    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    const uint32_t alltoAllvReduceType = 0u;

    mc2tiling::HcclDataType alltoallvHcclDataType =
        mc2tiling::ConvertGeTypeToHcclType(context_->GetNodeName(), gmmXDataType_);
    OP_TILING_CHECK(alltoallvHcclDataType == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gmmXDataType",
                                              Ops::Base::ToString(gmmXDataType_).c_str(), "supported HCCL data types"),
                    return ge::GRAPH_FAILED);
    uint8_t alltoAllvDataType = static_cast<uint8_t>(alltoallvHcclDataType);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType, alltoAllvDataType,
                                         alltoAllvDataType);
    uint8_t commMode = 0;
    if (GetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "CommMode is %u.", commMode);
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commMode == Mc2Comm::COMM_MODE_CCU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
                    OP_LOGE(context_->GetNodeName(), "HCCL init tiling config failed, expected success."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
                    OP_LOGE(context_->GetNodeName(), "HCCL alltoallv tiling config failed, expected success."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::setNumBlocks(gert::TilingContext *context)
{
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Failed to get common platform info.");
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Common platform info check failed, expected valid UB/L1/L0C sizes.");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t numBlocks = mc2tiling::GetNumBlocks(aicNum, aivNum, context_->GetNodeName());

    tilingData->taskTilingInfo.ubSize = ubSize_;
    context->SetBlockDim(static_cast<uint32_t>(numBlocks));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::Init(gert::TilingContext *context)
{
    context_ = context;
    tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OP_TILING_CHECK(GetContextAttr(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to get context attr."), return ge::GRAPH_FAILED);

    if (IsShapePresent(context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        OP_TILING_CHECK(!IsShapePresent(context_->GetOutputShape(OUTPUT_MM_Y_INDEX)),
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "mmY"), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(CheckShapeDims(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Shape dim check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDType(context_) != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Dtype check failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckShapeRelation(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Shape relation check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetShapeAndFormat(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to get shape and format."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckShapeSize(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Shape size check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrsShapeSize(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Attrs shape size check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrsShapeRelation(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Attrs shape relation check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSendRecvDataVolumn(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Send/recv data volume check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMKN(context_) != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "MKN check failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMmShapeDims(context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Mm shape dims check failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmTiling::GetTilingKey() const
{
    uint8_t commMode = 0;
    auto platformInfo = context_->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        if (GetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        commMode = Mc2Comm::COMM_MODE_AICPU;
    }
    bool tilingekyGmmTrans = transGmmWeight_;
    bool tilingekyMmTrans = transMmWeight_;
    uint64_t tilingKey = GET_TPL_TILING_KEY(tilingekyGmmTrans, tilingekyMmTrans, commMode);
    OP_LOGD(context_->GetNodeName(), "TilingKey is %llu, gmmTrans=%d, mmTrans=%d, commMode=%u.", tilingKey,
            tilingekyGmmTrans, tilingekyMmTrans, commMode);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmTiling::RunFusionKernelTiling(gert::TilingContext *context)
{
    OP_TILING_CHECK(SetHcclTiling(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "HCCL tiling config failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(setNumBlocks(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Set numBlocks failed."), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*epWorldSizePtr == 0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epWorldSize",
                                              std::to_string(*epWorldSizePtr).c_str(), "positive value"),
                    return ge::GRAPH_FAILED);
    epWorldSize_ = *epWorldSizePtr;

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts or recvCounts"),
                    return ge::GRAPH_FAILED);

    sendCounts = static_cast<const int64_t *>(sendCountsPtr->GetData());
    OP_TILING_CHECK(sendCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts data"),
                    return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t *>(recvCountsPtr->GetData());
    OP_TILING_CHECK(recvCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts data"),
                    return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    e_ = sendCountsSize / epWorldSize_;

    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    bsk_ = gmmXShape.GetDim(0);
    h1_ = gmmXShape.GetDim(1);

    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK((transGmmWeightPtr == nullptr) || (transMmWeightPtr == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "transGmmWeight or transMmWeight"),
                    return ge::GRAPH_FAILED);
    transGmmWeight_ = *transGmmWeightPtr;
    transMmWeight_ = IsShapePresent(context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)) ? *transMmWeightPtr : false;

    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    e_ = gmmWeightShape.GetDim(0);
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    n1_ = gmmWeightShape.GetDim(n1DimIndex);

    auto gmmYShape = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape();
    a_ = gmmYShape.GetDim(0);

    if (IsShapePresent(context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        bs_ = mmXShape.GetDim(0);
        h2_ = mmXShape.GetDim(1);
        hasSharedExpertFlag_ = true;
        auto mmWeightShape = context->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        n2_ = mmWeightShape.GetDim(n2DimIndex);
    }

    OP_TILING_CHECK(DoAiCoreTiling(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "AI core tiling failed."), return ge::GRAPH_FAILED);

    uint64_t tilingKey = GetTilingKey();
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::DoAiCoreTiling(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts"),
                    return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t *>(recvCountsPtr->GetData());
    OP_TILING_CHECK(recvCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts data"),
                    return ge::GRAPH_FAILED);
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    uint64_t maxMSize = 0;
    OP_TILING_CHECK(
        e_ * epWorldSize_ > recvCountsSize,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "e * epWorldSize",
                                              (std::to_string(e_) + " * " + std::to_string(epWorldSize_)).c_str(),
                                              "should not exceed recvCounts size"),
        return ge::GRAPH_FAILED);
    for (uint64_t expertIdx = 0; expertIdx < e_; expertIdx++) {
        uint64_t mSize = 0;
        for (uint64_t rankIdx = 0; rankIdx < epWorldSize_; rankIdx++) {
            mSize += recvCounts[rankIdx * e_ + expertIdx];
        }
        maxMSize = std::max(mSize, maxMSize);
    }

    if (maxMSize != 0) {
        uint32_t expertNum = CalcExpertNum(e_, epWorldSize_, bsk_, n1_);
        uint32_t gmmGroupNum = expertNum * static_cast<uint32_t>(epWorldSize_);
        AlltoAllvGmmTilingHelper gmmHelper(*this);
        MC2_CHECK_LOG_RET(context_->GetNodeName(),
                          gmmHelper.SetInputParams(maxMSize, n1_, h1_, transGmmWeight_, gmmXDataType_,
                                                   gmmWeightDataType_, gmmXDataType_, gmmGroupNum));
        MC2_CHECK_LOG_RET(context_->GetNodeName(), gmmHelper.Process());
        tilingData->gmmQuantTilingData = gmmHelper.GetAlltoAllvQuantHelperData();
    }

    if (bs_ != 0) {
        AlltoAllvGmmTilingHelper mmHelper(*this);
        MC2_CHECK_LOG_RET(context_->GetNodeName(), mmHelper.SetInputParams(bs_, n2_, h2_, transMmWeight_, mmXDataType_,
                                                                           mmWeightDataType_, mmXDataType_,
                                                                           SINGLE_GROUP_NUM));
        MC2_CHECK_LOG_RET(context_->GetNodeName(), mmHelper.Process());
        tilingData->mmQuantTilingData = mmHelper.GetAlltoAllvQuantHelperData();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingHelper::SetInputParams(uint64_t M, uint64_t N, uint64_t K, bool transB,
                                                         ge::DataType aDtype, ge::DataType bDtype, ge::DataType cDtype,
                                                         uint32_t groupNum)
{
    GetPlatformInfo();
    inputParams_.opName = context_->GetNodeName();
    inputParams_.kernelType = 0UL;
    inputParams_.splitItem = 0;
    inputParams_.actType = GMM_ACT_TYPE_NONE;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = false;
    inputParams_.transB = transB;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = (groupNum > SINGLE_GROUP_NUM) ? true : false;
    inputParams_.isSingleW = (groupNum > SINGLE_GROUP_NUM) ? true : false;
    inputParams_.isSingleY = (groupNum > SINGLE_GROUP_NUM) ? true : false;

    inputParams_.mSize = M;
    inputParams_.kSize = K;
    inputParams_.nSize = N;
    inputParams_.groupNum = groupNum;
    inputParams_.aQuantMode = Mc2GroupedMatmulTiling::QuantMode::DEFAULT;
    inputParams_.bQuantMode = Mc2GroupedMatmulTiling::QuantMode::DEFAULT;
    inputParams_.groupType = Mc2GroupedMatmul::SPLIT_M;
    inputParams_.groupListType = 1;

    inputParams_.aDtype = aDtype;
    inputParams_.bDtype = bDtype;
    inputParams_.cDtype = cDtype;
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = ge::DT_FLOAT;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT;

    OP_TILING_CHECK(M > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "M", std::to_string(M).c_str(),
                                                          "should be less than 2147483647"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(K > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "K", std::to_string(K).c_str(),
                                                          "should be less than 2147483647"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(N > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "N", std::to_string(N).c_str(),
                                                          "should be less than 2147483647"),
                    return ge::GRAPH_FAILED);
    constexpr uint32_t MAX_TENSOR_CONT = 128U;
    OP_TILING_CHECK(groupNum > MAX_TENSOR_CONT,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "groupNum",
                        std::to_string(groupNum).c_str(),
                        (std::string("<=") + std::to_string(MAX_TENSOR_CONT)).c_str()),
                    return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < groupNum; i++) {
        mList_[i] = static_cast<int32_t>(M);
    }
    kList_[0] = static_cast<int32_t>(K);
    nList_[0] = static_cast<int32_t>(N);
    SetKernelType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingHelper::Process()
{
    MC2_CHECK_LOG_RET(context_->GetNodeName(), DoOpTiling());
    MC2_CHECK_LOG_RET(context_->GetNodeName(), DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmTilingStruct::GetTilingKey() const
{
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(context_->GetNodeName(), "TilingKey is %llu.", tilingKey);
    return tilingKey;
}

bool AlltoAllvGmmTilingStruct::IsCapable()
{
    return true;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetPlatformInfo()
{
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    groupPtr_ = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(groupPtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "group"),
                    return ge::GRAPH_FAILED);
    group_ = groupPtr_;
    epWorldSizePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*epWorldSizePtr_ == 0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "epWorldSize",
                                              std::to_string(*epWorldSizePtr_).c_str(), "positive value"),
                    return ge::GRAPH_FAILED);
    epWorldSize_ = *epWorldSizePtr_;
    sendCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts"),
                    return ge::GRAPH_FAILED);
    sendCounts = static_cast<const int64_t *>(sendCountsPtr_->GetData());
    OP_TILING_CHECK(sendCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts data"),
                    return ge::GRAPH_FAILED);
    recvCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts"),
                    return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t *>(recvCountsPtr_->GetData());
    OP_TILING_CHECK(recvCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts data"),
                    return ge::GRAPH_FAILED);
    transGmmWeightPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    if (transGmmWeightPtr_ != nullptr) {
        transGmmWeight_ = *transGmmWeightPtr_;
    }
    transMmWeightPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    if (transMmWeightPtr_ != nullptr) {
        transMmWeight_ = *transMmWeightPtr_;
    }
    if (!IsShapePresent(context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        transMmWeight_ = false;
    }
    permuteOutFlagPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX);
    if (permuteOutFlagPtr_ != nullptr) {
        permuteOutFlag_ = *permuteOutFlagPtr_;
    }

    if (IsShapePresent(context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        auto mmXShape = context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        bs_ = mmXShape.GetDim(0);
        h2_ = mmXShape.GetDim(1);
        hasSharedExpertFlag_ = true;
        mmXDataType_ = context_->GetOptionalInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
        mmWeightDataType_ = context_->GetOptionalInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();
        auto mmWeightShape = context_->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        n2_ = mmWeightShape.GetDim(n2DimIndex);
    } else {
        transMmWeight_ = false;
    }

    constexpr uint32_t SEND_COUNTS_TENSOR_INDEX = 2U;
    constexpr uint32_t RECV_COUNTS_TENSOR_INDEX = 3U;
    OP_TILING_CHECK(IsShapePresent(context_->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "sendCounts input", "present",
                                                          "The value of sendCounts input must be nullptr."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(IsShapePresent(context_->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "recvCounts input", "present",
                                                          "The value of recvCounts input must be nullptr."),
                    return ge::GRAPH_FAILED);

    auto gmmXDesc = context_->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmX"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmX",
                                               Ops::Base::ToString(gmmXDesc->GetStorageFormat()).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    auto gmmWeightDesc = context_->GetInputDesc(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeight"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmWeight",
                                               Ops::Base::ToString(gmmWeightDesc->GetStorageFormat()).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    auto mmXDesc = context_->GetOptionalInputDesc(NON_QUANT_MM_X_INDEX);
    if (mmXDesc != nullptr) {
        OP_TILING_CHECK(mmXDesc->GetStorageFormat() != ge::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmX",
                                                   Ops::Base::ToString(mmXDesc->GetStorageFormat()).c_str(), "ND"),
                        return ge::GRAPH_FAILED);
    }
    auto mmWeightDesc = context_->GetOptionalInputDesc(NON_QUANT_MM_WEIGHT_INDEX);
    if (mmWeightDesc != nullptr) {
        OP_TILING_CHECK(mmWeightDesc->GetStorageFormat() != ge::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmWeight",
                                                   Ops::Base::ToString(mmWeightDesc->GetStorageFormat()).c_str(), "ND"),
                        return ge::GRAPH_FAILED);
    }
    auto gmmYDesc = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX);
    OP_TILING_CHECK(gmmYDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmY"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmYDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gmmY",
                                               Ops::Base::ToString(gmmYDesc->GetStorageFormat()).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX);
        if (mmYDesc != nullptr) {
            OP_TILING_CHECK(mmYDesc->GetStorageFormat() != ge::FORMAT_ND,
                            OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "mmY",
                                                       Ops::Base::ToString(mmYDesc->GetStorageFormat()).c_str(), "ND"),
                            return ge::GRAPH_FAILED);
        }
    }
    if (permuteOutFlag_) {
        auto permuteOutDesc = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX);
        if (permuteOutDesc != nullptr) {
            OP_TILING_CHECK(permuteOutDesc->GetStorageFormat() != ge::FORMAT_ND,
                            OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "permuteOut",
                                                       Ops::Base::ToString(permuteOutDesc->GetStorageFormat()).c_str(),
                                                       "ND"),
                            return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetShapeInfo()
{
    OP_TILING_CHECK(context_->GetInputShape(GMM_X_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmX"), return ge::GRAPH_FAILED);
    bsk_ = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    h1_ = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    gmmXDataType_ = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();

    OP_TILING_CHECK(context_->GetInputShape(GMM_WEIGHT_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeight"), return ge::GRAPH_FAILED);
    e_ = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    n1_ = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(n1DimIndex);
    gmmWeightDataType_ = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();

    OP_TILING_CHECK(context_->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmY"), return ge::GRAPH_FAILED);
    a_ = context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(DIM_ZERO);

    if (IsShapePresent(context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX))) {
        hasSharedExpertFlag_ = true;
        bs_ = context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        h2_ = context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
        mmXDataType_ = context_->GetOptionalInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
    }

    if (IsShapePresent(context_->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX))) {
        auto mmWeightShape = context_->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        n2_ = mmWeightShape.GetDim(n2DimIndex);
        mmWeightDataType_ = context_->GetOptionalInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::DoLibApiTiling()
{
    return GetShapeInfo();
}

ge::graphStatus AlltoAllvGmmTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "workspace"),
                    return ge::GRAPH_FAILED);
    const uint64_t tensorListSize = 512;
    uint64_t xDataTypeSize = GetSizeByDataType(gmmXDataType_);
    uint64_t commOutSize = Ops::Base::CeilAlign(a_ * h1_ * xDataTypeSize, tensorListSize);
    uint64_t groupListSize = sizeof(int64_t) * e_ * epWorldSize_;
    uint64_t aGroupOffsetTableSize = sizeof(uint64_t) * e_ * epWorldSize_;
    uint64_t xScaleOffsetTableSize = sizeof(uint64_t) * e_ * epWorldSize_;
    workspaces[0] = libApiWorkSpaceSize_ + commOutSize + groupListSize + aGroupOffsetTableSize +
                    xScaleOffsetTableSize + tensorListSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::PostTiling()
{
    auto tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->taskTilingInfo.BSK = bsk_;
    tilingData->taskTilingInfo.BS = bs_;
    tilingData->taskTilingInfo.H1 = h1_;
    tilingData->taskTilingInfo.H2 = h2_;
    tilingData->taskTilingInfo.A = a_;
    tilingData->taskTilingInfo.N1 = n1_;
    tilingData->taskTilingInfo.N2 = n2_;
    tilingData->taskTilingInfo.epWorldSize = epWorldSize_;
    tilingData->taskTilingInfo.e = e_;
    tilingData->taskTilingInfo.ubSize = ubSize_;
    tilingData->taskTilingInfo.aivCoreNum = aivCoreNum_;
    tilingData->taskTilingInfo.aicCoreNum = aicCoreNum_;
    uint32_t expertNum = CalcExpertNum(e_, epWorldSize_, bsk_, n1_);
    tilingData->taskTilingInfo.expertNum = expertNum;
    tilingData->taskTilingInfo.mainLoopExpertNum = expertNum;
    tilingData->taskTilingInfo.tailLoopExpertNum =
        (e_ % expertNum == 0) ? 0 : static_cast<uint32_t>(e_ % expertNum);
    tilingData->taskTilingInfo.totalLoopCount =
        static_cast<uint32_t>((e_ + expertNum - 1U) / expertNum);

    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts"),
                    return ge::GRAPH_FAILED);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts"),
                    return ge::GRAPH_FAILED);
    const int64_t *sendCountsData = static_cast<const int64_t *>(sendCountsPtr->GetData());
    OP_TILING_CHECK(sendCountsData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "sendCounts data"),
                    return ge::GRAPH_FAILED);
    const int64_t *recvCountsData = static_cast<const int64_t *>(recvCountsPtr->GetData());
    OP_TILING_CHECK(recvCountsData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "recvCounts data"),
                    return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < e_ * epWorldSize_; i++) {
        tilingData->taskTilingInfo.sendCnt[i] = sendCountsData[i];
        tilingData->taskTilingInfo.recvCnt[i] = recvCountsData[i];
    }

    tilingData->isPermuteOut = permuteOutFlag_;
    tilingData->isNeedMM = hasSharedExpertFlag_;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AlltoAllvGmmTilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct AlltoAllvGmmCompileInfo {};
static ge::graphStatus TilingParseForAlltoAllvGmm(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<AlltoAllvGmmCompileInfo>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AlltoAllvGroupedMatMul)
    .Tiling(AlltoAllvGmmTilingFunc)
    .TilingParse<AlltoAllvGmmCompileInfo>(TilingParseForAlltoAllvGmm);
} // namespace optiling
