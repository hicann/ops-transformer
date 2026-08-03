/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file all_gather_quant_bmm_tiling.cpp
 * \brief
 */
#ifndef _ALL_GATHER_QUANT_BMM_TILING_CPP_
#define _ALL_GATHER_QUANT_BMM_TILING_CPP_
#include "all_gather_quant_bmm_tiling.h"
#include "all_gather_fit_balance_tiling.h"
#include "all_gather_hccl_utils.h"
#include "common/utils/op_mc2.h"
#include "mc2_comm_utils.h"
#include "mc2_log.h"
#include "../../../op_kernel/all_gather_matmul_v2_apt_tiling_key.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace optiling {
namespace {
const gert::Shape defaultShape = gert::Shape();

constexpr uint32_t DAVIDVER = 1;
constexpr uint32_t HIF8 = 34;
constexpr uint32_t FP8_E5M2 = 35;
constexpr uint32_t FP8_E4M3 = 36;
constexpr uint64_t MX_SCALE_OFFSET = 63;
constexpr uint64_t MX_SCALE_ALIGN = 64;
constexpr uint64_t EVEN_ALIGN = 2;
constexpr uint64_t MX_SCALE_K_VALUE_ALIGN = 32;
constexpr uint64_t MX_SCALE_BLOCK_M = 1;
constexpr uint64_t MX_SCALE_BLOCK_K = 32;
constexpr uint64_t MX_SCALE_BLOCK_N = 1;
constexpr uint64_t DIM_IS_ONE = 1;
constexpr uint64_t DIM_NUM_IS_ONE = 1;
constexpr uint64_t DIM_NUM_IS_TWO = 2;
constexpr uint64_t DIM_NUM_IS_THREE = 3;
constexpr uint64_t DTYPE_ENUM_FLOAT = 0;
constexpr uint64_t DTYPE_ENUM_FLOAT16 = 1;
constexpr uint64_t DTYPE_ENUM_BF16 = 27;
constexpr uint64_t GROUP_SIZE_INDEX = 7;
constexpr uint64_t YDTYPE_INDEX = 10;
constexpr uint8_t OUTPUT_TYPE_FLOAT = 2;
constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;

constexpr int32_t IDX_K_LOW = 2;
constexpr int32_t IDX_K_HIGH = 3;
constexpr int32_t IDX_N_LOW = 4;
constexpr int32_t IDX_N_HIGH = 5;
constexpr int32_t IDX_B_LOW = 6;

constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
} // namespace

static constexpr int64_t CMP_MAX_LEN = 7;

bool AllGatherQuantBmmTiling::IsCapable()
{
    // geAType 和 geBType 为fp8/hif8时且amax为空时做tiling
    auto amaxShape = context_->GetOutputShape(OUTPUT_AMAX);
    if (amaxShape != nullptr) {
        OP_LOGI(opName_, "amaxShapeDim0 is %lu", amaxShape->GetStorageShape().GetDim(0));
    }
    OP_TILING_CHECK((amaxShape != nullptr) && (amaxShape->GetStorageShape().GetDim(0) != 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "amaxOut", std::to_string(amaxShape->GetStorageShape().GetDim(0)).c_str(),
                        "If the dtype of input is fp8 or hifp8, amaxOut must be nullptr or empty tensor"),
                    return false);
    // geAType 和 geBType 为fp8/hif8时做tiling
    if (mc2tiling::CheckDataTypeVaild(args_.geAType, mc2tiling::FP8DTYPE_SUPPORT_LIST) &&
        mc2tiling::CheckDataTypeVaild(args_.geBType, mc2tiling::FP8DTYPE_SUPPORT_LIST)) {
        OP_LOGI(opName_, "Start with allgather quantbmm tiling.");
        return true;
    }
    // geAType 和 geBType 为fp4时做tiling
    if ((args_.geAType == ge::DataType::DT_FLOAT4_E2M1) && (args_.geBType == ge::DataType::DT_FLOAT4_E2M1)) {
        OP_LOGI(opName_, "Start with allgather quantbmm mxfp4 tiling.");
        isFp4_ = true;
        return true;
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "x1/x2 dtype", Ops::Base::ToString(args_.geAType).c_str(),
                                          "The dtype of x1/x2 must be within the supported fp8/fp4 range");
    return false;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckGroupSize()
{
    uint64_t groupSize = static_cast<uint64_t>(*context_->GetAttrs()->GetAttrPointer<int64_t>(GROUP_SIZE_INDEX));
    OP_LOGI(opName_, "groupSize=%lu", groupSize);
    mc2tiling::Mc2MatmulShapeInfo shapeInfo = {context_->GetInputShape(INPUT_X1),
                                               context_->GetInputShape(INPUT_X2),
                                               context_->GetOptionalInputShape(SCALE_INV1),
                                               context_->GetOptionalInputShape(SCALE_INV2),
                                               false,
                                               *context_->GetAttrs()->GetAttrPointer<bool>(IS_TRANS_B),
                                               opName_};
    uint64_t groupSizeK = groupSize & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (groupSize >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (groupSize >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    auto scaleInv1Desc = context_->GetOptionalInputDesc(SCALE_INV1);
    if ((quantMmMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) &&
        (scaleInv1Desc->GetDataType() == ge::DataType::DT_FLOAT)) {
        OP_TILING_CHECK(groupSize != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "groupSize", std::to_string(groupSize).c_str(),
                                                              "The value of groupSize must be 0 in pertensor scene"),
                        return ge::GRAPH_FAILED);
    } else if ((quantMmMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) &&
               (scaleInv1Desc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        shapeInfo.isMxfp = true;
        OP_TILING_CHECK(!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, groupSizeM, groupSizeN, groupSizeK),
                        OP_LOGE(opName_, "Failed to execute inferGroupSize in mx scene."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (groupSizeM != MX_SCALE_BLOCK_M) || (groupSizeN != MX_SCALE_BLOCK_N) || (groupSizeK != MX_SCALE_BLOCK_K),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "groupSize",
                                                  (std::string("[") + std::to_string(groupSizeM) + ", " +
                                                   std::to_string(groupSizeN) + ", " + std::to_string(groupSizeK) + "]")
                                                      .c_str(),
                                                  "The value of groupSize must be [1, 1, 32] in mx scene"),
            return ge::GRAPH_FAILED);
    } else if (quantMmMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) {
        OP_TILING_CHECK(!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, groupSizeM, groupSizeN, groupSizeK),
                        OP_LOGE(opName_, "Failed to execute inferGroupSize in perblock scene."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (groupSizeM != PERBLOCK_SCALE_SIZE) || (groupSizeN != PERBLOCK_SCALE_SIZE) ||
                (groupSizeK != PERBLOCK_SCALE_SIZE),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "groupSize",
                                                  (std::string("[") + std::to_string(groupSizeM) + ", " +
                                                   std::to_string(groupSizeN) + ", " + std::to_string(groupSizeK) + "]")
                                                      .c_str(),
                                                  "The value of groupSize must be [128, 128, 128] in perblock scene"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckMXFP4Input()
{
    auto scaleInv1Desc = context_->GetOptionalInputDesc(SCALE_INV1);
    // mxfp4: K must be even
    OP_TILING_CHECK(args_.kValue % EVEN_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "kValue", std::to_string(args_.kValue).c_str(),
                                                          "The value of K must be even in mxfp4 scene"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((!args_.isBTrans),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "BTrans", std::to_string(args_.isBTrans).c_str(),
                                                          "x2 only support transpose"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((scaleInv1Desc->GetDataType() != ge::DataType::DT_FLOAT8_E8M0),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "scale1", (Ops::Base::ToString(scaleInv1Desc->GetDataType())).c_str(),
                        "The dtypes of scale1 must be float8_e8m0 when x1/x2 is float4_e2m1"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckX1Input()
{
    OP_TILING_CHECK(
        (args_.mValue % PERBLOCK_SCALE_SIZE != 0) && (quantMmMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1Dim0", std::to_string(args_.mValue).c_str(),
                                              "The value of x1Dim0 must be divisible by 128 in perblock scene"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMmMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) && (isFp4_) &&
                        (CheckMXFP4Input() != ge::GRAPH_SUCCESS),
                    OP_LOGE(opName_, "Check mxfp4 input failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMmMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) && (isFp4_),
                    OP_LOGE(opName_, "fp4 only support mx quant."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckPerBlockScaleInput()
{
    auto scaleInv1Shape = context_->GetOptionalInputShape(SCALE_INV1);
    auto scaleInv2Shape = context_->GetOptionalInputShape(SCALE_INV2);
    uint64_t scale1FirstDim = scaleInv1Shape->GetStorageShape().GetDim(0);
    uint64_t scale1SecondDim = scaleInv1Shape->GetStorageShape().GetDim(1);
    uint64_t scale2FirstDim = scaleInv2Shape->GetStorageShape().GetDim(0);
    uint64_t scale2SecondDim = scaleInv2Shape->GetStorageShape().GetDim(1);
    uint64_t ceilMValue = (args_.mValue + PERBLOCK_SCALE_SIZE - 1) / PERBLOCK_SCALE_SIZE;
    uint64_t ceilKValue = (args_.kValue + PERBLOCK_SCALE_SIZE - 1) / PERBLOCK_SCALE_SIZE;
    uint64_t ceilNValue = (args_.nValue + PERBLOCK_SCALE_SIZE - 1) / PERBLOCK_SCALE_SIZE;
    const gert::StorageShape *x1Shape = context_->GetInputShape(INPUT_X1);
    const gert::StorageShape *x2Shape = context_->GetInputShape(INPUT_X2);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    if (x1Dim1 != x2Dim0) {
        scale2FirstDim = scaleInv2Shape->GetStorageShape().GetDim(1);
        scale2SecondDim = scaleInv2Shape->GetStorageShape().GetDim(0);
    }

    OP_LOGI(opName_,
            "scale1FirstDim=%lu, scale1SecondDim=%lu, scale2FirstDim=%lu, scale2SecondDim=%lu, ceilMValue=%lu, "
            "ceilKValue=%lu, ceilNValue=%lu.",
            scale1FirstDim, scale1SecondDim, scale2FirstDim, scale2SecondDim, ceilMValue, ceilKValue, ceilNValue);

    scale1kSpaceSize_ = args_.rankDim * scale1FirstDim * scale1SecondDim * sizeof(float);
    OP_LOGI(opName_, "scale1kSpaceSize_=%lu.", scale1kSpaceSize_);

    OP_TILING_CHECK(
        (scale1FirstDim != ceilMValue) || (scale1SecondDim != ceilKValue),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "x1Scale",
            (std::string("[") + std::to_string(scale1FirstDim) + ", " + std::to_string(scale1SecondDim) + "]").c_str(),
            (std::string("The shape of x1Scale must be [") + std::to_string(ceilMValue) + ", " +
             std::to_string(ceilKValue) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (scale2FirstDim != ceilKValue) || (scale2SecondDim != ceilNValue),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "x2Scale",
            (std::string("[") + std::to_string(scale2FirstDim) + ", " + std::to_string(scale2SecondDim) + "]").c_str(),
            (std::string("The shape of x2Scale must be [") + std::to_string(ceilKValue) + ", " +
             std::to_string(ceilNValue) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckMXFPScaleInput()
{
    auto scaleInv1Shape = context_->GetOptionalInputShape(SCALE_INV1);
    auto scaleInv2Shape = context_->GetOptionalInputShape(SCALE_INV2);
    const gert::StorageShape *x2Shape = context_->GetInputShape(INPUT_X2);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((scaleInv1Shape->GetStorageShape().GetDimNum() != DIM_NUM_IS_THREE) ||
                        (scaleInv2Shape->GetStorageShape().GetDimNum() != DIM_NUM_IS_THREE),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "x1Scale and x2Scale",
                        (std::to_string(scaleInv1Shape->GetStorageShape().GetDimNum()) + "D and " +
                         std::to_string(scaleInv2Shape->GetStorageShape().GetDimNum()) + "D")
                            .c_str(),
                        "The shape dims of x1Scale and x2Scale must be 3D in mxfp scene"),
                    return ge::GRAPH_FAILED);
    uint64_t scale1FirstDim = scaleInv1Shape->GetStorageShape().GetDim(0);
    uint64_t scale1SecondDim = scaleInv1Shape->GetStorageShape().GetDim(1);
    uint64_t scale1ThirdDim = scaleInv1Shape->GetStorageShape().GetDim(2);
    uint64_t scale2FirstDim = scaleInv2Shape->GetStorageShape().GetDim(0);
    uint64_t scale2SecondDim = scaleInv2Shape->GetStorageShape().GetDim(1);
    uint64_t scale2ThirdDim = scaleInv2Shape->GetStorageShape().GetDim(2);
    uint64_t ceilKAlign = (args_.kValue + MX_SCALE_OFFSET) / MX_SCALE_ALIGN;
    OP_LOGI(opName_,
            "scale1FirstDim=%lu, scale1SecondDim=%lu, scale1ThirdDim=%lu, "
            "scale2FirstDim=%lu, scale2SecondDim=%lu, scale2ThirdDim=%lu, mValue=%lu, ceilKAlign=%lu, nValue=%lu.",
            scale1FirstDim, scale1SecondDim, scale1ThirdDim, scale2FirstDim, scale2SecondDim, scale2ThirdDim,
            args_.mValue, ceilKAlign, args_.nValue);
    OP_TILING_CHECK((scale1FirstDim != args_.mValue) || (scale1SecondDim != ceilKAlign) ||
                        (scale1ThirdDim != EVEN_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "x1Scale",
                        (std::string("[") + std::to_string(scale1FirstDim) + ", " + std::to_string(scale1SecondDim) +
                         ", " + std::to_string(scale1ThirdDim) + "]")
                            .c_str(),
                        (std::string("The shape of x1Scale must be [") + std::to_string(args_.mValue) + ", " +
                         std::to_string(ceilKAlign) + ", " + std::to_string(EVEN_ALIGN) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);

    bool isTransB = *context_->GetAttrs()->GetAttrPointer<bool>(IS_TRANS_B);
    if (isTransB) {
        OP_TILING_CHECK((scale2FirstDim != args_.nValue) || (scale2SecondDim != ceilKAlign) ||
                            (scale2ThirdDim != EVEN_ALIGN),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "x2Scale",
                            (std::string("[") + std::to_string(scale2FirstDim) + ", " +
                             std::to_string(scale2SecondDim) + ", " + std::to_string(scale2ThirdDim) + "]")
                                .c_str(),
                            (std::string("The shape of x2Scale must be [") + std::to_string(args_.nValue) + ", " +
                             std::to_string(ceilKAlign) + ", " + std::to_string(EVEN_ALIGN) + "]")
                                .c_str()),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((scale2FirstDim != ceilKAlign) || (scale2SecondDim != args_.nValue) ||
                            (scale2ThirdDim != EVEN_ALIGN),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "x2Scale",
                            (std::string("[") + std::to_string(scale2FirstDim) + ", " +
                             std::to_string(scale2SecondDim) + ", " + std::to_string(scale2ThirdDim) + "]")
                                .c_str(),
                            (std::string("The shape of x2Scale must be [") + std::to_string(ceilKAlign) + ", " +
                             std::to_string(args_.nValue) + ", " + std::to_string(EVEN_ALIGN) + "]")
                                .c_str()),
                        return ge::GRAPH_FAILED);
    }
    scale1kSpaceSize_ = args_.rankDim * scale1FirstDim * scale1SecondDim * sizeof(ge::DT_FLOAT8_E8M0);
    OP_LOGI(opName_, "scale1kSpaceSize_=%lu.", scale1kSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckPerTensorScaleInput()
{
    auto scaleInv1Shape = context_->GetOptionalInputShape(SCALE_INV1);
    auto scaleInv2Shape = context_->GetOptionalInputShape(SCALE_INV2);
    OP_TILING_CHECK(
        (scaleInv1Shape->GetStorageShape().GetDim(0) != DIM_IS_ONE) ||
            (scaleInv2Shape->GetStorageShape().GetDim(0) != DIM_IS_ONE),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "x1Scale or x2Scale",
            (std::string("scaleInv1Shape dim=") + std::to_string(scaleInv1Shape->GetStorageShape().GetDim(0)) +
             " and scaleInv2Shape dim=" + std::to_string(scaleInv2Shape->GetStorageShape().GetDim(0)))
                .c_str(),
            "The shape of x1Scale or x2Scale must be scalar"),
        return ge::GRAPH_FAILED);

    // pertensor模式：每个卡有1个scale值，需要通信scale1
    scale1kSpaceSize_ = args_.rankDim * sizeof(float);
    OP_LOGI(opName_, "scale1kSpaceSize_=%lu.", scale1kSpaceSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::SetQuantScene()
{
    auto scaleInv1Shape = context_->GetOptionalInputShape(SCALE_INV1);
    auto scaleInv2Shape = context_->GetOptionalInputShape(SCALE_INV2);
    auto scaleInv1Desc = context_->GetOptionalInputDesc(SCALE_INV1);
    auto scaleInv2Desc = context_->GetOptionalInputDesc(SCALE_INV2);
    OP_TILING_CHECK(isFp4_ && ((scaleInv1Shape->GetStorageShape().GetDimNum() != DIM_NUM_IS_THREE) ||
                                   (scaleInv2Shape->GetStorageShape().GetDimNum() != DIM_NUM_IS_THREE)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "x1Scale and x2Scale",
                        (std::to_string(scaleInv1Shape->GetStorageShape().GetDimNum()) + "D and " +
                         std::to_string(scaleInv2Shape->GetStorageShape().GetDimNum()) + "D")
                            .c_str(),
                        "The shape dims of x1Scale and x2Scale must be 3D when x1 and x2 is float4_e2m1"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((scaleInv1Shape->GetStorageShape().GetDimNum() != scaleInv2Shape->GetStorageShape().GetDimNum()),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "x1Scale and x2Scale",
                        (std::to_string(scaleInv1Shape->GetStorageShape().GetDimNum()) + "D and " +
                         std::to_string(scaleInv2Shape->GetStorageShape().GetDimNum()) + "D")
                            .c_str(),
                        "The shape dims of x1Scale and x2Scale must be the same"),
                    return ge::GRAPH_FAILED);
    if ((scaleInv1Shape->GetStorageShape().GetDimNum() == DIM_NUM_IS_ONE) &&
        (scaleInv2Shape->GetStorageShape().GetDimNum() == DIM_NUM_IS_ONE)) {
        quantMmMode_ = mc2tiling::Mc2QuantMode::PERTENSOR_MODE;
        return ge::GRAPH_SUCCESS;
    } else if ((scaleInv1Shape->GetStorageShape().GetDimNum() == DIM_NUM_IS_THREE) &&
               (scaleInv2Shape->GetStorageShape().GetDimNum() == DIM_NUM_IS_THREE) &&
               (scaleInv1Desc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        quantMmMode_ = mc2tiling::Mc2QuantMode::PERTENSOR_MODE;
        return ge::GRAPH_SUCCESS;
    } else if ((scaleInv1Shape->GetStorageShape().GetDimNum() == DIM_NUM_IS_TWO) &&
               (scaleInv2Shape->GetStorageShape().GetDimNum() == DIM_NUM_IS_TWO) &&
               (scaleInv1Desc->GetDataType() == ge::DataType::DT_FLOAT)) {
        quantMmMode_ = mc2tiling::Mc2QuantMode::PERBLOCK_MODE;
        return ge::GRAPH_SUCCESS;
    } else {
        quantMmMode_ = mc2tiling::Mc2QuantMode::INVALID_MODE;
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "scaleInv1/scaleInv2 dtype", Ops::Base::ToString(scaleInv1Desc->GetDataType()).c_str(),
            "The dtype of scaleInv1/scaleInv2 must be within the range {pertensor, mxfp, perblock}");
    }

    return ge::GRAPH_FAILED;
}

mc2tiling::Mc2QuantMode AllGatherQuantBmmTiling::GetQuantScene()
{
    return quantMmMode_;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckScaleInvShape()
{
    auto scaleInv1Desc = context_->GetOptionalInputDesc(SCALE_INV1);
    if ((quantMmMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) &&
        (scaleInv1Desc->GetDataType() == ge::DataType::DT_FLOAT)) {
        OP_LOGI(opName_, "Check pertensor scale input!");
        OP_TILING_CHECK(CheckPerTensorScaleInput() == ge::GRAPH_FAILED,
                        OP_LOGE(opName_, "Check pertensor scale input failed."), return ge::GRAPH_FAILED);
    } else if ((quantMmMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) &&
               (scaleInv1Desc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        OP_LOGI(opName_, "Check mxfp scale input!");
        OP_TILING_CHECK(CheckMXFPScaleInput() == ge::GRAPH_FAILED, OP_LOGE(opName_, "Check mxfp scale input failed."),
                        return ge::GRAPH_FAILED);
    } else if (quantMmMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) {
        OP_LOGI(opName_, "Check perblock scale input!");
        OP_TILING_CHECK(CheckPerBlockScaleInput() == ge::GRAPH_FAILED,
                        OP_LOGE(opName_, "Check perblock scale input failed."), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "quantMmMode", std::to_string(static_cast<int>(quantMmMode_)).c_str(),
                                  "pertensor, mxfp or perblock");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckInputValid()
{
    OP_TILING_CHECK((args_.geAType == ge::DataType::DT_HIFLOAT8) && (args_.geAType != args_.geBType),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "BType", Ops::Base::ToString(args_.geBType).c_str(),
                        "If the dtype of AType is hifp8, the value of BType must be equal to that of AType"),
                    return ge::GRAPH_FAILED);

    auto scaleInv1Desc = context_->GetOptionalInputDesc(SCALE_INV1);
    auto scaleInv2Desc = context_->GetOptionalInputDesc(SCALE_INV2);
    OP_TILING_CHECK((scaleInv1Desc == nullptr) || (scaleInv2Desc == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "scale1 or scale2"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(scaleInv1Desc->GetDataType() != scaleInv2Desc->GetDataType(),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "scale1 and scale2",
                                                           (Ops::Base::ToString(scaleInv1Desc->GetDataType()) +
                                                            " and " + Ops::Base::ToString(scaleInv2Desc->GetDataType()))
                                                               .c_str(),
                                                           "The dtypes of scale1 and scale2 must be the same"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((args_.geAType == ge::DataType::DT_HIFLOAT8) &&
                        (scaleInv1Desc->GetDataType() != ge::DataType::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "scale1",
                                                          Ops::Base::ToString(scaleInv1Desc->GetDataType()).c_str(),
                                                          "The dtype of scale1 must be float32"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (((scaleInv1Desc->GetDataType() != ge::DataType::DT_FLOAT) &&
          (scaleInv1Desc->GetDataType() != ge::DataType::DT_FLOAT8_E8M0)) ||
         ((scaleInv2Desc->GetDataType() != ge::DataType::DT_FLOAT) &&
          (scaleInv2Desc->GetDataType() != ge::DataType::DT_FLOAT8_E8M0))),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "scale1 and scale2",
                                               (Ops::Base::ToString(scaleInv1Desc->GetDataType()) + " and " +
                                                Ops::Base::ToString(scaleInv2Desc->GetDataType()))
                                                   .c_str(),
                                               "The dtypes of scale1 and scale2 must be float32 or float8_e8m0"),
        return ge::GRAPH_FAILED);

    auto scaleInv1Shape = context_->GetOptionalInputShape(SCALE_INV1);
    auto scaleInv2Shape = context_->GetOptionalInputShape(SCALE_INV2);
    OP_TILING_CHECK((scaleInv1Shape == nullptr) || (scaleInv2Shape == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "scale1Shape or scale2Shape"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckBiasInput()
{
    auto biasShape = context_->GetOptionalInputShape(BIAS);
    OP_TILING_CHECK((quantMmMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) && (biasShape != nullptr),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "bias", "not nullptr",
                                                          "The value of bias must be nullptr in perblock scene"),
                    return ge::GRAPH_FAILED);
    auto biasDesc = context_->GetOptionalInputDesc(BIAS);
    OP_TILING_CHECK(
        (biasDesc != nullptr) && (biasDesc->GetDataType() != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "bias", Ops::Base::ToString(biasDesc->GetDataType()).c_str(), "FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::CheckInput()
{
    OP_TILING_CHECK(CheckInputValid() == ge::GRAPH_FAILED, OP_LOGE(opName_, "Check input valid failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetQuantScene() == ge::GRAPH_FAILED, OP_LOGE(opName_, "Fail to set quant scene."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckBiasInput() == ge::GRAPH_FAILED, OP_LOGE(opName_, "Check input bias failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckScaleInvShape() == ge::GRAPH_FAILED, OP_LOGE(opName_, "check scale inv shape failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckX1Input() == ge::GRAPH_FAILED, OP_LOGE(opName_, "Check x1 input failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckGroupSize() == ge::GRAPH_FAILED, OP_LOGE(opName_, "Check block size and axis failed."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::SetMc2Hcomm()
{
    int index = 0;
    auto group = context_->GetAttrs()->GetAttrPointer<char>(index++);
    std::string algConfig = "AllGather=level0:fullmesh";
    // mxfp4场景使用int8类型通信，通信量减半
    uint32_t hcclDataType;
    if (isFp4_) {
        hcclDataType = static_cast<uint32_t>(mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT8);
    } else {
        hcclDataType = static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType));
    }
    Mc2CcTilingConfig mc2CcTilingConfig(group, static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER),
                                        algConfig, 0, hcclDataType, hcclDataType);
    // Set hccl comm engine with comm_mode
    uint8_t commEngine = Mc2Comm::ENGINE_AICPU;
    if (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0) {
        commEngine = Mc2Comm::ENGINE_CCU_SCHED;
    }
    OP_LOGD(opName_, "Tiling SetMc2Hcom commMode_: %s", commMode_);
    OP_LOGD(opName_, "Tiling SetMc2Hcom commEngine: %d", commEngine);
    mc2CcTilingConfig.SetCommEngine(commEngine);
    uint8_t skipBufferWindowCopy = (allGatherMatmulTilingDataFp8_->param.gatherLen == 0) ?
                                       static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT) :
                                       static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
    mc2CcTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingDataFp8_->mc2InitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingDataFp8_->mc2CcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmTiling::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckHCCLSize());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    SetTilingKeyParams();
    OP_TILING_CHECK(SetMc2Hcomm() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Tiling SetHcommCfg failed."),
                    return ge::GRAPH_FAILED);
    SetRcsTilingData(MutableRCSTilingDataA5());
    DoSplitMTiling(MutableRCSTilingDataA5());
    if (quantMmMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) {
        PostDoSplitMTiling(MutableRCSTilingDataA5(), GetQuantScene());
    }
    if (args_.nValue != 0) {
        MC2_CHECK_LOG_RET(opName_, DoAdaptSlidWindowTiling());
    }
    DoAllGatherTiling(MutableRCSTilingDataA5(), MutableTCubeTileTilingData(), MutableTCubeTailTilingData(),
                      allGatherMatmulTilingDataFp8_->debugMode, allGatherMatmulTilingDataFp8_->dataType);
    return ge::GRAPH_SUCCESS;
}

void AllGatherQuantBmmTiling::SetTilingKeyParams()
{
    outputIsFp8_ = false;
    inputIsBf16Fp16_ = false;
    commAlgorithm_ = args_.commAlg;
    enableNd2Nz_ = false;
    castBias_ = false;
}

ge::graphStatus AllGatherQuantBmmTiling::GetWorkspaceSize()
{
    MC2_CHECK_LOG_RET(opName_, AllGatherMatmulTilingBase::GetWorkspaceSize());
    // 当N=0时，Helper不执行，myWorkSpaceSize_未被设置，需要手动赋值
    // workspaceSize_包含：libApiWorkSpaceSize_ + storageA_(含gatherLen) + biasLen_
    // scale1kSpaceSize_为scale1通信所需空间
    if (args_.nValue == 0) {
        myWorkSpaceSize_ = workspaceSize_ + scale1kSpaceSize_;
    } else {
        myWorkSpaceSize_ = myWorkSpaceSize_ + MutableRCSTilingDataA5().gatherLen + scale1kSpaceSize_;
    }
    OP_LOGI(opName_, "Set max workspace size %lu to context", myWorkSpaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = myWorkSpaceSize_;

    return ge::GRAPH_SUCCESS;
}

uint64_t AllGatherQuantBmmTiling::GetTilingKey() const
{
    uint8_t outputType = 0;
    uint32_t yDtype = static_cast<uint32_t>(*context_->GetAttrs()->GetAttrPointer<uint64_t>(YDTYPE_INDEX));
    if (yDtype == DTYPE_ENUM_FLOAT) {
        outputType = OUTPUT_TYPE_FLOAT;
    } else if ((yDtype == DTYPE_ENUM_FLOAT16) || (yDtype == DTYPE_ENUM_BF16)) {
        outputType = static_cast<uint8_t>(0);
    } else {
        outputType = static_cast<uint8_t>(1);
    }

    uint8_t scaleType = 0;

    auto scaleInv1Desc = context_->GetOptionalInputDesc(SCALE_INV1);
    auto scaleInv2Desc = context_->GetOptionalInputDesc(SCALE_INV2);
    if ((scaleInv1Desc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0) &&
        (scaleInv2Desc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        scaleType = static_cast<uint8_t>(1);
    } else {
        scaleType = static_cast<uint8_t>(0);
    }

    uint8_t quanMmMode = static_cast<uint8_t>(quantMmMode_) - 1;
    // Set hccl comm engine with comm_mode
    uint8_t commMode = Mc2Comm::COMM_MODE_AICPU;
    if (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0) {
        commMode = Mc2Comm::COMM_MODE_CCU;
    }

    // Non-A5 platform must use AICPU mode
    if (npuArch_ != NpuArch::DAV_3510) {
        commMode = Mc2Comm::COMM_MODE_AICPU;
    }
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(inputIsBf16Fp16_, args_.isBTrans, outputType, quanMmMode, scaleType, commMode);
    OP_LOGD(opName_,
            "AllGatherMatmulV2, inputIsBf16Fp16_, outputType, "
            "args_.isBTrans, quanMmMode, scaleType, commMode: [%d,%u,%d,%u,%u,%u]",
            inputIsBf16Fp16_, outputType, args_.isBTrans, quanMmMode, scaleType, commMode);
    OP_LOGD(opName_, "tilingkey=%lu", tilingKey);
    return tilingKey;
}

void Mc2PrintTCubeTilingWindowParam(const std::string &opName, DequantBmm::Mc2SlidingWindowParams &tiling)
{
    OP_LOGD(opName, " tiling.mTailTile %d", tiling.mTailTile);
    OP_LOGD(opName, " tiling.nTailTile %d", tiling.nTailTile);
}

void Mc2PrintTCubeTilingParams(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3DataParams &tiling)
{
    OP_LOGD(opName, " tiling.batchA %d", tiling.batchA);
    OP_LOGD(opName, " tiling.batchB %d", tiling.batchB);
    OP_LOGD(opName, " tiling.batchC %d", tiling.batchC);
    OP_LOGD(opName, " tiling.batchA1 %d", tiling.batchA1);
    OP_LOGD(opName, " tiling.batchA2 %d", tiling.batchA2);
    OP_LOGD(opName, " tiling.batchA3 %d", tiling.batchA3);
    OP_LOGD(opName, " tiling.batchA4 %d", tiling.batchA4);
    OP_LOGD(opName, " tiling.batchB1 %d", tiling.batchB1);
    OP_LOGD(opName, " tiling.batchB2 %d", tiling.batchB2);
    OP_LOGD(opName, " tiling.batchB3 %d", tiling.batchB3);
    OP_LOGD(opName, " tiling.batchB4 %d", tiling.batchB4);
    OP_LOGD(opName, " tiling.batchC1 %d", tiling.batchC1);
    OP_LOGD(opName, " tiling.batchC2 %d", tiling.batchC2);
    OP_LOGD(opName, " tiling.batchC3 %d", tiling.batchC3);
    OP_LOGD(opName, " tiling.batchC4 %d", tiling.batchC4);
    OP_LOGD(opName, " tiling.singleCoreBatch %d", tiling.singleCoreBatch);
    OP_LOGD(opName, " tiling.isPerTensor %d", tiling.isPerTensor);
    OP_LOGD(opName, " tiling.isPertoken %d", tiling.isPertoken);
    OP_LOGD(opName, " tiling.isDoubleScale %d", tiling.isDoubleScale);
    OP_LOGD(opName, " tiling.biasThreeDim %d", tiling.biasThreeDim);
    OP_LOGD(opName, " tiling.ubCalcM %d", tiling.ubCalcM);
    OP_LOGD(opName, " tiling.ubCalcN %d", tiling.ubCalcN);
    OP_LOGD(opName, " tiling.needUbBuffer %d", tiling.needUbBuffer);
    OP_LOGD(opName, " tiling.realSingleCoreM %d", tiling.realSingleCoreM);
    OP_LOGD(opName, " tiling.realSingleCoreN %d", tiling.realSingleCoreN);
    OP_LOGD(opName, " tiling.biasDtype %d", tiling.biasDtype);
    OP_LOGD(opName, " tiling.ubSize %d", tiling.ubSize);
    OP_LOGD(opName, " tiling.isMClash %d", tiling.isMClash);
    OP_LOGD(opName, " tiling.isNClash %d", tiling.isNClash);
    OP_LOGD(opName, " tiling.groupSizeM %d", tiling.groupSizeM);
    OP_LOGD(opName, " tiling.groupSizeN %d", tiling.groupSizeN);
    OP_LOGD(opName, " tiling.groupSizeK %d", tiling.groupSizeK);
}

void Mc2PrintTCubeTilingL2cache(const std::string &opName, DequantBmm::Mc2L2cacheTileParams &tiling)
{
    OP_LOGD(opName, " tiling.mTileCntL2 %d", tiling.mTileCntL2);
    OP_LOGD(opName, " tiling.nTileCntL2 %d", tiling.nTileCntL2);
    OP_LOGD(opName, " tiling.mTileBlock %d", tiling.mTileBlock);
    OP_LOGD(opName, " tiling.nTileBlock %d", tiling.nTileBlock);
    OP_LOGD(opName, " tiling.calOrder %d", tiling.calOrder);
    OP_LOGD(opName, " tiling.isBasicTiling %d", tiling.isBasicTiling);
}

ge::graphStatus AllGatherQuantBmmTiling::PostTiling()
{
    OP_LOGD(opName_, "The final tiling data size=%zu and context capacity size=%zu ",
            sizeof(AllGatherMatmulTilingDataFp8), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(AllGatherMatmulTilingDataFp8));

    OP_TILING_CHECK(
        sizeof(AllGatherMatmulTilingDataFp8) % sizeof(uint64_t) != 0,
        OP_LOGE(opName_, "The tiling data size[%zu] is not aligned to 8.", sizeof(AllGatherMatmulTilingDataFp8)),
        return ge::GRAPH_FAILED);
    if (MutableRCSTilingDataA5().rankID == 0) {
        PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingDataA5());
        OP_LOGD(opName_, "local tilingdata");
        PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeLocalTilingData());
        Mc2PrintTCubeTilingParams(context_->GetNodeName(), MutableTCubeLocalTilingParams());
        Mc2PrintTCubeTilingL2cache(context_->GetNodeName(), MutableTCubeLocalTilingTileL2());
        Mc2PrintTCubeTilingWindowParam(context_->GetNodeName(), MutableTCubeLocalTilingWindowParam());

        OP_LOGD(opName_, "tile tilingdata");
        PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTileTilingData());
        Mc2PrintTCubeTilingParams(context_->GetNodeName(), MutableTCubeTileTilingParams());
        Mc2PrintTCubeTilingL2cache(context_->GetNodeName(), MutableTCubeTileTilingTileL2());
        Mc2PrintTCubeTilingWindowParam(context_->GetNodeName(), MutableTCubeTileTilingWindowParam());

        if (MutableRCSTilingDataA5().tailM > 0) {
            OP_LOGD(opName_, "tail exist");
            PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTailTilingData());
            Mc2PrintTCubeTilingParams(context_->GetNodeName(), MutableTCubeTailTilingParams());
            Mc2PrintTCubeTilingL2cache(context_->GetNodeName(), MutableTCubeTailTilingTileL2());
            Mc2PrintTCubeTilingWindowParam(context_->GetNodeName(), MutableTCubeTailTilingWindowParam());
        }
    }

    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

CutResult AllGatherQuantBmmTiling::GetTilingResult()
{
    AllGatherMMFitBalanceTiling tileFormulate(args_, KernelType::ALL_GATHER, TopoType::STANDARD_CARD);
    CutResult result = tileFormulate.GetTiling();

    bool isCcuMode = (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0);

    auto kValue = args_.kValue;
    auto inputDtypeSize = args_.inputDtypeSize;
    if (isFp4_) {
        kValue = args_.kValue / EVEN_ALIGN;
        inputDtypeSize = 1;
    }

    if (isCcuMode) {
        CommSizeInfo commSize{kValue, inputDtypeSize, args_.rankDim};
        AdjustCutResultForCCU(result, args_.mValue, commSize, opName_);
    } else {
        AdjustCutResultForAICPU(result, args_.mValue, tileFormulate.matmulPerf_.GetBaseM(), opName_);
    }

    return result;
}


Mc2Tiling::RCSTiling &AllGatherQuantBmmTiling::MutableRCSTilingDataA5()
{
    return allGatherMatmulTilingDataFp8_->param;
}

::TCubeTiling &AllGatherQuantBmmTiling::MutableTCubeLocalTilingData()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3LocalTiling.matmulTiling;
}

DequantBmm::Mc2QuantBatchMatmulV3DataParams &AllGatherQuantBmmTiling::MutableTCubeLocalTilingParams()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3LocalTiling.params;
}

DequantBmm::Mc2L2cacheTileParams &AllGatherQuantBmmTiling::MutableTCubeLocalTilingTileL2()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3LocalTiling.tileL2cacheTiling;
}

DequantBmm::Mc2SlidingWindowParams &AllGatherQuantBmmTiling::MutableTCubeLocalTilingWindowParam()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3LocalTiling.adaptiveSlidingWin;
}

::TCubeTiling &AllGatherQuantBmmTiling::MutableTCubeTileTilingData()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3TileTiling.matmulTiling;
}

DequantBmm::Mc2QuantBatchMatmulV3DataParams &AllGatherQuantBmmTiling::MutableTCubeTileTilingParams()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3TileTiling.params;
}

DequantBmm::Mc2L2cacheTileParams &AllGatherQuantBmmTiling::MutableTCubeTileTilingTileL2()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3TileTiling.tileL2cacheTiling;
}

DequantBmm::Mc2SlidingWindowParams &AllGatherQuantBmmTiling::MutableTCubeTileTilingWindowParam()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3TileTiling.adaptiveSlidingWin;
}

::TCubeTiling &AllGatherQuantBmmTiling::MutableTCubeTailTilingData()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3TailTiling.matmulTiling;
}

DequantBmm::Mc2QuantBatchMatmulV3DataParams &AllGatherQuantBmmTiling::MutableTCubeTailTilingParams()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3TailTiling.params;
}

DequantBmm::Mc2L2cacheTileParams &AllGatherQuantBmmTiling::MutableTCubeTailTilingTileL2()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3TailTiling.tileL2cacheTiling;
}

DequantBmm::Mc2SlidingWindowParams &AllGatherQuantBmmTiling::MutableTCubeTailTilingWindowParam()
{
    return allGatherMatmulTilingDataFp8_->quantBmmv3TailTiling.adaptiveSlidingWin;
}

ge::graphStatus AllGatherQuantBmmTiling::DoAdaptSlidWindowTiling()
{
    // local块切分
    args_.mValue = args_.orgMValue;
    // 当N=0时，跳过matmul tiling，避免shape检查失败
    if (args_.nValue != 0) {
        AllGatherQuantBmmHelper mmLocalTile(*this, allGatherMatmulTilingDataFp8_->quantBmmv3LocalTiling, true);
        MC2_CHECK_LOG_RET(opName_, mmLocalTile.DoTiling());
    }
    args_.mValue = (quantMmMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) ?
                       (tileMValue_ * (args_.rankDim - 1) * (MutableRCSTilingDataA5().tileCnt)) :
                       tileMValue_;
    AllGatherQuantBmmHelper mmTile(*this, allGatherMatmulTilingDataFp8_->quantBmmv3TileTiling, false);

    // 当N=0时，跳过matmul tiling，避免shape检查失败
    if (args_.nValue != 0) {
        MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
    }
    if (quantMmMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) {
        MutableTCubeTileTilingData().M = tileMValue_;
    } else {
        MutableTCubeTileTilingData().M = tileMValue_ * (args_.rankDim - 1);
    }
    if (MutableRCSTilingDataA5().tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }
    args_.mValue = (quantMmMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) ?
                       (tailMValue_ * (args_.rankDim - 1) * (MutableRCSTilingDataA5().tailCnt)) :
                       tailMValue_;
    AllGatherQuantBmmHelper mmTail(*this, allGatherMatmulTilingDataFp8_->quantBmmv3TailTiling, false);
    // 当N=0时，跳过matmul tiling，避免shape检查失败
    if (args_.nValue != 0) {
        MC2_CHECK_LOG_RET(opName_, mmTail.DoTiling());
    }
    if (quantMmMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) {
        MutableTCubeTailTilingData().M = tailMValue_;
    } else {
        MutableTCubeTailTilingData().M = tailMValue_ * (args_.rankDim - 1);
    }
    return ge::GRAPH_SUCCESS;
}

AllGatherQuantBmmTiling::AllGatherQuantBmmTiling(gert::TilingContext *context)
    : AllGatherMatmulTilingBase(context), allGatherMatmulTilingDataFp8_(&allGatherMatmulTilingDataFp8Self_)
{
    allGatherMatmulTilingDataFp8_ = context_->GetTilingData<AllGatherMatmulTilingDataFp8>();
}

void AllGatherQuantBmmHelper::AnalyzeBatchInfo(const gert::Shape &oriShapeA, const gert::Shape &oriShapeB)
{
    (void)oriShapeA;
    (void)oriShapeB;
    auto x1Shape = GetX1Shape(X1_INDEX);
    auto x2Shape = GetX2Shape(X2_INDEX);
    auto outShape = GetOutputShape(0);
    int32_t numDimA = static_cast<int32_t>(x1Shape.GetDimNum());
    int32_t numDimB = static_cast<int32_t>(x2Shape.GetDimNum());
    int32_t numDimC = static_cast<int32_t>(outShape.GetDimNum());
    inputParams_.batchA4 = numDimA > IDX_K_LOW ? x1Shape.GetDim(numDimA - IDX_K_HIGH) : 1;
    inputParams_.batchA3 = numDimA > IDX_K_HIGH ? x1Shape.GetDim(numDimA - IDX_N_LOW) : 1;
    inputParams_.batchA2 = numDimA > IDX_N_LOW ? x1Shape.GetDim(numDimA - IDX_N_HIGH) : 1;
    inputParams_.batchA1 = numDimA > IDX_N_HIGH ? x1Shape.GetDim(numDimA - IDX_B_LOW) : 1;
    inputParams_.batchB4 = numDimB > IDX_K_LOW ? x2Shape.GetDim(numDimB - IDX_K_HIGH) : 1;
    inputParams_.batchB3 = numDimB > IDX_K_HIGH ? x2Shape.GetDim(numDimB - IDX_N_LOW) : 1;
    inputParams_.batchB2 = numDimB > IDX_N_LOW ? x2Shape.GetDim(numDimB - IDX_N_HIGH) : 1;
    inputParams_.batchB1 = numDimB > IDX_N_HIGH ? x2Shape.GetDim(numDimB - IDX_B_LOW) : 1;
    inputParams_.batchC4 = numDimC > IDX_K_LOW ? outShape.GetDim(numDimC - IDX_K_HIGH) : 1UL;
    inputParams_.batchC3 = numDimC > IDX_K_HIGH ? outShape.GetDim(numDimC - IDX_N_LOW) : 1UL;
    inputParams_.batchC2 = numDimC > IDX_N_LOW ? outShape.GetDim(numDimC - IDX_N_HIGH) : 1UL;
    inputParams_.batchC1 = numDimC > IDX_N_HIGH ? outShape.GetDim(numDimC - IDX_B_LOW) : 1UL;
}

void AllGatherQuantBmmHelper::SetBatch()
{
    if (inputParams_.isPerBlock) {
        if (!isLocal_) {
            batch4_ = tilingProcesser_.args_.rankDim -
                      1; // 本卡数据先在本卡计算，因此batch matmul只用计算rankDim - 1 张卡通信的数据
        }
    }
}

const gert::Shape AllGatherQuantBmmHelper::GetX1Shape(const size_t index)
{
    (void)index;
    SetBatch();
    if (tilingProcesser_.args_.isATrans) {
        return gert::Shape({batch1_, batch2_, batch3_, batch4_, static_cast<int64_t>(tilingProcesser_.args_.kValue),
                            static_cast<int64_t>(tilingProcesser_.args_.mValue)});
    }
    return gert::Shape({batch1_, batch2_, batch3_, batch4_, static_cast<int64_t>(tilingProcesser_.args_.mValue),
                        static_cast<int64_t>(tilingProcesser_.args_.kValue)});
}

const gert::Shape AllGatherQuantBmmHelper::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.args_.isBTrans) {
        return gert::Shape({1, 1, 1, 1, static_cast<int64_t>(tilingProcesser_.args_.nValue),
                            static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    }
    return gert::Shape(
        {1, 1, 1, 1, static_cast<int64_t>(tilingProcesser_.args_.kValue),
         static_cast<int64_t>(
             tilingProcesser_.args_.nValue)}); // x2构造4维全1 Batch，防止matmul规则自动融合x1/output的batch轴
}

const gert::Shape AllGatherQuantBmmHelper::GetOutputShape(const size_t index)
{
    (void)index;
    SetBatch();
    return gert::Shape({batch1_, batch2_, batch3_, batch4_, static_cast<int64_t>(tilingProcesser_.args_.mValue),
                        static_cast<int64_t>(tilingProcesser_.args_.nValue)});
}

const gert::Shape &AllGatherQuantBmmHelper::GetScaleShape(const size_t index)
{
    (void)index;
    if (context_->GetOptionalInputShape(static_cast<size_t>(SCALE_INV1)) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "scale_inv1");
        return defaultShape;
    }
    return context_->GetOptionalInputShape(static_cast<size_t>(SCALE_INV1))->GetStorageShape();
}

// matmul 将protoken 作为第二路scale输入
const gert::StorageShape *AllGatherQuantBmmHelper::GetPertokenShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(SCALE_INV2));
}

const gert::StorageShape *AllGatherQuantBmmHelper::GetBiasShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(BIAS));
}

const gert::StorageShape *AllGatherQuantBmmHelper::GetOffsetShape(const size_t index)
{
    (void)index;
    return (gert::StorageShape *)nullptr;
}

ge::graphStatus AllGatherQuantBmmHelper::GetShapeAttrsInfo()
{
    OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto &&tilingArgs = tilingProcesser_.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transA = tilingArgs.isATrans;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    inputParams_.mSize = tilingArgs.mValue;
    inputParams_.kSize = tilingArgs.kValue;
    inputParams_.nSize = tilingArgs.nValue;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    int yDType = *context_->GetAttrs()->GetAttrPointer<uint64_t>(Y_DTYPE);
    auto scaleTensorDesc = context_->GetOptionalInputDesc(SCALE_INV1);
    auto perTokenScaleTensorDesc = context_->GetOptionalInputDesc(SCALE_INV2);
    OP_TILING_CHECK((scaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(tilingProcesser_.opName_, "scale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((perTokenScaleTensorDesc == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(tilingProcesser_.opName_, "pertokenScale"), return ge::GRAPH_FAILED);
    inputParams_.scaleDtype = scaleTensorDesc->GetDataType();
    inputParams_.perTokenScaleDtype = perTokenScaleTensorDesc->GetDataType();

    inputParams_.cDtype = static_cast<ge::DataType>(yDType);
    inputParams_.outDtype = static_cast<int64_t>(yDType);
    OP_LOGI(tilingProcesser_.opName_, "yDType is %ld", inputParams_.outDtype);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    inputParams_.isPerTensor = (tilingProcesser_.GetQuantScene() == mc2tiling::Mc2QuantMode::PERTENSOR_MODE);
    inputParams_.isPerBlock = (tilingProcesser_.GetQuantScene() == mc2tiling::Mc2QuantMode::PERBLOCK_MODE);
    inputParams_.isDoubleScale = true;
    if (inputParams_.isPerBlock) {
        inputParams_.groupSizeM = PERBLOCK_SCALE_SIZE;
        inputParams_.groupSizeN = PERBLOCK_SCALE_SIZE;
        inputParams_.groupSizeK = PERBLOCK_SCALE_SIZE;
    } else if ((scaleTensorDesc->GetDataType() == ge::DT_FLOAT8_E8M0) &&
               (perTokenScaleTensorDesc->GetDataType() == ge::DT_FLOAT8_E8M0)) {
        inputParams_.groupSizeM = MX_SCALE_BLOCK_M;
        inputParams_.groupSizeN = MX_SCALE_BLOCK_N;
        inputParams_.groupSizeK = MX_SCALE_BLOCK_K;
    }
    MC2_CHECK_TRUE_RET(tilingProcesser_.opName_, AnalyzeInputs());
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmHelper::PostTiling()
{
    // 此处需要后续考虑补充相关tiling打印，当前该函数已不存在。
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherQuantBmmHelper::DoLibApiTiling()
{
    MC2_CHECK_LOG_RET(tilingProcesser_.opName_, Mc2AdaptiveSlidingWindowTiling::DoLibApiTiling());
    isBf16Opt_ = false;
    return ge::GRAPH_SUCCESS;
}

void AllGatherQuantBmmHelper::PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantBatchMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_, " transA_ %d transB_ %d, hasBias_ %d", quantBatchMatmulInfo.transA,
            quantBatchMatmulInfo.transB, quantBatchMatmulInfo.hasBias);
    OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld libApiWorkSpaceSize %u",
            quantBatchMatmulInfo.mSize, quantBatchMatmulInfo.kSize, quantBatchMatmulInfo.nSize,
            quantBatchMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(tilingProcesser_.opName_,
            "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d outDtype %ld"
            " scaleDtype %d perTokenScaleDtype %d",
            static_cast<int32_t>(quantBatchMatmulInfo.aDtype), static_cast<int32_t>(quantBatchMatmulInfo.bDtype),
            static_cast<int32_t>(quantBatchMatmulInfo.cDtype), static_cast<int32_t>(quantBatchMatmulInfo.biasDtype),
            quantBatchMatmulInfo.outDtype, static_cast<int32_t>(quantBatchMatmulInfo.scaleDtype),
            static_cast<int32_t>(quantBatchMatmulInfo.perTokenScaleDtype));
    OP_LOGD(tilingProcesser_.opName_,
            "batchA %lu batchA1-A4[%lu:%lu:%lu:%lu];"
            " batchB %lu batchB1-B4[%lu:%lu:%lu:%lu]; batchC %lu; batchBias %lu",
            quantBatchMatmulInfo.batchA, quantBatchMatmulInfo.batchA1, quantBatchMatmulInfo.batchA2,
            quantBatchMatmulInfo.batchA3, quantBatchMatmulInfo.batchA4, quantBatchMatmulInfo.batchB,
            quantBatchMatmulInfo.batchB1, quantBatchMatmulInfo.batchB2, quantBatchMatmulInfo.batchB3,
            quantBatchMatmulInfo.batchB4, quantBatchMatmulInfo.batchC, quantBatchMatmulInfo.batchBias);
    OP_LOGD(tilingProcesser_.opName_, "isPerTensor %d, isPerBlock %d isDoubleScale %d",
            static_cast<int32_t>(quantBatchMatmulInfo.isPerTensor),
            static_cast<int32_t>(quantBatchMatmulInfo.isPerBlock),
            static_cast<int32_t>(quantBatchMatmulInfo.isDoubleScale));
}
AllGatherQuantBmmHelper::AllGatherQuantBmmHelper(AllGatherQuantBmmTiling &allGatherQuantBmmTiling,
                                                 DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &data, bool isLocal)
    : Mc2AdaptiveSlidingWindowTiling(allGatherQuantBmmTiling.context_, &data),
      tilingProcesser_(allGatherQuantBmmTiling), isLocal_(isLocal)
{
}
// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(AllGatherMatmulV2, AllGatherQuantBmmTiling, static_cast<int32_t>(NpuArch::DAV_3510),
                                   1);
} // namespace optiling

#endif // _QUANT_MATMUL_ALL_REDUCE_TILING_CC_
