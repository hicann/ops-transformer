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
 * \file quant_bmm_reduce_scatter_tiling.cpp
 * \brief
 */

#ifndef _QUANT_BMM_MATMUL_REDUCE_SCATTER_TILING_CC_
#define _QUANT_BMM_MATMUL_REDUCE_SCATTER_TILING_CC_

#include "quant_bmm_reduce_scatter_tiling.h"
#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_tiling_utils.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/matmul_reduce_scatter_v2_apt_tiling_key.h"
#include "reduce_scatter_fit_balance_tiling.h"
#include "../reduce_scatter_formulaic_tiling.h"

using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optiling {
namespace {
const gert::Shape defaultShape = gert::Shape();

constexpr uint32_t DTYPE_ENUM_FLOAT = 0;
constexpr uint32_t DTYPE_ENUM_FLOAT16 = 1;
constexpr uint32_t DTYPE_ENUM_BF16 = 27;
constexpr uint64_t PERBLOCK_SIZE = 128;
constexpr uint64_t MXFP8_SIZE = 32;
constexpr uint64_t MXFP8_SIZE_K = 32;
constexpr uint64_t MXFP8_SIZE_M = 1;
constexpr uint64_t MXFP8_SIZE_N = 1;
constexpr uint64_t MX_SCALE_OFFSET = 64;
constexpr uint64_t EVEN_ALIGN = 2;
constexpr uint64_t PERTENSOR_SCALE_DIM = 1;
constexpr uint64_t PERBLOCK_SCALE_DIM = 2;
constexpr uint64_t MX_SCALE_DIM = 3;
constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;

constexpr int32_t IDX_K_LOW = 2;
constexpr int32_t IDX_K_HIGH = 3;
constexpr int32_t IDX_N_LOW = 4;
constexpr int32_t IDX_N_HIGH = 5;
constexpr int32_t IDX_B_LOW = 6;

// output
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t AMAX_INDEX = 1;
// input
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t X1SCALE_INDEX = 3;
constexpr uint32_t X2SCALE_INDEX = 4;
constexpr uint32_t QUANTSCALE_INDEX = 5;
// Attr
constexpr uint32_t YDTYPE_INDEX = 9;
constexpr uint32_t OUTPUT_TYPE_FLOAT = 2;
constexpr uint32_t BLOCKSIZE_INDEX = 6;
constexpr uint32_t GROUPSIZE_INDEX = 7;
constexpr uint32_t TRANSPOSEB_INDEX = 3;
constexpr uint32_t COMMMODE_INDEX = 10;
// DataType
constexpr std::initializer_list<ge::DataType> HIF8FP8FP4DTYPE_SUPPORT_LIST = {
    ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2, ge::DataType::DT_HIFLOAT8,
    ge::DataType::DT_FLOAT4_E2M1};
} // namespace

bool QuantBmmReduceScatterTiling::IsCapable()
{
    if (npuArch_ != NpuArch::DAV_3510) {
        OP_LOGI(opName_, "skip quantbmm reducescatter tiling when npuArch is not 3510.");
        return false;
    }
    // geAType 和 geBType 为fp8e4m3/fp8e5m2/hif8时, 走该tiling流程
    if (mc2tiling::CheckDataTypeVaild(args_.geAType, HIF8FP8FP4DTYPE_SUPPORT_LIST) &&
        mc2tiling::CheckDataTypeVaild(args_.geBType, HIF8FP8FP4DTYPE_SUPPORT_LIST)) {
        OP_LOGI(opName_, "start with quantbmm reducescatter tiling.");
        return true;
    }
    OP_LOGI(opName_, "skip quantbmm reducescatter tiling as dtype not support");
    return false;
}

// amaxout 和 quantscale为nullptr
bool QuantBmmReduceScatterTiling::CommonParamCheck() const
{
    auto quantscale = context_->GetOptionalInputShape(QUANTSCALE_INDEX);
    OP_TILING_CHECK(quantscale != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "quantscale", "not nullptr",
                                                          "The value of quantscale must be nullptr"),
                    return false);

    auto amaxShape = context_->GetOutputShape(AMAX_INDEX);
    if (amaxShape != nullptr) {
        OP_LOGI(opName_, "amaxShapeDim0 is %lu", amaxShape->GetStorageShape().GetDim(0));
    }
    OP_TILING_CHECK((amaxShape != nullptr) && (amaxShape->GetStorageShape().GetDim(0) != 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "amax", std::to_string(amaxShape->GetStorageShape().GetDim(0)).c_str(),
                        "The value of amax must be nullptr or empty tensor"),
                    return false);
    return true;
}

ge::graphStatus QuantBmmReduceScatterTiling::CheckGroupSize() const
{
    auto attrsPtr = context_->GetAttrs();
    OP_TILING_CHECK(attrsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);

    auto groupSizePtr = attrsPtr->GetAttrPointer<int64_t>(GROUPSIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "groupSize"), return ge::GRAPH_FAILED);
    mc2tiling::Mc2MatmulShapeInfo shapeInfo = {context_->GetInputShape(X1_INDEX),
                                               context_->GetInputShape(X2_INDEX),
                                               context_->GetOptionalInputShape(X1SCALE_INDEX),
                                               context_->GetOptionalInputShape(X2SCALE_INDEX),
                                               false,
                                               *context_->GetAttrs()->GetAttrPointer<bool>(TRANSPOSEB_INDEX),
                                               opName_};
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    if (quantMode_ == mc2tiling::Mc2QuantMode::MXFP_MODE) {
        shapeInfo.isMxfp = true;
        OP_TILING_CHECK(!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, groupSizeM, groupSizeN, groupSizeK),
                        OP_LOGE(opName_, "Failed to execute inferGroupSize in mx scene."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupSizeM != MXFP8_SIZE_M) || (groupSizeN != MXFP8_SIZE_N) || (groupSizeK != MXFP8_SIZE_K),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "groupSize",
                                                              ("[groupSizeM=" + std::to_string(groupSizeM) +
                                                               ", groupSizeN=" + std::to_string(groupSizeN) +
                                                               ", groupSizeK=" + std::to_string(groupSizeK) + "]")
                                                                  .c_str(),
                                                              "The value of groupSize must be [1, 1, 32] in mx scene"),
                        return ge::GRAPH_FAILED);
    } else if (quantMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) {
        OP_TILING_CHECK(
            *groupSizePtr != 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "groupSize", std::to_string(*groupSizePtr).c_str(),
                                                  "The value of groupSize must be 0 in pertensor scene"),
            return ge::GRAPH_FAILED);
    } else if (quantMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) {
        OP_TILING_CHECK(!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, groupSizeM, groupSizeN, groupSizeK),
                        OP_LOGE(opName_, "Failed to execute inferGroupSize in perblock scene."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupSizeM != PERBLOCK_SIZE) || (groupSizeN != PERBLOCK_SIZE) || (groupSizeK != PERBLOCK_SIZE),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            opName_, "groupSize",
                            ("[groupSizeM=" + std::to_string(groupSizeM) + ", groupSizeN=" +
                             std::to_string(groupSizeN) + ", groupSizeK=" + std::to_string(groupSizeK) + "]")
                                .c_str(),
                            "The value of groupSize must be [128, 128, 128] in perblock scene"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "quantMode", "unknown",
                                              "The value of quantMode must be pertensor, perblock or mxfp");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBmmReduceScatterTiling::CheckMxScaleDim(const gert::StorageShape *x1ScaleShape,
                                                             const gert::StorageShape *x2ScaleShape) const
{
    auto x1shape = context_->GetInputShape(X1_INDEX);
    auto x2shape = context_->GetInputShape(X2_INDEX);
    uint64_t x1Dim0 = x1shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2shape->GetStorageShape().GetDim(1);
    uint64_t x1ScaleDim0 = static_cast<uint64_t>(x1ScaleShape->GetStorageShape().GetDim(0));
    uint64_t x1ScaleDim1 = static_cast<uint64_t>(x1ScaleShape->GetStorageShape().GetDim(1));
    uint64_t x1ScaleDim2 = static_cast<uint64_t>(x1ScaleShape->GetStorageShape().GetDim(2));
    uint64_t x1Dim1DivMxFp8Size =
        static_cast<uint64_t>(((static_cast<int64_t>(x1Dim1) + MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    uint64_t x2ScaleDim0 = static_cast<uint64_t>(x2ScaleShape->GetStorageShape().GetDim(0));
    uint64_t x2Dim0DivMxFp8Size =
        static_cast<uint64_t>(((static_cast<int64_t>(x2Dim0) + MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    uint64_t x2Dim1DivMxFp8Size =
        static_cast<uint64_t>(((static_cast<int64_t>(x2Dim1) + MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    uint64_t x2ScaleDim1 = static_cast<uint64_t>(x2ScaleShape->GetStorageShape().GetDim(1));
    uint64_t x2ScaleDim2 = static_cast<uint64_t>(x2ScaleShape->GetStorageShape().GetDim(2));

    OP_TILING_CHECK((x1ScaleDim0 != x1Dim0) || (x1ScaleDim1 != x1Dim1DivMxFp8Size) || (x1ScaleDim2 != EVEN_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "x1Scale",
                        ("(" + std::to_string(x1ScaleDim0) + ", " + std::to_string(x1ScaleDim1) + ", " +
                         std::to_string(x1ScaleDim2) + ")")
                            .c_str(),
                        ("The shape of x1Scale must be (" + std::to_string(x1Dim0) + ", " +
                         std::to_string(x1Dim1DivMxFp8Size) + ", " + std::to_string(EVEN_ALIGN) + ")")
                            .c_str()),
                    return ge::GRAPH_FAILED);

    bool isTransposeB = *context_->GetAttrs()->GetAttrPointer<bool>(TRANSPOSEB_INDEX);
    bool nIsOne = (x1Dim1 == x2Dim0) ? (x2Dim1 == 1) : (x2Dim0 == 1);
    if (!nIsOne) {
        // Transposed Scenario
        if (isTransposeB) {
            OP_TILING_CHECK(
                (x2ScaleDim0 != x2Dim0) || (x2ScaleDim1 != x2Dim1DivMxFp8Size) || (x2ScaleDim2 != EVEN_ALIGN),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "x2Scale",
                    ("(" + std::to_string(x2ScaleDim0) + ", " + std::to_string(x2ScaleDim1) + ", " +
                     std::to_string(x2ScaleDim2) + ")")
                        .c_str(),
                    ("The shape of x2Scale must be (" + std::to_string(x2Dim0) + ", " +
                     std::to_string(x2Dim1DivMxFp8Size) + ", " + std::to_string(EVEN_ALIGN) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
        } else {
            OP_TILING_CHECK(
                (x2ScaleDim0 != x2Dim0DivMxFp8Size) || (x2ScaleDim1 != x2Dim1) || (x2ScaleDim2 != EVEN_ALIGN),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "x2Scale",
                    ("(" + std::to_string(x2ScaleDim0) + ", " + std::to_string(x2ScaleDim1) + ", " +
                     std::to_string(x2ScaleDim2) + ")")
                        .c_str(),
                    ("The shape of x2Scale must be (" + std::to_string(x2Dim0DivMxFp8Size) + ", " +
                     std::to_string(x2Dim1) + ", " + std::to_string(EVEN_ALIGN) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool QuantBmmReduceScatterTiling::PertensorSceneParamCheck(const gert::StorageShape *x1ScaleShape,
                                                           const gert::StorageShape *x2ScaleShape)
{
    OP_TILING_CHECK(
        (x1ScaleShape->GetStorageShape().GetDim(0) != 1) || (x2ScaleShape->GetStorageShape().GetDim(0) != 1),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "x1Scale/x2Scale",
                                              ("(" + std::to_string(x1ScaleShape->GetStorageShape().GetDim(0)) + ", " +
                                               std::to_string(x2ScaleShape->GetStorageShape().GetDim(0)) + ")")
                                                  .c_str(),
                                              "The shapes of x1Scale and x2Scale must be scalar (1D)"),
        return false);
    auto biasDesc = context_->GetOptionalInputDesc(static_cast<size_t>(BIAS_INDEX));
    auto biasShape = context_->GetInputShape(static_cast<size_t>(BIAS_INDEX));
    if ((biasDesc != nullptr) && (biasShape != nullptr)) {
        OP_TILING_CHECK(
            biasDesc->GetDataType() != ge::DataType::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "bias", Ops::Base::ToString(biasDesc->GetDataType()).c_str(),
                                                  "The dtype of bias must be DT_FLOAT"),
            return false);
        OP_TILING_CHECK(biasShape->GetStorageShape().GetDimNum() != 1,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "bias", std::to_string(biasShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of bias must be 1"),
                        return false);
        uint64_t biasDimValue = static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0));
        OP_TILING_CHECK(biasDimValue != args_.nValue,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "bias", std::to_string(biasDimValue).c_str(),
                                                              "The dimension of bias must be equal to n"),
                        return false);
    }

    return true;
}

bool QuantBmmReduceScatterTiling::PerblockSceneParamCheck(const gert::StorageShape *x1ScaleShape,
                                                          const gert::StorageShape *x2ScaleShape) const
{
    auto biasDesc = context_->GetOptionalInputShape(static_cast<size_t>(BIAS_INDEX));
    OP_TILING_CHECK(biasDesc != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "bias", "not nullptr",
                                                          "The value of bias must be nullptr in perblock scene"),
                    return false);

    auto x1shape = context_->GetInputShape(X1_INDEX);
    auto x2shape = context_->GetInputShape(X2_INDEX);
    uint64_t x1ScaleDim0 = static_cast<uint64_t>(x1ScaleShape->GetStorageShape().GetDim(0));
    uint64_t x1Dim0DivBlocksize =
        ops::CeilDiv(static_cast<uint64_t>(x1shape->GetStorageShape().GetDim(0)), PERBLOCK_SIZE);
    uint64_t x1ScaleDim1 = static_cast<uint64_t>(x1ScaleShape->GetStorageShape().GetDim(1));
    uint64_t x1Dim1DivBlocksize =
        ops::CeilDiv(static_cast<uint64_t>(x1shape->GetStorageShape().GetDim(1)), PERBLOCK_SIZE);
    uint64_t x2ScaleDim0 = static_cast<uint64_t>(x2ScaleShape->GetStorageShape().GetDim(0));
    uint64_t x2Dim0DivBlocksize =
        ops::CeilDiv(static_cast<uint64_t>(x2shape->GetStorageShape().GetDim(0)), PERBLOCK_SIZE);
    uint64_t x2ScaleDim1 = static_cast<uint64_t>(x2ScaleShape->GetStorageShape().GetDim(1));
    uint64_t x2Dim1DivBlocksize =
        ops::CeilDiv(static_cast<uint64_t>(x2shape->GetStorageShape().GetDim(1)), PERBLOCK_SIZE);
    OP_TILING_CHECK(
        (x1ScaleDim0 != x1Dim0DivBlocksize) || (x1ScaleDim1 != x1Dim1DivBlocksize),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "x1Scale", ("(" + std::to_string(x1ScaleDim0) + ", " + std::to_string(x1ScaleDim1) + ")").c_str(),
            "The shape of x1Scale must be (x1Dim0DivBlocksize, x1Dim1DivBlocksize)"),
        return false);
    OP_TILING_CHECK(
        (x2ScaleDim0 != x2Dim0DivBlocksize) || (x2ScaleDim1 != x2Dim1DivBlocksize),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "x2Scale", ("(" + std::to_string(x2ScaleDim0) + ", " + std::to_string(x2ScaleDim1) + ")").c_str(),
            "The shape of x2Scale must be (x2Dim0DivBlocksize, x2Dim1DivBlocksize)"),
        return false);
    return true;
}

bool QuantBmmReduceScatterTiling::MxfpSceneParamCheck(const gert::StorageShape *x1ScaleShape,
                                                      const gert::StorageShape *x2ScaleShape)
{
    OP_TILING_CHECK(!mc2tiling::CheckDataTypeVaild(args_.geAType, MXDTYPE_SUPPORT_LIST) ||
                        !mc2tiling::CheckDataTypeVaild(args_.geBType, MXDTYPE_SUPPORT_LIST),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "x1/x2",
                        (std::string(Ops::Base::ToString(args_.geAType).c_str()) + "/" +
                         std::string(Ops::Base::ToString(args_.geBType).c_str()))
                            .c_str(),
                        "The dtype of x1 and x2 must be float8_e4m3fn, float8_e5m2 or float4_e2m1"),
                    return false);
    // MXFP4场景强制x1/x2类型相同
    if (isMxfp4_) {
        OP_TILING_CHECK(args_.geAType != args_.geBType,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "x1/x2",
                            (std::string(Ops::Base::ToString(args_.geAType).c_str()) + "/" +
                             std::string(Ops::Base::ToString(args_.geBType).c_str()))
                                .c_str(),
                            "In MXFP4 quant mode, the dtype of x1 and x2 must be the same (DT_FLOAT4_E2M1)"),
                        return false);
        OP_TILING_CHECK(args_.kValue % EVEN_ALIGN != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "kValue", std::to_string(args_.kValue).c_str(),
                                                              "In MXFP4 quant mode, kValue must be even"),
                        return false);
    }
    OP_TILING_CHECK(CheckMxScaleDim(x1ScaleShape, x2ScaleShape) == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "CheckMxScaleDim failed!"), return false);
    return true;
}

ge::graphStatus QuantBmmReduceScatterTiling::CheckScale() const
{
    // x1scale x2scale判空
    auto x1ScaleShape = context_->GetOptionalInputShape(X1SCALE_INDEX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2SCALE_INDEX);
    OP_TILING_CHECK(((x1ScaleShape == nullptr)), OP_LOGE_WITH_INVALID_INPUT(opName_, "x1Scale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((x2ScaleShape == nullptr)), OP_LOGE_WITH_INVALID_INPUT(opName_, "x2Scale"),
                    return ge::GRAPH_FAILED);

    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1SCALE_INDEX);
    auto x2ScaleDesc = context_->GetOptionalInputDesc(X2SCALE_INDEX);
    OP_TILING_CHECK(((x1ScaleDesc == nullptr)), OP_LOGE_WITH_INVALID_INPUT(opName_, "x1Scale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((x2ScaleDesc == nullptr)), OP_LOGE_WITH_INVALID_INPUT(opName_, "x2Scale"),
                    return ge::GRAPH_FAILED);

    // 低精度x1scale x2scale 数据类型为fp32或fp8_e8m0
    OP_TILING_CHECK(((x1ScaleDesc->GetDataType() != ge::DataType::DT_FLOAT) &&
                     (x2ScaleDesc->GetDataType() != ge::DataType::DT_FLOAT)) &&
                        ((x1ScaleDesc->GetDataType() != ge::DataType::DT_FLOAT8_E8M0) &&
                         (x2ScaleDesc->GetDataType() != ge::DataType::DT_FLOAT8_E8M0)),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "x1Scale/x2Scale",
                        (std::string(Ops::Base::ToString(x1ScaleDesc->GetDataType()).c_str()) + "/" +
                         std::string(Ops::Base::ToString(x2ScaleDesc->GetDataType()).c_str()))
                            .c_str(),
                        "The dtype of x1Scale and x2Scale must be float32 or float8_e8m0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x1ScaleDesc->GetDataType() != x2ScaleDesc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "x1Scale/x2Scale",
                        (std::string(Ops::Base::ToString(x1ScaleDesc->GetDataType()).c_str()) + "/" +
                         std::string(Ops::Base::ToString(x2ScaleDesc->GetDataType()).c_str()))
                            .c_str(),
                        "The dtype of x1Scale and x2Scale must be the same"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBmmReduceScatterTiling::CheckInput()
{
    OP_TILING_CHECK(CheckScale() == ge::GRAPH_FAILED, OP_LOGE(opName_, "Check scale failed!"), return ge::GRAPH_FAILED);
    auto x1ScaleShape = context_->GetOptionalInputShape(X1SCALE_INDEX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2SCALE_INDEX);
    // 目前只有pertensor和perblock和mxfp场景
    SetScene();
    OP_TILING_CHECK(
        ((quantMode_ != mc2tiling::Mc2QuantMode::PERBLOCK_MODE) &&
         (quantMode_ != mc2tiling::Mc2QuantMode::PERTENSOR_MODE) && (quantMode_ != mc2tiling::Mc2QuantMode::MXFP_MODE)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "quantMode",
            ("x1scaleDim=" + std::to_string(x1ScaleShape->GetStorageShape().GetDimNum()) +
             ", x2scaleDim=" + std::to_string(x2ScaleShape->GetStorageShape().GetDimNum()))
                .c_str(),
            "The value of quantMode must be pertensor, perblock or mxfp"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((args_.geAType == ge::DataType::DT_HIFLOAT8) && (args_.geAType != args_.geBType)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "BType", Ops::Base::ToString(args_.geBType).c_str(),
            "If the dtype of AType is hifp8 in pertensor scene, the value of BType must be equal to that of AType"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK((!CommonParamCheck()), OP_LOGE(opName_, "Common params check failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckGroupSize() == ge::GRAPH_FAILED, OP_LOGE(opName_, "Check block size and axis failed!"),
                    return ge::GRAPH_FAILED);
    if (quantMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) {
        OP_TILING_CHECK((!PertensorSceneParamCheck(x1ScaleShape, x2ScaleShape)),
                        OP_LOGE(opName_, "Pertensor scene params check failed."), return ge::GRAPH_FAILED);
    } else if (quantMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) {
        OP_TILING_CHECK((!PerblockSceneParamCheck(x1ScaleShape, x2ScaleShape)),
                        OP_LOGE(opName_, "perblock scene params check failed."), return ge::GRAPH_FAILED);
    } else if (quantMode_ == mc2tiling::Mc2QuantMode::MXFP_MODE) {
        OP_TILING_CHECK((!MxfpSceneParamCheck(x1ScaleShape, x2ScaleShape)),
                        OP_LOGE(opName_, "Mx scene params check failed."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBmmReduceScatterTiling::SetMc2Hcomm()
{
    const uint32_t reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    const uint32_t opType = isA2APath_ ? static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL) :
                                         static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER);

    const std::string rsConfig = isA2APath_ ? "AlltoAll=level0:fullmesh" : "ReduceScatter=level0:fullmesh";

    int index = 0;
    auto group = context_->GetAttrs()->GetAttrPointer<char>(index++);
    uint32_t dataType = static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType));

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, rsConfig, reduceType, dataType, dataType);
    if (commMode_ == TPL_AICPU_COMM_MODE) {
        mc2CcTilingConfig.SetCommEngine(mc2tiling::A5_AICPU_TS_ENGINE);
        OP_LOGD(opName_, "[SetCommEngine] Set CommEngine to AiCPU for quant_bmm_reduce_scatter_tiling.");
    } else {
        mc2CcTilingConfig.SetCommEngine(mc2tiling::A5_CCU_ENGINE);
        OP_LOGD(opName_, "[SetCommEngine] Set CommEngine to CCU for quant_bmm_reduce_scatter_tiling.");
    }
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(quantBmmMatmulReducescatterTilingData_->mc2InitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(quantBmmMatmulReducescatterTilingData_->mc2CcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QuantBmmReduceScatterTiling::SetScene()
{
    quantMode_ = mc2tiling::Mc2QuantMode::INVALID_MODE;
    isMxfp4_ = false; // 初始化为false
    auto x1ScaleShape = context_->GetOptionalInputShape(X1SCALE_INDEX);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2SCALE_INDEX);
    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1SCALE_INDEX);
    auto x2ScaleDesc = context_->GetOptionalInputDesc(X2SCALE_INDEX);

    OP_TILING_CHECK((x1ScaleShape->GetStorageShape().GetDimNum() != x2ScaleShape->GetStorageShape().GetDimNum()),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "x1Scale/x2Scale",
                        ("x1Scale=" + std::to_string(x1ScaleShape->GetStorageShape().GetDimNum()) +
                         "D, x2Scale=" + std::to_string(x2ScaleShape->GetStorageShape().GetDimNum()) + "D")
                            .c_str(),
                        "The current shape combination does not match any scenario."
                        "(PerTensor scenario: [x1Scale=1D, x2Scale=1D]; "
                        "PerBlock scenario: [x1Scale=2D, x2Scale=2D]; MXFP scenario: [x1Scale=3D, x2Scale=3D])"),
                    return);

    if ((x1ScaleShape->GetStorageShape().GetDimNum() == PERTENSOR_SCALE_DIM) &&
        (x2ScaleShape->GetStorageShape().GetDimNum() == PERTENSOR_SCALE_DIM)) {
        quantMode_ = mc2tiling::Mc2QuantMode::PERTENSOR_MODE;
    } else if ((x1ScaleShape->GetStorageShape().GetDimNum() == MX_SCALE_DIM) &&
               (x2ScaleShape->GetStorageShape().GetDimNum() == MX_SCALE_DIM) &&
               (x1ScaleDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0) &&
               (x2ScaleDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        quantMode_ = mc2tiling::Mc2QuantMode::MXFP_MODE;
        // MXFP4场景识别：当x1和x2均为FP4_E2M1时，设置isMxfp4_=true
        if (args_.geAType == ge::DataType::DT_FLOAT4_E2M1 && args_.geBType == ge::DataType::DT_FLOAT4_E2M1) {
            isMxfp4_ = true;
            OP_LOGI(opName_, "MXFP4 quant scene detected: x1/x2 dtype is DT_FLOAT4_E2M1");
        }
    } else if ((x1ScaleShape->GetStorageShape().GetDimNum() == PERBLOCK_SCALE_DIM) &&
               (x2ScaleShape->GetStorageShape().GetDimNum() == PERBLOCK_SCALE_DIM) &&
               (x1ScaleDesc->GetDataType() == ge::DataType::DT_FLOAT) &&
               (x2ScaleDesc->GetDataType() == ge::DataType::DT_FLOAT)) {
        quantMode_ = mc2tiling::Mc2QuantMode::PERBLOCK_MODE;
    }
    return;
}

bool QuantBmmReduceScatterTiling::CheckPerblockM() { return args_.orgMValue % (PERBLOCK_SIZE * args_.rankDim) == 0; }

ge::graphStatus QuantBmmReduceScatterTiling::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckHCCLSize());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    OP_TILING_CHECK(SetMc2Hcomm() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Tiling SetHcommCfg failed."),
                    return ge::GRAPH_FAILED);
    SetRcsTilingData(MutableRCSTilingDataA5());

    // 在perblock量化场景，orgMValue不满足PERBLOCK_SIZE * args_.rankDim整数倍时，不做公式化切分，走计算通信串行
    if ((quantMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) || (quantMode_ == mc2tiling::Mc2QuantMode::MXFP_MODE) ||
        (quantMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE && CheckPerblockM())) {
        MC2_CHECK_LOG_RET(opName_, DoSplitMTiling(MutableRCSTilingDataA5()));
    }
    MC2_CHECK_LOG_RET(opName_, AdjustHCCLLimit(MutableRCSTilingDataA5(), GetQuantMode()));
    MC2_CHECK_LOG_RET(opName_, DoAdaptSlidWindowTiling());
    if ((GetQuantMode() == mc2tiling::Mc2QuantMode::PERBLOCK_MODE) && (MutableRCSTilingDataA5().tileCnt == 0) &&
        (MutableRCSTilingDataA5().tailCnt == 0)) {
        MutableRCSTilingDataA5().tileCnt = 1;
    }
    SetTilingResult(MutableRCSTilingDataA5(), MutableTCubeTileTilingData(), MutableTCubeTailTilingData(),
                    quantBmmMatmulReducescatterTilingData_->debugMode,
                    quantBmmMatmulReducescatterTilingData_->dataType);
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantBmmReduceScatterTiling::GetTilingKey() const
{
    bool enableNd2Nz = false;              // quantbatchmatmul输入数据格式为ND
    bool isCastBias = false;               // david上支持bias数据类型为bf16
    uint8_t commAlgorithm = args_.commAlg; // fullmesh
    uint8_t outputType = 0;
    uint32_t yDType = static_cast<uint32_t>(*context_->GetAttrs()->GetAttrPointer<uint64_t>(YDTYPE_INDEX));
    if (yDType == DTYPE_ENUM_FLOAT) {
        outputType = OUTPUT_TYPE_FLOAT;
    } else if ((yDType == DTYPE_ENUM_FLOAT16) || (yDType == DTYPE_ENUM_BF16)) {
        outputType = static_cast<uint8_t>(0);
    }

    uint8_t scaleType = 0;
    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1SCALE_INDEX);
    auto x2ScaleDesc = context_->GetOptionalInputDesc(X2SCALE_INDEX);
    if ((x1ScaleDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0) &&
        (x2ScaleDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        scaleType = static_cast<uint8_t>(1);
    } else {
        scaleType = static_cast<uint8_t>(0);
    }

    // MXFP4场景设置inputType为INPUT_TYPE_IS_FP4
    uint8_t inputType = INPUT_TYPE_IS_FP8;
    if (isMxfp4_) {
        inputType = INPUT_TYPE_IS_FP4;
        OP_LOGD(opName_, "MXFP4 scene detected, setting inputType to INPUT_TYPE_IS_FP4 (2)");
    }

    bool isPerBlock = quantMode_ == mc2tiling::Mc2QuantMode::PERBLOCK_MODE;
    uint8_t commAlg = isA2APath_ ? TPL_CCU_ALL2ALL_VEC_REDUCE : TPL_CCU_REDUCESUM;
    uint64_t tilingKey = GET_TPL_TILING_KEY(isPerBlock, args_.isATrans, args_.isBTrans, inputType, outputType,
                                            scaleType, commAlg, commMode_);
    OP_LOGD(opName_, "isPerBlock, transA, transB is: [%d, %d, %d]", isPerBlock, args_.isATrans, args_.isBTrans);
    OP_LOGD(opName_, "inputType, outputType, scaleType is: [%u, %u, %u]", inputType, outputType, scaleType);
    return tilingKey;
}

ge::graphStatus QuantBmmReduceScatterTiling::GetWorkspaceSize()
{
    // A2A 路径需要 3 倍 cToFloatLen (senBuf + recvBuf + matmul)，否则只需 1 倍
    uint32_t factor = isA2APath_ ? 3 : 1;
    myWorkSpaceSize_ = myWorkSpaceSize_ + MutableRCSTilingDataA5().cToFloatLen * factor;

    OP_LOGI(opName_, "set max workspace size %lu to context", myWorkSpaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "WorkspaceSizes");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

void PrintTCubeTilingParams(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3DataParams &tiling)
{
    OP_LOGD(opName, "tiling.batchA %d", tiling.batchA);
    OP_LOGD(opName, "tiling.batchB %d", tiling.batchB);
    OP_LOGD(opName, "tiling.batchC %d", tiling.batchC);
    OP_LOGD(opName, "tiling.batchA1 %d", tiling.batchA1);
    OP_LOGD(opName, "tiling.batchA2 %d", tiling.batchA2);
    OP_LOGD(opName, "tiling.batchA3 %d", tiling.batchA3);
    OP_LOGD(opName, "tiling.batchA4 %d", tiling.batchA4);
    OP_LOGD(opName, "tiling.batchB1 %d", tiling.batchB1);
    OP_LOGD(opName, "tiling.batchB2 %d", tiling.batchB2);
    OP_LOGD(opName, "tiling.batchB3 %d", tiling.batchB3);
    OP_LOGD(opName, "tiling.batchB4 %d", tiling.batchB4);
    OP_LOGD(opName, "tiling.batchC1 %d", tiling.batchC1);
    OP_LOGD(opName, "tiling.batchC2 %d", tiling.batchC2);
    OP_LOGD(opName, "tiling.batchC3 %d", tiling.batchC3);
    OP_LOGD(opName, "tiling.batchC4 %d", tiling.batchC4);
    OP_LOGD(opName, "tiling.singleCoreBatch %d", tiling.singleCoreBatch);
    OP_LOGD(opName, "tiling.realSingleCoreM %d", tiling.realSingleCoreM);
    OP_LOGD(opName, "tiling.realSingleCoreN %d", tiling.realSingleCoreN);
    OP_LOGD(opName, "tiling.isPerTensor %d", tiling.isPerTensor);
    OP_LOGD(opName, "tiling.isDoubleScale %d", tiling.isDoubleScale);
    OP_LOGD(opName, "tiling.isPertoken %d", tiling.isPertoken);
    OP_LOGD(opName, "tiling.biasThreeDim %d", tiling.biasThreeDim);
    OP_LOGD(opName, "tiling.ubCalcM %d", tiling.ubCalcM);
    OP_LOGD(opName, "tiling.ubCalcN %d", tiling.ubCalcN);
    OP_LOGD(opName, "tiling.needUbBuffer %d", tiling.needUbBuffer);
    OP_LOGD(opName, "tiling.biasDtype %d", tiling.biasDtype);
    OP_LOGD(opName, "tiling.ubSize %d", tiling.ubSize);
    OP_LOGD(opName, "tiling.groupSizeM %d", tiling.groupSizeM);
    OP_LOGD(opName, "tiling.groupSizeN %d", tiling.groupSizeN);
    OP_LOGD(opName, "tiling.groupSizeK %d", tiling.groupSizeK);
    OP_LOGD(opName, "tiling.isMClash %d", tiling.isMClash);
    OP_LOGD(opName, "tiling.isNClash %d", tiling.isNClash);
}

void PrintTCubeTilingWindowParam(const std::string &opName, DequantBmm::Mc2SlidingWindowParams &tiling)
{
    OP_LOGD(opName, "tiling.mTailTile %d", tiling.mTailTile);
    OP_LOGD(opName, "tiling.nTailTile %d", tiling.nTailTile);
}

void PrintTCubeTilingL2cache(const std::string &opName, DequantBmm::Mc2L2cacheTileParams &tiling)
{
    OP_LOGD(opName, "tiling.calOrder %d", tiling.calOrder);
    OP_LOGD(opName, "tiling.mTileCntL2 %d", tiling.mTileCntL2);
    OP_LOGD(opName, "tiling.nTileCntL2 %d", tiling.nTileCntL2);
    OP_LOGD(opName, "tiling.mTileBlock %d", tiling.mTileBlock);
    OP_LOGD(opName, "tiling.nTileBlock %d", tiling.nTileBlock);
    OP_LOGD(opName, "tiling.isBasicTiling %d", tiling.isBasicTiling);
}

ge::graphStatus QuantBmmReduceScatterTiling::PostTiling()
{
    auto rawTilingDataPtr = context_->GetRawTilingData();
    OP_TILING_CHECK((rawTilingDataPtr == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "rawTilingDataPtr"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(opName_, "final tiling data size: %zu and context capacity size: %zu ",
            sizeof(QuantBatchMatmulV3ReduceScatterTilingData), rawTilingDataPtr->GetCapacity());
    rawTilingDataPtr->SetDataSize(sizeof(QuantBatchMatmulV3ReduceScatterTilingData));

    OP_TILING_CHECK(
        sizeof(QuantBatchMatmulV3ReduceScatterTilingData) % sizeof(uint64_t) != 0,
        OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", sizeof(QuantBatchMatmulV3ReduceScatterTilingData)),
        return ge::GRAPH_FAILED);
    if (MutableRCSTilingDataA5().rankID == 0) {
        PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingDataA5());
        PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTileTilingData());
        PrintTCubeTilingParams(context_->GetNodeName(), MutableTCubeTilingParam());
        PrintTCubeTilingWindowParam(context_->GetNodeName(), MutableTCubeTilingSlidingWindow());
        PrintTCubeTilingL2cache(context_->GetNodeName(), MutableTCubeTilingL2cache());
        if (MutableRCSTilingDataA5().tailM > 0) {
            OP_LOGD(opName_, "tail exist");
            PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTailTilingData());
            PrintTCubeTilingParams(context_->GetNodeName(), MutableTailTCubeTilingParam());
            PrintTCubeTilingWindowParam(context_->GetNodeName(), MutableTailTCubeTilingSlidingWindow());
            PrintTCubeTilingL2cache(context_->GetNodeName(), MutableTailTCubeTilingL2cache());
        }
    }
    uint32_t usedCoreNum = std::max(MutableTCubeTileTilingData().usedCoreNum, MutableTCubeTailTilingData().usedCoreNum);
    context_->SetBlockDim(usedCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::RCSTiling &QuantBmmReduceScatterTiling::MutableRCSTilingDataA5() const
{
    return quantBmmMatmulReducescatterTilingData_->param;
}

::TCubeTiling &QuantBmmReduceScatterTiling::MutableTCubeTileTilingData() const
{
    return quantBmmMatmulReducescatterTilingData_->quantBmmV3TileTiling.matmulTiling;
}

DequantBmm::Mc2QuantBatchMatmulV3DataParams &QuantBmmReduceScatterTiling::MutableTCubeTilingParam() const
{
    return quantBmmMatmulReducescatterTilingData_->quantBmmV3TileTiling.params;
}

DequantBmm::Mc2L2cacheTileParams &QuantBmmReduceScatterTiling::MutableTCubeTilingL2cache() const
{
    return quantBmmMatmulReducescatterTilingData_->quantBmmV3TileTiling.tileL2cacheTiling;
}

DequantBmm::Mc2SlidingWindowParams &QuantBmmReduceScatterTiling::MutableTCubeTilingSlidingWindow() const
{
    return quantBmmMatmulReducescatterTilingData_->quantBmmV3TileTiling.adaptiveSlidingWin;
}

::TCubeTiling &QuantBmmReduceScatterTiling::MutableTCubeTailTilingData() const
{
    return quantBmmMatmulReducescatterTilingData_->quantBmmV3TailTiling.matmulTiling;
}

DequantBmm::Mc2QuantBatchMatmulV3DataParams &QuantBmmReduceScatterTiling::MutableTailTCubeTilingParam() const
{
    return quantBmmMatmulReducescatterTilingData_->quantBmmV3TailTiling.params;
}

DequantBmm::Mc2L2cacheTileParams &QuantBmmReduceScatterTiling::MutableTailTCubeTilingL2cache() const
{
    return quantBmmMatmulReducescatterTilingData_->quantBmmV3TailTiling.tileL2cacheTiling;
}

DequantBmm::Mc2SlidingWindowParams &QuantBmmReduceScatterTiling::MutableTailTCubeTilingSlidingWindow() const
{
    return quantBmmMatmulReducescatterTilingData_->quantBmmV3TailTiling.adaptiveSlidingWin;
}

ge::graphStatus QuantBmmReduceScatterTiling::DoAdaptSlidWindowTiling()
{
    // 主块切分
    uint32_t tempMValue = tileMValue_ == 0 ? MutableRCSTilingDataA5().rankM : tileMValue_;
    args_.mValue = ((quantMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) ||
                    (quantMode_ == mc2tiling::Mc2QuantMode::MXFP_MODE)) ?
                       (tempMValue * args_.rankDim) :
                       tempMValue;

    QuantBmmReduceScatterHelper mmTile(*this, quantBmmMatmulReducescatterTilingData_->quantBmmV3TileTiling);
    MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
    if (MutableRCSTilingDataA5().tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }

    // 尾块切分
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams tailQuantBatchMatmulParams;
    args_.mValue = ((quantMode_ == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) ||
                    (quantMode_ == mc2tiling::Mc2QuantMode::MXFP_MODE)) ?
                       (tailMValue_ * args_.rankDim) :
                       tailMValue_;
    QuantBmmReduceScatterHelper mmTail(*this, quantBmmMatmulReducescatterTilingData_->quantBmmV3TailTiling);
    MC2_CHECK_LOG_RET(opName_, mmTail.DoTiling());
    return ge::GRAPH_SUCCESS;
}

QuantBmmReduceScatterTiling::QuantBmmReduceScatterTiling(gert::TilingContext *context)
    : MatmulReduceScatterTilingBase(context),
      quantBmmMatmulReducescatterTilingData_(&quantBmmMatmulReducescatterTilingDataSelf_)
{
    quantBmmMatmulReducescatterTilingData_ = context->GetTilingData<QuantBatchMatmulV3ReduceScatterTilingData>();
}

void QuantBmmReduceScatterHelper::AnalyzeBatchInfo(const gert::Shape &oriShapeA, const gert::Shape &oriShapeB)
{
    (void)oriShapeA;
    (void)oriShapeB;
    auto x1Shape = GetX1Shape(X1_INDEX);
    auto x2Shape = GetX2Shape(X2_INDEX);
    int32_t numDimA = static_cast<int32_t>(x1Shape.GetDimNum());
    int32_t numDimB = static_cast<int32_t>(x2Shape.GetDimNum());
    inputParams_.batchA4 = numDimA > IDX_K_LOW ? x1Shape.GetDim(numDimA - IDX_K_HIGH) : 1;
    inputParams_.batchA3 = numDimA > IDX_K_HIGH ? x1Shape.GetDim(numDimA - IDX_N_LOW) : 1;
    inputParams_.batchA2 = numDimA > IDX_N_LOW ? x1Shape.GetDim(numDimA - IDX_N_HIGH) : 1;
    inputParams_.batchA1 = numDimA > IDX_N_HIGH ? x1Shape.GetDim(numDimA - IDX_B_LOW) : 1;
    inputParams_.batchB4 = numDimB > IDX_K_LOW ? x2Shape.GetDim(numDimB - IDX_K_HIGH) : 1;
    inputParams_.batchB3 = numDimB > IDX_K_HIGH ? x2Shape.GetDim(numDimB - IDX_N_LOW) : 1;
    inputParams_.batchB2 = numDimB > IDX_N_LOW ? x2Shape.GetDim(numDimB - IDX_N_HIGH) : 1;
    inputParams_.batchB1 = numDimB > IDX_N_HIGH ? x2Shape.GetDim(numDimB - IDX_B_LOW) : 1;
    auto outShape = GetOutputShape(0);
    int32_t numDimC = static_cast<int32_t>(outShape.GetDimNum());
    inputParams_.batchC4 = numDimC > IDX_K_LOW ? outShape.GetDim(numDimC - IDX_K_HIGH) : 1UL;
    inputParams_.batchC3 = numDimC > IDX_K_HIGH ? outShape.GetDim(numDimC - IDX_N_LOW) : 1UL;
    inputParams_.batchC2 = numDimC > IDX_N_LOW ? outShape.GetDim(numDimC - IDX_N_HIGH) : 1UL;
    inputParams_.batchC1 = numDimC > IDX_N_HIGH ? outShape.GetDim(numDimC - IDX_B_LOW) : 1UL;
}

void QuantBmmReduceScatterHelper::SetBatch()
{
    // perblock量化非串行场景将batch4_设置成卡数
    if (inputParams_.isPerBlock && !isSerial_) {
        batch4_ = tilingArgs_.rankDim;
    }
}

const gert::Shape QuantBmmReduceScatterHelper::GetX1Shape(const size_t index)
{
    (void)index;
    SetBatch();
    if (tilingArgs_.isATrans) {
        return gert::Shape({batch1_, batch2_, batch3_, batch4_, static_cast<int64_t>(tilingArgs_.kValue),
                            static_cast<int64_t>(tilingArgs_.mValue)});
    }
    return gert::Shape({batch1_, batch2_, batch3_, batch4_, static_cast<int64_t>(tilingArgs_.mValue),
                        static_cast<int64_t>(tilingArgs_.kValue)});
}

const gert::Shape QuantBmmReduceScatterHelper::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingArgs_.isBTrans) {
        return gert::Shape(
            {1, 1, 1, 1, static_cast<int64_t>(tilingArgs_.nValue), static_cast<int64_t>(tilingArgs_.kValue)});
    }
    return gert::Shape(
        {1, 1, 1, 1, static_cast<int64_t>(tilingArgs_.kValue),
         static_cast<int64_t>(tilingArgs_.nValue)}); // x2构造4维全1 Batch，防止matmul规则自动融合x1/output的batch轴
}

const gert::Shape QuantBmmReduceScatterHelper::GetOutputShape(const size_t index)
{
    (void)index;
    SetBatch();
    return gert::Shape({batch1_, batch2_, batch3_, batch4_, static_cast<int64_t>(tilingArgs_.mValue),
                        static_cast<int64_t>(tilingArgs_.nValue)});
}

const gert::Shape &QuantBmmReduceScatterHelper::GetScaleShape(const size_t index)
{
    (void)index;
    if (context_->GetOptionalInputShape(static_cast<size_t>(X2SCALE_INDEX)) == nullptr) {
        OP_LOGE(inputParams_.opName, "has dequant, but has no x2scale shape");
        return defaultShape;
    }
    return context_->GetOptionalInputShape(static_cast<size_t>(X2SCALE_INDEX))->GetStorageShape();
}

// matmul 将protoken 作为第二路scale输入
const gert::StorageShape *QuantBmmReduceScatterHelper::GetPertokenShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(X1SCALE_INDEX));
}

const gert::StorageShape *QuantBmmReduceScatterHelper::GetBiasShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(BIAS_INDEX));
}

const gert::StorageShape *QuantBmmReduceScatterHelper::GetOffsetShape(const size_t index) const
{
    (void)index;
    return static_cast<const gert::StorageShape *>(nullptr);
}

ge::graphStatus QuantBmmReduceScatterHelper::GetShapeAttrsInfo()
{
    inputParams_.opName = tilingProcesser_.GetOpName();
    OP_LOGI(inputParams_.opName, "Start assemble input params for slidWindow matmul tiling");
    inputParams_.transA = tilingArgs_.isATrans;
    inputParams_.transB = tilingArgs_.isBTrans;
    inputParams_.hasBias = tilingArgs_.isBias;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.GetLibApiWorkSpaceSize();
    inputParams_.mSize = tilingArgs_.mValue;
    inputParams_.kSize = tilingArgs_.kValue;
    inputParams_.nSize = tilingArgs_.nValue;
    inputParams_.aDtype = tilingArgs_.geAType;
    inputParams_.bDtype = tilingArgs_.geBType;
    int yDType = *context_->GetAttrs()->GetAttrPointer<uint64_t>(YDTYPE_INDEX);
    // quantbatchmatmul输出的数据类型和output类型一致
    inputParams_.cDtype = tilingArgs_.geCType;
    inputParams_.outDtype = static_cast<int64_t>(yDType);
    OP_LOGI(inputParams_.opName, "yDType is %ld", inputParams_.outDtype);
    inputParams_.biasDtype = tilingArgs_.isBias ? tilingArgs_.geBiasType : ge::DT_FLOAT;
    auto scaleTensorDesc = context_->GetOptionalInputDesc(X2SCALE_INDEX);
    auto perTokenScaleTensorDesc = context_->GetOptionalInputDesc(X1SCALE_INDEX);
    OP_TILING_CHECK((scaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "scale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((perTokenScaleTensorDesc == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "pertoken_scale"), return ge::GRAPH_FAILED);
    inputParams_.scaleDtype = scaleTensorDesc->GetDataType();
    inputParams_.perTokenScaleDtype = perTokenScaleTensorDesc->GetDataType();
    inputParams_.isPerTensor = ((tilingProcesser_.GetQuantMode() == mc2tiling::Mc2QuantMode::PERTENSOR_MODE) ||
                                (tilingProcesser_.GetQuantMode() == mc2tiling::Mc2QuantMode::MXFP_MODE));
    inputParams_.isPerBlock = (tilingProcesser_.GetQuantMode() == mc2tiling::Mc2QuantMode::PERBLOCK_MODE);
    inputParams_.isDoubleScale = true;
    // 当orgMValue在perblock量化场景不满足PERBLOCK_SIZE * rankDim的整数倍，mValue不做切分，设置为计算和通信串行
    isSerial_ = (tilingArgs_.mValue == tilingArgs_.orgMValue) && inputParams_.isPerBlock;
    if (inputParams_.isPerBlock) {
        inputParams_.groupSizeM = PERBLOCK_SIZE;
        inputParams_.groupSizeK = PERBLOCK_SIZE;
        inputParams_.groupSizeN = PERBLOCK_SIZE;
    } else if ((scaleTensorDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0) &&
               (perTokenScaleTensorDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        inputParams_.groupSizeM = MXFP8_SIZE_M;
        inputParams_.groupSizeK = MXFP8_SIZE;
        inputParams_.groupSizeN = MXFP8_SIZE_N;
    }
    MC2_CHECK_TRUE_RET(inputParams_.opName, AnalyzeInputs());
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBmmReduceScatterHelper::PostTiling()
{
    tilingProcesser_.SetMyWorkSpaceSize(std::max(tilingProcesser_.GetMyWorkSpaceSize(), workspaceSize_));
    OP_LOGI(inputParams_.opName, "set mm workspace size %lu to mc2", tilingProcesser_.GetMyWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBmmReduceScatterHelper::DoLibApiTiling()
{
    MC2_CHECK_LOG_RET(inputParams_.opName, Mc2AdaptiveSlidingWindowTiling::DoLibApiTiling());
    isBf16Opt_ = false;
    return ge::GRAPH_SUCCESS;
}

void QuantBmmReduceScatterHelper::PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantBatchMatmulInfo) const
{
    OP_LOGD(inputParams_.opName, "transA_ = %d transB_ = %d, hasBias_ = %d", quantBatchMatmulInfo.transA,
            quantBatchMatmulInfo.transB, quantBatchMatmulInfo.hasBias);
    OP_LOGD(inputParams_.opName, "mSize_ = %ld kSize_ = %ld nSize_ = %ld libApiWorkSpaceSize = %u",
            quantBatchMatmulInfo.mSize, quantBatchMatmulInfo.kSize, quantBatchMatmulInfo.nSize,
            quantBatchMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(inputParams_.opName, "aDtype_ = %d bDtype_ = %d cDtype_ = %d biasDtype_ = %d outDtype = %ld",
            static_cast<int32_t>(quantBatchMatmulInfo.aDtype), static_cast<int32_t>(quantBatchMatmulInfo.bDtype),
            static_cast<int32_t>(quantBatchMatmulInfo.cDtype), static_cast<int32_t>(quantBatchMatmulInfo.biasDtype),
            quantBatchMatmulInfo.outDtype);
    OP_LOGD(inputParams_.opName,
            "batchA = %lu batchA1-A4[%lu:%lu:%lu:%lu];"
            " batchB = %lu batchB1-B4[%lu:%lu:%lu:%lu]; batchC = %lu; batchBias = %lu",
            quantBatchMatmulInfo.batchA, quantBatchMatmulInfo.batchA1, quantBatchMatmulInfo.batchA2,
            quantBatchMatmulInfo.batchA3, quantBatchMatmulInfo.batchA4, quantBatchMatmulInfo.batchB,
            quantBatchMatmulInfo.batchB1, quantBatchMatmulInfo.batchB2, quantBatchMatmulInfo.batchB3,
            quantBatchMatmulInfo.batchB4, quantBatchMatmulInfo.batchC, quantBatchMatmulInfo.batchBias);
    OP_LOGD(inputParams_.opName, "isPerTensor = %d", static_cast<int32_t>(quantBatchMatmulInfo.isPerTensor));
}

QuantBmmReduceScatterHelper::QuantBmmReduceScatterHelper(QuantBmmReduceScatterTiling &quantBmmReduceScatterTiling,
                                                         DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &data)
    : Mc2AdaptiveSlidingWindowTiling(quantBmmReduceScatterTiling.GetContext(), &data),
      tilingProcesser_(quantBmmReduceScatterTiling),
      tilingArgs_(quantBmmReduceScatterTiling.GetArgs())
{}

CutResult QuantBmmReduceScatterTiling::GetTilingResult()
{
    if (mc2tiling::IsStandardCard4P(args_.rankDim, npuArch_)) {
        MMReduceScatterFitBalanceTiling quantBmmScatterTiling(args_, KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL,
                                                              TopoType::STANDARD_CARD);
        return quantBmmScatterTiling.GetTiling();
    } else if (mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        MMReduceScatterFitBalanceTiling quantBmmScatterTiling(args_, KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL,
                                                              TopoType::EIGHT_P, SocVersion::SOC950,
                                                              (commMode_ == TPL_AICPU_COMM_MODE));
        return quantBmmScatterTiling.GetTiling();
    } else {
        SocVersion inputSocVersion = (npuArch_ == NpuArch::DAV_3510) ? SocVersion::SOC950 : SocVersion::SOC910_B;
        MMPlusReduceScatter quantBmmScatterTiling(args_, args_.rankDim, KernelType::REDUCE_SCATTER, inputSocVersion,
                                                  false, (commMode_ == TPL_AICPU_COMM_MODE));
        quantBmmScatterTiling.GetTiling();
        return quantBmmScatterTiling.tilingM_.cutRes;
    }
}
// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulReduceScatterV2, QuantBmmReduceScatterTiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace optiling

#endif //_QUANT_BMM_MATMUL_REDUCE_SCATTER_TILING_CC_
