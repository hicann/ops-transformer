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
 * \file all_gather_matmul_v3_tiling.cpp
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "graph/types.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "ascendc/host_api/tiling/template_argument.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/all_gather_matmul_v3_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace Mc2Tiling {

// ---- input/output indices ----
constexpr uint32_t IDX_INPUT_CONTEXT = 0U;
constexpr uint32_t IDX_INPUT_X1 = 1U;
constexpr uint32_t IDX_INPUT_X2 = 2U;
constexpr uint32_t IDX_INPUT_BIAS = 3U;
constexpr uint32_t IDX_INPUT_X1_SCALE = 4U;
constexpr uint32_t IDX_INPUT_X2_SCALE = 5U;
constexpr uint32_t IDX_OUTPUT_Y = 0U;

// ---- attr indices (must match def order) ----
constexpr uint32_t ATTR_HCCL_BUFFER_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_IS_TRANS_B_INDEX = 3U;
constexpr uint32_t ATTR_RANK_SIZE_INDEX = 4U;
constexpr uint32_t ATTR_GROUP_SIZE_INDEX = 5U;
constexpr uint32_t ATTR_COMM_MODE_INDEX = 7U;

// ---- constants ----
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t DIM_FOUR = 4U;
constexpr uint64_t NZ_FRACTAL_ROW = 16UL;
constexpr uint64_t NZ_FRACTAL_COL = 32UL;
constexpr uint32_t SCALE_DIM_NUM = 3U;
constexpr uint64_t MX_SCALE_BLOCK = 64UL;
constexpr uint64_t SCALE_LAST_DIM = 2UL;
constexpr int64_t MAX_INT32_VAL = 2147483647;
constexpr int64_t K_MIN_VAL = 256;
constexpr int64_t K_MAX_VAL = 65535;

constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr uint64_t GROUP_N_OFFSET = 16UL;
constexpr uint64_t GROUP_M_OFFSET = 32UL;
constexpr uint64_t MX_GROUP_M = 1UL;
constexpr uint64_t MX_GROUP_N = 1UL;
constexpr uint64_t MX_GROUP_K = 32UL;

static const std::vector<ge::DataType> X_DTYPE_LIST = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1};
static const std::vector<ge::DataType> OUT_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16};

static bool IsContains(const std::vector<ge::DataType>& list, ge::DataType value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}
struct ShapeInfo {
    int64_t mPerRank{0};
    int64_t k{0};
    int64_t n{0};
    uint64_t rankSize{0};
};

/**
 * @brief 校验tensor指针非空
 */
static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext* context)
{
    auto contextDesc = context->GetInputDesc(IDX_INPUT_CONTEXT);
    auto x1Desc = context->GetInputDesc(IDX_INPUT_X1);
    auto x2Desc = context->GetInputDesc(IDX_INPUT_X2);
    auto yDesc = context->GetOutputDesc(IDX_OUTPUT_Y);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    // scale 在 def 中为 OPTIONAL，但 V3 仅支持 MX 量化，tiling 强制要求必传
    auto x1ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
    auto x2ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
    OP_TILING_CHECK(x1ScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x1_scale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2ScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x2_scale"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor数据类型
 */
static ge::graphStatus CheckTensorDataType(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto x1Desc = context->GetInputDesc(IDX_INPUT_X1);
    auto x2Desc = context->GetInputDesc(IDX_INPUT_X2);
    auto yDesc = context->GetOutputDesc(IDX_OUTPUT_Y);

    ge::DataType x1Dtype = x1Desc->GetDataType();
    OP_TILING_CHECK(!IsContains(X_DTYPE_LIST, x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "x1", Ops::Base::ToString(x1Dtype).c_str(),
                        "The dtype of x1 must be DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2 or DT_FLOAT4_E2M1"),
                    return ge::GRAPH_FAILED);

    ge::DataType x2Dtype = x2Desc->GetDataType();
    OP_TILING_CHECK(!IsContains(X_DTYPE_LIST, x2Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "x2", Ops::Base::ToString(x2Dtype).c_str(),
                        "The dtype of x2 must be DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2 or DT_FLOAT4_E2M1"),
                    return ge::GRAPH_FAILED);

    // FP4 要求 x1/x2 同时为 FP4
    OP_TILING_CHECK((x1Dtype == ge::DT_FLOAT4_E2M1) != (x2Dtype == ge::DT_FLOAT4_E2M1),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x1/x2", Ops::Base::ToString(x1Dtype).c_str(),
                                                          "FP4 requires both x1 and x2 to be DT_FLOAT4_E2M1"),
                    return ge::GRAPH_FAILED);
    // fp4 暂不支持 NZ 右矩阵
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(x2Format == ge::FORMAT_FRACTAL_NZ && x1Dtype == ge::DT_FLOAT4_E2M1,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x2", Ops::Base::ToString(x2Dtype).c_str(),
                                                          "FP4 does not support FRACTAL_NZ x2"),
                    return ge::GRAPH_FAILED);

    // scale dtype
    auto x1ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
    OP_TILING_CHECK(x1ScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x1_scale",
                                                          Ops::Base::ToString(x1ScaleDesc->GetDataType()).c_str(),
                                                          "The dtype of x1_scale must be DT_FLOAT8_E8M0"),
                    return ge::GRAPH_FAILED);
    auto x2ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
    OP_TILING_CHECK(x2ScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x2_scale",
                                                          Ops::Base::ToString(x2ScaleDesc->GetDataType()).c_str(),
                                                          "The dtype of x2_scale must be DT_FLOAT8_E8M0"),
                    return ge::GRAPH_FAILED);

    // output dtype
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(OUT_DTYPE_LIST, yDtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "y", Ops::Base::ToString(yDtype).c_str(),
                                                          "The dtype of y must be DT_BF16 or DT_FLOAT16"),
                    return ge::GRAPH_FAILED);

    // bias dtype (optional, 若非空，则必须为 DT_FLOAT)
    auto biasDesc = context->GetOptionalInputDesc(IDX_INPUT_BIAS);
    if (biasDesc != nullptr) {
        OP_TILING_CHECK(biasDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "bias",
                                                              Ops::Base::ToString(biasDesc->GetDataType()).c_str(),
                                                              "The dtype of bias must be DT_FLOAT"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor格式
 */
static ge::graphStatus CheckTensorFormat(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto contextDesc = context->GetInputDesc(IDX_INPUT_CONTEXT);
    ge::Format contextFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat()));
    OP_TILING_CHECK(contextFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "context", Ops::Base::ToString(contextFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto x1Desc = context->GetInputDesc(IDX_INPUT_X1);
    ge::Format x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x1", Ops::Base::ToString(x1Format).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto x2Desc = context->GetInputDesc(IDX_INPUT_X2);
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(x2Format != ge::FORMAT_ND && x2Format != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x2", Ops::Base::ToString(x2Format).c_str(), "ND/FRACTAL_NZ"),
                    return ge::GRAPH_FAILED);

    auto yDesc = context->GetOutputDesc(IDX_OUTPUT_Y);
    ge::Format yFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat()));
    OP_TILING_CHECK(yFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "y", Ops::Base::ToString(yFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto biasDesc = context->GetOptionalInputDesc(IDX_INPUT_BIAS);
    if (biasDesc != nullptr) {
        ge::Format biasFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat()));
        OP_TILING_CHECK(biasFormat != ge::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "bias", Ops::Base::ToString(biasFormat).c_str(), "ND"),
                        return ge::GRAPH_FAILED);
    }

    auto x1ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
    if (x1ScaleDesc != nullptr) {
        ge::Format x1ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x1ScaleDesc->GetStorageFormat()));
        OP_TILING_CHECK(
            x1ScaleFormat != ge::FORMAT_ND,
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x1_scale", Ops::Base::ToString(x1ScaleFormat).c_str(), "ND"),
            return ge::GRAPH_FAILED);
    }

    auto x2ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
    if (x2ScaleDesc != nullptr) {
        ge::Format x2ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2ScaleDesc->GetStorageFormat()));
        OP_TILING_CHECK(
            x2ScaleFormat != ge::FORMAT_ND,
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x2_scale", Ops::Base::ToString(x2ScaleFormat).c_str(), "ND"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验单个 MX scale shape: [dim0, Ceil(K/64), 2]
 */
static ge::graphStatus CheckOneScaleShape(const gert::StorageShape* scaleShape, const char* nodeName,
                                          const char* paramName, int64_t dim0, int64_t scaleKDim)
{
    OP_TILING_CHECK(scaleShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, paramName), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scaleShape->GetStorageShape().GetDimNum() != SCALE_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, paramName, (std::to_string(scaleShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of scale must be 3D"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scaleShape->GetStorageShape().GetDim(0) != dim0 ||
                        scaleShape->GetStorageShape().GetDim(1) != scaleKDim ||
                        scaleShape->GetStorageShape().GetDim(2) != static_cast<int64_t>(SCALE_LAST_DIM),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, paramName,
                        (std::string("[") + std::to_string(scaleShape->GetStorageShape().GetDim(0)) + "," +
                         std::to_string(scaleShape->GetStorageShape().GetDim(1)) + "," +
                         std::to_string(scaleShape->GetStorageShape().GetDim(2)) + "]")
                            .c_str(),
                        (std::string("expected [") + std::to_string(dim0) + "," + std::to_string(scaleKDim) + "," +
                         std::to_string(SCALE_LAST_DIM) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 读取 rank_size attr（必须由外界显式传入，不做 group 反查）
 */
static ge::graphStatus ResolveRankSize(const gert::TilingContext* context, int64_t& rankSize)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    auto rankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE_INDEX);
    OP_TILING_CHECK(rankSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "rank_size"), return ge::GRAPH_FAILED);
    rankSize = *rankSizePtr;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief x2 是否为 FRACTAL_NZ 格式
 */
static bool IsX2NzFormat(const gert::TilingContext* context)
{
    auto x2Desc = context->GetInputDesc(IDX_INPUT_X2);
    return x2Desc != nullptr &&
           static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ;
}

/**
 * @brief 读取 is_trans_b: true=[N,K](转置 B), false=[K,N](非转置 B)
 */
static bool GetIsTransB(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    auto ptr = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B_INDEX);
    return ptr != nullptr && *ptr;
}

static uint64_t CeilDivU64(uint64_t a, uint64_t b)
{
    return (a + b - 1UL) / b;
}

/**
 * @brief 校验 x1/x2 维度与取值，并提取基础 shape 信息（m/k/n/rankSize）
 *
 * x2 支持 4 种右矩阵布局（is_trans_b 决定语义）:
 *   ND + is_trans_b=true : [N, K] 转置存储
 *   ND + is_trans_b=false: [K, N] 连续存储
 *   NZ + is_trans_b=true : [N, K] 的 FRACTAL_NZ 分块 (ceil(K/32), ceil(N/16), 16, 32)
 *   NZ + is_trans_b=false: [K, N] 的 FRACTAL_NZ 分块 (ceil(N/32), ceil(K/16), 16, 32)
 */
static ge::graphStatus CheckInputShape(const gert::TilingContext* context, ShapeInfo& shapeInfo)
{
    const char* nodeName = context->GetNodeName();
    const bool isX2Nz = IsX2NzFormat(context);
    const bool isTransB = GetIsTransB(context);

    const gert::StorageShape* x1Shape = context->GetInputShape(IDX_INPUT_X1);
    OP_TILING_CHECK(x1Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x1Shape->GetStorageShape().GetDimNum() != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x1", (std::to_string(x1Shape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of x1 must be 2D"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* x2Shape = context->GetInputShape(IDX_INPUT_X2);
    OP_TILING_CHECK(x2Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x2"), return ge::GRAPH_FAILED);
    const uint64_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(!isX2Nz && x2DimNum != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "x2", (std::to_string(x2DimNum) + "D").c_str(),
                                                             "The shape dim of x2 must be 2D in ND format"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        isX2Nz && x2DimNum != DIM_TWO && x2DimNum != DIM_FOUR,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "x2", (std::to_string(x2DimNum) + "D").c_str(),
                                                 "The shape dim of x2 must be 2D (logical) or 4D (FRACTAL_NZ storage)"),
        return ge::GRAPH_FAILED);

    int64_t mPerRank = x1Shape->GetStorageShape().GetDim(0);
    int64_t kX1 = x1Shape->GetStorageShape().GetDim(1);
    int64_t x2N = 0;
    if (isX2Nz) {
        // NZ storage 4D 时最后两维固定为 (16, 32) 分形块
        if (x2DimNum == DIM_FOUR) {
            const int64_t nzD0 = x2Shape->GetStorageShape().GetDim(0);
            const int64_t nzD1 = x2Shape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(x2Shape->GetStorageShape().GetDim(2) != NZ_FRACTAL_ROW ||
                                x2Shape->GetStorageShape().GetDim(3) != NZ_FRACTAL_COL,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                nodeName, "x2",
                                (std::string("[") + std::to_string(nzD0) + "," + std::to_string(nzD1) + "," +
                                 std::to_string(x2Shape->GetStorageShape().GetDim(2)) + "," +
                                 std::to_string(x2Shape->GetStorageShape().GetDim(3)) + "]")
                                    .c_str(),
                                "FRACTAL_NZ storage must be [*, *, 16, 32]"),
                            return ge::GRAPH_FAILED);
            // 方向一致性: 转置 [N,K] -> dim0=ceil(K/32); 非转置 [K,N] -> dim1=ceil(K/16)
            if (isTransB) {
                OP_TILING_CHECK(static_cast<uint64_t>(nzD0) != CeilDivU64(static_cast<uint64_t>(kX1), NZ_FRACTAL_COL),
                                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                    nodeName, "x2.NZ.dim0", std::to_string(nzD0).c_str(),
                                    (std::string("ceil(K/32)=") +
                                     std::to_string(CeilDivU64(static_cast<uint64_t>(kX1), NZ_FRACTAL_COL)) +
                                     " for transposed [N,K] storage")
                                        .c_str()),
                                return ge::GRAPH_FAILED);
            } else {
                OP_TILING_CHECK(static_cast<uint64_t>(nzD1) != CeilDivU64(static_cast<uint64_t>(kX1), NZ_FRACTAL_ROW),
                                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                    nodeName, "x2.NZ.dim1", std::to_string(nzD1).c_str(),
                                    (std::string("ceil(K/16)=") +
                                     std::to_string(CeilDivU64(static_cast<uint64_t>(kX1), NZ_FRACTAL_ROW)) +
                                     " for non-transposed [K,N] storage")
                                        .c_str()),
                                return ge::GRAPH_FAILED);
            }
        } else {
            int64_t kX2 = isTransB ? x2Shape->GetStorageShape().GetDim(1) : x2Shape->GetStorageShape().GetDim(0);
            OP_TILING_CHECK(kX1 != kX2,
                            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                nodeName, "x1 and x2", (std::to_string(kX1) + " and " + std::to_string(kX2)).c_str(),
                                "The k-axis of x1 and x2 must be the same"),
                            return ge::GRAPH_FAILED);
        }
        const gert::StorageShape* yShape = context->GetOutputShape(IDX_OUTPUT_Y);
        OP_TILING_CHECK(yShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);
        x2N = yShape->GetStorageShape().GetDim(1);
    } else {
        x2N = isTransB ? x2Shape->GetStorageShape().GetDim(0) : x2Shape->GetStorageShape().GetDim(1);
        int64_t kX2 = isTransB ? x2Shape->GetStorageShape().GetDim(1) : x2Shape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(kX1 != kX2,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            nodeName, "x1 and x2", (std::to_string(kX1) + " and " + std::to_string(kX2)).c_str(),
                            "The k-axis of x1 and x2 must be the same"),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(mPerRank == 0 || kX1 == 0 || x2N == 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1/x2",
                                              (std::string("x1=[") + std::to_string(mPerRank) + "," +
                                               std::to_string(kX1) + "] x2.N=" + std::to_string(x2N))
                                                  .c_str(),
                                              "non-zero"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        kX1 < K_MIN_VAL || kX1 >= K_MAX_VAL,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "K", std::to_string(kX1).c_str(),
            (std::string("in range [") + std::to_string(K_MIN_VAL) + ", " + std::to_string(K_MAX_VAL) + ")").c_str()),
        return ge::GRAPH_FAILED);

    // fp4 按 4bit 打包，K 必须为偶数
    if (context->GetInputDesc(IDX_INPUT_X1)->GetDataType() == ge::DT_FLOAT4_E2M1) {
        OP_TILING_CHECK(
            kX1 % 2 != 0,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "K", std::to_string(kX1).c_str(), "must be even in mxfp4 scene"),
            return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        mPerRank > MAX_INT32_VAL || x2N > MAX_INT32_VAL,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1.M/x2.N", (std::to_string(mPerRank) + "/" + std::to_string(x2N)).c_str(),
                                  "must not exceed INT32_MAX"),
        return ge::GRAPH_FAILED);

    // rank_size: 直接读取外界传入的 attr
    int64_t rankSize = 0;
    OP_TILING_CHECK(ResolveRankSize(context, rankSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "resolve rank_size failed"), return ge::GRAPH_FAILED);

    shapeInfo.mPerRank = mPerRank;
    shapeInfo.k = kX1;
    shapeInfo.n = x2N;
    shapeInfo.rankSize = static_cast<uint64_t>(rankSize);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 y shape: [M * rankSize, N]
 */
static ge::graphStatus CheckOutputShape(const gert::TilingContext* context, const ShapeInfo& shapeInfo)
{
    const char* nodeName = context->GetNodeName();

    const gert::StorageShape* yShape = context->GetOutputShape(IDX_OUTPUT_Y);
    OP_TILING_CHECK(yShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yShape->GetStorageShape().GetDimNum() != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "y", (std::to_string(yShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of y must be 2D"),
                    return ge::GRAPH_FAILED);

    int64_t yM = yShape->GetStorageShape().GetDim(0);
    int64_t yN = yShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        yM != shapeInfo.mPerRank * static_cast<int64_t>(shapeInfo.rankSize) || yN != shapeInfo.n,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yM) + "," + std::to_string(yN) + "]").c_str(),
            (std::string("expected [") + std::to_string(shapeInfo.mPerRank * static_cast<int64_t>(shapeInfo.rankSize)) +
             "," + std::to_string(shapeInfo.n) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 bias shape: [N]，非空时校验
 */
static ge::graphStatus CheckBiasShape(const gert::TilingContext* context, const ShapeInfo& shapeInfo)
{
    const char* nodeName = context->GetNodeName();

    auto biasShape = context->GetOptionalInputShape(IDX_INPUT_BIAS);
    if (biasShape != nullptr) {
        OP_TILING_CHECK(biasShape->GetStorageShape().GetDimNum() != 1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "bias", (std::to_string(biasShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                            "The shape dim of bias must be 1D"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            biasShape->GetStorageShape().GetDim(0) != shapeInfo.n,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "bias", std::to_string(biasShape->GetStorageShape().GetDim(0)).c_str(),
                                      (std::string("equal to N=") + std::to_string(shapeInfo.n)).c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 scale shapes: x1_scale [M, K/64, 2]; x2_scale [N, K/64, 2]
 */
static ge::graphStatus CheckScaleShapes(const gert::TilingContext* context, const ShapeInfo& shapeInfo)
{
    const char* nodeName = context->GetNodeName();

    int64_t scaleKDim = (shapeInfo.k + static_cast<int64_t>(MX_SCALE_BLOCK) - 1) / static_cast<int64_t>(MX_SCALE_BLOCK);
    OP_TILING_CHECK(CheckOneScaleShape(context->GetOptionalInputShape(IDX_INPUT_X1_SCALE), nodeName, "x1_scale",
                                       shapeInfo.mPerRank, scaleKDim) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check x1_scale shape failed"), return ge::GRAPH_FAILED);
    // x2_scale: 转置 B 为 [N, K/64, 2]; 非转置 B 为 [K/64, N, 2]
    if (GetIsTransB(context)) {
        OP_TILING_CHECK(CheckOneScaleShape(context->GetOptionalInputShape(IDX_INPUT_X2_SCALE), nodeName, "x2_scale",
                                           shapeInfo.n, scaleKDim) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "check x2_scale shape failed"), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(CheckOneScaleShape(context->GetOptionalInputShape(IDX_INPUT_X2_SCALE), nodeName, "x2_scale",
                                           scaleKDim, shapeInfo.n) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "check x2_scale shape failed"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor维度和shape
 */
static ge::graphStatus CheckTensorShape(const gert::TilingContext* context, ShapeInfo& shapeInfo)
{
    const char* nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckInputShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check input shape failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check output shape failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckBiasShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check bias shape failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckScaleShapes(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check scale shape failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor (ptr + dtype + format + shape)
 */
static ge::graphStatus CheckTensor(const gert::TilingContext* context, ShapeInfo& shapeInfo)
{
    const char* nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckTensorPtrNullptr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check tensor nullptr failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDataType(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check tensor dtype failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorFormat(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check tensor format failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check tensor shape failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 rank_size: 合法取值且不超过 1:1 核配比下实际可用 block 数
 */
static ge::graphStatus CheckRankSizeAttr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    // rank_size: 仅校验外界传入的 attr，不做 group 反查
    int64_t rankSize = 0;
    OP_TILING_CHECK(ResolveRankSize(context, rankSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "resolve rank_size failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSize != 2 && rankSize != 4 && rankSize != 8 && rankSize != 16,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "rank_size", std::to_string(rankSize).c_str(), "2/4/8/16"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t actualBlockNum = std::min(aicNum, aivNum);
    OP_TILING_CHECK(
        static_cast<uint32_t>(rankSize) > actualBlockNum,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "rank_size",
                                  (std::to_string(rankSize) + " > min(aicNum=" + std::to_string(aicNum) +
                                   ", aivNum=" + std::to_string(aivNum) + ")=" + std::to_string(actualBlockNum))
                                      .c_str(),
                                  "<= min(aicNum, aivNum)"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 is_trans_b: V3 仅支持 x2 [N,K] 布局（is_trans_b=true）
 */
static ge::graphStatus CheckIsTransBAttr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    // true=[N,K] 转置 B; false=[K,N] 非转置 B, 两种均支持
    auto isTransBPtr = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B_INDEX);
    OP_TILING_CHECK(isTransBPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "is_trans_b"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 */
/**
 * @brief 校验 comm_mode: experimental 版仅支持 ubmem 通路
 */
static ge::graphStatus CheckCommModeAttr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto commModePtr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE_INDEX);
    OP_TILING_CHECK(commModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "comm_mode"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::strcmp(commModePtr, "ubmem") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "comm_mode", commModePtr,
                                              "ubmem (experimental version supports the ubmem path only)"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 group_size (MX quant: [1,1,32])，V3 仅支持 MX 量化，
 *        值为 0 的维度自动从 scale shape 推导，推导后校验是否为 [1,1,32]
 */
static ge::graphStatus CheckGroupSizeAttr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group_size"),
                    return ge::GRAPH_FAILED);
    uint64_t gs = static_cast<uint64_t>(*groupSizePtr);
    uint64_t gsK = gs & GROUP_MNK_BIT_SIZE;
    uint64_t gsN = (gs >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t gsM = (gs >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;

    // 自动推导：值为 0 的维度从 scale shape 反推
    const gert::StorageShape* x1Shape = context->GetInputShape(IDX_INPUT_X1);
    const gert::StorageShape* x2Shape = context->GetInputShape(IDX_INPUT_X2);
    auto x1ScaleShape = context->GetOptionalInputShape(IDX_INPUT_X1_SCALE);
    auto x2ScaleShape = context->GetOptionalInputShape(IDX_INPUT_X2_SCALE);
    int64_t mValue = x1Shape->GetStorageShape().GetDim(0);
    int64_t kValue = x1Shape->GetStorageShape().GetDim(1);
    int64_t nValue = x2Shape->GetStorageShape().GetDim(0);
    if (IsX2NzFormat(context)) {
        // NZ storage 的 dim0 是 K 方向 block 数, N 取输出 y 的 dim1
        const gert::StorageShape* yShape = context->GetOutputShape(IDX_OUTPUT_Y);
        nValue = (yShape != nullptr) ? yShape->GetStorageShape().GetDim(1) : nValue;
    } else if (!GetIsTransB(context)) {
        // 非转置 [K, N]
        nValue = x2Shape->GetStorageShape().GetDim(1);
    }

    if (gsM == 0) {
        int64_t scaleM = x1ScaleShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(scaleM == 0, OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1_scale M", "0", "non-zero"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            mValue % scaleM != 0,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1 M / x1_scale M",
                                      (std::to_string(mValue) + " / " + std::to_string(scaleM)).c_str(), "divisible"),
            return ge::GRAPH_FAILED);
        gsM = static_cast<uint64_t>(mValue / scaleM);
    }
    if (gsN == 0) {
        int64_t scaleN = x2ScaleShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(scaleN == 0, OP_LOGE_FOR_INVALID_VALUE(nodeName, "x2_scale N", "0", "non-zero"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            nValue % scaleN != 0,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "x2 N / x2_scale N",
                                      (std::to_string(nValue) + " / " + std::to_string(scaleN)).c_str(), "divisible"),
            return ge::GRAPH_FAILED);
        gsN = static_cast<uint64_t>(nValue / scaleN);
    }
    if (gsK == 0) {
        // mxfp scale shape [M, CeilDiv(K,64), 2]，K维 group size = K / (CeilDiv(K,64) * 2)
        int64_t scaleKDim = x1ScaleShape->GetStorageShape().GetDim(1);
        int64_t scaleKValue = scaleKDim * 2;
        OP_TILING_CHECK(scaleKValue == 0, OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1_scale K dim", "0", "non-zero"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(kValue % scaleKValue != 0,
                        OP_LOGE_FOR_INVALID_VALUE(
                            nodeName, "x1 K / x1_scale K",
                            (std::to_string(kValue) + " / " + std::to_string(scaleKValue)).c_str(), "divisible"),
                        return ge::GRAPH_FAILED);
        gsK = static_cast<uint64_t>(kValue / scaleKValue);
    }

    OP_LOGI(nodeName, "group_size: attr=[%lu,%lu,%lu], inferred [M=%lu N=%lu K=%lu]",
            (gs >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE, (gs >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE,
            gs & GROUP_MNK_BIT_SIZE, gsM, gsN, gsK);
    OP_TILING_CHECK(gsM != MX_GROUP_M || gsN != MX_GROUP_N || gsK != MX_GROUP_K,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "group_size",
                                              (std::string("[") + std::to_string(gsM) + "," + std::to_string(gsN) +
                                               "," + std::to_string(gsK) + "]")
                                                  .c_str(),
                                              "[1,1,32]"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckHcclBufferSizeAttr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto hcclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HCCL_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(hcclBufferSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "hccl_buffer_size"),
                    return ge::GRAPH_FAILED);
    int64_t hcclBufferSize = *hcclBufferSizePtr;
    OP_TILING_CHECK(hcclBufferSize <= 0,
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "hccl_buffer_size", std::to_string(hcclBufferSize).c_str(),
                                              "positive integer"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "hccl_buffer_size=%ld (ubmem path, window size managed by CommContextManager)", hcclBufferSize);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验算子属性
 */
static ge::graphStatus CheckAttrs(const gert::TilingContext* context, const ShapeInfo& shapeInfo)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckRankSizeAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check rank_size attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckIsTransBAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check is_trans_b attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckCommModeAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check comm_mode attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckGroupSizeAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check group_size attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckHcclBufferSizeAttr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check hccl_buffer_size attr failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

struct UbmemWsLayout {
    uint64_t libApiSize{0};
    uint64_t totalSize{0};
};

/**
 * @brief workspace 布局：AscendC 基础库预留区
 */
static UbmemWsLayout CalcUbmemWorkspaceLayout(const gert::TilingContext* context)
{
    UbmemWsLayout layout;
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    layout.libApiSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    layout.totalSize = layout.libApiSize;
    return layout;
}

/**
 * @brief 设置tiling数据（shape 切分、SWAT 引擎、workspace 布局）
 */
static ge::graphStatus SetTilingData(gert::TilingContext* context, const ShapeInfo& shapeInfo)
{
    const char* nodeName = context->GetNodeName();

    auto* rawTilingData = context->GetRawTilingData();
    OP_TILING_CHECK(rawTilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "rawTilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rawTilingData->GetCapacity() < sizeof(QuantMatmulTilingData),
                    OP_LOGE(nodeName, "rawTilingData capacity %zu < %zu", rawTilingData->GetCapacity(),
                            sizeof(QuantMatmulTilingData)),
                    return ge::GRAPH_FAILED);

    memset_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), 0, rawTilingData->GetCapacity());
    auto* tilingData = reinterpret_cast<QuantMatmulTilingData*>(rawTilingData->GetData());

    uint64_t m = static_cast<uint64_t>(shapeInfo.mPerRank);
    uint64_t k = static_cast<uint64_t>(shapeInfo.k);
    uint64_t n = static_cast<uint64_t>(shapeInfo.n);
    uint64_t rankSize = shapeInfo.rankSize;
    uint64_t totalM = m * rankSize;

    // fp4 用 e2m1 引擎, fp8 系(e4m3/e5m2)统一用 e4m3 引擎
    const bool isTransB = GetIsTransB(context);
    if (context->GetInputDesc(IDX_INPUT_X1)->GetDataType() == ge::DT_FLOAT4_E2M1) {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1, mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.GetTilingData(totalM, n, k, false, isTransB, *tilingData);
    } else {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                              mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.GetTilingData(totalM, n, k, false, isTransB, *tilingData);
    }

    uint32_t usedCoreNum = tilingData->usedCoreNum < static_cast<uint32_t>(rankSize) ? static_cast<uint32_t>(rankSize) :
                                                                                       tilingData->usedCoreNum;
    tilingData->usedCoreNum = usedCoreNum;

    UbmemWsLayout layout = CalcUbmemWorkspaceLayout(context);

    context->SetScheduleMode(1);
    context->SetBlockDim(usedCoreNum);
    rawTilingData->SetDataSize(sizeof(QuantMatmulTilingData));

    OP_LOGI(nodeName,
            "AllGatherMatmulV3 ubmem tiling: mPerRank=%lu totalM=%lu k=%lu n=%lu rankSize=%lu usedCoreNum=%u "
            "libApiWsSize=%lu wsTotalSize=%lu",
            m, totalM, k, n, rankSize, usedCoreNum, layout.libApiSize, layout.totalSize);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling key
 */
static void SetTilingKey(gert::TilingContext* context)
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(GetIsTransB(context));
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "tilingKey is [%lu]", tilingKey);
}

/**
 * @brief 设置workspace大小
 */
static ge::graphStatus SetWorkSpace(gert::TilingContext* context)
{
    uint64_t workspaceSize = CalcUbmemWorkspaceLayout(context).totalSize;
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(context->GetNodeName(), "workSpaces is nullptr"),
                    return ge::GRAPH_FAILED);
    workSpaces[0] = workspaceSize;
    OP_LOGD(context->GetNodeName(), "workspace size is %lu", workspaceSize);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief AllGatherMatmulV3 算子的 tiling 函数
 */
static ge::graphStatus AllGatherMatmulV3TilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("AllGatherMatmulV3", "failed to get tiling context"),
                    return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "nodeName"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Enter AllGatherMatmulV3 tiling func");

    // 1. tensor check (ptr + dtype + format + shape)
    ShapeInfo shapeInfo;
    OP_TILING_CHECK(CheckTensor(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check input/output failed"), return ge::GRAPH_FAILED);

    // ubmem暂不支持 bias，未实现 bias 通路）
    OP_TILING_CHECK(context->GetOptionalInputShape(IDX_INPUT_BIAS) != nullptr,
                    OP_LOGE(nodeName, "bias is not supported in the ubmem path"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "AllGatherMatmulV3 comm path: ubmem");

    // 2. attr check
    OP_TILING_CHECK(CheckAttrs(context, shapeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check attrs failed"),
                    return ge::GRAPH_FAILED);

    // 3. set tiling data
    OP_TILING_CHECK(SetTilingData(context, shapeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set tiling data failed"),
                    return ge::GRAPH_FAILED);

    // 4. set tiling key
    SetTilingKey(context);

    // 5. set workspace
    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set workspace failed"),
                    return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "AllGatherMatmulV3 tiling end");
    return ge::GRAPH_SUCCESS;
}

struct AllGatherMatmulV3CompileInfo {};

static ge::graphStatus TilingParseForAllGatherMatmulV3(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AllGatherMatmulV3)
    .Tiling(AllGatherMatmulV3TilingFunc)
    .TilingParse<AllGatherMatmulV3CompileInfo>(TilingParseForAllGatherMatmulV3);

} // namespace Mc2Tiling
