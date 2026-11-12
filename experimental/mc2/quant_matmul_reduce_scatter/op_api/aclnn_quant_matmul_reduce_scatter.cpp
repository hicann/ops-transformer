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
 * \file aclnn_quant_matmul_reduce_scatter.cpp
 * \brief aclnn API implementation for QuantMatmulReduceScatter (MX-quant FP8, apace ubmem path)
 */

#include "aclnnInner_quant_matmul_reduce_scatter.h"

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "mc2_log_compat.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"

using namespace op;

namespace {
enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

constexpr int64_t TWO_DIMS = 2;
constexpr int64_t THREE_DIMS = 3;
constexpr int64_t FOUR_DIMS = 4;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t NZ_FRACTAL_ROW = 16;
constexpr int64_t NZ_FRACTAL_COL = 32;
constexpr int64_t MX_SCALE_GROUP_SIZE = 64; // MX 分组宽度（kd 粒度，与 device 侧 MXFP_DIVISOR 一致）

// group_size 位打包布局（groupM<<32 | groupN<<16 | groupK），各字段 16 位
constexpr int64_t GROUP_SIZE_OFFSET_M_BITS = 32;
constexpr int64_t GROUP_SIZE_OFFSET_N_BITS = 16;
constexpr int64_t GROUP_SIZE_FIELD_MASK = 0xFFFF;
constexpr int64_t GROUP_SIZE_AUTO = 0; // 0：自动推断 [1,1,32]
constexpr int64_t GROUP_SIZE_DEFAULT_M = 1;
constexpr int64_t GROUP_SIZE_DEFAULT_N = 1;
constexpr int64_t GROUP_SIZE_DEFAULT_K = 32;

// x2 为最后两维转置 view（view [K,N]、物理 storage [N,K]）时返回 true。
// 判据基于 view 与 storage 最后两维互换（物理 [N,K] 的 [K,N] 视图），不依赖 stride——
// allow_internal_format 下连续 tensor 的 stride 元数据可能失真，不能用 stride 判断。
static bool IsTransposedView2D(const aclTensor* x2)
{
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        return false;
    }
    const auto& view = x2->GetViewShape();
    const auto& strides = x2->GetViewStrides();
    if (static_cast<int64_t>(strides.size()) < TWO_DIMS) {
        return false;
    }
    // 列优先/转置连续（如 .t() 视图）：strides[DIM_ZERO]==1 且 strides[DIM_ONE]==行数(view.dim0)。
    // 用 stride 判据而非 storage/view shape 互换：torch extension 传入的 storage shape 为
    // 1D 展平（nbytes/itemsize），原互换判据在 storage 非 2D 时恒 false，会漏判 ND+trans 的
    // .t() 视图，导致 view 位置把 [K,N] 视图误判为非转置而与 x2_scale 方向冲突。
    if (strides[DIM_ZERO] == 1 && strides[DIM_ONE] == view.GetDim(DIM_ZERO)) {
        return !(view.GetDim(DIM_ZERO) == 1 && view.GetDim(DIM_ONE) == 1);
    }
    return false;
}

// 将 [K,N] 转置 view 翻回 [N,K] 视图（物理存储不变），使框架/tiling 以 [N,K]（DN）语义接收 x2
static const aclTensor* TransX2Tensor(const aclTensor* x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (uint64_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }

    uint64_t viewDimsNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDims(viewDimsNum);
    for (uint64_t i = 0; i < viewDimsNum; i++) {
        viewDims[i] = x2->GetViewShape().GetDim(i);
    }
    viewDims[DIM_ZERO] = x2->GetViewShape().GetDim(DIM_ONE);
    viewDims[DIM_ONE] = x2->GetViewShape().GetDim(DIM_ZERO);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(viewDimsNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    stride[DIM_ZERO] = transStride[DIM_ONE];
    stride[DIM_ONE] = transStride[DIM_ZERO];
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDims.data(), viewDimsNum, dataType, stride.data(), offset, format, storageDims.data(),
                           storageDimsNum, x2->GetTensor()->GetAddr());
}

// x2 为 FRACTAL_NZ 时重建一个保持原始物理存储的 aclTensor，供 inner 侧透传
static const aclTensor* CopyTensor(const aclTensor* x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (size_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }
    OP_LOGD("QuantMatmulReduceScatter, CopyTensor storageDimsNum is %lu.", storageDimsNum);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    auto stride = x2->GetViewStrides();
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    auto stgFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    if (stgFormat == Format::FORMAT_ND) {
        OP_LOGD("QuantMatmulReduceScatter, CopyTensor format is ACL_FORMAT_ND");
        format = aclFormat::ACL_FORMAT_ND;
    } else if (stgFormat == Format::FORMAT_FRACTAL_NZ) {
        format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    return aclCreateTensor(storageDims.data(), storageDimsNum, dataType, stride.data(), offset, format,
                           storageDims.data(), storageDimsNum, x2->GetTensor()->GetAddr());
}
} // namespace

static constexpr const char* K_QMMRS_OP_NAME = "aclnnQuantMatmulReduceScatter";

// dtype/format 校验（对齐 tiling CheckDtypeAndFormat 口径）
static aclnnStatus CheckDtypeAndFormat(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                                       const aclTensor* x2Scale, const aclTensor* y)
{
    auto isSupportedX = [](op::DataType dt) {
        return dt == op::DataType::DT_FLOAT8_E4M3FN || dt == op::DataType::DT_FLOAT8_E5M2 ||
               dt == op::DataType::DT_FLOAT4_E2M1;
    };
    if (!isSupportedX(x1->GetDataType())) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(K_QMMRS_OP_NAME, "x1",
                                              std::to_string(static_cast<int>(x1->GetDataType())),
                                              "expect FP8_E4M3FN/FP8_E5M2/FP4_E2M1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!isSupportedX(x2->GetDataType())) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(K_QMMRS_OP_NAME, "x2",
                                              std::to_string(static_cast<int>(x2->GetDataType())),
                                              "expect FP8_E4M3FN/FP8_E5M2/FP4_E2M1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // fp4 双向拦截：x1/x2 必须同为 fp4 或同为 fp8
    if ((x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1) != (x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(K_QMMRS_OP_NAME, "x1, x2",
                                               (std::to_string(static_cast<int>(x1->GetDataType())) + ", " +
                                                std::to_string(static_cast<int>(x2->GetDataType())))
                                                   .c_str(),
                                               "FP4 requires both x1 and x2 to be FP4_E2M1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0 ||
        x2Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(K_QMMRS_OP_NAME, "x1_scale, x2_scale",
                                               (std::to_string(static_cast<int>(x1Scale->GetDataType())) + ", " +
                                                std::to_string(static_cast<int>(x2Scale->GetDataType())))
                                                   .c_str(),
                                               "scale dtype must be FP8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (y->GetDataType() != op::DataType::DT_BF16 && y->GetDataType() != op::DataType::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(K_QMMRS_OP_NAME, "y", std::to_string(static_cast<int>(y->GetDataType())),
                                              "expect BF16 or FP16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // format：x1 恒 ND；x2 支持 ND/FRACTAL_NZ；fp4 仅 ND
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT(K_QMMRS_OP_NAME, "x1", std::to_string(static_cast<int>(x1->GetStorageFormat())),
                                   "ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool x2Nz = (x2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ);
    if (!x2Nz && x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            K_QMMRS_OP_NAME, "x2", std::to_string(static_cast<int>(x2->GetStorageFormat())), "ND or FRACTAL_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1 && x2Nz) {
        OP_LOGE_FOR_INVALID_FORMAT(K_QMMRS_OP_NAME, "x2", std::to_string(static_cast<int>(x2->GetStorageFormat())),
                                   "ND (FP4)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAttrs(const char* group, int64_t rankSize, int64_t hcclBufferSize, int64_t groupSize,
                              const char* commMode)
{
    if (group == nullptr) {
        OP_LOGE_WITH_INVALID_ATTR(K_QMMRS_OP_NAME, "group", "nullptr", "non-empty string");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (rankSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE(K_QMMRS_OP_NAME, "rank_size", std::to_string(rankSize), "positive integer");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hcclBufferSize <= 0) {
        OP_LOGE_WITH_INVALID_ATTR(K_QMMRS_OP_NAME, "hccl_buffer_size", std::to_string(hcclBufferSize).c_str(),
                                  "positive integer");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // group_size：0（自动推断）或位打包 [1,1,32]（groupM<<32|groupN<<16|groupK）
    if (groupSize != GROUP_SIZE_AUTO) {
        const int64_t gsM = (groupSize >> GROUP_SIZE_OFFSET_M_BITS) & GROUP_SIZE_FIELD_MASK;
        const int64_t gsN = (groupSize >> GROUP_SIZE_OFFSET_N_BITS) & GROUP_SIZE_FIELD_MASK;
        const int64_t gsK = groupSize & GROUP_SIZE_FIELD_MASK;
        if (gsM != GROUP_SIZE_DEFAULT_M || gsN != GROUP_SIZE_DEFAULT_N || gsK != GROUP_SIZE_DEFAULT_K) {
            OP_LOGE_FOR_INVALID_VALUE(
                K_QMMRS_OP_NAME, "group_size",
                (std::to_string(gsM) + "," + std::to_string(gsN) + "," + std::to_string(gsK)).c_str(), "[1,1,32]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (commMode == nullptr || std::string(commMode) != "ubmem") {
        OP_LOGE_FOR_INVALID_VALUE(K_QMMRS_OP_NAME, "comm_mode", commMode != nullptr ? commMode : "nullptr", "ubmem");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                              const aclTensor* x2Scale, const aclTensor* y, bool isTransB)
{
    auto shapeStr = [](const aclTensor* t) {
        std::string s = "[";
        const int64_t dimNum = static_cast<int64_t>(t->GetViewShape().GetDimNum());
        for (int64_t i = 0; i < dimNum; ++i) {
            if (i != 0) {
                s += ",";
            }
            s += std::to_string(t->GetViewShape().GetDim(i));
        }
        s += "]";
        return s;
    };
    if (x1->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_QMMRS_OP_NAME, "x1",
                                                 (std::to_string(x1->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x1 must be 2D");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (y->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_QMMRS_OP_NAME, "y",
                                                 (std::to_string(y->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of y must be 2D");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool isX2Nz =
        (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ);
    const int64_t mVal = x1->GetViewShape().GetDim(DIM_ZERO);
    const int64_t kX1 = x1->GetViewShape().GetDim(DIM_ONE);
    int64_t nVal = y->GetViewShape().GetDim(DIM_ONE);
    const char* layoutDesc = isTransB ? "x2 must be [N, K] layout" : "x2 must be [K, N] layout";
    if (isX2Nz) {
        const int64_t x2DimNum = x2->GetViewShape().GetDimNum();
        if (x2DimNum == TWO_DIMS) {
            nVal = isTransB ? x2->GetViewShape().GetDim(DIM_ZERO) : x2->GetViewShape().GetDim(DIM_ONE);
            const int64_t kX2 = isTransB ? x2->GetViewShape().GetDim(DIM_ONE) : x2->GetViewShape().GetDim(DIM_ZERO);
            if (kX1 != kX2) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(K_QMMRS_OP_NAME, "x1, x2",
                                                       (shapeStr(x1) + ", " + shapeStr(x2)).c_str(), layoutDesc);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else if (x2DimNum == FOUR_DIMS) {
            if (x2->GetViewShape().GetDim(DIM_TWO) != NZ_FRACTAL_ROW ||
                x2->GetViewShape().GetDim(DIM_THREE) != NZ_FRACTAL_COL) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(K_QMMRS_OP_NAME, "x2", shapeStr(x2).c_str(),
                                                      "FRACTAL_NZ storage must be [*, *, 16, 32]");
                return ACLNN_ERR_PARAM_INVALID;
            }
            const int64_t nzD0 = x2->GetViewShape().GetDim(DIM_ZERO);
            const int64_t nzD1 = x2->GetViewShape().GetDim(DIM_ONE);
            const int64_t kt32 = (kX1 + NZ_FRACTAL_COL - 1) / NZ_FRACTAL_COL;
            const int64_t kt16 = (kX1 + NZ_FRACTAL_ROW - 1) / NZ_FRACTAL_ROW;
            if ((isTransB && nzD0 != kt32) || (!isTransB && nzD1 != kt16)) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    K_QMMRS_OP_NAME, "x1, x2", (shapeStr(x2) + ", K=" + std::to_string(kX1)).c_str(), layoutDesc);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_QMMRS_OP_NAME, "x2", (std::to_string(x2DimNum) + "D").c_str(),
                                                     "2D (logical) or 4D (FRACTAL_NZ storage)");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_QMMRS_OP_NAME, "x2",
                                                     (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                     "The shape of x2 must be 2D in ND format");
            return ACLNN_ERR_PARAM_INVALID;
        }
        nVal = isTransB ? x2->GetViewShape().GetDim(DIM_ZERO) : x2->GetViewShape().GetDim(DIM_ONE);
        const int64_t kX2 = isTransB ? x2->GetViewShape().GetDim(DIM_ONE) : x2->GetViewShape().GetDim(DIM_ZERO);
        if (kX1 != kX2) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(K_QMMRS_OP_NAME, "x1, x2",
                                                   (shapeStr(x1) + ", " + shapeStr(x2)).c_str(), layoutDesc);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (y->GetViewShape().GetDim(DIM_ONE) != nVal) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(K_QMMRS_OP_NAME, "y", shapeStr(y).c_str(), "y.N must be equal to x2.N");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // scale shape：kd 从 x1_scale 形状取（逻辑 K 粒度 ceil(K/64)，fp4 打包无关）
    if (x1Scale->GetViewShape().GetDimNum() != THREE_DIMS || x2Scale->GetViewShape().GetDimNum() != THREE_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(K_QMMRS_OP_NAME, "x1_scale, x2_scale",
                                               (shapeStr(x1Scale) + ", " + shapeStr(x2Scale)).c_str(),
                                               "scale must be 3D [dim0, kd, 2]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t kd = x1Scale->GetViewShape().GetDim(DIM_ONE);
    if (x1Scale->GetViewShape().GetDim(DIM_ZERO) != mVal || x1Scale->GetViewShape().GetDim(DIM_TWO) != SCALE_LAST_DIM) {
        OP_LOGE_FOR_INVALID_SHAPE(K_QMMRS_OP_NAME, "x1_scale", shapeStr(x1Scale).c_str(),
                                  ("[" + std::to_string(mVal) + "," + std::to_string(kd) + ",2]").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t x2ScaleDim0 = isTransB ? nVal : kd;
    const int64_t x2ScaleDim1 = isTransB ? kd : nVal;
    if (x2Scale->GetViewShape().GetDim(DIM_ZERO) != x2ScaleDim0 ||
        x2Scale->GetViewShape().GetDim(DIM_ONE) != x2ScaleDim1 ||
        x2Scale->GetViewShape().GetDim(DIM_TWO) != SCALE_LAST_DIM) {
        OP_LOGE_FOR_INVALID_SHAPE(
            K_QMMRS_OP_NAME, "x2_scale", shapeStr(x2Scale).c_str(),
            ("[" + std::to_string(x2ScaleDim0) + "," + std::to_string(x2ScaleDim1) + ",2]").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnQuantMatmulReduceScatterGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
    const aclTensor* x2Scale, const char* group, int64_t rankSize, int64_t hcclBufferSize, int64_t groupSize,
    const char* commMode, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnnQuantMatmulReduceScatterGetWorkspaceSize start");
    if (context == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_QMMRS_OP_NAME, "context");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x1 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_QMMRS_OP_NAME, "x1");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x2 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_QMMRS_OP_NAME, "x2");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x1Scale == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_QMMRS_OP_NAME, "x1_scale");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x2Scale == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_QMMRS_OP_NAME, "x2_scale");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (y == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_QMMRS_OP_NAME, "y");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_QMMRS_OP_NAME, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_QMMRS_OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    aclnnStatus checkRet = CheckDtypeAndFormat(x1, x2, x1Scale, x2Scale, y);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = CheckAttrs(group, rankSize, hcclBufferSize, groupSize, commMode);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    OP_LOGD("aclnnQuantMatmulReduceScatterGetWorkspaceSize: x2Format=%d",
            static_cast<int>(ge::GetPrimaryFormat(x2->GetStorageFormat())));
    const bool isX2Nz =
        (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ);

    if (x1->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_QMMRS_OP_NAME, "x1",
                                                 (std::to_string(x1->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "x1 must be 2D before transB inference");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1Scale->GetViewShape().GetDimNum() != THREE_DIMS || x2Scale->GetViewShape().GetDimNum() != THREE_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_QMMRS_OP_NAME, "x1_scale, x2_scale",
                                                 (std::to_string(x1Scale->GetViewShape().GetDimNum()) + "D, " +
                                                  std::to_string(x2Scale->GetViewShape().GetDimNum()) + "D")
                                                     .c_str(),
                                                 "x1_scale/x2_scale must be 3D before transB inference");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // fp4 打包存储（2 元素/字节）时 x1 末维为 K/2，需翻倍还原逻辑 K。
    // 逻辑 K 从 x1_scale 形状推导：x1_scale=[m, kd, 2]，kd=ceil(K/64)（MX 量化粒度）。
    // 不依赖 x1 aclTensor dtype——fp4 打包时 x1 为 uint8（dim1=K/2），DT_FLOAT4_E2M1 判断不触发。
    const int64_t kd = x1Scale->GetViewShape().GetDim(DIM_ONE);
    const int64_t kX1 = kd * MX_SCALE_GROUP_SIZE;
    // x1 view 口径 K（fp4 打包时为 K/2），与 x2 view dim 同口径，用于 2D view 位置判断。
    // 不能使用 kd*64（fp4 打包时与 view dim 不一致，会误判）。
    const int64_t kX1View = x1->GetViewShape().GetDim(DIM_ONE);
    // x2_scale 为显式构造的普通 uint8 tensor（不随 allow_internal_format 失真），
    // 其形状给出方向的权威信号：非转置 B -> scale=[kd,n,2]（sD0==kd）；转置 B -> scale=[n,kd,2]（sD1==kd）。
    const int64_t sD0 = x2Scale->GetViewShape().GetDim(DIM_ZERO);
    const int64_t sD1 = x2Scale->GetViewShape().GetDim(DIM_ONE);
    const bool scaleNonTrans = (sD0 == kd && sD1 != kd);
    const bool scaleTrans = (sD1 == kd && sD0 != kd);
    // scale 形状无法给出方向（kd==n 方形 或 两者都不匹配 kd）
    const bool scaleAmbiguous = !scaleNonTrans && !scaleTrans;

    // ---- 方向推断（inner 的 transpose_x2 attr）：true=[N,K] 转置 B / false=[K,N] 非转置 B ----
    bool isTransB = true;
    const int64_t x2DimNum = x2->GetViewShape().GetDimNum();
    if (isX2Nz && x2DimNum == FOUR_DIMS) {
        // NZ 4D storage：转置 [N,K] -> (ceil(K/32), ceil(N/16), 16, 32)；
        //                非转置 [K,N] -> (ceil(N/32), ceil(K/16), 16, 32)
        const int64_t nzD0 = x2->GetViewShape().GetDim(DIM_ZERO);
        const int64_t nzD1 = x2->GetViewShape().GetDim(DIM_ONE);
        const int64_t kt32 = (kX1 + NZ_FRACTAL_COL - 1) / NZ_FRACTAL_COL;
        const int64_t kt16 = (kX1 + NZ_FRACTAL_ROW - 1) / NZ_FRACTAL_ROW;
        const bool matchTrans = (nzD0 == kt32);
        const bool matchNonTrans = (nzD1 == kt16);
        if (matchTrans != matchNonTrans) {
            // 布局单边确定方向，与 x2_scale 交叉校验
            isTransB = matchTrans;
            if ((scaleNonTrans && isTransB) || (scaleTrans && !isTransB)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "aclnnQuantMatmulReduceScatter: x2_scale direction conflicts with FRACTAL_NZ layout");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else if (scaleTrans) {
            isTransB = true; // NZ 4D 歧义（方形），scale 消歧
        } else if (scaleNonTrans) {
            isTransB = false; // NZ 4D 歧义（方形），scale 消歧
        } else if (scaleAmbiguous) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantMatmulReduceScatter: transB direction cannot be determined: FRACTAL_NZ layout "
                    "ambiguous and x2_scale is square (kd==n) or not matching kd");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (x2DimNum == TWO_DIMS) {
        // ND 或 NZ 2D 逻辑 shape：按 view 形状 [N,K]/[K,N] 判定，x2_scale 形状交叉校验
        if (IsTransposedView2D(x2)) {
            // view 是 storage 的转置表达（.t() 视图）：物理 [K,N] 与物理 [N,K] 的 .t() 都会命中，
            // 且 view 位置给出与物理相反的方向，故方向必须以 scale 为准。
            if (scaleTrans) {
                isTransB = true; // scale [n,kd,2] → 转置 B
            } else if (scaleNonTrans) {
                isTransB = false; // scale [kd,n,2] → 非转置 B
            } else if (scaleAmbiguous) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "aclnnQuantMatmulReduceScatter: transB direction cannot be determined: x2 is a "
                        "transposed view and x2_scale is square (kd==n) or not matching kd");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            const int64_t x2D0 = x2->GetViewShape().GetDim(DIM_ZERO);
            const int64_t x2D1 = x2->GetViewShape().GetDim(DIM_ONE);
            if (x2D1 == kX1View && x2D0 != kX1View) {
                isTransB = true; // [N,K]
            } else if (x2D0 == kX1View && x2D1 != kX1View) {
                isTransB = false; // [K,N]
            } else if (scaleTrans) {
                isTransB = true; // K==N 方阵或非法维度，scale 消歧
            } else if (scaleNonTrans) {
                isTransB = false; // K==N 方阵或非法维度，scale 消歧
            } else if (scaleAmbiguous) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "aclnnQuantMatmulReduceScatter: transB direction cannot be determined: x2 view is "
                        "square/illegal (x2=[%ld,%ld], kX1View=%ld) and x2_scale is square (kd==n) or not "
                        "matching kd",
                        x2D0, x2D1, kX1View);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 位置确定后交叉校验 scale 方向信号
            if ((scaleNonTrans && isTransB) || (scaleTrans && !isTransB)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "aclnnQuantMatmulReduceScatter: x2_scale direction conflicts with x2 view layout");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    } else {
        // 其它维度（ND 非 2D、NZ 非 4D 等）：无法推断方向，直接报错兜底
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulReduceScatter: transB direction cannot be determined: x2 is %ldD (unsupported "
                "for direction inference)",
                x2DimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // ---- x2 归一化（在方向推断之后，依据 isTransB）----
    auto tempX2 = x2;
    const aclTensor* transX2Scale = x2Scale;
    if (isX2Nz) {
        tempX2 = CopyTensor(x2);
    } else if (isTransB && IsTransposedView2D(x2)) {
        // ND 转置 B 且以 .t() 视图表达（view [K,N] over 物理 [N,K]）时翻回 [N,K] 视图
        tempX2 = TransX2Tensor(x2);
        if (tempX2 == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TransX2Tensor failed for x2.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (MC2Aclnn::IsNeedScaleTrans(x2Scale)) {
        // x2_scale 以转置 view 传入（如 [kd,n,2] view over [n,kd,2]）时翻回 [n,kd,2]
        transX2Scale = TransX2Tensor(x2Scale);
        if (transX2Scale == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TransX2Tensor failed for x2Scale.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    aclnnStatus checkShapeRet = CheckShape(x1, tempX2, x1Scale, x2Scale, y, isTransB);
    CHECK_RET(checkShapeRet == ACLNN_SUCCESS, checkShapeRet);

    aclnnStatus ret = aclnnInnerQuantMatmulReduceScatterGetWorkspaceSize(
        context, x1, tempX2, x1Scale, transX2Scale, const_cast<char*>(group), hcclBufferSize, rankSize, isTransB,
        groupSize, const_cast<char*>(commMode), y, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerQuantMatmulReduceScatterGetWorkspaceSize failed, ret=%d", ret);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("aclnnQuantMatmulReduceScatterGetWorkspaceSize end: workspaceSize=%lu", *workspaceSize);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnQuantMatmulReduceScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     const aclrtStream stream)
{
    OP_LOGD("aclnnQuantMatmulReduceScatter start");
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "aclnnQuantMatmulReduceScatter: executor is null.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (NnopbaseSetHcclServerType != nullptr) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerQuantMatmulReduceScatter(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerQuantMatmulReduceScatter failed, ret=%d", ret);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("aclnnQuantMatmulReduceScatter end");
    return ACLNN_SUCCESS;
}
