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
 * \file quant_matmul_reduce_scatter_tiling.cpp
 * \brief host侧tiling实现
 */

#include <register/op_def_registry.h>
#include "util/math_util.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling_data.h"
#include "../../op_kernel/quant_matmul_reduce_scatter_tiling_key.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling.h"

#include "mc2_log.h"
#include "mc2_log_compat.h"
using namespace AscendC;
using namespace ge;
namespace MC2Tiling {
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint64_t CUBE_BLOCK_NUM = 16;
constexpr uint64_t MAX_BASIC_BLOCK_SIZE = 256;
constexpr uint64_t M_BLOCK_WINDOW_LEN = 4;
constexpr uint64_t UB_TOTAL_BYTES = 248UL * 1024UL;
constexpr uint64_t UB_COST_FACTOR = 12UL; // sumfp32(4b)+localbf16(2b)+localfp32(4b)+sumbf16(2b)

constexpr size_t X1_INDEX = 1;
constexpr size_t X2_INDEX = 2;
constexpr size_t X1_SCALE_INDEX = 3;
constexpr size_t X2_SCALE_INDEX = 4;
constexpr size_t OUTPUT_INDEX = 0;
constexpr uint64_t MX_SCALE_GROUP = 64UL;  // MX 分组宽度（与 device 侧 MXFP_DIVISOR 一致）
constexpr uint64_t MX_SCALE_SUB_DIM = 2UL; // scale 末维：64 组内两个 32 子组

constexpr size_t HCCL_BUFFER_SIZE_INDEX = 1;
constexpr size_t RANK_SIZE_INDEX = 2;
constexpr size_t TRANSPOSE_X2_INDEX = 3;
constexpr size_t GROUP_SIZE_INDEX = 4;
constexpr size_t COMM_MODE_INDEX = 5;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;

constexpr int64_t K_MIN_VAL = 64; // 最小 K=64：MX scale 组/baseK 对齐粒度（tiling baseKAlignNum=MXFP_DIVISOR_SIZE=64）
constexpr int64_t K_MAX_VAL = 65535;
constexpr int64_t MAX_INT32_VAL = 2147483647;

constexpr uint32_t BATCH_MODE_SCHEDULE = 1; // batch mode：所有核同时启动（对齐同仓 MC2 算子 SetScheduleMode 用法）
constexpr uint64_t C_TYPE_BYTES = 2UL;   // CType（bf16/fp16）元素字节宽
constexpr uint64_t UB_ALIGN_UNIT = 32UL; // UB/DMA 搬运 32 字节对齐单位
constexpr uint64_t M_SPLIT_HALVES = 2UL; // M 对半切分（L0C2UB DUAL_DST 双半区 / baseM 回退取半）

// x1/x2 支持的 dtype 白名单（与 def 的 9 组组合一致）：fp8 任意配对，fp4 仅 fp4+fp4
static bool IsSupportedXDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2 || dtype == ge::DT_FLOAT4_E2M1;
}

/**
 * @brief 校验 x1/x2/scale/y 的 dtype 与 x1/x2 format（dtype 泛化后新增）
 *        拦截口径：
 *        1) x1/x2 dtype ∈ {e4m3, e5m2, fp4}；2) fp4 只能与 fp4 配对（双向拦截）；
 *        3) fp4 仅支持 x2 ND（def 配对 + tiling 双保险）；4) x1 恒 ND；
 *        5) scale 固定 e8m0；6) y ∈ {bf16, fp16}（与 def 配对）
 */
static ge::graphStatus CheckDtypeAndFormat(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto x1Desc = context->GetInputDesc(X1_INDEX);
    auto x2Desc = context->GetInputDesc(X2_INDEX);
    OP_TILING_CHECK(x1Desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x2"), return ge::GRAPH_FAILED);
    ge::DataType x1Dtype = x1Desc->GetDataType();
    ge::DataType x2Dtype = x2Desc->GetDataType();
    OP_TILING_CHECK(!IsSupportedXDtype(x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x1", std::to_string(static_cast<int>(x1Dtype)),
                                                          "expect FP8_E4M3FN/FP8_E5M2/FP4_E2M1"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsSupportedXDtype(x2Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x2", std::to_string(static_cast<int>(x2Dtype)),
                                                          "expect FP8_E4M3FN/FP8_E5M2/FP4_E2M1"),
                    return ge::GRAPH_FAILED);
    // fp4 双向拦截：x1/x2 必须同为 fp4 或同为 fp8
    OP_TILING_CHECK(
        (x1Dtype == ge::DT_FLOAT4_E2M1) != (x2Dtype == ge::DT_FLOAT4_E2M1),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            nodeName, "x1, x2",
            (std::to_string(static_cast<int>(x1Dtype)) + ", " + std::to_string(static_cast<int>(x2Dtype))).c_str(),
            "FP4 requires both x1 and x2 to be FP4_E2M1"),
        return ge::GRAPH_FAILED);

    ge::Format x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x1", std::to_string(static_cast<int>(x1Format)), "ND"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Format != ge::FORMAT_ND && x2Format != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(nodeName, "x2", std::to_string(static_cast<int>(x2Format)),
                                                            "ND or FRACTAL_NZ"),
                    return ge::GRAPH_FAILED);
    // fp4 仅支持 ND：def 中 fp4 只配 ND format，此处为 tiling 侧双保险
    OP_TILING_CHECK(x1Dtype == ge::DT_FLOAT4_E2M1 && x2Format != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x2", std::to_string(static_cast<int>(x2Format)), "ND (FP4)"),
                    return ge::GRAPH_FAILED);

    auto x1ScaleDesc = context->GetInputDesc(X1_SCALE_INDEX);
    auto x2ScaleDesc = context->GetInputDesc(X2_SCALE_INDEX);
    OP_TILING_CHECK(x1ScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x1_scale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2ScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x2_scale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        x1ScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0 || x2ScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, "x1_scale, x2_scale",
                                               (std::to_string(static_cast<int>(x1ScaleDesc->GetDataType())) + ", " +
                                                std::to_string(static_cast<int>(x2ScaleDesc->GetDataType())))
                                                   .c_str(),
                                               "scale dtype must be FP8_E8M0"),
        return ge::GRAPH_FAILED);

    auto yDesc = context->GetOutputDesc(OUTPUT_INDEX);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yDesc->GetDataType() != ge::DT_BF16 && yDesc->GetDataType() != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "y", std::to_string(static_cast<int>(yDesc->GetDataType())), "expect BF16 or FP16"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingKey, TPL 模板实参方式生成
 *        TRANSB 由 attr transpose_x2 决定（对应 def 的 Attr(2)），而非 x2 format：
 *        transpose_x2=true  → x2 为转置右矩阵 [N,K]（ND/DN） → LayoutB=DNExt，key=GET_TPL_TILING_KEY(false)
 *        transpose_x2=false → x2 为 [K,N]（NZ）             → LayoutB=NZ，    key=GET_TPL_TILING_KEY(true)
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param transposeX2: attr transpose_x2（B 是否转置）
 */
// hccl_buffer_size attr 校验：> 0（通信 buffer 容量）
static ge::graphStatus CheckHcclBufferSizeAttr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto hcclBufferSizePtr = context->GetAttrs()->GetAttrPointer<int64_t>(HCCL_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(hcclBufferSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "hccl_buffer_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *hcclBufferSizePtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hccl_buffer_size", std::to_string(*hcclBufferSizePtr), "positive integer"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupSizeAttr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto groupSizePtr = context->GetAttrs()->GetAttrPointer<int64_t>(GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group_size"),
                    return ge::GRAPH_FAILED);
    const int64_t gs = *groupSizePtr;
    if (gs == 0) {
        return ge::GRAPH_SUCCESS; // 自动推断 [1,1,32]
    }
    const int64_t gsM = (gs >> 32) & 0xFFFF;
    const int64_t gsN = (gs >> 16) & 0xFFFF;
    const int64_t gsK = gs & 0xFFFF;
    OP_TILING_CHECK(
        gsM != 1 || gsN != 1 || gsK != 32,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "group_size",
                                  (std::to_string(gsM) + "," + std::to_string(gsN) + "," + std::to_string(gsK)).c_str(),
                                  "[1,1,32]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCommModeAttr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto commModePtr = context->GetAttrs()->GetAttrPointer<char>(COMM_MODE_INDEX);
    OP_TILING_CHECK(commModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "comm_mode"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::string(commModePtr) != "ubmem",
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "comm_mode", commModePtr, "ubmem"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingKey(gert::TilingContext* context, const bool transposeX2)
{
    context->SetTilingKey(GET_TPL_TILING_KEY(transposeX2));
}

static ge::graphStatus SetWorkSpace(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"), return ge::GRAPH_FAILED);

    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

// 解析 M, K, N 并校验 x2 format 与 K 对齐：
//   ND 分支按 transpose_x2 区分 DN（[N,K] 的 .t() 视图，N=dim0）与 ND（[K,N]，N=dim1），并校验 x2.K == x1.K；
//   FRACTAL_NZ 分支（NZ/ZN）N 从 output shape 取（与转置无关）
static ge::graphStatus ParseMNK(gert::TilingContext* context, const bool transposeX2, uint64_t& M, uint64_t& K,
                                uint64_t& N)
{
    const char* nodeName = context->GetNodeName();
    const gert::StorageShape* x1Shape = context->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context->GetInputShape(X2_INDEX);
    const gert::StorageShape* outputShape = context->GetOutputShape(OUTPUT_INDEX);

    OP_TILING_CHECK(x1Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x2"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(outputShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);

    M = x1Shape->GetStorageShape().GetDim(DIM_ZERO);
    K = x1Shape->GetStorageShape().GetDim(DIM_ONE);
    N = 0;

    ge::Format x2Format =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    if (x2Format == ge::FORMAT_ND) {
        if (transposeX2) {
            // DN：x2 为 [N,K] 物理存储（[K,N] 的 .t() 视图），N = storage dim(0)
            N = x2Shape->GetStorageShape().GetDim(DIM_ZERO);
        } else {
            // ND：x2 为 [K,N] 连续，N = dim(1)
            N = x2Shape->GetStorageShape().GetDim(DIM_ONE);
        }
        uint64_t kX2 =
            (transposeX2 ? x2Shape->GetStorageShape().GetDim(DIM_ONE) : x2Shape->GetStorageShape().GetDim(DIM_ZERO));
        OP_TILING_CHECK(
            kX2 != K,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                nodeName, "x1, x2", (std::to_string(K) + ", " + std::to_string(kX2)).c_str(), "x1.K must equal x2.K"),
            return ge::GRAPH_FAILED);
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        // NZ/ZN：N 从 output shape 取（与转置无关）
        N = static_cast<uint64_t>(outputShape->GetStorageShape().GetDim(DIM_ONE));
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x2", std::to_string(static_cast<int>(x2Format)), "ND or FRACTAL_NZ");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// scale shape 校验：x1_scale 恒为 [M, kd, 2]；DN 分支 x2_scale 为 [N, kd, 2]，NZ/ND 分支为 [kd, n, 2]
static ge::graphStatus CheckScaleShape(gert::TilingContext* context, const uint64_t M, const uint64_t N,
                                       const bool transposeX2)
{
    const char* nodeName = context->GetNodeName();
    const gert::StorageShape* x1ScaleShape = context->GetInputShape(X1_SCALE_INDEX);
    const gert::StorageShape* x2ScaleShape = context->GetInputShape(X2_SCALE_INDEX);
    OP_TILING_CHECK(x1ScaleShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x1_scale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2ScaleShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x2_scale"), return ge::GRAPH_FAILED);
    // kd 从 x1_scale 形状取（逻辑 K 粒度 ceil(K/64)，fp4 打包无关，与 aclnn 推断一致）
    OP_TILING_CHECK(x1ScaleShape->GetStorageShape().GetDimNum() != DIM_THREE,
                    OP_LOGE_FOR_INVALID_SHAPE(nodeName, "x1_scale", "mismatch", "must be 3D [M, kd, 2]"),
                    return ge::GRAPH_FAILED);
    const uint64_t kd = static_cast<uint64_t>(x1ScaleShape->GetStorageShape().GetDim(DIM_ONE));
    OP_TILING_CHECK(x1ScaleShape->GetStorageShape().GetDim(DIM_ZERO) != M ||
                        x1ScaleShape->GetStorageShape().GetDim(DIM_TWO) != MX_SCALE_SUB_DIM,
                    OP_LOGE_FOR_INVALID_SHAPE(nodeName, "x1_scale", "mismatch",
                                              ("[" + std::to_string(M) + "," + std::to_string(kd) + ",2]").c_str()),
                    return ge::GRAPH_FAILED);
    if (transposeX2) {
        // 转置（DN/ZN）：scale 语义 [N, kd, 2]
        OP_TILING_CHECK(x2ScaleShape->GetStorageShape().GetDimNum() != DIM_THREE ||
                            x2ScaleShape->GetStorageShape().GetDim(DIM_ZERO) != N ||
                            x2ScaleShape->GetStorageShape().GetDim(DIM_ONE) != kd ||
                            x2ScaleShape->GetStorageShape().GetDim(DIM_TWO) != MX_SCALE_SUB_DIM,
                        OP_LOGE_FOR_INVALID_SHAPE(nodeName, "x2_scale", "mismatch (transposed)",
                                                  ("[" + std::to_string(N) + "," + std::to_string(kd) + ",2]").c_str()),
                        return ge::GRAPH_FAILED);
    } else {
        // 非转置（ND/NZ）：scale 语义 [kd, N, 2]
        OP_TILING_CHECK(x2ScaleShape->GetStorageShape().GetDimNum() != DIM_THREE ||
                            x2ScaleShape->GetStorageShape().GetDim(DIM_ZERO) != kd ||
                            x2ScaleShape->GetStorageShape().GetDim(DIM_ONE) != N ||
                            x2ScaleShape->GetStorageShape().GetDim(DIM_TWO) != MX_SCALE_SUB_DIM,
                        OP_LOGE_FOR_INVALID_SHAPE(nodeName, "x2_scale", "mismatch (non-transposed)",
                                                  ("[" + std::to_string(kd) + "," + std::to_string(N) + ",2]").c_str()),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// 计算 dynMaxBaseM：baseM 取 tpSizeM 的 ceil(rankSize/4) 倍，使 mBlockCnt = rankSize/ceil(rankSize/4) <= 窗口宽4，
// 单窗口行线性走位，每一轮分核都覆盖全部 M 块，alltoall 全程喂满所有 peer 链路；
// rankSize <= 4 时倍数为 1，退化为原有逻辑；不足一个 cube block 时逐级回退
static uint64_t ComputeDynMaxBaseM(const uint64_t M, const uint64_t rankSize)
{
    const uint64_t tpSizeM = M / rankSize;
    const uint64_t mSplitFactor = (rankSize + M_BLOCK_WINDOW_LEN - 1) / M_BLOCK_WINDOW_LEN;
    uint64_t dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, (mSplitFactor * tpSizeM / CUBE_BLOCK_NUM) * CUBE_BLOCK_NUM);
    if (dynMaxBaseM < CUBE_BLOCK_NUM) {
        dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, (M / M_SPLIT_HALVES / CUBE_BLOCK_NUM) * CUBE_BLOCK_NUM);
    }
    if (dynMaxBaseM < CUBE_BLOCK_NUM) {
        dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, M);
    }
    return dynMaxBaseM;
}

// 按 x1 dtype 分发 SWAT 引擎实例（fp8 统一按 e4m3 实例化，fp4 统一按 e2m1 实例化；
// e5m2 与 e4m3 位宽/分组一致，tiling 结果相同，无需按 fp8 具体类型细分）
static ge::graphStatus GetMatmulTilingData(gert::TilingContext* context, const uint64_t M, const uint64_t N,
                                           const uint64_t K, const bool transB, const uint64_t dynMaxBaseM,
                                           QuantMatmulTilingData* tilingData)
{
    if (context->GetInputDesc(X1_INDEX)->GetDataType() == ge::DT_FLOAT4_E2M1) {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1, mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetMaxBaseM(dynMaxBaseM);
        tilingEngine.GetTilingData(M, N, K, false, transB, *tilingData);
    } else {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                              mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetMaxBaseM(dynMaxBaseM);
        tilingEngine.GetTilingData(M, N, K, false, transB, *tilingData);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckUbBudget(const QuantMatmulTilingData* tilingData, const uint64_t N, const char* nodeName)
{
    const uint64_t mmHalfRows = (static_cast<uint64_t>(tilingData->baseM) + 1UL) / M_SPLIT_HALVES;
    const uint64_t mmAlignNF32 =
        (static_cast<uint64_t>(tilingData->baseN) + UB_ALIGN_UNIT - 1UL) & ~(UB_ALIGN_UNIT - 1UL);
    const uint64_t mmPadRowBytes =
        Ops::Base::CeilDiv(static_cast<uint64_t>(tilingData->baseN) * C_TYPE_BYTES, UB_ALIGN_UNIT) * UB_ALIGN_UNIT;
    const uint64_t mmUbCost = mmHalfRows * mmAlignNF32 * sizeof(float) + mmHalfRows * mmPadRowBytes;
    if (mmUbCost > UB_TOTAL_BYTES) {
        OP_LOGE(nodeName, "Exceeds UB limit in matmul stage: baseM=%u baseN=%u cost=%lu", tilingData->baseM,
                tilingData->baseN, mmUbCost);
        return ge::GRAPH_FAILED;
    }

    const uint64_t rowPaddingBytes =
        Ops::Base::CeilDiv(static_cast<uint64_t>(N) * C_TYPE_BYTES, static_cast<uint64_t>(UB_ALIGN_UNIT)) *
        UB_ALIGN_UNIT;
    const uint64_t paddingN = rowPaddingBytes / C_TYPE_BYTES;
    const uint64_t ubTotalCost = paddingN * UB_COST_FACTOR;
    if (ubTotalCost > UB_TOTAL_BYTES) {
        OP_LOGE(nodeName, "Exceeds UB limit when N = %lu", N);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext* context, QuantMatmulTilingData* tilingData,
                                     const bool transposeX2)
{
    const char* nodeName = context->GetNodeName();

    uint64_t M = 0;
    uint64_t K = 0;
    uint64_t N = 0;
    OP_TILING_CHECK(ParseMNK(context, transposeX2, M, K, N) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "ParseMNK failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckScaleShape(context, M, N, transposeX2) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckScaleShape failed."), return ge::GRAPH_FAILED);

    OP_LOGD(nodeName, "Input shape: M=%lu, K=%lu, N=%lu", M, K, N);

    const gert::StorageShape* x1ScaleShapeForK = context->GetInputShape(X1_SCALE_INDEX);
    const uint64_t kLogic = static_cast<uint64_t>(x1ScaleShapeForK->GetStorageShape().GetDim(DIM_ONE)) * MX_SCALE_GROUP;
    const bool x1IsFp4 = context->GetInputDesc(X1_INDEX)->GetDataType() == ge::DT_FLOAT4_E2M1;
    const uint64_t kExact = x1IsFp4 ? K * 2 : K;
    OP_TILING_CHECK(
        kExact < K_MIN_VAL || static_cast<int64_t>(kLogic) >= K_MAX_VAL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "K", std::to_string(kExact),
            (std::string("must be in range [") + std::to_string(K_MIN_VAL) + ", " + std::to_string(K_MAX_VAL) + ")")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(M > static_cast<uint64_t>(MAX_INT32_VAL) || N > static_cast<uint64_t>(MAX_INT32_VAL),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "M/N", (std::to_string(M) + "/" + std::to_string(N)), "must not exceed INT32_MAX"),
                    return ge::GRAPH_FAILED);

    // 使用 SWAT tiling engine 计算 matmul tiling 参数
    const uint32_t* rankSizePtr = context->GetAttrs()->GetAttrPointer<uint32_t>(RANK_SIZE_INDEX);
    OP_TILING_CHECK(rankSizePtr == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "rank_size", "nullptr", "required attr"),
                    return ge::GRAPH_FAILED);
    uint32_t rankSize = *rankSizePtr;
    OP_TILING_CHECK(rankSize == 0, OP_LOGE_FOR_INVALID_VALUE(nodeName, "rank_size", "0", "positive integer"),
                    return ge::GRAPH_FAILED);

    // transB 由 attr transpose_x2 决定（与 x2 format 正交，对齐 kernel 侧 TRANSB 模板参数）：
    //   转置 DN/ZN → true（B [N,K]）；非转置 ND/NZ → false（B [K,N]）
    bool transB = transposeX2;
    OP_TILING_CHECK(
        M % rankSize != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "M", (std::to_string(M) + " % " + std::to_string(rankSize)).c_str(),
                                  "divisible by rank_size"),
        return ge::GRAPH_FAILED);
    const gert::StorageShape* outputShapeForY = context->GetOutputShape(OUTPUT_INDEX);
    OP_TILING_CHECK(outputShapeForY == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);
    const uint64_t yM = M / rankSize;
    OP_TILING_CHECK(outputShapeForY->GetStorageShape().GetDimNum() != DIM_TWO ||
                        outputShapeForY->GetStorageShape().GetDim(DIM_ZERO) != yM ||
                        outputShapeForY->GetStorageShape().GetDim(DIM_ONE) != N,
                    OP_LOGE_FOR_INVALID_SHAPE(nodeName, "y", "mismatch",
                                              ("[" + std::to_string(yM) + "," + std::to_string(N) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    const uint64_t dynMaxBaseM = ComputeDynMaxBaseM(M, rankSize);

    OP_TILING_CHECK(GetMatmulTilingData(context, M, N, K, transB, dynMaxBaseM, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "GetMatmulTilingData failed."), return ge::GRAPH_FAILED);

    OP_LOGD(nodeName, "QuantMatmulTilingSwat computed tiling data successfully");
    OP_LOGD(nodeName, "  tilingData.m=%u, n=%u, k=%u", tilingData->m, tilingData->n, tilingData->k);
    OP_LOGD(nodeName, "  tilingData.baseM=%u, baseN=%u, baseK=%u", tilingData->baseM, tilingData->baseN,
            tilingData->baseK);
    OP_LOGD(nodeName, "  tilingData.usedCoreNum=%u", tilingData->usedCoreNum);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());

    OP_LOGD(nodeName, "Custom tiling parameters filled: rankSize=%u, baseM=%u, baseN=%u", rankSize, tilingData->baseM,
            tilingData->baseN);

    OP_TILING_CHECK(CheckUbBudget(tilingData, N, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckUbBudget failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief quant_matmul_reduce_scatter算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus QuantMatmulReduceScatterTilingFunc(gert::TilingContext* context)
{
    OP_LOGD("QuantMatmulReduceScatter tiling start.");
    OP_TILING_CHECK(
        context == nullptr,
        OP_LOGE("quant_matmul_reduce_scatter", "failed to get tiling context in quant_matmul_reduce_scatter."),
        return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
                    OP_LOGE("quant_matmul_reduce_scatter", "failed to get nodeName in quant_matmul_reduce_scatter."),
                    return ge::GRAPH_FAILED);

    QuantMatmulTilingData* tilingData = context->GetTilingData<QuantMatmulTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr in quant_matmul_reduce_scatter."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckDtypeAndFormat(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check dtype/format failed in quant_matmul_reduce_scatter."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckHcclBufferSizeAttr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check hccl_buffer_size attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckGroupSizeAttr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check group_size attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckCommModeAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check comm_mode attr failed."),
                    return ge::GRAPH_FAILED);

    // tiling参数设置
    auto transposeX2Ptr = context->GetAttrs()->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(transposeX2Ptr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "transpose_x2"),
                    return ge::GRAPH_FAILED);
    const bool transposeX2 = *transposeX2Ptr;
    OP_LOGD(nodeName, "attr transpose_x2 = %d, x2Format = %d", static_cast<int>(transposeX2),
            static_cast<int>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat())));
    SetWorkSpace(context);
    OP_TILING_CHECK(SetTilingData(context, tilingData, transposeX2) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "SetTilingData failed."), return ge::GRAPH_FAILED);
    SetTilingKey(context, transposeX2);
    OP_LOGD("QuantMatmulReduceScatter tiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForQuantMatmulReduceScatter(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
struct QuantMatmulReduceScatterCompileInfo {};

IMPL_OP_OPTILING(QuantMatmulReduceScatter)
    .Tiling(QuantMatmulReduceScatterTilingFunc)
    .TilingParse<QuantMatmulReduceScatterCompileInfo>(TilingParseForQuantMatmulReduceScatter);

} // namespace MC2Tiling
