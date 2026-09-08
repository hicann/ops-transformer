/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "block_attention_residuals_grad_tiling.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/block_attention_residuals_grad_tiling_key.h"
namespace optiling {
using namespace Ops::Transformer::OpTiling;
constexpr uint32_t ALIGN_512B = 512;
constexpr uint32_t ALIGN_256B = 256;
constexpr uint32_t ALIGN_32B = 32;
constexpr uint64_t UB_RESERVE_BYTES = 8UL * 1024UL;
// FULL_H kernel 的 H 轴 buffer 至少预留 64 个元素（归约尾部清零需要）。
constexpr int64_t H_REDUCE_LANE_NUM = 64;
// 各路径 Buffer 构成，与 kernel InitBuffer 逐项对应，避免 UB 估算公式散落魔鬼数字。
constexpr uint64_t A22_H_FP32_BUF_NUM = 6UL;
constexpr uint64_t A22_FULL_H_META_BUF_NUM = 4UL;
constexpr uint64_t A22_SPLIT_H_META_BUF_NUM = 6UL;
constexpr uint64_t A35_FULL_H_FP32_BUF_NUM = 6UL;
constexpr uint64_t A35_FULL_H_INPUT_BUF_NUM = 4UL;
constexpr uint64_t A35_FULL_H_META_BUF_NUM = 4UL;
constexpr uint64_t A35_SPLIT_H_FP32_BUF_NUM = 7UL;
constexpr uint64_t A35_SPLIT_H_INPUT_BUF_NUM = 4UL;
constexpr uint64_t A35_SPLIT_H_META_BUF_NUM = 6UL;
constexpr uint64_t SCALAR_SLOT_NUM = 6UL;
constexpr uint64_t SCALAR_ELEM_PER_SLOT = 8UL;
constexpr uint64_t A35_H_INPUT_ALIGN_ELEMS = 16UL;
constexpr int64_t FP32_ELEM_PER_32B = 8;
constexpr int64_t FP16_ELEM_PER_32B = 16;
constexpr int64_t MAX_NUM_BLOCKS = 128;
constexpr size_t PARTIAL_BLOCK_RANK = 2;
constexpr size_t BLOCK_RES_RANK = 3;
// Input tensor positions, keep in sync with the op def registration order.
constexpr size_t INPUT_PARTIAL_BLOCK = 0;
constexpr size_t INPUT_BLOCK_RES = 1;
constexpr size_t INPUT_PROJ_WEIGHT = 2;
constexpr size_t INPUT_NORM_WEIGHT = 3;
constexpr size_t INPUT_GRAD_HIDDEN_STATES = 4;
constexpr size_t INPUT_INV_NORM = 5;
constexpr size_t INPUT_PROBS = 6;

static uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1UL) / alignment * alignment;
}

// 向上取整除法：value / divisor，不足 1 个 tile 时按 1 个处理。
static int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

// 返回算子支持的输入类型字节数；0 表示不支持的类型。
static uint32_t GetSupportedDtypeBytes(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2U; // FP16/BF16 每个元素占 2 字节
        case ge::DT_FLOAT:
            return 4U; // FP32 每个元素占 4 字节
        default:
            return 0;
    }
}

static bool IsSupportedMainDtype(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT;
}

static const char *DtypeName(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return "FLOAT16";
        case ge::DT_BF16:
            return "BF16";
        case ge::DT_FLOAT:
            return "FLOAT32";
        default:
            return "UNKNOWN";
    }
}

static uint64_t CalcArch22FullHUbBytes(int64_t H, int64_t totalBlocks, uint32_t dtypeBytes)
{
    const int64_t hBufElems = H > H_REDUCE_LANE_NUM ? H : H_REDUCE_LANE_NUM;
    const uint64_t hFloatBytes = AlignUp(static_cast<uint64_t>(hBufElems) * sizeof(float), ALIGN_32B);
    const uint64_t hInputBytes = AlignUp(static_cast<uint64_t>(hBufElems) * dtypeBytes, ALIGN_32B);
    const uint64_t metaBytes = AlignUp(static_cast<uint64_t>(totalBlocks) * sizeof(float), ALIGN_32B);
    return A22_H_FP32_BUF_NUM * hFloatBytes + hInputBytes + A22_FULL_H_META_BUF_NUM * metaBytes + ALIGN_32B;
}

// A2/A3 SPLIT_H 比 FULL_H 多两个 K 轴 Kahan 补偿 Buffer。
static uint64_t CalcArch22SplitHUbBytes(int64_t hiddenTileSize, int64_t totalBlocks, uint32_t dtypeBytes)
{
    const uint64_t hFloatBytes = AlignUp(static_cast<uint64_t>(hiddenTileSize) * sizeof(float), ALIGN_32B);
    const uint64_t hInputBytes = AlignUp(static_cast<uint64_t>(hiddenTileSize) * dtypeBytes, ALIGN_32B);
    const uint64_t metaBytes = AlignUp(static_cast<uint64_t>(totalBlocks) * sizeof(float), ALIGN_32B);
    return A22_H_FP32_BUF_NUM * hFloatBytes + hInputBytes + A22_SPLIT_H_META_BUF_NUM * metaBytes + ALIGN_32B;
}

// A5 FULL_H 用 6 个 H 轴 FP32 Buffer（gswAcc 直接 MTE3 写 Workspace）+ 4 个输入 Buffer
// + 4 个 K 轴 Buffer；FP32 按 256B、输入按 16 元素对齐。
static uint64_t CalcArch35FullHUbBytes(int64_t H, int64_t totalBlocks, uint32_t dtypeBytes)
{
    const uint64_t hFloatBytes = AlignUp(static_cast<uint64_t>(H) * sizeof(float), ALIGN_256B);
    const uint64_t hInputBytes = AlignUp(static_cast<uint64_t>(H), A35_H_INPUT_ALIGN_ELEMS) * dtypeBytes;
    const uint64_t metaBytes = AlignUp(static_cast<uint64_t>(totalBlocks) * sizeof(float), ALIGN_256B);
    constexpr uint64_t scalarBytes = SCALAR_SLOT_NUM * SCALAR_ELEM_PER_SLOT * sizeof(float);
    return A35_FULL_H_FP32_BUF_NUM * hFloatBytes + A35_FULL_H_INPUT_BUF_NUM * hInputBytes +
           A35_FULL_H_META_BUF_NUM * metaBytes + scalarBytes;
}

static int64_t CalcArch22HiddenTileSize(uint64_t ubSize, int64_t H, int64_t totalBlocks, uint32_t dtypeBytes)
{
    // 为运行时和编译器未体现在显式 InitBuffer 中的开销预留安全空间。
    if (ubSize <= UB_RESERVE_BYTES) {
        return 0;
    }
    const uint64_t availableUb = ubSize - UB_RESERVE_BYTES;

    // K 轴 Buffer 必须整段驻留 UB，不能随 H 切分缩小。
    const uint64_t fixedBytes =
        A22_SPLIT_H_META_BUF_NUM * AlignUp(static_cast<uint64_t>(totalBlocks) * sizeof(float), ALIGN_32B) + ALIGN_32B;
    if (availableUb <= fixedBytes) {
        return 0;
    }

    // 每个 H 元素占 6 个 FP32 Buffer + 1 个 T 输入/输出 Buffer，仅这部分随 H 切分缩小。
    const uint64_t bytesPerH = A22_H_FP32_BUF_NUM * sizeof(float) + dtypeBytes;

    // FP32 每 32 字节包含 8 个元素，FP16/BF16 每 32 字节包含 16 个元素。
    const int64_t alignElements = dtypeBytes == sizeof(float) ? FP32_ELEM_PER_32B : FP16_ELEM_PER_32B;

    // 先根据线性 UB 模型计算能够容纳的最大对齐 H tile：
    // fixedBytes + maxTile * bytesPerH <= availableUb。
    int64_t maxTile = static_cast<int64_t>((availableUb - fixedBytes) / bytesPerH);
    maxTile = maxTile / alignElements * alignElements;

    // 用独立对齐的精确模型回退校验，防止 Host Tiling 超 Kernel UB。
    while (maxTile > 0 && CalcArch22SplitHUbBytes(maxTile, totalBlocks, dtypeBytes) > availableUb) {
        maxTile -= alignElements;
    }
    if (maxTile <= 0) {
        return 0;
    }

    // 先求最少 tile 数再均分 H，避免尾块过小。
    const int64_t initialTileNum = CeilDiv(H, maxTile);
    int64_t tileSize = CeilDiv(H, initialTileNum);
    tileSize = CeilDiv(tileSize, alignElements) * alignElements;

    // 均分后若超 maxTile，回退到已校验的 maxTile。
    return tileSize <= maxTile ? tileSize : maxTile;
}

// A5 SPLIT_H 沿用 FULL_H tile Buffer 构成：7 个 H 轴 FP32 Buffer（含 wkspOutQue）、
// 4 个输入 Buffer、6 个 K 轴 FP32 Buffer + 标量槽。
static uint64_t CalcArch35SplitHUbBytes(int64_t hiddenTileSize, int64_t totalBlocks, uint32_t dtypeBytes)
{
    const uint64_t hFloatBytes = AlignUp(static_cast<uint64_t>(hiddenTileSize) * sizeof(float), ALIGN_256B);
    const uint64_t hInputBytes = AlignUp(static_cast<uint64_t>(hiddenTileSize) * dtypeBytes, ALIGN_256B);
    const uint64_t metaBytes = AlignUp(static_cast<uint64_t>(totalBlocks) * sizeof(float), ALIGN_256B);
    constexpr uint64_t scalarBytes = SCALAR_SLOT_NUM * SCALAR_ELEM_PER_SLOT * sizeof(float);
    return A35_SPLIT_H_FP32_BUF_NUM * hFloatBytes + A35_SPLIT_H_INPUT_BUF_NUM * hInputBytes +
           A35_SPLIT_H_META_BUF_NUM * metaBytes + scalarBytes;
}

static int64_t CalcArch35HiddenTileSize(uint64_t ubSize, int64_t H, int64_t totalBlocks, uint32_t dtypeBytes)
{
    if (ubSize <= UB_RESERVE_BYTES) {
        return 0;
    }
    const uint64_t availableUb = ubSize - UB_RESERVE_BYTES;
    constexpr uint64_t scalarBytes = SCALAR_SLOT_NUM * SCALAR_ELEM_PER_SLOT * sizeof(float);
    const uint64_t fixedBytes =
        A35_SPLIT_H_META_BUF_NUM * AlignUp(static_cast<uint64_t>(totalBlocks) * sizeof(float), ALIGN_256B) +
        scalarBytes;
    if (availableUb <= fixedBytes) {
        return 0;
    }

    const uint64_t bytesPerH = A35_SPLIT_H_FP32_BUF_NUM * sizeof(float) + A35_SPLIT_H_INPUT_BUF_NUM * dtypeBytes;
    // tile 同时满足 FP32 和原始 dtype Buffer 的 256B 对齐。
    // FP16/BF16 为 128 元素，FP32 为 64 元素。
    const int64_t alignElements = static_cast<int64_t>(ALIGN_256B / dtypeBytes);
    int64_t maxTile = static_cast<int64_t>((availableUb - fixedBytes) / bytesPerH);
    maxTile = maxTile / alignElements * alignElements;
    while (maxTile > 0 && CalcArch35SplitHUbBytes(maxTile, totalBlocks, dtypeBytes) > availableUb) {
        maxTile -= alignElements;
    }
    if (maxTile <= 0) {
        return 0;
    }

    const int64_t tileNum = CeilDiv(H, maxTile);
    int64_t tileSize = CeilDiv(H, tileNum);
    tileSize = CeilDiv(tileSize, alignElements) * alignElements;
    return tileSize <= maxTile ? tileSize : maxTile;
}

struct HiddenTilingResult {
    const char *archName;
    uint64_t requiredFull;
    int64_t hiddenTileSize;
    bool splitH;
};

// 架构差异集中在这一个函数中。TilingBlockAttentionResidualsGrad 只消费最终计算结果。
static HiddenTilingResult CalcHiddenTiling(const gert::TilingContext *context, uint64_t ub, int64_t H,
                                           int64_t totalBlocks, uint32_t dtypeBytes)
{
    const uint64_t availableUb = ub > UB_RESERVE_BYTES ? ub - UB_RESERVE_BYTES : 0;
    HiddenTilingResult result{};
    if (IsRegbaseSocVersion(context)) {
        result.archName = "arch35";
        result.requiredFull = CalcArch35FullHUbBytes(H, totalBlocks, dtypeBytes);
        result.splitH = result.requiredFull > availableUb;
        result.hiddenTileSize = H;
        if (result.splitH) {
            result.hiddenTileSize = CalcArch35HiddenTileSize(ub, H, totalBlocks, dtypeBytes);
        }
    } else {
        result.archName = "arch22";
        result.requiredFull = CalcArch22FullHUbBytes(H, totalBlocks, dtypeBytes);
        result.splitH = result.requiredFull > availableUb;
        result.hiddenTileSize = H;
        if (result.splitH) {
            result.hiddenTileSize = CalcArch22HiddenTileSize(ub, H, totalBlocks, dtypeBytes);
        }
    }
    return result;
}

static void PrintInfo(gert::TilingContext *context, BlockAttentionResidualsGradTilingData &tilingData)
{
    OP_LOGD(context,
            " B=%ld N=%ld N1=%ld H=%ld hTile=%ld hTileNum=%ld cores=%ld perCoreWkspBytes=%lu "
            "gradScoresOff=%lu varianceScaleOff=%lu",
            tilingData.get_batchSize(), tilingData.get_numBlocks(), tilingData.get_totalBlocks(),
            tilingData.get_hiddenSize(), tilingData.get_hiddenTileSize(), tilingData.get_hiddenTileNum(),
            tilingData.get_coreNum(), tilingData.get_perCoreWkspBytes(), tilingData.get_gradScoresWkspOff(),
            tilingData.get_varianceScaleWkspOff());
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ub, uint32_t &coreNum)
{
    const BlockAttentionResidualsGradCompileInfo *compileInfo =
        context->GetCompileInfo<BlockAttentionResidualsGradCompileInfo>();
    coreNum = compileInfo->coreNum;
    ub = compileInfo->ubSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeBlockAttentionResidualsGrad(gert::TilingContext *context)
{
    const gert::Shape &partialBlockShape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    const gert::Shape &blockResShape = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape());
    if (partialBlockShape.GetDimNum() != PARTIAL_BLOCK_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "partial_block",
                                     std::to_string(partialBlockShape.GetDimNum()).c_str(),
                                     std::to_string(PARTIAL_BLOCK_RANK).c_str());
        return ge::GRAPH_FAILED;
    }
    if (blockResShape.GetDimNum() != BLOCK_RES_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "block_res",
                                     std::to_string(blockResShape.GetDimNum()).c_str(),
                                     std::to_string(BLOCK_RES_RANK).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t B = partialBlockShape.GetDim(0);
    int64_t H = partialBlockShape.GetDim(1);
    int64_t blockResBatch = blockResShape.GetDim(0);
    int64_t N = blockResShape.GetDim(1);
    int64_t blockResHidden = blockResShape.GetDim(2);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *validBlockNum = attrs->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, validBlockNum);
    if (*validBlockNum != -1 && *validBlockNum != N) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "valid_block_num",
                                              std::to_string(*validBlockNum).c_str(),
                                              "valid_block_num must be -1 or block_res.shape[1]");
        return ge::GRAPH_FAILED;
    }
    if (B <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "partialBlock.shape[0]",
                                              std::to_string(B).c_str(),
                                              "partialBlock.shape[0] must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (H <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "partialBlock.shape[1]",
                                              std::to_string(H).c_str(),
                                              "partialBlock.shape[1] must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (N < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockRes.shape[1]", std::to_string(N).c_str(),
                                              "blockRes.shape[1] must be greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    if (B != blockResBatch) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockRes.shape[0]",
                                              std::to_string(blockResBatch).c_str(),
                                              "blockRes.shape[0] must be same as partialBlock.shape[0]");
        return ge::GRAPH_FAILED;
    }
    if (H != blockResHidden) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockRes.shape[2]",
                                              std::to_string(blockResHidden).c_str(),
                                              "blockRes.shape[2] must be same as partialBlock.shape[1]");
        return ge::GRAPH_FAILED;
    }
    // K 轴 meta Buffer 按 totalBlocks = N + 1 驻留 UB，N > MAX_NUM_BLOCKS 时超出设计上限。
    if (N > MAX_NUM_BLOCKS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockRes.shape[1]", std::to_string(N).c_str(),
                                              "blockRes.shape[1] must be less than or equal to 128");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "shape: B=%ld N=%ld H=%ld", B, N, H);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckMainInputDtypeSupported(gert::TilingContext *context, size_t inputIndex,
                                                    const char *inputName)
{
    const gert::CompileTimeTensorDesc *desc = context->GetInputDesc(inputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const ge::DataType dataType = desc->GetDataType();
    if (!IsSupportedMainDtype(dataType)) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), inputName, DtypeName(dataType), "FLOAT16/BFLOAT16/FLOAT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDtypeBlockAttentionResidualsGrad(gert::TilingContext *context)
{
    if (CheckMainInputDtypeSupported(context, INPUT_PARTIAL_BLOCK, "partial_block") != ge::GRAPH_SUCCESS ||
        CheckMainInputDtypeSupported(context, INPUT_BLOCK_RES, "block_res") != ge::GRAPH_SUCCESS ||
        CheckMainInputDtypeSupported(context, INPUT_PROJ_WEIGHT, "proj_weight") != ge::GRAPH_SUCCESS ||
        CheckMainInputDtypeSupported(context, INPUT_NORM_WEIGHT, "norm_weight") != ge::GRAPH_SUCCESS ||
        CheckMainInputDtypeSupported(context, INPUT_GRAD_HIDDEN_STATES, "grad_hidden_states") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType partialBlockDtype = context->GetInputDesc(INPUT_PARTIAL_BLOCK)->GetDataType();
    const ge::DataType blockResDtype = context->GetInputDesc(INPUT_BLOCK_RES)->GetDataType();
    const ge::DataType projWeightDtype = context->GetInputDesc(INPUT_PROJ_WEIGHT)->GetDataType();
    const ge::DataType normWeightDtype = context->GetInputDesc(INPUT_NORM_WEIGHT)->GetDataType();
    const ge::DataType gradHiddenStatesDtype = context->GetInputDesc(INPUT_GRAD_HIDDEN_STATES)->GetDataType();
    if (blockResDtype != partialBlockDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "block_res", DtypeName(blockResDtype),
                                              "main input dtype must be same as partial_block");
        return ge::GRAPH_FAILED;
    }
    if (projWeightDtype != partialBlockDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "proj_weight", DtypeName(projWeightDtype),
                                              "main input dtype must be same as partial_block");
        return ge::GRAPH_FAILED;
    }
    if (normWeightDtype != partialBlockDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "norm_weight", DtypeName(normWeightDtype),
                                              "main input dtype must be same as partial_block");
        return ge::GRAPH_FAILED;
    }
    if (gradHiddenStatesDtype != partialBlockDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "grad_hidden_states",
                                              DtypeName(gradHiddenStatesDtype),
                                              "main input dtype must be same as partial_block");
        return ge::GRAPH_FAILED;
    }

    const ge::DataType invNormDtype = context->GetInputDesc(INPUT_INV_NORM)->GetDataType();
    const ge::DataType probsDtype = context->GetInputDesc(INPUT_PROBS)->GetDataType();
    if (invNormDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "inv_norm", DtypeName(invNormDtype), "FLOAT32");
        return ge::GRAPH_FAILED;
    }
    if (probsDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "probs", DtypeName(probsDtype), "FLOAT32");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(gert::TilingContext *context, int64_t B, int64_t N, int64_t H, uint32_t coreNum,
                                int64_t hiddenTileSize, bool splitH)
{
    // number of blocks (including the prefix-sum block)
    int64_t N1 = N + 1;

    // per-core workspace: H floats, 512B-aligned for DMA efficiency
    uint64_t perCoreWkspBytes = AlignUp(static_cast<uint64_t>(H) * sizeof(float), ALIGN_512B);

    BlockAttentionResidualsGradTilingData tilingData;
    tilingData.set_batchSize(B);
    tilingData.set_numBlocks(N);
    tilingData.set_totalBlocks(N1);
    tilingData.set_hiddenSize(H);
    tilingData.set_hiddenTileSize(hiddenTileSize);
    tilingData.set_hiddenTileNum(CeilDiv(H, hiddenTileSize));
    // per-core batch range and workspace offset are computed in kernel via GetBlockIdx()
    tilingData.set_coreBatchStart(0);
    tilingData.set_coreBatchEnd(B);
    tilingData.set_coreBatchCount(0);
    tilingData.set_gradScoreWeightWkspOff(0);
    tilingData.set_coreNum(static_cast<int64_t>(coreNum));
    tilingData.set_perCoreWkspBytes(perCoreWkspBytes);
    const uint64_t metaWkspBytes =
        AlignUp(static_cast<uint64_t>(B) * static_cast<uint64_t>(N1) * sizeof(float), ALIGN_512B);
    const uint64_t gradScoresWkspOff = splitH ? perCoreWkspBytes * static_cast<uint64_t>(coreNum) : 0UL;
    tilingData.set_gradScoresWkspOff(gradScoresWkspOff);
    tilingData.set_varianceScaleWkspOff(splitH ? gradScoresWkspOff + metaWkspBytes : 0UL);

    // Hard SyncAll requires all launched cores to start together in batch mode.
    context->SetScheduleMode(1);
    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    PrintInfo(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcWorkspaceSize(gert::TilingContext *context, bool splitH)
{
    int64_t H = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape()).GetDim(1);
    int64_t B = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape()).GetDim(0);
    int64_t N = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape()).GetDim(1);
    const BlockAttentionResidualsGradCompileInfo *compileInfo =
        context->GetCompileInfo<BlockAttentionResidualsGradCompileInfo>();
    uint64_t perCoreWkspBytes = AlignUp(static_cast<uint64_t>(H) * sizeof(float), ALIGN_512B);
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    uint64_t userWorkspaceSize = perCoreWkspBytes * static_cast<uint64_t>(compileInfo->coreNum);
    if (splitH) {
        const uint64_t metaWkspBytes =
            AlignUp(static_cast<uint64_t>(B) * static_cast<uint64_t>(N + 1) * sizeof(float), ALIGN_512B);
        // 分别保存 gradScore[B, K] 和 varianceScale[B, K]。
        userWorkspaceSize += 2UL * metaWkspBytes;
    }
    ws[0] = static_cast<size_t>(userWorkspaceSize + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingBlockAttentionResidualsGrad(gert::TilingContext *context)
{
    uint64_t ub = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ub, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "get platform info failed"), return ge::GRAPH_FAILED);
    if (CheckShapeBlockAttentionResidualsGrad(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDtypeBlockAttentionResidualsGrad(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t dtypeBytes = GetSupportedDtypeBytes(context->GetInputDesc(0)->GetDataType());
    const gert::Shape &partialBlockShape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    const gert::Shape &blockResShape = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape());
    const int64_t H = partialBlockShape.GetDim(1);
    const int64_t totalBlocks = blockResShape.GetDim(1) + 1;
    const uint64_t availableUb = ub > UB_RESERVE_BYTES ? ub - UB_RESERVE_BYTES : 0;
    const HiddenTilingResult hiddenTiling = CalcHiddenTiling(context, ub, H, totalBlocks, dtypeBytes);
    OP_CHECK_IF(hiddenTiling.hiddenTileSize <= 0, OP_LOGE(context, "failed to calculate a valid H tile"),
                return ge::GRAPH_FAILED);

    const uint64_t tilingKey =
        hiddenTiling.splitH ? GET_TPL_TILING_KEY(TPL_H_MODE_SPLIT) : GET_TPL_TILING_KEY(TPL_H_MODE_FULL);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context, "tiling path: path=%s arch=%s H=%ld K=%ld dtypeBytes=%u requiredFull=%lu available=%lu hTile=%ld",
            hiddenTiling.splitH ? "SPLIT_H" : "FULL_H", hiddenTiling.archName, H, totalBlocks, dtypeBytes,
            hiddenTiling.requiredFull, availableUb, hiddenTiling.hiddenTileSize);

    OP_CHECK_IF(DoTiling(context, partialBlockShape.GetDim(0), blockResShape.GetDim(1), H, coreNum,
                         hiddenTiling.hiddenTileSize, hiddenTiling.splitH) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "tiling failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcWorkspaceSize(context, hiddenTiling.splitH) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "workspace failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForBlockAttentionResidualsGrad(gert::TilingParseContext *context)
{
    BlockAttentionResidualsGradCompileInfo *compileInfo =
        context->GetCompiledInfo<BlockAttentionResidualsGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    platform_ascendc::PlatformAscendC plat(context->GetPlatformInfo());
    uint32_t aivNum = plat.GetCoreNumAiv();
    compileInfo->coreNum = aivNum;
    OP_CHECK_IF(compileInfo->coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    OP_LOGD(context, "parse: ub=%luB coreNum=%u", compileInfo->ubSize, compileInfo->coreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockAttentionResidualsGrad)
    .Tiling(TilingBlockAttentionResidualsGrad)
    .TilingParse<BlockAttentionResidualsGradCompileInfo>(TilingPrepareForBlockAttentionResidualsGrad);
} // namespace optiling
