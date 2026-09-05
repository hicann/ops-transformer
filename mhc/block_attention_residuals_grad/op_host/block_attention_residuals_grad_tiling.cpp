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

static uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1UL) / alignment * alignment;
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
    const int64_t initialTileNum = (H + maxTile - 1) / maxTile;
    int64_t tileSize = (H + initialTileNum - 1) / initialTileNum;
    tileSize = (tileSize + alignElements - 1) / alignElements * alignElements;

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

    const int64_t tileNum = (H + maxTile - 1) / maxTile;
    int64_t tileSize = (H + tileNum - 1) / tileNum;
    tileSize = (tileSize + alignElements - 1) / alignElements * alignElements;
    return tileSize <= maxTile ? tileSize : maxTile;
}

struct HiddenTilingResult {
    const char *archName;
    uint64_t requiredFull;
    int64_t hiddenTileSize;
    bool splitH;
};

// 架构差异集中在这一个函数中。TilingFunc 只消费最终计算结果。
static HiddenTilingResult CalcHiddenTiling(const gert::TilingContext *ctx, uint64_t ub, int64_t H, int64_t totalBlocks,
                                           uint32_t dtypeBytes)
{
    const uint64_t availableUb = ub > UB_RESERVE_BYTES ? ub - UB_RESERVE_BYTES : 0;
    HiddenTilingResult result{};
    if (IsRegbaseSocVersion(ctx)) {
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

static void PrintInfo(gert::TilingContext *ctx, BlockAttentionResidualsGradTilingData &td)
{
    OP_LOGD(ctx,
            " B=%ld N=%ld N1=%ld H=%ld hTile=%ld hTileNum=%ld cores=%ld perWksp=%lu "
            "gradScoresOff=%lu varianceScaleOff=%lu",
            td.get_batchSize(), td.get_numBlocks(), td.get_totalBlocks(), td.get_hiddenSize(), td.get_hiddenTileSize(),
            td.get_hiddenTileNum(), td.get_coreNum(), td.get_perCoreWkspBytes(), td.get_gradScoresWkspOff(),
            td.get_varianceScaleWkspOff());
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext *ctx, uint64_t &ub, uint32_t &coreNum)
{
    auto ci = ctx->GetCompileInfo<BlockAttentionResidualsGradCompileInfo>();
    coreNum = ci->coreNum;
    ub = ci->ubSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Check(gert::TilingContext *ctx)
{
    auto ps = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape());
    auto br = EnsureNotScalar(ctx->GetInputShape(1)->GetStorageShape());
    OP_CHECK_IF(ps.GetDimNum() != 2, OP_LOGE(ctx, "partial_block must be 2D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(br.GetDimNum() != 3, OP_LOGE(ctx, "block_res must be 3D"), return ge::GRAPH_FAILED);

    int64_t B = ps.GetDim(0);
    int64_t H = ps.GetDim(1);
    int64_t Bb = br.GetDim(0);
    int64_t N = br.GetDim(1);
    int64_t Hb = br.GetDim(2);
    OP_CHECK_IF(B <= 0 || N < 0 || H <= 0, OP_LOGE(ctx, "B and H must be positive and N must be non-negative"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(B != Bb || H != Hb, OP_LOGE(ctx, "shape mismatch"), return ge::GRAPH_FAILED);
    // K 轴 meta Buffer 按 totalBlocks = N + 1 驻留 UB，N > MAX_NUM_BLOCKS 时超出设计上限。
    OP_CHECK_IF(N > MAX_NUM_BLOCKS, OP_LOGE(ctx, "numBlocks N must be <= %ld, got %ld", MAX_NUM_BLOCKS, N),
                return ge::GRAPH_FAILED);

    OP_LOGD(ctx, "shape: B=%ld N=%ld H=%ld", B, N, H);
    return ge::GRAPH_SUCCESS;
}

// 返回算子支持的输入类型字节数；0 表示不支持的类型。
static uint32_t GetSupportedDtypeBytes(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return sizeof(uint16_t);
        case ge::DT_FLOAT:
            return sizeof(float);
        default:
            return 0;
    }
}

static ge::graphStatus DoTiling(gert::TilingContext *ctx, int64_t B, int64_t N, int64_t H, uint32_t coreNum,
                                int64_t hiddenTileSize, bool splitH)
{
    // number of blocks (including the prefix-sum block)
    int64_t N1 = N + 1;

    // per-core workspace: H floats, 512B-aligned for DMA efficiency
    uint64_t perWksp = (static_cast<uint64_t>(H) * sizeof(float) + ALIGN_512B - 1) / ALIGN_512B * ALIGN_512B;

    BlockAttentionResidualsGradTilingData td;
    td.set_batchSize(B);
    td.set_numBlocks(N);
    td.set_totalBlocks(N1);
    td.set_hiddenSize(H);
    td.set_hiddenTileSize(hiddenTileSize);
    td.set_hiddenTileNum((H + hiddenTileSize - 1) / hiddenTileSize);
    // per-core batch range and workspace offset are computed in kernel via GetBlockIdx()
    td.set_coreBatchStart(0);
    td.set_coreBatchEnd(B);
    td.set_coreBatchCount(0);
    td.set_gradScoreWeightWkspOff(0);
    td.set_coreNum(static_cast<int64_t>(coreNum));
    td.set_perCoreWkspBytes(perWksp);
    const uint64_t metaWkspBytes =
        AlignUp(static_cast<uint64_t>(B) * static_cast<uint64_t>(N1) * sizeof(float), ALIGN_512B);
    const uint64_t gradScoresWkspOff = splitH ? perWksp * static_cast<uint64_t>(coreNum) : 0UL;
    td.set_gradScoresWkspOff(gradScoresWkspOff);
    td.set_varianceScaleWkspOff(splitH ? gradScoresWkspOff + metaWkspBytes : 0UL);

    // Hard SyncAll requires all launched cores to start together in batch mode.
    ctx->SetScheduleMode(1);
    ctx->SetBlockDim(coreNum);
    td.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(td.GetDataSize());

    PrintInfo(ctx, td);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcWorkspaceSize(gert::TilingContext *ctx, bool splitH)
{
    int64_t H = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape()).GetDim(1);
    int64_t B = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape()).GetDim(0);
    int64_t N = EnsureNotScalar(ctx->GetInputShape(1)->GetStorageShape()).GetDim(1);
    auto ci = ctx->GetCompileInfo<BlockAttentionResidualsGradCompileInfo>();
    uint64_t per = (static_cast<uint64_t>(H) * sizeof(float) + ALIGN_512B - 1) / ALIGN_512B * ALIGN_512B;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *ws = ctx->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ws);
    uint64_t userWorkspaceSize = per * static_cast<uint64_t>(ci->coreNum);
    if (splitH) {
        const uint64_t metaWkspBytes =
            AlignUp(static_cast<uint64_t>(B) * static_cast<uint64_t>(N + 1) * sizeof(float), ALIGN_512B);
        // 分别保存 gradScore[B, K] 和 varianceScale[B, K]。
        userWorkspaceSize += 2UL * metaWkspBytes;
    }
    ws[0] = static_cast<size_t>(userWorkspaceSize + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *ctx)
{
    uint64_t ub = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(ctx, ub, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "get platform info failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Check(ctx) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "check shape failed"), return ge::GRAPH_FAILED);

    auto inputDesc = ctx->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    const uint32_t dtypeBytes = GetSupportedDtypeBytes(dataType);
    OP_CHECK_IF(dtypeBytes == 0, OP_LOGE(ctx, "unsupported dtype: %d", static_cast<int32_t>(dataType)),
                return ge::GRAPH_FAILED);
    auto ps = EnsureNotScalar(ctx->GetInputShape(0)->GetStorageShape());
    auto br = EnsureNotScalar(ctx->GetInputShape(1)->GetStorageShape());
    const int64_t H = ps.GetDim(1);
    const int64_t totalBlocks = br.GetDim(1) + 1;
    const uint64_t availableUb = ub > UB_RESERVE_BYTES ? ub - UB_RESERVE_BYTES : 0;
    const HiddenTilingResult hiddenTiling = CalcHiddenTiling(ctx, ub, H, totalBlocks, dtypeBytes);
    OP_CHECK_IF(hiddenTiling.hiddenTileSize <= 0, OP_LOGE(ctx, "failed to calculate a valid H tile"),
                return ge::GRAPH_FAILED);

    const uint64_t tilingKey =
        hiddenTiling.splitH ? GET_TPL_TILING_KEY(TPL_H_MODE_SPLIT) : GET_TPL_TILING_KEY(TPL_H_MODE_FULL);
    ctx->SetTilingKey(tilingKey);
    OP_LOGD(ctx, "tiling path: path=%s arch=%s H=%ld K=%ld dtypeBytes=%u requiredFull=%lu available=%lu hTile=%ld",
            hiddenTiling.splitH ? "SPLIT_H" : "FULL_H", hiddenTiling.archName, H, totalBlocks, dtypeBytes,
            hiddenTiling.requiredFull, availableUb, hiddenTiling.hiddenTileSize);

    OP_CHECK_IF(DoTiling(ctx, ps.GetDim(0), br.GetDim(1), H, coreNum, hiddenTiling.hiddenTileSize,
                         hiddenTiling.splitH) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "tiling failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcWorkspaceSize(ctx, hiddenTiling.splitH) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "workspace failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Parse(gert::TilingParseContext *ctx)
{
    auto ci = ctx->GetCompiledInfo<BlockAttentionResidualsGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ci);

    auto plat = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint32_t aivNum = plat.GetCoreNumAiv();
    ci->coreNum = aivNum;
    OP_CHECK_IF(ci->coreNum == 0, OP_LOGE(ctx, "coreNum is 0"), return ge::GRAPH_FAILED);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ci->ubSize);
    OP_CHECK_IF(ci->ubSize == 0, OP_LOGE(ctx, "ubSize is 0"), return ge::GRAPH_FAILED);

    OP_LOGD(ctx, "parse: ub=%luB coreNum=%u", ci->ubSize, ci->coreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockAttentionResidualsGrad)
    .Tiling(TilingFunc)
    .TilingParse<BlockAttentionResidualsGradCompileInfo>(Parse);
} // namespace optiling
