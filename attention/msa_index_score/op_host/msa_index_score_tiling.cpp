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
 * \file msa_index_score_tiling.cpp
 * \brief MsaIndexScore Tiling 实现（A2/A3 / 950：PA BBND/BNBD + TND packed key；sparse_mode 0/3；
 *        PA key 允许 dim0 非连续，strideKvBlock 来自 GetInputStride）。
 */

#include "msa_index_score_tiling.h"
#include "../op_kernel/msa_index_score_common.h"

#include <cstring>
#include <limits>
#include <string>
#include "graph/utils/type_utils.h"

using namespace ge;
using namespace MsaIndexScoreNs;

namespace optiling {
namespace {
constexpr uint32_t MSA_QUERY_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_TND_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_PA_DIM_NUM = 4;
constexpr uint32_t MSA_BLOCK_TABLE_DIM_NUM = 2;
constexpr uint32_t MSA_SCALE_PA_DIM_NUM = 3;
constexpr uint32_t MSA_SCALE_TND_DIM_NUM = 2;
constexpr uint32_t MSA_ATTEN_MASK_DIM_NUM = 2;
constexpr uint32_t MSA_DIM_0 = 0;
constexpr uint32_t MSA_DIM_1 = 1;
constexpr uint32_t MSA_DIM_2 = 2;
constexpr uint32_t MSA_DIM_3 = 3;
constexpr uint32_t MSA_RANK_1D = 1;
constexpr uint32_t MSA_QLEN_MIN_DIM0 = 2; // prefix-sum 长度至少为 B+1

inline bool IsFp8ComputeDtype(ge::DataType dt)
{
    return dt == ge::DT_HIFLOAT8 || dt == ge::DT_FLOAT8_E5M2 || dt == ge::DT_FLOAT8_E4M3FN;
}

inline bool IsNonQuantQueryDtype(ge::DataType dt, bool isAscend950)
{
    if (dt == ge::DT_BF16 || dt == ge::DT_FLOAT16) {
        return true;
    }
    return isAscend950 && IsFp8ComputeDtype(dt);
}

inline uint32_t RoundUpU32(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

// MIX：按估计 M-task 数启动 AIC，避免短 decode 打满空核（scheduler.Init GetValue / MODE4 VC 空转）。
// Host 读不到 per-request q_len。kernel TotalTasks = Σ CeilDiv(qLen_b * Hq, MSA_ROW_TILE_M)。
// Σ ceil(x_i) <= ceil(Σ x_i) + B；B=1 时 packed == actual。B>1 用 packed+B 上界，避免少估把
// 多 tile 请求串到已启动核上。多 M-tile / 大 batch 仍截到全量 AIC。
inline uint32_t EstPackedMTasks(const MsaIndexScoreInfo& info, uint32_t aicNum)
{
    if (info.totalQ == 0U || aicNum == 0U) {
        return 1U;
    }
    const uint64_t packedRows = static_cast<uint64_t>(info.totalQ) * static_cast<uint64_t>(info.numQHeads);
    const uint32_t packedM =
        static_cast<uint32_t>((packedRows + static_cast<uint64_t>(MSA_ROW_TILE_M) - 1U) / MSA_ROW_TILE_M);
    uint32_t est = packedM;
    if (info.batch > 1U) {
        const uint32_t add = packedM + info.batch;
        est = (add < packedM) ? aicNum : add;
    }
    if (est < 1U) {
        est = 1U;
    }
    if (est > aicNum) {
        est = aicNum;
    }
    return est;
}

// 可见 KV page 上界：优先 host 可读的 actual_seq_klen；否则 min(表宽, PA numPages)。
// GetData() 对 device tensor 也返回非空指针，解引用会段错误；仅 kOnHost/kFollowing 可读。
inline bool IsHostVisibleTensor(const gert::Tensor* tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto placement = tensor->GetPlacement();
    return (placement == gert::kOnHost) || (placement == gert::kFollowing);
}

// 手写 op_api 对未传的可选输入给 AllocTensor 空占位（0 维）。占位视同未传。
inline bool IsOptionalProvided(const gert::StorageShape* shape)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() > 0;
}

// 可选输出未实例化时，运行时下标会前移。按 IR 下标取实例起点。
struct MsaIrOutput {
    const gert::StorageShape* shape = nullptr;
    const gert::CompileTimeTensorDesc* desc = nullptr;
};

inline MsaIrOutput IrOutputAt(gert::TilingContext* context, size_t irIndex)
{
    MsaIrOutput out;
    const auto* node = context->GetComputeNodeInfo();
    if (node == nullptr || irIndex >= node->GetIrOutputsNum()) {
        return out;
    }
    const auto* inst = node->GetOutputInstanceInfo(irIndex);
    if (inst == nullptr || inst->GetInstanceNum() == 0U) {
        return out;
    }
    const size_t runtime = inst->GetInstanceStart();
    out.shape = context->GetOutputShape(runtime);
    out.desc = context->GetOutputDesc(runtime);
    return out;
}

// 分页逻辑块数是页表宽度；非分页没有页表，上界是 score 末维。
// score 末维是对齐后的输出宽度，分页时只作不超过已分配 score 的钳位。
inline uint32_t LogicalBlockLimit(const MsaIndexScoreInfo& info)
{
    if (MsaIsPaKeyLayout(info.keyLayout) && info.tableWidth > 0U && info.tableWidth < info.maxBlocksPerBatch) {
        return info.tableWidth;
    }
    return info.maxBlocksPerBatch;
}

inline uint32_t EstMaxVisibleBlocks(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    // 打分块数（pool_block_size 粒度）上界；klen host 可见时用真实可见 token 收紧。
    uint32_t vis = LogicalBlockLimit(info);
    auto klenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_KLEN);
    if (!IsHostVisibleTensor(klenTensor)) {
        return vis;
    }
    const int32_t* data = klenTensor->GetData<int32_t>();
    if (data == nullptr) {
        return vis;
    }
    uint32_t maxTok = 0U;
    // 顺序布局统一使用 [B+1] 前缀和；PA 使用 [B] 可见长度。
    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        for (uint32_t b = 0; b < info.batch; ++b) {
            const int32_t delta = data[b + 1U] - data[b];
            if (delta > 0 && static_cast<uint32_t>(delta) > maxTok) {
                maxTok = static_cast<uint32_t>(delta);
            }
        }
    } else {
        for (uint32_t b = 0; b < info.batch; ++b) {
            if (data[b] > 0 && static_cast<uint32_t>(data[b]) > maxTok) {
                maxTok = static_cast<uint32_t>(data[b]);
            }
        }
    }
    const uint32_t blkLen = info.poolBlockSize == 0U ? MSA_BLOCK_SIZE : info.poolBlockSize;
    const uint32_t fromLen = (maxTok + blkLen - 1U) / blkLen;
    if (fromLen < vis) {
        vis = fromLen;
    }
    return vis;
}

inline uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0U) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return (a == 0U) ? 1U : a;
}

// A2/A3：M-task 少于 AIC 时，kvChunks 取 aic/gcd(mTasks, aic)，让总 task 能铺满核。
// 仅当估算 makespan（每核 stile 数）下降才切开，避免短 KV 被握手开销拖慢。
inline uint32_t EstKvChunksA2(uint32_t aicNum, uint32_t packedM, uint32_t visBlocks)
{
    if (packedM == 0U || aicNum == 0U || packedM >= aicNum) {
        return 1U;
    }
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles <= 1U) {
        return 1U;
    }
    uint32_t chunks = aicNum / GcdU32(packedM, aicNum);
    if (chunks > nStiles) {
        chunks = nStiles;
    }
    if (chunks <= 1U) {
        return 1U;
    }
    const uint32_t oldSpan = ((packedM + aicNum - 1U) / aicNum) * nStiles;
    const uint32_t newTasks = packedM * chunks;
    const uint32_t tasksPerCore = (newTasks + aicNum - 1U) / aicNum;
    const uint32_t stilePerChunk = (nStiles + chunks - 1U) / chunks;
    if (tasksPerCore * stilePerChunk >= oldSpan) {
        return 1U;
    }
    return chunks;
}

// A2/A3：batch 已超过 AIC、但每个请求仍是一个 M-tile（短 decode）时，按最长 KV 的 stile
// 再切 S。否则 32 个长短不一的请求会按任务号跨步，最慢核拿到约 2 倍平均页数。
// 仅当估算 makespan 下降才切。prefill（单请求行数 > M-tile）和 950 不走这里。
inline uint32_t EstKvChunksA2BatchBalance(const MsaIndexScoreInfo& info, uint32_t aicNum, uint32_t visBlocks)
{
    if (info.batch <= aicNum || aicNum == 0U) {
        return 1U;
    }
    const uint64_t rows = static_cast<uint64_t>(info.totalQ) * static_cast<uint64_t>(info.numQHeads);
    if (rows > static_cast<uint64_t>(info.batch) * MSA_ROW_TILE_M) {
        return 1U;
    }
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles <= 1U) {
        return 1U;
    }
    const uint32_t oldSpan = ((info.batch + aicNum - 1U) / aicNum) * nStiles;
    uint32_t bestChunks = 1U;
    uint32_t bestSpan = oldSpan;
    for (uint32_t chunks = 2U; chunks <= nStiles && chunks <= 16U; chunks *= 2U) {
        const uint32_t newTasks = info.batch * chunks;
        const uint32_t tasksPerCore = (newTasks + aicNum - 1U) / aicNum;
        const uint32_t stilePerChunk = (nStiles + chunks - 1U) / chunks;
        const uint32_t span = tasksPerCore * stilePerChunk;
        // 与当前最优 span 相同时取更大 chunks：stile 估算打平，但请求长短不一时
        // 更细的 S 切分能把长请求拆开。span 不小于初始 makespan 时不切，避免空切。
        if (span < bestSpan || (span == bestSpan && bestChunks > 1U)) {
            bestSpan = span;
            bestChunks = chunks;
        }
    }
    return bestChunks;
}

// 每请求只有一个 M-tile 时，packed+B 上界会让 chunks 整除不了真实 batch。
// B=4、上界 packedM=5 时 chunks=4，实际只有 16 个 task，4 个 AIC 空转。
// 这里改用 batch 本身。makespan 不差过约 1/6 时取更大的 chunks，
// 避免长短不一的请求只被切成两三份、长请求堆在少数核上。
inline uint32_t ChunkSpan(uint32_t mTasks, uint32_t chunks, uint32_t aicNum, uint32_t nStiles)
{
    const uint32_t tasks = mTasks * chunks;
    const uint32_t tasksPerCore = (tasks + aicNum - 1U) / aicNum;
    const uint32_t stilePerChunk = (nStiles + chunks - 1U) / chunks;
    return tasksPerCore * stilePerChunk;
}

inline uint32_t EstKvChunksA2ByBatch(uint32_t aicNum, uint32_t mTasks, uint32_t nStiles)
{
    if (mTasks == 0U || aicNum == 0U || mTasks >= aicNum || nStiles <= 1U) {
        return 1U;
    }
    uint32_t base = aicNum / GcdU32(mTasks, aicNum);
    if (base > nStiles) {
        base = nStiles;
    }
    if (base <= 1U) {
        return 1U;
    }
    uint32_t bestChunks = base;
    uint32_t bestSpan = ChunkSpan(mTasks, base, aicNum, nStiles);
    const uint32_t spanLimit = bestSpan + ((bestSpan / 6U) > 1U ? (bestSpan / 6U) : 1U);
    for (uint32_t chunks = base + 1U; chunks <= nStiles && chunks <= aicNum; ++chunks) {
        const uint32_t tasks = mTasks * chunks;
        if (tasks / chunks != mTasks) {
            break;
        }
        if ((tasks % aicNum) != 0U) {
            continue;
        }
        const uint32_t span = ChunkSpan(mTasks, chunks, aicNum, nStiles);
        if (span < bestSpan || (span <= spanLimit && chunks > bestChunks)) {
            if (span < bestSpan) {
                bestSpan = span;
            }
            bestChunks = chunks;
        }
    }
    const uint32_t oldSpan = ((mTasks + aicNum - 1U) / aicNum) * nStiles;
    if (bestSpan >= oldSpan) {
        return 1U;
    }
    return bestChunks;
}

// 每请求只有一个 M-tile（短 decode）。packed+B 上界会让 chunks 对不齐真实 batch，950 上同样空核。
inline bool OneTilePerRequest(const MsaIndexScoreInfo& info, uint32_t aicNum)
{
    if (info.batch < 1U || info.batch >= aicNum) {
        return false;
    }
    const uint64_t rows = static_cast<uint64_t>(info.totalQ) * static_cast<uint64_t>(info.numQHeads);
    return (rows > 0U) && (rows <= static_cast<uint64_t>(info.batch) * MSA_ROW_TILE_M);
}

// 950：M-task 填不满 AIC 且可见 KV stile 足够时，沿 S 切开打满核。A1 类 packedM≈aicNum 时 kvChunks=1。
// 短 decode（每请求一个 M-tile）与 A2/A3 一样按真实 batch 取 chunks，避免 packed+B 上界留下空核。
// 其余 950 形状仍用 chunks=aic/packedM。A2/A3 分支不改。
inline uint32_t EstKvChunks(const MsaIndexScoreInfo& info, uint32_t aicNum, uint32_t packedM, uint32_t visBlocks)
{
    if (packedM == 0U || aicNum == 0U) {
        return 1U;
    }
    if (packedM >= aicNum) {
        if (!info.isAscend950) {
            return EstKvChunksA2BatchBalance(info, aicNum, visBlocks);
        }
        return 1U;
    }
    if (!info.isAscend950) {
        if (OneTilePerRequest(info, aicNum)) {
            const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
            return EstKvChunksA2ByBatch(aicNum, info.batch, nStiles);
        }
        return EstKvChunksA2(aicNum, packedM, visBlocks);
    }
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles <= 1U) {
        return 1U;
    }
    if (OneTilePerRequest(info, aicNum)) {
        return EstKvChunksA2ByBatch(aicNum, info.batch, nStiles);
    }
    uint32_t chunks = aicNum / packedM;
    if (chunks > nStiles) {
        chunks = nStiles;
    }
    if (chunks < 1U) {
        chunks = 1U;
    }
    return chunks;
}

// 950 块长 16：host 读不到逐条长度，packedM 只能按 packed+batch 估，再除 aicNum 会把核丢掉
// （8 条请求时 32/9=3，只派出 24 个任务）。改成把核数当预算下发，kernel 在 Init 里按真实
// M-task 数重算每个 M-tile 的 chunk 数。其余 SoC 与块长 128 返回 0，保持原路径。
inline uint32_t EstChunkBudget(const MsaIndexScoreInfo& info, uint32_t aicNum)
{
    if (!info.isAscend950 || info.poolBlockSize != MSA_BLOCK16_SIZE) {
        return 0U;
    }
    return aicNum;
}

// 预算路径下 kernel 的任务数由真实长度决定，host 只给一个不小于它的上界：
// 任务数 = Σ_b mTiles_b × min(chunks, nStiles_b) ≤ packedM × nStiles，再截到 aicNum。
inline uint32_t EstLaunchAicBudget(uint32_t aicNum, uint32_t packedM, uint32_t visBlocks)
{
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles == 0U || packedM == 0U) {
        return 1U;
    }
    const uint64_t est = static_cast<uint64_t>(packedM) * static_cast<uint64_t>(nStiles);
    if (est >= static_cast<uint64_t>(aicNum)) {
        return aicNum;
    }
    return static_cast<uint32_t>(est);
}

inline uint32_t EstLaunchAic(uint32_t aicNum, uint32_t packedM, uint32_t kvChunks)
{
    if (aicNum == 0U) {
        return 1U;
    }
    uint32_t est = packedM;
    if (kvChunks > 1U) {
        const uint32_t prod = packedM * kvChunks;
        est = (prod / kvChunks != packedM) ? aicNum : prod;
    }
    if (est < 1U) {
        est = 1U;
    }
    if (est > aicNum) {
        est = aicNum;
    }
    return est;
}

inline uint64_t GetDefaultStride0(const gert::Shape& shape)
{
    uint64_t stride = 1;
    for (size_t dim = 1; dim < shape.GetDimNum(); ++dim) {
        stride *= static_cast<uint64_t>(shape.GetDim(dim));
    }
    return stride;
}

// 逻辑 view：优先 OriginShape（aclCreateTensor 的 viewDims / torch sizes）。
// StorageShape 在 dim0 插空时可能是 [NP*gap, ...]，不能当 numPages。
inline const gert::Shape& KeyLogicalShape(const gert::StorageShape* keyShape, uint32_t keyLayout)
{
    const size_t expectRank = (keyLayout == MSA_KEY_LAYOUT_TND) ? MSA_KEY_TND_DIM_NUM : MSA_KEY_PA_DIM_NUM;
    const gert::Shape& origin = keyShape->GetOriginShape();
    if (origin.GetDimNum() == expectRank) {
        return origin;
    }
    return keyShape->GetStorageShape();
}

// aclnn TensorV2 优先 GetRequiredInputStride；
// 文档接口 GetInputStride 作为兼容回退。
inline const gert::Stride* GetKeyStrideDesc(gert::TilingContext* context)
{
    const gert::Stride* stride = context->GetRequiredInputStride(MSA_IDX_KEY);
    if (stride != nullptr) {
        return stride;
    }
    return context->GetInputStride(MSA_IDX_KEY);
}

// 对齐 MlaProlog GetCacheStride0：dim1…末维须等于紧凑 stride，仅 dim0 可非连续。
// size-1 轴不参与寻址：PyTorch / torch_npu 对其 stride 不做连续约束（BBND↔BNBD 且 N2=1
// 时 permute().contiguous() 仍可能是 dim1 stride=P 而非 N2*P*D），跳过该轴比较。
inline ge::graphStatus ValidateKeyInnerAxesContiguous(gert::TilingContext* context, const gert::Stride* stride,
                                                      const gert::Shape& shape)
{
    if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t expectedStride = 1U;
    for (int64_t dim = static_cast<int64_t>(shape.GetDimNum()) - 1; dim >= 1; --dim) {
        const int64_t dimSize = shape.GetDim(static_cast<size_t>(dim));
        OP_CHECK_IF(dimSize < 0, OP_LOGE(context, "key dim%ld size is negative.", dim), return ge::GRAPH_FAILED);
        if (dimSize > 1) {
            const uint64_t actualStride = static_cast<uint64_t>(stride->GetStride(static_cast<size_t>(dim)));
            OP_CHECK_IF(actualStride != expectedStride,
                        OP_LOGE(context,
                                "key dim%ld must be contiguous, actual stride is %lu, expected stride is %lu. "
                                "Only dim0 may be non-contiguous.",
                                dim, actualStride, expectedStride),
                        return ge::GRAPH_FAILED);
        }
        expectedStride *= static_cast<uint64_t>(dimSize);
    }
    return ge::GRAPH_SUCCESS;
}

// PA：strideKvBlock 取 key dim0 元素 stride。连续输入写入值与 shape 推算相同。
inline ge::graphStatus ResolveStrideKvBlock(gert::TilingContext* context, const MsaIndexScoreInfo& info,
                                            uint32_t defaultStrideKvBlock, uint32_t& strideKvBlock)
{
    strideKvBlock = defaultStrideKvBlock;

    const gert::StorageShape* keyShape = context->GetInputShape(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape& kc = KeyLogicalShape(keyShape, info.keyLayout);
    if (kc.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t defaultStride0 = GetDefaultStride0(kc);
    const gert::Stride* stride = GetKeyStrideDesc(context);
    if (stride == nullptr || stride->GetDimNum() != kc.GetDimNum()) {
        OP_LOGD(context->GetNodeName(), "key has no stride desc, treat as contiguous, strideKvBlock=%u.",
                strideKvBlock);
        return ge::GRAPH_SUCCESS;
    }
    if (ValidateKeyInnerAxesContiguous(context, stride, kc) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t actualStride0 = stride->GetStride(MSA_DIM_0);
    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        OP_CHECK_IF(actualStride0 < 0 || static_cast<uint64_t>(actualStride0) != defaultStride0,
                    OP_LOGE(context, "TND key dim0 must be contiguous, actual=%ld expected=%lu.", actualStride0,
                            defaultStride0),
                    return ge::GRAPH_FAILED);
        strideKvBlock = 0;
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(actualStride0 < 0 || static_cast<uint64_t>(actualStride0) < defaultStride0,
                OP_LOGE(context, "PA key dim0 stride must be >= %lu, got %ld.", defaultStride0, actualStride0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(actualStride0) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context, "PA key dim0 stride %ld exceeds uint32 strideKvBlock.", actualStride0),
                return ge::GRAPH_FAILED);
    strideKvBlock = static_cast<uint32_t>(actualStride0);
    OP_LOGI(context->GetNodeName(), "PA key strideKvBlock=%u (contiguous default=%lu).", strideKvBlock, defaultStride0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseLayoutKeyAttr(gert::TilingContext* context, const std::string& layoutKey, uint32_t& keyLayout)
{
    const std::string s = layoutKey.empty() ? "BBND" : layoutKey;
    if (s == "TND") {
        keyLayout = MSA_KEY_LAYOUT_TND;
    } else if (s == "BSND") {
        keyLayout = MSA_KEY_LAYOUT_BSND;
    } else if (s == "BNSD") {
        keyLayout = MSA_KEY_LAYOUT_BNSD;
    } else if (s == "BBND") {
        keyLayout = MSA_KEY_LAYOUT_BBND;
    } else if (s == "BNBD") {
        keyLayout = MSA_KEY_LAYOUT_BNBD;
    } else {
        OP_LOGE(context, "layout_key must be TND, BSND, BNSD, BBND or BNBD, got %s.", s.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseLayoutQueryAttr(gert::TilingContext* context, const char* layoutQuery, uint32_t& queryLayout)
{
    const char* s = (layoutQuery == nullptr || layoutQuery[0] == '\0') ? "TND" : layoutQuery;
    if (std::strcmp(s, "TND") == 0) {
        queryLayout = MSA_QUERY_LAYOUT_TND;
    } else if (std::strcmp(s, "BSND") == 0) {
        queryLayout = MSA_QUERY_LAYOUT_BSND;
    } else if (std::strcmp(s, "BNSD") == 0) {
        queryLayout = MSA_QUERY_LAYOUT_BNSD;
    } else {
        OP_LOGE(context, "layout_query must be TND, BSND or BNSD, got %s.", s);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// return_choice!=0 时没有用户 score。末维与 InferShape 相同：逻辑块数 16 对齐。
uint32_t DeriveAlignedMaxBlocks(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    uint32_t raw = 0;
    const uint32_t pool = (info.poolBlockSize == 0U) ? MSA_BLOCK_SIZE : info.poolBlockSize;
    if (MsaIsPaKeyLayout(info.keyLayout)) {
        raw = info.tableWidth;
    } else {
        const gert::Tensor* klenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_KLEN);
        const int32_t* klenData = nullptr;
        if (IsHostVisibleTensor(klenTensor)) {
            klenData = klenTensor->GetData<int32_t>();
        }
        if (klenData != nullptr && info.batch > 0U) {
            for (uint32_t b = 0; b < info.batch; ++b) {
                int32_t kv = klenData[b + 1U] - klenData[b];
                if (kv < 0) {
                    kv = 0;
                }
                if (info.kMaxSeq > 0U && static_cast<uint32_t>(kv) > info.kMaxSeq) {
                    kv = static_cast<int32_t>(info.kMaxSeq);
                }
                const uint32_t blocks = (kv <= 0) ? 0U : (static_cast<uint32_t>(kv) + pool - 1U) / pool;
                if (blocks > raw) {
                    raw = blocks;
                }
            }
        } else if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
            raw = (pool == 0U) ? 0U : (info.totalK + pool - 1U) / pool;
        } else {
            raw = (pool == 0U) ? 0U : (info.kMaxSeq + pool - 1U) / pool;
        }
    }
    if (raw == 0U) {
        raw = 1U;
    }
    return RoundUpU32(raw, MSA_SCORE_STRIDE_ALIGN);
}

ge::graphStatus ParseAndCheck(gert::TilingContext* context, MsaIndexScoreInfo& info)
{
    info.platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(info.platformInfo == nullptr, OP_LOGE(context, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatformEarly = platform_ascendc::PlatformAscendC(info.platformInfo);
    info.isAscend950 = (ascendcPlatformEarly.GetCurNpuArch() == NpuArch::DAV_3510);

    const gert::StorageShape* queryShape = context->GetInputShape(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::StorageShape* keyShape = context->GetInputShape(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::StorageShape* blockTableShape = context->GetOptionalInputShape(MSA_IDX_BLOCK_TABLE);
    const gert::StorageShape* actualSeqQlenShape = context->GetOptionalInputShape(MSA_IDX_ACTUAL_SEQ_QLEN);
    OP_CHECK_IF(!IsOptionalProvided(actualSeqQlenShape), OP_LOGE(context, "TND query requires actual_seq_qlen."),
                return ge::GRAPH_FAILED);
    const gert::StorageShape* actualSeqKlenShape = context->GetOptionalInputShape(MSA_IDX_ACTUAL_SEQ_KLEN);
    OP_CHECK_IF(!IsOptionalProvided(actualSeqKlenShape), OP_LOGE(context, "actual_seq_klen is required."),
                return ge::GRAPH_FAILED);
    // start_loc 是必选输入，排在 query、key 之后。
    const gert::StorageShape* startLocShape = context->GetRequiredInputShape(MSA_IDX_START_LOC);
    OP_CHECK_NULL_WITH_CONTEXT(context, startLocShape);

    const gert::Shape& q = queryShape->GetStorageShape();
    const gert::Shape& qlen = actualSeqQlenShape->GetStorageShape();
    const gert::Shape& klen = actualSeqKlenShape->GetStorageShape();

    const auto* keyDesc = context->GetInputDesc(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    info.keyDtype = keyDesc->GetDataType();
    OP_CHECK_IF(static_cast<ge::Format>(ge::GetPrimaryFormat(keyDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context, "FRACTAL_NZ key is not supported."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(qlen.GetDimNum() != MSA_RANK_1D || qlen.GetDim(MSA_DIM_0) < MSA_QLEN_MIN_DIM0,
                OP_LOGE(context, "actual_seq_qlen must be 1D [B+1]."), return ge::GRAPH_FAILED);
    info.batch = static_cast<uint32_t>(qlen.GetDim(MSA_DIM_0) - 1);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* layoutKeyPtr = attrs->GetStr(MSA_ATTR_LAYOUT_KEY);
    const std::string layoutKey = (layoutKeyPtr == nullptr) ? "" : layoutKeyPtr;
    if (ParseLayoutKeyAttr(context, layoutKey, info.keyLayout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseLayoutQueryAttr(context, attrs->GetStr(MSA_ATTR_LAYOUT_QUERY), info.queryLayout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // pool_block_size：缺省 128。只允许 16 / 128。顺序布局的 klen 统一为 [B+1] 前缀和。
    // V2（query 与 key 同布局）的 BSND/BNSD 固定 16，仅 950。
    // 非 V2（layout_query=TND、pool=128）允许 key 为 BSND/BNSD，A2/A3 与 950 都支持。
    const int64_t* poolBlockSizeAttrPtr = attrs->GetInt(MSA_ATTR_POOL_BLOCK_SIZE);
    const uint32_t poolBlockSizeAttr = (poolBlockSizeAttrPtr == nullptr || *poolBlockSizeAttrPtr <= 0) ?
                                           MSA_BLOCK_SIZE :
                                           static_cast<uint32_t>(*poolBlockSizeAttrPtr);
    OP_CHECK_IF(poolBlockSizeAttr != MSA_BLOCK_SIZE && poolBlockSizeAttr != MSA_BLOCK16_SIZE,
                OP_LOGE(context, "pool_block_size must be %u or %u, got %u.", MSA_BLOCK_SIZE, MSA_BLOCK16_SIZE,
                        poolBlockSizeAttr),
                return ge::GRAPH_FAILED);
    const int64_t* returnChoicePtr = attrs->GetInt(MSA_ATTR_RETURN_CHOICE);
    const int64_t returnChoice = (returnChoicePtr == nullptr) ? 0 : *returnChoicePtr;
    OP_CHECK_IF(returnChoice != static_cast<int64_t>(MSA_RETURN_SCORE) &&
                    returnChoice != static_cast<int64_t>(MSA_RETURN_INDICES) &&
                    returnChoice != static_cast<int64_t>(MSA_RETURN_INDICES_VALUES),
                OP_LOGE(context, "return_choice must be 0, 1 or 2, got %ld.", returnChoice), return ge::GRAPH_FAILED);
    info.returnChoice = static_cast<uint32_t>(returnChoice);
    if (info.returnChoice == MSA_RETURN_SCORE) {
        info.topk = 0;
    } else {
        OP_CHECK_IF(!info.isAscend950,
                    OP_LOGE(context, "return_choice=%ld is only supported on Ascend 950.", returnChoice),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(poolBlockSizeAttr == MSA_BLOCK16_SIZE,
                    OP_LOGE(context, "return_choice=%ld does not support pool_block_size=16.", returnChoice),
                    return ge::GRAPH_FAILED);
        const int64_t* topkPtr = attrs->GetInt(MSA_ATTR_TOPK);
        const int64_t topkVal = (topkPtr == nullptr) ? static_cast<int64_t>(MSA_TOPK_DEFAULT) : *topkPtr;
        OP_CHECK_IF(topkVal < 1 || topkVal > static_cast<int64_t>(MSA_TOPK_MAX),
                    OP_LOGE(context, "topk must be in [1, 128], got %ld.", topkVal), return ge::GRAPH_FAILED);
        info.topk = static_cast<uint32_t>(topkVal);
    }
    // compress_factor 只做接口预留。1 为不压缩，计算路径与原先一致。
    const int64_t* compressFactorPtr = attrs->GetInt(MSA_ATTR_COMPRESS_FACTOR);
    const int64_t compressFactor = (compressFactorPtr == nullptr) ? 1 : *compressFactorPtr;
    OP_CHECK_IF(compressFactor != 1, OP_LOGE(context, "compress_factor only supports 1, got %ld.", compressFactor),
                return ge::GRAPH_FAILED);
    const bool batchKey128 = (info.keyLayout == MSA_KEY_LAYOUT_BSND || info.keyLayout == MSA_KEY_LAYOUT_BNSD) &&
                             poolBlockSizeAttr == MSA_BLOCK_SIZE;
    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        if (batchKey128) {
            OP_CHECK_IF(info.queryLayout != MSA_QUERY_LAYOUT_TND,
                        OP_LOGE(context, "pool_block_size=%u BSND/BNSD requires layout_query=TND.", MSA_BLOCK_SIZE),
                        return ge::GRAPH_FAILED);
        } else {
            const uint32_t queryOfKey = (info.keyLayout == MSA_KEY_LAYOUT_BSND) ? MSA_QUERY_LAYOUT_BSND :
                                        (info.keyLayout == MSA_KEY_LAYOUT_BNSD) ? MSA_QUERY_LAYOUT_BNSD :
                                                                                  MSA_QUERY_LAYOUT_TND;
            OP_CHECK_IF(info.queryLayout != queryOfKey,
                        OP_LOGE(context, "non-paged key layout must match layout_query."), return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(info.queryLayout > MSA_QUERY_LAYOUT_BNSD, OP_LOGE(context, "layout_query is invalid."),
                    return ge::GRAPH_FAILED);
    }
    if (info.queryLayout == MSA_QUERY_LAYOUT_TND) {
        OP_CHECK_IF(q.GetDimNum() != MSA_QUERY_DIM_NUM,
                    OP_LOGE(context, "layout_query=TND requires query rank 3 [T1,N1,D], got %zu.", q.GetDimNum()),
                    return ge::GRAPH_FAILED);
        info.totalQ = static_cast<uint32_t>(q.GetDim(MSA_DIM_0));
        info.numQHeads = static_cast<uint32_t>(q.GetDim(MSA_DIM_1));
        info.headDim = static_cast<uint32_t>(q.GetDim(MSA_DIM_2));
        info.qMaxSeq = 0;
    } else {
        OP_CHECK_IF(q.GetDimNum() != MSA_KEY_PA_DIM_NUM,
                    OP_LOGE(context, "layout_query=BSND/BNSD requires query rank 4, got %zu.", q.GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<uint32_t>(q.GetDim(MSA_DIM_0)) != info.batch,
            OP_LOGE(context, "query batch(%ld) must equal actual_seq_qlen batch(%u).", q.GetDim(MSA_DIM_0), info.batch),
            return ge::GRAPH_FAILED);
        if (info.queryLayout == MSA_QUERY_LAYOUT_BSND) {
            info.qMaxSeq = static_cast<uint32_t>(q.GetDim(MSA_DIM_1));
            info.numQHeads = static_cast<uint32_t>(q.GetDim(MSA_DIM_2));
        } else {
            info.numQHeads = static_cast<uint32_t>(q.GetDim(MSA_DIM_1));
            info.qMaxSeq = static_cast<uint32_t>(q.GetDim(MSA_DIM_2));
        }
        info.headDim = static_cast<uint32_t>(q.GetDim(MSA_DIM_3));
        info.totalQ = info.batch * info.qMaxSeq;
        // GetData() 对 device tensor 返回非空指针，host 解引用会段错误。只在 kOnHost/kFollowing 时读前缀。
        bool qPrefixApplied = false;
        const gert::Tensor* qlenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_QLEN);
        if (IsHostVisibleTensor(qlenTensor)) {
            const int32_t* qlenData = qlenTensor->GetData<int32_t>();
            if (qlenData != nullptr) {
                info.totalQ = static_cast<uint32_t>(qlenData[info.batch]);
                qPrefixApplied = true;
                for (uint32_t b = 0; b < info.batch; ++b) {
                    const int32_t qlenB = qlenData[b + 1] - qlenData[b];
                    OP_CHECK_IF(qlenB < 0 || static_cast<uint32_t>(qlenB) > info.qMaxSeq,
                                OP_LOGE(context, "query length %d exceeds layout S=%u.", qlenB, info.qMaxSeq),
                                return ge::GRAPH_FAILED);
                }
            }
        }
        // score 的 T1 决定 head 步长。return_choice!=0 时没有 score，totalQ 保持前缀和或 B*S。
        if (info.returnChoice != MSA_RETURN_SCORE) {
            // 非 TND query 的 TopK 仍用上面算好的 totalQ。
        } else {
            const gert::StorageShape* scoreShapeEarly = context->GetOutputShape(0);
            OP_CHECK_NULL_WITH_CONTEXT(context, scoreShapeEarly);
            const gert::Shape& scoreEarly = scoreShapeEarly->GetStorageShape();
            OP_CHECK_IF(scoreEarly.GetDimNum() != MSA_QUERY_DIM_NUM,
                        OP_LOGE(context, "score must be 3D [N1,T1,stride]."), return ge::GRAPH_FAILED);
            // pool_block_size=16 的 head 间 max 把 score 收到 1 个 head；块长 128 仍是每个 query head 一份。
            const uint32_t expectScoreHeads = (poolBlockSizeAttr == MSA_BLOCK16_SIZE) ? 1U : info.numQHeads;
            OP_CHECK_IF(
                static_cast<uint32_t>(scoreEarly.GetDim(MSA_DIM_0)) != expectScoreHeads,
                OP_LOGE(context, "score head dim(%ld) must be %u.", scoreEarly.GetDim(MSA_DIM_0), expectScoreHeads),
                return ge::GRAPH_FAILED);
            const int64_t scoreTokens = scoreEarly.GetDim(MSA_DIM_1);
            OP_CHECK_IF(scoreTokens < 0, OP_LOGE(context, "score T1 must be concrete, got %ld.", scoreTokens),
                        return ge::GRAPH_FAILED);
            if (qPrefixApplied) {
                OP_CHECK_IF(
                    static_cast<uint32_t>(scoreTokens) != info.totalQ,
                    OP_LOGE(context, "score T1(%ld) must equal actual_seq_qlen prefix(%u).", scoreTokens, info.totalQ),
                    return ge::GRAPH_FAILED);
            } else {
                const uint32_t cap = info.batch * info.qMaxSeq;
                OP_CHECK_IF(static_cast<uint32_t>(scoreTokens) > cap,
                            OP_LOGE(context, "score T1(%ld) exceeds batch*S(%u).", scoreTokens, cap),
                            return ge::GRAPH_FAILED);
                info.totalQ = static_cast<uint32_t>(scoreTokens);
            }
        }
    }
    // 块长 16 的 BSND/BNSD（及 query BSND/BNSD）仅 950。非 V2 的 pool=128 key BSND/BNSD 不在此列。
    const bool m3Layout =
        info.queryLayout == MSA_QUERY_LAYOUT_BSND || info.queryLayout == MSA_QUERY_LAYOUT_BNSD ||
        ((info.keyLayout == MSA_KEY_LAYOUT_BSND || info.keyLayout == MSA_KEY_LAYOUT_BNSD) && !batchKey128);
    OP_CHECK_IF(m3Layout && !info.isAscend950, OP_LOGE(context, "BSND/BNSD is only supported on Ascend 950."),
                return ge::GRAPH_FAILED);

    const gert::Shape& kc = KeyLogicalShape(keyShape, info.keyLayout);

    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        // 块长来自 pool_block_size。TND 与非 V2 的 BSND/BNSD 为 128；V2 的 BSND/BNSD 为 16。
        info.poolBlockSize = poolBlockSizeAttr;
        info.pageSize = poolBlockSizeAttr;
        info.numPages = 0;
        info.kMaxSeq = 0;
        OP_CHECK_IF(IsOptionalProvided(blockTableShape), OP_LOGE(context, "non-paged key must not pass block_table."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(klen.GetDimNum() != MSA_RANK_1D || static_cast<uint32_t>(klen.GetDim(MSA_DIM_0)) != info.batch + 1,
                    OP_LOGE(context, "non-paged actual_seq_klen must be [B+1] prefix-sum."), return ge::GRAPH_FAILED);
        if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
            OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_TND_DIM_NUM,
                        OP_LOGE(context, "layout_key=TND requires key rank 3 [T2,N2,D], got %zu.", kc.GetDimNum()),
                        return ge::GRAPH_FAILED);
            info.totalK = static_cast<uint32_t>(kc.GetDim(MSA_DIM_0));
            info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_2)) != info.headDim,
                        OP_LOGE(context, "TND key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_2),
                                info.headDim),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_PA_DIM_NUM,
                        OP_LOGE(context, "layout_key=BSND/BNSD requires key rank 4, got %zu.", kc.GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                static_cast<uint32_t>(kc.GetDim(MSA_DIM_0)) != info.batch,
                OP_LOGE(context, "key batch(%ld) must equal query batch(%u).", kc.GetDim(MSA_DIM_0), info.batch),
                return ge::GRAPH_FAILED);
            if (info.keyLayout == MSA_KEY_LAYOUT_BSND) {
                info.kMaxSeq = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
                info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            } else {
                info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
                info.kMaxSeq = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            }
            info.totalK = info.batch * info.kMaxSeq;
            const gert::Tensor* klenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_KLEN);
            if (IsHostVisibleTensor(klenTensor)) {
                const int32_t* klenData = klenTensor->GetData<int32_t>();
                if (klenData != nullptr) {
                    for (uint32_t b = 0; b < info.batch; ++b) {
                        const int32_t kvB = klenData[b + 1U] - klenData[b];
                        OP_CHECK_IF(kvB < 0 || static_cast<uint32_t>(kvB) > info.kMaxSeq,
                                    OP_LOGE(context, "key length %d exceeds layout S=%u.", kvB, info.kMaxSeq),
                                    return ge::GRAPH_FAILED);
                    }
                }
            }
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_3)) != info.headDim,
                        OP_LOGE(context, "BSND/BNSD key headDim(%ld) must equal query headDim(%u).",
                                kc.GetDim(MSA_DIM_3), info.headDim),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(!IsOptionalProvided(blockTableShape),
                    OP_LOGE(context, "PageAttention (BBND/BNBD) requires block_table."), return ge::GRAPH_FAILED);
        const gert::Shape& bt = blockTableShape->GetStorageShape();
        OP_CHECK_IF(bt.GetDimNum() != MSA_BLOCK_TABLE_DIM_NUM,
                    OP_LOGE(context, "block_table must have 2 dims, got %zu.", bt.GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(bt.GetDim(MSA_DIM_0)) != info.batch,
                    OP_LOGE(context, "block_table batch(%ld) must equal actual_seq_qlen batch(%u).",
                            bt.GetDim(MSA_DIM_0), info.batch),
                    return ge::GRAPH_FAILED);
        // 页表按存储页粒度：宽度 = 存储页数，与打分块数（pool_block_size attr）解耦。
        info.tableWidth = static_cast<uint32_t>(bt.GetDim(MSA_DIM_1));
        OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_PA_DIM_NUM,
                    OP_LOGE(context, "layout_key=%s requires key rank 4, got %zu.",
                            info.keyLayout == MSA_KEY_LAYOUT_BNBD ? "BNBD" : "BBND", kc.GetDimNum()),
                    return ge::GRAPH_FAILED);
        info.numPages = static_cast<uint32_t>(kc.GetDim(MSA_DIM_0));
        // 存储页长 = key 的页维度；打分块长 = pool_block_size attr。两者解耦：打分块必须放得进存储页。
        uint32_t pageSize = 0;
        if (info.keyLayout == MSA_KEY_LAYOUT_BBND) {
            pageSize = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
            info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_3)) != info.headDim,
                        OP_LOGE(context, "BBND key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_3),
                                info.headDim),
                        return ge::GRAPH_FAILED);
        } else {
            info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
            pageSize = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_3)) != info.headDim,
                        OP_LOGE(context, "BNBD key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_3),
                                info.headDim),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(pageSize != MSA_BLOCK_SIZE && pageSize != MSA_BLOCK16_SIZE,
                    OP_LOGE(context, "BBND/BNBD page size must be %u or %u, got %u.", MSA_BLOCK_SIZE, MSA_BLOCK16_SIZE,
                            pageSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(poolBlockSizeAttr > pageSize,
                    OP_LOGE(context, "pool_block_size(%u) must not exceed BBND/BNBD page size(%u).", poolBlockSizeAttr,
                            pageSize),
                    return ge::GRAPH_FAILED);
        info.poolBlockSize = poolBlockSizeAttr;
        info.pageSize = pageSize;
        OP_CHECK_IF(klen.GetDimNum() != MSA_RANK_1D || static_cast<uint32_t>(klen.GetDim(MSA_DIM_0)) != info.batch,
                    OP_LOGE(context, "PA actual_seq_klen must be [B] visible S2."), return ge::GRAPH_FAILED);
    }

    // score 末维 = 打分块数（16 对齐）。return_choice=0 时取自用户 score；否则按输入推导，与 InferShape 一致。
    const MsaIrOutput scoreOutIr = IrOutputAt(context, 0);
    const MsaIrOutput indicesOutIr = IrOutputAt(context, 1);
    const MsaIrOutput valuesOutIr = IrOutputAt(context, 2);
    const gert::StorageShape* scoreShape = scoreOutIr.shape;
    const gert::StorageShape* indicesShape = indicesOutIr.shape;
    const gert::StorageShape* valuesShape = valuesOutIr.shape;
    const bool hasScore = IsOptionalProvided(scoreShape);
    const bool hasIndices = IsOptionalProvided(indicesShape);
    const bool hasValues = IsOptionalProvided(valuesShape);
    if (info.returnChoice == MSA_RETURN_SCORE) {
        OP_CHECK_IF(!hasScore, OP_LOGE(context, "outputs do not match return_choice=%u.", info.returnChoice),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hasIndices || hasValues,
                    OP_LOGE(context, "outputs do not match return_choice=%u.", info.returnChoice),
                    return ge::GRAPH_FAILED);
        const gert::Shape& scOut = scoreShape->GetStorageShape();
        OP_CHECK_IF(scOut.GetDimNum() != MSA_QUERY_DIM_NUM, OP_LOGE(context, "score must be 3D [N1,T1,stride]."),
                    return ge::GRAPH_FAILED);
        info.maxBlocksPerBatch = static_cast<uint32_t>(scOut.GetDim(MSA_DIM_2));
        OP_CHECK_IF(info.maxBlocksPerBatch == 0, OP_LOGE(context, "score last dim must be positive."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((info.maxBlocksPerBatch % MSA_SCORE_STRIDE_ALIGN) != 0,
                    OP_LOGE(context, "score last dim(%u) must be a multiple of %u.", info.maxBlocksPerBatch,
                            MSA_SCORE_STRIDE_ALIGN),
                    return ge::GRAPH_FAILED);
        OP_CHECK_NULL_WITH_CONTEXT(context, scoreOutIr.desc);
        OP_CHECK_IF(scoreOutIr.desc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context, "fp16 score is reserved and not supported."), return ge::GRAPH_FAILED);
    } else {
        const bool outputsOk = !hasScore && hasIndices &&
                               ((info.returnChoice == MSA_RETURN_INDICES && !hasValues) ||
                                (info.returnChoice == MSA_RETURN_INDICES_VALUES && hasValues));
        if (!outputsOk) {
            OP_LOGE(context, "outputs do not match return_choice=%u.", info.returnChoice);
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& idxOut = indicesShape->GetStorageShape();
        const auto* idxDesc = indicesOutIr.desc;
        OP_CHECK_NULL_WITH_CONTEXT(context, idxDesc);
        const bool idxOk =
            idxOut.GetDimNum() == MSA_QUERY_DIM_NUM && idxDesc->GetDataType() == ge::DT_INT32 &&
            static_cast<ge::Format>(ge::GetPrimaryFormat(idxDesc->GetStorageFormat())) == ge::FORMAT_ND &&
            static_cast<uint32_t>(idxOut.GetDim(MSA_DIM_0)) == info.numQHeads &&
            static_cast<uint32_t>(idxOut.GetDim(MSA_DIM_1)) == info.totalQ &&
            static_cast<uint32_t>(idxOut.GetDim(MSA_DIM_2)) == info.topk;
        if (!idxOk) {
            const int dtype = (idxDesc == nullptr) ? -1 : static_cast<int>(idxDesc->GetDataType());
            const int fmt =
                (idxDesc == nullptr) ? -1 : static_cast<int>(ge::GetPrimaryFormat(idxDesc->GetStorageFormat()));
            OP_LOGE(context,
                    "sparse_indices must be [N1, T1, topk] int32 ND, got rank=%zu dims=[%ld,%ld,%ld] dtype=%d "
                    "format=%d heads=%u totalQ=%u topk=%u.",
                    idxOut.GetDimNum(), idxOut.GetDimNum() > 0 ? idxOut.GetDim(0) : -1,
                    idxOut.GetDimNum() > 1 ? idxOut.GetDim(1) : -1, idxOut.GetDimNum() > 2 ? idxOut.GetDim(2) : -1,
                    dtype, fmt, info.numQHeads, info.totalQ, info.topk);
            return ge::GRAPH_FAILED;
        }
        if (info.returnChoice == MSA_RETURN_INDICES_VALUES) {
            const gert::Shape& valOut = valuesShape->GetStorageShape();
            const auto* valDesc = valuesOutIr.desc;
            OP_CHECK_NULL_WITH_CONTEXT(context, valDesc);
            const bool valOk = valDesc->GetDataType() == ge::DT_FLOAT && valOut.GetDimNum() == idxOut.GetDimNum() &&
                               valOut.GetDim(MSA_DIM_0) == idxOut.GetDim(MSA_DIM_0) &&
                               valOut.GetDim(MSA_DIM_1) == idxOut.GetDim(MSA_DIM_1) &&
                               valOut.GetDim(MSA_DIM_2) == idxOut.GetDim(MSA_DIM_2);
            OP_CHECK_IF(!valOk,
                        OP_LOGE(context, "sparse_values must match sparse_indices and be float32. fp16 values are "
                                         "reserved and not supported."),
                        return ge::GRAPH_FAILED);
        }
        info.maxBlocksPerBatch = DeriveAlignedMaxBlocks(context, info);
    }
    // 块长 16：TND 16、BSND/BNSD，以及 BBND/BNBD 页长 16，仅 Ascend 950。
    // 取值白名单与 BSND/BNSD 组合已在 attr 解析处校验，PA 页长已与 attr 交叉校验。
    const bool logicalBlock16 = info.poolBlockSize == MSA_BLOCK16_SIZE;
    OP_CHECK_IF(logicalBlock16 && !info.isAscend950,
                OP_LOGE(context, "pool_block_size=%u is only supported on Ascend 950.", MSA_BLOCK16_SIZE),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(logicalBlock16 && (info.numQHeads == 0 || (MSA_ROW_TILE_M % info.numQHeads) != 0),
                OP_LOGE(context, "pool_block_size=%u requires numQHeads(%u) to divide %u.", MSA_BLOCK16_SIZE,
                        info.numQHeads, MSA_ROW_TILE_M),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        info.headDim == 0 || (info.headDim % MSA_SCORE_STRIDE_ALIGN) != 0,
        OP_LOGE(context, "headDim(%u) must be a positive multiple of %u.", info.headDim, MSA_SCORE_STRIDE_ALIGN),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.headDim > MSA_K_TILE,
                OP_LOGE(context, "headDim(%u) larger than %u is not supported yet.", info.headDim, MSA_K_TILE),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.numKvHeads == 0 || (info.numQHeads % info.numKvHeads) != 0,
                OP_LOGE(context, "numQHeads(%u) must be divisible by numKvHeads(%u).", info.numQHeads, info.numKvHeads),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.batch == 0, OP_LOGE(context, "batch must be positive."), return ge::GRAPH_FAILED);
    // q_len/kv_len 全 0 跳过计算。InferShape 已把空 KV 的 score 末维钳到 16。
    OP_CHECK_IF(info.maxBlocksPerBatch == 0,
                OP_LOGE(context, "maxBlocksPerBatch must be positive (empty kv uses aligned score stride)."),
                return ge::GRAPH_FAILED);
    // 仅 PA 有 page 池；TND / BSND 直接按 token 寻址，numPages 恒为 0。
    OP_CHECK_IF((info.keyLayout == MSA_KEY_LAYOUT_BBND || info.keyLayout == MSA_KEY_LAYOUT_BNBD) && info.numPages == 0,
                OP_LOGE(context, "PA numPages must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint32_t>(startLocShape->GetStorageShape().GetDim(MSA_DIM_0)) != info.batch,
                OP_LOGE(context, "start_loc size must equal batch."), return ge::GRAPH_FAILED);

    const auto* queryDesc = context->GetInputDesc(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    info.queryDtype = queryDesc->GetDataType();
    OP_CHECK_IF(!IsNonQuantQueryDtype(info.queryDtype, info.isAscend950),
                OP_LOGE(context, "query dtype must be bf16 or fp16 on A2/A3; Ascend 950 also allows hifloat8 / "
                                 "float8_e5m2 / float8_e4m3fn. int8 query and float4 are reserved."),
                return ge::GRAPH_FAILED);

    info.isQuant = (info.keyDtype == ge::DT_INT8);
    // 预留int8场景，当前拦截
    OP_CHECK_IF(info.queryDtype == ge::DT_FLOAT16 && info.keyDtype == ge::DT_INT8,
                OP_LOGE(context, "float16 query with int8 key is not supported yet, got query %d key %d.",
                        static_cast<int32_t>(info.queryDtype), static_cast<int32_t>(info.keyDtype)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.isAscend950 && info.queryDtype == ge::DT_HIFLOAT8 && info.keyDtype == ge::DT_HIFLOAT8,
                OP_LOGE(context, "hifloat8 query with hifloat8 key is not supported yet, got query %d key %d.",
                        static_cast<int32_t>(info.queryDtype), static_cast<int32_t>(info.keyDtype)),
                return ge::GRAPH_FAILED);
    // 打分块长 16（TND 16、BSND/BNSD、页长 16 的两级 maxpool）仅 fp16 / bf16；int8 与 FP8 固定 128。
    OP_CHECK_IF(info.poolBlockSize == MSA_BLOCK16_SIZE &&
                    (info.isQuant || IsFp8ComputeDtype(info.queryDtype) || IsFp8ComputeDtype(info.keyDtype)),
                OP_LOGE(context, "pool_block_size=%u supports fp16 / bf16 only.", MSA_BLOCK16_SIZE),
                return ge::GRAPH_FAILED);
    // PA 存储页长 16（M3 短页存储）同样仅 950 上的 fp16 / bf16。
    OP_CHECK_IF(info.pageSize == MSA_BLOCK16_SIZE &&
                    (info.isQuant || IsFp8ComputeDtype(info.queryDtype) || IsFp8ComputeDtype(info.keyDtype)),
                OP_LOGE(context, "BBND/BNBD page size %u supports fp16 / bf16 only.", MSA_BLOCK16_SIZE),
                return ge::GRAPH_FAILED);
    if (info.isQuant) {
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT16, OP_LOGE(context, "ND int8 key currently requires fp16 query."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(info.keyDtype != info.queryDtype, OP_LOGE(context, "non-quant key dtype must match query dtype."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(IsFp8ComputeDtype(info.queryDtype) && !info.isAscend950,
                    OP_LOGE(context, "FP8 query/key is only supported on Ascend 950."), return ge::GRAPH_FAILED);
    }

    const gert::StorageShape* scaleShape = context->GetOptionalInputShape(MSA_IDX_KEY_SCALE);
    if (info.isQuant) {
        OP_CHECK_IF(!IsOptionalProvided(scaleShape), OP_LOGE(context, "int8 key requires dequant scale."),
                    return ge::GRAPH_FAILED);
        const gert::Shape& sc = scaleShape->GetStorageShape();
        if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
            const bool ok2d = (sc.GetDimNum() == MSA_SCALE_TND_DIM_NUM &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) == info.totalK &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_1)) == info.numKvHeads);
            const bool ok1d = (sc.GetDimNum() == MSA_RANK_1D && info.numKvHeads == 1 &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) == info.totalK);
            OP_CHECK_IF(!ok2d && !ok1d,
                        OP_LOGE(context, "TND dequant scale must be [%u, %u] or [%u] (N2=1).", info.totalK,
                                info.numKvHeads, info.totalK),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(sc.GetDimNum() != MSA_SCALE_PA_DIM_NUM,
                        OP_LOGE(context, "PA dequant scale must be 3D [NP, N_kv, P], got rank %zu.", sc.GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) != info.numPages ||
                            static_cast<uint32_t>(sc.GetDim(MSA_DIM_1)) != info.numKvHeads ||
                            static_cast<uint32_t>(sc.GetDim(MSA_DIM_2)) != info.pageSize,
                        OP_LOGE(context, "dequant scale shape must be [%u, %u, %u], got [%ld, %ld, %ld].",
                                info.numPages, info.numKvHeads, info.pageSize, sc.GetDim(MSA_DIM_0),
                                sc.GetDim(MSA_DIM_1), sc.GetDim(MSA_DIM_2)),
                        return ge::GRAPH_FAILED);
        }
        const auto* scaleDesc = context->GetOptionalInputDesc(MSA_IDX_KEY_SCALE);
        OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
        OP_CHECK_IF(scaleDesc->GetDataType() != ge::DT_FLOAT, OP_LOGE(context, "dequant scale dtype must be float32."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(IsOptionalProvided(scaleShape), OP_LOGE(context, "non-quant path must not pass key_scale."),
                    return ge::GRAPH_FAILED);
    }
    const gert::StorageShape* queryScaleShape = context->GetOptionalInputShape(MSA_IDX_QUERY_SCALE);
    OP_CHECK_IF(IsOptionalProvided(queryScaleShape), OP_LOGE(context, "query_scale is reserved and not supported."),
                return ge::GRAPH_FAILED);

    const int64_t* sparseModeAttr = attrs->GetInt(MSA_ATTR_SPARSE_MODE);
    info.sparseMode = (sparseModeAttr == nullptr) ? MSA_SPARSE_MODE_RIGHT_DOWN : static_cast<uint32_t>(*sparseModeAttr);
    OP_CHECK_IF(info.sparseMode != MSA_SPARSE_MODE_DEFAULT && info.sparseMode != MSA_SPARSE_MODE_RIGHT_DOWN,
                OP_LOGE(context, "sparse_mode must be %u or %u, got %u.", MSA_SPARSE_MODE_DEFAULT,
                        MSA_SPARSE_MODE_RIGHT_DOWN, info.sparseMode),
                return ge::GRAPH_FAILED);

    const gert::StorageShape* attenMaskShape = context->GetOptionalInputShape(MSA_IDX_ATTEN_MASK);
    if (info.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN) {
        OP_CHECK_IF(!IsOptionalProvided(attenMaskShape),
                    OP_LOGE(context, "sparse_mode=%u requires atten_mask of shape [%u, %u].",
                            MSA_SPARSE_MODE_RIGHT_DOWN, MSA_ATTEN_MASK_SIZE, MSA_ATTEN_MASK_SIZE),
                    return ge::GRAPH_FAILED);
        const gert::Shape& am = attenMaskShape->GetStorageShape();
        OP_CHECK_IF(am.GetDimNum() != MSA_ATTEN_MASK_DIM_NUM ||
                        static_cast<uint32_t>(am.GetDim(MSA_DIM_0)) != MSA_ATTEN_MASK_SIZE ||
                        static_cast<uint32_t>(am.GetDim(MSA_DIM_1)) != MSA_ATTEN_MASK_SIZE,
                    OP_LOGE(context, "atten_mask must be [%u, %u], got rank=%zu dims=[%ld,%ld].", MSA_ATTEN_MASK_SIZE,
                            MSA_ATTEN_MASK_SIZE, am.GetDimNum(), am.GetDimNum() > 0 ? am.GetDim(MSA_DIM_0) : -1,
                            am.GetDimNum() > 1 ? am.GetDim(MSA_DIM_1) : -1),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(IsOptionalProvided(attenMaskShape),
                    OP_LOGE(context, "sparse_mode=%u must not pass atten_mask.", MSA_SPARSE_MODE_DEFAULT),
                    return ge::GRAPH_FAILED);
    }

    const int64_t* initBlocksAttr = attrs->GetInt(MSA_ATTR_INIT_BLOCKS);
    const int64_t* localBlocksAttr = attrs->GetInt(MSA_ATTR_LOCAL_BLOCKS);
    info.initBlocks = (initBlocksAttr == nullptr) ? MSA_DEFAULT_INIT_BLOCKS : static_cast<uint32_t>(*initBlocksAttr);
    info.localBlocks =
        (localBlocksAttr == nullptr) ? MSA_DEFAULT_LOCAL_BLOCKS : static_cast<uint32_t>(*localBlocksAttr);
    const uint32_t logicalBlocks = LogicalBlockLimit(info);
    OP_CHECK_IF(info.initBlocks > logicalBlocks || info.localBlocks > logicalBlocks,
                OP_LOGE(context, "init_blocks/local_blocks must be <= logical blocks(%u).", logicalBlocks),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DoTiling(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(info.platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0, OP_LOGE(context, "aic core num is 0."), return ge::GRAPH_FAILED);

    const uint32_t scoreBlockStride = RoundUpU32(info.maxBlocksPerBatch, MSA_SCORE_STRIDE_ALIGN);
    // 950 C_to_UB：S 不进 GM；int8/TND 尾页仍要一页 K scratch。A2 与 950 回退（MSA_A5_USE_C2UB=0）走 8-stage S。
    const uint32_t sElemBytes = info.isQuant ? sizeof(float) : sizeof(uint16_t);
    const bool c2ub = info.isAscend950 && (MSA_A5_USE_C2UB != 0);
    const uint32_t sWsBytes = c2ub ? 0U : (aicNum * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM * sElemBytes);
    // BSND 的尾窗可能越过 key 末尾（仅最后一个 batch），与 TND 尾页一样走 scratch 补零。
    const bool paBlock16 = (info.keyLayout == MSA_KEY_LAYOUT_BBND || info.keyLayout == MSA_KEY_LAYOUT_BNBD) &&
                           info.poolBlockSize == MSA_BLOCK16_SIZE;
    // 块长 16 的 8 个 page 不连续，必须先收到 scratch 再做 128 列 GEMM。
    const bool useKScratch = info.isQuant || MsaIsSeqKeyLayout(info.keyLayout) || paBlock16;
    uint32_t scratchElemBytes = sizeof(uint16_t);
    if (info.isAscend950 && !info.isQuant && IsFp8ComputeDtype(info.queryDtype)) {
        scratchElemBytes = 1U;
    }
    uint32_t kScratchElems = c2ub ? MSA_A5_K_SCRATCH_ELEM_NUM : MSA_K_SCRATCH_ELEM_NUM;
    // A2/A3 int8 四槽 K scratch；950 C2UB / 非量化 TND 尾页保持单槽。
    // 块长 16 的 C2UB 用两槽：Cube 读当前窗时 Vector 预取下一窗。
    if (!info.isAscend950 && info.isQuant) {
        kScratchElems *= MSA_K_SCRATCH_STAGES_A2;
    } else if (c2ub && info.poolBlockSize == MSA_BLOCK16_SIZE) {
        kScratchElems *= MSA_A5_S_STAGES;
    }
    const uint32_t kScratchBytes = useKScratch ? (aicNum * kScratchElems * scratchElemBytes) : 0U;
    const uint32_t kScratchOffsetElems = sWsBytes / sizeof(float);
    // BSND/BNSD 的 M-tile 先收成连续 (token, head) 再给 Cube，不能直接按 TND 步长读。
    const uint32_t qScratchBytes =
        (info.queryLayout != MSA_QUERY_LAYOUT_TND) ? (aicNum * MSA_ROW_TILE_M * info.headDim * scratchElemBytes) : 0U;
    const uint32_t qScratchElemOff = (qScratchBytes == 0U) ? 0U : ((sWsBytes + kScratchBytes) / scratchElemBytes);

    uint32_t launchAic = aicNum;
    uint32_t kvChunks = 1U;
    uint32_t chunkBudget = 0U;
    if (info.totalQ == 0U) {
        launchAic = 1U;
    } else {
        const uint32_t packedM = EstPackedMTasks(info, aicNum);
        const uint32_t visBlocks = EstMaxVisibleBlocks(context, info);
        kvChunks = EstKvChunks(info, aicNum, packedM, visBlocks);
        chunkBudget = EstChunkBudget(info, aicNum);
        launchAic = (chunkBudget > 0U) ? EstLaunchAicBudget(aicNum, packedM, visBlocks) :
                                         EstLaunchAic(aicNum, packedM, kvChunks);
    }

    MsaIndexScoreTilingData tilingData;
    tilingData.set_batch(info.batch);
    tilingData.set_totalQ(info.totalQ);
    tilingData.set_numQHeads(info.numQHeads);
    tilingData.set_numKvHeads(info.numKvHeads);
    tilingData.set_gqaGroup(info.numQHeads / info.numKvHeads);
    tilingData.set_headDim(info.headDim);
    tilingData.set_poolBlockSize(info.poolBlockSize);
    tilingData.set_pageSize(info.pageSize);
    tilingData.set_tableWidth(info.tableWidth);
    tilingData.set_maxBlocksPerBatch(info.maxBlocksPerBatch);
    tilingData.set_scoreBlockStride(scoreBlockStride);
    tilingData.set_usedCoreNum(launchAic);
    tilingData.set_isQuant(info.isQuant ? 1U : 0U);
    tilingData.set_sparseMode(info.sparseMode);
    tilingData.set_initBlocks(info.initBlocks);
    tilingData.set_localBlocks(info.localBlocks);
    tilingData.set_numPages(info.numPages);
    tilingData.set_keyLayout(info.keyLayout);
    tilingData.set_totalK(info.totalK);
    tilingData.set_kvChunks(kvChunks);
    tilingData.set_chunkBudget(chunkBudget);
    tilingData.set_queryLayout(info.queryLayout);
    tilingData.set_qMaxSeq(info.qMaxSeq);
    tilingData.set_kMaxSeq(info.kMaxSeq);
    tilingData.set_qScratchElems(qScratchElemOff);

    tilingData.set_strideQt(info.numQHeads * info.headDim);
    tilingData.set_strideQn(info.headDim);
    uint32_t defaultStrideKvBlock = 0;
    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        defaultStrideKvBlock = 0;
        // BNSD 的 head0 沿 S 连续，token 步长是 D；BSND/TND 的 head0 步长是 N2*D。
        const uint32_t seqTokenStride =
            (info.keyLayout == MSA_KEY_LAYOUT_BNSD) ? info.headDim : (info.numKvHeads * info.headDim);
        tilingData.set_strideKvToken(seqTokenStride);
        tilingData.set_strideScalePage(info.numKvHeads);
        tilingData.set_strideScaleHead(1);
    } else if (info.keyLayout == MSA_KEY_LAYOUT_BNBD) {
        // PA 的页内步长按存储页长（key 的页维度）算，与打分块长解耦。
        defaultStrideKvBlock = info.numKvHeads * info.pageSize * info.headDim;
        tilingData.set_strideKvToken(info.headDim);
        tilingData.set_strideScalePage(info.numKvHeads * info.pageSize);
        tilingData.set_strideScaleHead(info.pageSize);
    } else {
        defaultStrideKvBlock = info.pageSize * info.numKvHeads * info.headDim;
        tilingData.set_strideKvToken(info.numKvHeads * info.headDim);
        tilingData.set_strideScalePage(info.numKvHeads * info.pageSize);
        tilingData.set_strideScaleHead(info.pageSize);
    }
    uint32_t strideKvBlock = defaultStrideKvBlock;
    if (ResolveStrideKvBlock(context, info, defaultStrideKvBlock, strideKvBlock) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingData.set_strideKvBlock(strideKvBlock);
    tilingData.set_strideOutHead(info.totalQ * scoreBlockStride);
    tilingData.set_strideOutToken(scoreBlockStride);
    tilingData.set_kScratchOffsetElems(kScratchOffsetElems);

    size_t userBytes =
        static_cast<size_t>(sWsBytes) + static_cast<size_t>(kScratchBytes) + static_cast<size_t>(qScratchBytes);
    if (info.isAscend950 && sWsBytes == 0U && kScratchBytes == 0U) {
        userBytes += 64U;
    }
    uint32_t scoreWorkspaceElems = 0;
    size_t scoreBytes = 0;
    if (info.returnChoice != MSA_RETURN_SCORE) {
        const size_t aligned = (userBytes + 255U) & ~static_cast<size_t>(255U);
        scoreWorkspaceElems = static_cast<uint32_t>(aligned / sizeof(float));
        scoreBytes = static_cast<size_t>(info.numQHeads) * static_cast<size_t>(info.totalQ) *
                     static_cast<size_t>(scoreBlockStride) * sizeof(float);
        userBytes = aligned + scoreBytes;
    }
    tilingData.set_returnChoice(info.returnChoice);
    tilingData.set_topk(info.topk);
    tilingData.set_scoreWorkspaceElems(scoreWorkspaceElems);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // MIX 1AIC:2AIV：CalcTschBlockDim 的 sliceNum 按 AIV 计数，内部再 / (aiv/aic)。
    // 传入 aicNum 会再除一次得到 blockDim=aic/2，只能打一半 Cube。
    // 整 batch q_len=0：queryS==0 → BlockDim=1，避免 totalTaskNum=0。
    // sliceNum = launchAic * 2：短 decode 只起实际 M-task（及 950 的 S-chunk）对应的 MIX；
    // 多 M-tile / 大 batch 仍打满 AIC。
    if (info.totalQ == 0U) {
        context->SetBlockDim(1);
    } else {
        uint32_t sliceAiv = launchAic * MSA_AIV_PER_AIC;
        if (sliceAiv > aivNum) {
            sliceAiv = aivNum;
        }
        context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(sliceAiv, aicNum, aivNum));
    }

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + userBytes;

    uint64_t tilingKey = MSA_TILING_KEY_FP16;
    if (info.poolBlockSize == MSA_BLOCK16_SIZE) {
        // 块长 16（TND 16 / BSND/BNSD / 页长 16，两级 maxpool）独占 TilingKey，128 路径保持原样。
        tilingKey = (info.queryDtype == ge::DT_BF16) ? MSA_TILING_KEY_BF16_BLK16 : MSA_TILING_KEY_FP16_BLK16;
    } else if (info.isQuant) {
        tilingKey = MSA_TILING_KEY_FP16_INT8;
    } else if (info.queryDtype == ge::DT_HIFLOAT8) {
        tilingKey = MSA_TILING_KEY_HIFLOAT8;
    } else if (info.queryDtype == ge::DT_FLOAT8_E5M2) {
        tilingKey = MSA_TILING_KEY_FP8_E5M2;
    } else if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
        tilingKey = MSA_TILING_KEY_FP8_E4M3FN;
    } else {
        tilingKey = (info.queryDtype == ge::DT_BF16) ? MSA_TILING_KEY_BF16 : MSA_TILING_KEY_FP16;
    }
    if (info.returnChoice != MSA_RETURN_SCORE) {
        if (info.queryDtype == ge::DT_BF16) {
            tilingKey = MSA_TILING_KEY_BF16_TOPK;
        } else if (info.queryDtype == ge::DT_FLOAT16) {
            tilingKey = MSA_TILING_KEY_FP16_TOPK;
        } else if (info.queryDtype == ge::DT_FLOAT8_E5M2) {
            tilingKey = MSA_TILING_KEY_FP8_E5M2_TOPK;
        } else if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
            tilingKey = MSA_TILING_KEY_FP8_E4M3FN_TOPK;
        } else {
            OP_LOGE(context, "return_choice=%u does not support this dtype.", info.returnChoice);
            return ge::GRAPH_FAILED;
        }
    }
    context->SetTilingKey(tilingKey);
    OP_LOGI(context->GetNodeName(),
            "MsaIndexScore tilingKey=%lu headDim=%u layout=%u strideKvBlock=%u launchAic=%u kvChunks=%u "
            "chunkBudget=%u aicNum=%u",
            tilingKey, info.headDim, info.keyLayout, strideKvBlock, launchAic, kvChunks, chunkBudget, aicNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingPrepareForMsaIndexScore(gert::TilingParseContext* /* context */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForMsaIndexScore(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("MsaIndexScore", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    MsaIndexScoreInfo info;
    if (ParseAndCheck(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return DoTiling(context, info);
}

IMPL_OP_OPTILING(MsaIndexScore)
    .Tiling(TilingForMsaIndexScore)
    .TilingParse<MsaIndexScoreCompileInfo>(TilingPrepareForMsaIndexScore);

} // namespace optiling
