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
 * \file quant_lightning_indexer_v2_fast_service_vector_arch35.h
 * \brief Vector侧权重准备、加权归约及TopK流水调度
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_VECTOR_ARCH35_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_VECTOR_ARCH35_H

#include "quant_lightning_indexer_v2_fast_common_arch35.h"
#include "vf/quant_lightning_indexer_v2_fast_topk.h"
#include "vf/quant_lightning_indexer_v2_fast_vector1.h"
#include "vf/quant_lightning_indexer_v2_vector1.h"

namespace QLIV2FastVectorFp8 {
using namespace AscendC;
using namespace QLIV2Fast;
using TopK = QLIV2FastTopK::PairTopK<false>;
constexpr int32_t ROWS_PER_AIV = 8;
// 上一对Query的TopK调度状态，在当前对归约期间保持不变
// queryPairIndex为本AIV的两行Query编号，-1表示无待处理行对；scoreBufferIndex为双缓冲槽编号
// keyTileStart/validKeyCount描述该K块的逻辑起始位置及有效Key数，isLastKeyTile标识任务末块
struct PendingQueryPair {
    int32_t queryPairIndex = -1;
    int32_t scoreBufferIndex = 0;
    int64_t keyTileStart = 0;
    int32_t validKeyCount = 0;
    bool isLastKeyTile = false;
};
// FP8先合并weight与Query scale，后续每个K分组复用；尾行跳过搬运，补齐头由LoadHeadRow置零
__aicore__ inline void PrepareWeights(LocalTensor<float> effective, LocalTensor<float> scratch,
                                      GlobalTensor<float> weights, GlobalTensor<float> queryScales, uint32_t sub,
                                      uint32_t validQueries, uint32_t paddedHeadNum, uint32_t heads)
{
    for (int32_t r = 0; r < ROWS_PER_AIV; ++r) {
        if (TopK::GlobalRow(sub, r) >= validQueries) {
            continue;
        }
        LoadHeadRow(effective[r * paddedHeadNum], weights[TopK::GlobalRow(sub, r) * heads], heads, paddedHeadNum);
        LoadHeadRow(scratch[r * paddedHeadNum], queryScales[TopK::GlobalRow(sub, r) * heads], heads, paddedHeadNum);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
    QLIV2FastVector1::PrepareWeightsVF(reinterpret_cast<__ubuf__ float*>(effective.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float*>(scratch.GetPhyAddr()), paddedHeadNum);
}
// 等待Cube交付当前槽：对两行各256个Key执行ReLU、按头加权求和及Key scale缩放，再归还槽
template <int32_t Slot>
__aicore__ inline void ReduceGroup(LocalTensor<float> effective, TopK& topk, int32_t queryPairIndex,
                                   int32_t scoreBufferIndex, int32_t keyGroupIndex, uint32_t paddedHeadNum,
                                   uint32_t heads)
{
    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_V>(static_cast<uint16_t>(Slot));
    LocalTensor<float> qk(TPosition::VECCALC, Slot * CUBE_RESULT_SLOT_OFFSET_BYTES, CUBE_RESULT_VIEW_ELEMENTS);
    auto weight = effective[queryPairIndex * 2 * paddedHeadNum];
    auto output = topk.Score(scoreBufferIndex, 0, keyGroupIndex * 256);
    auto keyScale = topk.scales[keyGroupIndex * 256];
    QLIV2FastVector1::WeightedReduceFp8(output, qk, weight, keyScale, KEY_TILE_COLUMNS, paddedHeadNum, heads);
    CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_V>(static_cast<uint16_t>(Slot));
}
// 普通分组共用归约与上一对TopK推进；第3/7组需在两者之间预取scale，仍由调用方显式编排
template <int32_t Group>
__aicore__ inline void ReduceGroupAndAdvance(LocalTensor<float> effective, TopK& topk, int32_t queryPairIndex,
                                             int32_t scoreBufferIndex, uint32_t paddedHeadNum, uint32_t heads,
                                             const PendingQueryPair& pending)
{
    ReduceGroup<Group % 2>(effective, topk, queryPairIndex, scoreBufferIndex, Group, paddedHeadNum, heads);
    if (pending.queryPairIndex >= 0) {
        topk.template AdvancePair<Group>(pending.queryPairIndex * 2, pending.scoreBufferIndex, pending.keyTileStart,
                                         pending.validKeyCount, pending.isLastKeyTile);
    }
}
template <int32_t Half, bool PAGE>
__aicore__ inline void PrefetchScales(TopK& topk, const QLIV2Fast::KeyView<PAGE>& keyScales, int64_t keyTileStart,
                                      int64_t s2)
{
    // 2048列Key tile的scale分为两个1024列半区，Half取0或1
    const int64_t start = keyTileStart + KEY_TILE_COLUMNS + Half * 1024;
    if (start >= s2) {
        return;
    }
    const int32_t validKeyCount = (s2 - start) < 1024 ? s2 - start : 1024;
    // 最后一对Query已读完该半区，可在此预取下一K块的scale；另一半区仍供当前加权归约使用
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    keyScales.CopyFp8Scale(topk.scales[Half * 1024], start, validKeyCount);
    // 前半区就绪沿用事件6，后半区就绪使用事件0，与消费者等待事件一一对应
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(Half == 0 ? 6 : 0));
}
} // namespace QLIV2FastVectorFp8

namespace QLIV2FastVectorMx {
using namespace AscendC;
using namespace QLIV2Fast;
using TopK = QLIV2FastTopK::PairTopK<true>;
constexpr int32_t ROWS_PER_AIV = 8;
// 每对Query预留128个float（512字节），两份权重相隔64个float（256字节）
constexpr int64_t WEIGHT_COPY_OFFSET_ELEMENTS = 64U;
constexpr int32_t WEIGHT_PAIR_STRIDE_ELEMENTS = 128;
constexpr uint32_t WEIGHT_PAIR_STRIDE_BYTES = 512U;
// MX反量化已在Cube侧完成；这里只搬运weight，并保留两份副本供不同结果槽读取
__aicore__ inline void PrepareWeights(LocalTensor<float> effective, GlobalTensor<float> weights, uint32_t sub,
                                      uint32_t validQueries, uint32_t paddedHeadNum, uint32_t heads)
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    // 尾Q或头数补齐时逐行搬运；完整对齐块使用下方跨行搬运路径
    if (validQueries < 16U || heads != paddedHeadNum) {
        for (uint32_t queryPairIndex = 0U; queryPairIndex < 4U; ++queryPairIndex) {
            for (uint32_t row = 0U; row < 2U; ++row) {
                const uint32_t globalQueryRow = queryPairIndex * 4U + sub * 2U + row;
                if (globalQueryRow >= validQueries) {
                    continue;
                }
                QLIV2Fast::LoadHeadRow(effective[queryPairIndex * WEIGHT_PAIR_STRIDE_ELEMENTS + row * paddedHeadNum],
                                       weights[globalQueryRow * heads], heads, paddedHeadNum);
                QLIV2Fast::LoadHeadRow(effective[queryPairIndex * WEIGHT_PAIR_STRIDE_ELEMENTS +
                                                 WEIGHT_COPY_OFFSET_ELEMENTS + row * paddedHeadNum],
                                       weights[globalQueryRow * heads], heads, paddedHeadNum);
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
        return;
    }
    auto global = weights[sub * 2 * paddedHeadNum];
    auto local = effective;
    // GM步长为字节间隔，UB步长为32字节块数间隔
    // 4个Query对，每对两行float共8*头数字节；源跳过另一AIV的两行
    // UB每对预留512字节，目的间隔按32字节块计；最后一项为保留字段0
    DataCopyExtParams copy{4, 8 * paddedHeadNum, 8 * paddedHeadNum,
                           (WEIGHT_PAIR_STRIDE_BYTES - 8 * paddedHeadNum) / DATA_BLOCK_BYTES, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(local, global, copy, pad);
    DataCopyPad(local[WEIGHT_COPY_OFFSET_ELEMENTS], global, copy, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
}
// 消费Cube结果并执行ReLU及按头加权求和；MX无需再次乘scale，读完后归还Cube结果槽
template <int32_t Slot>
__aicore__ inline void ReduceGroup(LocalTensor<float> effective, TopK& topk, int32_t queryPairIndex,
                                   int32_t scoreBufferIndex, int32_t keyGroupIndex, uint32_t paddedHeadNum,
                                   uint32_t heads)
{
    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_V>(static_cast<uint16_t>(Slot));
    LocalTensor<float> qk(TPosition::VECCALC, Slot * CUBE_RESULT_SLOT_OFFSET_BYTES, CUBE_RESULT_VIEW_ELEMENTS);
    auto weight = effective[queryPairIndex * WEIGHT_PAIR_STRIDE_ELEMENTS + (1 - Slot) * WEIGHT_COPY_OFFSET_ELEMENTS];
    auto output = topk.Score(scoreBufferIndex, 0, keyGroupIndex * 256);
    QLIV2FastVector1::WeightedReduceMx(output, qk, weight, KEY_TILE_COLUMNS, paddedHeadNum, heads);
    CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_V>(static_cast<uint16_t>(Slot));
}
// 编译期确定分组及结果槽，在当前归约结束后利用间隙推进上一对Query的合并
template <int32_t Group>
__aicore__ inline void ReduceGroupAndAdvance(LocalTensor<float> effective, TopK& topk, int32_t queryPairIndex,
                                             int32_t scoreBufferIndex, uint32_t paddedHeadNum, uint32_t heads,
                                             bool hasPendingUpdate)
{
    ReduceGroup<Group % 2>(effective, topk, queryPairIndex, scoreBufferIndex, Group, paddedHeadNum, heads);
    if (hasPendingUpdate) {
        // AdvanceMerge检查待合并行掩码，无待合并任务时直接返回
        topk.AdvanceMerge();
    }
}
// 复用指定行的候选和阈值前，持续推进合并，直到这些行的待合并状态全部清除
__aicore__ inline void DrainMerges(TopK& topk, uint32_t rowMask)
{
    while ((topk.pendingMergeRowMask & rowMask) != 0U) {
        topk.AdvanceMerge();
    }
}
} // namespace QLIV2FastVectorMx
namespace QLIV2FastVector {
using namespace AscendC;
using namespace QLIV2Fast;
// 两种量化共用结果收尾，保持GM写入到重读之间的同步依赖
template <class TopK>
__aicore__ inline void WriteResults(TopK& topk, GlobalTensor<uint16_t> stateKeys, GlobalTensor<int32_t> stateIndices,
                                    GlobalTensor<uint16_t> globalValues, GM_ADDR indices, GM_ADDR values,
                                    uint32_t requested, uint32_t validQueries, int64_t tileBase, uint32_t sub,
                                    bool returnValues, bool useIndexWorkspace)
{
    // WaitStateKeys()仅将缓冲区使用权交还V；最终输出需从GM重读状态，因此读取前还须建立MTE3到MTE2的同步依赖
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
    // 暂存路径按实际Q行数和请求TopK写出；直写路径的索引已就位，仅在需要时补写value
    if (useIndexWorkspace) {
        QLIV2FastTopK::Finalize(topk, stateKeys, stateIndices, indices, values, requested, validQueries, tileBase,
                                returnValues);
    } else if (returnValues) {
        for (uint32_t row = 0; row < 8U; ++row) {
            const uint32_t globalQueryRow = TopK::GlobalRow(sub, row);
            QLIV2Fast::Load(topk.mergeKeys, stateKeys[globalQueryRow * TOPK_STATE_CAPACITY], TOPK_STATE_CAPACITY);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            vector1::UIntToFloatReturnValueWithInfMaskVF(
                reinterpret_cast<__ubuf__ bfloat16_t*>(topk.intermediateKeyResults.GetPhyAddr()),
                // 内部2048个状态元素，每次VF重复转换128个uint16，共16次
                reinterpret_cast<__ubuf__ uint16_t*>(topk.mergeKeys.GetPhyAddr()), 16U,
                QLIV2Fast::BF16_NEGATIVE_INF_BITS);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
            QLIV2Fast::Store(globalValues[(tileBase + globalQueryRow) * TOPK_STATE_CAPACITY],
                             topk.intermediateKeyResults, TOPK_STATE_CAPACITY);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        }
    }
}

// 两种量化共用物理K尾块和逐行mask屏蔽，屏障顺序与原分支一致
template <class TopK>
__aicore__ inline void MaskScoreTail(TopK& topk, int32_t scoreBufferIndex, int32_t queryPairIndex, int64_t keyTileStart,
                                     int32_t validKeyCount, int64_t keyLengthOffset, int64_t tileBase, uint32_t sub)
{
    if (validKeyCount < KEY_TILE_COLUMNS) {
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0U; r < 2U; ++r) {
            QLIV2FastVector1::FillAlignedRangeVF<uint16_t>(
                reinterpret_cast<__ubuf__ uint16_t*>(topk.Score(scoreBufferIndex, r).GetPhyAddr()),
                static_cast<uint16_t>(0), static_cast<uint32_t>(validKeyCount), KEY_TILE_COLUMNS);
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t r = 0; r < 2U; ++r) {
        const int64_t rowValid =
            keyLengthOffset + tileBase + TopK::GlobalRow(sub, queryPairIndex * 2 + r) + 1 - keyTileStart;
        if (rowValid < validKeyCount) {
            PipeBarrier<PIPE_V>();
            QLIV2FastVector1::FillAlignedRangeVF<uint16_t>(
                reinterpret_cast<__ubuf__ uint16_t*>(topk.Score(scoreBufferIndex, r).GetPhyAddr()),
                static_cast<uint16_t>(0), static_cast<uint32_t>(rowValid > 0 ? rowValid : 0),
                static_cast<uint32_t>(validKeyCount));
            PipeBarrier<PIPE_V>();
        }
    }
}

template <bool IS_MX, bool PAGE>
__aicore__ inline void RunVector(GM_ADDR weights, GM_ADDR queryScales, GM_ADDR keyScales, GM_ADDR scores,
                                 GM_ADDR indices, GM_ADDR values, uint32_t s1, int64_t keyLengthOffset,
                                 int64_t tileBase, bool returnValues, uint32_t requested, uint32_t taskK,
                                 const QLIV2Fast::KeyView<PAGE>& keyView, uint32_t paddedHeadNum, uint32_t heads)
{
    using TopK = QLIV2FastTopK::PairTopK<IS_MX>;
    constexpr int32_t ROWS_PER_AIV = 8;
    // 初始化两个Cube结果槽的空闲令牌，并绑定本任务的TopK状态与候选工作区
    const uint32_t sub = GetSubBlockIdx();
    CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_V>(static_cast<uint16_t>(0));
    CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_V>(static_cast<uint16_t>(1));
    GlobalTensor<uint16_t> stateKeys, candidateIndices;
    GlobalTensor<int32_t> stateIndices;
    stateKeys.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scores));
    const uint32_t validQueries =
        static_cast<uint32_t>((s1 - tileBase) < QUERY_TILE_ROWS ? s1 - tileBase : QUERY_TILE_ROWS);
    // 内部状态固定为16x2048 ；输出TopK不足2048或Q有尾块时使用暂存区，避免越界
    const bool useIndexWorkspace = (requested != TOPK_STATE_CAPACITY) || (validQueries != QUERY_TILE_ROWS);
    stateIndices.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>((useIndexWorkspace ? scores + STATE_INDEX_OFFSET : indices)));
    if (!useIndexWorkspace) {
        stateIndices = stateIndices[tileBase * TOPK_STATE_CAPACITY];
    }
    auto candidateKeys = stateKeys[STATE_KEY_BYTES / sizeof(uint16_t)];
    candidateIndices.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>((scores + CANDIDATE_INDEX_OFFSET)));
    GlobalTensor<float> globalWeights;
    GlobalTensor<uint16_t> globalValues;
    globalWeights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    globalValues.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(values));
    TopK topk(stateKeys, stateIndices, candidateKeys, candidateIndices, sub,
              returnValues || requested < TOPK_STATE_CAPACITY);

    if constexpr (IS_MX) {
        using namespace QLIV2FastVectorMx;

        // MX无需在Vector侧处理scale；复用该暂存区前2 KiB存放四对权重，每对权重分别复制到两个相差256字节的存储体相位位置
        static_assert((ROWS_PER_AIV / 2) * WEIGHT_PAIR_STRIDE_BYTES <= KEY_TILE_COLUMNS * sizeof(float));
        auto effective = topk.scales;
        PrepareWeights(effective, globalWeights[tileBase * heads], sub, validQueries, paddedHeadNum, heads);
        topk.PrepareIdentity();
        // 索引初始化仅写索引及分段边界暂存区，首次加权归约读取权重副本前再等待两次权重搬运完成
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);

        // pending记录上一对Query的分数；计算当前对的同时推进上一对TopK，首对尚无待处理数据
        // 每个AIV负责8行即4个Query对；行掩码3（二进制11）表示一对相邻行
        int32_t pendingQueryPairIndex = -1, pendingScoreBufferIndex = 0, pendingValidKeyCount = 0;
        int64_t pendingKeyTileStart = 0;
        bool pendingIsLastKeyTile = false;
        uint32_t keyTileStart;
        int32_t validKeyCount, queryPairIndex, scoreBufferIndex;
        bool isLastKeyTile;
        // 逐个2048列K块处理；每个AIV的8行拆成4对，每对由8次256列归约填满分数缓冲区
        for (keyTileStart = 0; keyTileStart < taskK; keyTileStart += KEY_TILE_COLUMNS) {
            validKeyCount = taskK - keyTileStart < KEY_TILE_COLUMNS ? taskK - keyTileStart : KEY_TILE_COLUMNS;
            isLastKeyTile = keyTileStart + validKeyCount == taskK;
            for (queryPairIndex = 0; queryPairIndex < ROWS_PER_AIV / 2; ++queryPairIndex) {
                // pending在本对归约期间保持不变；动态合并状态由TopK内部判断
                const bool hasPendingQueryPair = pendingQueryPairIndex >= 0;
                const bool hasPendingUpdate = hasPendingQueryPair && (pendingKeyTileStart != 0);
                // 两组score缓冲区交替使用，BeforeScores保证复用前相关搬运已完成
                scoreBufferIndex = queryPairIndex & 1;
                topk.BeforeScores(scoreBufferIndex);
                ReduceGroup<0>(effective, topk, queryPairIndex, scoreBufferIndex, 0, paddedHeadNum, heads);
                // 首K块直接建立TopK初始状态；后续块先完成同一行的旧合并，再按当前阈值筛选候选
                if (hasPendingQueryPair) {
                    if (pendingKeyTileStart == 0) {
                        topk.InitializePair(pendingQueryPairIndex * 2, pendingScoreBufferIndex);
                    } else {
                        topk.WaitCandidates();
                        DrainMerges(topk, 3U << (pendingQueryPairIndex * 2));
                        topk.PackPair(pendingQueryPairIndex * 2, pendingScoreBufferIndex, pendingValidKeyCount);
                    }
                }
                ReduceGroup<1>(effective, topk, queryPairIndex, scoreBufferIndex, 1, paddedHeadNum, heads);
                // 候选提交GM并登记合并任务，后续归约之间的空隙用于推进合并
                if (hasPendingUpdate) {
                    topk.CommitPair(pendingQueryPairIndex * 2, pendingScoreBufferIndex, pendingKeyTileStart);
                    topk.EnqueuePair(pendingQueryPairIndex * 2, pendingIsLastKeyTile, pendingKeyTileStart);
                }
                ReduceGroupAndAdvance<2>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         hasPendingUpdate);
                // 在归约间隙分步推进合并，同一行状态复用前再通过DrainMerges完成必要等待
                ReduceGroupAndAdvance<3>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         hasPendingUpdate);
                ReduceGroupAndAdvance<4>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         hasPendingUpdate);
                ReduceGroupAndAdvance<5>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         hasPendingUpdate);
                ReduceGroupAndAdvance<6>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         hasPendingUpdate);
                ReduceGroup<1>(effective, topk, queryPairIndex, scoreBufferIndex, 7, paddedHeadNum, heads);
                if (hasPendingQueryPair && !hasPendingUpdate) {
                    topk.FinishInitialPair();
                } else if (hasPendingUpdate) {
                    topk.AdvanceMerge();
                }
                MaskScoreTail(topk, scoreBufferIndex, queryPairIndex, keyTileStart, validKeyCount, keyLengthOffset,
                              tileBase, sub);
                // 当前两行分数已完整，尝试提前筛选首行，并留到下一对归约时继续处理
                topk.PrepareFirstRow(queryPairIndex * 2, scoreBufferIndex, keyTileStart, validKeyCount);
                pendingQueryPairIndex = queryPairIndex;
                pendingScoreBufferIndex = scoreBufferIndex;
                pendingKeyTileStart = keyTileStart;
                pendingValidKeyCount = validKeyCount;
                pendingIsLastKeyTile = isLastKeyTile;
            }
        }
        // 流水排空：最后一对没有下一对归约可交错，先清理同一行旧合并，再提交最后的候选
        topk.WaitCandidates();
        DrainMerges(topk, 3U << (pendingQueryPairIndex * 2));
        topk.PackPair(pendingQueryPairIndex * 2, pendingScoreBufferIndex, pendingValidKeyCount);
        topk.CommitPair(pendingQueryPairIndex * 2, pendingScoreBufferIndex, pendingKeyTileStart);
        topk.EnqueuePair(pendingQueryPairIndex * 2, pendingIsLastKeyTile, pendingKeyTileStart);
        DrainMerges(topk, ~0U);
        // 所有合并结束后，等待状态搬运完成，再根据returnValues及暂存策略生成最终输出
        topk.WaitCandidates();
        topk.WaitStateKeys();
        topk.WaitStateIndices();
        WriteResults(topk, stateKeys, stateIndices, globalValues, indices, values, requested, validQueries, tileBase,
                     sub, returnValues, useIndexWorkspace);
        return;
    } else {
        using namespace QLIV2FastVectorFp8;

        // 视图上限为8行乘32头，即256个float；实际按补齐后的头数访问
        LocalTensor<float> effective(TPosition::VECCALC, EFFECTIVE_WEIGHT_UB_OFFSET,
                                     QUERY_TILE_ROWS / AIVS_PER_CUBE * MAX_HEAD_NUM);
        GlobalTensor<float> globalQueryScale;
        globalQueryScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queryScales));
        // FP8在UB预合并Query scale和weight，Key scale则按K块独立搬运
        PrepareWeights(effective, topk.scales, globalWeights[tileBase * heads], globalQueryScale[tileBase * heads], sub,
                       validQueries, paddedHeadNum, heads);
        topk.PrepareIdentity();

        // pending记录上一对Query的分数；计算当前对的同时推进上一对TopK，首对尚无待处理数据
        PendingQueryPair pending;
        // 每个K块内交错执行当前Query对的加权归约与上一对的TopK预筛、候选提交和合并
        for (uint32_t keyTileStart = 0; keyTileStart < taskK; keyTileStart += KEY_TILE_COLUMNS) {
            const int32_t validKeyCount =
                taskK - keyTileStart < KEY_TILE_COLUMNS ? taskK - keyTileStart : KEY_TILE_COLUMNS;
            const bool isLastKeyTile = keyTileStart + validKeyCount == taskK;
            // 首块显式搬运scale；后续块的前半区由上一块预取，此处等其完成
            if (keyTileStart == 0) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
                keyView.CopyFp8Scale(topk.scales, keyTileStart, validKeyCount);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
            }
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
            for (int32_t queryPairIndex = 0; queryPairIndex < ROWS_PER_AIV / 2; ++queryPairIndex) {
                const bool hasPendingQueryPair = pending.queryPairIndex >= 0;
                // score按Query对乒乓；AdvancePair的0/1步负责候选筛选与提交，后续步推进合并
                const int32_t scoreBufferIndex = queryPairIndex & 1;
                topk.BeforeScores(scoreBufferIndex);
                ReduceGroupAndAdvance<0>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         pending);
                ReduceGroupAndAdvance<1>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         pending);
                ReduceGroupAndAdvance<2>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         pending);
                ReduceGroup<1>(effective, topk, queryPairIndex, scoreBufferIndex, 3, paddedHeadNum, heads);
                // 最后一对已消费当前scale前半区，立即预取下一K块前半区，与后半区归约重叠
                if (queryPairIndex == ROWS_PER_AIV / 2 - 1) {
                    PrefetchScales<0>(topk, keyView, keyTileStart, taskK);
                }
                if (hasPendingQueryPair) {
                    topk.template AdvancePair<3>(pending.queryPairIndex * 2, pending.scoreBufferIndex,
                                                 pending.keyTileStart, pending.validKeyCount, pending.isLastKeyTile);
                }
                // 新K块首对切到后1024列前，等待上一K块发起的scale后半区预取完成
                if (queryPairIndex == 0 && keyTileStart != 0 && validKeyCount > 1024) {
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
                ReduceGroupAndAdvance<4>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         pending);
                ReduceGroupAndAdvance<5>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         pending);
                ReduceGroupAndAdvance<6>(effective, topk, queryPairIndex, scoreBufferIndex, paddedHeadNum, heads,
                                         pending);
                ReduceGroup<1>(effective, topk, queryPairIndex, scoreBufferIndex, 7, paddedHeadNum, heads);
                // 当前后半区也已消费完毕，预取下一K块后半区
                if (queryPairIndex == ROWS_PER_AIV / 2 - 1) {
                    PrefetchScales<1>(topk, keyView, keyTileStart, taskK);
                }
                if (hasPendingQueryPair) {
                    topk.template AdvancePair<7>(pending.queryPairIndex * 2, pending.scoreBufferIndex,
                                                 pending.keyTileStart, pending.validKeyCount, pending.isLastKeyTile);
                }
                MaskScoreTail(topk, scoreBufferIndex, queryPairIndex, keyTileStart, validKeyCount, keyLengthOffset,
                              tileBase, sub);
                // 当前两行分数已完整，尝试提前筛选首行，并留到下一对归约时继续处理
                topk.PrepareFirstRow(queryPairIndex * 2, scoreBufferIndex, keyTileStart, validKeyCount);
                pending.queryPairIndex = queryPairIndex;
                pending.scoreBufferIndex = scoreBufferIndex;
                pending.keyTileStart = keyTileStart;
                pending.validKeyCount = validKeyCount;
                pending.isLastKeyTile = isLastKeyTile;
            }
        }
        // 排空最后一对的候选及待合并任务，等待状态落GM后统一输出
        topk.CompletePair(2 * pending.queryPairIndex, pending.scoreBufferIndex, pending.keyTileStart,
                          pending.validKeyCount, pending.isLastKeyTile);
        WriteResults(topk, stateKeys, stateIndices, globalValues, indices, values, requested, validQueries, tileBase,
                     sub, returnValues, useIndexWorkspace);
    }
    // FP8直接输出2048个结果或经内部状态输出后，均须将UB使用权交还V，供下一任务复用暂存区
    SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
}

} // namespace QLIV2FastVector

#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_VECTOR_ARCH35_H
