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
 * \file quant_lightning_indexer_v2_fast_topk.h
 * \brief TopK候选与状态管理、数据搬运同步及最终结果输出
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_TOPK_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_TOPK_H

#include "../quant_lightning_indexer_v2_fast_common_arch35.h"
#include "vf_fast_topk.h"
#include "vf_topk_16_gather_quant_v2.h"
#include "quant_lightning_indexer_v2_vector1.h"

namespace QLIV2FastTopK {
using namespace AscendC;
using QLIV2Fast::Load;
using QLIV2Fast::Store;
using QLIV2Fast::TOPK_STATE_CAPACITY;
constexpr int32_t EQ_COMPACT_MAX_REPEATS = 32; // 单次EQ压缩上限，分段后继续累计AR
constexpr int32_t ROWS_PER_PAIR = 2;
constexpr int32_t AIV_SUB_BLOCK_COUNT = 2;
constexpr int32_t ROWS_PER_GROUP = ROWS_PER_PAIR * AIV_SUB_BLOCK_COUNT;
constexpr int32_t HISTOGRAM_BIN_COUNT = 256;       // 一个字节取值对应256个直方图桶
constexpr int32_t CANDIDATE_CUT_ROW_ELEMENTS = 16; // 单行边界表按32字节对齐，单位uint16

__aicore__ inline void CompactEquals(LocalTensor<uint16_t> keys, LocalTensor<int16_t> positions,
                                     LocalTensor<uint32_t> kth, uint16_t repeats)
{
    // 每次重复覆盖128个uint16元素；按API单次重复上限分段压缩
    for (uint16_t repeatOffset = 0; repeatOffset < repeats; repeatOffset += EQ_COMPACT_MAX_REPEATS) {
        const uint32_t count = static_cast<uint32_t>(GetSpr<SpecialPurposeReg::AR>()) / sizeof(int16_t);
        if (count >= static_cast<uint32_t>(TOPK_STATE_CAPACITY)) {
            break;
        }
        const uint16_t remaining = repeats - repeatOffset;
        const uint16_t n = remaining < EQ_COMPACT_MAX_REPEATS ? remaining : EQ_COMPACT_MAX_REPEATS;
        // 原模板EQ压缩保留AR累计状态；输入和逻辑索引同步偏移，延续当前压缩序列
        topkb16gather::FindIdxEQOutputVFImpl(
            reinterpret_cast<__ubuf__ uint16_t*>(positions.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(keys[static_cast<uint32_t>(repeatOffset) * 128U].GetPhyAddr()),
            static_cast<uint16_t>(repeatOffset * 128U), reinterpret_cast<__ubuf__ uint32_t*>(kth.GetPhyAddr()), n);
    }
}

template <bool IS_MX>
struct PairTopK {
    static constexpr int32_t Rows = 8;
    static constexpr int32_t topk = TOPK_STATE_CAPACITY, batchColumns = QLIV2Fast::KEY_TILE_COLUMNS;
    // 每个tile最多追加2048个候选（包含搬运对齐）
    // 合并阈值6144=8192-2048，为下一批候选预留空间；外部TopK大小不改变内部2048个状态元素及该安全阈值
    static constexpr int32_t candidateCapacity = QLIV2Fast::TOPK_CANDIDATE_CAPACITY;
    static constexpr int32_t mergeThreshold = candidateCapacity - batchColumns;
    // 这些UB视图按生命周期复用同一片区域，末地址为249344字节，不能视为独立分配
    // 直方图、压缩、分数和索引阶段通过V/MTE2/MTE3事件交接缓冲区使用权
    // Cube交接区为[0,131072)，有效权重区为[131072,132096)；249344字节以上的剩余UB未使用
    static constexpr int64_t UB_USED_BYTES = 249344U; // 所有复用视图的共同末地址，非各区间容量之和
    static constexpr int64_t SCORE_STAGES_UB_OFFSET = 203264U;
    static constexpr int64_t PACKED_RECORDS_UB_OFFSET = 219648U;
    static constexpr int64_t SCALES_UB_OFFSET = 236032U;
    static constexpr int64_t COUNTS_UB_OFFSET = 244224U;
    static constexpr int64_t CANDIDATE_CUTS_UB_OFFSET = 244736U;
    static constexpr int64_t HIGH_CDFS_UB_OFFSET = 245248U;
    static constexpr int64_t HISTOGRAM_UB_OFFSET = 132096U;
    static constexpr int64_t TARGET_HIGH_UB_OFFSET = 133120U;
    static constexpr int64_t TARGET_LOW_UB_OFFSET = 134144U;
    static constexpr int64_t THRESHOLD_SELECTION_UB_OFFSET = 137216U;
    static constexpr int64_t COMPACT_INDICES_UB_OFFSET = 137472U;
    static constexpr int64_t MERGE_KEYS_UB_OFFSET = 149760U;
    static constexpr int64_t MERGE_INDICES_UB_OFFSET = 170240U;
    static constexpr int64_t INDEX_SCRATCH_UB_OFFSET = 194816U;
    static constexpr int64_t INTERMEDIATE_KEY_RESULTS_UB_OFFSET = 141568U;
    LocalTensor<uint16_t> scoreStages{TPosition::VECCALC, SCORE_STAGES_UB_OFFSET,
                                      (UB_USED_BYTES - SCORE_STAGES_UB_OFFSET) / sizeof(uint16_t)};
    LocalTensor<uint32_t> packedRecords{TPosition::VECCALC, PACKED_RECORDS_UB_OFFSET,
                                        (UB_USED_BYTES - PACKED_RECORDS_UB_OFFSET) / sizeof(uint32_t)};
    LocalTensor<float> scales{TPosition::VECCALC, SCALES_UB_OFFSET, (UB_USED_BYTES - SCALES_UB_OFFSET) / sizeof(float)};
    LocalTensor<uint32_t> counts{TPosition::VECCALC, COUNTS_UB_OFFSET,
                                 (UB_USED_BYTES - COUNTS_UB_OFFSET) / sizeof(uint32_t)};
    // 每行边界表占16个uint16（32字节），保存8个64K索引分段之间的7个边界
    LocalTensor<uint16_t> candidateCuts{TPosition::VECCALC, CANDIDATE_CUTS_UB_OFFSET,
                                        (UB_USED_BYTES - CANDIDATE_CUTS_UB_OFFSET) / sizeof(uint16_t)};
    LocalTensor<uint16_t> highCdfs{TPosition::VECCALC, HIGH_CDFS_UB_OFFSET,
                                   (UB_USED_BYTES - HIGH_CDFS_UB_OFFSET) / sizeof(uint16_t)};
    LocalTensor<uint32_t> histogram{TPosition::VECCALC, HISTOGRAM_UB_OFFSET,
                                    (UB_USED_BYTES - HISTOGRAM_UB_OFFSET) / sizeof(uint32_t)};
    LocalTensor<uint32_t> targetHigh{TPosition::VECCALC, TARGET_HIGH_UB_OFFSET,
                                     (UB_USED_BYTES - TARGET_HIGH_UB_OFFSET) / sizeof(uint32_t)};
    LocalTensor<uint32_t> targetLow{TPosition::VECCALC, TARGET_LOW_UB_OFFSET,
                                    (UB_USED_BYTES - TARGET_LOW_UB_OFFSET) / sizeof(uint32_t)};
    // 阈值选择的复用通信区：高字节筛选阶段保存桶内排名，初始化及最终压缩后保存阈值
    LocalTensor<uint32_t> thresholdSelectionScratch{TPosition::VECCALC, THRESHOLD_SELECTION_UB_OFFSET,
                                                    (UB_USED_BYTES - THRESHOLD_SELECTION_UB_OFFSET) / sizeof(uint32_t)};
    LocalTensor<int16_t> compactIndices{TPosition::VECCALC, COMPACT_INDICES_UB_OFFSET,
                                        (UB_USED_BYTES - COMPACT_INDICES_UB_OFFSET) / sizeof(int16_t)};
    LocalTensor<uint16_t> mergeKeys{TPosition::VECCALC, MERGE_KEYS_UB_OFFSET,
                                    (UB_USED_BYTES - MERGE_KEYS_UB_OFFSET) / sizeof(uint16_t)};
    LocalTensor<int32_t> mergeIndices{TPosition::VECCALC, MERGE_INDICES_UB_OFFSET,
                                      (UB_USED_BYTES - MERGE_INDICES_UB_OFFSET) / sizeof(int32_t)};
    LocalTensor<int32_t> indexScratch{TPosition::VECCALC, INDEX_SCRATCH_UB_OFFSET,
                                      (UB_USED_BYTES - INDEX_SCRATCH_UB_OFFSET) / sizeof(int32_t)};
    LocalTensor<uint16_t> intermediateKeyResults{
        TPosition::VECCALC, INTERMEDIATE_KEY_RESULTS_UB_OFFSET,
        (UB_USED_BYTES - INTERMEDIATE_KEY_RESULTS_UB_OFFSET) / sizeof(uint16_t)};
    GlobalTensor<uint16_t> stateKeysGm;
    GlobalTensor<int32_t> stateIndicesGm;
    GlobalTensor<uint16_t> candidateKeysGm;
    GlobalTensor<uint16_t> candidateIndicesGm;
    uint32_t subBlock;
    // 返回value或请求TopK小于内部2048时，保留最终分数排序键
    bool keepFinalKey;
    // 候选低位索引限制在单个512K窗口内，合并状态保存完整int32索引
    static constexpr uint32_t indexWindowSize = 1U << 19;
    int64_t candidateIndexBase[Rows]{};
    uint32_t candidateCount[Rows]{};
    uint32_t pendingCount[Rows]{};
    uint32_t pendingDmaCount[Rows]{};
    uint16_t threshold[Rows]{};
    // 本行状态是否已完成过候选合并；首次初始化仅拷贝首个K块，最终收尾仍需完成合并
    bool stateCanonical[Rows]{};
    // 两者均为本AIV的Query行位掩码：标记待合并行，以及其中需要完成最终合并的行
    uint32_t pendingMergeRowMask = 0U;
    uint32_t pendingFinalMergeRowMask = 0U;
    // 当前合并行及已预取的下一行，均为本AIV内行号；-1表示没有对应任务
    int32_t activeRow = -1;
    int32_t prefetchedRow = -1;
    // 合并按时隙推进，状态值连续，完成当前步骤后由AdvanceMerge递增
    static constexpr int32_t MERGE_IDLE = 0;
    static constexpr int32_t MERGE_SELECT_HIGH = 1;
    static constexpr int32_t MERGE_SELECT_LOW = 2;
    static constexpr int32_t MERGE_COMPACT = 3;
    static constexpr int32_t MERGE_GATHER_KEYS = 4;
    int32_t mergeStep = MERGE_IDLE;
    // 正在向GM写回候选的分数缓冲槽，-1表示无待等待的候选搬运
    int32_t candidateDmaStage = -1;
    bool stateKeysPending = false;
    bool stateIndicesPending = false;
    bool firstRowPacked = false;
    __aicore__ inline PairTopK(GlobalTensor<uint16_t> keys, GlobalTensor<int32_t> indices,
                               GlobalTensor<uint16_t> candidates, GlobalTensor<uint16_t> candidateIndices, uint32_t sub,
                               bool keepFinalKey = false)
        : stateKeysGm(keys),
          stateIndicesGm(indices),
          candidateKeysGm(candidates),
          candidateIndicesGm(candidateIndices),
          subBlock(sub),
          keepFinalKey(keepFinalKey)
    {}
    __aicore__ inline static int64_t GlobalRow(uint32_t sub, uint32_t local)
    {
        // 两个AIV交错处理行对，每组覆盖四个全局Query行
        return static_cast<int64_t>(local / ROWS_PER_PAIR) * ROWS_PER_GROUP +
               static_cast<int64_t>(sub) * ROWS_PER_PAIR + local % ROWS_PER_PAIR;
    }
    __aicore__ inline GlobalTensor<uint16_t> CandidateKeyRow(int32_t row) const
    {
        return candidateKeysGm[GlobalRow(subBlock, static_cast<uint32_t>(row)) * candidateCapacity];
    }
    __aicore__ inline bool NeedsFinalMerge(int32_t row) const
    {
        return candidateCount[row] != 0U || !stateCanonical[row];
    }
    __aicore__ inline static uint32_t AlignCandidateCount(uint32_t n)
    {
        // GM搬运按32字节对齐，即16个uint16元素；15为向上取整偏置
        return ((n + 15U) / 16U) * 16U;
    }
    __aicore__ inline void IssueMergeKeys(LocalTensor<uint16_t> destination, int32_t row, uint32_t count)
    {
        Load(destination, stateKeysGm[GlobalRow(subBlock, row) * topk], topk);
        Load(destination[topk], CandidateKeyRow(row), count);
    }
    __aicore__ inline void StartStateKeys(int32_t row, LocalTensor<uint16_t> keys)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        Store(stateKeysGm[GlobalRow(subBlock, row) * topk], keys, topk);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    __aicore__ inline void StartStateIndices(int32_t row, LocalTensor<int32_t> indices)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        Store(stateIndicesGm[GlobalRow(subBlock, row) * topk], indices, topk);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }
    __aicore__ inline auto Score(int32_t scoreBufferIndex, int32_t row,
                                 int32_t column = 0) const -> LocalTensor<uint16_t>
    {
        return scoreStages[scoreBufferIndex * ROWS_PER_PAIR * batchColumns + row * batchColumns + column];
    }

    __aicore__ inline void PrepareIdentity()
    {
        PrepareIdentityVF(reinterpret_cast<__ubuf__ int32_t*>((indexScratch).GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint16_t*>((candidateCuts).GetPhyAddr()));
    }

    __aicore__ inline void InitializePair(int32_t rowBase, int32_t scoreBufferIndex)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
        for (int32_t r = 0; r < ROWS_PER_PAIR; ++r) {
            const int32_t row = rowBase + r;
            const auto global = GlobalRow(subBlock, static_cast<uint32_t>(row));
            Store(stateKeysGm[global * topk], Score(scoreBufferIndex, r), topk);
            Store(stateIndicesGm[global * topk], indexScratch, topk);
            InitializeHistogramVF(
                reinterpret_cast<__ubuf__ uint16_t*>((Score(scoreBufferIndex, r)).GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>((highCdfs[row * HISTOGRAM_BIN_COUNT]).GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>((thresholdSelectionScratch).GetPhyAddr()));
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            threshold[row] = static_cast<uint16_t>(thresholdSelectionScratch.GetValue(0));
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
    }

    __aicore__ inline void PrepareFirstRow(int32_t rowBase, int32_t scoreBufferIndex, int64_t keyTileStart,
                                           int32_t validKeyCount)
    {
        // 两行分数均已就绪时，利用空闲合并时隙提前打包第一行；有待处理合并时，仍保留原合并时隙和双行打包路径
        if (keyTileStart == 0 || pendingMergeRowMask != 0U) {
            return;
        }
        WaitCandidates();
        // 32位候选记录高16位为分数、低16位为索引；0xffff排除等于阈值的记录
        const uint32_t kth = (static_cast<uint32_t>(threshold[rowBase]) << 16) | 0xffffU;
        PackRowVF(reinterpret_cast<__ubuf__ uint16_t*>((Score(scoreBufferIndex, 0)).GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>((packedRecords).GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>((counts).GetPhyAddr()), kth,
                  static_cast<uint32_t>(validKeyCount));
        firstRowPacked = true;
    }

    __aicore__ inline void PackPair(int32_t rowBase, int32_t scoreBufferIndex, int32_t validKeyCount)
    {
        if constexpr (!IS_MX) {
            WaitCandidates();
            // 3（二进制11）同时标记当前Query对的两行
            const uint32_t rows = 3U << rowBase;
            // 当前行前次合并完成后，才能读取阈值并复用候选区
            while ((pendingMergeRowMask & rows) != 0U) {
                AdvanceMerge();
            }
        }
        // 两行分别设置阈值：高16位分数、低16位0xffff，筛选严格大于阈值的记录
        const uint32_t kth0 = (static_cast<uint32_t>(threshold[rowBase]) << 16) | 0xffffU;
        const uint32_t kth1 = (static_cast<uint32_t>(threshold[rowBase + 1]) << 16) | 0xffffU;
        if (firstRowPacked) {
            PackRowVF(reinterpret_cast<__ubuf__ uint16_t*>((Score(scoreBufferIndex, 1)).GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t*>((packedRecords[batchColumns]).GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t*>((counts[1]).GetPhyAddr()), kth1,
                      static_cast<uint32_t>(validKeyCount));
            firstRowPacked = false;
        } else {
            PackPairVF(reinterpret_cast<__ubuf__ uint16_t*>((Score(scoreBufferIndex, 0)).GetPhyAddr()),
                       reinterpret_cast<__ubuf__ uint16_t*>((Score(scoreBufferIndex, 1)).GetPhyAddr()),
                       reinterpret_cast<__ubuf__ uint32_t*>((packedRecords).GetPhyAddr()),
                       reinterpret_cast<__ubuf__ uint32_t*>((counts).GetPhyAddr()), kth0, kth1,
                       static_cast<uint32_t>(validKeyCount));
        }
        SetFlag<HardEvent::V_S>(EVENT_ID7);
    }

    __aicore__ inline void CommitPair(int32_t rowBase, int32_t scoreBufferIndex, int64_t keyTileStart)
    {
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (int32_t r = 0; r < ROWS_PER_PAIR; ++r) {
            const int32_t row = rowBase + r;
            pendingCount[row] = counts.GetValue(r) / sizeof(uint32_t);
            pendingDmaCount[row] = AlignCandidateCount(pendingCount[row]);
            if (pendingDmaCount[row] != 0U) {
                // 窗口大小为2的幂，64位掩码对齐可保留高位索引并避免热路径除法
                candidateIndexBase[row] = keyTileStart & ~(static_cast<int64_t>(indexWindowSize) - 1);
                UnpackCandidates(
                    reinterpret_cast<__ubuf__ uint32_t*>((packedRecords[r * batchColumns]).GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>((Score(scoreBufferIndex, r)).GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(
                        (candidateCuts[row * CANDIDATE_CUT_ROW_ELEMENTS]).GetPhyAddr()),
                    static_cast<uint32_t>(keyTileStart & (indexWindowSize - 1U)), pendingCount[row],
                    pendingDmaCount[row], candidateCount[row],
                    reinterpret_cast<__ubuf__ uint16_t*>((highCdfs[row * HISTOGRAM_BIN_COUNT]).GetPhyAddr()));
            }
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        for (int32_t r = 0; r < ROWS_PER_PAIR; ++r) {
            const int32_t row = rowBase + r;
            const auto count = pendingDmaCount[row];
            if (count == 0U) {
                continue;
            }
            const auto offset = candidateCount[row];
            Store(CandidateKeyRow(row)[offset], Score(scoreBufferIndex, r), count);
            Store(CandidateIndexRow(row)[offset], packedRecords[r * batchColumns].ReinterpretCast<uint16_t>(), count);
            candidateCount[row] = offset + count;
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        candidateDmaStage = scoreBufferIndex;
    }

    __aicore__ inline void WaitCandidates()
    {
        if (candidateDmaStage < 0) {
            return;
        }
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        candidateDmaStage = -1;
    }

    __aicore__ inline auto CandidateIndexRow(int32_t row) const -> GlobalTensor<uint16_t>
    {
        return candidateIndicesGm[GlobalRow(subBlock, static_cast<uint32_t>(row)) * candidateCapacity];
    }

    __aicore__ inline auto MergeCandidateIndices() const -> LocalTensor<uint16_t>
    {
        return mergeIndices[topk].ReinterpretCast<uint16_t>();
    }

    __aicore__ inline void BeforeScores(int32_t scoreBufferIndex)
    {
        if (candidateDmaStage == scoreBufferIndex) {
            WaitCandidates();
        }
    }

    __aicore__ inline void WaitStateKeys()
    {
        if (!stateKeysPending) {
            return;
        }
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        stateKeysPending = false;
    }

    __aicore__ inline void WaitStateIndices()
    {
        if (!stateIndicesPending) {
            return;
        }
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        stateIndicesPending = false;
    }

    __aicore__ inline void EnqueuePair(int32_t rowBase, bool isFinalMerge, int64_t keyTileStart)
    {
        for (int32_t r = rowBase; r < rowBase + ROWS_PER_PAIR; ++r) {
            // 跨窗口复用低位索引前完成合并，候选数量较少时也不能跳过
            const bool windowEnd = ((keyTileStart + batchColumns) & (indexWindowSize - 1U)) == 0U;
            if (isFinalMerge ? NeedsFinalMerge(r) :
                               (candidateCount[r] > static_cast<uint32_t>(mergeThreshold) ||
                                (windowEnd && candidateCount[r] != 0U))) {
                pendingMergeRowMask |= 1U << r;
                if (isFinalMerge) {
                    pendingFinalMergeRowMask |= 1U << r;
                }
            }
        }
        if constexpr (IS_MX) {
            if ((pendingMergeRowMask & (3U << rowBase)) != 0U) {
                // 合并重读前须等待候选和状态写入GM完成，仅等待V计算完成并不足够
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            }
        }
        BeginMergeIfReady();
    }

    __aicore__ inline void IssueKeys(int32_t row)
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
        IssueMergeKeys(mergeKeys, row, candidateCount[row]);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
    }

    __aicore__ inline void FinishInitialPair()
    {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    }
    __aicore__ inline void BeginMergeIfReady()
    {
        if constexpr (IS_MX) {
            if (pendingMergeRowMask == 0U || activeRow >= 0) {
                return;
            }
            if (prefetchedRow >= 0) {
                activeRow = prefetchedRow;
                prefetchedRow = -1;
            } else {
                activeRow = 0;
                while ((pendingMergeRowMask & (1U << activeRow)) == 0U) {
                    ++activeRow;
                }
                IssueKeys(activeRow);
            }
            mergeStep = MERGE_SELECT_HIGH;
        } else {
            if (pendingMergeRowMask == 0U || activeRow >= 0) {
                return;
            }
            activeRow = 0;
            while ((pendingMergeRowMask & (1U << activeRow)) == 0U) {
                ++activeRow;
            }
            const int32_t row = activeRow;
            // 在入队或合并完成时发起搬运，避免仅发起搬运的状态占用后续加权归约时隙
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
            IssueMergeKeys(mergeKeys, row, candidateCount[row]);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
            const auto global = GlobalRow(subBlock, static_cast<uint32_t>(row));
            Load(mergeIndices, stateIndicesGm[global * topk], topk);
            Load(MergeCandidateIndices(), CandidateIndexRow(row), candidateCount[row]);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
            mergeStep = MERGE_SELECT_HIGH;
        }
    }

    __aicore__ inline void GatherRowIndices(int32_t row)
    {
        // 首窗口沿用原VF路径，覆盖所有S2<=512K的场景
        if (candidateIndexBase[row] == 0U) {
            GatherIndices<false>(
                reinterpret_cast<__ubuf__ int32_t*>(mergeIndices.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(MergeCandidateIndices().GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(candidateCuts[row * CANDIDATE_CUT_ROW_ELEMENTS].GetPhyAddr()),
                reinterpret_cast<__ubuf__ int16_t*>(compactIndices.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(indexScratch.GetPhyAddr()), 0U);
        } else {
            GatherIndices<true>(
                reinterpret_cast<__ubuf__ int32_t*>(mergeIndices.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(MergeCandidateIndices().GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(candidateCuts[row * CANDIDATE_CUT_ROW_ELEMENTS].GetPhyAddr()),
                reinterpret_cast<__ubuf__ int16_t*>(compactIndices.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(indexScratch.GetPhyAddr()),
                static_cast<uint32_t>(candidateIndexBase[row]));
        }
    }

    __aicore__ inline void AdvanceMerge()
    {
        if (pendingMergeRowMask == 0U) {
            return;
        }
        if constexpr (!IS_MX) {
            BeginMergeIfReady();
        }
        // MX已在入队时启动当前行；FP8在此保持原有的启动检查
        const int32_t row = activeRow;
        const bool isFinalMerge = (pendingFinalMergeRowMask & (1U << row)) != 0U;
        const uint32_t length = (static_cast<uint32_t>(topk) + candidateCount[row] + HISTOGRAM_BIN_COUNT - 1U) /
                                HISTOGRAM_BIN_COUNT * HISTOGRAM_BIN_COUNT;
        if (mergeStep == MERGE_SELECT_HIGH) {
            HighCdf::Select<!IS_MX>(highCdfs[row * HISTOGRAM_BIN_COUNT], targetHigh, thresholdSelectionScratch,
                                    length - topk + 1U, length - topk - candidateCount[row]);
        } else if (mergeStep == MERGE_SELECT_LOW) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            RepairPadding(reinterpret_cast<__ubuf__ uint16_t*>((mergeKeys).GetPhyAddr()), candidateCount[row]);
            if constexpr (IS_MX) {
                FindThresholdMxVF(reinterpret_cast<__ubuf__ uint16_t*>(mergeKeys.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ uint32_t*>(targetHigh.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ uint32_t*>(thresholdSelectionScratch.GetPhyAddr()), length);
            } else {
                // 复用原模板低字节直方图和阈值拼接，rank由SelectHighVF按原VF布局广播
                topkb16gather::HistogramsLowVFImpl<uint16_t>(
                    reinterpret_cast<__ubuf__ uint32_t*>(histogram.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(mergeKeys.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(targetHigh.GetPhyAddr()),
                    static_cast<uint16_t>(length / HISTOGRAM_BIN_COUNT), true);
                topkb16gather::FindKthVFImpl(
                    reinterpret_cast<__ubuf__ uint32_t*>(thresholdSelectionScratch.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(histogram.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(targetHigh.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(targetLow.GetPhyAddr()));
            }
            if constexpr (IS_MX) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
                const auto global = GlobalRow(subBlock, static_cast<uint32_t>(row));
                Load(mergeIndices, stateIndicesGm[global * topk], topk);
                Load(MergeCandidateIndices(), CandidateIndexRow(row), candidateCount[row]);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
            }
            // 阈值通信区使用独立的V到Scalar同步事件
            SetFlag<HardEvent::V_S>(EVENT_ID3);
        } else if (mergeStep == MERGE_COMPACT) {
            WaitStateKeys();
            // 严格大于和等于阈值的压缩过程连续使用AR寄存器；打包可穿插于其他合并步骤之间，但不能打断此压缩过程
            topkb16gather::FindIdxGTOutputVFImpl(
                reinterpret_cast<__ubuf__ uint16_t*>(compactIndices.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(mergeKeys.GetPhyAddr()), 0U,
                reinterpret_cast<__ubuf__ uint32_t*>(thresholdSelectionScratch.GetPhyAddr()),
                static_cast<uint16_t>(length / 128U));
            CompactEquals(mergeKeys, compactIndices, thresholdSelectionScratch, static_cast<uint16_t>(length / 128U));
            // 最终仅输出索引时，跳过压缩与索引收集之间无需执行的分数收集步骤
            if (isFinalMerge && !keepFinalKey) {
                mergeStep = MERGE_GATHER_KEYS;
            }
        } else if (mergeStep == MERGE_GATHER_KEYS) {
            if (!isFinalMerge || keepFinalKey) {
                GatherKeysVF(reinterpret_cast<__ubuf__ int16_t*>((compactIndices).GetPhyAddr()),
                             reinterpret_cast<__ubuf__ uint16_t*>((mergeKeys).GetPhyAddr()),
                             reinterpret_cast<__ubuf__ uint16_t*>((intermediateKeyResults).GetPhyAddr()),
                             reinterpret_cast<__ubuf__ uint16_t*>((highCdfs[row * HISTOGRAM_BIN_COUNT]).GetPhyAddr()));
                StartStateKeys(row, intermediateKeyResults);
                stateKeysPending = true;
            }
        } else {
            // 分数收集完成或跳过后，状态为MERGE_GATHER_KEYS+1；收集索引并发布本行状态
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
            WaitStateIndices();
            GatherRowIndices(row);
            StartStateIndices(row, indexScratch);
            stateIndicesPending = true;
            WaitFlag<HardEvent::V_S>(EVENT_ID3);
            threshold[row] = static_cast<uint16_t>(thresholdSelectionScratch.GetValue(0));
            candidateCount[row] = 0U;
            stateCanonical[row] = true;
            pendingMergeRowMask &= ~(1U << row);
            pendingFinalMergeRowMask &= ~(1U << row);
            activeRow = -1;
            BeginMergeIfReady();
            return;
        }
        if constexpr (IS_MX) {
            if (mergeStep == MERGE_GATHER_KEYS) {
                const uint32_t remaining = pendingMergeRowMask & ~(1U << row);
                if (remaining != 0U) {
                    prefetchedRow = 0;
                    while ((remaining & (1U << prefetchedRow)) == 0U) {
                        ++prefetchedRow;
                    }
                    IssueKeys(prefetchedRow);
                }
            }
        }
        ++mergeStep;
    }

    template <int32_t Step>
    __aicore__ inline void AdvancePair(int32_t rowBase, int32_t scoreBufferIndex, int64_t keyTileStart,
                                       int32_t validKeyCount, bool isLastKeyTile)
    {
        if (keyTileStart == 0) {
            if constexpr (Step == 0) {
                InitializePair(rowBase, scoreBufferIndex);
            } else if constexpr (Step == 7) {
                // 加权归约与首块状态写回交错；复用当前源缓冲区和事件6前，须等待搬运完成
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
            }
            return;
        }
        if constexpr (Step == 0) {
            PackPair(rowBase, scoreBufferIndex, validKeyCount);
        } else if constexpr (Step == 1) {
            CommitPair(rowBase, scoreBufferIndex, keyTileStart);
            EnqueuePair(rowBase, isLastKeyTile, keyTileStart);
        } else {
            AdvanceMerge();
        }
    }

    __aicore__ inline void CompletePair(int32_t rowBase, int32_t scoreBufferIndex, int64_t keyTileStart,
                                        int32_t validKeyCount, bool isLastKeyTile)
    {
        PackPair(rowBase, scoreBufferIndex, validKeyCount);
        CommitPair(rowBase, scoreBufferIndex, keyTileStart);
        EnqueuePair(rowBase, isLastKeyTile, keyTileStart);
        while (pendingMergeRowMask != 0U) {
            AdvanceMerge();
        }
        WaitCandidates();
        WaitStateKeys();
        WaitStateIndices();
    }
};

} // namespace QLIV2FastTopK

namespace QLIV2FastTopK {
// 内部保持2048个状态元素，与接口输出步长解耦；最后通过基数筛选支持请求的TopK大小，包括奇数TopK
template <class TopK>
__aicore__ inline void Finalize(TopK& topk, GlobalTensor<uint16_t> keys, GlobalTensor<int32_t> indices, GM_ADDR output,
                                GM_ADDR values, uint32_t requested, uint32_t validQueries, int64_t tileBase,
                                bool returnValues)
{
    GlobalTensor<int32_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
    GlobalTensor<bfloat16_t> val;
    val.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(values));
    for (uint32_t row = 0U; row < 8U; ++row) {
        const uint32_t globalQueryRow = TopK::GlobalRow(GetSubBlockIdx(), row);
        if (globalQueryRow >= validQueries) {
            continue;
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        Load(topk.mergeKeys, keys[globalQueryRow * TOPK_STATE_CAPACITY], TOPK_STATE_CAPACITY);
        Load(topk.mergeIndices, indices[globalQueryRow * TOPK_STATE_CAPACITY], TOPK_STATE_CAPACITY);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
        if (requested < TOPK_STATE_CAPACITY) {
            // 直接复用原模板完整筛选入口，避免改变原VF实现或重复维护筛选计算体
            topkb16gather::QLiV2TopKVF<false>(topk.compactIndices.template ReinterpretCast<uint16_t>(),
                                              topk.intermediateKeyResults, topk.mergeKeys, topk.histogram,
                                              topk.targetHigh, topk.targetLow, topk.thresholdSelectionScratch,
                                              requested, TOPK_STATE_CAPACITY);
            // 仅返回索引时无需再次收集最终分数，与原模板的ISOUTVALUE语义一致
            if (returnValues) {
                topkb16gather::FindValueOutputVFImpl(
                    reinterpret_cast<__ubuf__ uint16_t*>(topk.intermediateKeyResults.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(topk.mergeKeys.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint16_t*>(topk.compactIndices.GetPhyAddr()),
                    // 每次处理128个uint16；127为尾轮向上取整偏置
                    static_cast<uint16_t>((requested + 127U) / 128U));
            }
            PipeBarrier<PIPE_V>();
            topkb16gather::QLiV2TopKLDGatherVF(topk.indexScratch.template ReinterpretCast<uint32_t>(),
                                               topk.compactIndices.template ReinterpretCast<uint16_t>(),
                                               topk.mergeIndices.template ReinterpretCast<uint32_t>(), requested);
        } else {
            DataCopy(topk.indexScratch, topk.mergeIndices, TOPK_STATE_CAPACITY);
            DataCopy(topk.intermediateKeyResults, topk.mergeKeys, TOPK_STATE_CAPACITY);
        }
        PipeBarrier<PIPE_V>();
        if (returnValues) {
            vector1::UIntToFloatReturnValueWithInfMask(topk.mergeKeys.template ReinterpretCast<bfloat16_t>(),
                                                       topk.intermediateKeyResults, requested,
                                                       QLIV2Fast::BF16_NEGATIVE_INF_BITS);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        const int64_t offset = (tileBase + globalQueryRow) * requested;
        Store(out[offset], topk.indexScratch, requested);
        if (returnValues) {
            Store(val[offset], topk.mergeKeys.template ReinterpretCast<bfloat16_t>(), requested);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
    }
    // 下一MX任务会在权重搬运就绪前执行索引初始化VF；必须等待最终输出搬运读完UB，才能将其使用权交还V
    SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
}
} // namespace QLIV2FastTopK

#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_TOPK_H
