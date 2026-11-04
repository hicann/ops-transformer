/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_FAST_SERVICE_VECTOR_ARCH35_H
#define LIGHTNING_INDEXER_FAST_SERVICE_VECTOR_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "vf/lightning_indexer_v2_fast_topk.h"

namespace LIV2Kernel {
using namespace LIV2Fast;

// Per-AIV TopK state: initial 2048 keys -> filtered candidate epochs -> final output.
// Buffer/workspace bindings are reused; BeginQueryTile resets only tile-local state.
template <typename LIT>
class LightningIndexerV2StreamingTopK {
public:
    using SCORE_T = uint32_t;

    __aicore__ inline void InitBuffers(FastVectorBuffers& buffers)
    {
        buffers_ = &buffers;
    }

    __aicore__ inline void InitWorkspace(const FastWorkspace& workspace, uint32_t subBlock, uint32_t outputTopK)
    {
        stateKeys_ = workspace.stateKeys;
        candidateKeys_ = workspace.candidateKeys;
        candidateIndices_ = workspace.candidateIndices;
        paddingIndices_ = workspace.paddingIndices;
        epochStride_ = workspace.epochStride;
        subBlock_ = subBlock;
        outputTopK_ = outputTopK;
    }

    // Finish retires all DMA before the next tile reuses the same workspace.
    __aicore__ inline void BeginQueryTile(__gm__ SCORE_T* outputIndices, __gm__ SCORE_T* outputValues,
                                          uint32_t validQueryRows)
    {
        outputIndices_ = outputIndices;
        outputValues_ = outputValues;
        stateIndices_ = outputTopK_ == kTopK ? outputIndices : paddingIndices_;
        validQueryRows_ = validQueryRows;
        for (uint32_t row = 0; row < kAivRows; ++row) {
            logCount_[row] = candidateCount_[row] = frozenCount_[row] = threshold_[row] = 0U;
        }
        activeEpoch_ = frozenEpoch_ = activeLogSlot_ = 0U;
        mergeRowCursor_ = mergePrefix_ = 0U;
        mergeStage_ = IDLE;
        mergeRank_ = kTopK;
        mergeStrict_ = mergeEqual_ = mergeEqualBegin_ = 0U;
        logDmaPending_ = stateStorePending_ = false;
    }

    __aicore__ inline uint32_t GetThreshold(uint32_t row) const
    {
        return threshold_[row];
    }

    // Log page: four key rows followed by four index rows, each reserving 512 words.
    // The index section therefore starts at 4 * 512 = 2048 uint32_t elements.
    __aicore__ inline __ubuf__ uint32_t* LogKeyRow(int32_t row) const
    {
        return reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetLogPage(activeLogSlot_).GetPhyAddr()) +
               static_cast<uint64_t>(row) * kLogRowElements + logCount_[row];
    }

    __aicore__ inline __ubuf__ uint32_t* LogIndexRow(int32_t row) const
    {
        return reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetLogPage(activeLogSlot_).GetPhyAddr()) +
               kLogIndexOffset + static_cast<uint64_t>(row) * kLogRowElements + logCount_[row];
    }

    __aicore__ inline void ConsumeInitialPair(int32_t localRowBase, const LocalTensor<uint32_t>& scores,
                                              uint64_t keyOffset)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::LOG_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::LOG_STORE);
        CopyOutUB(KeyStateRow(localRowBase) + keyOffset, scores, static_cast<uint32_t>(kFixTile));
        CopyOutUB(KeyStateRow(localRowBase + 1) + keyOffset, scores[static_cast<uint32_t>(kFixTile)],
                  static_cast<uint32_t>(kFixTile));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::LOG_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::LOG_STORE);
        if (keyOffset + static_cast<uint32_t>(kFixTile) == kTopK && localRowBase == kQueryPairRows) {
            auto* indices = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetIndexScratch().GetPhyAddr());
            vf::GenerateInitialTopkIndices(indices);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
            for (int32_t row = 0; row < kAivRows; ++row) {
                CopyOutUB(IndexStateRow(row), buffers_->GetIndexScratch(), kTopK);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        }
    }

    __aicore__ inline void ProgressFrozenMerge()
    {
        if (mergeStage_ == IDLE) {
            return;
        }
        if (mergeStage_ <= SELECT_BYTE0) {
            ProgressMergeKeySelection();
        } else if (mergeStage_ <= SELECT_EQUAL) {
            ProgressMergePositions();
        } else {
            ProgressMergeState();
        }
    }

    __aicore__ inline void CommitFilteredRows()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(FastEvent::SCALAR_READ);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(FastEvent::SCALAR_READ);
        auto counts = buffers_->GetCounts();
        for (int32_t row = 0; row < kAivRows; ++row) {
            // VF compact counts are byte offsets; one uint32_t key occupies four bytes.
            logCount_[row] += counts.GetValue(row) / sizeof(uint32_t);
        }
        FlushLogsIfNeeded();
        bool freeze = false;
        for (int32_t row = 0; row < kAivRows; ++row) {
            // Reserve 512 of 4096 candidate slots for the next log flush.
            freeze = freeze || candidateCount_[row] > kCandidateFreezeThreshold;
        }
        if (!freeze) {
            return;
        }
        if (mergeStage_ != IDLE) {
            DrainFrozenMerge();
        }
        FlushAllLogs();
        frozenEpoch_ = activeEpoch_;
        // Toggle the two epochs: merge the frozen one while collecting in the other.
        activeEpoch_ ^= 1U;
        for (int32_t row = 0; row < kAivRows; ++row) {
            frozenCount_[row] = candidateCount_[row];
            candidateCount_[row] = 0U;
        }
        mergeRowCursor_ = 0U;
        SeekNextFrozenRow();
    }

    __aicore__ inline void Finish()
    {
        DrainFrozenMerge();
        FlushAllLogs();
        for (int32_t row = 0; row < kAivRows; ++row) {
            if (outputTopK_ == kTopK && outputValues_ == nullptr) {
                MergeFinalRow(row);
            } else {
                MergeRow(row, candidateCount_[row], activeEpoch_);
            }
        }
        FinishStateStore();
        if (outputTopK_ != kTopK || outputValues_ != nullptr) {
            for (int32_t row = 0; row < kAivRows; ++row) {
                if (PhysicalRow(row) >= validQueryRows_) {
                    continue;
                }
                if (outputTopK_ != kTopK) {
                    WriteRequestedTopK(row);
                } else {
                    WriteTopKValues(row);
                }
            }
        }
    }

private:
    // One step is advanced between WRS calls to overlap merge DMA with score computation.
    // Ordered state IDs; the dispatcher compares contiguous stage ranges.
    enum MergeStage : uint32_t {
        IDLE = 0U,
        LOAD_KEYS = 1U,
        WAIT_KEYS = 2U,
        SELECT_BYTE3 = 3U,
        SELECT_BYTE2 = 4U,
        SELECT_BYTE1 = 5U,
        SELECT_BYTE0 = 6U,
        SELECT_STRICT = 7U,
        SELECT_EQUAL = 8U,
        GATHER_KEYS = 9U,
        STORE_KEYS = 10U,
        LOAD_INDICES = 11U,
        WAIT_INDICES = 12U,
        GATHER_INDICES = 13U,
        STORE_INDICES = 14U,
    };

    __aicore__ inline void ProgressMergeKeySelection()
    {
        const int32_t row = static_cast<int32_t>(mergeRowCursor_);
        const uint32_t count = frozenCount_[row];
        const uint32_t length = kTopK + frozenCount_[mergeRowCursor_];
        auto* merge = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetMerge().GetPhyAddr());
        if (mergeStage_ == LOAD_KEYS) {
            SubmitMergeData(KeyStateRow(row), CandidateKeyRow(row, frozenEpoch_), count);
            mergeStage_ = WAIT_KEYS;
            return;
        }
        // Keep one WRS interval between issuing MTE2 and consuming its data.
        if (mergeStage_ == WAIT_KEYS) {
            mergeStage_ = SELECT_BYTE3;
            return;
        }
        if (mergeStage_ == SELECT_BYTE3) {
            FinishMergeLoad();
            mergePrefix_ = 0U;
            mergeRank_ = kTopK;
            vf::SelectRadixByte(*buffers_, merge, length, kRadixMostSignificantByte, mergePrefix_, mergeRank_);
            mergeStage_ = SELECT_BYTE2;
            return;
        }
        if (mergeStage_ == SELECT_BYTE2) {
            vf::SelectRadixByte(*buffers_, merge, length, kRadixSecondByte, mergePrefix_, mergeRank_);
            mergeStage_ = SELECT_BYTE1;
            return;
        }
        if (mergeStage_ == SELECT_BYTE1) {
            vf::SelectRadixByte(*buffers_, merge, length, 1, mergePrefix_, mergeRank_);
            mergeStage_ = SELECT_BYTE0;
            return;
        }
        if (mergeStage_ == SELECT_BYTE0) {
            vf::SelectRadixByte(*buffers_, merge, length, 0, mergePrefix_, mergeRank_);
            mergeStage_ = SELECT_STRICT;
            return;
        }
    }

    __aicore__ inline void ProgressMergePositions()
    {
        const uint32_t length = kTopK + frozenCount_[mergeRowCursor_];
        auto* merge = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetMerge().GetPhyAddr());
        auto* positions = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetPositions().GetPhyAddr());
        if (mergeStage_ == SELECT_STRICT) {
            mergeStrict_ = vf::CompactStrictPositions(merge, length, mergePrefix_, positions);
            mergeEqual_ = 0U;
            mergeEqualBegin_ = 0U;
            mergeStage_ = SELECT_EQUAL;
            return;
        }
        if (mergeStage_ == SELECT_EQUAL) {
            const uint32_t chunk = length - mergeEqualBegin_ < static_cast<uint32_t>(kFixTile) ?
                                       length - mergeEqualBegin_ :
                                       static_cast<uint32_t>(kFixTile);
            mergeEqual_ += vf::CompactEqualPositionsChunk(merge, mergeEqualBegin_, chunk, mergePrefix_,
                                                          positions + mergeStrict_ + mergeEqual_);
            mergeEqualBegin_ += static_cast<uint32_t>(kFixTile);
            if (mergeStrict_ + mergeEqual_ >= kTopK) {
                mergeStage_ = GATHER_KEYS;
            }
            return;
        }
    }

    __aicore__ inline void ProgressMergeState()
    {
        const int32_t row = static_cast<int32_t>(mergeRowCursor_);
        const uint32_t count = frozenCount_[row];
        auto* merge = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetMerge().GetPhyAddr());
        auto* scratch = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetIndexScratch().GetPhyAddr());
        if (mergeStage_ == GATHER_KEYS) {
            FinishStateStore();
            vf::GatherSelectedTopk(*buffers_, merge, scratch);
            mergeStage_ = STORE_KEYS;
            return;
        }
        if (mergeStage_ == STORE_KEYS) {
            BeginStateStore(row, true);
            mergeStage_ = LOAD_INDICES;
            return;
        }
        if (mergeStage_ == LOAD_INDICES) {
            SubmitMergeData(IndexStateRow(row), CandidateIndexRow(row, frozenEpoch_), count);
            mergeStage_ = WAIT_INDICES;
            return;
        }
        // The index load uses the same deferred-completion protocol as the keys.
        if (mergeStage_ == WAIT_INDICES) {
            mergeStage_ = GATHER_INDICES;
            return;
        }
        if (mergeStage_ == GATHER_INDICES) {
            FinishMergeLoad();
            FinishStateStore();
            vf::GatherSelectedTopk(*buffers_, merge, scratch);
            mergeStage_ = STORE_INDICES;
            return;
        }
        BeginStateStore(row, false);
        threshold_[row] = mergePrefix_;
        frozenCount_[row] = 0U;
        ++mergeRowCursor_;
        SeekNextFrozenRow();
    }

    __aicore__ inline void WriteTopKValues(int32_t row)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(FastEvent::GM_REUSE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(FastEvent::GM_REUSE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        CopyInUB(buffers_->GetMerge(), KeyStateRow(row), kTopK);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
        vf::SortableKeysToFloatBits(reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetMerge().GetPhyAddr()), kTopK);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        CopyOutUB(outputValues_ + static_cast<uint64_t>(PhysicalRow(row)) * kTopK, buffers_->GetMerge(), kTopK);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
    }

    __aicore__ inline void WriteRequestedTopK(int32_t row)
    {
        auto* keys = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetMerge().GetPhyAddr());
        auto* indices = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetFinalIndices().GetPhyAddr());
        auto* selected = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetIndexScratch().GetPhyAddr());
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(FastEvent::GM_REUSE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(FastEvent::GM_REUSE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        CopyInUB(buffers_->GetMerge(), KeyStateRow(row), kTopK);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
        const uint32_t kth = vf::KthSortableKey(*buffers_, keys, kTopK, outputTopK_);
        vf::SelectStableTopkPositions(*buffers_, keys, kTopK, kth, outputTopK_);
        if (outputValues_ != nullptr) {
            vf::GatherSelectedCount(*buffers_, keys, selected, outputTopK_);
            vf::SortableKeysToFloatBits(selected, outputTopK_);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
            CopyOutUB(outputValues_ + static_cast<uint64_t>(PhysicalRow(row)) * outputTopK_,
                      buffers_->GetIndexScratch(), outputTopK_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        CopyInUB(buffers_->GetFinalIndices(), IndexStateRow(row), kTopK);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
        vf::GatherSelectedCount(*buffers_, indices, selected, outputTopK_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        CopyOutUB(outputIndices_ + static_cast<uint64_t>(PhysicalRow(row)) * outputTopK_, buffers_->GetIndexScratch(),
                  outputTopK_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
    }

    __aicore__ inline uint32_t PhysicalRow(uint32_t local) const
    {
        return local < kQueryPairRows ? kQueryPairRows * subBlock_ + local :
                                        kQuerySlotRows + kQueryPairRows * subBlock_ + local - kQueryPairRows;
    }

    __aicore__ inline __gm__ uint32_t* KeyStateRow(int32_t row) const
    {
        return stateKeys_ + static_cast<uint64_t>(PhysicalRow(static_cast<uint32_t>(row))) * kTopK;
    }

    __aicore__ inline __gm__ uint32_t* IndexStateRow(int32_t row) const
    {
        const uint64_t physical = PhysicalRow(static_cast<uint32_t>(row));
        return (physical < validQueryRows_ ? stateIndices_ : paddingIndices_) + physical * kTopK;
    }

    __aicore__ inline __gm__ uint32_t* CandidateKeyRow(int32_t row, uint32_t epoch) const
    {
        return candidateKeys_ + static_cast<uint64_t>(epoch) * epochStride_ +
               static_cast<uint64_t>(PhysicalRow(static_cast<uint32_t>(row))) * kCandidateCapacity;
    }

    __aicore__ inline __gm__ uint32_t* CandidateIndexRow(int32_t row, uint32_t epoch) const
    {
        return candidateIndices_ + static_cast<uint64_t>(epoch) * epochStride_ +
               static_cast<uint64_t>(PhysicalRow(static_cast<uint32_t>(row))) * kCandidateCapacity;
    }

    __aicore__ inline void RetireLogDma()
    {
        if (!logDmaPending_) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::LOG_STORE);
        logDmaPending_ = false;
    }

    __aicore__ inline void SubmitLogFlush(int32_t row)
    {
        const uint32_t count = logCount_[row];
        if (count == 0U) {
            return;
        }
        auto page = buffers_->GetLogPage(activeLogSlot_);
        CopyOutUB(CandidateKeyRow(row, activeEpoch_) + candidateCount_[row],
                  page[static_cast<uint64_t>(row) * kLogRowElements], count);
        CopyOutUB(CandidateIndexRow(row, activeEpoch_) + candidateCount_[row],
                  page[kLogIndexOffset + static_cast<uint64_t>(row) * kLogRowElements], count);
    }

    __aicore__ inline void FlushLogsIfNeeded()
    {
        bool flush = false;
        for (int32_t row = 0; row < kAivRows; ++row) {
            flush = flush || logCount_[row] > static_cast<uint32_t>(kFixTile);
        }
        if (!flush) {
            return;
        }
        RetireLogDma();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::LOG_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::LOG_STORE);
        for (int32_t row = 0; row < kAivRows; ++row) {
            SubmitLogFlush(row);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::LOG_STORE);
        for (int32_t row = 0; row < kAivRows; ++row) {
            candidateCount_[row] += logCount_[row];
            logCount_[row] = 0U;
        }
        activeLogSlot_ = 1U - activeLogSlot_;
        logDmaPending_ = true;
    }

    __aicore__ inline void FlushAllLogs()
    {
        RetireLogDma();
        bool any = false;
        for (int32_t row = 0; row < kAivRows; ++row) {
            any = any || logCount_[row] != 0U;
        }
        if (!any) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::LOG_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::LOG_STORE);
        for (int32_t row = 0; row < kAivRows; ++row) {
            SubmitLogFlush(row);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::LOG_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::LOG_STORE);
        for (int32_t row = 0; row < kAivRows; ++row) {
            candidateCount_[row] += logCount_[row];
            logCount_[row] = 0U;
        }
    }

    __aicore__ inline void SubmitMergeData(__gm__ uint32_t* previous, __gm__ uint32_t* candidate, uint32_t count)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(FastEvent::GM_REUSE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(FastEvent::GM_REUSE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        auto* merge = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetMerge().GetPhyAddr());
        CopyInUB(buffers_->GetMerge(), previous, kTopK);
        CopyInUB(buffers_->GetMerge()[kTopK], candidate, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
    }

    __aicore__ inline void FinishMergeLoad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
    }

    __aicore__ inline void LoadMergeData(__gm__ uint32_t* previous, __gm__ uint32_t* candidate, uint32_t count)
    {
        SubmitMergeData(previous, candidate, count);
        FinishMergeLoad();
    }

    __aicore__ inline void FinishStateStore()
    {
        if (!stateStorePending_) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        stateStorePending_ = false;
    }

    __aicore__ inline void BeginStateStore(int32_t row, bool key)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        CopyOutUB(key ? KeyStateRow(row) : IndexStateRow(row), buffers_->GetIndexScratch(), kTopK);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        stateStorePending_ = true;
    }

    __aicore__ inline void MergeRow(int32_t row, uint32_t count, uint32_t epoch)
    {
        if (count == 0U) {
            return;
        }
        auto* merge = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetMerge().GetPhyAddr());
        auto* scratch = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetIndexScratch().GetPhyAddr());
        LoadMergeData(KeyStateRow(row), CandidateKeyRow(row, epoch), count);
        const uint32_t length = kTopK + count;
        const uint32_t kth = vf::KthSortableKey(*buffers_, merge, length);
        vf::SelectStableTopkPositions(*buffers_, merge, length, kth);
        vf::GatherSelectedTopk(*buffers_, merge, scratch);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        CopyOutUB(KeyStateRow(row), buffers_->GetIndexScratch(), kTopK);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        LoadMergeData(IndexStateRow(row), CandidateIndexRow(row, epoch), count);
        vf::GatherSelectedTopk(*buffers_, merge, scratch);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FastEvent::STATE_STORE);
        CopyOutUB(IndexStateRow(row), buffers_->GetIndexScratch(), kTopK);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FastEvent::STATE_STORE);
        threshold_[row] = kth;
    }

    __aicore__ inline void MergeFinalRow(int32_t row)
    {
        const uint32_t count = candidateCount_[row];
        if (count == 0U) {
            return;
        }
        auto* merge = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetMerge().GetPhyAddr());
        auto* finalIndices = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetFinalIndices().GetPhyAddr());
        auto* scratch = reinterpret_cast<__ubuf__ uint32_t*>(buffers_->GetIndexScratch().GetPhyAddr());
        // Every compact-log write has retired. The key and index reads are
        // disjoint; publish separate completions so radix work can proceed
        // while the index prefetch is still in the MTE2 queue.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(FastEvent::STATE_LOAD);
        CopyInUB(buffers_->GetMerge(), KeyStateRow(row), kTopK);
        CopyInUB(buffers_->GetMerge()[kTopK], CandidateKeyRow(row, activeEpoch_), count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
        CopyInUB(buffers_->GetFinalIndices(), IndexStateRow(row), kTopK);
        CopyInUB(buffers_->GetFinalIndices()[kTopK], CandidateIndexRow(row, activeEpoch_), count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FastEvent::INDEX_PREFETCH);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FastEvent::STATE_LOAD);
        const uint32_t length = kTopK + count;
        const uint32_t kth = vf::KthSortableKey(*buffers_, merge, length);
        vf::SelectStableTopkPositions(*buffers_, merge, length, kth);
        FinishStateStore();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FastEvent::INDEX_PREFETCH);
        vf::GatherSelectedTopk(*buffers_, finalIndices, scratch);
        BeginStateStore(row, false);
        candidateCount_[row] = 0U;
    }

    __aicore__ inline void SeekNextFrozenRow()
    {
        while (mergeRowCursor_ < kAivRows && frozenCount_[mergeRowCursor_] == 0U) {
            ++mergeRowCursor_;
        }
        mergeStage_ = mergeRowCursor_ < kAivRows ? LOAD_KEYS : IDLE;
    }

    __aicore__ inline void DrainFrozenMerge()
    {
        while (mergeStage_ != IDLE) {
            ProgressFrozenMerge();
        }
        FinishStateStore();
    }

    FastVectorBuffers* buffers_ = nullptr;
    __gm__ uint32_t* stateKeys_;
    __gm__ uint32_t* stateIndices_;
    __gm__ uint32_t* paddingIndices_;
    uint32_t validQueryRows_ = kTileS1;
    __gm__ uint32_t* candidateKeys_;
    __gm__ uint32_t* candidateIndices_;
    uint64_t epochStride_;
    uint32_t subBlock_;
    uint32_t logCount_[kAivRows];
    uint32_t candidateCount_[kAivRows];
    uint32_t frozenCount_[kAivRows];
    uint32_t threshold_[kAivRows];
    uint32_t activeEpoch_;
    uint32_t frozenEpoch_;
    uint32_t activeLogSlot_;
    uint32_t mergeRowCursor_;
    uint32_t mergeStage_;
    uint32_t mergePrefix_;
    uint32_t outputTopK_ = kTopK;
    __gm__ SCORE_T* outputIndices_ = nullptr;
    __gm__ SCORE_T* outputValues_ = nullptr;
    uint32_t mergeRank_;
    uint32_t mergeStrict_;
    uint32_t mergeEqual_;
    uint32_t mergeEqualBegin_;
    bool logDmaPending_;
    bool stateStorePending_;
};

template <typename LIT>
class LightningIndexerV2FastServiceVector {
public:
    using W_T = float;
    using OUT_T = typename LIT::outputType;

    __aicore__ inline void InitBuffers(TPipe* pipe)
    {
        buffers_.InitBuffers(pipe);
    }

    __aicore__ inline void InitParams(const LIV2Fast::FastConstInfo& constInfo)
    {
        constInfo_ = constInfo;
    }

    __aicore__ inline void InitVecInputTensor(const GlobalTensor<W_T>& weightsGm,
                                              const GlobalTensor<OUT_T>& indiceOutGm, __gm__ float* valueOut = nullptr)
    {
        weightsGm_ = weightsGm;
        indiceOutGm_ = indiceOutGm;
        valueOut_ = valueOut;
    }

    __aicore__ inline void InitWorkspace(__gm__ uint8_t* workspace)
    {
        workspace_ = LIV2Fast::GetFastWorkspace(workspace, constInfo_);
    }

    __aicore__ inline void Process()
    {
        if (constInfo_.headCount == kHeads) {
            ProcessImpl<kHeads>();
        } else if (constInfo_.headCount == kSpecializedHeads) {
            ProcessImpl<kSpecializedHeads>();
        } else {
            ProcessImpl<0U>();
        }
    }

private:
    // Head count is a property of the sequence, independent of alignment
    // and sparse mode. Keep G24/G32 specialized; zero selects runtime heads.
    template <uint32_t HEADS>
    __aicore__ inline void ProcessImpl()
    {
        const auto& info = constInfo_;
        const uint32_t subBlock = info.subBlockIdx;
        if (subBlock >= kAivPerAic) {
            return;
        }
        AscendC::CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(
            static_cast<uint16_t>(FastEvent::QK_SLOT_0));
        AscendC::CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(
            static_cast<uint16_t>(FastEvent::QK_SLOT_1));
        // Reuse one batch-local state across Q tiles without extending its lifetime to the kernel.
        LightningIndexerV2StreamingTopK<LIT> topk;
        topk.InitBuffers(buffers_);
        topk.InitWorkspace(workspace_, subBlock, info.outputTopK);
        const uint32_t headCount = HEADS == 0U ? info.headCount : HEADS;
        const bool causal = info.sparseMode == kCausalMaskMode;
        if (!causal) {
            InitTileBounds(0U);
        }
        for (uint32_t tile = info.firstQueryTile; tile < info.queryTileEnd; ++tile) {
            tileRowBase_ = tile * kTileS1;
            const uint32_t validQueryRows = min(static_cast<uint32_t>(kTileS1), info.s1Size - tileRowBase_);
            queryHeadStride_ = LIV2Fast::GetQueryHeadStride(headCount, validQueryRows);
            // Set index and optional score output addresses for the current Q tile.
            SetQueryTileAddress(topk, validQueryRows);
            // Copy each query's G head weights from GM to UB for weighted score accumulation.
            LIV2Fast::vf::LoadTileWeights(weightsGm_.GetPhyAddr(static_cast<uint64_t>(tileRowBase_) * headCount),
                                          subBlock, buffers_.GetWeights(), headCount, validQueryRows);
            // Set the shared K scan bound and per-row visibility for causal and tail masking.
            if (causal) {
                InitTileBounds(tile);
            }
            if (validQueryRows == kTileS1) {
                ProcessQueryTile<HEADS, true>(topk);
            } else {
                ProcessQueryTile<HEADS, false>(topk);
            }
            topk.Finish();
        }
    }

    __aicore__ inline void SetQueryTileAddress(LightningIndexerV2StreamingTopK<LIT>& topk, uint32_t validQueryRows)
    {
        const uint64_t offset = static_cast<uint64_t>(tileRowBase_) * constInfo_.outputTopK;
        topk.BeginQueryTile(reinterpret_cast<__gm__ uint32_t*>(indiceOutGm_.GetPhyAddr(offset)),
                            valueOut_ != nullptr ? reinterpret_cast<__gm__ uint32_t*>(valueOut_) + offset : nullptr,
                            validQueryRows);
    }

    __aicore__ inline void InitTileBounds(uint32_t tile)
    {
        const auto& info = constInfo_;
        const bool causal = info.sparseMode == kCausalMaskMode;
        const uint32_t firstValidKey =
            causal ? min(info.s2Size - info.s1Size + tileRowBase_ + 1U, info.s2Size) : info.s2Size;
        fullScanSize_ = firstValidKey / kKTile * kKTile;
        scanSize_ = LIV2Fast::GetQueryTileScanSize(info, tile);
        if (fullScanSize_ < scanSize_) {
            for (uint32_t row = 0; row < kAivRows; ++row) {
                const uint32_t physicalRow =
                    (row / kQueryPairRows) * kQuerySlotRows + kQueryPairRows * info.subBlockIdx + row % kQueryPairRows;
                validKeys_[row] = causal ? min(firstValidKey + physicalRow, info.s2Size) : info.s2Size;
            }
        }
    }

    template <uint32_t HEADS, bool FULL_QUERY_TILE>
    __aicore__ inline void ProcessQueryTile(LightningIndexerV2StreamingTopK<LIT>& topk)
    {
        // All supported sequences have at least 2048 valid keys per row.
        // Interior K512 pairs use the same unmasked path in both sparse modes.
        const uint32_t fullScanSize = fullScanSize_;
        const uint32_t scanSize = scanSize_;
        uint64_t offset = 0;
        for (; offset < fullScanSize; offset += kKTile) {
            // Process K tiles that are fully visible to every query row.
            ProcessKeyTile<HEADS, FULL_QUERY_TILE, false>(topk, offset);
        }
        // Only the last one or two pairs can intersect a causal/physical edge.
        for (; offset < scanSize; offset += kKTile) {
            // Process the remaining K tiles with per-row visibility masks.
            ProcessKeyTile<HEADS, FULL_QUERY_TILE, true>(topk, offset);
        }
    }

    template <uint32_t HEADS, bool FULL_QUERY_TILE, bool MASKED>
    __aicore__ inline void ProcessKeyTile(LightningIndexerV2StreamingTopK<LIT>& topk, uint64_t offset)
    {
        const bool initial = offset < kTopK;
        // QK bank planes interleave K64 fragments from the two Q4 slots.
        // Byte ranges 0-255 and 256-511 hold one FP32 fragment from each slot;
        // ranges 512-767 and 768-1023 hold their next fragments.
        ProcessVec1<0, 0, HEADS, FULL_QUERY_TILE, MASKED>(topk, offset, initial); // First Q4 slot, first K256 half.
        ProcessVec1<1, 1, HEADS, FULL_QUERY_TILE, MASKED>(topk, offset, initial); // Second Q4 slot, first K256 half.
        if (!initial) {
            topk.CommitFilteredRows();
        }
        ProcessVec1<0, 0, HEADS, FULL_QUERY_TILE, MASKED>(topk, offset + kFixTile,
                                                          initial); // First Q4 slot, second K256 half.
        ProcessVec1<1, 1, HEADS, FULL_QUERY_TILE, MASKED>(topk, offset + kFixTile,
                                                          initial); // Second Q4 slot, second K256 half.
        if (!initial) {
            topk.CommitFilteredRows();
        }
    }

    // qBlock selects rows 0..3 or 4..7; each AIV handles two rows in that group.
    template <int32_t SLOT, int32_t Q_BLOCK, uint32_t HEADS, bool FULL_QUERY_TILE, bool MASKED>
    __aicore__ inline void ProcessVec1(LightningIndexerV2StreamingTopK<LIT>& topk, uint64_t offset, bool initial)
    {
        if constexpr (FULL_QUERY_TILE) {
            ProcessVec1Impl<SLOT, Q_BLOCK, HEADS, MASKED, false, false>(topk, offset, initial);
        } else {
            // Preserve the mainline's tail-row accumulation order. This
            // dispatch is absent from every complete Q tile's K loop.
            const uint32_t physicalRow =
                tileRowBase_ + Q_BLOCK * kQuerySlotRows + kQueryPairRows * constInfo_.subBlockIdx;
            const uint32_t tailRows = constInfo_.s1Size % kQuerySlotRows;
            // splitFirstRow/splitSecondRow select separate even/odd-head accumulation
            // for each query row. A false flag uses sequential head accumulation.
            if (tailRows == kQueryPairRows && physicalRow + kQueryPairRows == constInfo_.s1Size) {
                ProcessVec1Impl<SLOT, Q_BLOCK, HEADS, MASKED, true, true>(topk, offset, initial);
            } else if ((tailRows == 1U || tailRows == (kQuerySlotRows - 1U)) && physicalRow + 1U == constInfo_.s1Size) {
                ProcessVec1Impl<SLOT, Q_BLOCK, HEADS, MASKED, true, false>(topk, offset, initial);
            } else {
                ProcessVec1Impl<SLOT, Q_BLOCK, HEADS, MASKED, false, false>(topk, offset, initial);
            }
        }
    }

    // SplitRow preserves the mainline even/odd-head accumulation order for tail rows.
    template <int32_t SLOT, int32_t Q_BLOCK, uint32_t HEADS, bool MASKED, bool SPLIT_FIRST_ROW, bool SPLIT_SECOND_ROW>
    __aicore__ inline void ProcessVec1Impl(LightningIndexerV2StreamingTopK<LIT>& topk, uint64_t offset, bool initial)
    {
        AscendC::CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(static_cast<uint16_t>(SLOT));
        auto scoreLocal = buffers_.GetWrsScores(SLOT);
        auto weightLocal = buffers_.GetWeights(Q_BLOCK, SLOT);
        auto qkLocal = buffers_.GetQk(SLOT);
        if (!initial) {
            constexpr int32_t row = Q_BLOCK * kQueryPairRows;
            // Compute scores for this AIV's query pair and compact candidates into the UB log.
            // Compare FP32 sortable keys against the corresponding row thresholds.
            LIV2Fast::vf::WrsSortCompactPair<MASKED, SPLIT_FIRST_ROW, SPLIT_SECOND_ROW, HEADS>(
                reinterpret_cast<__ubuf__ float*>(qkLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(weightLocal.GetPhyAddr()), topk.LogKeyRow(row),
                topk.LogKeyRow(row + 1), topk.LogIndexRow(row), topk.LogIndexRow(row + 1),
                reinterpret_cast<__ubuf__ uint32_t*>(buffers_.GetCounts().GetPhyAddr()) + row, topk.GetThreshold(row),
                topk.GetThreshold(row + 1), offset, MASKED ? validKeys_[row] : 0U, MASKED ? validKeys_[row + 1] : 0U,
                HEADS == 0U ? constInfo_.headCount : HEADS, queryHeadStride_);
            AscendC::CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(static_cast<uint16_t>(SLOT));
            topk.ProgressFrozenMerge();
            return;
        }
        LIV2Fast::vf::RunBankPlaneWrs<SPLIT_FIRST_ROW, SPLIT_SECOND_ROW, HEADS>(
            reinterpret_cast<__ubuf__ float*>(qkLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(weightLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(scoreLocal.GetPhyAddr()), HEADS == 0U ? constInfo_.headCount : HEADS,
            queryHeadStride_);
        AscendC::CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(static_cast<uint16_t>(SLOT));
        LIV2Fast::vf::ConvertWrsScoresToKeys(reinterpret_cast<__ubuf__ float*>(scoreLocal.GetPhyAddr()),
                                             reinterpret_cast<__ubuf__ uint32_t*>(scoreLocal.GetPhyAddr()));
        topk.ConsumeInitialPair(Q_BLOCK * kQueryPairRows, scoreLocal.template ReinterpretCast<uint32_t>(), offset);
    }

    GlobalTensor<W_T> weightsGm_;
    GlobalTensor<OUT_T> indiceOutGm_;
    __gm__ float* valueOut_ = nullptr;
    LIV2Fast::FastConstInfo constInfo_{};
    LIV2Fast::FastVectorBuffers buffers_;
    LIV2Fast::FastWorkspace workspace_;
    uint32_t tileRowBase_ = 0;
    uint32_t queryHeadStride_ = kHeads;
    uint32_t validKeys_[kAivRows]{};
    uint32_t fullScanSize_ = 0;
    uint32_t scanSize_ = 0;
};

} // namespace LIV2Kernel
#endif
