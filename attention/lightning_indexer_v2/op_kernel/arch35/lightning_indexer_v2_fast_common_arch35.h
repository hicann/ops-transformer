/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_FAST_COMMON_ARCH35_H
#define LIGHTNING_INDEXER_V2_FAST_COMMON_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lightning_indexer_v2_common_arch35.h"

namespace LIV2Fast {

inline constexpr int32_t kTileS1 = 8;
inline constexpr int32_t kHeads = 32;
inline constexpr int32_t kHeadDim = 128;
inline constexpr int32_t kQRows = kTileS1 * kHeads;
// Maximum Q4 slot: 4 queries * 32 heads.
inline constexpr int32_t kQHalfRows = 128;
// Scan K512 tiles; Fixpipe/vector consume K256 portions.
inline constexpr int32_t kKTile = 512;
inline constexpr int32_t kKSubtile = 128;
inline constexpr int32_t kFixTile = 256;
inline constexpr int32_t kTopK = 2048;
inline constexpr int32_t kCandidateCapacity = 4096;
inline constexpr int32_t kCandidateEpochs = 2;
inline constexpr int32_t kAivRows = 4;
inline constexpr int32_t kDoubleBuffer = 2;
inline constexpr int32_t kQueryPairRows = 2;
inline constexpr int32_t kRadixBins = 256;
inline constexpr int32_t kRadixMostSignificantByte = 3;
inline constexpr int32_t kRadixSecondByte = 2;
inline constexpr int32_t kWrsFragment2 = 2;
inline constexpr int32_t kWrsFragment3 = 3;

// Cube NZ matrices use 16-element inner blocks; a Q8 tile has two Q4 slots.
inline constexpr uint32_t kNzBlockElements = 16U;
inline constexpr uint32_t kQuerySlotRows = 4U;
inline constexpr uint32_t kCausalMaskMode = 3U;
// One 256-byte vector register holds 64 FP32/uint32_t lanes.
inline constexpr uint32_t kFloatRegLanes = 64U;
inline constexpr uint64_t kWrsFragment1Offset = kFloatRegLanes;
inline constexpr uint64_t kWrsFragment2Offset = 2U * kFloatRegLanes;
inline constexpr uint64_t kWrsFragment3Offset = 3U * kFloatRegLanes;
// WRS bank planes interleave pairs of heads; offsets below count float elements.
inline constexpr uint32_t kWrsHeadStride = 128U;
inline constexpr uint32_t kWrsPlaneStride = kAivRows * kHeads * kFloatRegLanes;
inline constexpr uint64_t kWrsPlane2Offset = 2U * kWrsPlaneStride;
inline constexpr uint64_t kWrsPlane3Offset = 3U * kWrsPlaneStride;
inline constexpr uint32_t kWeightPairElements = kQueryPairRows * kHeads;
inline constexpr uint32_t kWeightSlotElements = 2U * kWeightPairElements;
inline constexpr uint32_t kLogRowElements = kQueryPairRows * kFixTile;
inline constexpr uint64_t kLogIndexOffset = kAivRows * kLogRowElements;
inline constexpr uint32_t kCandidateFreezeThreshold = kCandidateCapacity - kLogRowElements;
inline constexpr uint32_t kRadixLastBin = kRadixBins - 1U;
inline constexpr uint32_t kBitsPerByte = 8U;
inline constexpr uint32_t kRadixKeysPerIteration = 256U;
inline constexpr uint32_t kRadixHalfwordsPerIteration = 2U * kRadixKeysPerIteration;
inline constexpr uint32_t kSelectionChunkElements = 256U;
inline constexpr uint32_t kUint32SignBit = 0x80000000U;
inline constexpr uint32_t kUint32AllBits = 0xffffffffU;

// Each AIC has two companion AIVs, each owning two query pairs in a Q8 tile.
inline constexpr uint32_t kAivPerAic = 2U;
inline constexpr uint32_t kSpecializedHeads = 24U;
inline constexpr uint32_t kHeadParityCount = 2U;
inline constexpr uint32_t kKeyIndexSections = 2U;
// One K256 Fixpipe result consists of four K64 fragments. Keep all dependent
// widths and strides derived from this transfer geometry, in element units.
inline constexpr uint32_t kFixpipeFragmentKeys = 64U;
inline constexpr int32_t kWrsFragmentCount = kFixTile / kFixpipeFragmentKeys;
inline constexpr int32_t kWrsPairFragmentCount = kQueryPairRows * kWrsFragmentCount;
inline constexpr uint32_t kKSubtilesPerFixTile = kFixTile / kKSubtile;
inline constexpr uint32_t kFixpipeSourceStrideFactor = kFixpipeFragmentKeys / kNzBlockElements;
// Radix processes four bytes per sortable FP32 key and two 128-bin histogram halves.
inline constexpr int32_t kRadixKeyBytes = sizeof(uint32_t);
inline constexpr uint32_t kRadixHalfBins = kRadixBins / 2U;
inline constexpr uint32_t kRadixVectorGroups = kRadixBins / kFloatRegLanes;
// Capacity limits for this Arch35 pipeline. KiB values are converted to bytes.
inline constexpr uint32_t kL1CapacityBytes = 512U * 1024U;
inline constexpr uint32_t kL0cCapacityBytes = 256U * 1024U;

// Pack complete Q8 tiles with the actual head count. Only the complete
// Cube matrix is aligned to 16 rows; tail tiles retain the established layout.
__aicore__ inline uint32_t GetQueryHeadStride(uint32_t heads, uint32_t validQueries)
{
    return validQueries == kTileS1 ? heads : static_cast<uint32_t>(kHeads);
}

// Event IDs are scoped by HardEvent. Query/key ping-pong slots share the
// cube protocol; vector state, log and scalar handoffs use separate roles.
struct FastEvent {
    static constexpr uint16_t QUERY_SLOT_0 = 0;
    static constexpr uint16_t QUERY_SLOT_1 = 1;
    static constexpr uint16_t KEY_SLOT_0 = 2;
    static constexpr uint16_t KEY_SLOT_1 = 3;
    static constexpr uint16_t ACCUMULATOR_SLOT_0 = 0;
    static constexpr uint16_t ACCUMULATOR_SLOT_1 = 1;
    static constexpr uint16_t QK_SLOT_0 = 0;
    static constexpr uint16_t QK_SLOT_1 = 1;
    static constexpr uint16_t COMPANION_AIV_OFFSET = 16;
    static constexpr uint16_t WEIGHTS_READY = 0;
    static constexpr uint16_t INDEX_PREFETCH = 3;
    static constexpr uint16_t STATE_LOAD = 4;
    static constexpr uint16_t STATE_STORE = 5;
    static constexpr uint16_t LOG_STORE = 6;
    static constexpr uint16_t GM_REUSE = 7;
    static constexpr uint16_t SCALAR_READ = 7;
};

// Per-core fast-path parameters shared by the kernel and its services.
struct FastConstInfo {
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t subBlockIdx;
    uint32_t sparseMode;
    uint32_t outputTopK;
    uint32_t headCount;
    uint32_t firstQueryTile;
    uint32_t queryTileEnd;
};

// Per-core views into the shared, fixed-size streaming workspace.
struct FastWorkspace {
    __gm__ uint32_t* stateKeys;
    __gm__ uint32_t* candidateKeys;
    __gm__ uint32_t* candidateIndices;
    __gm__ uint32_t* paddingIndices;
    uint64_t epochStride;
};

__aicore__ inline FastWorkspace GetFastWorkspace(__gm__ uint8_t* workspace, const FastConstInfo& info)
{
    const uint64_t rows = static_cast<uint64_t>(info.blockNum) * kTileS1;
    const uint64_t rowBase = static_cast<uint64_t>(info.blockIdx) * kTileS1;
    const uint64_t stateWords = rows * kTopK;
    const uint64_t candidateWords = rows * kCandidateCapacity * kCandidateEpochs;
    auto* base = reinterpret_cast<__gm__ uint32_t*>(workspace);
    // GM sections in uint32_t words: state keys, candidate keys, candidate indices,
    // padded state indices. Each candidate section holds two epochs.
    return {base + rowBase * kTopK, base + stateWords + rowBase * kCandidateCapacity,
            base + stateWords + candidateWords + rowBase * kCandidateCapacity,
            base + stateWords + kKeyIndexSections * candidateWords + rowBase * kTopK, rows * kCandidateCapacity};
}

// Skip only K512 pairs fully masked for every row of this Q tile.
__aicore__ inline uint32_t GetQueryTileScanSize(const FastConstInfo& info, uint32_t tile)
{
    if (info.sparseMode != kCausalMaskMode) {
        return info.s2Size;
    }
    const uint32_t queryEnd = (tile + 1U) * kTileS1 < info.s1Size ? (tile + 1U) * kTileS1 : info.s1Size;
    // Bottom-right causal mask: row r sees K-Q+r+1 keys. The last valid row
    // sets the shared tile bound, which is then rounded up to a K512 scan unit.
    const uint32_t maskedKeys = info.s1Size - queryEnd;
    const uint32_t keys = info.s2Size > maskedKeys ? info.s2Size - maskedKeys : 0U;
    return (keys + kKTile - 1U) / kKTile * kKTile;
}

// The fixed UB bank-plane layout reserves Q8 * G32 = 256 matrix rows.
static_assert(kQRows == 256);
static_assert(kKTile % kFixTile == 0);

// FP16 and BF16 share the same 16-bit matrix tiles and ping-pong slots.
inline constexpr uint32_t kQl1Bytes = kQHalfRows * kHeadDim * sizeof(uint16_t);
inline constexpr uint32_t kKl1Bytes = kKTile * kHeadDim * sizeof(uint16_t);
inline constexpr uint32_t kL0bBytes = kKSubtile * kHeadDim * sizeof(uint16_t);
inline constexpr uint32_t kL0cBytes = kQHalfRows * kFixTile * sizeof(float);
inline constexpr uint32_t kQkBankPlaneBytes = kAivRows * kHeads * kFixTile * sizeof(float);
inline constexpr uint32_t kQkOneAivPayloadBytes = kQkBankPlaneBytes / kAivPerAic;
static_assert(kDoubleBuffer * kQl1Bytes + kDoubleBuffer * kKl1Bytes <= kL1CapacityBytes);
static_assert(kDoubleBuffer * kL0cBytes <= kL0cCapacityBytes);

// The buffers retain the current bank placement and phase-specific reuse.
// Sizes describe allocations; consumers only use LocalTensor slices.
class FastVectorBuffers {
public:
    static constexpr uint32_t LOCAL_ROWS = kTileS1 / kAivPerAic;
    static constexpr uint32_t REG_LANES = kFloatRegLanes;
    static constexpr uint32_t LOG_ROW_CAPACITY = kQueryPairRows * kFixTile;
    static constexpr uint32_t POSITION_CAPACITY = kTopK + kFixTile;
    static constexpr uint32_t MERGE_CAPACITY = kTopK + kCandidateCapacity;
    static constexpr uint32_t QK_BYTES = LOCAL_ROWS * kHeads * kFixTile * sizeof(float);
    static constexpr uint32_t LOG_BYTES = LOCAL_ROWS * LOG_ROW_CAPACITY * kKeyIndexSections * sizeof(uint32_t);
    // Preserve the existing gap before the merge buffer and the WRS slot gap.
    static constexpr uint32_t POSITION_BANK_PADDING = 6400;
    static constexpr uint64_t NEXT_RANK_OFFSET = 48;

    __aicore__ inline void InitBuffers(TPipe* pipe)
    {
        pipe->InitBuffer(qkBuf_, QK_BYTES);
        pipe->InitBuffer(weightBuf_, kDoubleBuffer * LOCAL_ROWS * kHeads * sizeof(float));
        pipe->InitBuffer(logBuf_, LOG_BYTES);
        pipe->InitBuffer(histogramBuf_, kDoubleBuffer * kRadixBins * sizeof(uint32_t));
        pipe->InitBuffer(positionBuf_, POSITION_CAPACITY * sizeof(uint32_t) + POSITION_BANK_PADDING);
        pipe->InitBuffer(mergeBuf_, MERGE_CAPACITY * sizeof(uint32_t));
        pipe->InitBuffer(indexBuf_, kTopK * sizeof(uint32_t));
        pipe->InitBuffer(countBuf_, kDoubleBuffer * REG_LANES * sizeof(uint32_t));
        // Log page 1 is dead before final indices use this allocation.
        pipe->InitBuffer(logAndFinalBuf_, (REG_LANES + MERGE_CAPACITY) * sizeof(uint32_t));
        pipe->InitBuffer(score0Buf_, (kQueryPairRows * kFixTile + REG_LANES) * sizeof(float));
        pipe->InitBuffer(score1Buf_, kQueryPairRows * kFixTile * sizeof(float));
        pipe->InitBuffer(weightScratchBuf_, REG_LANES * sizeof(float));

        qkLocal_ = qkBuf_.Get<float>();
        weightLocal_ = weightBuf_.Get<float>();
        logLocal_[0] = logBuf_.Get<uint32_t>();
        logLocal_[1] = logAndFinalBuf_.Get<uint32_t>();
        finalIndexLocal_ = logLocal_[1][REG_LANES];
        histogramLocal_ = histogramBuf_.Get<uint32_t>();
        positionLocal_ = positionBuf_.Get<uint32_t>();
        mergeLocal_ = mergeBuf_.Get<uint32_t>();
        indexLocal_ = indexBuf_.Get<uint32_t>();
        countLocal_ = countBuf_.Get<uint32_t>();
        scoreLocal_[0] = score0Buf_.Get<float>();
        scoreLocal_[1] = score1Buf_.Get<float>();
        weightScratchLocal_ = weightScratchBuf_.Get<float>();
    }

    __aicore__ inline LocalTensor<float> GetQk(uint32_t slot) const
    {
        return qkLocal_[slot * REG_LANES];
    }
    __aicore__ inline LocalTensor<float> GetWeights(uint32_t qBlock = 0, uint32_t slot = 0) const
    {
        return weightLocal_[qBlock * kDoubleBuffer * REG_LANES + slot * REG_LANES];
    }
    __aicore__ inline LocalTensor<uint32_t> GetLogPage(uint32_t slot) const
    {
        return logLocal_[slot];
    }
    __aicore__ inline LocalTensor<uint32_t> GetHistogram() const
    {
        return histogramLocal_;
    }
    __aicore__ inline LocalTensor<uint32_t> GetPositions() const
    {
        return positionLocal_;
    }
    __aicore__ inline LocalTensor<uint32_t> GetMerge() const
    {
        return mergeLocal_;
    }
    __aicore__ inline LocalTensor<uint32_t> GetIndexScratch() const
    {
        return indexLocal_;
    }
    __aicore__ inline LocalTensor<uint32_t> GetCounts() const
    {
        return countLocal_;
    }
    __aicore__ inline LocalTensor<uint32_t> GetNextRank() const
    {
        return countLocal_[NEXT_RANK_OFFSET];
    }
    __aicore__ inline LocalTensor<uint32_t> GetFinalIndices() const
    {
        return finalIndexLocal_;
    }
    __aicore__ inline LocalTensor<float> GetWrsScores(uint32_t slot) const
    {
        return scoreLocal_[slot];
    }
    __aicore__ inline LocalTensor<float> GetWeightScratch() const
    {
        return weightScratchLocal_;
    }

private:
    TBuf<TPosition::VECCALC> qkBuf_;
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> logBuf_;
    TBuf<TPosition::VECCALC> histogramBuf_;
    TBuf<TPosition::VECCALC> positionBuf_;
    TBuf<TPosition::VECCALC> mergeBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> countBuf_;
    TBuf<TPosition::VECCALC> logAndFinalBuf_;
    TBuf<TPosition::VECCALC> score0Buf_;
    TBuf<TPosition::VECCALC> score1Buf_;
    TBuf<TPosition::VECCALC> weightScratchBuf_;
    LocalTensor<float> qkLocal_, weightLocal_, scoreLocal_[kDoubleBuffer];
    LocalTensor<uint32_t> logLocal_[kDoubleBuffer], histogramLocal_, positionLocal_, mergeLocal_;
    LocalTensor<uint32_t> indexLocal_, countLocal_, finalIndexLocal_;
    LocalTensor<float> weightScratchLocal_;
};

// DMA consumes the LocalTensor views allocated by the vector service.
template <typename T>
__aicore__ inline void CopyInUB(const LocalTensor<T>& destination, __gm__ T* source, uint32_t count)
{
    if (count == 0) {
        return;
    }
    GlobalTensor<T> global;
    global.SetGlobalBuffer(source);
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    DataCopyPad(destination, global, params, padding);
}

template <typename T>
__aicore__ inline void CopyOutUB(__gm__ T* destination, const LocalTensor<T>& source, uint32_t count)
{
    if (count == 0) {
        return;
    }
    GlobalTensor<T> global;
    global.SetGlobalBuffer(destination);
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(global, source, params);
}

} // namespace LIV2Fast

namespace LIV2Kernel {

struct FastSequenceInfo {
    uint32_t querySize;
    uint32_t keySize;
    uint64_t queryOffset;
    uint64_t keyOffset;
};

template <typename LIT>
class LightningIndexerV2FastSequencePlan {
public:
    __aicore__ inline void Init(const LIV2TilingData* tiling, __gm__ uint8_t* queryLengths, __gm__ uint8_t* keyLengths,
                                __gm__ uint8_t* usedKeyLengths)
    {
        batchSize_ = tiling->bSize;
        querySize_ = tiling->s1Size;
        keySize_ = tiling->s2Size;
        hasUsedKey_ = usedKeyLengths != nullptr;
        if (hasUsedKey_) {
            usedKeyLengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(usedKeyLengths), batchSize_);
        }
        if constexpr (LIT::layout == LIV2Common::LI_V2_LAYOUT::TND) {
            queryLengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryLengths), batchSize_ + 1U);
        }
        if constexpr (LIT::keyLayout == LIV2Common::LI_V2_LAYOUT::TND) {
            keyLengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(keyLengths), batchSize_ + 1U);
        }
    }

    __aicore__ inline uint32_t GetBatchSize() const
    {
        return batchSize_;
    }

    __aicore__ inline FastSequenceInfo GetSequence(uint32_t batch) const
    {
        FastSequenceInfo info;
        if constexpr (LIT::layout == LIV2Common::LI_V2_LAYOUT::TND) {
            info.queryOffset = queryLengths_.GetValue(batch);
            info.querySize = queryLengths_.GetValue(batch + 1) - info.queryOffset;
        } else {
            info.queryOffset = static_cast<uint64_t>(batch) * querySize_;
            info.querySize = querySize_;
        }
        if constexpr (LIT::pageAttention) {
            info.keyOffset = 0;
            info.keySize = usedKeyLengths_.GetValue(batch);
        } else if constexpr (LIT::keyLayout == LIV2Common::LI_V2_LAYOUT::TND) {
            info.keyOffset = keyLengths_.GetValue(batch);
            info.keySize = keyLengths_.GetValue(batch + 1) - info.keyOffset;
        } else {
            info.keyOffset = static_cast<uint64_t>(batch) * keySize_;
            info.keySize = keySize_;
        }
        if (hasUsedKey_) {
            info.keySize = usedKeyLengths_.GetValue(batch);
        }
        return info;
    }

private:
    GlobalTensor<int32_t> queryLengths_, keyLengths_, usedKeyLengths_;
    bool hasUsedKey_ = false;
    uint32_t batchSize_, querySize_, keySize_;
};

} // namespace LIV2Kernel
#endif
