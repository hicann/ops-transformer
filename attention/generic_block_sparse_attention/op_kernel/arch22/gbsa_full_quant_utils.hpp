/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GBSA_FULL_QUANT_UTILS_HPP
#define GBSA_FULL_QUANT_UTILS_HPP

#include <cstdint>

namespace GbsaKernelArch22::Int8 {

// Shared by host sizing, device addressing and the standalone contract tests.
struct FullQuantLayout {
    static constexpr uint32_t KIB = 1024;
    static constexpr uint32_t ROWS = 128;
    static constexpr uint32_t VECTOR_ROWS = 8;
    static constexpr uint32_t MAX_VECTOR_ROWS = ROWS / 2;
    static constexpr uint32_t PV_ROWS = 64;
    static constexpr uint32_t HEAD_DIM = 256;
    static constexpr uint32_t KV_TILE = 512;
    static constexpr uint32_t PAGE_TOKENS = 128;
    static constexpr uint32_t PAGES_PER_TILE = KV_TILE / PAGE_TOKENS;
    static constexpr uint32_t L1_KV_ROWS = 256;
    static constexpr uint32_t QK_COLUMNS = 256;
    static constexpr uint32_t QK_REDUCTION = 128;
    static constexpr uint32_t PV_REDUCTION = 256;
    static constexpr uint32_t STAGES = 3;
    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t DOUBLE_BUFFER = 2;

    static constexpr uint32_t Q_BYTES = ROWS * HEAD_DIM;
    static constexpr uint32_t KV_SLOT_BYTES = L1_KV_ROWS * HEAD_DIM;
    static constexpr uint32_t P_BYTES = ROWS * KV_TILE;
    static constexpr uint32_t S_BYTES = ROWS * KV_TILE * sizeof(int32_t);
    static constexpr uint32_t O_BYTES = ROWS * HEAD_DIM * sizeof(int32_t);
    static constexpr uint32_t L1_Q = 0;
    static constexpr uint32_t L1_K = L1_Q + Q_BYTES;
    static constexpr uint32_t L1_V = L1_K + DOUBLE_BUFFER * KV_SLOT_BYTES;
    static constexpr uint32_t L1_P = L1_V + DOUBLE_BUFFER * KV_SLOT_BYTES;
    static constexpr uint32_t L1_END = L1_P + STAGES * P_BYTES;
    static constexpr uint32_t L0A_Q = 0;
    static constexpr uint32_t L0A_P = Q_BYTES;
    static constexpr uint32_t L0A_END = L0A_P + PV_ROWS * KV_TILE;
    static constexpr uint32_t L0B_SLOT_BYTES = QK_REDUCTION * QK_COLUMNS;
    static constexpr uint32_t L0C_S_SLOT_BYTES = 32 * KIB;
    static constexpr uint32_t L0C_SHARED_SLOT_BYTES = 64 * KIB;
    static constexpr uint32_t L0C_O = DOUBLE_BUFFER * L0C_S_SLOT_BYTES;
    static constexpr uint32_t L0C_END = L0C_O + DOUBLE_BUFFER * L0C_S_SLOT_BYTES;
    static constexpr uint32_t WORKSPACE_BYTES = STAGES * (S_BYTES + P_BYTES + O_BYTES);
};

// Host tiling uses only FullQuantLayout. Sparse copy plans depend on Ascend C device types.
#if defined(__aicore__) || defined(__CCE_AICORE__)
struct Page {
    // One compact 128-column compute chunk, potentially spanning many sparse blocks.
    uint64_t sourceToken = 0;
    uint32_t firstIndex = 0;
    uint32_t endIndex = 0;
    uint32_t validTokens = 0;
    uint32_t planBegin = 0;
    uint32_t planEnd = 0;
    bool contiguous = true;
};

// Token spans preserve sparse-list order. Cube retains one plan per pipeline
// slot for QK/PV; each Vector builds the same mapping for its own Kscale copy.
struct CopySpan {
    uint64_t token;
    uint32_t length;
};

struct CopyPlan {
    CopySpan spans[FullQuantLayout::KV_TILE];
    __aicore__ inline void SetSpan(uint32_t index, uint64_t token, uint32_t length)
    {
        spans[index] = {token, length};
    }
    __aicore__ inline CopySpan GetSpan(uint32_t index) const
    {
        return spans[index];
    }
};

struct KvCopyRecord {
    uint64_t key;
    uint64_t value;
    uint16_t keyStride;
    uint16_t valueStride;
    uint16_t rows;
    uint16_t dstRow;
    uint16_t ndNum;
};

struct KvCopyPlan {
    KvCopyRecord records[FullQuantLayout::KV_TILE];
    uint32_t begin[FullQuantLayout::PAGES_PER_TILE];
    uint32_t end[FullQuantLayout::PAGES_PER_TILE];
};

// Like the FP16 plan, only retain the peer operand after issuing K.
struct VCopyRecord {
    uint64_t source;
    uint16_t stride;
    uint16_t rows;
    uint16_t dstRow;
    uint16_t ndNum;
};
static_assert(sizeof(VCopyRecord) == 16U);

template <uint32_t recordsPerChunk>
struct VCopyPlanStorage {
    static constexpr uint32_t RECORDS_PER_CHUNK = recordsPerChunk;
    VCopyRecord records[recordsPerChunk * FullQuantLayout::PAGES_PER_TILE];
    uint32_t begin[FullQuantLayout::PAGES_PER_TILE];
    uint32_t end[FullQuantLayout::PAGES_PER_TILE];
};
using VCopyPlan = VCopyPlanStorage<FullQuantLayout::PAGE_TOKENS>;
using CompactVCopyPlan = VCopyPlanStorage<FullQuantLayout::PAGE_TOKENS / 4U>;
static_assert(sizeof(CompactVCopyPlan) == 2080U);

struct Tile {
    Page pages[FullQuantLayout::PAGES_PER_TILE];
};
// INT8 PV reduces K in 32-element units. Capacity stays 512; execution is compact.
__aicore__ inline uint32_t TileColumns(const Tile& tile)
{
    uint32_t valid = 0;
    for (uint32_t i = 0; i < FullQuantLayout::PAGES_PER_TILE; ++i) {
        valid += tile.pages[i].validTokens;
    }
    return (valid + 31U) / 32U * 32U;
}
__aicore__ inline uint32_t HalfColumns(const Tile& tile, uint32_t half)
{
    const uint32_t columns = TileColumns(tile);
    const uint32_t begin = half * FullQuantLayout::L1_KV_ROWS;
    return columns <= begin ?
               0U :
               (columns - begin < FullQuantLayout::L1_KV_ROWS ? columns - begin : FullQuantLayout::L1_KV_ROWS);
}

struct SparseRun {
    uint64_t token = 0;
    uint32_t length = 0;
};

struct KvCopyWindow {
    uint32_t channels;
    uint32_t headOffset;
};

// The source window stays inside one token. Backward expansion uses complete
// C0s so this head stays aligned in L1; forward expansion can use partial C0s.
__aicore__ inline KvCopyWindow MakeKvCopyWindow(uint32_t dim, uint32_t heads, uint32_t head)
{
    // Keep narrow ND2NZ rows at up to two C0s; never widen D >= 64.
    const uint32_t capacity = dim < 64U ? 64U - dim : 0U;
    const uint32_t wanted = dim < capacity ? dim : capacity;
    const uint32_t left = head * dim / 32U;
    const uint32_t backward = (wanted / 32U < left ? wanted / 32U : left) * 32U;
    const uint32_t right = (heads - head - 1U) * dim;
    const uint32_t forward = wanted - backward < right ? wanted - backward : right;
    const uint32_t required = dim + backward;
    const uint32_t aligned = (required + forward) / 32U * 32U;
    // Avoid allocating an extra L1 C0 for surplus channels alone.
    return {required > aligned ? required : aligned, backward};
}

// Sparse block IDs are logical blockY units, never physical page IDs. Keep the
// same compact token order for K, V and Kscale. Only merge adjacent source spans;
// unlike FP16 pair-copy, reordering K/V alone would detach their token scales.
template <bool isPaged, bool hasResidualBlock, class IndexTensor>
class SparseSource {
public:
    __aicore__ inline SparseSource(IndexTensor indices, IndexTensor table, uint32_t blockY, uint32_t visibleEnd,
                                   uint32_t maxPages, uint32_t explicitCount = 0,
                                   uint32_t headDim = FullQuantLayout::HEAD_DIM, uint64_t tokenStride = 0,
                                   uint64_t headStride = 0, bool includeResidual = true)
        : indices_(indices),
          table_(table),
          blockY_(blockY),
          visibleEnd_(visibleEnd),
          maxPages_(maxPages),
          explicitCount_(explicitCount),
          headDim_(headDim),
          tokenStride_(tokenStride),
          headStride_(headStride == 0U ? headDim : headStride)
    {
        if constexpr (hasResidualBlock) {
            residualBlockId_ = visibleEnd_ / blockY_;
            residualSize_ = includeResidual ? visibleEnd_ % blockY_ : 0U;
        }
    }

    __aicore__ inline bool ReadBlock(int32_t id, SparseRun& run) const
    {
        const uint64_t token = static_cast<uint64_t>(id) * blockY_;
        if (id < 0 || token >= visibleEnd_ || (isPaged && token / FullQuantLayout::PAGE_TOKENS >= maxPages_)) {
            return false;
        }
        const uint32_t remaining = visibleEnd_ - token;
        run.length = remaining < blockY_ ? remaining : blockY_;
        if constexpr (isPaged) {
            const uint32_t logicalPage = token / FullQuantLayout::PAGE_TOKENS;
            if (logicalPage != cachedLogicalPage_) {
                cachedLogicalPage_ = logicalPage;
                cachedPhysicalPage_ = table_.GetValue(logicalPage);
            }
            run.token = static_cast<uint64_t>(cachedPhysicalPage_) * FullQuantLayout::PAGE_TOKENS +
                        token % FullQuantLayout::PAGE_TOKENS;
        } else {
            run.token = token;
        }
        return true;
    }

    __aicore__ inline IndexTensor GetIndexTensor() const
    {
        return indices_;
    }
    __aicore__ inline uint32_t ExplicitCount() const
    {
        return explicitCount_;
    }
    __aicore__ inline uint32_t ResidualId() const
    {
        return residualBlockId_;
    }
    __aicore__ inline int32_t PhysicalPage(uint32_t logical) const
    {
        if constexpr (isPaged) {
            return table_.GetValue(logical);
        }
        return static_cast<int32_t>(logical);
    }

    __aicore__ inline uint32_t CountValid(uint32_t count) const
    {
        if constexpr (hasResidualBlock) {
            // Mode 1 lists only full visible blocks; the partial tail is virtual.
            return count + static_cast<uint32_t>(residualSize_ != 0U);
        }
        // Sparse indices are strictly increasing by contract. Like FP16, the
        // last index proves that the entire prefix is visible in the common case.
        if (count != 0U) {
            const int32_t last = indices_.GetValue(count - 1U);
            const uint64_t token = static_cast<uint64_t>(last) * blockY_;
            if (last >= 0 && token < visibleEnd_ && (!isPaged || token / FullQuantLayout::PAGE_TOKENS < maxPages_)) {
                return count;
            }
        }
        uint32_t valid = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t id = indices_.GetValue(i);
            const uint64_t token = static_cast<uint64_t>(id) * blockY_;
            valid += id >= 0 && token < visibleEnd_ && (!isPaged || token / FullQuantLayout::PAGE_TOKENS < maxPages_);
        }
        return valid;
    }

    // Cube only needs chunk bounds before issuing K. Do not walk the same
    // sparse list here and again when generating DMA descriptors.
    __aicore__ inline Tile DescribeTile(uint32_t& cursor, uint32_t count) const
    {
        Tile tile{};
        for (auto& part : tile.pages) {
            part.firstIndex = cursor;
            const uint32_t room = FullQuantLayout::PAGE_TOKENS / blockY_;
            const uint32_t blocks = count - cursor < room ? count - cursor : room;
            cursor += blocks;
            part.endIndex = cursor;
            part.validTokens = blocks * blockY_;
            if constexpr (hasResidualBlock) {
                if (part.firstIndex <= explicitCount_ && cursor > explicitCount_) {
                    part.validTokens -= blockY_ - residualSize_;
                }
            } else if (blocks != 0U) {
                const uint64_t last = static_cast<uint64_t>(indices_.GetValue(cursor - 1U)) * blockY_;
                const uint32_t tail = visibleEnd_ - last;
                if (tail < blockY_) {
                    part.validTokens -= blockY_ - tail;
                }
            }
        }
        return tile;
    }

    // One pass resolves each selected block once, merges contiguous spans,
    // pairs forward sources, and immediately emits K plus the saved V record.
    // Preserve token order so the independently built Kscale plan stays aligned.
    template <class Emit>
    __aicore__ inline void ForEachKvRecord(const Page& part, uint64_t kPageStride, uint64_t vPageStride, uint32_t heads,
                                           uint32_t head, uint64_t kvStart, Emit emit) const
    {
        KvCopyRecord pending{};
        uint64_t pendingSpan = 0;
        const uint64_t tokenStride = TokenStride(heads);
        const uint64_t headOffset = static_cast<uint64_t>(head) * headStride_;
        uint32_t dstRow = 0;
        uint32_t explicitEnd = part.endIndex;
        if constexpr (hasResidualBlock) {
            explicitEnd = explicitEnd < explicitCount_ ? explicitEnd : explicitCount_;
        }
        // DescribeTile has already clipped the sorted visible prefix. Like
        // FP16's loader, resolve full-block addresses without rechecking every
        // token bound or reconstructing and then decoding a physical token.
        const uint32_t fullByTokens = part.firstIndex + part.validTokens / blockY_;
        const uint32_t fullEnd = explicitEnd < fullByTokens ? explicitEnd : fullByTokens;
        for (uint32_t i = part.firstIndex; i < fullEnd; ++i) {
            const uint32_t token = static_cast<uint32_t>(indices_.GetValue(i)) * blockY_;
            uint64_t key, value;
            if constexpr (isPaged) {
                const uint32_t logicalPage = token / FullQuantLayout::PAGE_TOKENS;
                if (logicalPage != cachedLogicalPage_) {
                    cachedLogicalPage_ = logicalPage;
                    cachedPhysicalPage_ = table_.GetValue(logicalPage);
                }
                const uint64_t inner = (token % FullQuantLayout::PAGE_TOKENS) * tokenStride + headOffset;
                key = static_cast<uint64_t>(cachedPhysicalPage_) * kPageStride + inner;
                value = static_cast<uint64_t>(cachedPhysicalPage_) * vPageStride + inner;
            } else {
                key = kvStart * heads * HeadDim() + token * tokenStride + headOffset;
                value = key;
            }
            AppendKvAddress(key, value, blockY_, pending, pendingSpan, dstRow, tokenStride, emit);
        }
        for (uint32_t i = fullEnd; i < explicitEnd; ++i) {
            SparseRun run;
            if (ReadBlock(indices_.GetValue(i), run)) {
                AppendKv(run, pending, pendingSpan, dstRow, kPageStride, vPageStride, heads, head, kvStart, emit);
            }
        }
        if constexpr (hasResidualBlock) {
            if (part.firstIndex <= explicitCount_ && part.endIndex > explicitCount_) {
                SparseRun run;
                if (ReadBlock(static_cast<int32_t>(residualBlockId_), run)) {
                    AppendKv(run, pending, pendingSpan, dstRow, kPageStride, vPageStride, heads, head, kvStart, emit);
                }
            }
        }
        if (pending.rows != 0U) {
            pending.dstRow = dstRow;
            emit(pending);
        }
    }

    __aicore__ inline bool Adjacent(const SparseRun& left, const SparseRun& right) const
    {
        return left.token + left.length == right.token &&
               (!isPaged || left.token / FullQuantLayout::PAGE_TOKENS == right.token / FullQuantLayout::PAGE_TOKENS);
    }

    template <class PlanType = CopyPlan>
    __aicore__ inline Tile NextTile(uint32_t& cursor, uint32_t count, PlanType* plan = nullptr) const
    {
        const uint32_t blocksPerChunk = FullQuantLayout::PAGE_TOKENS / blockY_;
        Tile tile{};
        uint32_t record = 0;
        for (uint32_t chunk = 0; chunk < FullQuantLayout::PAGES_PER_TILE; ++chunk) {
            auto& part = tile.pages[chunk];
            part.firstIndex = cursor;
            part.planBegin = record;
            uint32_t blocks = 0;
            SparseRun pending{};
            part.contiguous = true;
            uint32_t explicitEnd = count;
            if constexpr (hasResidualBlock) {
                explicitEnd = explicitEnd < explicitCount_ ? explicitEnd : explicitCount_;
            }
            while (cursor < explicitEnd && blocks < blocksPerChunk) {
                SparseRun run;
                const int32_t id = indices_.GetValue(cursor++);
                if constexpr (hasResidualBlock) {
                    // Mode 1 explicitly lists only complete visible blocks.
                    // The separately appended residual still uses ReadBlock.
                    ReadFullBlock(id, run);
                } else {
                    if (!ReadBlock(id, run)) {
                        continue;
                    }
                }
                AppendScale(run, pending, part, plan, record, blocks);
            }
            if constexpr (hasResidualBlock) {
                if (cursor == explicitCount_ && cursor < count && blocks < blocksPerChunk) {
                    SparseRun run;
                    ++cursor;
                    if (ReadBlock(static_cast<int32_t>(residualBlockId_), run)) {
                        AppendScale(run, pending, part, plan, record, blocks);
                    }
                }
            }
            if (plan != nullptr && pending.length != 0) {
                plan->SetSpan(record++, pending.token, pending.length);
            }
            part.endIndex = cursor;
            part.planEnd = record;
        }
        return tile;
    }

private:
    __aicore__ inline void ReadFullBlock(int32_t id, SparseRun& run) const
    {
        const uint64_t token = static_cast<uint64_t>(id) * blockY_;
        run.length = blockY_;
        if constexpr (isPaged) {
            const uint32_t logicalPage = token / FullQuantLayout::PAGE_TOKENS;
            if (logicalPage != cachedLogicalPage_) {
                cachedLogicalPage_ = logicalPage;
                cachedPhysicalPage_ = table_.GetValue(logicalPage);
            }
            run.token = static_cast<uint64_t>(cachedPhysicalPage_) * FullQuantLayout::PAGE_TOKENS +
                        token % FullQuantLayout::PAGE_TOKENS;
        } else {
            run.token = token;
        }
    }

    __aicore__ inline uint32_t HeadDim() const
    {
        return headDim_;
    }

    // Keep the tail in the same merge/pair stream as the explicit prefix.
    template <class Emit>
    __aicore__ inline void AppendKv(const SparseRun& run, KvCopyRecord& pending, uint64_t& pendingSpan,
                                    uint32_t& dstRow, uint64_t kPageStride, uint64_t vPageStride, uint32_t heads,
                                    uint32_t head, uint64_t kvStart, Emit& emit) const
    {
        const uint64_t key = KvOffset(run.token, kPageStride, heads, head, kvStart);
        const uint64_t value = KvOffset(run.token, vPageStride, heads, head, kvStart);
        AppendKvAddress(key, value, run.length, pending, pendingSpan, dstRow, TokenStride(heads), emit);
    }

    template <class Emit>
    __aicore__ inline void AppendKvAddress(uint64_t key, uint64_t value, uint32_t rows, KvCopyRecord& pending,
                                           uint64_t& pendingSpan, uint32_t& dstRow, uint64_t tokenStride,
                                           Emit& emit) const
    {
        if (pending.rows != 0U) {
            if (key == pending.key + pendingSpan && value == pending.value + pendingSpan) {
                pending.rows += rows;
                pendingSpan += static_cast<uint64_t>(rows) * tokenStride;
                return;
            }
            if (pending.rows == rows && key > pending.key && value > pending.value && key - pending.key < 65536U &&
                value - pending.value < 65536U && ((key - pending.key) & 31U) == 0U &&
                ((value - pending.value) & 31U) == 0U) {
                pending.keyStride = key - pending.key;
                pending.valueStride = value - pending.value;
                pending.ndNum = 2;
                pending.dstRow = dstRow;
                emit(pending);
                dstRow += pending.rows * 2U;
                pending.rows = 0;
                return;
            }
            pending.dstRow = dstRow;
            emit(pending);
            dstRow += pending.rows;
        }
        pending = {key, value, 0, 0, static_cast<uint16_t>(rows), 0, 1};
        pendingSpan = static_cast<uint64_t>(rows) * tokenStride;
    }

    template <class PlanType>
    __aicore__ inline void AppendScale(const SparseRun& run, SparseRun& pending, Page& part, PlanType* plan,
                                       uint32_t& record, uint32_t& blocks) const
    {
        if (blocks == 0) {
            part.sourceToken = run.token;
        }
        if (pending.length == 0) {
            pending = run;
        } else if (Adjacent(pending, run)) {
            pending.length += run.length;
        } else {
            part.contiguous = false;
            if (plan != nullptr) {
                plan->SetSpan(record++, pending.token, pending.length);
            }
            pending = run;
        }
        part.validTokens += run.length;
        ++blocks;
    }

    __aicore__ inline uint64_t KvOffset(uint64_t token, uint64_t pageStride, uint32_t heads, uint32_t head,
                                        uint64_t kvStart) const
    {
        if constexpr (isPaged) {
            return token / FullQuantLayout::PAGE_TOKENS * pageStride +
                   token % FullQuantLayout::PAGE_TOKENS * TokenStride(heads) + head * headStride_;
        }
        return kvStart * heads * HeadDim() + token * TokenStride(heads) + head * headStride_;
    }

    __aicore__ inline uint64_t TokenStride(uint32_t heads) const
    {
        return tokenStride_ == 0U ? static_cast<uint64_t>(heads) * HeadDim() : tokenStride_;
    }

    IndexTensor indices_, table_;
    uint32_t blockY_, visibleEnd_, maxPages_, explicitCount_, headDim_;
    uint64_t tokenStride_, headStride_;
    uint32_t residualBlockId_ = 0, residualSize_ = 0;
    mutable uint32_t cachedLogicalPage_ = UINT32_MAX;
    mutable int32_t cachedPhysicalPage_ = 0;
};

#endif

} // namespace GbsaKernelArch22::Int8
#endif
