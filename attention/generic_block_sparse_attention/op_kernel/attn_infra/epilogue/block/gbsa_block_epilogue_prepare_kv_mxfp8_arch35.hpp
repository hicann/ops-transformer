/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_PREPARE_KV_MXFP8_ARCH35_HPP
#define GBSA_BLOCK_EPILOGUE_PREPARE_KV_MXFP8_ARCH35_HPP

#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/epilogue/gbsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_utils.hpp"
#include "../../../arch35/gbsa_layout_utils.hpp"
#include "../../gemm/tile_common/gbsa_build_kv_records_mxfp8.hpp"

namespace NpuArch::Epilogue::Block {

template <bool PAGED_CACHE_FLAG_, bool RESIDUAL_BLOCK_FLAG_>
class PrepareMxfp8FullOffsets {
public:
    static constexpr uint32_t TOKEN_BYTES_ = Gemm::Tile::RUNTIME_TOKEN_BYTES;
    using ArchTag = Arch::AtlasA5;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool RESIDUAL_BLOCK_FLAG = RESIDUAL_BLOCK_FLAG_;
    // Preserve the source descriptor policy's pair-alignment rule. Runtime
    // token bytes belong to this producer, not the shared nonquantized policy.
    static constexpr uint32_t COPY_ALIGNMENT = 16U;

    __aicore__ inline PrepareMxfp8FullOffsets(Arch::Resource<ArchTag>& resource, UBufTileHelper& uBufTileHelper,
                                              uint32_t tileSize, uint32_t l1TileN, uint32_t blockY, uint32_t pageSize,
                                              uint64_t tokenStride, uint64_t kPageStride, uint64_t vPageStride,
                                              uint32_t runtimeTokenBytes = 0U)
        : tileSize_(tileSize),
          l1TileN_(l1TileN),
          blockY_(blockY),
          pageSize_(pageSize),
          tokenStride_(tokenStride),
          kPageStride_(kPageStride),
          vPageStride_(vPageStride),
          tokenBytes_(runtimeTokenBytes)
    {
        // The mask reservation is unused by the current sparse softmax path.
        groupWords_ = GbsaKernelArch35::Mxfp8Descriptor::GroupWords(l1TileN_, blockY_, PAGED_CACHE_FLAG);
        recordsBufMask_ = uBufTileHelper.kvOffsetUbBufNum - 1U;
        for (uint32_t i = 0; i < uBufTileHelper.kvOffsetUbBufNum; ++i) {
            recordsBuf_[i] = resource.ubBuf.template GetBufferByByte<uint64_t>(uBufTileHelper.maskStartOffset +
                                                                               i * groupWords_ * sizeof(uint64_t));
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<uint64_t> info,
                                      const __gm__ int32_t* __restrict__ sparseIdx,
                                      const __gm__ int32_t* __restrict__ blockTable, uint32_t tileIdx,
                                      uint32_t tileTokens, uint32_t explicitCount, uint32_t residualId)
    {
        Gemm::Tile::KvRecordsOnly consumer;
        (*this)(info, sparseIdx, blockTable, tileIdx, tileTokens, explicitCount, residualId, consumer);
    }

    template <class Consumer>
    __aicore__ inline void operator()(AscendC::GlobalTensor<uint64_t> info,
                                      const __gm__ int32_t* __restrict__ sparseIdx,
                                      const __gm__ int32_t* __restrict__ blockTable, uint32_t tileIdx,
                                      uint32_t tileTokens, uint32_t explicitCount, uint32_t residualId,
                                      Consumer& consumer)
    {
        // Meta packs (sourceStride << 32) | (length << 1) | (ndNum - 1).
        // MXFP8 requires equal K/V shapes and contiguous PA page strides, so
        // both matrices share [offset, meta]. Non-quantized PA still permits
        // distinct page strides and retains [K offset, V offset, K meta, V meta].
        // Keep groupWords_ and the reserved workspace/UB slot sizes unchanged.
        constexpr bool compactPaged = PAGED_CACHE_FLAG && TOKEN_BYTES_ != 0U;
        constexpr uint32_t recordWords = PAGED_CACHE_FLAG && !compactPaged ?
                                             GbsaKernelArch35::Mxfp8Descriptor::CAPACITY_RECORD_WORDS :
                                             GbsaKernelArch35::Mxfp8Descriptor::RECORD_WORDS;
        const uint32_t groupNum = (tileTokens + l1TileN_ - 1U) / l1TileN_;
        for (uint32_t groupIdx = AscendC::GetSubBlockIdx(); groupIdx < groupNum; groupIdx += 2U) {
            // Alternate across actual groups, including groups in successive tiles/tasks.
            // Only wait when reusing this half; the other half may still be copying.
            const auto eventId = static_cast<event_t>(EVENT_ID5 + recordsBufIdx_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
            auto records_ = recordsBuf_[recordsBufIdx_];
            // This UB region is private to V0. Events 5/6 protect each half
            // explicitly; direct stores avoid a conservative tensor-wide
            // auto-sync dependency on the other half's outstanding MTE3 copy.
            auto records = reinterpret_cast<__ubuf__ uint64_t*>(records_.GetPhyAddr());
            auto groupInfo = info[groupIdx * groupWords_];
            const uint32_t groupBegin = groupIdx * l1TileN_;
            const uint32_t groupTokens = min(l1TileN_, tileTokens - groupBegin);
            const uint32_t compactBegin = tileIdx * tileSize_ + groupBegin;
            consumer.BeginGroup(groupIdx, groupTokens);
            bool fullRecords = false;
            uint32_t inner = compactBegin % blockY_;
            uint32_t count = 0;
            uint32_t dealtLenAccum = 0;
            uint32_t previousLen = 0;
            uint32_t previousPage = 0;
            uint32_t previousPageEnd = 0;
            uint64_t previousOffset = 0;
            uint64_t previousVOffset = 0;
            if constexpr (!PAGED_CACHE_FLAG && TOKEN_BYTES_ != 0) {
                const uint32_t firstBlock = compactBegin / blockY_;
                const uint32_t blocks = groupTokens / blockY_;
                if (tokenStride_ == (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_) &&
                    inner == 0 && blocks * blockY_ == groupTokens && firstBlock + blocks <= explicitCount) {
                    if (blockY_ == 4U) {
                        count = Gemm::Tile::BuildFullKvRecordsRuntime<4U>(
                            sparseIdx + firstBlock, records, groupTokens / 4U, 4U, consumer,
                            (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_));
                    } else if (blockY_ == 8U) {
                        count = Gemm::Tile::BuildFullKvRecordsRuntime<8U>(
                            sparseIdx + firstBlock, records, groupTokens / 8U, 8U, consumer,
                            (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_));
                    } else if (blockY_ == 16U) {
                        count = Gemm::Tile::BuildFullKvRecordsRuntime<16U>(
                            sparseIdx + firstBlock, records, groupTokens / 16U, 16U, consumer,
                            (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_));
                    } else {
                        count = Gemm::Tile::BuildFullKvRecordsRuntime<>(
                            sparseIdx + firstBlock, records, blocks, blockY_, consumer,
                            (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_));
                    }
                    fullRecords = true;
                    dealtLenAccum = groupTokens;
                }
            }
            if (compactPaged &&
                ((TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_) == 96U ||
                 (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_) == 128U ||
                 (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_) == 192U ||
                 (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_) == 256U)) {
                const uint32_t firstBlock = compactBegin / blockY_;
                const uint32_t blocks = groupTokens / blockY_;
                if (pageSize_ == 128U && (blockY_ == 2U || blockY_ == 4U || blockY_ == 8U || blockY_ == 16U) &&
                    inner == 0U && blocks * blockY_ == groupTokens && firstBlock + blocks <= explicitCount) {
                    if (tokenStride_ ==
                            (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_) &&
                        blockY_ == 4U) {
                        count = Gemm::Tile::BuildFullPagedKvRecords<128U, 4U>(sparseIdx + firstBlock, blockTable,
                                                                              records, groupTokens / 4U, 4U,
                                                                              tokenStride_, kPageStride_, consumer);
                    } else if (tokenStride_ ==
                                   (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_) &&
                               blockY_ == 8U) {
                        count = Gemm::Tile::BuildFullPagedKvRecords<128U, 8U>(sparseIdx + firstBlock, blockTable,
                                                                              records, groupTokens / 8U, 8U,
                                                                              tokenStride_, kPageStride_, consumer);
                    } else if (tokenStride_ ==
                                   (TOKEN_BYTES_ == Gemm::Tile::RUNTIME_TOKEN_BYTES ? tokenBytes_ : TOKEN_BYTES_) &&
                               blockY_ == 16U) {
                        count = Gemm::Tile::BuildFullPagedKvRecords<128U, 16U>(sparseIdx + firstBlock, blockTable,
                                                                               records, groupTokens / 16U, 16U,
                                                                               tokenStride_, kPageStride_, consumer);
                    } else {
                        count = Gemm::Tile::BuildFullPagedKvRecords<128U>(sparseIdx + firstBlock, blockTable, records,
                                                                          blocks, blockY_, tokenStride_, kPageStride_,
                                                                          consumer);
                    }
                    fullRecords = true;
                    dealtLenAccum = groupTokens;
                }
            }
            for (uint32_t blockIdx = compactBegin / blockY_; dealtLenAccum < groupTokens; ++blockIdx) {
                const uint32_t rows = min(blockY_ - inner, groupTokens - dealtLenAccum);
                // These read-only inputs never alias the descriptor workspace.
                int32_t blockId;
                if constexpr (RESIDUAL_BLOCK_FLAG) {
                    blockId = blockIdx == explicitCount ? static_cast<int32_t>(residualId) : sparseIdx[blockIdx];
                } else {
                    blockId = sparseIdx[blockIdx];
                }
                const uint64_t sourceToken = static_cast<uint64_t>(blockId) * blockY_ + inner;
                // A page split produces ordinary entries; Cube never parses pages.
                for (uint32_t copied = 0; copied < rows;) {
                    uint32_t len = rows - copied;
                    uint64_t token = sourceToken + copied;
                    uint64_t kOffset = 0;
                    uint64_t vOffset = 0;
                    uint32_t pageInner = 0;
                    uint32_t page = 0;
                    if constexpr (PAGED_CACHE_FLAG) {
                        pageInner = token % pageSize_;
                        page = blockTable[token / pageSize_];
                        len = min(len, pageSize_ - pageInner);
                        kOffset = static_cast<uint64_t>(page) * kPageStride_;
                        if constexpr (!compactPaged) {
                            vOffset = static_cast<uint64_t>(page) * vPageStride_;
                        }
                        token = pageInner;
                    }
                    kOffset += token * tokenStride_;
                    if constexpr (!compactPaged) {
                        vOffset += token * tokenStride_;
                    }
                    const uint32_t word = GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS + count * recordWords;
                    if constexpr (PAGED_CACHE_FLAG) {
                        const bool reverse = kOffset < previousOffset;
                        const uint64_t kStride = reverse ? previousOffset - kOffset : kOffset - previousOffset;
                        const uint64_t vStride =
                            vOffset < previousVOffset ? previousVOffset - vOffset : vOffset - previousVOffset;
                        if (previousLen != 0U && page == previousPage && pageInner == previousPageEnd) {
                            // Within one physical page both K and V use tokenStride_.
                            // Extend an unpaired run without changing valid token order.
                            previousLen += len;
                            records[word - (compactPaged ? 1U : 2U)] = static_cast<uint64_t>(previousLen) << 1U;
                            if constexpr (!compactPaged) {
                                records[word - 1U] = static_cast<uint64_t>(previousLen) << 1U;
                            }
                        } else if (previousLen == blockY_ && len == blockY_ &&
                                   (compactPaged || reverse == (vOffset < previousVOffset)) &&
                                   kStride <= 0xFFFFFFFFULL && kStride % COPY_ALIGNMENT == 0U &&
                                   (compactPaged || (vStride <= 0xFFFFFFFFULL && vStride % COPY_ALIGNMENT == 0U))) {
                            // Reorder K and V together, only within this group. Short page/tail
                            // fragments stay single, so valid tokens never move into padded columns.
                            if (reverse) {
                                records[word - recordWords] = kOffset;
                                if constexpr (!compactPaged) {
                                    records[word - 3U] = vOffset;
                                }
                            }
                            records[word - (compactPaged ? 1U : 2U)] =
                                (kStride << 32U) | (static_cast<uint64_t>(len) << 1U) | 1U;
                            if constexpr (!compactPaged) {
                                records[word - 1U] = (vStride << 32U) | (static_cast<uint64_t>(len) << 1U) | 1U;
                            }
                            if constexpr (TOKEN_BYTES_ != 0U) {
                                consumer.Record(reverse ? kOffset : previousOffset, len, kStride, 2U);
                            }
                            previousLen = 0;
                        } else {
                            if constexpr (TOKEN_BYTES_ != 0U) {
                                if (previousLen != 0U) {
                                    consumer.Record(previousOffset, previousLen, 0U, 1U);
                                }
                            }
                            records[word] = kOffset;
                            if constexpr (compactPaged) {
                                records[word + 1U] = static_cast<uint64_t>(len) << 1U;
                            } else {
                                records[word + 1U] = vOffset;
                                records[word + 2U] = static_cast<uint64_t>(len) << 1U;
                                records[word + 3U] = static_cast<uint64_t>(len) << 1U;
                            }
                            ++count;
                            previousLen = len;
                            previousOffset = kOffset;
                            previousVOffset = vOffset;
                        }
                        previousPage = page;
                        previousPageEnd = pageInner + len;
                    } else {
                        const uint64_t stride = kOffset - previousOffset;
                        if (previousLen != 0U && kOffset == previousOffset + previousLen * tokenStride_) {
                            // Adjacent blocks share the token stride and need only one copy.
                            previousLen += len;
                            records[word - 1U] = static_cast<uint64_t>(previousLen) << 1U;
                        } else if (previousLen == blockY_ && len == blockY_ && stride <= 0xFFFFFFFFULL &&
                                   stride % COPY_ALIGNMENT == 0U) {
                            records[word - 1U] = (stride << 32U) | (static_cast<uint64_t>(len) << 1U) | 1U;
                            previousLen = 0;
                        } else {
                            records[word] = kOffset;
                            records[word + 1U] = static_cast<uint64_t>(len) << 1U;
                            ++count;
                            previousLen = len;
                            previousOffset = kOffset;
                        }
                    }
                    copied += len;
                }
                dealtLenAccum += rows;
                inner = 0;
            }
            if constexpr (PAGED_CACHE_FLAG && TOKEN_BYTES_ != 0U) {
                if (previousLen != 0U) {
                    consumer.Record(previousOffset, previousLen, 0U, 1U);
                }
                fullRecords = true; // Every finalized K record has already issued its scale copy.
            }
            consumer.EndGroup(records, count, fullRecords);
            records[0] = count;
            const uint32_t copyWords =
                (GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS + count * recordWords + 3U) / 4U * 4U;
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
            AscendC::DataCopy(groupInfo, records_, copyWords);
            if (TOKEN_BYTES_ == 0U && groupNum > 2U && groupIdx < 2U) {
                // Both AIVs publish round 0 after their first group copy.
                // The caller publishes round 1 (or the only round for <= 2 groups).
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(GbsaKernelArch35::KV_OFFSET_FIRST_READY_FLAG_BASE +
                                                        tileIdx % GbsaKernelArch35::KV_OFFSET_STAGES);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
            recordsBufIdx_ ^= recordsBufMask_;
        }
    }

    // The most recently produced local group stays valid until this producer
    // reuses its slot. Consumers may read it while MTE3 publishes the same data.
    __aicore__ inline AscendC::LocalTensor<uint64_t> GetLocalRecords(uint32_t previous = 0) const
    {
        return recordsBuf_[(recordsBufIdx_ ^ recordsBufMask_) ^ (previous & recordsBufMask_)];
    }

private:
    AscendC::LocalTensor<uint64_t> recordsBuf_[2];
    uint32_t recordsBufIdx_ = 0;
    uint32_t recordsBufMask_;
    uint32_t tileSize_;
    uint32_t l1TileN_;
    uint32_t blockY_;
    uint32_t pageSize_;
    uint32_t groupWords_;
    uint64_t tokenStride_;
    uint64_t kPageStride_;
    uint64_t vPageStride_;
    uint32_t tokenBytes_; // Per-head width D, not the uint64_t interleaved token stride.
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_PREPARE_KV_MXFP8_ARCH35_HPP
