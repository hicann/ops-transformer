/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BUILD_KV_RECORDS_MXFP8_HPP
#define GBSA_BUILD_KV_RECORDS_MXFP8_HPP

#include <cstdint>

#include "../../../arch35/gbsa_layout_utils.hpp"

namespace NpuArch::Gemm::Tile {

// Runtime token size is separate from the fixed-token descriptor API.
constexpr uint32_t RUNTIME_TOKEN_BYTES = 0xFFFFFFFFU;

// Complete BNSD blocks need no token/page splitting. Keep the pending run in
// registers and write it only after its length and optional second matrix are
// known. The descriptor format and selected-token order remain unchanged.
struct KvRecordsOnly {
    __aicore__ inline void Record(uint64_t, uint32_t, uint64_t, uint32_t) {}
    template <uint32_t BlockRows>
    __aicore__ inline void Record(uint64_t, uint32_t, uint64_t, uint32_t)
    {}
    __aicore__ inline void BeginGroup(uint32_t, uint32_t) {}
    template <class Records>
    __aicore__ inline void EndGroup(Records, uint32_t, bool)
    {}
};

template <uint32_t BlockRows = 0U, class Indices, class Records, class Consumer>
__aicore__ inline uint32_t BuildFullKvRecordsRuntime(Indices indices, Records records, uint32_t blocks,
                                                     uint32_t runtimeRows, Consumer& consumer, uint32_t TokenBytes)
{
    const uint32_t blockRows = BlockRows == 0U ? runtimeRows : BlockRows;
    const uint64_t blockBytes = static_cast<uint64_t>(blockRows) * TokenBytes;
    uint64_t offset = static_cast<uint64_t>(indices[0]) * blockBytes;
    uint32_t rows = blockRows;
    uint32_t count = 0;
    for (uint32_t index = 1; index < blocks; ++index) {
        const uint64_t next = static_cast<uint64_t>(indices[index]) * blockBytes;
        if (next == offset + static_cast<uint64_t>(rows) * TokenBytes) {
            rows += blockRows;
            continue;
        }
        const uint64_t stride = next - offset;
        const uint32_t word =
            GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS + count * GbsaKernelArch35::Mxfp8Descriptor::RECORD_WORDS;
        records[word] = offset;
        ++count;
        if (rows == blockRows && stride <= 0xFFFFFFFFULL && stride % 16 == 0) {
            records[word + 1] = (stride << 32) | (static_cast<uint64_t>(rows) << 1) | 1;
            if constexpr (BlockRows == 0U) {
                consumer.Record(offset, rows, stride, 2);
            } else {
                consumer.template Record<BlockRows>(offset, rows, stride, 2);
            }
            if (++index == blocks) {
                return count;
            }
            offset = static_cast<uint64_t>(indices[index]) * blockBytes;
        } else {
            records[word + 1] = static_cast<uint64_t>(rows) << 1;
            if constexpr (BlockRows == 0U) {
                consumer.Record(offset, rows, 0, 1);
            } else {
                consumer.template Record<BlockRows>(offset, rows, 0, 1);
            }
            offset = next;
        }
        rows = blockRows;
    }
    records[GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS + count * GbsaKernelArch35::Mxfp8Descriptor::RECORD_WORDS] =
        offset;
    records[9 + count * 2] = static_cast<uint64_t>(rows) << 1;
    if constexpr (BlockRows == 0U) {
        consumer.Record(offset, rows, 0, 1);
    } else {
        consumer.template Record<BlockRows>(offset, rows, 0, 1);
    }
    return count + 1;
}

template <class Indices, class Records>
__aicore__ inline uint32_t BuildFullKvRecordsRuntime(Indices indices, Records records, uint32_t blocks,
                                                     uint32_t blockRows, uint32_t TokenBytes)
{
    KvRecordsOnly consumer;
    return BuildFullKvRecordsRuntime<>(indices, records, blocks, blockRows, consumer, TokenBytes);
}

// Preserve the fixed-token API for existing callers and standalone descriptor tests.
template <uint32_t TokenBytes, uint32_t BlockRows = 0U, class Indices, class Records, class Consumer>
__aicore__ inline uint32_t BuildFullKvRecords(Indices indices, Records records, uint32_t blocks, uint32_t runtimeRows,
                                              Consumer& consumer)
{
    return BuildFullKvRecordsRuntime<BlockRows>(indices, records, blocks, runtimeRows, consumer, TokenBytes);
}

template <uint32_t TokenBytes, class Indices, class Records>
__aicore__ inline uint32_t BuildFullKvRecords(Indices indices, Records records, uint32_t blocks, uint32_t runtimeRows)
{
    KvRecordsOnly consumer;
    return BuildFullKvRecordsRuntime<>(indices, records, blocks, runtimeRows, consumer, TokenBytes);
}

// MXFP8_FULL PA uses identical K/V page strides and a shared page table, so a
// complete block needs one relative offset and one meta word. Keep pending
// runs in registers until they are final; cache only equal logical pages,
// without assuming that the sparse block list is sorted.
template <uint32_t PageRows, uint32_t BlockRows = 0U, class Indices, class Pages, class Records, class Consumer>
__aicore__ inline uint32_t BuildFullPagedKvRecords(Indices indices, Pages pages, Records records, uint32_t blocks,
                                                   uint32_t runtimeRows, uint64_t tokenStride, uint64_t pageStride,
                                                   Consumer& consumer)
{
    const uint32_t blockRows = BlockRows == 0U ? runtimeRows : BlockRows;
    uint64_t cachedPage = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t pageBase = 0, offset = 0;
    uint32_t page = 0, rows = 0, previousPage = 0, previousEnd = 0;
    uint32_t count = 0;
    for (uint32_t index = 0; index < blocks; ++index) {
        const uint64_t token = static_cast<uint64_t>(indices[index]) * blockRows;
        const uint64_t logicalPage = token / PageRows;
        const uint32_t inner = token % PageRows;
        if (logicalPage != cachedPage) {
            page = pages[logicalPage];
            pageBase = static_cast<uint64_t>(page) * pageStride;
            cachedPage = logicalPage;
        }
        const uint64_t next = pageBase + inner * tokenStride;
        if (rows != 0U && page == previousPage && inner == previousEnd) {
            rows += blockRows;
        } else {
            const bool reverse = next < offset;
            const uint64_t stride = reverse ? offset - next : next - offset;
            if (rows == blockRows && stride <= 0xFFFFFFFFULL && stride % 16U == 0U) {
                const uint32_t word = 8U + count * 2U;
                const uint64_t first = reverse ? next : offset;
                records[word] = first;
                records[word + 1U] = (stride << 32U) | (static_cast<uint64_t>(blockRows) << 1U) | 1U;
                if constexpr (BlockRows == 0U) {
                    consumer.Record(first, blockRows, stride, 2U);
                } else {
                    consumer.template Record<BlockRows>(first, blockRows, stride, 2U);
                }
                ++count;
                rows = 0U;
            } else {
                if (rows != 0U) {
                    const uint32_t word = 8U + count * 2U;
                    records[word] = offset;
                    records[word + 1U] = static_cast<uint64_t>(rows) << 1U;
                    if constexpr (BlockRows == 0U) {
                        consumer.Record(offset, rows, 0U, 1U);
                    } else {
                        consumer.template Record<BlockRows>(offset, rows, 0U, 1U);
                    }
                    ++count;
                }
                offset = next;
                rows = blockRows;
            }
        }
        previousPage = page;
        previousEnd = inner + blockRows;
    }
    if (rows != 0U) {
        const uint32_t word = 8U + count * 2U;
        records[word] = offset;
        records[word + 1U] = static_cast<uint64_t>(rows) << 1U;
        if constexpr (BlockRows == 0U) {
            consumer.Record(offset, rows, 0U, 1U);
        } else {
            consumer.template Record<BlockRows>(offset, rows, 0U, 1U);
        }
        ++count;
    }
    return count;
}

} // namespace NpuArch::Gemm::Tile

#endif // GBSA_BUILD_KV_RECORDS_MXFP8_HPP
