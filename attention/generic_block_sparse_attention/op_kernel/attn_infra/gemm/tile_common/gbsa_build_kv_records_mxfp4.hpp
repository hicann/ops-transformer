/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BUILD_KV_RECORDS_MXFP4_HPP
#define GBSA_BUILD_KV_RECORDS_MXFP4_HPP

#include <cstdint>

namespace NpuArch::Gemm::Tile {

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
        const uint32_t word = 8 + count * 2;
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
    records[8 + count * 2] = offset;
    records[9 + count * 2] = static_cast<uint64_t>(rows) << 1;
    if constexpr (BlockRows == 0U) {
        consumer.Record(offset, rows, 0, 1);
    } else {
        consumer.template Record<BlockRows>(offset, rows, 0, 1);
    }
    return count + 1;
}

// MXFP4 PA uses identical K/V page strides and a shared page table, so a
// complete block needs one relative offset and one meta word. Keep pending
// runs in registers until they are final; cache only equal logical pages,
// without assuming that the sparse block list is sorted.
template <uint32_t PageRows, uint32_t BlockRows = 0U, class Indices, class Pages, class Records, class Consumer>
__aicore__ inline uint32_t BuildFullPagedKvRecords(Indices indices, Pages pages, Records records, uint32_t blocks,
                                                   uint32_t runtimeRows, uint64_t tokenStride, uint64_t pageStride,
                                                   Consumer& consumer)
{
    const uint32_t blockRows = BlockRows == 0U ? runtimeRows : BlockRows;
    // A complete logical page may be represented by one exact contiguous
    // record. Check every sparse key; endpoint equality alone is insufficient.
    if constexpr (BlockRows != 0U) {
        constexpr uint32_t blocksPerPage = PageRows / BlockRows;
        if (blocks == blocksPerPage) {
            const int32_t first = indices[0];
            if (first >= 0 && static_cast<uint32_t>(first) % blocksPerPage == 0U &&
                static_cast<uint32_t>(indices[blocksPerPage - 1U]) ==
                    static_cast<uint32_t>(first) + blocksPerPage - 1U) {
                bool contiguous = true;
                for (uint32_t i = 1U; i + 1U < blocksPerPage; ++i) {
                    if (static_cast<uint32_t>(indices[i]) != static_cast<uint32_t>(first) + i) {
                        contiguous = false;
                        break;
                    }
                }
                if (contiguous) {
                    const uint64_t offset =
                        static_cast<uint64_t>(pages[static_cast<uint32_t>(first) / blocksPerPage]) * pageStride;
                    records[8] = offset;
                    records[9] = static_cast<uint64_t>(PageRows) << 1U;
                    consumer.template Record<BlockRows>(offset, PageRows, 0U, 1U);
                    return 1U;
                }
            }
        }
    }

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

#endif // GBSA_BUILD_KV_RECORDS_MXFP4_HPP
