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

namespace NpuArch::Gemm::Tile {

// One bounded group of full BSND/BNSD blocks. Read each index once, keep the
// pending contiguous run in registers, and submit its scale copy as soon as
// the final {offset, rows, stride, matrices} record is known. Token stride is
// Nkv * D bytes for BSND, D bytes for BNSD; FP8 pair strides remain 32-byte aligned.
template <class Indices, class Records, class Consumer>
__aicore__ inline uint32_t BuildFullKvRecordsMxfp8(Indices indices, Records records, uint32_t blocks,
                                                   uint32_t blockRows, uint64_t tokenStride, Consumer& consumer)
{
    if (blocks == 0U) {
        return 0U;
    }
    const uint64_t blockBytes = static_cast<uint64_t>(blockRows) * tokenStride;
    uint64_t offset = static_cast<uint64_t>(indices[0]) * blockBytes;
    uint32_t rows = blockRows;
    uint32_t count = 0U;
    for (uint32_t index = 1U; index < blocks; ++index) {
        const uint64_t next = static_cast<uint64_t>(indices[index]) * blockBytes;
        if (next == offset + static_cast<uint64_t>(rows) * tokenStride) {
            rows += blockRows;
            continue;
        }
        const uint64_t stride = next - offset;
        const uint32_t word = 8U + count * 2U;
        records[word] = offset;
        ++count;
        if (rows == blockRows && next > offset && stride <= 0xffffffffULL && stride % 32U == 0U) {
            records[word + 1U] = (stride << 32U) | (static_cast<uint64_t>(rows) << 1U) | 1U;
            consumer.Record(offset, rows, stride, 2U);
            if (++index == blocks) {
                return count;
            }
            offset = static_cast<uint64_t>(indices[index]) * blockBytes;
        } else {
            records[word + 1U] = static_cast<uint64_t>(rows) << 1U;
            consumer.Record(offset, rows, 0U, 1U);
            offset = next;
        }
        rows = blockRows;
    }
    records[8U + count * 2U] = offset;
    records[9U + count * 2U] = static_cast<uint64_t>(rows) << 1U;
    consumer.Record(offset, rows, 0U, 1U);
    return count + 1U;
}

// Page IDs may be in arbitrary physical order. Preserve the selected-token
// order, merge only contiguous spans in one page, and pair only forward full
// blocks with a representable DMA stride. Cube consumes ordinary byte offsets.
template <class Indices, class Pages, class Records, class Consumer>
__aicore__ inline uint32_t BuildFullPagedKvRecordsMxfp8(Indices indices, Pages pages, Records records, uint32_t blocks,
                                                        uint32_t blockRows, uint64_t tokenStride, uint64_t pageStride,
                                                        Consumer& consumer)
{
    constexpr uint32_t PAGE = 128U;
    if (blocks == 0U) {
        return 0U;
    }
    uint32_t token = static_cast<uint32_t>(indices[0]) * blockRows;
    uint32_t page = static_cast<uint32_t>(pages[token / PAGE]);
    uint64_t offset = static_cast<uint64_t>(page) * pageStride + (token % PAGE) * tokenStride;
    uint32_t rows = blockRows;
    uint32_t count = 0U;
    for (uint32_t index = 1U; index < blocks; ++index) {
        token = static_cast<uint32_t>(indices[index]) * blockRows;
        const uint32_t nextPage = static_cast<uint32_t>(pages[token / PAGE]);
        const uint64_t next = static_cast<uint64_t>(nextPage) * pageStride + (token % PAGE) * tokenStride;
        if (page == nextPage && next == offset + static_cast<uint64_t>(rows) * tokenStride) {
            rows += blockRows;
            continue;
        }
        const uint64_t stride = next - offset;
        const uint32_t word = 8U + count * 2U;
        records[word] = offset;
        ++count;
        if (rows == blockRows && next > offset && stride <= 0xffffffffULL && stride % 32U == 0U) {
            records[word + 1U] = (stride << 32U) | (static_cast<uint64_t>(rows) << 1U) | 1U;
            consumer.Record(offset, rows, stride, 2U);
            if (++index == blocks) {
                return count;
            }
            token = static_cast<uint32_t>(indices[index]) * blockRows;
            page = static_cast<uint32_t>(pages[token / PAGE]);
            offset = static_cast<uint64_t>(page) * pageStride + (token % PAGE) * tokenStride;
        } else {
            records[word + 1U] = static_cast<uint64_t>(rows) << 1U;
            consumer.Record(offset, rows, 0U, 1U);
            offset = next;
            page = nextPage;
        }
        rows = blockRows;
    }
    records[8U + count * 2U] = offset;
    records[9U + count * 2U] = static_cast<uint64_t>(rows) << 1U;
    consumer.Record(offset, rows, 0U, 1U);
    return count + 1U;
}

} // namespace NpuArch::Gemm::Tile

#endif // GBSA_BUILD_KV_RECORDS_MXFP8_HPP
