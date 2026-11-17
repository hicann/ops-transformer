/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_PREPARE_K_SCALE_MXFP4_A5_HPP
#define GBSA_PREPARE_K_SCALE_MXFP4_A5_HPP

#include "gbsa_copy_gm_to_l1_a5.hpp"

namespace NpuArch::Gemm::Tile {

// Copy one active descriptor group into 32B-aligned blocks of scale rows.
// The epilogue owns packing, UB->L1 publication, and the buffer's events.
template <bool Paged>
struct CopyKScaleGmToUbMxfp4 {
    uint32_t D;
    __aicore__ CopyKScaleGmToUbMxfp4(uint32_t dim)
        : D(dim)
    {}
    __aicore__ inline uint32_t operator()(const AscendC::LocalTensor<uint8_t>& scratch,
                                          const AscendC::GlobalTensor<uint8_t>& keyScale,
                                          const AscendC::GlobalTensor<uint64_t>& descriptor, uint32_t groupWords,
                                          uint32_t tokenStrideBytes, uint32_t blockRows = 1,
                                          uint32_t sourceBytes = 0) const
    {
        RefreshMxfp4Descriptor<Paged>(descriptor, groupWords);
        return CopyRecords(scratch, keyScale, descriptor.GetPhyAddr(), tokenStrideBytes, blockRows, sourceBytes);
    }

    __aicore__ inline uint32_t operator()(const AscendC::LocalTensor<uint8_t>& scratch,
                                          const AscendC::GlobalTensor<uint8_t>& keyScale,
                                          const AscendC::LocalTensor<uint64_t>& descriptor, uint32_t tokenStrideBytes,
                                          uint32_t blockRows, uint32_t sourceBytes) const
    {
        // The producer's UB records remain live; no GM round trip or cache refresh.
        auto records = reinterpret_cast<__ubuf__ uint64_t*>(descriptor.GetPhyAddr());
        return CopyRecords(scratch, keyScale, records, tokenStrideBytes, blockRows, sourceBytes);
    }

    // The descriptor builder already has these fields in scalar registers.
    // Issue the same padded copies without encoding and reading a record back.
    template <uint32_t BlockRows>
    __aicore__ inline void CopyRecord(const AscendC::LocalTensor<uint8_t>& scratch,
                                      const AscendC::GlobalTensor<uint8_t>& keyScale, uint64_t offset, uint32_t rows,
                                      uint64_t stride, uint32_t matrices, uint32_t dstRow, uint32_t sourceBytes,
                                      uint32_t tokenStrideBytes) const
    {
        const uint32_t blockBytes = BlockRows * (D / 32);
        const uint32_t blockStride = (blockBytes + 31) / 32 * 32;
        // A paired full-block record can gather both interleaved-head spans
        // with the same per-block UB padding as the two original CopyRun calls.
        if (BlockRows > 1U && matrices == 2U && rows == BlockRows && tokenStrideBytes > D / 2U) {
            const uint32_t span = (BlockRows - 1U) * (tokenStrideBytes / 16U) + D / 32U;
            const uint32_t ubPitch = (span + 31U) / 32U * 32U;
            AscendC::DataCopyExtParams params;
            params.blockCount = 2;
            params.blockLen = span;
            params.srcStride = stride / 16U - span;
            params.dstStride = 0;
            AscendC::DataCopyPadExtParams<uint8_t> padding;
            padding.isPad = false;
            AscendC::DataCopyPad(scratch[dstRow / BlockRows * ubPitch], keyScale[offset / 16U], params, padding);
            return;
        }

        if (BlockRows != 1 && matrices == 2 && tokenStrideBytes == D / 2U) {
            AscendC::DataCopyExtParams params;
            params.blockCount = 2;
            params.blockLen = rows * (D / 32);
            params.srcStride = stride / 16 - params.blockLen;
            params.dstStride = 0;
            AscendC::DataCopyPadExtParams<uint8_t> padding;
            padding.isPad = false;
            if (blockBytes < blockStride) {
                if (rows == BlockRows && stride / 16 == blockStride && offset / 16 + 2 * blockStride <= sourceBytes) {
                    params.blockCount = 1;
                    params.blockLen = 2 * blockStride;
                    params.srcStride = 0;
                }
            }
            AscendC::DataCopyPad(scratch[dstRow / BlockRows * blockStride], keyScale[offset / 16], params, padding);
        } else {
            for (uint32_t matrix = 0; matrix < matrices; ++matrix) {
                CopyRun<BlockRows>(scratch, keyScale, offset + matrix * stride, dstRow + matrix * rows, rows,
                                   tokenStrideBytes);
            }
        }
    }

private:
    template <class Records>
    __aicore__ inline uint32_t CopyRecords(const AscendC::LocalTensor<uint8_t>& scratch,
                                           const AscendC::GlobalTensor<uint8_t>& keyScale, Records records,
                                           uint32_t tokenStrideBytes, uint32_t blockRows, uint32_t sourceBytes) const
    {
        if constexpr (!Paged) {
            if (CanCopyKScaleDirect(keyScale, records, tokenStrideBytes, D, 128U)) {
                return 0; // Keep the direct path ahead of block-layout dispatch.
            }
        }
        const AscendC::LocalTensor<uint8_t> ubBase(AscendC::TPosition::VECCALC, 0, Arch::AtlasA5::UB_SIZE);
        const uint32_t scratchAddress =
            static_cast<uint32_t>((uint64_t)scratch.GetPhyAddr() - (uint64_t)ubBase.GetPhyAddr());
        const AscendC::LocalTensor<uint8_t> scratchData(AscendC::TPosition::VECCALC, scratchAddress, scratch.GetSize());
        if (blockRows == 4) {
            return Copy<4>(scratchData, keyScale, records, tokenStrideBytes, sourceBytes);
        }
        if (blockRows == 8) {
            return Copy<8>(scratchData, keyScale, records, tokenStrideBytes, sourceBytes);
        }
        if (blockRows == 16) {
            return Copy<16>(scratchData, keyScale, records, tokenStrideBytes, sourceBytes);
        }
        return Copy<1>(scratchData, keyScale, records, tokenStrideBytes, sourceBytes);
    }

    template <uint32_t BlockRows, class Records>
    __aicore__ inline uint32_t Copy(const AscendC::LocalTensor<uint8_t>& scratch,
                                    const AscendC::GlobalTensor<uint8_t>& keyScale, Records records,
                                    uint32_t tokenStrideBytes, uint32_t sourceBytes) const
    {
        constexpr uint32_t recordWords = 2;
        const uint32_t scaleBytesPerToken = D / 32;
        uint32_t copiedRows = 0;
        const uint32_t count = records[0];
        const uint64_t firstMeta = records[9];
        if (count == 1 && (firstMeta & 1) == 0) {
            const uint32_t rows = static_cast<uint32_t>(firstMeta) >> 1;
            CopyRun<BlockRows>(scratch, keyScale, records[8], 0, rows, tokenStrideBytes);
            return rows;
        }
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t word = 8 + index * recordWords;
            const uint64_t sourceByteOffset = records[word];
            const uint64_t meta = records[word + 1];
            const uint32_t rows = static_cast<uint32_t>(meta) >> 1;
            const uint32_t matrices = 1 + static_cast<uint32_t>(meta & 1);
            const uint32_t matrixStrideBytes = static_cast<uint32_t>(meta >> 32);
            AscendC::DataCopyExtParams params;
            params.dstStride = 0;
            AscendC::DataCopyPadExtParams<uint8_t> padding;
            padding.isPad = false;
            const uint32_t blockBytes = BlockRows * scaleBytesPerToken;
            const uint32_t blockStride = (blockBytes + 31) / 32 * 32;
            // Paired full sparse blocks have equal lengths and one source gap.
            // For BlockRows < 16 each record is exactly one block; otherwise
            // every 16-token segment is 32B aligned and the whole run is dense.
            if (BlockRows != 1 && matrices == 2 && tokenStrideBytes == D / 2U) {
                params.blockCount = 2;
                params.blockLen = rows * scaleBytesPerToken;
                params.srcStride = matrixStrideBytes / 16 - params.blockLen;
                if (blockBytes < blockStride) {
                    // A pair separated by exactly one UB block can be read as
                    // one bounded span. Extra GM bytes land only in UB padding;
                    // Gather never selects them. Do not read past this head.
                    if (rows == BlockRows && matrixStrideBytes / 16 == blockStride &&
                        sourceByteOffset / 16 + 2 * blockStride <= sourceBytes) {
                        params.blockCount = 1;
                        params.blockLen = 2 * blockStride;
                        params.srcStride = 0;
                    }
                }
                AscendC::DataCopyPad(scratch[copiedRows / BlockRows * blockStride], keyScale[sourceByteOffset / 16],
                                     params, padding);
                copiedRows += rows * 2;
                continue;
            }
            for (uint32_t matrix = 0; matrix < matrices; ++matrix) {
                const uint64_t offset = sourceByteOffset + static_cast<uint64_t>(matrix) * matrixStrideBytes;
                CopyRun<BlockRows>(scratch, keyScale, offset, copiedRows, rows, tokenStrideBytes);
                copiedRows += rows;
            }
        }
        return copiedRows;
    }

    template <uint32_t BlockRows>
    __aicore__ inline void CopyRun(const AscendC::LocalTensor<uint8_t>& scratch,
                                   const AscendC::GlobalTensor<uint8_t>& keyScale, uint64_t offset, uint32_t dstRow,
                                   uint32_t rows, uint32_t tokenStrideBytes) const
    {
        const uint32_t scaleBytesPerToken = D / 32;
        // Read the span of one sparse block, retaining interleaved heads in
        // UB gaps. The pack VF selects only this head's scale pairs.
        const uint32_t scaleStride = tokenStrideBytes / 16U;
        const uint32_t blockBytes = (BlockRows - 1U) * scaleStride + scaleBytesPerToken;
        const uint32_t blockStride = (blockBytes + 31) / 32 * 32;
        AscendC::DataCopyExtParams params;
        params.dstStride = 0;
        AscendC::DataCopyPadExtParams<uint8_t> padding;
        padding.isPad = false;
        const uint32_t fullRows = rows / BlockRows * BlockRows;
        if (fullRows != 0) {
            params.blockCount = fullRows / BlockRows;
            params.blockLen = blockBytes;
            params.srcStride = BlockRows * tokenStrideBytes / 16 - blockBytes;
            AscendC::DataCopyPad(scratch[dstRow / BlockRows * blockStride], keyScale[offset / 16], params, padding);
        }
        // Only the final record can have a partial block in the packed path.
        if (fullRows != rows) {
            params.blockCount = 1;
            params.blockLen = (rows - fullRows - 1U) * scaleStride + scaleBytesPerToken;
            params.srcStride = 0;
            AscendC::DataCopyPad(scratch[(dstRow + fullRows) / BlockRows * blockStride],
                                 keyScale[(offset + fullRows * tokenStrideBytes) / 16], params, padding);
        }
    }
};

} // namespace NpuArch::Gemm::Tile
#endif // GBSA_PREPARE_K_SCALE_MXFP4_A5_HPP
