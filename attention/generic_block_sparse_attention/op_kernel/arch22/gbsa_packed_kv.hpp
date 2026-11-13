/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_PACKED_KV_HPP
#define GBSA_PACKED_KV_HPP

namespace GbsaKernelArch22::Int8::PackedKv {
template <class Tiling>
__aicore__ inline bool Enabled(const Tiling& t)
{
    const bool reuse = t.groupSize <= 16U && t.embeddingSize >= 64U && t.embeddingSize % 32U == 0U &&
                       t.totalQTokens >= static_cast<uint64_t>(t.batch) * 128U;
    // Do not expand the former physical-pack workspace budget when PA pages are heavily shared.
    const bool capacity = t.blockSize == 0U || static_cast<uint64_t>(t.batch) * t.maxBlocksPerBatch <=
                                                   (t.totalKvTokens + static_cast<uint64_t>(t.batch) * 128U) / 128U;
    return reuse && capacity;
}
template <class Tiling>
__aicore__ inline uint64_t Bytes(const Tiling& t)
{
    const uint64_t tokens = t.blockSize != 0U ? static_cast<uint64_t>(t.batch) * t.maxBlocksPerBatch * 128U :
                                                t.totalKvTokens + static_cast<uint64_t>(t.batch) * 128U;
    const uint64_t raw = tokens * t.kvHeads * t.embeddingSize;
    return (raw + 511U) / 512U * 512U;
}
template <class Tiling>
__aicore__ inline uint64_t Offset(const Tiling& t)
{
    const uint64_t originalEnd = t.fdStaticEnabled != 0U ?
                                     t.fdPartialOOffset + static_cast<uint64_t>(t.fdPartialCapacity) * t.groupSize *
                                                              ((t.embeddingSize + 15U) / 16U * 16U) * sizeof(float) :
                                     t.mm1OutSize + t.smOnlineOutSize + t.mm2OutSize;
    return (originalEnd + 511U) / 512U * 512U;
}

template <bool isKey, class Layout>
__aicore__ inline void Emit(AscendC::LocalTensor<int8_t> dst, AscendC::GlobalTensor<int8_t> src, const Layout& layout,
                            uint32_t dstBegin, uint64_t source, uint32_t rows, uint32_t ndNum, uint32_t sourceStride,
                            uint32_t dim, uint32_t blockY)
{
    const uint32_t pitch = static_cast<uint32_t>(isKey ? layout.stride(1) : layout.stride(3)) / 32U;
    if (rows == blockY) {
        // Every selected sparse block is aligned to this packed unit: the dummy ND block address is its NZ base.
        AscendC::DataCopyParams copy{static_cast<uint16_t>(dim / 32U), static_cast<uint16_t>(blockY), 0,
                                     static_cast<uint16_t>(pitch - blockY)};
        const uint32_t offset = static_cast<uint32_t>(isKey ? layout.GetOffset(NpuArch::MatrixCoord{0, dstBegin}) :
                                                              layout.GetOffset(NpuArch::MatrixCoord{dstBegin, 0}));
        AscendC::DataCopy(dst[offset], src[source], copy);
        if (ndNum == 2U) {
            const uint32_t second =
                static_cast<uint32_t>(isKey ? layout.GetOffset(NpuArch::MatrixCoord{0, dstBegin + blockY}) :
                                              layout.GetOffset(NpuArch::MatrixCoord{dstBegin + blockY, 0}));
            AscendC::DataCopy(dst[second], src[source + sourceStride], copy);
        }
        return;
    }
    // Adjacent merged blocks or a partial residual retain their original token order.
    for (uint32_t nd = 0; nd < ndNum; ++nd) {
        uint64_t current = source + static_cast<uint64_t>(nd) * sourceStride;
        uint32_t copied = 0U;
        while (copied < rows) {
            const uint32_t take = rows - copied < blockY ? rows - copied : blockY;
            const uint32_t outputRow = dstBegin + nd * rows + copied;
            const uint32_t offset = static_cast<uint32_t>(isKey ? layout.GetOffset(NpuArch::MatrixCoord{0, outputRow}) :
                                                                  layout.GetOffset(NpuArch::MatrixCoord{outputRow, 0}));
            AscendC::DataCopyParams copy{static_cast<uint16_t>(dim / 32U), static_cast<uint16_t>(take),
                                         static_cast<uint16_t>(blockY - take), static_cast<uint16_t>(pitch - take)};
            AscendC::DataCopy(dst[offset], src[current], copy);
            current += static_cast<uint64_t>(take) * dim;
            copied += take;
        }
    }
}

template <class Resource>
__aicore__ inline void BuildMap(Resource& resource, uint32_t dim, uint32_t blockY)
{
    auto offsets = resource.ubBuf.template GetBufferByByte<uint32_t>(64U * 1024U);
    const uint32_t words = 128U * dim / 4U;
    for (uint32_t i = 0; i < words; ++i) {
        const uint32_t blockWords = blockY * dim / 4U;
        const uint32_t block = i / blockWords;
        const uint32_t inner = i % blockWords;
        const uint32_t stripe = inner / (blockY * 8U);
        const uint32_t token = (inner % (blockY * 8U)) / 8U;
        const uint32_t channelWord = inner % 8U;
        offsets.SetValue(i, (block * blockY + token) * dim + stripe * 32U + channelWord * 4U);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(7);
}

template <class Resource>
__aicore__ inline void PackPage(Resource& resource, AscendC::GlobalTensor<int8_t> source,
                                AscendC::GlobalTensor<int8_t> packed, uint64_t sourceOffset, uint64_t targetOffset,
                                uint32_t tokenStride, uint32_t dim, uint32_t rows)
{
    auto input = resource.ubBuf.template GetBufferByByte<uint32_t>(0);
    auto output = resource.ubBuf.template GetBufferByByte<uint32_t>(32U * 1024U);
    auto offsets = resource.ubBuf.template GetBufferByByte<uint32_t>(64U * 1024U);
    const uint32_t words = 128U * dim / 4U;
    AscendC::Duplicate(input, uint32_t{0}, words);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);
    AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows), dim, tokenStride - dim, 0U, 0U};
    AscendC::DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(input.template ReinterpretCast<int8_t>(), source[sourceOffset], copy, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::Gather(output, input, offsets, uint32_t{0}, words);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
    AscendC::DataCopy(packed[targetOffset], output.template ReinterpretCast<int8_t>(), 128U * dim);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
}

template <bool isPaged, class Resource, class Params, class Tiling>
__aicore__ inline void PackAll(Resource& resource, const Params& params, const Tiling& t)
{
    AscendC::GlobalTensor<int8_t> key, value, packedKey, packedValue;
    key.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.k));
    value.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.v));
    packedKey.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.workSpace + Offset(t)));
    packedValue.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.workSpace + Offset(t) + Bytes(t)));
    const uint32_t worker = AscendC::GetBlockIdx();
    const uint32_t workers = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    const uint32_t dim = t.embeddingSize;
    const bool headMajor = t.kvLayout == 2U || t.kvLayout == 3U;
    const uint32_t tokenStride = headMajor ? dim : t.kvHeads * dim;
    BuildMap(resource, dim, t.blockShapeY);
    if constexpr (isPaged) {
        AscendC::GlobalTensor<int32_t> table, usedKv;
        table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.blockTable));
        usedKv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedKv));
        uint64_t jobBase = 0U;
        for (uint32_t batch = 0; batch < t.batch; ++batch) {
            const uint32_t length = usedKv.GetValue(batch);
            const uint32_t pages = (length + 127U) / 128U;
            const uint64_t jobs = static_cast<uint64_t>(pages) * t.kvHeads;
            const uint32_t first = (worker + workers - jobBase % workers) % workers;
            for (uint64_t job = first; job < jobs; job += workers) {
                const uint32_t page = job / t.kvHeads;
                const uint32_t head = job % t.kvHeads;
                const uint32_t physical = table.GetValue(static_cast<uint64_t>(batch) * t.maxBlocksPerBatch + page);
                const uint32_t rows = length - page * 128U < 128U ? length - page * 128U : 128U;
                const uint64_t inner = static_cast<uint64_t>(head) * (headMajor ? 128U : 1U) * dim;
                const uint64_t target =
                    ((static_cast<uint64_t>(batch) * t.kvHeads + head) * t.maxBlocksPerBatch + page) * 128U * dim;
                PackPage(resource, key, packedKey, static_cast<uint64_t>(physical) * t.kStride0 + inner, target,
                         tokenStride, dim, rows);
                PackPage(resource, value, packedValue, static_cast<uint64_t>(physical) * t.vStride0 + inner, target,
                         tokenStride, dim, rows);
            }
            jobBase += jobs;
        }
    } else {
        AscendC::GlobalTensor<int64_t> cuKv;
        AscendC::GlobalTensor<int32_t> usedKv;
        cuKv.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengthsKv));
        usedKv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedKv));
        const bool batched = t.qLayout != 0U;
        uint64_t jobBase = 0U;
        for (uint32_t batch = 0; batch < t.batch; ++batch) {
            const uint64_t start = batched ? static_cast<uint64_t>(batch) * t.kvSeqCapacity : cuKv.GetValue(batch);
            const uint32_t storageLength =
                batched ? t.kvSeqCapacity : static_cast<uint32_t>(cuKv.GetValue(batch + 1U) - start);
            const uint32_t length = params.sequsedKv == nullptr ? storageLength : usedKv.GetValue(batch);
            const uint32_t pages = (length + 127U) / 128U;
            const uint64_t jobs = static_cast<uint64_t>(pages) * t.kvHeads;
            const uint64_t packedBase = (start / 128U + batch) * 128U * t.kvHeads * dim;
            const uint32_t first = (worker + workers - jobBase % workers) % workers;
            for (uint64_t job = first; job < jobs; job += workers) {
                const uint32_t page = job / t.kvHeads;
                const uint32_t head = job % t.kvHeads;
                const uint32_t rowStart = page * 128U;
                const uint32_t rows = length - rowStart < 128U ? length - rowStart : 128U;
                const uint64_t source = start * t.kvHeads * dim +
                                        static_cast<uint64_t>(head) * (headMajor ? t.kvSeqCapacity : 1U) * dim +
                                        static_cast<uint64_t>(rowStart) * tokenStride;
                const uint64_t target = packedBase + (static_cast<uint64_t>(head) * pages + page) * 128U * dim;
                PackPage(resource, key, packedKey, source, target, tokenStride, dim, rows);
                PackPage(resource, value, packedValue, source, target, tokenStride, dim, rows);
            }
            jobBase += jobs;
        }
    }
}
} // namespace GbsaKernelArch22::Int8::PackedKv
#endif
