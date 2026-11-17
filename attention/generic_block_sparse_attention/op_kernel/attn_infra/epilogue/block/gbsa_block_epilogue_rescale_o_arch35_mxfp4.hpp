/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_RESCALE_O_ARCH35_MXFP4_HPP
#define GBSA_BLOCK_EPILOGUE_RESCALE_O_ARCH35_MXFP4_HPP

#include "gbsa_block_epilogue.hpp"
#include "../gbsa_epilogue_dispatch_policy.hpp"
#include "gbsa_block_epilogue_arch35_mxfp4_utils.hpp"
#include "mxfp4_vf/vf_rescale_o_fp32_per_channel.h"

namespace NpuArch::Epilogue::Block {

template <typename OutputT>
class BlockEpilogue<EpilogueAtlasA5Mxfp4RescaleO, OutputT> {
public:
    using DispatchPolicy = EpilogueAtlasA5Mxfp4RescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;
    uint32_t D, Q, R;
    using ElementO = OutputT;
    using ElementLse = float;
    using Layout = GbsaKernelArch35::Mxfp4::UBufTileHelper;
    Layout layout;
    using Resource = Arch::Resource<Arch::AtlasA5>;
    __aicore__ inline explicit BlockEpilogue(Resource& resource, const Layout& ubInfo, float log2CxCeil)
        : D(ubInfo.D),
          Q(ubInfo.Q),
          R(ubInfo.R),
          layout(ubInfo),
          log2CxCeil_(log2CxCeil)
    {
        outputUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.O_OFFSET);
        for (uint32_t slot = 0; slot < layout.PV_SLOTS; ++slot) {
            pvUBTensor[slot] =
                resource.ubBuf.template GetBufferByByte<float>(layout.PV_OFFSET + slot * layout.PV_SLOT_BYTES);
        }
        lsePackedUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LSE_PACKED_OFFSET);
        rowsumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LOCAL_ROW_SUM_OFFSET);
        sumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.OUT_L_OFFSET);
        maxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.OUT_G_OFFSET);
        lseUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LSE_OFFSET);
        resultUBTensor = resource.ubBuf.template GetBufferByByte<OutputT>(layout.OUTPUT_SCRATCH_OFFSET);
        for (uint32_t slot = 0; slot < 4; ++slot) {
            tileMaxUBTensor[slot] = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_G_OFFSET + slot * 256);
            dmUBTensor[slot] = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_DM_OFFSET + slot * 256);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
    }
    __aicore__ inline ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
    }
    __aicore__ inline void BeginTask(uint32_t rows, bool flashDecode)
    {
        if (flashDecode) {
            // A previous partial DMA reads N/G/L until this credit arrives.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        }
        // The frontend may already own the statistics of the following task.
        GbsaMxfp4VF::InitTaskCallVF<false>(outputUBTensor, sumUBTensor, maxUBTensor, tileMaxUBTensor[0], rows, D);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ConsumeTile(uint32_t statsSlot, uint32_t pvSlot, uint32_t rows, bool saveMax)
    {
        GbsaMxfp4VF::RescaleCallVF(outputUBTensor, pvUBTensor[pvSlot], rowsumUBTensor[statsSlot * 64], sumUBTensor,
                                   dmUBTensor[statsSlot], rows, D, R);
        AscendC::PipeBarrier<PIPE_V>();
        // Only final LSE or FD partial output consumes this G snapshot.
        if (saveMax) {
            GbsaMxfp4VF::CopyVectorCallVF(maxUBTensor, tileMaxUBTensor[statsSlot], rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void StorePartial(const AscendC::GlobalTensor<float>& numerator,
                                        const AscendC::GlobalTensor<float>& stats, uint32_t validRows,
                                        uint32_t subStride)
    {
        if (validRows == 0U) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::DataCopyExtParams copy = {1, validRows * D * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(numerator, outputUBTensor, copy);
        copy.blockLen = validRows * sizeof(float);
        AscendC::DataCopyPad(stats, maxUBTensor, copy);
        AscendC::DataCopyPad(stats[2U * subStride], sumUBTensor, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
    }

    __aicore__ inline void Store(const AscendC::GlobalTensor<OutputT>& output, const AscendC::GlobalTensor<float>& lse,
                                 uint32_t validRows, bool outputLse, uint32_t headStride, uint32_t lseHeadStride)
    {
        StoreOutput(output, lse, validRows, outputLse, headStride, lseHeadStride, outputUBTensor, sumUBTensor,
                    maxUBTensor);
    }

    __aicore__ inline void StoreTile(const AscendC::GlobalTensor<OutputT>& output,
                                     const AscendC::GlobalTensor<float>& lse, uint32_t validRows, bool outputLse,
                                     uint32_t headStride, uint32_t lseHeadStride, uint32_t statsSlot, uint32_t pvSlot)
    {
        // A complete single-tile task has no earlier numerator to rescale.
        StoreOutput(output, lse, validRows, outputLse, headStride, lseHeadStride, pvUBTensor[pvSlot],
                    rowsumUBTensor[statsSlot * 64U], tileMaxUBTensor[statsSlot]);
    }

    __aicore__ inline void StoreEmpty(const AscendC::GlobalTensor<OutputT>& output,
                                      const AscendC::GlobalTensor<float>& lse, uint32_t validRows, bool outputLse,
                                      uint32_t headStride, uint32_t lseHeadStride)
    {
        if (validRows == 0U) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
        const uint32_t chunkRows = layout.OUTPUT_SCRATCH_BYTES / (D * static_cast<uint32_t>(sizeof(OutputT)));
        auto lseSource = lseHeadStride == 1U ? lseUBTensor : lsePackedUBTensor;
        GbsaMxfp4VF::FillEmptyOutputCallVF(resultUBTensor, lseSource, min(validRows, chunkRows) * D, validRows,
                                           outputLse, lseHeadStride != 1U);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        for (uint32_t first = 0U; first < validRows; first += chunkRows) {
            const uint32_t rows = min(validRows - first, chunkRows);
            AscendC::DataCopyExtParams copy = {static_cast<uint16_t>(rows), D * static_cast<uint32_t>(sizeof(OutputT)),
                                               0, static_cast<int64_t>((headStride - 1U) * D * sizeof(OutputT)), 0};
            AscendC::DataCopyPad(output[first * headStride * D], resultUBTensor, copy);
        }
        if (outputLse) {
            AscendC::DataCopyExtParams copy = {
                static_cast<uint16_t>(lseHeadStride == 1U ? 1U : validRows),
                static_cast<uint32_t>((lseHeadStride == 1U ? validRows : 1U) * sizeof(float)), 0,
                static_cast<int64_t>((lseHeadStride - 1U) * sizeof(float)), 0};
            AscendC::DataCopyPad(lse, lseSource, copy);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }

private:
    __aicore__ __attribute__((noinline)) inline void StoreOutput(
        const AscendC::GlobalTensor<OutputT>& output, const AscendC::GlobalTensor<float>& lse, uint32_t validRows,
        bool outputLse, uint32_t headStride, uint32_t lseHeadStride, const AscendC::LocalTensor<float>& numerator,
        const AscendC::LocalTensor<float>& denominator, const AscendC::LocalTensor<float>& maximum)
    {
        if (validRows == 0) {
            return;
        }
        // Output conversion shares P storage with the frontend. Preserve its
        // credits while waiting for the current P/PScale stores to complete.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        const uint32_t CHUNK_ROWS = layout.OUTPUT_SCRATCH_BYTES / (D * static_cast<uint32_t>(sizeof(OutputT)));
        auto scratch = resultUBTensor;
        for (uint32_t first = 0; first < validRows; first += CHUNK_ROWS) {
            const uint32_t rows = validRows - first < CHUNK_ROWS ? validRows - first : CHUNK_ROWS;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
            GbsaMxfp4VF::NormalizeCallVF<OutputT>(scratch, numerator, denominator, first, rows, D, R);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
            AscendC::DataCopyExtParams copy = {
                static_cast<uint16_t>(rows), D * static_cast<uint32_t>(sizeof(OutputT)), 0,
                static_cast<int64_t>((headStride - 1) * D * static_cast<uint32_t>(sizeof(OutputT))), 0};
            AscendC::DataCopyPad(output[first * headStride * D], scratch, copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
        if (outputLse) {
            GbsaMxfp4VF::LseCallVF(lseUBTensor, maximum, denominator, validRows, log2CxCeil_);
            if (lseHeadStride == 1U) {
                // ComputeLse already stores contiguous FP32 values. Keep one
                // transfer for a contiguous GM view, including short row tails.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::DataCopyExtParams copy = {1, validRows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(lse, lseUBTensor, copy);
            } else {
                GbsaMxfp4VF::PackLseCallVF(lsePackedUBTensor, lseUBTensor, validRows);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::DataCopyExtParams copy = {static_cast<uint16_t>(validRows), sizeof(float), 0,
                                                   static_cast<int64_t>((lseHeadStride - 1U) * sizeof(float)), 0};
                AscendC::DataCopyPad(lse, lsePackedUBTensor, copy);
            }
        }
        // Only Vector reuses the output and LSE sources. Keep its credits;
        // independent Scalar descriptor work may proceed during the writeback.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }

private:
    float log2CxCeil_;
    AscendC::LocalTensor<float> outputUBTensor, pvUBTensor[2], rowsumUBTensor;
    AscendC::LocalTensor<float> lsePackedUBTensor, sumUBTensor, maxUBTensor, lseUBTensor;
    AscendC::LocalTensor<float> tileMaxUBTensor[4], dmUBTensor[4];
    AscendC::LocalTensor<OutputT> resultUBTensor;
};

} // namespace NpuArch::Epilogue::Block
#endif
