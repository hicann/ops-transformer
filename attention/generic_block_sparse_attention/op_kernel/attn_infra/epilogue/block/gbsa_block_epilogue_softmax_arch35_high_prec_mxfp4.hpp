/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_SOFTMAX_ARCH35_HIGH_PREC_MXFP4_HPP
#define GBSA_BLOCK_EPILOGUE_SOFTMAX_ARCH35_HIGH_PREC_MXFP4_HPP

#include "gbsa_block_epilogue.hpp"
#include "../gbsa_epilogue_dispatch_policy.hpp"
#include "gbsa_block_epilogue_arch35_mxfp4_utils.hpp"
#include "mxfp4_vf/vf_softmax_nd_fp32_mxfp4.h"
#include "../../../tla/gbsa_tla_tensor.hpp"

namespace NpuArch::Epilogue::Block {

template <>
class BlockEpilogue<EpilogueAtlasA5Mxfp4Softmax> {
public:
    using DispatchPolicy = EpilogueAtlasA5Mxfp4Softmax;
    using ArchTag = typename DispatchPolicy::ArchTag;
    uint32_t D, Q, R;
    using Layout = GbsaKernelArch35::Mxfp4::UBufTileHelper;
    Layout layout;
    using Resource = Arch::Resource<Arch::AtlasA5>;

    __aicore__ inline explicit BlockEpilogue(Resource& resource, const Layout& ubInfo, float log2Cx)
        : D(ubInfo.D),
          Q(ubInfo.Q),
          R(ubInfo.R),
          layout(ubInfo),
          negLog2Cx_(-log2Cx)
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            sUBTensor[slot] =
                resource.ubBuf.template GetBufferByByte<float>(Layout::S_OFFSET + slot * layout.S_SLOT_BYTES);
            pUBTensor[slot] =
                resource.ubBuf.template GetBufferByByte<uint8_t>(layout.P_OFFSET + slot * layout.P_SLOT_BYTES);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
        localMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LOCAL_MAX_OFFSET);
        groupSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.GROUP_SUM_OFFSET);
        outputUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.O_OFFSET);
        sumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.OUT_L_OFFSET);
        maxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.OUT_G_OFFSET);
        tileMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_G_OFFSET);
        initialMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_MAX_OFFSET);
    }
    __aicore__ inline ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }
    // Output conversion reuses P and invalidates both slots' padding.
    __aicore__ inline void InvalidatePadding()
    {
        paddingPlanes_[0] = 0;
        paddingPlanes_[1] = 0;
    }
    __aicore__ inline void InitTask(uint32_t validRows, float attentionScale)
    {
        InvalidatePadding();
        validRows_ = validRows;
        attentionScale_ = attentionScale;
        // The first tile reads this immutable seed directly.
        // RescaleO owns the numerator and denominator state for each task.
        GbsaMxfp4VF::InitInitialMaxCallVF(initialMaxUBTensor, validRows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void operator()(uint32_t sSlot, uint32_t pSlot, uint32_t statsSlot, uint32_t validKv,
                                      const AscendC::LocalTensor<uint8_t>& l1P, uint32_t rowOffset, uint32_t queryRows,
                                      const AscendC::LocalTensor<float>& previousMax)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pSlot);
        if (validRows_ != 0) {
            const uint32_t bankRows = (validRows_ + 15U) / 16U * 16U;
            const uint32_t planes = (validKv + 63U) / 64U;
            const bool initializePadding = paddingPlanes_[pSlot] < planes && validRows_ != bankRows;
            paddingPlanes_[pSlot] = max(paddingPlanes_[pSlot], planes);
            if (validKv == layout.KV) {
                GbsaMxfp4VF::SoftmaxCallVF<true>(pUBTensor[pSlot], sUBTensor[sSlot], localMaxUBTensor, groupSumUBTensor,
                                                 tileMaxUBTensor[statsSlot * 64], previousMax, validRows_, validKv,
                                                 attentionScale_, negLog2Cx_, R, layout.KV, initializePadding);
            } else {
                GbsaMxfp4VF::SoftmaxCallVF<false>(pUBTensor[pSlot], sUBTensor[sSlot], localMaxUBTensor,
                                                  groupSumUBTensor, tileMaxUBTensor[statsSlot * 64], previousMax,
                                                  validRows_, validKv, attentionScale_, negLog2Cx_, R, layout.KV,
                                                  initializePadding);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pSlot);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pSlot);
            const uint32_t rows0 = (queryRows + 1) / 2;
            const uint32_t alignedRows = 2U * ((rows0 + 15U) / 16U * 16U);
            auto pLayout = tla::MakeLayout<uint8_t, layout::zN>(alignedRows, layout.KV / 2);
            auto pL1Tensor = tla::MakeTensor(l1P, pLayout, Arch::PositionL1{});
            auto pTile = tla::GetTile(pL1Tensor, tla::MakeCoord(rowOffset, 0), tla::MakeShape(bankRows, layout.KV / 2));
            AscendC::DataCopyParams copy = {static_cast<uint16_t>((validKv + 63) / 64), static_cast<uint16_t>(bankRows),
                                            static_cast<uint16_t>(R - bankRows),
                                            static_cast<uint16_t>(alignedRows - bankRows)};
            AscendC::DataCopy(pTile.data()[pLayout(pTile.coord())], pUBTensor[pSlot], copy);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pSlot);
    }

private:
    AscendC::LocalTensor<float> sUBTensor[2], localMaxUBTensor, groupSumUBTensor;
    AscendC::LocalTensor<float> outputUBTensor, sumUBTensor, maxUBTensor, tileMaxUBTensor, initialMaxUBTensor;
    AscendC::LocalTensor<uint8_t> pUBTensor[2];
    uint32_t paddingPlanes_[2] = {0, 0};
    uint32_t validRows_ = 0;
    float attentionScale_ = 1.0f;
    float negLog2Cx_;
};

} // namespace NpuArch::Epilogue::Block
#endif
