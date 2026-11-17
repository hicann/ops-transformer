/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP4_HPP
#define GBSA_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP4_HPP

#include "gbsa_block_epilogue.hpp"
#include "../gbsa_epilogue_dispatch_policy.hpp"
#include "gbsa_block_epilogue_arch35_mxfp4_utils.hpp"
#include "mxfp4_vf/vf_compute_pscale_group32.h"

namespace NpuArch::Epilogue::Block {

template <>
class BlockEpilogue<EpilogueAtlasA5Mxfp4PScale> {
public:
    using DispatchPolicy = EpilogueAtlasA5Mxfp4PScale;
    using ArchTag = typename DispatchPolicy::ArchTag;
    uint32_t D, Q, R;
    using Layout = GbsaKernelArch35::Mxfp4::UBufTileHelper;
    Layout layout;
    using Resource = Arch::Resource<Arch::AtlasA5>;
    static constexpr uint32_t SCALE_EVENT = 2;
    __aicore__ inline explicit BlockEpilogue(Resource& resource, const Layout& ubInfo, float log2CxCeil)
        : D(ubInfo.D),
          Q(ubInfo.Q),
          R(ubInfo.R),
          layout(ubInfo),
          scaleOffset_(-log2CxCeil),
          scaleBias_(127.0f - log2CxCeil)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
        initialMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_MAX_OFFSET);
        localMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LOCAL_MAX_OFFSET);
        groupSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.GROUP_SUM_OFFSET);
        rowsumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LOCAL_ROW_SUM_OFFSET);
        scaleUBTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(layout.PSCALE_OFFSET);
        for (uint32_t slot = 0; slot < 4; ++slot) {
            maxUBTensor[slot] = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_G_OFFSET + slot * 256);
            dmUBTensor[slot] = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_DM_OFFSET + slot * 256);
        }
    }
    __aicore__ inline ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
    }
    __aicore__ inline const AscendC::LocalTensor<float>& BeginTile(uint32_t statsSlot, bool firstTile)
    {
        firstTile_ = firstTile;
        // Softmax reads the previous snapshot and writes its own slot directly.
        return firstTile ? initialMaxUBTensor : maxUBTensor[(statsSlot + 3) % 4];
    }
    __aicore__ inline void operator()(uint32_t statsSlot, const AscendC::LocalTensor<uint8_t>& l1Scale, uint32_t bank,
                                      uint32_t validRows, uint32_t queryRows, uint32_t validKv)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
        const uint32_t scaleKv = validKv <= 128U ? 128U : layout.KV;
        const auto& previousMax = firstTile_ ? initialMaxUBTensor : maxUBTensor[(statsSlot + 3) % 4];
        if (validKv <= 32U && validRows <= 16U) {
            GbsaMxfp4VF::SingleGroupPScaleCallVF(localMaxUBTensor, scaleUBTensor, maxUBTensor[statsSlot], previousMax,
                                                 dmUBTensor[statsSlot], groupSumUBTensor,
                                                 rowsumUBTensor[statsSlot * 64], validRows, scaleOffset_, scaleBias_);
        } else {
            GbsaMxfp4VF::PScaleCallVF(localMaxUBTensor, scaleUBTensor, maxUBTensor[statsSlot], previousMax,
                                      dmUBTensor[statsSlot], groupSumUBTensor, rowsumUBTensor[statsSlot * 64],
                                      validRows, scaleOffset_, scaleBias_, scaleKv);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(SCALE_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(SCALE_EVENT);
        const uint32_t rows0 = (queryRows + 1) / 2;
        const uint32_t offset = bank * ((rows0 + 15) / 16 * 16) * layout.KV / 32;
        const uint32_t bytes = (validRows + 15) / 16 * 16 * layout.KV / 32;
        if (bytes != 0) {
            if (scaleKv == layout.KV) {
                AscendC::DataCopy(l1Scale[offset], scaleUBTensor,
                                  AscendC::DataCopyParams{1, static_cast<uint16_t>(bytes / 32), 0, 0});
            } else {
                // Compact UB statistics retain the full L1 row-block stride.
                AscendC::DataCopy(l1Scale[offset], scaleUBTensor,
                                  AscendC::DataCopyParams{static_cast<uint16_t>((validRows + 15U) / 16U),
                                                          static_cast<uint16_t>(scaleKv / 64U), 0,
                                                          static_cast<uint16_t>((layout.KV - scaleKv) / 64U)});
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SCALE_EVENT);
    }

private:
    bool firstTile_ = false;
    float scaleOffset_, scaleBias_;
    AscendC::LocalTensor<float> initialMaxUBTensor;
    AscendC::LocalTensor<float> localMaxUBTensor, groupSumUBTensor, rowsumUBTensor, maxUBTensor[4], dmUBTensor[4];
    AscendC::LocalTensor<uint8_t> scaleUBTensor;
};

} // namespace NpuArch::Epilogue::Block
#endif
