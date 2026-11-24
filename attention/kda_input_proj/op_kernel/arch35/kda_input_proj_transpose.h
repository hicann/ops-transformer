/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KDA_INPUT_PROJ_TRANSPOSE_H
#define KDA_INPUT_PROJ_TRANSPOSE_H

#include "kernel_operator.h"

namespace KdaInputProj {
namespace {
constexpr uint32_t TRANSPOSE_UB_ROW_BYTES = 128;
constexpr uint32_t TRANSPOSE_DMA_ALIGN = 32;

#ifndef __DAV_CUBE__
template <typename T, uint32_t TILE>
__aicore__ inline void TransposeQkvTile(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t rows,
                                        uint16_t cols)
{
    auto srcAddr = reinterpret_cast<__ubuf__ T*>(src.GetPhyAddr());
    auto dstAddr = reinterpret_cast<__ubuf__ uint16_t*>(dst.GetPhyAddr());
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int16_t> rowOffset;
        AscendC::Reg::RegTensor<uint16_t> data;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<uint16_t>(rows);
        AscendC::Reg::Arange(rowOffset, static_cast<int16_t>(0));
        AscendC::Reg::Muls(rowOffset, rowOffset, static_cast<int16_t>(TILE), mask);
        for (uint16_t col = 0; col < cols; ++col) {
            AscendC::Reg::DataCopyGather(data, srcAddr + col,
                                         reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(rowOffset), mask);
            if constexpr (sizeof(T) == 1) {
                AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(
                    dstAddr + col * (TILE / sizeof(uint16_t)), data, mask);
            } else {
                AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM_B16>(dstAddr + col * TILE, data,
                                                                                           mask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void TransposeQkvWeight(__gm__ uint8_t* src, __gm__ uint8_t* dst, uint32_t rows, uint32_t cols)
{
    using namespace AscendC;
    constexpr uint32_t ELEM_BYTES = sizeof(T);
    constexpr uint32_t TILE = TRANSPOSE_UB_ROW_BYTES / ELEM_BYTES;
    constexpr uint32_t TILE_ELEMS = TILE * TILE;
    const uint64_t rowTiles = (static_cast<uint64_t>(rows) + TILE - 1U) / TILE;
    const uint64_t colTiles = (static_cast<uint64_t>(cols) + TILE - 1U) / TILE;
    const uint32_t aivNum = GetBlockNum() * GetTaskRation();
    GlobalTensor<T> srcGm;
    GlobalTensor<T> dstGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), static_cast<uint64_t>(rows) * cols);
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst), static_cast<uint64_t>(rows) * cols);
    LocalTensor<T> srcUb(TPosition::VECIN, 0, TILE_ELEMS);
    LocalTensor<T> dstUb(TPosition::VECOUT, TILE_ELEMS * sizeof(T), TILE_ELEMS);
    const DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (uint64_t tile = GetBlockIdx(); tile < rowTiles * colTiles; tile += aivNum) {
        const uint64_t row = tile / colTiles * TILE;
        const uint64_t col = tile % colTiles * TILE;
        const uint32_t tileRows = (rows - row < TILE) ? rows - row : TILE;
        const uint32_t tileCols = (cols - col < TILE) ? cols - col : TILE;
        const uint32_t rowBytes = tileRows * ELEM_BYTES;
        const uint32_t colBytes = tileCols * ELEM_BYTES;
        // Normal DataCopyPad rounds each UB row to 32 bytes. Extend the stride to the fixed tile pitch.
        const DataCopyExtParams copyIn{static_cast<uint16_t>(tileRows), colBytes, (cols - tileCols) * ELEM_BYTES,
                                       (TRANSPOSE_UB_ROW_BYTES - colBytes) / TRANSPOSE_DMA_ALIGN, 0};
        const DataCopyExtParams copyOut{static_cast<uint16_t>(tileCols), rowBytes,
                                        (TRANSPOSE_UB_ROW_BYTES - rowBytes) / TRANSPOSE_DMA_ALIGN,
                                        (rows - tileRows) * ELEM_BYTES, 0};
        DataCopyPad(srcUb, srcGm[row * cols + col], copyIn, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        TransposeQkvTile<T, TILE>(dstUb, srcUb, tileRows, tileCols);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dstGm[col * rows + row], dstUb, copyOut);
        // Finish writeback before the next tile or stage reuses UB.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}
#endif
} // namespace
} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_TRANSPOSE_H
