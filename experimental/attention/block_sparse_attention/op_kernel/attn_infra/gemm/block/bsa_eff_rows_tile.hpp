/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BSA_EFF_ROWS_TILE_HPP
#define BSA_EFF_ROWS_TILE_HPP

#include <cstdint>

namespace NpuArch::Gemm::Block {

// 把块的有效行数先 clamp 到 blockShapeY，再 clamp 到尾块剩余行数(非尾块时第二步为 no-op)
__aicore__ inline uint32_t ClampEffectiveRows(uint32_t effectiveRows, uint32_t oriBlockIdx, uint32_t blockShapeY,
                                              int64_t kvSeqlen)
{
    int64_t oriBlockStartRow = static_cast<int64_t>(oriBlockIdx) * blockShapeY;
    uint32_t blockValidRows = effectiveRows < blockShapeY ? effectiveRows : blockShapeY;
    return kvSeqlen <= oriBlockStartRow ?
               0 :
               static_cast<uint32_t>(kvSeqlen - oriBlockStartRow < blockValidRows ? kvSeqlen - oriBlockStartRow :
                                                                                    blockValidRows);
}

// per-tile 下一次搬运的 kv 片段
struct EffRowsTile {
    uint32_t oriBlockIdx = 0; // 原始块号, 同时用于索引该块的反量化 scale。
    int64_t oriSeqOffset = 0; // 该片段在原始 kv 序列中的起始行偏移。
    uint32_t validRows = 0;   // 该片段的有效行数(<= kvBaseTile)。
};

// QK 与延后的 PV 在 AIC/AIV 上各使用独立游标。
// 一个 tile 不跨量化块(含未填满的块)。
template <class SparseTensor, class RowsTensor>
__aicore__ inline EffRowsTile NextEffRowsTile(SparseTensor gSparseBlockIdx, RowsTensor gBlockEffRows,
                                              uint64_t effRowsBase, uint32_t yBlockNumRsvd, uint32_t blockShapeY,
                                              int64_t kvSeqlen, uint32_t tileMaxRows, uint32_t &curBlockIdx,
                                              uint32_t &curBlockCopied)
{
    while (curBlockIdx < yBlockNumRsvd) {
        uint32_t oriBlockIdx = gSparseBlockIdx.GetValue(curBlockIdx);
        uint32_t effectiveRows = ClampEffectiveRows(gBlockEffRows.GetValue(effRowsBase + oriBlockIdx * 2 + 1),
                                                    oriBlockIdx, blockShapeY, kvSeqlen);
        if (curBlockCopied >= effectiveRows) {
            ++curBlockIdx;
            curBlockCopied = 0;
            continue;
        }
        uint32_t blockRemainRows = effectiveRows - curBlockCopied;
        uint32_t curTileRows = blockRemainRows < tileMaxRows ? blockRemainRows : tileMaxRows;
        EffRowsTile curTile{oriBlockIdx, static_cast<int64_t>(oriBlockIdx) * blockShapeY + curBlockCopied, curTileRows};
        curBlockCopied += curTileRows;
        if (curBlockCopied == effectiveRows) {
            ++curBlockIdx;
            curBlockCopied = 0;
        }
        return curTile;
    }
    return {};
}

} // namespace NpuArch::Gemm::Block
#endif // BSA_EFF_ROWS_TILE_HPP
