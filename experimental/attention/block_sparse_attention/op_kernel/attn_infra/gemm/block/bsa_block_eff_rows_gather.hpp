/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_BLOCK_EFF_ROWS_GATHER_HPP
#define BSA_BLOCK_EFF_ROWS_GATHER_HPP

#include <cstdint>

#include <kernel_operator.h>

#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

namespace NpuArch::Gemm::Block {

// effectiveRows (per-block active rows)上下文，打包effRows相关参数避免函数签名过长
struct EffRowsCtx {
    AscendC::GlobalTensor<int32_t> gBlockEffRows;
    uint64_t gmOffset = 0;
    bool enabled = false;
    uint32_t *curBlockIdx = nullptr;
    uint32_t *curBlockCopied = nullptr;
};

// effRows 模式的搬运：按 effectiveY 逐 block 搬移填充目标 tile
// Transposed=true  用于 K（L1B 布局为 [embed, N]，tile 坐标 (0, dealtLenAccum)）
// Transposed=false 用于 V（L1B 布局为 [N, embed]，tile 坐标 (dealtLenAccum, 0)）
// 返回实际填充的累积长度 dealtLenAccum
template <bool Transposed, class TensorB, class TensorL1B, class CopyFn>
__aicore__ inline uint32_t EffRowsGatherCopy(TensorB &gBTensor, TensorL1B &l1BTensorTla,
                                             AscendC::GlobalTensor<int32_t> gSparseBlockIdx, const EffRowsCtx &ctx,
                                             uint32_t yBlockNumRsvd, uint32_t kvSeqlen, uint32_t blockShapeY,
                                             uint32_t embed, uint32_t targetLen, CopyFn copyFn)
{
    if (!ctx.enabled || ctx.curBlockIdx == nullptr || ctx.curBlockCopied == nullptr) {
        return 0;
    }
    uint32_t dealtLenAccum = 0;
    while (dealtLenAccum < targetLen && (*ctx.curBlockIdx) < yBlockNumRsvd) {
        uint32_t oriYBlockIdx = gSparseBlockIdx.GetValue(*ctx.curBlockIdx);
        uint32_t effectiveY = ctx.gBlockEffRows.GetValue(ctx.gmOffset + oriYBlockIdx * 2 + 1);
        // 先 clamp 到 blockShapeY，再 clamp 到尾块剩余行数（非尾块时第二步为 no-op）
        int64_t oriBlockStartY = static_cast<int64_t>(oriYBlockIdx) * blockShapeY;
        effectiveY = (effectiveY < blockShapeY) ? effectiveY : blockShapeY;
        uint32_t tailBlockY = (kvSeqlen > oriBlockStartY) ? static_cast<uint32_t>(kvSeqlen - oriBlockStartY) : 0;
        effectiveY = (effectiveY < tailBlockY) ? effectiveY : tailBlockY;
        uint32_t remaining = (effectiveY > (*ctx.curBlockCopied)) ? (effectiveY - (*ctx.curBlockCopied)) : 0;
        uint32_t toCopy = (remaining < (targetLen - dealtLenAccum)) ? remaining : (targetLen - dealtLenAccum);
        if (toCopy > 0) {
            uint32_t oriStartOffset = static_cast<uint32_t>(oriBlockStartY) + (*ctx.curBlockCopied);
            if constexpr (Transposed) {
                // K: L1B 布局 [embed, N]，tile 起始 (0, dealtLenAccum)，形状 (embed, toCopy)
                auto l1BTile =
                    tla::GetTile(l1BTensorTla, tla::MakeCoord(0, dealtLenAccum), tla::MakeShape(embed, toCopy));
                auto gBTile = tla::GetTile(gBTensor, tla::MakeCoord(0, oriStartOffset), tla::MakeShape(embed, toCopy));
                copyFn(l1BTile, gBTile);
            } else {
                // V: L1B 布局 [N, embed]，tile 起始 (dealtLenAccum, 0)，形状 (toCopy, embed)
                auto l1BTile =
                    tla::GetTile(l1BTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(toCopy, embed));
                auto gBTile = tla::GetTile(gBTensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(toCopy, embed));
                copyFn(l1BTile, gBTile);
            }
            dealtLenAccum += toCopy;
            (*ctx.curBlockCopied) += toCopy;
        }
        if ((*ctx.curBlockCopied) >= effectiveY) {
            (*ctx.curBlockIdx)++;
            (*ctx.curBlockCopied) = 0;
        }
    }
    return dealtLenAccum;
}

} // namespace NpuArch::Gemm::Block

#endif // BSA_BLOCK_EFF_ROWS_GATHER_HPP
