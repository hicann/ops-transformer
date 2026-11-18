/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_grad_tiling_data.h
 * \brief CompressorGrad arch22 tiling data：cast/scatter/gemm 的 init 参数与 workspace 偏移，
 *        host tiling 与 kernel 共用（与 pypto 生成的 CompressorGradTiling 区分命名）
 */

#ifndef COMPRESSOR_GRAD_TILING_DATA_A3_H
#define COMPRESSOR_GRAD_TILING_DATA_A3_H

#include <cstdint>

namespace optiling {
struct CompressorGradTilingDataA3 {
    int64_t batchSize = 0;    // B
    int64_t seqSize = 0;      // S
    int64_t tLen = 0;         // T = B * S
    int64_t tPad = 0;         // T padded up to a multiple of 128
    int64_t hiddenSize = 0;   // H
    int64_t headDim = 0;      // D
    int64_t totalHeadDim = 0; // C = coff * D
    int64_t cmpRatio = 0;     // cr
    int64_t coff = 0;         // 1 or 2
    int64_t cmpRowCnt = 0;    // N = coff * cr
    int64_t nblk = 0; // BSH: 调度用每 batch 块数——默认 S/cr（无值语义输入时的精确值）；seqused/startPos
                      // 提供时为安全上界 min(S/cr + 1, nblkIn)（实际块数由 kernel 按 GM 值逐 batch 算）；
                      // TH: 恒 0（块数完全由 kernel 运行时走查决定）。可为 0（S<cr 时全零输出，合法）
    int64_t nblkIn = 0;       // d_cmp_kv block dim (>= nblk)
    int64_t seqUsedValid = 0; // 1 = seqused 张量已提供且值可读，scatter 需逐 batch 读值截断实际块数
    int64_t startPosValid = 0; // 1 = start_pos 张量已提供且值可读，scatter 需逐 batch 读值算块锚点偏移(sp%cr)
    int64_t dnPlanes = 0;       // dn workspace planes: fp16 HiLo = 2, bf16 = 1
    int64_t castCoreNum = 0;    // AIV cores used by the cast/zero phase
    int64_t scatterCoreNum = 0; // AIV cores used by the scatter phase
    int64_t blocksPerCore = 0;  // scatter blocks per AIV core
    int64_t segLen = 0;         // scatter D-segment length (64/32/16)
    int64_t gemmDxCores = 0;    // gDx: AIC cores assigned to the d_x branch
    // workspace byte offsets from the user workspace base
    int64_t xF16WsOff = 0;     // x cast to fp16, blocked-NZ, tPad * H elements
    int64_t wkvF16WsOff = 0;   // wkv cast to fp16, blocked-NZ, C * H elements
    int64_t wgateF16WsOff = 0; // wgate cast to fp16, blocked-NZ, C * H elements
    int64_t dnKvWsOff = 0;     // dn_kv fp16, dnPlanes * tPad * C elements
    int64_t dnScWsOff = 0;     // dn_sc fp16, dnPlanes * tPad * C elements
    int64_t apePartWsOff = 0;  // d_ape per-core partials f32, apePartElems elements
    int64_t apePartElems = 0;  // scatterCoreNum * cmpRatio * totalHeadDim
    int64_t layout = 0;        // 0: BSH (x [B,S,H]), 1: TH (x [T,H] + cu_seqlens)
};
} // namespace optiling

#endif // COMPRESSOR_GRAD_TILING_DATA_A3_H
