/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEXT_GATED_RESIDUAL_GRAD_NN_A5_TILING_H
#define TEXT_GATED_RESIDUAL_GRAD_NN_A5_TILING_H

#include <algorithm>
#include <cstdint>
#include "../../../op_kernel/arch35/text_gated_residual_grad_pipeline_arch35.h"
#include "../../../op_kernel/arch35/text_gated_residual_grad_tiling_data_arch35.h"

namespace HcGradA5Tiling {
namespace hcgrad = TextGatedResidualGrad;

constexpr uint32_t BASIC_BASE_M = 128U;
constexpr uint32_t BASIC_BASE_N = 256U;
constexpr uint32_t BASIC_BASE_K = 64U;
constexpr uint8_t ASW_WINDOW_LENGTH = 4U;
constexpr uint32_t L0C_DOUBLE_BUFFER_TILE_BYTES = 131072U; // 128 KiB per FP32 accumulator.
constexpr uint32_t STRIDED_ND_SLICE_ROWS = 1U;
constexpr uint32_t CONTIGUOUS_ND_STRIDE = 1U;

constexpr uint32_t TUNED_MIN_ROWS = 512U;
constexpr uint32_t C1A_FULL_CHUNK_BASE_K = 32U;
constexpr uint32_t C1A_WIDE_TAIL_MIN_ROWS = 1536U;
constexpr uint32_t C1A_WIDE_TAIL_MAX_BASE_M = 256U;
constexpr uint32_t C1A_WIDE_TAIL_BASE_N = 192U;
constexpr uint32_t C1A_WIDE_TAIL_N_TILES = 2U;
constexpr uint32_t C1A_MEDIUM_TAIL_BASE_N = 128U;
constexpr uint32_t C1A_SMALL_TAIL_BASE_N = 64U;
constexpr uint32_t C1A_SHORT_TAIL_BASE_K = 128U;
constexpr uint32_t DX_BASE_M = 256U;
constexpr uint32_t DX_BASE_N = 256U;
constexpr uint32_t DX_SMALL_M_TAIL_SPLIT_LIMIT = 2U;
constexpr uint32_t DX_LARGE_M_TAIL_SPLITS = 3U;
constexpr uint32_t DX_LARGE_N_TAIL_SPLITS = 2U;
constexpr uint32_t DX_LARGE_TAIL_FANOUT = DX_LARGE_M_TAIL_SPLITS * DX_LARGE_N_TAIL_SPLITS;

inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return value / divisor + static_cast<bool>(value % divisor);
}

inline HcGradA5NnTiling MakePlan(uint32_t m, uint32_t n, uint32_t k, uint32_t lda, uint32_t cores, uint32_t stage,
                                 bool tuned)
{
    const bool transA = stage == hcgrad::HC_GRAD_NN_STAGE_C1B || stage == hcgrad::HC_GRAD_NN_STAGE_G56;
    HcGradA5NnTiling plan{};
    plan.m = m;
    plan.n = n;
    plan.k = plan.skSingleCoreK = k;
    plan.baseM = hcgrad::HcGradAlign16(std::min(m, BASIC_BASE_M));
    plan.baseN = hcgrad::HcGradAlign16(std::min(n, BASIC_BASE_N));
    plan.baseK = hcgrad::HcGradAlign16(std::min(k, BASIC_BASE_K));
    plan.mTailCnt = plan.nTailCnt = hcgrad::HC_GRAD_A5_UNSPLIT_TAIL;
    plan.mBaseTailSplitCnt = plan.nBaseTailSplitCnt = hcgrad::HC_GRAD_A5_UNSPLIT_TAIL;
    plan.l1BufferNum = hcgrad::HC_GRAD_A5_DOUBLE_BUFFER;
    plan.ubDB = hcgrad::HC_GRAD_A5_DOUBLE_BUFFER;
    plan.aswWindowLen = ASW_WINDOW_LENGTH;

    // Keep the full-chunk Cube/Vector fast paths on the 28-AIC model.
    if (tuned && stage == hcgrad::HC_GRAD_NN_STAGE_C1A) {
        if (m == hcgrad::HC_GRAD_CHUNK_ROWS) {
            plan.baseM = hcgrad::HC_GRAD_C1A_UB_TILE_M;
            plan.baseN = hcgrad::HC_GRAD_C1A_UB_TILE_N;
            plan.baseK = C1A_FULL_CHUNK_BASE_K;
            plan.l2CacheDisable = true;
        } else if (m >= C1A_WIDE_TAIL_MIN_ROWS) {
            const uint32_t mTiles = cores / C1A_WIDE_TAIL_N_TILES;
            plan.baseM = std::min(hcgrad::HcGradAlign16(CeilDiv(m, mTiles)), C1A_WIDE_TAIL_MAX_BASE_M);
            plan.baseN = C1A_WIDE_TAIL_BASE_N;
        } else {
            plan.baseN = m >= TUNED_MIN_ROWS ? C1A_MEDIUM_TAIL_BASE_N : C1A_SMALL_TAIL_BASE_N;
            plan.baseK = C1A_SHORT_TAIL_BASE_K;
        }
    }
    if (tuned && transA && k >= TUNED_MIN_ROWS) {
        plan.baseM = hcgrad::HC_GRAD_A5_WEIGHT_TILE_M;
        plan.baseN = hcgrad::HC_GRAD_A5_WEIGHT_TILE_N;
        plan.baseK = hcgrad::HC_GRAD_A5_WEIGHT_TILE_K;
        if (stage == hcgrad::HC_GRAD_NN_STAGE_G56) {
            plan.mTailCnt = hcgrad::HC_GRAD_A5_G56_M_TAIL_SPLITS;
        } else if (k == hcgrad::HC_GRAD_CHUNK_ROWS) {
            plan.l2CacheDisable = true;
        }
    }
    if (tuned && stage == hcgrad::HC_GRAD_NN_STAGE_DX && m >= TUNED_MIN_ROWS) {
        plan.baseM = DX_BASE_M;
        plan.baseN = DX_BASE_N;
        plan.ubDB = hcgrad::HC_GRAD_A5_SINGLE_BUFFER;
        const uint32_t remainder = (CeilDiv(m, DX_BASE_M) * CeilDiv(n, DX_BASE_N)) % cores;
        const uint32_t fanout = remainder ? cores / remainder : hcgrad::HC_GRAD_A5_UNSPLIT_TAIL;
        const bool splitBothAxes = fanout >= DX_LARGE_TAIL_FANOUT;
        plan.mTailCnt = splitBothAxes ? DX_LARGE_M_TAIL_SPLITS : std::min(fanout, DX_SMALL_M_TAIL_SPLIT_LIMIT);
        plan.nTailCnt = splitBothAxes ? DX_LARGE_N_TAIL_SPLITS : hcgrad::HC_GRAD_A5_UNSPLIT_TAIL;
    }

    plan.mL1 = plan.baseM;
    plan.nL1 = plan.baseN;
    plan.kL1 = hcgrad::HC_GRAD_A5_L1_K_TILES * plan.baseK;
    plan.l0cDB = (plan.baseM * plan.baseN * sizeof(float) <= L0C_DOUBLE_BUFFER_TILE_BYTES) ?
                     hcgrad::HC_GRAD_A5_DOUBLE_BUFFER :
                     hcgrad::HC_GRAD_A5_SINGLE_BUFFER;
    const uint64_t blocks = static_cast<uint64_t>(CeilDiv(m, plan.mL1)) * CeilDiv(n, plan.nL1);
    plan.usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(cores, blocks));
    const bool hasNdRowPadding = !transA && lda != k;
    plan.sliceM = hasNdRowPadding ? STRIDED_ND_SLICE_ROWS : plan.mL1;
    plan.srcNdStride = hasNdRowPadding ? lda : CONTIGUOUS_ND_STRIDE;
    plan.rowStride = lda;
    return plan;
}
} // namespace HcGradA5Tiling

#endif // TEXT_GATED_RESIDUAL_GRAD_NN_A5_TILING_H
