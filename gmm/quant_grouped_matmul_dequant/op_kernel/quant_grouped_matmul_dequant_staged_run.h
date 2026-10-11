/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_dequant_staged_run.h
 * \brief Staged-path dispatch shim (separate stack frame).
 *
 * The unified-kernel entry sits at the 16000-byte aicore frame ceiling, so
 * the staged grouped object cannot live in the entry. RunStagedB0 is a
 * non-inline device fn with its own frame: it builds the staged tiling
 * struct (most fields = our TilingData; the ~9 host-derived fields are
 * recomputed on-device with master's formulas) and runs the vendored master
 * grouped kernel. Phase 1 baseline = master split-distribution; perf levers
 * layer on top in Phase 2.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_STAGED_RUN_H
#define QUANT_GROUPED_MATMUL_DEQUANT_STAGED_RUN_H

#include "quant_grouped_matmul_dequant_tiling_key.h"     // QgmmdFlagBit / GetFlag
#include "quant_grouped_matmul_dequant_staged_grouped.h" // qgmmd_staged kernel + tiling struct

namespace AscendC {

// Map our TilingData → master staged struct. Derived fields recomputed with
// master tiling.cpp GROUPED-branch formulas (UB=256KB constant on 310P).
__aicore__ inline void BuildStagedTiling(const QuantMatmulDequantTilingData* __restrict td,
                                         qgmmd_staged::QuantGroupedMatmulDequantTilingData& s)
{
    constexpr uint32_t UB = 256u * 1024u;
    constexpr uint32_t REDUCE_THRES = 32u * 1024u;
    const bool perTok = GetFlag(td->flags, FLAG_BIT_PER_TOKEN);
    const bool dyn = GetFlag(td->flags, FLAG_BIT_DYNAMIC_QUANT);
    const bool smooth = GetFlag(td->flags, FLAG_BIT_HAS_SMOOTH);

    s.CoreNum = td->coreNum;
    s.perToken = perTok;
    s.dynamicQuant = dyn;
    s.smoothScale = smooth;
    s.originE = td->E;
    s.originM = td->M;
    s.originN = td->N;
    s.originK = td->K;
    s.originKAligned32 = td->originKAligned32;
    s.originKAligned512 = td->originKAligned512;
    s.fracN = td->fracN;
    s.fracK = td->fracK;
    s.fracM = td->fracM;
    s.tailM = (td->M - 1u) % 16u + 1u;
    s.ubKMask = td->ubKMask;
    s.isXScaleHalf = 0u;
    s.MMmod = 0u;

    // GROUPED host fields (master tiling.cpp:346-352).
    s.processXKBaseNMax =
        (UB - (perTok ? 16u * 32u : 0u)) / (16u * 2u + 16u * 2u + 16u * 4u + (smooth ? (2u + 4u) : 0u)) / 256u * 256u;
    s.singleCoreFracN = (td->fracN + td->coreNum - 1u) / td->coreNum;
    s.singleCoreFracNTail = (td->fracN - 1u) % td->coreNum + 1u;
    s.baseFracK = 16u;
    s.baseFracN = 4u;

    // dynamic scale (master tiling.cpp:439-444).
    uint32_t dBaseK = (UB / 2u - 32u * 2u - 32u * 2u) / (1u + (smooth ? 1u : 0u)) / 2u / 256u * 256u;
    if (dBaseK > REDUCE_THRES)
        dBaseK = REDUCE_THRES;
    s.dynamicBaseK = dBaseK;
    s.dynamicIterK = (td->K + dBaseK - 1u) / dBaseK;
    s.dynamicBaseKTail = (td->K - 1u) % dBaseK + 1u;

    // Computed in-kernel by TilingInKernelNormal; zero is fine.
    s.singleCoreFracNTail = (td->fracN - 1u) % td->coreNum + 1u;
    s.MCoreNum = 0u;
    s.NCoreNum = 0u;
    s.singleCoreM = 0u;
    s.singleCoreN = 0u;
    s.singleCoreMTail = 0u;
    s.singleCoreNTail = 0u;
    s.baseMNum = 0u;
    s.baseNNum = 0u;
    s.baseKNum = 0u;
    s.swiftGEMVThreshold = 0u;
    s.baseNNum_2 = 0u;
    s.baseKNum_2 = 0u;
    s.baseK = 0u;
    s.baseKTail = 0u;
    s.baseK_2 = 0u;
    s.baseKTail_2 = 0u;
    s.baseFracNL0C = 0u;
    s.ubBaseK = 0u;
    s.ubIterK = 0u;
    s.ubBaseKTail = 0u;
}

// Own frame (entry is at the 16000 ceiling). __noinline__ keeps the staged
// object out of the entry; it sits in this fn's separate frame.
__aicore__ inline __attribute__((noinline)) void RunStagedB0(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale,
                                                             GM_ADDR group_list, GM_ADDR bias, GM_ADDR x_scale,
                                                             GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y,
                                                             GM_ADDR usr_workspace,
                                                             const QuantMatmulDequantTilingData* __restrict td)
{
    qgmmd_staged::QuantGroupedMatmulDequantTilingData s;
    BuildStagedTiling(td, s);
    qgmmd_staged::QuantMatmulDequantGrouped op;
    op.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, y, usr_workspace, &s);
    op.Process();
}

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_STAGED_RUN_H
