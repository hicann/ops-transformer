/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP8_PACKED_QK_MMAD_HPP
#define GBSA_MXFP8_PACKED_QK_MMAD_HPP

#include "gbsa_mxfp8_tile.hpp"

namespace NpuArch::Gemm::Tile {

// Packed host dispatch proves D192/256; a covering L0 K tile computes each QK
// output with one MMAD, initialized from zero. The caller waits M_FIX before
// each Fixpipe and FIX_M before reusing its L0C bank. There is no L0C read for
// accumulation between these independent output slices.
//
// The documented small-MN PIPE_M requirement concerns multiple K fragments
// accumulating into the same L0C. Keep a defensive pre-barrier and the original
// MMAD behavior if an accumulating call is ever introduced by a future change.
template <class L0TileShape>
struct GbsaMxPackedQKSingleKMmad {
    static_assert(tla::get<2>(L0TileShape{}) == 128U || tla::get<2>(L0TileShape{}) == 256U,
                  "Packed single-K QK MMAD requires L0 K128 or K256");

    template <class C, class A, class B>
    __aicore__ inline void operator()(C const& c, A const& a, B const& b, uint32_t m, uint32_t n, uint32_t k,
                                      bool initC = true)
    {
        if (!initC) {
            // The preceding initialized call intentionally had no post-barrier.
            // Synchronize before any newly introduced accumulator read instead.
            AscendC::PipeBarrier<PIPE_M>();
            GbsaMxMmad original;
            original(c, a, b, m, n, k, initC);
            return;
        }
        AscendC::MmadParams p;
        p.m = m;
        p.n = n;
        p.k = k;
        p.cmatrixInitVal = true;
        p.disableGemv = true;
        AscendC::Mmad(c.data(), a.data().template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                      b.data().template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(), p);
    }
};

} // namespace NpuArch::Gemm::Tile
#endif
