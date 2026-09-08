/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_CATLASS_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_CATLASS_H

#include <cstdint>

#include "allto_allv_grouped_mat_mul_aiv_comm.h"
#include "allto_allv_grouped_mat_mul_tiling.h"

namespace AlltoAllvGroupedMatMulCatlass {

struct GemmLaunchSpec {
    uint64_t offsetA = 0U;
    uint64_t offsetB = 0U;
    uint64_t offsetC = 0U;
    uint32_t m = 0U;
    uint32_t k = 0U;
    uint32_t n = 0U;
    uint32_t lda = 0U;
    uint32_t ldb = 0U;
    uint32_t ldc = 0U;
    bool transposeB = false;
    bool hasWork = false;
};

#if defined(__CCE_AICORE__)
#define A2AVGMM_CATLASS_HOST_DEVICE __aicore__ inline
#else
#define A2AVGMM_CATLASS_HOST_DEVICE inline
#endif

A2AVGMM_CATLASS_HOST_DEVICE bool IsSupportedTile(uint32_t m0, uint32_t k0, uint32_t n0)
{
    return (m0 == 128U && k0 == 64U && n0 == 128U) || (m0 == 128U && k0 == 32U && n0 == 128U) ||
           (m0 == 64U && k0 == 64U && n0 == 128U) || (m0 == 64U && k0 == 32U && n0 == 64U);
}

template <bool TRANSPOSE_B>
A2AVGMM_CATLASS_HOST_DEVICE bool BuildExpertGemmSpec(uint32_t expertIdx,
                                                     const AlltoAllvGroupedMatMulAiv::ExpertMeta &expert, uint32_t k,
                                                     uint32_t n, GemmLaunchSpec &spec)
{
    if (k == 0U || n == 0U || expert.recvTokenBase > 0xffffffffULL) {
        return false;
    }

    const uint64_t weightStride = AlltoAllvGroupedMatMulAiv::MulU32ToU64(k, n);
    uint64_t weightOffset = 0U;
    for (uint32_t index = 0U; index < expertIdx; ++index) {
        if (!AlltoAllvGroupedMatMulAiv::SafeAddU64(weightOffset, weightStride, weightOffset)) {
            return false;
        }
    }
    spec.offsetA = AlltoAllvGroupedMatMulAiv::MulU32ToU64(static_cast<uint32_t>(expert.recvTokenBase), k);
    spec.offsetC = AlltoAllvGroupedMatMulAiv::MulU32ToU64(static_cast<uint32_t>(expert.recvTokenBase), n);

    spec.offsetB = weightOffset;
    spec.m = expert.tokenCount;
    spec.k = k;
    spec.n = n;
    spec.lda = k;
    spec.ldb = TRANSPOSE_B ? k : n;
    spec.ldc = n;
    spec.transposeB = TRANSPOSE_B;
    spec.hasWork = expert.tokenCount != 0U;
    return true;
}

template <bool TRANSPOSE_B>
A2AVGMM_CATLASS_HOST_DEVICE bool BuildSharedGemmSpec(bool hasSharedExpert, uint32_t m, uint32_t k, uint32_t n,
                                                     GemmLaunchSpec &spec)
{
    if (!hasSharedExpert || m == 0U) {
        spec = {};
        return false;
    }
    if (k == 0U || n == 0U) {
        return false;
    }

    spec = {};
    spec.m = m;
    spec.k = k;
    spec.n = n;
    spec.lda = k;
    spec.ldb = TRANSPOSE_B ? k : n;
    spec.ldc = n;
    spec.transposeB = TRANSPOSE_B;
    spec.hasWork = true;
    return true;
}

#undef A2AVGMM_CATLASS_HOST_DEVICE

} // namespace AlltoAllvGroupedMatMulCatlass

#if defined(__CCE_AICORE__) && defined(__CCE_KT_TEST__)

namespace AlltoAllvGroupedMatMulCatlass {

template <typename Element, bool TRANSPOSE_B>
__aicore__ inline bool RunExpertGemm(GM_ADDR recvTokenBuffer, GM_ADDR weight, GM_ADDR output, uint32_t expertIdx,
                                     const AlltoAllvGroupedMatMulAiv::ExpertMeta &expert, const AlltoAllvGmmInfo &info,
                                     const AlltoAllvGmmCoCTiling &tiling)
{
    (void)recvTokenBuffer;
    (void)weight;
    (void)output;
    if ASCEND_IS_AIV {
        return true;
    }

    GemmLaunchSpec spec = {};
    return BuildExpertGemmSpec<TRANSPOSE_B>(expertIdx, expert, info.K, info.N, spec) &&
           (!spec.hasWork || IsSupportedTile(tiling.m0, tiling.k0, tiling.n0));
}

template <typename Element, bool TRANSPOSE_B>
__aicore__ inline bool RunSharedExpertGemm(GM_ADDR input, GM_ADDR weight, GM_ADDR output, const AlltoAllvGmmInfo &info,
                                           const AlltoAllvGmmCoCTiling &tiling)
{
    (void)input;
    (void)weight;
    (void)output;
    if ASCEND_IS_AIV {
        return true;
    }

    GemmLaunchSpec spec = {};
    if (!BuildSharedGemmSpec<TRANSPOSE_B>(info.hasSharedExpert != 0U, info.M, info.K, info.N, spec)) {
        return info.hasSharedExpert == 0U || info.M == 0U;
    }
    return IsSupportedTile(tiling.m0, tiling.k0, tiling.n0);
}

__aicore__ inline void CompileCheckCatlassWrappers(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                                   const AlltoAllvGmmInfo &info, const AlltoAllvGmmCoCTiling &tiling)
{
    if (false) {
        AlltoAllvGroupedMatMulAiv::ExpertMeta expert{};
        (void)RunExpertGemm<bfloat16_t, false>(input, weight, output, 0U, expert, info, tiling);
        (void)RunExpertGemm<half, true>(input, weight, output, 0U, expert, info, tiling);
        (void)RunSharedExpertGemm<bfloat16_t, false>(input, weight, output, info, tiling);
        (void)RunSharedExpertGemm<half, true>(input, weight, output, info, tiling);
    }
}

} // namespace AlltoAllvGroupedMatMulCatlass

#elif defined(__CCE_AICORE__)

#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace AlltoAllvGroupedMatMulCatlass {
namespace detail {

constexpr int32_t kL1TileK = 256;

template <typename Element, bool TRANSPOSE_B, int32_t TILE_M, int32_t TILE_K, int32_t TILE_N>
__aicore__ inline void RunElementGemm(const GemmLaunchSpec &spec, GM_ADDR ptrA, GM_ADDR ptrB, GM_ADDR ptrC)
{
    using namespace Catlass;
    using ArchTag = Arch::AtlasA2;
    using LayoutA = layout::RowMajor;
    using LayoutB = typename std::conditional<TRANSPOSE_B, layout::ColumnMajor, layout::RowMajor>::type;
    using LayoutC = layout::RowMajor;
    using AType = Gemm::GemmType<Element, LayoutA>;
    using BType = Gemm::GemmType<Element, LayoutB>;
    using CType = Gemm::GemmType<Element, LayoutC>;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<false>;
    using L1TileShape = GemmShape<TILE_M, TILE_N, kL1TileK>;
    using L0TileShape = GemmShape<TILE_M, TILE_N, TILE_K>;
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<1, 0>;
    using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, void, BlockScheduler>;

    static_assert(std::is_same<typename BlockMmad::ElementAccumulator, float>::value,
                  "FP16/BF16 CATLASS GEMM must accumulate in FP32.");

    LayoutA layoutA{spec.m, spec.k, spec.lda};
    LayoutB layoutB{spec.k, spec.n, spec.ldb};
    LayoutC layoutC{spec.m, spec.n, spec.ldc};
    GemmCoord problemShape{spec.m, spec.n, spec.k};
    typename MatmulKernel::Params params{problemShape, ptrA, layoutA, ptrB, layoutB, ptrC, layoutC};
    MatmulKernel matmul;
    matmul(params);
}

template <typename Element, bool TRANSPOSE_B>
__aicore__ inline bool RunConfiguredGemm(const GemmLaunchSpec &spec, const AlltoAllvGmmCoCTiling &tiling, GM_ADDR ptrA,
                                         GM_ADDR ptrB, GM_ADDR ptrC)
{
    if (!spec.hasWork) {
        return true;
    }
    if (tiling.m0 == 128U && tiling.k0 == 64U && tiling.n0 == 128U) {
        RunElementGemm<Element, TRANSPOSE_B, 128, 64, 128>(spec, ptrA, ptrB, ptrC);
    } else if (tiling.m0 == 128U && tiling.k0 == 32U && tiling.n0 == 128U) {
        RunElementGemm<Element, TRANSPOSE_B, 128, 32, 128>(spec, ptrA, ptrB, ptrC);
    } else if (tiling.m0 == 64U && tiling.k0 == 64U && tiling.n0 == 128U) {
        RunElementGemm<Element, TRANSPOSE_B, 64, 64, 128>(spec, ptrA, ptrB, ptrC);
    } else if (tiling.m0 == 64U && tiling.k0 == 32U && tiling.n0 == 64U) {
        RunElementGemm<Element, TRANSPOSE_B, 64, 32, 64>(spec, ptrA, ptrB, ptrC);
    } else {
        return false;
    }
    return true;
}

} // namespace detail

template <typename Element, bool TRANSPOSE_B>
__aicore__ inline bool RunExpertGemm(GM_ADDR recvTokenBuffer, GM_ADDR weight, GM_ADDR output, uint32_t expertIdx,
                                     const AlltoAllvGroupedMatMulAiv::ExpertMeta &expert, const AlltoAllvGmmInfo &info,
                                     const AlltoAllvGmmCoCTiling &tiling)
{
    if ASCEND_IS_AIV {
        return true;
    }

    GemmLaunchSpec spec = {};
    if (!BuildExpertGemmSpec<TRANSPOSE_B>(expertIdx, expert, info.K, info.N, spec)) {
        return false;
    }
    constexpr uint64_t elementBytes = sizeof(Element);
    return detail::RunConfiguredGemm<Element, TRANSPOSE_B>(spec, tiling, recvTokenBuffer + spec.offsetA * elementBytes,
                                                           weight + spec.offsetB * elementBytes,
                                                           output + spec.offsetC * elementBytes);
}

template <typename Element, bool TRANSPOSE_B>
__aicore__ inline bool RunSharedExpertGemm(GM_ADDR input, GM_ADDR weight, GM_ADDR output, const AlltoAllvGmmInfo &info,
                                           const AlltoAllvGmmCoCTiling &tiling)
{
    if ASCEND_IS_AIV {
        return true;
    }

    GemmLaunchSpec spec = {};
    if (!BuildSharedGemmSpec<TRANSPOSE_B>(info.hasSharedExpert != 0U, info.M, info.K, info.N, spec)) {
        return info.hasSharedExpert == 0U || info.M == 0U;
    }
    return detail::RunConfiguredGemm<Element, TRANSPOSE_B>(spec, tiling, input, weight, output);
}

__aicore__ inline void CompileCheckCatlassWrappers(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                                   const AlltoAllvGmmInfo &info, const AlltoAllvGmmCoCTiling &tiling)
{
    if (false) {
        AlltoAllvGroupedMatMulAiv::ExpertMeta expert{};
        (void)RunExpertGemm<bfloat16_t, false>(input, weight, output, 0U, expert, info, tiling);
        (void)RunExpertGemm<half, true>(input, weight, output, 0U, expert, info, tiling);
        (void)RunSharedExpertGemm<bfloat16_t, false>(input, weight, output, info, tiling);
        (void)RunSharedExpertGemm<half, true>(input, weight, output, info, tiling);
    }
}

} // namespace AlltoAllvGroupedMatMulCatlass

#endif // defined(__CCE_AICORE__) && defined(__CCE_KT_TEST__)

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_CATLASS_H
