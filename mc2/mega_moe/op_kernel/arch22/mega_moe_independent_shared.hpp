/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_INDEPENDENT_SHARED_HPP
#define MEGA_MOE_INDEPENDENT_SHARED_HPP

namespace MegaMoeImpl {
// A serial, fully drained stage within the same mixed kernel. Routing keeps its
// own type/layout and original fast path; this stage owns the shared types/I.
template <typename Weight, bool Nz, bool Transpose, bool Mixed, bool SharedW2Nz>
__aicore__ inline GM_ADDR IndependentShared(const MegaMoeA2A3TilingData& common, GM_ADDR input, GM_ADDR workspace,
                                            GM_ADDR w1, GM_ADDR w2, GM_ADDR s1, GM_ADDR s2, GM_ADDR b1, GM_ADDR b2,
                                            uint32_t hidden, uint32_t activation, float clamp, float param1,
                                            float param2)
{
    using namespace Catlass;
    constexpr bool w4 = std::is_same_v<Weight, AscendC::int4b_t> || std::is_same_v<Weight, int32_t>;
    constexpr bool w8 = std::is_same_v<Weight, int8_t>;
    using B = std::conditional_t<w4, AscendC::int4b_t, Weight>;
    using A = std::conditional_t<w4, AscendC::int4b_t, std::conditional_t<w8, int8_t, bfloat16_t>>;
    using C = std::conditional_t < w8 && !Mixed, int32_t, std::conditional_t < w4 || w8, float16_t, bfloat16_t >> ;
    using D = std::conditional_t<Mixed || !(w4 || w8), bfloat16_t, int8_t>;
    using LA = layout::RowMajor;
    using LB = std::conditional_t<Nz, layout::zN, std::conditional_t<Transpose, layout::ColumnMajor, LA>>;
    using AT = Gemm::GemmType<A, LA>;
    using BT = Gemm::GemmType<B, LB>;
    using CT = Gemm::GemmType<C, LA>;
    using DT = Gemm::GemmType<D, LA>;
    using Out = Gemm::GemmType<bfloat16_t, LA>;
    using Scale = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using TokenScale = Gemm::GemmType<float, layout::VectorLayout>;
    using Policy = Gemm::MmadDispatchPolicyFor_t<B, false, 1, 2, 2, 2, 1, false, true>;
    using Tile = std::conditional_t<w4, GemmShape<128, 256, 1024>,
                                    std::conditional_t<w8, GemmShape<128, 256, 512>, GemmShape<128, 256, 256>>>;
    using L0 = std::conditional_t<w4, GemmShape<128, 256, 256>,
                                  std::conditional_t<w8, GemmShape<128, 256, 128>, GemmShape<128, 256, 64>>>;
    using QuantCopy =
        Gemm::Tile::QuantTileCopy<Arch::AtlasA2, AT, BT, CT, void, Gemm::Tile::ScaleGranularity::PER_CHANNEL>;
    using Mmad = std::conditional_t<w4, Gemm::Block::BlockMmad<Policy, Tile, L0, AT, BT, CT, void, QuantCopy>,
                                    Gemm::Block::BlockMmad<Policy, Tile, L0, AT, BT, CT>>;
    using Mul = Epilogue::Tile::TileElemWiseMuls<Arch::AtlasA2, Gemm::GemmType<float, LA>, 0>;
    using Copy1 = Epilogue::Tile::TileCopy<Arch::AtlasA2, CT, Scale, TokenScale, DT>;
    using Copy2 = Epilogue::Tile::TileCopy<Arch::AtlasA2, CT, Scale, TokenScale, Out>;
    using P1 = std::conditional_t<w4, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<2>,
                                  Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<2>>;
    using P2 = std::conditional_t<w4, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<2>,
                                  Epilogue::EpilogueAtlasA2PerTokenDequant<2>>;
    using E1 = Epilogue::Block::BlockEpilogue<P1, CT, TokenScale, DT, Mul, Copy1>;
    using E2 = std::conditional_t<
        w4, Epilogue::Block::BlockEpilogue<P2, CT, TokenScale, Out, Copy2, Epilogue::Block::IS_A2_T<false>>,
        Epilogue::Block::BlockEpilogue<P2, CT, TokenScale, Out, Copy2>>;
    using Bf = Gemm::GemmType<bfloat16_t, LA>;
    using LBF = std::conditional_t<Mixed && !SharedW2Nz, LA, LB>;
    using BfW = Gemm::GemmType<bfloat16_t, LBF>;
    using BfPolicy = Gemm::MmadDispatchPolicyFor_t<bfloat16_t, false, 1, 2, 2, 2, 1, false, true>;
    using BfMmad = Gemm::Block::BlockMmad<BfPolicy, GemmShape<128, 256, 256>, GemmShape<128, 256, 64>, Bf, BfW, Bf>;
    using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using Kernel = Gemm::Kernel::SharedExpert<Mmad, Scheduler, E1, E2, BfMmad, Mixed>;
    const uint32_t n = common.sharedN;
    typename Kernel::Params params{GemmCoord{common.M, n, hidden},
                                   input,
                                   workspace,
                                   w1,
                                   w2,
                                   s1,
                                   s2,
                                   b1,
                                   b2,
                                   LA{common.M, n / 2},
                                   LayoutBInitializer<LB, B>::create(hidden, n),
                                   LayoutBInitializer<LB, B>::create(n / 2, hidden),
                                   LayoutBInitializer<LBF, bfloat16_t>::create(n / 2, hidden),
                                   layout::VectorLayout{n},
                                   layout::VectorLayout{hidden},
                                   common.sharedWorkspaceOffset,
                                   common.sharedExpertNum,
                                   common.aivNum,
                                   clamp,
                                   activation,
                                   param1,
                                   param2};
    Kernel kernel(params);
    kernel.Process(params);
    return kernel.Result();
}
} // namespace MegaMoeImpl
#endif
