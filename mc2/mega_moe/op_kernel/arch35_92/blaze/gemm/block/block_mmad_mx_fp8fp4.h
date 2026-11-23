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
 * \file block_mmad_mx_fp8fp4.h
 * \brief MMAD block implementation for weight-quant grouped matmul split-M pipeline.
 */
#pragma once
#include "../../../common/mega_moe_constants.h"
#include "../../../common/mega_moe_sync.h"
#include "../tile/copy_gmm1_concat.h"
#include "../policy/dispatch_policy_mega_moe.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "kernel_basic_intf.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"
#include "../utils/common_utils_mega_moe.h"
#include "../../../mega_moe_tiling.h"

namespace Blaze {
namespace Gemm {
namespace Block {

using AscendC::BLOCK_CUBE;
using AscendC::TEventID;
using MegaMoeImpl::SyncIntraArrive;
using MegaMoeImpl::PipeFence;
using MegaMoeImpl::PipeNotify;
using MegaMoeImpl::PipeWait;
using MegaMoeImpl::SyncIntraWait;
using Blaze::Gemm::DOUBLE_BUFFER;
using Blaze::Gemm::FINAL_ACCUMULATION;
using Blaze::Gemm::MXFP_DIVISOR_SIZE;
using Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
using Blaze::Gemm::NON_FINAL_ACCUMULATION;

// Macro aliases keep the specialization declaration compact for this dispatch-policy binding.
#define BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS \
    template <class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_, class CType_, \
              class LayoutC_, class BiasType_, class LayoutBias_>

#define BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION \
    BlockMmad<MatmulMxFp8Fp4DynamicKL1TailResplit, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_, CType_, \
              LayoutC_, BiasType_, LayoutBias_>

/*!
 * \brief AIC tile-compute unit for one tile inside one group in the weight-quant grouped matmul pipeline.
 *
 * Design reason:
 * - This class handles MMAD compute for a single-group tile only.
 * - AIC does not support direct FP4E2M1 -> FP8E4M3 conversion in the MMAD path.
 * - Therefore B must be preprocessed by prologue first, and this class consumes the converted B tiles.
 *
 * Distinctive behaviors:
 * 1) It synchronizes with prologue through cross-core flags, and the sync semantics must match prologue exactly.
 * 2) Host tiling chooses A/B buffer counts and the shared scale K window.
 * 3) A and B use K256 L1 tiles, including M/N tails; scales use two larger K windows.
 * 4) Each L0 B tile is reused across M strips, with independent L0A/L0B rotation.
 *
 * Key constraints:
 * 1) A must satisfy ND format.
 * 2) B must be prologue-converted ZN format and use the same compute type as AType.
 * 3) Scale layout must satisfy MXFP_DIVISOR_SIZE = 64.
 *
 * When to use:
 * - Use this block on the mxfp8fp4 path when pipelined L1 copies and L0 M-strip reuse are needed to increase per-tile
 * data workload and overall throughput.
 */
BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
class BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION {
public:
    using DispatchPolicy = MatmulMxFp8Fp4DynamicKL1TailResplit;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using AType = typename AscendC::Std::tuple_element<0, ATypeTuple_>::type;
    using ScaleBType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using ScaleAType = typename AscendC::Std::tuple_element<1, ATypeTuple_>::type;
    using CType = CType_;

    using LayoutA = typename AscendC::Std::tuple_element<0, LayoutATuple_>::type;
    using LayoutScaleA = typename AscendC::Std::tuple_element<1, LayoutATuple_>::type;
    using LayoutB = typename AscendC::Std::tuple_element<0, LayoutBTuple_>::type;
    using LayoutScaleB = typename AscendC::Std::tuple_element<1, LayoutBTuple_>::type;
    using LayoutC = LayoutC_;

    static_assert(asc::te::is_satisfied_ptn_format_v<
                  decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ AType*)0),
                                                LayoutA{}(0UL, 0UL))),
                  asc::te::nd_ext_layout_ptn>);

    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t SCALE_C0 = 2;
    constexpr static int32_t L0C_C0 = 16;

    // Common scheduler type slot; A8W4 L1 parameters are supplied only by host layout.
    struct L1Params {};

    template <bool ResetBufferIndices = true>
    __aicore__ inline void Init(const ProblemShape& problemShape, const BlockShape& l0TileShape,
                                bool interleave = false);

    __aicore__ inline BlockMmad(const MegaMoeL1Layout& layout);
    // When copyUbToV1 is true, L0C->UB copies the result to V1 core instead of V0.
    // Only meaningful when C tensor resides in UB; ignored for GM output.
    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(const TensorA& tensorA, const TensorScaleA& tensorScaleA,
                                      const TensorScaleB& tensorScaleB, const TensorC& tensorC,
                                      bool copyUbToV1 = false);
    __aicore__ inline ~BlockMmad();

private:
    struct BlockMmadOffsetParam {
        uint64_t mL1Size;
        uint64_t kaL1Size;
        uint64_t kbL1Size;
        uint64_t nL1Size;
        uint64_t kSize;
        uint64_t scaleKL1Size;
        uint64_t l0KSize;
    };

    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();

    template <typename TensorC>
    __aicore__ inline void ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize, const BlockMmadOffsetParam& param,
                                         const TensorC& tensorC, bool copyUbToV1);
    __aicore__ inline void CopyAAndScaleAL1ToL0(uint64_t l1KOffset, uint64_t kbOffset, uint64_t realL0K,
                                                uint64_t realL0ScaleK, uint64_t mOffset, uint64_t mSize,
                                                const BlockMmadOffsetParam& param);
    template <typename TensorBL1Type>
    __aicore__ inline void CopyBAndScaleBL1ToL0(const TensorBL1Type& tensorBL1, uint64_t l1KOffset, uint64_t kbOffset,
                                                uint64_t realL0K, uint64_t realL0ScaleK,
                                                const BlockMmadOffsetParam& param);
    template <typename TensorC>
    __aicore__ inline auto GetBlockMmadOffsetParam(const TensorC& tensorC) const;
    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void ProcessTilesL1(const TensorA& tensorA, const TensorScaleA& tensorScaleA,
                                          const TensorScaleB& tensorScaleB, const TensorC& tensorC,
                                          const BlockMmadOffsetParam& blockParam, bool copyUbToV1);
    __aicore__ inline void WaitAMTE1ToMTE2();
    __aicore__ inline void SetMTE1ToMTE2();
    __aicore__ inline void WaitScaleMTE1ToMTE2();
    __aicore__ inline void SetScaleMTE1ToMTE2();
    template <typename TensorA>
    __aicore__ inline auto CopyAGmToL1(const TensorA& tensorA, const BlockMmadOffsetParam& param, int64_t kaGmOffset,
                                       uint64_t slot);
    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyMxScaleGmToL1(const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB,
                                             const BlockMmadOffsetParam& param, uint64_t kbL1Offset, uint64_t slot);

    template <typename TensorC>
    __aicore__ inline void CopyCL0c2GmOrUb(const TensorC& tensorC, const BlockMmadOffsetParam& param, bool copyUbToV1,
                                           uint64_t mOffset, uint64_t mSize);

    using MakeLayoutAL1 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL1 =
        typename asc::te::frame_layout_format<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL1 =
        typename asc::te::frame_layout_format<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>;

    using MakeLayoutAL0 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutBL0 = asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL0 =
        typename asc::te::frame_layout_format<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL0 =
        typename asc::te::frame_layout_format<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>;

    // Init state used by the Tensor-based operator() path.
    const MegaMoeL1Layout layout_;
    uint64_t concatHalfN_{0};
    uint64_t k_{1};
    uint64_t baseK_{128};

    uint64_t aL1BufIdx_ = 0;
    uint64_t bL1BufIdx_ = 0;
    uint64_t scaleL1BufIdx_ = 0;
    uint64_t l0aBufIdx_ = 0;
    uint64_t l0bBufIdx_ = 0;

    template <typename DType, typename Layout>
    using L1TensorType =
        decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, DType>(0UL), Layout{}(16UL, 16UL)));

    using TensorAL1 = L1TensorType<AType, MakeLayoutAL1>;
    using TensorScaleAL1 = L1TensorType<ScaleAType, MakeLayoutScaleAL1>;
    using TensorScaleBL1 = L1TensorType<ScaleBType, MakeLayoutScaleBL1>;

    TensorAL1 tensorAL1_;
    TensorScaleAL1 tensorScaleAL1_;
    TensorScaleBL1 tensorScaleBL1_;

    // --- sync ---
    static constexpr TEventID eventIdsMte1ToMte2_ = 0;
    const TEventID eventIdsMxScaleMte1ToMte2_ = layout_.aBufferNum;
    static constexpr TEventID eventIdsAReady_ = 0;
    const TEventID eventIdsScaleReady_ = layout_.aBufferNum;
    static constexpr TEventID eventIdMte2ToMte1_ = 0;
    static constexpr TEventID eventIdMToMte1_ = 0;
    static constexpr TEventID eventIdMToMte1B_ = 2;
    static constexpr TEventID eventIdMte1ToM_ = 0;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 1;

    // Host layout packs A, interleaved B and two scale pairs; see CalcMegaMoeL1Layout.

    /**
     * L0 64KB Memory Map (double-buffered B tiles)
     * [0k]      [32KB]    [64KB]
     * |--- B0 ---|--- B1 ---|
     *    (32KB)     (32KB)
     */
    static constexpr uint64_t L0_BUF_OFFSET = 32 * 1024;
    static constexpr uint64_t MX_FP8FP4_M_STRIP = 128;
};

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyAAndScaleAL1ToL0(uint64_t l1KOffset, uint64_t kbOffset,
                                                                                 uint64_t realL0K,
                                                                                 uint64_t realL0ScaleK,
                                                                                 uint64_t mOffset, uint64_t mSize,
                                                                                 const BlockMmadOffsetParam& param)
{
    auto layoutAL0 = MakeLayoutAL0{}(mSize, realL0K);
    auto tensorAL0 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0aBufIdx_ * L0_BUF_OFFSET), layoutAL0);
    auto tensorBlockAL1 = tensorAL1_.slice(asc::te::make_coord(mOffset, (l1KOffset + kbOffset) % param.kaL1Size),
                                           asc::te::make_shape(mSize, realL0K));
    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0a{}), tensorAL0, tensorBlockAL1);

    auto layoutScaleAL0 = MakeLayoutScaleAL0{}(mSize, realL0ScaleK);
    auto tensorScaleAL0 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l0scalea, fp8_e8m0_t>((l0aBufIdx_ * L0_BUF_OFFSET) >> 4),
        layoutScaleAL0);
    auto tensorBlockScaleAL1 = tensorScaleAL1_.slice(
        asc::te::make_coord(
            mOffset, CeilDiv(((l1KOffset + kbOffset) % param.scaleKL1Size), MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
        asc::te::make_shape(mSize, realL0ScaleK));
    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0scalea{}), tensorScaleAL0, tensorBlockScaleAL1);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorBL1Type>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyBAndScaleBL1ToL0(const TensorBL1Type& tensorBL1,
                                                                                 uint64_t l1KOffset, uint64_t kbOffset,
                                                                                 uint64_t realL0K,
                                                                                 uint64_t realL0ScaleK,
                                                                                 const BlockMmadOffsetParam& param)
{
    auto layoutBL0 = MakeLayoutBL0{}(realL0K, param.nL1Size);
    auto tensorBL0 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l0b, AType>(l0bBufIdx_ * L0_BUF_OFFSET), layoutBL0);
    auto tensorBlockBL1 =
        tensorBL1.slice(asc::te::make_coord(l1KOffset, 0), asc::te::make_shape(realL0K, param.nL1Size));
    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0b{}), tensorBL0, tensorBlockBL1);

    auto layoutScaleBL0 = MakeLayoutScaleBL0{}(realL0ScaleK, param.nL1Size);
    auto tensorScaleBL0 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l0scaleb, fp8_e8m0_t>((l0bBufIdx_ * L0_BUF_OFFSET) >> 4),
        layoutScaleBL0);
    auto tensorBlockScaleBL1 = tensorScaleBL1_.slice(
        asc::te::make_coord(
            CeilDiv(((l1KOffset + kbOffset) % param.scaleKL1Size), MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0),
        asc::te::make_shape(realL0ScaleK, param.nL1Size));
    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{}), tensorScaleBL0, tensorBlockScaleBL1);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorC>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize,
                                                                          const BlockMmadOffsetParam& param,
                                                                          const TensorC& tensorC, bool copyUbToV1)
{
    const uint64_t bL1Offset = layout_.bOffsets[bL1BufIdx_];
    auto tensorBL1 =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(bL1Offset),
                             asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>{}(
                                 kbL1RealSize, param.nL1Size));
    const bool isLastGmK = kbOffset + kbL1RealSize >= param.kSize;
    const bool isFirstGmK = kbOffset == 0;
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += param.l0KSize) {
        const uint64_t realL0K = min(param.l0KSize, kbL1RealSize - l1KOffset);
        const uint64_t realL0ScaleK = CeilDiv(realL0K, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        const bool finalK = isLastGmK && l1KOffset + realL0K == kbL1RealSize;
        // One B/scaleB load serves all M strips for this K split. Its free event
        // is returned only after the last strip's MAD has stopped reading it.
        PipeWait<PIPE_M, PIPE_MTE1>(eventIdMToMte1B_ + l0bBufIdx_);
        CopyBAndScaleBL1ToL0(tensorBL1, l1KOffset, kbOffset, realL0K, realL0ScaleK, param);
        for (uint64_t mOffset = 0; mOffset < param.mL1Size; mOffset += MX_FP8FP4_M_STRIP) {
            const uint64_t mSize = min(MX_FP8FP4_M_STRIP, param.mL1Size - mOffset);

            auto tensorBL0 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, AType>(l0bBufIdx_ * L0_BUF_OFFSET),
                                     MakeLayoutBL0{}(realL0K, param.nL1Size));
            PipeWait<PIPE_M, PIPE_MTE1>(eventIdMToMte1_ + l0aBufIdx_);
            CopyAAndScaleAL1ToL0(l1KOffset, kbOffset, realL0K, realL0ScaleK, mOffset, mSize, param);
            auto tensorAL0 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0aBufIdx_ * L0_BUF_OFFSET),
                                     MakeLayoutAL0{}(mSize, realL0K));
            PipeFence<PIPE_MTE1, PIPE_M, eventIdMte1ToM_>();

            // Each strip owns a contiguous, 512-byte-aligned L0C range. Retain
            // the original per-element K order and the 2/3 unit-flag protocol.
            auto layoutL0C =
                asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(mSize, param.nL1Size);
            auto tensorL0C =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(
                                         mOffset * CeilAlign(param.nL1Size, uint64_t{L0C_C0}) * sizeof(float)),
                                     layoutL0C);
            asc::te::mmad_params params;
            params.m = static_cast<uint16_t>(mSize);
            params.k = static_cast<uint16_t>(realL0K);
            params.n = static_cast<uint16_t>(param.nL1Size);
            params.unit_flag = finalK ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
            params.init_with_zero = isFirstGmK && l1KOffset == 0;
            asc::te::mmad(
                asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0);
            PipeNotify<PIPE_M, PIPE_MTE1>(eventIdMToMte1_ + l0aBufIdx_);
            l0aBufIdx_ ^= 1;

            if (finalK) {
                CopyCL0c2GmOrUb(tensorC, param, copyUbToV1, mOffset, mSize);
            }
        }
        PipeNotify<PIPE_M, PIPE_MTE1>(eventIdMToMte1B_ + l0bBufIdx_);
        l0bBufIdx_ ^= 1;
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::WaitAMTE1ToMTE2()
{
    PipeWait<PIPE_MTE1, PIPE_MTE2>(eventIdsMte1ToMte2_ + aL1BufIdx_);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::WaitScaleMTE1ToMTE2()
{
    PipeWait<PIPE_MTE1, PIPE_MTE2>(eventIdsMxScaleMte1ToMte2_ + scaleL1BufIdx_);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::SetScaleMTE1ToMTE2()
{
    PipeNotify<PIPE_MTE1, PIPE_MTE2>(eventIdsMxScaleMte1ToMte2_ + scaleL1BufIdx_);
    scaleL1BufIdx_ ^= 1;
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::SetMTE1ToMTE2()
{
    PipeNotify<PIPE_MTE1, PIPE_MTE2>(eventIdsMte1ToMte2_ + aL1BufIdx_);
    aL1BufIdx_ = (aL1BufIdx_ + 1) % layout_.aBufferNum;
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorA>
__aicore__ inline auto BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyAGmToL1(const TensorA& tensorA,
                                                                        const BlockMmadOffsetParam& param,
                                                                        int64_t kaGmOffset, uint64_t slot)
{
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;
    auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    auto layoutAL1 = MakeLayoutAL1{}(param.mL1Size, kaL1RealSize);
    auto gmBlockA = tensorA.slice(asc::te::make_coord(0, kaGmOffset), asc::te::make_shape(param.mL1Size, kaL1RealSize));
    const uint64_t aL1Offset = layout_.aOffsets[slot];
    auto destination = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Offset), layoutAL1);
    asc::te::copy(copyGM2L1, destination, gmBlockA);
    return destination;
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorScaleA, typename TensorScaleB>
__aicore__ inline auto BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyMxScaleGmToL1(const TensorScaleA& tensorScaleA,
                                                                              const TensorScaleB& tensorScaleB,
                                                                              const BlockMmadOffsetParam& param,
                                                                              uint64_t kbL1Offset, uint64_t slot)
{
    uint64_t scaleKGmSize = CeilDiv(param.kSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint64_t scaleKL1StandardLen = CeilDiv(param.scaleKL1Size, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + param.scaleKL1Size) > param.kSize ?
                                    CeilDiv(param.kSize - kbL1Offset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE :
                                    scaleKL1StandardLen;
    const uint64_t scaleAL1Offset = layout_.scaleAOffsets[slot];
    const uint64_t scaleBL1Offset = layout_.scaleBOffsets[slot];
    auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    auto layoutScaleAL1 = MakeLayoutScaleAL1{}(param.mL1Size, scaleKL1RealSize);
    auto destinationA =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(scaleAL1Offset), layoutScaleAL1);
    auto gmBlockScaleA =
        tensorScaleA.slice(asc::te::make_coord(0, CeilDiv(kbL1Offset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
                           asc::te::make_shape(param.mL1Size, scaleKL1RealSize));
    asc::te::copy(CopyScaleGM2L1, destinationA, gmBlockScaleA);

    auto layoutScaleBL1 = MakeLayoutScaleBL1{}(scaleKL1RealSize, param.nL1Size);
    auto destinationB =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(scaleBL1Offset), layoutScaleBL1);
    auto gmBlockScaleB =
        tensorScaleB.slice(asc::te::make_coord(CeilDiv(kbL1Offset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0),
                           asc::te::make_shape(scaleKL1RealSize, param.nL1Size));
    if (concatHalfN_ != 0U) {
        MegaMoeImpl::CopyGmm1ScaleConcatToL1(destinationB, gmBlockScaleB, concatHalfN_, k_);
    } else {
        asc::te::copy(CopyScaleGM2L1, destinationB, gmBlockScaleB);
    }
    return AscendC::Std::tuple<decltype(destinationA), decltype(destinationB)>{destinationA, destinationB};
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorC>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyCL0c2GmOrUb(const TensorC& tensorC,
                                                                            const BlockMmadOffsetParam& param,
                                                                            bool copyUbToV1, uint64_t mOffset,
                                                                            uint64_t mSize)
{
    constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
    auto layoutL0C =
        asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(mSize, param.nL1Size);
    auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(
                                              mOffset * CeilAlign(param.nL1Size, uint64_t{L0C_C0}) * sizeof(float)),
                                          layoutL0C);
    auto tensorStripC = tensorC.slice(asc::te::make_coord(mOffset, 0), asc::te::make_shape(mSize, param.nL1Size));
    if constexpr (AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::ub>) {
        // C L0C->UB
        auto CopyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
        asc::te::copy(CopyL0C2UB.with(asc::te::l0c_to_ub_params(asc::te::unit_flag_mode::enable_update, copyUbToV1)),
                      tensorStripC, tensorL0C, FP32_64_AS_UINT64);
    } else {
        // C L0C->GM
        auto CopyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        asc::te::copy(CopyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), tensorStripC,
                      tensorL0C, FP32_64_AS_UINT64);
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::BlockMmad(const MegaMoeL1Layout& layout)
    : layout_(layout)
{
    for (uint64_t i = 0; i < layout_.bBufferNum; i++) {
        SetAicToAiv();
    }
    for (uint64_t i = 0; i < DOUBLE_BUFFER; i++) {
        PipeNotify<PIPE_M, PIPE_MTE1>(eventIdMToMte1_ + i);
        PipeNotify<PIPE_M, PIPE_MTE1>(eventIdMToMte1B_ + i);
        PipeNotify<PIPE_MTE1, PIPE_MTE2>(eventIdsMxScaleMte1ToMte2_ + i);
    }
    for (uint64_t i = 0; i < layout_.aBufferNum; i++) {
        PipeNotify<PIPE_MTE1, PIPE_MTE2>(eventIdsMte1ToMte2_ + i);
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::~BlockMmad()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER; i++) {
        PipeWait<PIPE_M, PIPE_MTE1>(eventIdMToMte1_ + i);
        PipeWait<PIPE_M, PIPE_MTE1>(eventIdMToMte1B_ + i);
        PipeWait<PIPE_MTE1, PIPE_MTE2>(eventIdsMxScaleMte1ToMte2_ + i);
    }
    for (uint64_t i = 0; i < layout_.aBufferNum; i++) {
        PipeWait<PIPE_MTE1, PIPE_MTE2>(eventIdsMte1ToMte2_ + i);
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <bool ResetBufferIndices>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::Init(const ProblemShape& problemShape,
                                                                 const BlockShape& l0TileShape, bool interleave)
{
    concatHalfN_ = interleave ? asc::te::get<IDX_N_IDX>(problemShape) / MegaMoeImpl::ACTIVATION_N_HALF : 0U;
    k_ = asc::te::get<IDX_K_IDX>(problemShape);
    baseK_ = asc::te::get<IDX_K_IDX>(l0TileShape);

    // Continuous AIC/AIV0 operation must preserve matching B-slot phase.
    if constexpr (ResetBufferIndices) {
        aL1BufIdx_ = 0;
        bL1BufIdx_ = 0;
        scaleL1BufIdx_ = 0;
        l0aBufIdx_ = 0;
        l0bBufIdx_ = 0;
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorC>
__aicore__ inline auto BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::GetBlockMmadOffsetParam(const TensorC& tensorC) const
{
    BlockMmadOffsetParam blockParam = {};
    blockParam.mL1Size = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 1>(tensorC.layout());
    blockParam.nL1Size =
        asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(tensorC.layout());
    blockParam.kSize = k_;
    blockParam.scaleKL1Size = layout_.scaleK;
    blockParam.l0KSize = baseK_;
    blockParam.kaL1Size = layout_.tileK;
    blockParam.kbL1Size = layout_.tileK;
    return blockParam;
}

/*
 * kaL1Size % kbL1Size == 0
 * scaleL1Size % kbL1Size == 0
 */
BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::operator()(const TensorA& tensorA,
                                                                       const TensorScaleA& tensorScaleA,
                                                                       const TensorScaleB& tensorScaleB,
                                                                       const TensorC& tensorC, bool copyUbToV1)
{
    const auto blockParam = GetBlockMmadOffsetParam(tensorC);
    ProcessTilesL1(tensorA, tensorScaleA, tensorScaleB, tensorC, blockParam, copyUbToV1);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::ProcessTilesL1(
    const TensorA& tensorA, const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB, const TensorC& tensorC,
    const BlockMmadOffsetParam& blockParam, bool copyUbToV1)
{
    uint64_t nextAKOffset = 0;
    uint64_t nextScaleKOffset = 0;
    uint64_t aCopySlot = aL1BufIdx_;
    uint64_t scaleCopySlot = scaleL1BufIdx_;
    for (uint64_t kbGmOffset = 0;; kbGmOffset += blockParam.kbL1Size) {
        // 启动时填充 A 槽，随后补充上一轮释放的槽。scale 只允许当前和下一窗口常驻，
        // 避免将第三窗口的等待排在释放该槽所需的消费操作之前。
        const uint64_t aCopyLimit = kbGmOffset + layout_.aBufferNum * blockParam.kaL1Size;
        const uint64_t scaleCopyLimit =
            (kbGmOffset / blockParam.scaleKL1Size + DOUBLE_BUFFER) * blockParam.scaleKL1Size;
        while (nextAKOffset < blockParam.kSize && nextAKOffset < aCopyLimit && nextAKOffset < scaleCopyLimit) {
            // 先搬对应 scale，再搬 A；不能让当前需要的 scale 排在未来 A 块之后。
            if (nextAKOffset == nextScaleKOffset) {
                PipeWait<PIPE_MTE1, PIPE_MTE2>(eventIdsMxScaleMte1ToMte2_ + scaleCopySlot);
                CopyMxScaleGmToL1(tensorScaleA, tensorScaleB, blockParam, nextScaleKOffset, scaleCopySlot);
                PipeNotify<PIPE_MTE2, PIPE_MTE1>(eventIdsScaleReady_ + scaleCopySlot);
                nextScaleKOffset += blockParam.scaleKL1Size;
                scaleCopySlot ^= 1;
            }
            PipeWait<PIPE_MTE1, PIPE_MTE2>(eventIdsMte1ToMte2_ + aCopySlot);
            CopyAGmToL1(tensorA, blockParam, nextAKOffset, aCopySlot);
            PipeNotify<PIPE_MTE2, PIPE_MTE1>(eventIdsAReady_ + aCopySlot);
            nextAKOffset += blockParam.kaL1Size;
            aCopySlot = (aCopySlot + 1) % layout_.aBufferNum;
        }
        // 补充 MTE2 队列后再归还上一轮的 B 槽。最后一轮也归还额度，但不再执行计算。
        if (kbGmOffset != 0) {
            SetAicToAiv();
            bL1BufIdx_ = (bL1BufIdx_ + 1) % layout_.bBufferNum;
        }
        if (kbGmOffset >= blockParam.kSize) {
            break;
        }
        const uint64_t kbL1RealSize = min(blockParam.kbL1Size, blockParam.kSize - kbGmOffset);
        // 消费端只绑定当前槽；提前搬入后续块不会修改这些 tensor 的地址和尾块布局。
        if (kbGmOffset % blockParam.scaleKL1Size == 0) {
            PipeWait<PIPE_MTE2, PIPE_MTE1>(eventIdsScaleReady_ + scaleL1BufIdx_);
            const uint64_t realScaleK =
                CeilDiv(min(blockParam.scaleKL1Size, blockParam.kSize - kbGmOffset), MXFP_DIVISOR_SIZE) *
                MXFP_MULTI_BASE_SIZE;
            tensorScaleAL1_ = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(layout_.scaleAOffsets[scaleL1BufIdx_]),
                MakeLayoutScaleAL1{}(blockParam.mL1Size, realScaleK));
            tensorScaleBL1_ = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(layout_.scaleBOffsets[scaleL1BufIdx_]),
                MakeLayoutScaleBL1{}(realScaleK, blockParam.nL1Size));
        }
        if (kbGmOffset % blockParam.kaL1Size == 0) {
            PipeWait<PIPE_MTE2, PIPE_MTE1>(eventIdsAReady_ + aL1BufIdx_);
            const uint64_t realKa = min(blockParam.kaL1Size, blockParam.kSize - kbGmOffset);
            tensorAL1_ =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(layout_.aOffsets[aL1BufIdx_]),
                                     MakeLayoutAL1{}(blockParam.mL1Size, realKa));
        }
        WaitAivToAic();
        ProcessTileL1(kbGmOffset, kbL1RealSize, blockParam, tensorC, copyUbToV1);
        const uint64_t nextKbGmOffset = kbGmOffset + blockParam.kbL1Size;
        if (nextKbGmOffset % blockParam.kaL1Size == 0 || nextKbGmOffset >= blockParam.kSize) {
            SetMTE1ToMTE2();
        }
        if (nextKbGmOffset % blockParam.scaleKL1Size == 0 || nextKbGmOffset >= blockParam.kSize) {
            SetScaleMTE1ToMTE2();
        }
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::WaitAivToAic()
{
    SyncIntraWait<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::SetAicToAiv()
{
    SyncIntraArrive<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}

#undef BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
#undef BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION
} // namespace Block
} // namespace Gemm
} // namespace Blaze

namespace MegaMoeImpl::GmmKernel {

// 小 M 门控 tile 使用独立的 A/B 环。通用 MMAD 基类仍负责事件生命周期和通用形状。
template <typename BaseMmad, uint64_t L0TileK>
class Gmm1Fp4Block : public BaseMmad {
    using Layout = MegaMoeFp4Gmm1L1Layout;
    using Fp4 = typename BaseMmad::AType;
    using Scale = typename BaseMmad::ScaleBType;
    using Fp4C0 = AscendC::Std::Int<Blaze::Gemm::C0_SIZE_B4>;
    using ScaleC0 = AscendC::Std::Int<Blaze::Gemm::SCALE_C0>;
    static constexpr uint16_t B_EVENT_BASE = Layout::A_BUFFER_COUNT;
    static constexpr uint16_t SCALE_EVENT_BASE = B_EVENT_BASE + Layout::B_BUFFER_COUNT;
    static_assert(Layout::TILE_M == Blaze::Gemm::MATMUL_MNK_ALIGN && Layout::TILE_N == MegaMoeImpl::L1_TILE_N &&
                  Layout::TILE_K % L0TileK == 0U);
    static_assert(Layout::HALF_L1_BYTES * Layout::A_BUFFER_COUNT == AscendC::TOTAL_L1_SIZE);
    static_assert(SCALE_EVENT_BASE + Layout::SCALE_BUFFER_COUNT ==
                  2U * Blaze::Gemm::TRIPLE_BUFFER_COUNT + Blaze::Gemm::SCALE_BUFFER_NUM);

public:
    using BaseMmad::operator();

    __aicore__ inline Gmm1Fp4Block() = default;
    __aicore__ inline ~Gmm1Fp4Block() = default;

    __aicore__ inline void Init(const typename BaseMmad::ProblemShape& shape,
                                const typename BaseMmad::MmadParams& params, const Layout* layout)
    {
        layout_ = layout;
        if (layout_ == nullptr) {
            BaseMmad::Init(shape, params);
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB,
                                      TensorC& output, const typename BaseMmad::BlockShape& shape)
    {
        if (layout_ == nullptr) {
            BaseMmad::operator()(gmA, gmB, gmScaleA, gmScaleB, output, shape);
            return;
        }

        using namespace asc::te;
        using namespace Blaze::Gemm;
        const uint64_t m = get<M_VALUE>(shape);
        const uint64_t n = Layout::TILE_N;
        const uint64_t k = get<K_VALUE>(shape);
        const uint64_t windowK = layout_->scaleK;
        constexpr uint64_t tileK = Layout::TILE_K;
        constexpr uint64_t halfN = Layout::TILE_N / BaseMmad::CONCAT_N_FACTOR;
        const uint64_t l0ScaleK = MxScaleKLen<Fp4>(L0TileK);
        auto aLayout = make_frame_layout<nz_layout_ptn, Fp4C0>(m, tileK);
        auto bLayout = make_frame_layout<zn_layout_ptn, Fp4C0>(tileK, n);
        auto aTile = [&](uint16_t slot) __attribute__((cce_aicore))
        {
            return make_tensor(make_mem_ptr<location::l1, Fp4>(slot * Layout::HALF_L1_BYTES), aLayout);
        };
        auto bTile = [&](uint16_t slot) __attribute__((cce_aicore))
        {
            const uint64_t half = slot % Layout::A_BUFFER_COUNT;
            const uint64_t indexInHalf = slot / Layout::A_BUFFER_COUNT;
            return make_tensor(make_mem_ptr<location::l1, Fp4>(half * Layout::HALF_L1_BYTES + Layout::A_SLOT_BYTES +
                                                               indexInHalf * Layout::B_SLOT_BYTES),
                               bLayout);
        };

        const auto copyGm = make_copy(copy_gm_to_l1{});
        const auto copyConcat = make_copy(Tile::CopyConcatGM2L1{}).with(Tile::CopyConcatGM2L1Params{layout_->fullN, k});
        auto loadA = [&](uint64_t offset, uint16_t slot) __attribute__((cce_aicore))
        {
            PipeWait<PIPE_MTE1, PIPE_MTE2>(slot);
            copy(copyGm, aTile(slot), gmA.slice(make_coord(0, offset), make_shape(m, tileK)));
            PipeNotify<PIPE_MTE2, PIPE_MTE1>(slot);
        };
        auto loadB = [&](uint64_t offset, uint16_t slot) __attribute__((cce_aicore))
        {
            const uint16_t event = B_EVENT_BASE + slot;
            PipeWait<PIPE_MTE1, PIPE_MTE2>(event);
            copy(copyConcat, bTile(slot), gmB.slice(make_coord(offset, 0), make_shape(tileK, halfN)));
            PipeNotify<PIPE_MTE2, PIPE_MTE1>(event);
        };

        auto l0c = make_tensor(make_mem_ptr<location::l0c, float>(0UL),
                               make_frame_layout<nz_layout_ptn, AscendC::Std::Int<C0_SIZE_L0C>>(
                                   Blaze::Gemm::CeilAlign(m, BaseMmad::SPLIT_M_ALIGN), n));
        const auto copyA = make_copy(copy_l1_to_l0a{});
        const auto copyB = make_copy(copy_l1_to_l0b{});
        const auto copyScaleA = make_copy(copy_l1_to_l0scalea{});
        const auto copyScaleB = make_copy(copy_l1_to_l0scaleb{});
        constexpr auto mma = make_mmad(mmad_operation{}, Tile::MmadTraitMX{});
        constexpr uint64_t l0HalfBytes = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        uint16_t l0Slot = 0U;

        for (uint64_t start = 0; start < k; start += windowK) {
            const uint64_t end = Min(start + windowK, k);
            const uint64_t scaleSpan = MxScaleKLen<Fp4>(end - start);
            const uint64_t scaleBase = scaleSlot_ * Layout::HALF_L1_BYTES;
            const uint16_t scaleEvent = SCALE_EVENT_BASE + scaleSlot_;
            auto scaleA = make_tensor(make_mem_ptr<location::l1, Scale>(scaleBase + Layout::SCALE_A_OFFSET),
                                      make_frame_layout<zz_layout_ptn, ScaleC0>(m, scaleSpan));
            auto scaleB = make_tensor(make_mem_ptr<location::l1, Scale>(scaleBase + layout_->scaleBOffset),
                                      make_frame_layout<nn_layout_ptn, ScaleC0>(scaleSpan, n));
            const uint64_t scaleStart = MxScaleKLen<Fp4>(start);
            PipeWait<PIPE_MTE1, PIPE_MTE2>(scaleEvent);
            copy(copyGm, scaleA, gmScaleA.slice(make_coord(0, scaleStart), make_shape(m, scaleSpan)));
            copy(copyConcat, scaleB, gmScaleB.slice(make_coord(scaleStart, 0), make_shape(scaleSpan, halfN)));

            // A0/B0 先入队，供 MTE1 开始计算；随后的 B 可以独立于 A 提前搬运。
            // 一个窗口恰好占满 B 槽时，先给下一块 A 发起搬运，再补上最后一块 B。
            const uint64_t bPrefetchCount =
                end - start == tileK * Layout::B_BUFFER_COUNT ? Layout::B_BUFFER_COUNT - 1U : Layout::B_BUFFER_COUNT;
            const uint64_t prefetchEnd = Min(end, start + tileK * bPrefetchCount);
            for (uint64_t offset = start; offset < prefetchEnd; offset += tileK) {
                const uint16_t slot = (offset - start) / tileK;
                if (slot < Layout::A_BUFFER_COUNT) {
                    loadA(offset, slot);
                }
                loadB(offset, slot);
            }
            for (uint64_t offset = start; offset < end; offset += tileK) {
                const uint16_t aSlot = ((offset - start) / tileK) % Layout::A_BUFFER_COUNT;
                const uint16_t bSlot = ((offset - start) / tileK) % Layout::B_BUFFER_COUNT;
                const uint16_t bEvent = B_EVENT_BASE + bSlot;
                // 等到 A 就绪也确保本窗口的 scale 已完成 MTE2 搬运。
                PipeWait<PIPE_MTE2, PIPE_MTE1>(aSlot);
                PipeWait<PIPE_MTE2, PIPE_MTE1>(bEvent);
                auto l1a = aTile(aSlot);
                auto l1b = bTile(bSlot);
#pragma unroll
                for (uint64_t innerK = 0; innerK < tileK; innerK += L0TileK) {
                    const uint16_t cubeDone = M_MTE1_FLAG_0 + l0Slot;
                    const uint64_t l0Offset = l0Slot * l0HalfBytes;
                    const uint64_t l0ScaleOffset = l0Offset / GetScaleDivisor<Fp4>();
                    const uint64_t scaleOffset = MxScaleKLen<Fp4>(offset - start + innerK);
                    auto l0a = make_tensor(make_mem_ptr<location::l0a, Fp4>(l0Offset),
                                           make_frame_layout<nz_layout_ptn, Fp4C0>(m, L0TileK));
                    auto l0b = make_tensor(make_mem_ptr<location::l0b, Fp4>(l0Offset),
                                           make_frame_layout<zn_layout_ptn, Fp4C0>(L0TileK, n));
                    auto l0ScaleA = make_tensor(make_mem_ptr<location::l0scalea, Scale>(l0ScaleOffset),
                                                make_frame_layout<zz_layout_ptn, ScaleC0>(m, l0ScaleK));
                    auto l0ScaleB = make_tensor(make_mem_ptr<location::l0scaleb, Scale>(l0ScaleOffset),
                                                make_frame_layout<nn_layout_ptn, ScaleC0>(l0ScaleK, n));
                    PipeWait<PIPE_M, PIPE_MTE1>(cubeDone);
                    copy(copyA, l0a, l1a.slice(make_coord(0, innerK), make_shape(m, L0TileK)));
                    copy(copyB, l0b, l1b.slice(make_coord(innerK, 0), make_shape(L0TileK, n)));
                    copy(copyScaleA, l0ScaleA, scaleA.slice(make_coord(0, scaleOffset), make_shape(m, l0ScaleK)));
                    copy(copyScaleB, l0ScaleB, scaleB.slice(make_coord(scaleOffset, 0), make_shape(l0ScaleK, n)));
                    PipeNotify<PIPE_MTE1, PIPE_M>(l0Slot);
                    PipeWait<PIPE_MTE1, PIPE_M>(l0Slot);

                    const uint64_t consumedK = offset + innerK;
                    asc::te::mmad_params params;
                    params.m = static_cast<uint16_t>(m);
                    params.n = static_cast<uint16_t>(n);
                    params.k = L0TileK;
                    params.unit_flag =
                        consumedK + L0TileK == k ? unit_flag_mode::enable_update : unit_flag_mode::enable_keep;
                    params.init_with_zero = consumedK == 0U;
                    asc::te::mmad(mma.with(params), l0c, l0a, l0b);
                    PipeNotify<PIPE_M, PIPE_MTE1>(cubeDone);
                    l0Slot ^= 1U;
                }
                PipeNotify<PIPE_MTE1, PIPE_MTE2>(aSlot);
                PipeNotify<PIPE_MTE1, PIPE_MTE2>(bEvent);
                const uint64_t nextA = offset + Layout::A_BUFFER_COUNT * tileK;
                const uint64_t nextB = offset + bPrefetchCount * tileK;
                if (nextA < end) {
                    loadA(nextA, aSlot);
                }
                if (nextB < end) {
                    const uint16_t nextBSlot = (bSlot + bPrefetchCount) % Layout::B_BUFFER_COUNT;
                    loadB(nextB, nextBSlot);
                }
            }
            PipeNotify<PIPE_MTE1, PIPE_MTE2>(scaleEvent);
            scaleSlot_ ^= 1U;
        }
        auto copyOut = make_copy(copy_l0c_to_ub{}, Tile::CopyL0C2UBTraitSplitM{});
        copy(copyOut.with(l0c_to_ub_params(unit_flag_mode::enable_update)), output, l0c);
    }

private:
    const Layout* layout_ = nullptr;
    // 跨计算 tile 保留 scale 轮转，避免每次都等待槽 0 归还。
    uint16_t scaleSlot_ = 0U;
};
} // namespace MegaMoeImpl::GmmKernel
