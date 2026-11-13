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
 * \file qmm_mx_block_mmad_tile_k_wait_flag.h
 * \brief Standalone BlockMmad class for MatmulWithScaleMx with split-K hard sync support.
 *
 * Based on Blaze's block_mmad_qbmm_mx.h, extended with:
 *   - split-K hard sync: CrossCoreWaitFlag(9) every AG_SYNC_INTERVAL/kL1 K-iterations
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"

namespace Blaze {
namespace Gemm {
namespace Block {

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201))

static constexpr uint64_t AG_SYNC_INTERVAL = 1024UL;
static constexpr uint16_t CROSS_CORE_FLAG_NUM_NINE = 9;
static constexpr uint8_t CROSS_CORE_INNER_CUBE_VEC_SYNC = 0x2U;

// 独立 BlockMmad 类(与 QmmMxBlockMmadFragment 组织方式一致), 不依赖 BlockMmad 偏特化派发
template <uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BType_, class LayoutB_,
          class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class QmmMxBlockMmadTileKWaitFlag {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using DispatchPolicy = MatmulWithScaleMx<A_FULL_LOAD_MODE, ATOMIC_ADD>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool weightNz = IsWeightNz<LayoutB>::value;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;
    using LayoutScaleB =
        AscendC::Std::conditional_t<transB, asc::te::scaleb_dn_layout_ptn, asc::te::scaleb_nd_layout_ptn>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline QmmMxBlockMmadTileKWaitFlag()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~QmmMxBlockMmadTileKWaitFlag()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
    }
    __aicore__ inline void Init(const ProblemShape& problemShape, const BlockShape& l0TileShape,
                                const L1Params& l1Params, bool isBias, bool dbL0C, uint64_t splitKNum = 1)
    {
        k_ = asc::te::get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        splitKNum_ = splitKNum;
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        uint64_t baseM = asc::te::get<IDX_M_IDX>(l0TileShape);
        baseN_ = asc::te::get<IDX_N_IDX>(l0TileShape);
        baseK_ = asc::te::get<IDX_K_IDX>(l0TileShape);
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
        const uint64_t l1HalfBufNum = l1BufNum_ >> 1;
        const uint64_t bL1OneBuffer = (baseN_ * kL1_) >> sizeShift;
        const uint64_t scaleBL1OneBuffer = baseN_ * (Align64(scaleKL1_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        const uint64_t biasL1OneBuffer = isBias_ ? baseN_ * sizeof(BiasType) : 0;
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            InitNoneFullLoadL1BufferOffsets(baseM, l1HalfBufNum, bL1OneBuffer, scaleBL1OneBuffer);
        } else {
            InitAFullLoadL1BufferOffsets(baseM, l1HalfBufNum, bL1OneBuffer, scaleBL1OneBuffer, biasL1OneBuffer);
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(TensorA const& gmA, TensorB const& gmB, TensorScaleA const& gmScaleA,
                                      TensorScaleB const& gmScaleB, TensorBias const& gmBias, TensorC const& gmC,
                                      BlockShape const& singleShape, uint64_t splitKIdx = 0)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        const uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset), layoutL0C);
        const bool isFirstSplitK = splitKIdx == 0;
        const bool isLastSplitK = splitKIdx == splitKNum_ - 1;

        const uint64_t scaleKIter = scaleKL1_ / kL1_;
        uint64_t scaleKIterIdx = 0;
        const uint64_t scaleKOffsetStride = (Align64(kL1_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        // split-K hard sync: AIV 每就绪 AG_SYNC_INTERVAL(1024) 元素 set(9) 一次，每核共 ceil(k/1024) 次。
        // AIC 读第 iter0 段 K [iter0*kL1,(iter0+1)*kL1) 前，需累计等到 ceil((iter0+1)*kL1/1024) 次，
        // 保证该段所有列已 gather 完成。循环末尾 drain 补齐，确保 wait 次数恒等于 AIV set 次数。
        const uint64_t totalSets = CeilDiv(k_, AG_SYNC_INTERVAL);
        uint64_t waited = 0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            // gather 等待下沉到 ProcessL1Iteration 内（B/bias 装载之后），见该函数注释。
            ProcessL1Iteration(gmA, gmB, gmScaleA, gmScaleB, gmBias, tensorL0C, tileL1L0Param, iter0,
                               scaleKOffsetStride, scaleKIter, scaleKIterIdx, isFirstSplitK, isLastSplitK, totalSets,
                               waited);
        }
        if (isLastSplitK) {
            if constexpr (AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::ub>) {
                // C L0C->UB
                auto CopyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
                asc::te::copy(CopyL0C2UB.with(asc::te::l0c_to_ub_params(asc::te::unit_flag_mode::enable_update)), gmC,
                              tensorL0C);
            } else {
                // C L0C->GM
                auto CopyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
                asc::te::copy(CopyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), gmC,
                              tensorL0C);
            }
            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
        if (firstRound) {
            // drain 剩余 set(9)，确保 AIC wait 次数恒等于 AIV set 次数（ceil(k/AG_SYNC_INTERVAL)），
            // 避免残留计数影响后续（虽然只有 firstRound wait，但计数须收支平衡）。
            while (waited < totalSets) {
                AscendC::CrossCoreWaitFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE2>(CROSS_CORE_FLAG_NUM_NINE);
                waited++;
            }
        }
        firstRound = false;
    }

private:
    static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;

    using MakeLayoutAL1 =
        AscendC::Std::conditional_t<transA,
                                    asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>,
                                    asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 =
        AscendC::Std::conditional_t<transB,
                                    asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>,
                                    asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>>;

    struct TileL1L0Param {
        uint64_t curM = 0;
        uint64_t curN = 0;
        uint64_t curGmKL1 = 0;
        uint64_t curPadKL1 = 0; // pad to 64 align
    };

    template <typename TensorScaleA, typename TensorScaleB>
    struct ScalePair {
        TensorScaleA scaleA;
        TensorScaleB scaleB;
    };

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyScalesInL1(TensorScaleA const& gmScaleA, TensorScaleB const& gmScaleB,
                                          const TileL1L0Param& tileL1L0Param, uint64_t scaleL1BufId, uint64_t kL1Offset,
                                          uint64_t scaleGmOffset, bool needCopyScale)
    {
        const uint64_t scaleKL1Len = (Align64(scaleKL1_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        const uint64_t scaleBL1Offset = l1BufferScaleBOffset_[scaleL1BufId];
        auto layoutScaleBL1 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL1Len, tileL1L0Param.curN);
        auto tensorScaleBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AscendC::fp8_e8m0_t>(scaleBL1Offset), layoutScaleBL1);
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            return CopyPartialScalesInL1(gmScaleA, gmScaleB, tensorScaleBL1, tileL1L0Param, scaleL1BufId, kL1Offset,
                                         scaleGmOffset, scaleKL1Len, needCopyScale);
        } else {
            return CopyFullLoadScalesInL1(gmScaleA, gmScaleB, tensorScaleBL1, tileL1L0Param, scaleL1BufId, kL1Offset,
                                          scaleGmOffset, needCopyScale);
        }
    }

    template <typename TensorScaleA, typename TensorScaleB, typename TensorScaleBL1>
    __aicore__ inline auto CopyPartialScalesInL1(TensorScaleA const& gmScaleA, TensorScaleB const& gmScaleB,
                                                 TensorScaleBL1& tensorScaleBL1, const TileL1L0Param& tileL1L0Param,
                                                 uint64_t scaleL1BufId, uint64_t kL1Offset, uint64_t scaleGmOffset,
                                                 uint64_t scaleKL1Len, bool needCopyScale)
    {
        // L1 uses the full scaleKL1_ length; GM uses the actual length, which may be a tail block.
        const uint64_t scaleAL1Offset = l1BufferScaleAOffset_[scaleL1BufId];
        auto layoutScaleAL1 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            tileL1L0Param.curM, scaleKL1Len);
        auto tensorScaleAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AscendC::fp8_e8m0_t>(scaleAL1Offset), layoutScaleAL1);
        if (needCopyScale) {
            auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
            const uint64_t curScaleKL1 = Min(scaleKL1_, k_ - kL1Offset);
            const uint64_t curScaleKLen = (Align64(curScaleKL1) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
            auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(0, scaleGmOffset),
                                                asc::te::make_shape(tileL1L0Param.curM, curScaleKLen));
            asc::te::copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            CopyScaleBInL1(gmScaleB, tensorScaleBL1, tileL1L0Param, scaleGmOffset, curScaleKLen);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
    }

    template <typename TensorScaleA, typename TensorScaleB, typename TensorScaleBL1>
    __aicore__ inline auto CopyFullLoadScalesInL1(TensorScaleA const& gmScaleA, TensorScaleB const& gmScaleB,
                                                  TensorScaleBL1& tensorScaleBL1, const TileL1L0Param& tileL1L0Param,
                                                  uint64_t scaleL1BufId, uint64_t kL1Offset, uint64_t scaleGmOffset,
                                                  bool needCopyScale)
    {
        const uint64_t scaleAL1Offset = l1BufferScaleAOffset_[0];
        const uint64_t scaleKLen = (Align64(k_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleAL1 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            tileL1L0Param.curM, scaleKLen);
        auto tensorScaleAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AscendC::fp8_e8m0_t>(scaleAL1Offset), layoutScaleAL1);
        auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        if (needCopyScale) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
            const uint64_t curScaleKL1 = Min(scaleKL1_, k_ - kL1Offset);
            const uint64_t curScaleKLen = (Align64(curScaleKL1) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
            CopyScaleBInL1(gmScaleB, tensorScaleBL1, tileL1L0Param, scaleGmOffset, curScaleKLen);
        }
        if (abL1LoopCnt_ == 0) {
            asc::te::copy(CopyScaleGM2L1, tensorScaleAL1, gmScaleA);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
    }

    template <typename TensorScaleB, typename TensorScaleBL1>
    __aicore__ inline void CopyScaleBInL1(TensorScaleB const& gmScaleB, TensorScaleBL1& tensorScaleBL1,
                                          const TileL1L0Param& tileL1L0Param, uint64_t scaleGmOffset,
                                          uint64_t curScaleKLen)
    {
        auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(scaleGmOffset, 0),
                                            asc::te::make_shape(curScaleKLen, tileL1L0Param.curN));
        asc::te::copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
    }

    template <typename TensorA>
    __aicore__ inline auto CopyAInL1(TensorA const& gmA, const TileL1L0Param& tileL1L0Param, uint64_t l1BufId,
                                     uint64_t kL1Offset)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            const uint64_t aL1Offset = l1BufferAOffset_[l1BufId];
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadKL1);
            auto gmBlockA = gmA.slice(asc::te::make_coord(0, kL1Offset),
                                      asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curGmKL1));
            auto tensorAL1 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Offset), layoutAL1);
            Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
            asc::te::copy(copyGM2L1, tensorAL1, gmBlockA);
            return tensorAL1;
        } else {
            const uint64_t aL1Offset = l1BufferAOffset_[0];
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, Align64(k_));
            auto tensorTotalAL1 =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Offset), layoutAL1);
            auto tensorAL1 = tensorTotalAL1.slice(asc::te::make_coord(0, kL1Offset),
                                                  asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curPadKL1));
            if (abL1LoopCnt_ < kL1Iter_) {
                auto gmBlockA = gmA.slice(asc::te::make_coord(0, kL1Offset),
                                          asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curGmKL1));
                Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
                asc::te::copy(copyGM2L1, tensorAL1, gmBlockA);
            }
            return tensorAL1;
        }
    }

    template <typename TensorBias>
    __aicore__ inline auto CopyBiasInL1(TensorBias const& gmBias, const TileL1L0Param& tileL1L0Param,
                                        uint64_t scaleL1BufId, uint64_t iter0, bool isFirstSplitK)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        const uint64_t biasL1Offset = l1BufferBiasOffset_[scaleL1BufId];
        auto layoutBiasL1 =
            asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>{}(1UL, Align16(tileL1L0Param.curN));
        auto tensorBiasL1 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Offset), layoutBiasL1);
        if (isBias_ && iter0 == 0 && isFirstSplitK) {
            asc::te::copy(copyGM2L1, tensorBiasL1, gmBias);
        }
        return tensorBiasL1;
    }

    template <typename TensorB>
    __aicore__ inline auto CopyBInL1(TensorB const& gmB, const TileL1L0Param& tileL1L0Param, uint64_t l1BufId,
                                     uint64_t kL1Offset)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        const uint64_t bL1Offset = l1BufferBOffset_[l1BufId];
        auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curPadKL1, tileL1L0Param.curN);
        auto tensorBL1 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Offset), layoutBL1);
        auto gmBlockB = gmB.slice(asc::te::make_coord(kL1Offset, 0),
                                  asc::te::make_shape(tileL1L0Param.curGmKL1, tileL1L0Param.curN));
        Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
        asc::te::copy(copyGM2L1, tensorBL1, gmBlockB);
        return tensorBL1;
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1, typename TensorAL1, typename TensorBL1,
              typename TensorBiasL1, typename TensorL0C>
    __aicore__ inline void Iterate(const TileL1L0Param& tileL1L0Param, uint64_t iter0, uint64_t scaleKL1Len,
                                   uint64_t scaleKOffset, uint64_t scaleAKOffset, uint64_t biasBtOffset,
                                   bool isFirstSplitK, bool isLastSplitK, TensorScaleAL1& tensorScaleAL1,
                                   TensorScaleBL1& tensorScaleBL1, TensorAL1& tensorAL1, TensorBL1& tensorBL1,
                                   TensorBiasL1& tensorBiasL1, TensorL0C& tensorL0C)
    {
        // Slice scaleKL1 to current kL1 window.
        auto tensorBlockScaleBL1 = tensorScaleBL1.slice(asc::te::make_coord(scaleKOffset, 0),
                                                        asc::te::make_shape(scaleKL1Len, tileL1L0Param.curN));
        auto tensorBlockScaleAL1 = tensorScaleAL1.slice(asc::te::make_coord(0, scaleAKOffset),
                                                        asc::te::make_shape(tileL1L0Param.curM, scaleKL1Len));

        const uint64_t kL0Iter = Blaze::Gemm::CeilDiv(tileL1L0Param.curGmKL1, baseK_);
        const uint64_t scaleK0OffsetStride = (Align64(baseK_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        auto layoutBt = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, Align16(tileL1L0Param.curN));
        auto tensorBt =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(biasBtOffset), layoutBt);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            ProcessL0Iteration(tileL1L0Param, iter0, iter1, kL0Iter, scaleK0OffsetStride, isFirstSplitK, isLastSplitK,
                               tensorBlockScaleAL1, tensorBlockScaleBL1, tensorAL1, tensorBL1, tensorBiasL1, tensorL0C,
                               tensorBt);
        }
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1, typename TensorAL1, typename TensorBL1,
              typename TensorBiasL1, typename TensorL0C, typename TensorBT>
    __aicore__ inline void ProcessL0Iteration(const TileL1L0Param& tileL1L0Param, uint64_t iter0, uint16_t iter1,
                                              uint64_t kL0Iter, uint64_t scaleK0OffsetStride, bool isFirstSplitK,
                                              bool isLastSplitK, TensorScaleAL1& tensorBlockScaleAL1,
                                              TensorScaleBL1& tensorBlockScaleBL1, TensorAL1& tensorAL1,
                                              TensorBL1& tensorBL1, TensorBiasL1& tensorBiasL1, TensorL0C& tensorL0C,
                                              TensorBT& tensorBt)
    {
        const uint64_t curKL0 =
            (iter1 * baseK_ + baseK_ > tileL1L0Param.curPadKL1) ? (tileL1L0Param.curPadKL1 - iter1 * baseK_) : baseK_;
        const uint64_t scaleKL0Len = (Align64(curKL0) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        const uint64_t scaleK0Offset = iter1 * scaleK0OffsetStride;
        // Load data to L0 and open DB
        const uint64_t l0PingPongId = l0PingPong_ & 1;
        const uint64_t l0Offset = HALF_L0_SIZE * l0PingPongId;
        const uint16_t mte1WaitMFlag = static_cast<uint16_t>(l0PingPongId + M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

        // A, ScaleA L1->L0
        auto tensorAL0 = CopyAInL0(tensorAL1, tensorBlockScaleAL1, tileL1L0Param, iter1, curKL0, scaleKL0Len,
                                   scaleK0Offset, l0Offset);

        // bias L1->BT
        const bool needBias = isBias_ && iter0 == 0 && iter1 == 0 && isFirstSplitK;
        if (needBias) {
            auto CopyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
            asc::te::copy(CopyL12BT, tensorBt, tensorBiasL1);
        }

        // B, scaleB L1->L0
        auto tensorBL0 = CopyBInL0(tensorBL1, tensorBlockScaleBL1, tileL1L0Param, iter1, curKL0, scaleKL0Len,
                                   scaleK0Offset, l0Offset);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongId);

        const bool isFinalAccumulation = (iter0 + 1 == kL1Iter_) && (iter1 + 1 == kL0Iter) && isLastSplitK;
        const bool initCMatrix = iter0 == 0 && iter1 == 0 && isFirstSplitK && !needBias;
        Mmad(tileL1L0Param, isFinalAccumulation, initCMatrix, needBias, curKL0, tensorL0C, tensorAL0, tensorBL0,
             tensorBt);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
        l0PingPong_++;
    }

    template <typename TensorAL1, typename TensorScaleAL1>
    __aicore__ inline auto CopyAInL0(TensorAL1& tensorAL1, TensorScaleAL1& tensorBlockScaleAL1,
                                     const TileL1L0Param& tileL1L0Param, uint16_t iter1, uint64_t curKL0,
                                     uint64_t scaleKL0Len, uint64_t scaleK0Offset, uint64_t l0Offset)
    {
        // A, ScaleA L1->L0
        auto layoutAL0 =
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(tileL1L0Param.curM, curKL0);
        auto tensorAL0 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset), layoutAL0);
        auto tensorBlockAL1 =
            tensorAL1.slice(asc::te::make_coord(0, iter1 * baseK_), asc::te::make_shape(tileL1L0Param.curM, curKL0));
        auto CopyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        asc::te::copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

        auto CopyL12L0ScaleA = asc::te::make_copy(asc::te::copy_l1_to_l0scalea{});
        auto layoutScaleAL0 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            tileL1L0Param.curM, scaleKL0Len);
        // L0Scale copy uses 16-byte address units, while l0Offset is in bytes.
        auto tensorScaleAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scalea, AscendC::fp8_e8m0_t>(l0Offset >> 4), layoutScaleAL0);
        asc::te::copy(CopyL12L0ScaleA, tensorScaleAL0,
                      tensorBlockScaleAL1.slice(asc::te::make_coord(0, scaleK0Offset),
                                                asc::te::make_shape(tileL1L0Param.curM, scaleKL0Len)));
        return tensorAL0;
    }

    template <typename TensorBL1, typename TensorScaleBL1>
    __aicore__ inline auto CopyBInL0(TensorBL1& tensorBL1, TensorScaleBL1& tensorBlockScaleBL1,
                                     const TileL1L0Param& tileL1L0Param, uint16_t iter1, uint64_t curKL0,
                                     uint64_t scaleKL0Len, uint64_t scaleK0Offset, uint64_t l0Offset)
    {
        // B, scaleB L1->L0
        auto layoutBL0 =
            asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL0, tileL1L0Param.curN);
        auto tensorBL0 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset), layoutBL0);
        auto tensorBlockBL1 =
            tensorBL1.slice(asc::te::make_coord(iter1 * baseK_, 0), asc::te::make_shape(curKL0, tileL1L0Param.curN));
        auto CopyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        asc::te::copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

        auto CopyL12L0ScaleB = asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{});
        auto layoutScaleBL0 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL0Len, tileL1L0Param.curN);
        // L0Scale copy uses 16-byte address units, while l0Offset is in bytes.
        auto tensorScaleBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scaleb, AscendC::fp8_e8m0_t>(l0Offset >> 4), layoutScaleBL0);
        asc::te::copy(CopyL12L0ScaleB, tensorScaleBL0,
                      tensorBlockScaleBL1.slice(asc::te::make_coord(scaleK0Offset, 0),
                                                asc::te::make_shape(scaleKL0Len, tileL1L0Param.curN)));
        return tensorBL0;
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorL0C>
    __aicore__ inline void ProcessL1Iteration(TensorA const& gmA, TensorB const& gmB, TensorScaleA const& gmScaleA,
                                              TensorScaleB const& gmScaleB, TensorBias const& gmBias,
                                              TensorL0C& tensorL0C, TileL1L0Param& tileL1L0Param, uint64_t iter0,
                                              uint64_t scaleKOffsetStride, uint64_t scaleKIter, uint64_t& scaleKIterIdx,
                                              bool isFirstSplitK, bool isLastSplitK, uint64_t totalSets,
                                              uint64_t& waited)
    {
        const uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
        const uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
        const uint64_t kL1Offset = iter0 * kL1_;
        const uint64_t scaleGmOffset = iter0 * scaleKOffsetStride;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        tileL1L0Param.curGmKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
        tileL1L0Param.curPadKL1 = Align64(tileL1L0Param.curGmKL1);

        // B/bias 不依赖 gather：先发 GM->L1，再等 gather。
        // bias GM->L1
        auto tensorBiasL1 = CopyBiasInL1(gmBias, tileL1L0Param, scaleL1BufId, iter0, isFirstSplitK);

        // B GM->L1; slice first, then copy.
        auto tensorBL1 = CopyBInL1(gmB, tileL1L0Param, l1BufId, kL1Offset);

        if (firstRound) {
            uint64_t waitTarget = CeilDiv((iter0 + 1) * kL1_, AG_SYNC_INTERVAL);
            if (waitTarget > totalSets) {
                waitTarget = totalSets;
            }
            while (waited < waitTarget) {
                AscendC::CrossCoreWaitFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE2>(CROSS_CORE_FLAG_NUM_NINE);
                waited++;
            }
        }

        // scaleA, scaleB GM->L1（scaleA 依赖 gather）
        auto scalePair = CopyScalesInL1(gmScaleA, gmScaleB, tileL1L0Param, scaleL1BufId, kL1Offset, scaleGmOffset,
                                        scaleKIterIdx == 0);
        auto& tensorScaleAL1 = scalePair.scaleA;
        auto& tensorScaleBL1 = scalePair.scaleB;

        // A GM->L1（依赖 gather）
        auto tensorAL1 = CopyAInL1(gmA, tileL1L0Param, l1BufId, kL1Offset);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

        const uint64_t scaleKOffset = scaleKIterIdx * scaleKOffsetStride;
        uint64_t scaleAKOffset = scaleKOffset;
        if constexpr (DispatchPolicy::FULL_LOAD_MODE != NONE_FULL_LOAD_MODE) {
            scaleAKOffset = scaleGmOffset;
        }
        const uint64_t biasBtOffset = baseN_ * scaleL1BufId * sizeof(float);
        Iterate(tileL1L0Param, iter0, scaleKOffsetStride, scaleKOffset, scaleAKOffset, biasBtOffset, isFirstSplitK,
                isLastSplitK, tensorScaleAL1, tensorScaleBL1, tensorAL1, tensorBL1, tensorBiasL1, tensorL0C);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        if (scaleKIterIdx + 1 == scaleKIter || iter0 + 1 == kL1Iter_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
            scaleLoopCnt_++;
            scaleKIterIdx = 0;
        } else {
            scaleKIterIdx++;
        }
        abL1LoopCnt_++;
    }

private:
    __aicore__ inline uint64_t AlignC0(uint64_t value)
    {
        if constexpr (IsFp4<AType>()) {
            return Align64(value);
        } else {
            return Align32(value);
        }
    }

    __aicore__ inline void InitNoneFullLoadL1BufferOffsets(uint64_t baseM, uint64_t l1HalfBufNum, uint64_t bL1OneBuffer,
                                                           uint64_t scaleBL1OneBuffer)
    {
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
        const uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE >> 1;
        const uint64_t aL1OneBuffer = (baseM * Align64(kL1_)) >> sizeShift;
        const uint64_t scaleAL1OneBuffer = baseM * (Align64(scaleKL1_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
            // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|bias0|...|A1A3|B1B3|AScale1|BScale1|bias1|...
            const uint64_t l1Offset = halfL1Size * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer * (bufferId >> 1);
            l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer * l1HalfBufNum + bL1OneBuffer * (bufferId >> 1);
        }
        const uint64_t scaleABaseOffset = bL1OneBuffer * l1HalfBufNum;
#pragma unroll
        for (int32_t bufferId = 0; bufferId < DOUBLE_BUFFER_COUNT; bufferId++) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + scaleABaseOffset;
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer;
        }
    }

    __aicore__ inline void InitAFullLoadL1BufferOffsets(uint64_t baseM, uint64_t l1HalfBufNum, uint64_t bL1OneBuffer,
                                                        uint64_t scaleBL1OneBuffer, uint64_t biasL1OneBuffer)
    {
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
        const uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE >> 1;
        uint64_t mAlign = 0;
        if constexpr (transA) {
            mAlign = AlignC0(baseM);
        } else {
            mAlign = Align16(baseM);
        }
        const uint64_t kAlign = Align64(k_);
        const uint64_t aL1OneBuffer = (mAlign * kAlign) >> sizeShift;
        const uint64_t scaleAL1OneBuffer = baseM * (Align64(k_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        // 2 buffer: L1 space is : B0|BScale0|bias0|A|AScale|...|B1|BScale1|bias1|
        // 4 buffer: L1 space is : B0B2|BScale0|bias0|A|AScale|...|B1B3|BScale1|bias1|...
        l1BufferAOffset_[0] = bL1OneBuffer * l1HalfBufNum + scaleBL1OneBuffer + biasL1OneBuffer;
        l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer;
        const uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer >= halfL1Size ?
                                      l1BufferScaleAOffset_[0] + scaleAL1OneBuffer :
                                      halfL1Size;
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer * (bufferId >> 1);
        }
        const uint64_t scaleBBaseOffset = bL1OneBuffer * l1HalfBufNum;
#pragma unroll
        for (int32_t bufferId = 0; bufferId < DOUBLE_BUFFER_COUNT; bufferId++) {
            l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + scaleBBaseOffset;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer;
        }
    }

    template <typename TensorL0C, typename TensorAL0, typename TensorBL0, typename TensorBT>
    __aicore__ inline void Mmad(const TileL1L0Param& tileL1L0Param, bool isFinalAccumulation, bool initCMatrix,
                                bool needBias, uint64_t curKL0, TensorL0C& tensorL0C, TensorAL0 const& tensorAL0,
                                TensorBL0 const& tensorBL0, TensorBT const& tensorBt)
    {
        asc::te::mmad_params params;
        params.m = static_cast<uint16_t>(tileL1L0Param.curM);
        params.k = static_cast<uint16_t>(Align64(curKL0));
        params.n = static_cast<uint16_t>(tileL1L0Param.curN);
        params.unit_flag =
            isFinalAccumulation ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
        params.init_with_zero = initCMatrix;
        if (needBias) {
            asc::te::mmad(
                asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            asc::te::mmad(
                asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0);
        }
    }

    static constexpr uint16_t SCALE_BUFFER_FLAG_0 = 4;
    static constexpr uint16_t SCALE_BUFFER_FLAG_1 = 5;
    static constexpr uint16_t M_MTE1_FLAG_0 = 4;
    static constexpr uint16_t M_MTE1_FLAG_1 = 5;

    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    bool firstRound{true};
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t splitKNum_{1};
    bool enableL0cPingPong_{false};

    uint64_t l1BufferAOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferScaleAOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferScaleBOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL};   // default 2 buffer
};
#endif
} // namespace Block
} // namespace Gemm
} // namespace Blaze
