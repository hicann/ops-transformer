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
 * \brief BlockMmad specialization for MatmulWithScaleMx with split-K hard sync support.
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

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)

static constexpr uint64_t AG_SYNC_INTERVAL = 1024UL;
static constexpr uint16_t CROSS_CORE_FLAG_NUM_NINE = 9;

template <uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class ScheduleType_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MatmulWithScaleMx<A_FULL_LOAD_MODE, ATOMIC_ADD, ScheduleType_>, AType_, LayoutA_, BType_, LayoutB_,
                CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using DispatchPolicy = MatmulWithScaleMx<A_FULL_LOAD_MODE, ATOMIC_ADD, ScheduleType_>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    static constexpr bool weightNz = IsWeightNz<LayoutB>::value;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;

    bool firstRound{true};

    static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr uint64_t SCALE_C0 = 2;

    __aicore__ inline uint64_t AlignC0(uint64_t value)
    {
        if constexpr (IsFp4<AType>()) {
            return Align64(value);
        } else {
            return Align32(value);
        }
    }

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;

    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t splitKNum_{1};
    bool enableL0cPingPong_{false};

    struct TileL1L0Param {
        uint64_t curM = 0;
        uint64_t curN = 0;
        uint64_t curGmKL1 = 0;
        uint64_t curPadKL1 = 0; // pad to 64 align
    };

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

    template <typename TensorScaleA, typename TensorScaleB>
    struct ScalePair {
        TensorScaleA scaleA;
        TensorScaleB scaleB;
    };

    __aicore__ inline BlockMmad()
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

    __aicore__ inline ~BlockMmad()
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

public:
    __aicore__ inline void Init(const ProblemShape &problemShape, const BlockShape &l0TileShape,
                                const L1Params &l1Params, bool isBias, bool dbL0C, uint64_t splitKNum = 1)
    {
        k_ = AscendC::Te::Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        splitKNum_ = splitKNum;
        baseM_ = AscendC::Te::Get<IDX_M_IDX>(l0TileShape);
        baseN_ = AscendC::Te::Get<IDX_N_IDX>(l0TileShape);
        baseK_ = AscendC::Te::Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
        const uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE >> 1;
        const uint64_t l1HalfBufNum = l1BufNum_ >> 1;
        bL1OneBuffer_ = (baseN_ * kL1_) >> sizeShift;
        scaleBL1OneBuffer_ = baseN_ * (Align64(scaleKL1_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        biasL1OneBuffer_ = isBias_ ? baseN_ * sizeof(BiasType) : 0;
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == 0) {
            aL1OneBuffer_ = (baseM_ * Align64(kL1_)) >> sizeShift;
            scaleAL1OneBuffer_ = baseM_ * (Align64(scaleKL1_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
                // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|bias0|...|A1A3|B1B3|AScale1|BScale1|bias1|...
                const uint64_t l1Offset = halfL1Size * (bufferId & 1);
                l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
                l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer_ * l1HalfBufNum + bL1OneBuffer_ * (bufferId >> 1);
            }
            const uint64_t scaleABaseOffset = bL1OneBuffer_ * l1HalfBufNum;
#pragma unroll
            for (int32_t bufferId = 0; bufferId < DOUBLE_BUFFER_COUNT; bufferId++) {
                l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + scaleABaseOffset;
                l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
                l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
            }
        } else {
            uint64_t mAlign = 0;
            if constexpr (transA) {
                mAlign = AlignC0(baseM_);
            } else {
                mAlign = Align16(baseM_);
            }
            const uint64_t kAlign = Align64(k_);
            aL1OneBuffer_ = (mAlign * kAlign) >> sizeShift;
            scaleAL1OneBuffer_ = baseM_ * (Align64(k_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
            // 2 buffer: L1 space is : B0|BScale0|bias0|A|AScale|...|B1|BScale1|bias1|
            // 4 buffer: L1 space is : B0B2|BScale0|bias0|A|AScale|...|B1B3|BScale1|bias1|...
            l1BufferAOffset_[0] = bL1OneBuffer_ * l1HalfBufNum + scaleBL1OneBuffer_ + biasL1OneBuffer_;
            l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer_;
            const uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ >= halfL1Size ?
                                          l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ :
                                          halfL1Size;
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            const uint64_t scaleBBaseOffset = bL1OneBuffer_ * l1HalfBufNum;
#pragma unroll
            for (int32_t bufferId = 0; bufferId < DOUBLE_BUFFER_COUNT; bufferId++) {
                l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + scaleBBaseOffset;
                l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
            }
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyScalesInL1(TensorScaleA const &gmScaleA, TensorScaleB const &gmScaleB,
                                          const TileL1L0Param &tileL1L0Param, uint64_t scaleL1BufId, uint64_t kL1Offset,
                                          uint64_t scaleGmOffset, bool needCopyScale)
    {
        const uint64_t scaleKL1Len = (Align64(scaleKL1_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        const uint64_t scaleBL1Offset = l1BufferScaleBOffset_[scaleL1BufId];
        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL1Len, tileL1L0Param.curN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AscendC::fp8_e8m0_t>(scaleBL1Offset), layoutScaleBL1);
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == 0) {
            // L1 uses the full scaleKL1_ length; GM uses the actual length, which may be a tail block.
            const uint64_t scaleAL1Offset = l1BufferScaleAOffset_[scaleL1BufId];
            auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                tileL1L0Param.curM, scaleKL1Len);
            auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AscendC::fp8_e8m0_t>(scaleAL1Offset),
                layoutScaleAL1);
            if (needCopyScale) {
                auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }
                const uint64_t curScaleKLen = (Align64(curScaleKL1) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
                auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(0, scaleGmOffset),
                                                    AscendC::Te::MakeShape(tileL1L0Param.curM, curScaleKLen));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

                auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(scaleGmOffset, 0),
                                                    AscendC::Te::MakeShape(curScaleKLen, tileL1L0Param.curN));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            }
            return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
        } else {
            const uint64_t scaleAL1Offset = l1BufferScaleAOffset_[0];
            const uint64_t scaleKLen = (Align64(k_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
            auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                tileL1L0Param.curM, scaleKLen);
            auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AscendC::fp8_e8m0_t>(scaleAL1Offset),
                layoutScaleAL1);
            auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            if (needCopyScale) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }
                const uint64_t curScaleKLen = (Align64(curScaleKL1) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
                auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(scaleGmOffset, 0),
                                                    AscendC::Te::MakeShape(curScaleKLen, tileL1L0Param.curN));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            }
            if (abL1LoopCnt_ == 0) {
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmScaleA);
            }
            return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
        }
    }

    template <typename TensorA>
    __aicore__ inline auto CopyAInL1(TensorA const &gmA, const TileL1L0Param &tileL1L0Param, uint64_t l1BufId,
                                     uint64_t kL1Offset)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == 0) {
            const uint64_t aL1Offset = l1BufferAOffset_[l1BufId];
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadKL1);
            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, kL1Offset),
                                      AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curGmKL1));
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(aL1Offset), layoutAL1);
            Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
            return tensorAL1;
        } else {
            const uint64_t aL1Offset = l1BufferAOffset_[0];
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, Align64(k_));
            auto tensorTotalAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(aL1Offset), layoutAL1);
            auto tensorAL1 = tensorTotalAL1.Slice(AscendC::Te::MakeCoord(0, kL1Offset),
                                                  AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curPadKL1));
            if (abL1LoopCnt_ < kL1Iter_) {
                auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, kL1Offset),
                                          AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curGmKL1));
                Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
                AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
            }
            return tensorAL1;
        }
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1, typename TensorAL1, typename TensorBL1,
              typename TensorBiasL1, typename TensorL0C>
    __aicore__ inline void Iterate(const TileL1L0Param &tileL1L0Param, uint64_t iter0, uint64_t scaleKL1Len,
                                   uint64_t scaleKOffset, uint64_t scaleAKOffset, uint64_t biasBtOffset,
                                   bool isFirstSplitK, bool isLastSplitK, TensorScaleAL1 &tensorScaleAL1,
                                   TensorScaleBL1 &tensorScaleBL1, TensorAL1 &tensorAL1, TensorBL1 &tensorBL1,
                                   TensorBiasL1 &tensorBiasL1, TensorL0C &tensorL0C)
    {
        // Slice scaleKL1 to current kL1 window.
        auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(AscendC::Te::MakeCoord(scaleKOffset, 0),
                                                        AscendC::Te::MakeShape(scaleKL1Len, tileL1L0Param.curN));
        auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(AscendC::Te::MakeCoord(0, scaleAKOffset),
                                                        AscendC::Te::MakeShape(tileL1L0Param.curM, scaleKL1Len));

        const uint64_t baseK = baseK_;
        const bool hasBias = isBias_;
        const uint64_t kL0Iter = Blaze::Gemm::CeilDiv(tileL1L0Param.curGmKL1, baseK);
        const uint64_t scaleK0OffsetStride = (Align64(baseK) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        const bool isLastL1Iter = (iter0 + 1 == kL1Iter_);
        auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto CopyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
        auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        auto CopyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            const uint64_t curKL0 =
                (iter1 * baseK + baseK > tileL1L0Param.curPadKL1) ? (tileL1L0Param.curPadKL1 - iter1 * baseK) : baseK;
            const uint64_t scaleKL0Len = (Align64(curKL0) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
            const uint64_t scaleK0Offset = iter1 * scaleK0OffsetStride;
            // Load data to L0 and open DB
            const uint64_t l0PingPongId = l0PingPong_ & 1;
            const uint64_t l0Offset = HALF_L0_SIZE * l0PingPongId;
            const uint16_t mte1WaitMFlag = static_cast<uint16_t>(l0PingPongId + M_MTE1_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            // A, ScaleA L1->L0
            auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(
                tileL1L0Param.curM, curKL0);
            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto tensorBlockAL1 = tensorAL1.Slice(AscendC::Te::MakeCoord(0, iter1 * baseK),
                                                  AscendC::Te::MakeShape(tileL1L0Param.curM, curKL0));
            AscendC::Te::Copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

            auto layoutScaleAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                tileL1L0Param.curM, scaleKL0Len);
            // L0Scale copy uses 16-byte address units, while l0Offset is in bytes.
            auto tensorScaleAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, AscendC::fp8_e8m0_t>(l0Offset >> 4),
                layoutScaleAL0);
            AscendC::Te::Copy(CopyL12L0ScaleA, tensorScaleAL0,
                              tensorBlockScaleAL1.Slice(AscendC::Te::MakeCoord(0, scaleK0Offset),
                                                        AscendC::Te::MakeShape(tileL1L0Param.curM, scaleKL0Len)));

            // bias L1->BT
            auto layoutBt = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, Align16(tileL1L0Param.curN));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(biasBtOffset), layoutBt);
            const bool needBias = hasBias && iter0 == 0 && iter1 == 0 && isFirstSplitK;
            if (needBias) {
                auto CopyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
                AscendC::Te::Copy(CopyL12BT, tensorBt, tensorBiasL1);
            }

            // B, scaleB L1->L0
            auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(
                curKL0, tileL1L0Param.curN);
            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
            auto tensorBlockBL1 = tensorBL1.Slice(AscendC::Te::MakeCoord(iter1 * baseK, 0),
                                                  AscendC::Te::MakeShape(curKL0, tileL1L0Param.curN));
            AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

            auto layoutScaleBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                scaleKL0Len, tileL1L0Param.curN);
            // L0Scale copy uses 16-byte address units, while l0Offset is in bytes.
            auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, AscendC::fp8_e8m0_t>(l0Offset >> 4),
                layoutScaleBL0);
            AscendC::Te::Copy(CopyL12L0ScaleB, tensorScaleBL0,
                              tensorBlockScaleBL1.Slice(AscendC::Te::MakeCoord(scaleK0Offset, 0),
                                                        AscendC::Te::MakeShape(scaleKL0Len, tileL1L0Param.curN)));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongId);

            const bool isFinalAccumulation = isLastL1Iter && (iter1 + 1 == kL0Iter) && isLastSplitK;
            const bool initCMatrix = iter0 == 0 && iter1 == 0 && isFirstSplitK && !needBias;
            Mmad(tileL1L0Param, isFinalAccumulation, initCMatrix, needBias, curKL0, tensorL0C, tensorAL0, tensorBL0,
                 tensorBt);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(TensorA const &gmA, TensorB const &gmB, TensorScaleA const &gmScaleA,
                                      TensorScaleB const &gmScaleB, TensorBias const &gmBias, TensorC const &gmC,
                                      BlockShape const &singleShape, uint64_t splitKIdx = 0)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        const uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);
        const bool isFirstSplitK = splitKIdx == 0;
        const bool isLastSplitK = splitKIdx == splitKNum_ - 1;

        const uint64_t scaleKIter = scaleKL1_ / kL1_;
        uint64_t scaleKIterIdx = 0;
        const uint64_t scaleKOffsetStride = (Align64(kL1_) >> ALIGN_64_BYTES_SHIFT) * MXFP_MULTI_BASE_SIZE;
        const uint64_t l1BufMask = l1BufNum_ - 1;
        // split-K hard sync: AIV 每就绪 AG_SYNC_INTERVAL(1024) 元素 set(9) 一次，每核共 ceil(k/1024) 次。
        // AIC 读第 iter0 段 K [iter0*kL1,(iter0+1)*kL1) 前，需累计等到 ceil((iter0+1)*kL1/1024) 次，
        // 保证该段所有列已 gather 完成。循环末尾 drain 补齐，确保 wait 次数恒等于 AIV set 次数。
        const uint64_t agTotalSets = CeilDiv(k_, AG_SYNC_INTERVAL);
        uint64_t agWaited = 0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            if (firstRound) {
                uint64_t waitTarget = CeilDiv((iter0 + 1) * kL1_, AG_SYNC_INTERVAL);
                if (waitTarget > agTotalSets) {
                    waitTarget = agTotalSets;
                }
                while (agWaited < waitTarget) {
                    AscendC::CrossCoreWaitFlag(CROSS_CORE_FLAG_NUM_NINE);
                    agWaited++;
                }
            }

            const uint64_t l1BufId = abL1LoopCnt_ & l1BufMask;
            const uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            const uint64_t kL1Offset = iter0 * kL1_;
            const uint64_t scaleGmOffset = iter0 * scaleKOffsetStride;

            // scaleA, scaleB GM->L1
            auto scalePair = CopyScalesInL1(gmScaleA, gmScaleB, tileL1L0Param, scaleL1BufId, kL1Offset, scaleGmOffset,
                                            scaleKIterIdx == 0);
            auto &tensorScaleAL1 = scalePair.scaleA;
            auto &tensorScaleBL1 = scalePair.scaleB;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            tileL1L0Param.curGmKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
            tileL1L0Param.curPadKL1 = Align64(tileL1L0Param.curGmKL1);

            // A GM->L1
            auto tensorAL1 = CopyAInL1(gmA, tileL1L0Param, l1BufId, kL1Offset);

            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            // bias GM->L1
            const uint64_t biasL1Offset = l1BufferBiasOffset_[scaleL1BufId];
            auto layoutBiasL1 =
                AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>{}(1UL, Align16(tileL1L0Param.curN));
            auto tensorBiasL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(biasL1Offset), layoutBiasL1);
            if (isBias_ && iter0 == 0 && isFirstSplitK) {
                AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
            }

            // B GM->L1; slice first, then copy.
            const uint64_t bL1Offset = l1BufferBOffset_[l1BufId];
            auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curPadKL1, tileL1L0Param.curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(bL1Offset), layoutBL1);
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kL1Offset, 0),
                                      AscendC::Te::MakeShape(tileL1L0Param.curGmKL1, tileL1L0Param.curN));
            Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            const uint64_t scaleKOffset = scaleKIterIdx * scaleKOffsetStride;
            uint64_t scaleAKOffset = scaleKOffset;
            if constexpr (DispatchPolicy::FULL_LOAD_MODE != 0) {
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
        if (isLastSplitK) {
            if constexpr (AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>, AscendC::Te::Location::UB>) {
                // C L0C->UB
                auto CopyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{});
                AscendC::Te::Copy(CopyL0C2UB.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
            } else {
                // C L0C->GM
                auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
                AscendC::Te::Copy(CopyL0C2GM.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
            }
            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
        if (firstRound) {
            // drain 剩余 set(9)，确保 AIC wait 次数恒等于 AIV set 次数（ceil(k/AG_SYNC_INTERVAL)），
            // 避免残留计数影响后续（虽然只有 firstRound wait，但计数须收支平衡）。
            while (agWaited < agTotalSets) {
                AscendC::CrossCoreWaitFlag(CROSS_CORE_FLAG_NUM_NINE);
                agWaited++;
            }
        }
        firstRound = false;
    }

private:
    template <typename TensorL0C, typename TensorAL0, typename TensorBL0, typename TensorBT>
    __aicore__ inline void Mmad(const TileL1L0Param &tileL1L0Param, bool isFinalAccumulation, bool initCMatrix,
                                bool needBias, uint64_t curKL0, TensorL0C &tensorL0C, TensorAL0 const &tensorAL0,
                                TensorBL0 const &tensorBL0, TensorBT const &tensorBt)
    {
        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(tileL1L0Param.curM);
        params.k = static_cast<uint16_t>(Align64(curKL0));
        params.n = static_cast<uint16_t>(tileL1L0Param.curN);
        params.unitFlag = isFinalAccumulation ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        params.cmatrixInitVal = initCMatrix;
        if (needBias) {
            AscendC::Te::Mmad(AscendC::Te::MmadAtom<
                                  AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                                  .with(params),
                              tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            AscendC::Te::Mmad(AscendC::Te::MmadAtom<
                                  AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                                  .with(params),
                              tensorL0C, tensorAL0, tensorBL0);
        }
    }

    static constexpr uint16_t SCALE_BUFFER_FLAG_0 = 4;
    static constexpr uint16_t SCALE_BUFFER_FLAG_1 = 5;
    static constexpr uint16_t M_MTE1_FLAG_0 = 4;
    static constexpr uint16_t M_MTE1_FLAG_1 = 5;

    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
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
