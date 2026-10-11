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
 * \file block_prologue_mx_fp8fp4.h
 * \brief Weight-quantized grouped matmul prologue with MXFP8/FP4 casting
 *
 * ## Overview
 * This prologue prepares weight data for grouped matrix multiplication by:
 * 1. Loading 4-bit quantized weights from Global Memory (GM)
 * 2. Anti-quantizing to 8-bit format using vector instructions
 * 3. Moving processed weights to L1 cache for Cube unit consumption
 *
 * ## Pipeline Architecture (3-Stage)
 *
 * ```
 * ┌─────────┐    ┌─────────┐    ┌─────────┐
 * │  MTE2   │───→│  Vector │───→│  MTE3   │
 * │  (Load) │    │(Compute)│    │ (Store) │
 * └─────────┘    └─────────┘    └─────────┘
 *      │              │              │
 *      ▼              ▼              ▼
 *   GM→UB       Anti-quant      UB→L1
 *   4-bit       4-bit→8-bit    8-bit
 * ```
 *
 * ### Stage 1: MTE2 (Memory Transfer Engine 2)
 * - Copies 4-bit weight from GM to Unified Buffer (UB)
 * - Uses quad-buffering for 4-way parallel access
 * - Triggered per K-block iteration
 *
 * ### Stage 2: Vector (Anti-quantization)
 * - Converts 4-bit weights to 8-bit using SIMD instructions
 * - Uses quad-buffering for 8-bit weight output
 *
 * ### Stage 3: MTE3 (Memory Transfer Engine 3)
 * - Moves 8-bit weights from UB to L1 cache
 * - Signals AIC (Cube unit) when data is ready
 *
 * ## Unified Buffer Layout
 *
 * | Offset    | Size  | Content                    | Buffering |
 * |-----------|-------|---------------------------|-----------|
 * | 0KB       | 64KB  | 4-bit weight quad         | Quad      |
 * | 64KB      | 128KB | 8-bit weight quad         | Quad      |
 * | **Total (used)** | **192KB** | Within 248KB UB hardware limit | |
 *
 * ## L1 Buffer Layout
 * Host tiling supplies the W8 B-slot count and byte offsets through MegaMoeL1Layout.
 * AIC and AIV0 read the same layout; logical B slots alternate between two physical halves.
 * Each slot holds one K256 x N256 tile; tail tiles copy only their valid extent.
 *
 * ## Key Design Decisions
 *
 * 1. **Quad buffering for 4-bit weight**: Allows MTE2 to load multiple blocks ahead
 *    while Vector unit processes earlier blocks
 *
 * 2. **Quad buffering for 8-bit weight**: Supports the 3-stage pipeline where
 *    MTE2, Vector, and MTE3 can operate on different buffers simultaneously
 *
 * 3. **K-block splitting**: Large K dimensions are split to fit in UB/L1 constraints,
 *    with each sub-block processed independently
 *
 * ## SIMD Model
 *
 * - **AIV (AI Vector) cores**: Execute MTE2, Vector, and MTE3 operations
 * - **AIC (AI Cube) cores**: Consume prepared weights from L1
 * - **Synchronization**: Cross-core flags (0/1) coordinate between AIV and AIC
 *
 * ## Data Types
 *
 * - Input weight: 4-bit per element (packed as 2 elements per byte)
 * - Output weight: 8-bit per element (fp8_e4m3)

 */
#pragma once
#include "../gemm/tile/copy_gmm1_concat.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/tile/copy_gm_to_ub.h"
#include "blaze/gemm/tile/shift_w4_to_w8.h"
#include "blaze/gemm/tile/copy_weight_ub_to_l1.h"
#include "block_prologue.h"
#include "../gemm/utils/common_utils_mega_moe.h"
#include "../../mega_moe_tiling.h"

namespace Blaze {
namespace Gemm {
namespace Prologue {

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

using Blaze::Gemm::DOUBLE_BUFFER;
using Blaze::Gemm::SYNC_MODE4;

static constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;
static constexpr uint64_t SERIAL_HALF_K_PARTS = 2;

struct PrologueMxCastWOffsetParam {
    uint64_t kSize;
    uint64_t kbL1Size;
    uint64_t nL1Size;
    uint64_t nOffset;
    uint64_t nAlign;
    uint64_t concatHalfN;
};

// Macro aliases keep this dispatch-policy specialization concise and readable.
#define BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS template <class OutType_, class InType_>

#define BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION BlockPrologue<MatmulMxFp8Fp4DynamicKL1TailResplit, OutType_, InType_>

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
class BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION {
public:
    using DispatchPolicy = MatmulMxFp8Fp4DynamicKL1TailResplit;
    using OutType = OutType_;
    using InType = InType_;

    static constexpr uint64_t kUbMte2BufferNum = 4;

    struct Params {
        __gm__ InType* ptrB;
    };

    __aicore__ explicit inline BlockPrologue(const MegaMoeL1Layout& layout);
    template <typename GMWeightTensorType>
    __aicore__ inline void operator()(const GMWeightTensorType& gmWeightTensor, uint64_t kSize, uint64_t nL1Size,
                                      uint64_t nOffset, uint64_t nAlign, uint64_t concatHalfN = 0);
    __aicore__ inline ~BlockPrologue();

protected:
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAicToAiv();
    template <typename GMWeightTensorType>
    __aicore__ inline void ComputeBasicBlockAivNdKnNzNk(const PrologueMxCastWOffsetParam& offsetParam,
                                                        const GMWeightTensorType& gmWeightTensor);

    template <typename GMWeightTensorType>
    __aicore__ inline void CopyWeightToUb(const PrologueMxCastWOffsetParam& param,
                                          const GMWeightTensorType& gmWeightTensor, uint64_t halfSeq);
    __aicore__ inline void WaitVectorToMTE2();
    __aicore__ inline void SetVectorToMTE2();
    template <typename Weight4BitTensorType, typename Weight8BitTensorType>
    __aicore__ inline void WeightAntiQuantComputeNzNk(const Weight4BitTensorType& weight4BitTensor,
                                                      const Weight8BitTensorType& weight8BitTensor);
    template <typename Weight8BitTensorType, typename L1TensorType>
    __aicore__ inline void CopyWeightToL1(uint64_t mte2RealK, const Weight8BitTensorType& weight8BitTensor,
                                          const L1TensorType& l1Tensor);

    __aicore__ inline void FinalizeVectorCompute();

    // GM to UB copy function
    template <typename GMWeightBaseTensorType, typename Weight4BitTensorType>
    __aicore__ inline void CopyGmToUb(uint64_t kOffset, uint64_t mte2RealK, const PrologueMxCastWOffsetParam& param,
                                      const GMWeightBaseTensorType& gmWeightBaseTensor,
                                      const Weight4BitTensorType& weight4BitTensor);

    // === Tensor Creation Helper Functions ===
    // Weight tensor creation functions
    __aicore__ inline auto MakeWeight4BitTensor(uint64_t mte2RealK, uint64_t nL1Size);
    __aicore__ inline auto MakeWeight8BitTensor(uint64_t mte2RealK, uint64_t nL1Size);

    // L1 tensor creation functions
    __aicore__ inline auto MakeL1WeightTensor(uint64_t l1RealLen, uint64_t nL1Size);

    const MegaMoeL1Layout layout_;
    uint64_t cvLoopIdx_ = 0;

    uint64_t ubMte2LoopIdx_ = 0;
    uint64_t ubComputeLoopIdx_ = 0;

    // === Buffer Size Unit ===
    static constexpr uint64_t KB = 1024;
    // === Pipeline Buffer Configuration ===
    static constexpr uint64_t WEIGHT_8BIT_BUFFER_NUM = QUADRUPLE_BUFFER_NUM; // 4

    // === UB Memory Layout (192KB actively used in this pipeline) ===
    // [0-64KB): 4-bit weight quad buffers
    static constexpr uint64_t WEIGHT_4BIT_TOTAL_SIZE = 64 * KB;
    static constexpr uint64_t WEIGHT_4BIT_SINGLE_BUFFER_SIZE = WEIGHT_4BIT_TOTAL_SIZE / kUbMte2BufferNum;

    // [64KB-192KB): 8-bit weight quad buffers
    static constexpr uint64_t WEIGHT_8BIT_TOTAL_SIZE = 128 * KB;

    // UB hardware usable ceiling: TOTAL_UB_SIZE (256KB) minus ~8KB system reserved.
    static constexpr uint64_t UB_MAX_USABLE_SIZE = 248 * KB;

    // Compile-time verification: active use stays within the UB hardware usable ceiling.
    static_assert(WEIGHT_4BIT_TOTAL_SIZE + WEIGHT_8BIT_TOTAL_SIZE <= UB_MAX_USABLE_SIZE,
                  "UB buffer total must not exceed 248KB hardware limit");

    // === UB Buffer Base Offsets ===
    static constexpr uint64_t WEIGHT_4BIT_INIT_OFFSET = 0;
    static constexpr uint64_t WEIGHT_8BIT_INIT_OFFSET = WEIGHT_4BIT_TOTAL_SIZE;

    // === UB Buffer Offset Arrays (Compile-time computed for fast lookup) ===
    static constexpr uint64_t WEIGHT_4BIT_OFFSETS[4] = {WEIGHT_4BIT_INIT_OFFSET + 0 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
                                                        WEIGHT_4BIT_INIT_OFFSET + 1 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
                                                        WEIGHT_4BIT_INIT_OFFSET + 2 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
                                                        WEIGHT_4BIT_INIT_OFFSET + 3 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE};

    // === Hardware/Architecture Parameters ===
#if __CCE_AICORE__ == 310
    constexpr static uint64_t VEC_REG_ELEM = AscendC::VECTOR_REG_WIDTH;
#else
    constexpr static uint64_t VEC_REG_ELEM = 256;
#endif

    static constexpr uint64_t WEIGHT_8BIT_OFFSETS[4] = {WEIGHT_8BIT_INIT_OFFSET + 0 * VEC_REG_ELEM * sizeof(OutType),
                                                        WEIGHT_8BIT_INIT_OFFSET + 1 * VEC_REG_ELEM * sizeof(OutType),
                                                        WEIGHT_8BIT_INIT_OFFSET + 2 * VEC_REG_ELEM * sizeof(OutType),
                                                        WEIGHT_8BIT_INIT_OFFSET + 3 * VEC_REG_ELEM * sizeof(OutType)};
    static constexpr uint64_t WEIGHT_8BIT_LAYOUT_INNER_SIZE = VEC_REG_ELEM * WEIGHT_8BIT_BUFFER_NUM;

    // === Event IDs for Pipeline Synchronization ===
    constexpr static TEventID vecEventIdVToMte2_ = 0;
    constexpr static TEventID vecEventIdMte3ToV_ = 0;
    constexpr static TEventID EVENT_ID_MTE2_TO_V = 0;

    // === Cross-Core Synchronization Flags ===
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 1;
};

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::BlockPrologue(const MegaMoeL1Layout& layout)
    : layout_(layout)
{
    for (uint16_t idx = 0; idx < kUbMte2BufferNum; idx++) {
        SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + idx);
    }

    for (uint16_t idx = 0; idx < WEIGHT_8BIT_BUFFER_NUM; idx++) {
        SetFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_ + idx);
    }
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
template <typename GMWeightTensorType>
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::ComputeBasicBlockAivNdKnNzNk(
    const PrologueMxCastWOffsetParam& param, const GMWeightTensorType& gmWeightTensor)
{
    const uint64_t kMte2BaseSize = param.kbL1Size >> 1;
    const uint64_t totalHalves = CeilDiv(param.kSize, param.kbL1Size) * SERIAL_HALF_K_PARTS;
    uint64_t nextHalf = 0;
    for (; nextHalf < min(uint64_t{2}, totalHalves); ++nextHalf) {
        CopyWeightToUb(param, gmWeightTensor, nextHalf);
    }
    for (uint64_t kOffset = 0; kOffset < param.kSize; kOffset += param.kbL1Size, cvLoopIdx_++) {
        const uint64_t l1RealLen = min(param.kbL1Size, param.kSize - kOffset);
        auto l1BaseTensor = MakeL1WeightTensor(l1RealLen, param.nL1Size);
        for (uint64_t halfIdx = 0; halfIdx < SERIAL_HALF_K_PARTS; ++halfIdx) {
            const uint64_t l1SplitOffset = halfIdx * kMte2BaseSize;
            const uint64_t mte2RealK = l1RealLen > l1SplitOffset ? min(kMte2BaseSize, l1RealLen - l1SplitOffset) : 0;
            const uint64_t slot = ubComputeLoopIdx_ & (kUbMte2BufferNum - 1);
            auto weight4BitTensor =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, InType>(WEIGHT_4BIT_OFFSETS[slot]),
                                     asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<32>>{}(
                                         static_cast<int64_t>(mte2RealK), static_cast<int64_t>(param.nL1Size)));
            auto weight8BitTensor = MakeWeight8BitTensor(mte2RealK, param.nL1Size);
            auto l1Tensor = l1BaseTensor.slice(asc::te::make_coord(l1SplitOffset, 0),
                                               asc::te::make_shape(mte2RealK, param.nL1Size));
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V + slot);
            WeightAntiQuantComputeNzNk(weight4BitTensor, weight8BitTensor);
            SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + slot);
            // Publish another independent GM load before the current MTE3 write.
            if (nextHalf < totalHalves) {
                CopyWeightToUb(param, gmWeightTensor, nextHalf++);
            }
            if (halfIdx == 0) {
                WaitAicToAiv();
            }
            CopyWeightToL1(mte2RealK, weight8BitTensor, l1Tensor);
        }
        SetAivToAic();
    }
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
template <typename GMWeightTensorType>
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::operator()(const GMWeightTensorType& gmWeightTensor,
                                                                           uint64_t kSize, uint64_t nL1Size,
                                                                           uint64_t nOffset, uint64_t nAlign,
                                                                           uint64_t concatHalfN)
{
    // Type assertions - __aicore__ guarantees these types are valid
    static_assert(std::is_same_v<OutType, __fp8e4m3>, "OutType must be __fp8e4m3");
    static_assert(std::is_same_v<InType, __fp4e2m1x2>, "InType must be __fp4e2m1x2");

    PrologueMxCastWOffsetParam offsetParam = {};
    offsetParam.kSize = kSize;
    offsetParam.nL1Size = nL1Size;
    offsetParam.nOffset = nOffset;
    offsetParam.kbL1Size = layout_.tileK;
    offsetParam.nAlign = nAlign;
    offsetParam.concatHalfN = concatHalfN;
    ComputeBasicBlockAivNdKnNzNk(offsetParam, gmWeightTensor);
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::~BlockPrologue()
{
    for (uint64_t idx = 0; idx < layout_.bBufferNum; ++idx) {
        WaitAicToAiv();
    }
    FinalizeVectorCompute();
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::SetAivToAic()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::WaitAicToAiv()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::WaitVectorToMTE2()
{
    WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + (ubMte2LoopIdx_ & (kUbMte2BufferNum - 1)));
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::SetVectorToMTE2()
{
    SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + (ubMte2LoopIdx_ & (kUbMte2BufferNum - 1)));
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
template <typename GMWeightBaseTensorType, typename Weight4BitTensorType>
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::CopyGmToUb(
    uint64_t kOffset, uint64_t mte2RealK, const PrologueMxCastWOffsetParam& param,
    const GMWeightBaseTensorType& gmWeightBaseTensor, const Weight4BitTensorType& weight4BitTensor)
{
    if (mte2RealK > 0) {
        auto gmSliceTensor = gmWeightBaseTensor.slice(asc::te::make_coord(kOffset, param.nOffset),
                                                      asc::te::make_shape(mte2RealK, param.nL1Size));
        auto copyGM2UBWeight = asc::te::make_copy(Blaze::Gemm::Tile::CopyGM2UBWeight{});
        if (param.concatHalfN != 0U) {
            MegaMoeImpl::CopyGmm1WeightConcatToUb(weight4BitTensor, gmSliceTensor, param.concatHalfN);
        } else {
            asc::te::copy(copyGM2UBWeight, weight4BitTensor, gmSliceTensor);
        }
    }
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
template <typename Weight4BitTensorType, typename Weight8BitTensorType>
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::WeightAntiQuantComputeNzNk(
    const Weight4BitTensorType& weight4BitTensor, const Weight8BitTensorType& weight8BitTensor)
{
    WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_ + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));

    // Pure compute: anti-quantization using tile helper
    Blaze::Gemm::Tile::ShiftW4ToW8<OutType, InType>(weight4BitTensor, weight8BitTensor);

    // Set/Wait flags AFTER compute
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
template <typename Weight8BitTensorType, typename L1TensorType>
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::CopyWeightToL1(
    uint64_t mte2RealK, const Weight8BitTensorType& weight8BitTensor, const L1TensorType& l1Tensor)
{
    if (likely(mte2RealK > 0)) {
        // Copy weight 8-bit from UB to L1 (inlined from CopyWeight8BitForAligned)
        auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyUB2L1Weight8Bit{});
        asc::te::copy(copyUB2L1, l1Tensor, weight8BitTensor);
    }
    SetFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_ + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));
    ubComputeLoopIdx_++;
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::FinalizeVectorCompute()
{
    for (uint16_t idx = 0; idx < WEIGHT_8BIT_BUFFER_NUM; idx++) {
        WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV_ + idx);
    }

    for (uint16_t idx = 0; idx < kUbMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2_ + idx);
    }
}

// === Tensor Creation Helper Function Implementations ===

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline auto BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::MakeWeight4BitTensor(uint64_t mte2RealK,
                                                                                     uint64_t nL1Size)
{
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, InType>(
                                    WEIGHT_4BIT_OFFSETS[ubMte2LoopIdx_ & (kUbMte2BufferNum - 1)]),
                                asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<32>>{}(
                                    static_cast<int64_t>(mte2RealK), static_cast<int64_t>(nL1Size)));
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline auto BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::MakeWeight8BitTensor(uint64_t mte2RealK,
                                                                                     uint64_t nL1Size)
{
    return asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, OutType>(
            WEIGHT_8BIT_OFFSETS[ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)]),
        Blaze::Gemm::Weight8BitUBLayout<OutType, WEIGHT_8BIT_LAYOUT_INNER_SIZE>{}(mte2RealK, nL1Size));
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline auto BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::MakeL1WeightTensor(uint64_t l1RealLen, uint64_t nL1Size)
{
    auto l1BaseLayout =
        asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<OutType>>(l1RealLen, nL1Size);
    const uint64_t weightL1Offset = layout_.bOffsets[cvLoopIdx_ % layout_.bBufferNum];
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, OutType>(weightL1Offset), l1BaseLayout);
}

BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
template <typename GMWeightTensorType>
__aicore__ inline void BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION::CopyWeightToUb(const PrologueMxCastWOffsetParam& param,
                                                                               const GMWeightTensorType& gmWeightTensor,
                                                                               uint64_t halfSeq)
{
    const uint64_t kMte2BaseSize = param.kbL1Size >> 1;
    const uint64_t kOffset =
        (halfSeq / SERIAL_HALF_K_PARTS) * param.kbL1Size + (halfSeq % SERIAL_HALF_K_PARTS) * kMte2BaseSize;
    const uint64_t mte2RealK = kOffset < param.kSize ? min(kMte2BaseSize, param.kSize - kOffset) : 0;
    const uint64_t slot = ubMte2LoopIdx_ & (kUbMte2BufferNum - 1);
    WaitVectorToMTE2();
    auto weight4BitTensor = MakeWeight4BitTensor(mte2RealK, param.nL1Size);
    CopyGmToUb(kOffset, mte2RealK, param, gmWeightTensor, weight4BitTensor);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V + slot);
    ++ubMte2LoopIdx_;
}

#undef BLOCK_PROLOGUE_MX_FP8FP4_SPECIALIZATION
#undef BLOCK_PROLOGUE_MX_FP8FP4_TEMPLATE_PARAMS
} // namespace Prologue
} // namespace Gemm
} // namespace Blaze
