/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP
#define BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP

#include "../../../attn_infra/generic_block_sparse_attention_base_defs.hpp"
#include "../../../attn_infra/arch/generic_block_sparse_attention_cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "../../../attn_infra/epilogue/generic_block_sparse_attention_dispatch_policy.hpp"

namespace NpuArch::Epilogue::Block {

template <class OutputType_, class InputType_, class MaskType_, LseMode LSE_MODE_>
class BlockEpilogue<EpilogueAtlasA2OnlineSoftmax<LSE_MODE_, float>, OutputType_, InputType_, MaskType_> {
public:
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmax<LSE_MODE_, float>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    static constexpr LseMode LSE_MODE = DispatchPolicy::LSE_MODE;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_16 = 16;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    static constexpr bool ANTIQUANT = !std::is_same<ElementInput, float>::value;
    static constexpr float antiqCoeff1 = 127.0f;
    static constexpr float antiqCoeff2 = 1.0f / 127.0f;
    static constexpr float antiquantExpandCoeff = 254.0f;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;

        scaleValue = scaleValue_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);
        maskUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(MASK32_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);

        if constexpr (ANTIQUANT) {
            // Antiquant buffers (reuse MASK region which is unused in this kernel)
            constexpr uint32_t ANTIQ_TMP_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;  // 65536
            constexpr uint32_t ANTIQ_TMP2_OFFSET = 6 * UB_UINT8_BLOCK_SIZE; // 98304
            constexpr uint32_t ANTIQ_AMAX_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 16 * UB_UINT8_VECTOR_SIZE;
            constexpr uint32_t ANTIQ_ROWMAX_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 17 * UB_UINT8_VECTOR_SIZE;
            // K/V scale buffers: the stacked-pages path loads up to eight
            // 128-token pages per tile. Keep them disjoint from Amax_Q.
            constexpr uint32_t ANTIQ_KSCALE_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 18 * UB_UINT8_VECTOR_SIZE;
            constexpr uint32_t ANTIQ_VSCALE_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 23 * UB_UINT8_VECTOR_SIZE;
            constexpr uint32_t ANTIQ_AMAXQ_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 27 * UB_UINT8_VECTOR_SIZE;

            antiqTmpInt32 = resource.ubBuf.template GetBufferByByte<int32_t>(ANTIQ_TMP_OFFSET);
            antiqTmpFloat = resource.ubBuf.template GetBufferByByte<float>(ANTIQ_TMP_OFFSET);
            antiqTmpFloor = resource.ubBuf.template GetBufferByByte<float>(ANTIQ_TMP2_OFFSET);
            antiqResHalf = resource.ubBuf.template GetBufferByByte<half>(ANTIQ_TMP2_OFFSET);
            antiqResInt8 = resource.ubBuf.template GetBufferByByte<int8_t>(ANTIQ_TMP2_OFFSET);
            antiqTmpRowMax = resource.ubBuf.template GetBufferByByte<float>(ANTIQ_ROWMAX_OFFSET);
            antiqAmaxBmm1 = resource.ubBuf.template GetBufferByByte<float>(ANTIQ_AMAX_OFFSET);
            antiqKScaleUb = resource.ubBuf.template GetBufferByByte<float>(ANTIQ_KSCALE_OFFSET);
            antiqVScaleUb = resource.ubBuf.template GetBufferByByte<float>(ANTIQ_VSCALE_OFFSET);
            antiqAmaxQUb = resource.ubBuf.template GetBufferByByte<float>(ANTIQ_AMAXQ_OFFSET);

            // Ping-pong staging slots for the int8 P store.  Slot stride
            // covers the max tile (MAX_UB_S_ELEM_NUM
            // halves = 16KB each).  Two slots let the per-tile MTE3_V full
            // drain degrade into a "wait before slot reuse" one full tile
            // earlier, so the V pipeline keeps flowing while MTE3 drains to
            // GM.  Placed past antiqTmpFloor (<=2KB at TMP2+0) and inside
            // the 64KB TMP2 region [98304, 163840).
            constexpr uint32_t ANTIQ_PP_BASE_OFFSET = ANTIQ_TMP2_OFFSET + 8 * UB_UINT8_VECTOR_SIZE;
            antiqResHalfPp = resource.ubBuf.template GetBufferByByte<half>(ANTIQ_PP_BASE_OFFSET);
        }
    }

    __aicore__ inline ~BlockEpilogue() {}

    __aicore__ inline void SetAntiquantParams(uint32_t msdIterNum_, uint32_t pMsdIterNum_, uint32_t groupSize_)
    {
        // GBSA supports MSD2 and MSD3 only.  Normalize malformed tiling
        // values here as a final guard before entering the MSD loops.
        msdIterNum = msdIterNum_ == 3U ? 3U : 2U;
        pMsdIterNum = pMsdIterNum_ == 3U ? 3U : 2U;
        antiqGroupSize = groupSize_;
    }

    __aicore__ inline void SetAntiquantPStoreFusion(bool enabled)
    {
        fuseTwoDigitStore = enabled;
    }

    __aicore__ inline void SetAntiquantScaleGm(AscendC::GlobalTensor<float> keyAntiqScaleGm_,
                                               AscendC::GlobalTensor<float> valueAntiqScaleGm_,
                                               AscendC::GlobalTensor<float> amaxQGm_,
                                               AscendC::GlobalTensor<float> amaxPGm_)
    {
        keyAntiqScaleGm = keyAntiqScaleGm_;
        valueAntiqScaleGm = valueAntiqScaleGm_;
        amaxQGm = amaxQGm_;
        amaxPGm = amaxPGm_;
    }

    __aicore__ inline void SetAntiquantScaleOffset(uint64_t kvScaleOffset_)
    {
        kvScaleOffset = kvScaleOffset_;
        kvScaleBlockCount = 1;
    }

    __aicore__ inline void SetAntiquantScalePages(const int32_t *physicalBlockIds, uint32_t blockCount,
                                                  uint32_t blockSize)
    {
        kvScaleBlockCount = blockCount > MAX_SCALE_BLOCKS ? MAX_SCALE_BLOCKS : blockCount;
        kvScaleBlockSize = blockSize;
        for (uint32_t i = 0; i < kvScaleBlockCount; ++i) {
            kvScalePageOffsets[i] = static_cast<uint64_t>(physicalBlockIds[i]) * blockSize;
        }
    }

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return (a > b) ? b : a;
    }

    __aicore__ inline void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE || len == 0) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = static_cast<uint64_t>(len) % static_cast<uint64_t>(FLOAT_VECTOR_SIZE);
        for (uint64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ inline void RowsumSPECTILE1024(const AscendC::LocalTensor<float> &srcUb,
                                              const AscendC::LocalTensor<float> &rowsumUb,
                                              const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound,
                                              uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
                                              AscendC::MASK_PLACEHOLDER, // (uint64_t)0
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                              numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
                                              AscendC::MASK_PLACEHOLDER, // (uint64_t)0
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        SetVecMask(ROW_OPS_SPEC_MASK_16);
        AscendC::WholeReduceSum<float, false>(rowsumUb, tvUbTensor[REDUCE_UB_SIZE],
                                              AscendC::MASK_PLACEHOLDER, // (uint64_t)0
                                              numRowsRound, 1, 1, 2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void RowsumSPECTILE512(const AscendC::LocalTensor<float> &srcUb,
                                             const AscendC::LocalTensor<float> &rowsumUb,
                                             const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound,
                                             uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                              numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(rowsumUb, tvUbTensor[REDUCE_UB_SIZE],
                                              numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowsumSPECTILE256(const AscendC::LocalTensor<float> &srcUb,
                                             const AscendC::LocalTensor<float> &rowsumUb,
                                             const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound,
                                             uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor, numRowsRound, 0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceSum<float, false>(rowsumUb, tvUbTensor[REDUCE_UB_SIZE],
                                              CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void RowsumTAILTILE(const AscendC::LocalTensor<float> &srcUb,
                                          const AscendC::LocalTensor<float> &rowsumUb,
                                          const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound,
                                          uint32_t numElems, uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRowsRound, 0, 1, 1,
                                                  numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb, tvUbTensor, CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowSumIdx = 1; rowSumIdx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb[rowSumIdx * FLOAT_VECTOR_SIZE], numRowsRound, 0,
                                                      1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                                      CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1,
                                                      1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(rowsumUb, rowsumUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                  numRowsRound, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb, tvUbTensor, CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                                      CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1,
                                                      1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(rowsumUb, rowsumUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    __aicore__ inline void RowmaxSPECTILE1024(const AscendC::LocalTensor<float> &srcUb,
                                              const AscendC::LocalTensor<float> &rowmaxUb,
                                              const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound,
                                              uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
                                              AscendC::MASK_PLACEHOLDER, // (uint64_t)0
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                              numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
                                              AscendC::MASK_PLACEHOLDER, // (uint64_t)0
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        SetVecMask(ROW_OPS_SPEC_MASK_16);
        AscendC::WholeReduceMax<float, false>(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE],
                                              AscendC::MASK_PLACEHOLDER, // (uint64_t)0
                                              numRowsRound, 1, 1, 2, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void RowmaxSPECTILE512(const AscendC::LocalTensor<float> &srcUb,
                                             const AscendC::LocalTensor<float> &rowmaxUb,
                                             const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound,
                                             uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                              numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE],
                                              numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowmaxSPECTILE256(const AscendC::LocalTensor<float> &srcUb,
                                             const AscendC::LocalTensor<float> &rowmaxUb,
                                             const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound,
                                             uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE, 0,
                                              1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor, numRowsRound, 0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceMax<float, false>(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE],
                                              CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void RowmaxTAILTILE(const AscendC::LocalTensor<float> &srcUb,
                                          const AscendC::LocalTensor<float> &rowmaxUb,
                                          const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound,
                                          uint32_t numElems, uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRowsRound, 0, 1, 1,
                                                  numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb, tvUbTensor, CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb[rowmax_idx * FLOAT_VECTOR_SIZE], numRowsRound,
                                                      0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                                      CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1,
                                                      1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb, rowmaxUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                  numRowsRound, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceMax<float, false>(
                    rowmaxUb, tvUbTensor, CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                                      CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1,
                                                      1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb, rowmaxUb, tvUbTensor[REDUCE_UB_SIZE], (uint64_t)0, 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    __aicore__ inline void CopySGmToUb(AscendC::GlobalTensor<ElementInput> gInput, uint32_t sUbOffset,
                                       uint32_t rowNumCurLoop, uint32_t columnNum, uint32_t columnNumRound,
                                       uint32_t columnNumPad)
    {
        if constexpr (ANTIQUANT) {
            // 伪量化：合并 MSD 分段 int32 结果为 float，最终 ÷127
            // 对应 AntiquantMatmulResCombine: (C_0 + C_1/254 + ...) / 127
            uint32_t copySize = rowNumCurLoop * columnNumRound;
            // Event 0/1 are the outer ping-pong hand-off and are also used by
            // rescale-O.  Event 3 is used by the PV/RescaleO pipeline.  Keep
            // the private int32 staging transfer on the unused event 5.
            constexpr uint32_t eventId = EVENT_ID5;
            float scale = 1.0f;
            uint32_t step = antiqGroupSize * columnNumPad;
            const bool fullTile = columnNum == columnNumRound && columnNumPad == columnNumRound;
            for (uint32_t i = 0; i < msdIterNum; i++) {
                if (fullTile) {
                    // Full KV blocks are already contiguous and 32-byte
                    // aligned.  A single 2-D transfer avoids one GM request
                    // per row of every MSD segment.
                    AscendC::DataCopy(antiqTmpInt32, gInput[step * i],
                                      AscendC::DataCopyParams(rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE, 0, 0));
                } else {
                    // Partial tiles need a zeroed destination for their
                    // padded lanes before the row-wise transfers.
                    AscendC::Duplicate<int32_t>(antiqTmpInt32, 0, copySize);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
                    for (uint32_t row = 0; row < rowNumCurLoop; ++row) {
                        AscendC::DataCopyPad(antiqTmpInt32[row * columnNumRound], gInput[step * i + row * columnNumPad],
                                             AscendC::DataCopyExtParams(1, columnNum * sizeof(ElementInput), 0, 0, 0),
                                             AscendC::DataCopyPadExtParams<ElementInput>(false, 0, 0, 0));
                    }
                }
                // GM -> UB is asynchronous; wait for the data before the
                // vector cast.  Without this dependency the first MSD tile
                // can consume stale UB contents.
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                if (i == 0) {
                    // 第 0 段直接 Cast int32 → float
                    AscendC::Cast(lsUbTensor[sUbOffset], antiqTmpInt32, AscendC::RoundMode::CAST_NONE, copySize);
                } else {
                    // 第 i 段 Cast 后乘递减 scale 权重，累加
                    AscendC::Cast(antiqTmpFloat, antiqTmpInt32, AscendC::RoundMode::CAST_NONE, copySize);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(antiqTmpFloat, antiqTmpFloat, scale, copySize);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], antiqTmpFloat, copySize);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (i + 1 < msdIterNum) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
                }
                scale = scale / antiquantExpandCoeff;
            }
            // The common 1/127 factor is folded into the row scale below.
            // This avoids one full-tile vector multiply for every QK tile.
            // CopySGmToUb is called repeatedly with the same int32 staging
            // buffer.  Close the final V -> MTE2 dependency as well as the
            // between-segment dependencies above, otherwise the next call can
            // overwrite the last MSD segment while Cast is still reading it.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        } else {
            AscendC::DataCopy(lsUbTensor[sUbOffset], gInput,
                              AscendC::DataCopyParams(rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE,
                                                      (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE, 0));
        }
    }

    __aicore__ inline void ZeroColumnPadding(uint32_t sUbOffset, uint32_t rowNum, uint32_t columnNum,
                                             uint32_t columnNumRound)
    {
        if (columnNum == columnNumRound) {
            return;
        }
        for (uint32_t vectorOffset = 0; vectorOffset < columnNumRound; vectorOffset += FLOAT_VECTOR_SIZE) {
            uint32_t paddingBegin = columnNum > vectorOffset ? columnNum - vectorOffset : 0;
            uint32_t vectorEnd = vectorOffset + FLOAT_VECTOR_SIZE;
            uint32_t paddingEnd = columnNumRound < vectorEnd ? columnNumRound - vectorOffset : FLOAT_VECTOR_SIZE;
            if (paddingBegin >= paddingEnd) {
                continue;
            }
            uint64_t mask = 0;
            for (uint32_t idx = paddingBegin; idx < paddingEnd; ++idx) {
                mask |= static_cast<uint64_t>(1) << idx;
            }
            AscendC::SetVectorMask<int8_t>(0, mask);
            for (uint32_t row = 0; row < rowNum; ++row) {
                uint32_t offset = sUbOffset + row * columnNumRound + vectorOffset;
                AscendC::Muls<float, false>(lsUbTensor[offset], lsUbTensor[offset], 0.0f, (uint64_t)0, 1,
                                            AscendC::UnaryRepeatParams(1, 1, 8, 8));
            }
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyMaskGmToUb(AscendC::GlobalTensor<ElementMask> gMask, uint32_t columnNum,
                                          uint32_t columnNumRound, uint32_t maskStride, uint32_t tokenNumPerHead,
                                          uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t integralHeadNum,
                                          uint32_t epiTokenNum)
    {
        uint32_t innerUbRowOffset = 0;
        if (proTokenNum != 0) {
            AscendC::DataCopyPad(maskUbTensor[innerUbRowOffset], gMask[proTokenIdx * maskStride],
                                 AscendC::DataCopyExtParams(proTokenNum, columnNum * sizeof(ElementMask),
                                                            (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                                 AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += proTokenNum * columnNumRound;
        }
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            AscendC::DataCopyPad(maskUbTensor[innerUbRowOffset], gMask,
                                 AscendC::DataCopyExtParams(tokenNumPerHead, columnNum * sizeof(ElementMask),
                                                            (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                                 AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        if (epiTokenNum != 0) {
            AscendC::DataCopyPad(maskUbTensor[innerUbRowOffset], gMask,
                                 AscendC::DataCopyExtParams(epiTokenNum, columnNum * sizeof(ElementMask),
                                                            (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                                 AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
        }
    }

    __aicore__ inline void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], scaleValue, (uint64_t)0,
                                    CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename ElementMaskDst, typename ElementMaskSrc>
    __aicore__ inline void UpCastMask(const AscendC::LocalTensor<ElementMaskDst> &maskUbTensorDst,
                                      const AscendC::LocalTensor<ElementMaskSrc> &maskUbTensorSrc,
                                      uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Cast<ElementMaskDst, ElementMaskSrc, false>(
            maskUbTensorDst, maskUbTensorSrc, AscendC::RoundMode::CAST_NONE, (uint64_t)0,
            CeilDiv(rowNumCurLoop * columnNumRound, (uint32_t)(REPEAT_SIZE_IN_BYTE / sizeof(ElementMaskDst))),
            AscendC::UnaryRepeatParams(1, 1, 8, 4));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound,
                                     uint32_t maskColumnRound, uint32_t addMaskUbOffset)
    {
        AscendC::Muls<float, false>(maskUbTensor32, maskUbTensor32, (float)-3e38, (uint64_t)0,
                                    CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
                                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (maskColumnRound == columnNumRound) {
            AscendC::Add<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], maskUbTensor32, (uint64_t)0,
                                       CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        } else {
            uint32_t loop = maskColumnRound / FLOAT_VECTOR_SIZE;
            for (uint32_t i = 0; i < loop; i++) {
                AscendC::Add<float, false>(
                    lsUbTensor[sUbOffset][addMaskUbOffset + i * FLOAT_VECTOR_SIZE],
                    lsUbTensor[sUbOffset][addMaskUbOffset + i * FLOAT_VECTOR_SIZE],
                    maskUbTensor32[i * FLOAT_VECTOR_SIZE], (uint64_t)0, rowNumCurLoop,
                    AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE,
                                                columnNumRound / FLOAT_BLOCK_SIZE, maskColumnRound / FLOAT_BLOCK_SIZE));
            }
            if (maskColumnRound % FLOAT_VECTOR_SIZE > 0) {
                SetVecMask(maskColumnRound % FLOAT_VECTOR_SIZE);
                AscendC::Add<float, false>(
                    lsUbTensor[sUbOffset][addMaskUbOffset + loop * FLOAT_VECTOR_SIZE],
                    lsUbTensor[sUbOffset][addMaskUbOffset + loop * FLOAT_VECTOR_SIZE],
                    maskUbTensor32[loop * FLOAT_VECTOR_SIZE], (uint64_t)0, rowNumCurLoop,
                    AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE,
                                                columnNumRound / FLOAT_BLOCK_SIZE, maskColumnRound / FLOAT_BLOCK_SIZE));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum,
                                           uint32_t columnNumRound, uint32_t rowOffset)
    {
        if (columnNum == 1024) {
            RowmaxSPECTILE1024(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                               columnNumRound);
        } else if (columnNum == 512) {
            RowmaxSPECTILE512(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                              columnNumRound);
        } else if (columnNum == 256) {
            RowmaxSPECTILE256(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                              columnNumRound);
        } else {
            RowmaxTAILTILE(lsUbTensor[sUbOffset], lmUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                           columnNumRound);
        }
    }

    __aicore__ inline void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
                                              uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset,
                                              uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            AscendC::DataCopy(hmUbTensor[rowOffset], lmUbTensor[rowOffset],
                              AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** hm = vmax(lm, gm)
            AscendC::Max<float, false>(hmUbTensor[rowOffset], lmUbTensor[rowOffset], gmUbTensor[rowOffset], (uint64_t)0,
                                       1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = gm - hm
            AscendC::Sub<float, false>(dmUbTensor[dmUbOffsetCurCycle], gmUbTensor[rowOffset], hmUbTensor[rowOffset],
                                       (uint64_t)0, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)
            AscendC::Exp<float, false>(dmUbTensor[dmUbOffsetCurCycle], dmUbTensor[dmUbOffsetCurCycle], (uint64_t)0, 1,
                                       AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm
        AscendC::DataCopy(gmUbTensor[rowOffset], hmUbTensor[rowOffset],
                          AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
                                   uint32_t columnNum, uint32_t columnNumRound, uint32_t rowOffset)
    {
        // *** hm_block = expand_to_block(hm), 存放于 tv
        AscendC::Brcb(tvUbTensor.template ReinterpretCast<uint32_t>(),
                      hmUbTensor[rowOffset].template ReinterpretCast<uint32_t>(), rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
                      AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = ls - hm_block
        for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0,
                                       rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                       tvUbTensor, (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = exp(ls)
        AscendC::Exp<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], (uint64_t)0,
                                   CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                                   AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum,
                                           uint32_t columnNumRound, uint32_t rowOffset)
    {
        // *** ll = rowsum(ls32)
        if (columnNum == 1024) {
            RowsumSPECTILE1024(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                               columnNumRound);
        } else if (columnNum == 512) {
            RowsumSPECTILE512(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                              columnNumRound);
        } else if (columnNum == 256) {
            RowsumSPECTILE256(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                              columnNumRound);
        } else {
            RowsumTAILTILE(lsUbTensor[sUbOffset], llUbTensor[rowOffset], tvUbTensor, rowNumCurLoopRound, columnNum,
                           columnNumRound);
        }
    }

    __aicore__ inline void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
                                              uint32_t dmUbOffsetCurCycle, uint32_t rowOffset,
                                              uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // *** gl = ll
            AscendC::DataCopy(glUbTensor[rowOffset], llUbTensor[rowOffset],
                              AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** gl = dm * gl
            AscendC::Mul<float, false>(glUbTensor[rowOffset], dmUbTensor[dmUbOffsetCurCycle], glUbTensor[rowOffset],
                                       (uint64_t)0, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** gl = ll + gl
            AscendC::Add<float, false>(glUbTensor[rowOffset], glUbTensor[rowOffset], llUbTensor[rowOffset], (uint64_t)0,
                                       1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    __aicore__ inline void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    __aicore__ inline void DownCastPAntiquant(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
                                              uint32_t columnNum, uint32_t columnNumRound, uint32_t rowOffset)
    {
        // 1. Compute per-row Amax of P (after V_scale, all non-negative → no Abs needed)
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        // 2. Brcb Amax_P_v to antiqAmaxBmm1 (8 floats per row for broadcast)
        AscendC::Brcb(antiqAmaxBmm1.ReinterpretCast<uint32_t>(), lmUbTensor[rowOffset].ReinterpretCast<uint32_t>(),
                      rowNumCurLoopRound / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // 3. Compute one 127/Amax scale per row, then multiply P.  This keeps
        // the original Amax values for BMM2 reconstruction while avoiding an
        // elementwise divide and a second full-tile multiply.
        AscendC::Duplicate(antiqTmpFloor, antiqCoeff1, rowNumCurLoop * FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(antiqTmpFloor, antiqTmpFloor, antiqAmaxBmm1, rowNumCurLoop * FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t vmul_idx = 0; vmul_idx < columnNum / FLOAT_VECTOR_SIZE; ++vmul_idx) {
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + vmul_idx * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset + vmul_idx * FLOAT_VECTOR_SIZE], antiqTmpFloor, (uint64_t)0,
                                       rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + (columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset + (columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       antiqTmpFloor, (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 4. The scaled P is expanded by StoreAntiquantP.
        // 5. Cast float → half → int8
    }

    // 伪量化：int8 P（MSD 分段）写回 GM
    __aicore__ inline void StoreAntiquantP(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset,
                                           uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        uint32_t calcSize = rowNumCurLoop * columnNumRound;
        uint32_t segmentStride = antiqGroupSize * columnNumRound;
        if (pMsdIterNum == 1U) {
            // A single P digit has no residual to preserve. Cast directly to
            // the half staging tensor and drain the GM write before reuse.
            AscendC::Cast(antiqResHalf, lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_ROUND, calcSize);
            AscendC::PipeBarrier<PIPE_V>();
            antiqResInt8 = antiqResHalf.template ReinterpretCast<int8_t>();
            antiqResInt8.SetSize(antiqResHalf.GetSize());
            AscendC::Cast(antiqResInt8, antiqResHalf, AscendC::RoundMode::CAST_ROUND, calcSize);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::DataCopy(gOutput, antiqResInt8, calcSize);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            return;
        }
        // Ping-pong the int8 staging across two 16KB slots (antiqResHalfPp,
        // pre-allocated in the constructor).  Digit i converts into slot
        // (i & 1); only a slot REUSE (i >= 2) must first wait for the MTE3
        // store of digit i-2.  With pMsdIterNum == 2 no reuse ever happens,
        // so the per-digit MTE3_V full drain collapses into a tail drain of
        // the last two pending stores and the V pipeline keeps converting
        // digit 1 while MTE3 is still writing digit 0 to GM.
        constexpr uint32_t PP_SLOT_STRIDE = MAX_UB_S_ELEM_NUM; // half elems per slot
        // For the common two-digit target tile, both int8 digits are compact
        // and contiguous at the same stride in UB and GM. Convert both
        // digits first, then issue one 2-D GM copy instead of two independent
        // MTE3 transfers. Keep the general path below for larger or
        // non-block-aligned tiles.
        const uint32_t slotBytes = PP_SLOT_STRIDE * sizeof(half);
        const bool canFuseTwoDigitStore = fuseTwoDigitStore && pMsdIterNum == 2U && rowNumCurLoop <= antiqGroupSize &&
                                          calcSize % 32U == 0U && slotBytes >= calcSize && segmentStride >= calcSize &&
                                          (slotBytes - calcSize) % 32U == 0U && (segmentStride - calcSize) % 32U == 0U;
        if (canFuseTwoDigitStore) {
            for (uint32_t i = 0; i < 2U; ++i) {
                if (i > 0U) {
                    AscendC::Sub(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], antiqTmpFloat, calcSize);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], antiquantExpandCoeff, calcSize);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Cast(antiqTmpFloat, lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_ROUND, calcSize);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::LocalTensor<half> halfSlot = antiqResHalfPp[i * PP_SLOT_STRIDE];
                AscendC::Cast(halfSlot, antiqTmpFloat, AscendC::RoundMode::CAST_ROUND, calcSize);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::LocalTensor<int8_t> int8Slot = halfSlot.template ReinterpretCast<int8_t>();
                int8Slot.SetSize(halfSlot.GetSize());
                AscendC::Cast(int8Slot, halfSlot, AscendC::RoundMode::CAST_ROUND, calcSize);
                AscendC::PipeBarrier<PIPE_V>();
            }
            auto int8Base = antiqResHalfPp[0].template ReinterpretCast<int8_t>();
            int8Base.SetSize(antiqResHalfPp[0].GetSize());
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::DataCopy(gOutput, int8Base,
                              AscendC::DataCopyParams(2U, calcSize / 32U, (slotBytes - calcSize) / 32U,
                                                      (segmentStride - calcSize) / 32U));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            return;
        }
        for (uint32_t i = 0; i < pMsdIterNum; ++i) {
            // Keep the hard event paired with the staging slot.  Reusing one
            // event id for both slots turns the two outstanding MTE3 writes
            // into a single serialized queue and defeats the ping-pong UB
            // buffer.  EVENT_ID6/7 are also the established query-expansion
            // pair; query preprocessing drains them before this epilogue.
            const uint32_t eventId = (i & 1U) == 0U ? EVENT_ID6 : EVENT_ID7;
            if (i >= 2) {
                // Slot (i & 1) is about to be overwritten: digit i-2 used the
                // same slot and its GM store must have retired first.
                const uint32_t releaseEventId = ((i - 2U) & 1U) == 0U ? EVENT_ID6 : EVENT_ID7;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(releaseEventId);
            }
            if (i > 0) {
                AscendC::Sub(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], antiqTmpFloat, calcSize);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], antiquantExpandCoeff, calcSize);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // Retain the rounded fp32 value for the next residual digit.
            AscendC::Cast(antiqTmpFloat, lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_ROUND, calcSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> halfSlot = antiqResHalfPp[(i & 1U) * PP_SLOT_STRIDE];
            AscendC::Cast(halfSlot, antiqTmpFloat, AscendC::RoundMode::CAST_ROUND, calcSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<int8_t> int8Slot = halfSlot.template ReinterpretCast<int8_t>();
            int8Slot.SetSize(halfSlot.GetSize());
            AscendC::Cast(int8Slot, halfSlot, AscendC::RoundMode::CAST_ROUND, calcSize);
            // The int8 P digit is produced on PIPE_V and consumed by the
            // UB-to-GM copy on PIPE_MTE3 (same hand-off as before).  The
            // MTE3_V completion flag is only SET here; it is consumed either
            // by digit i+2 (slot reuse) or by the tail drain below.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::DataCopy(gOutput[static_cast<uint64_t>(i) * segmentStride], int8Slot, calcSize);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
        }
        // Drain the last min(pMsdIterNum, 2) pending GM stores so the caller
        // (and the next tile) observes a fully written P segment and the
        // event counter stays balanced.
        for (uint32_t k = 0; k < AscendC::Std::min(pMsdIterNum, 2U); ++k) {
            const uint32_t eventId = (k & 1U) == 0U ? EVENT_ID6 : EVENT_ID7;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        }
    }

    // 伪量化：加载 K/V per-token scale 到 UB（MTE2 操作，在 row loop 前调用一次）
    __aicore__ inline void LoadAntiquantScales(uint32_t columnNum, uint32_t columnNumRound)
    {
        if constexpr (!ANTIQUANT) {
            return;
        }
        if (kvScaleBlockCount > 1) {
            return;
        }
        // K scale: per-token, columnNum 个 float
        AscendC::DataCopyExtParams copyInParams;
        AscendC::DataCopyPadExtParams<float> copyInPadParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = columnNum * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (columnNumRound - columnNum) % FLOAT_BLOCK_SIZE;
        copyInPadParams.paddingValue = 0;
        AscendC::DataCopyPad(antiqKScaleUb, keyAntiqScaleGm[kvScaleOffset], copyInParams, copyInPadParams);
        AscendC::DataCopyPad(antiqVScaleUb, valueAntiqScaleGm[kvScaleOffset], copyInParams, copyInPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
    }

    __aicore__ inline void LoadAntiquantScalePages(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src)
    {
        const uint32_t pageCopyBlocks = kvScaleBlockSize / FLOAT_BLOCK_SIZE;
        bool contiguousPages = true;
        for (uint32_t blockIdx = 1U; blockIdx < kvScaleBlockCount; ++blockIdx) {
            contiguousPages =
                contiguousPages && kvScalePageOffsets[blockIdx] ==
                                       kvScalePageOffsets[0] + static_cast<uint64_t>(blockIdx) * kvScaleBlockSize;
        }
        if (contiguousPages) {
            // Keep the page dimension in the descriptor so MTE2 submits one
            // command for the whole scale tile.
            AscendC::DataCopy(dst, src[kvScalePageOffsets[0]],
                              AscendC::DataCopyParams(kvScaleBlockCount, pageCopyBlocks, 0, 0));
        } else {
            const auto copyParams = AscendC::DataCopyParams(1, pageCopyBlocks, 0, 0);
            for (uint32_t blockIdx = 0U; blockIdx < kvScaleBlockCount; ++blockIdx) {
                AscendC::DataCopy(dst[blockIdx * kvScaleBlockSize], src[kvScalePageOffsets[blockIdx]], copyParams);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
    }

    __aicore__ inline void ApplyScaleRange(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound,
                                           uint32_t columnOffset, uint32_t columnCount,
                                           AscendC::LocalTensor<float> scaleUb)
    {
        for (uint32_t vidx = 0; vidx < columnCount / FLOAT_VECTOR_SIZE; ++vidx) {
            const uint32_t offset = vidx * FLOAT_VECTOR_SIZE;
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + columnOffset + offset], scaleUb[offset],
                                       lsUbTensor[sUbOffset + columnOffset + offset], (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE, 0,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE));
        }
        if (columnCount % FLOAT_VECTOR_SIZE > 0) {
            const uint32_t tailOffset = columnCount / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE;
            SetMask(columnCount % FLOAT_VECTOR_SIZE);
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + columnOffset + tailOffset], scaleUb[tailOffset],
                                       lsUbTensor[sUbOffset + columnOffset + tailOffset], (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE, 0,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
    }

    // 伪量化：应用 Amax_Q (RowMuls) 和 K scale (VecMulMat) 到 BMM1 结果
    // 在 CopySGmToUb 之后、ScaleS 之前调用
    __aicore__ inline void ApplyAmaxQAndKScale(uint32_t sUbOffset, uint32_t rowOffsetIoGm, uint32_t rowNumCurLoop,
                                               uint32_t columnNum, uint32_t columnNumRound)
    {
        if constexpr (!ANTIQUANT) {
            return;
        }
        // Amax_Q is already in Brcb format in the dedicated UB region written
        // by QueryPreProcess on this AIV sub-block.
        auto antiqAmaxQCur = antiqAmaxQUb[rowOffsetIoGm * FLOAT_BLOCK_SIZE];
        // Fold the uniform attention scale into the small per-row Amax
        // vector, avoiding a separate full-tile ScaleS pass.
        AscendC::Muls(antiqTmpFloor, antiqAmaxQCur, scaleValue * antiqCoeff2, rowNumCurLoop * FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        // Mul: lsUb *= scaled Amax_Q.
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t vmul_idx = 0; vmul_idx < columnNum / FLOAT_VECTOR_SIZE; ++vmul_idx) {
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + vmul_idx * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset + vmul_idx * FLOAT_VECTOR_SIZE], antiqTmpFloor, (uint64_t)0,
                                       rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + (columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset + (columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       antiqTmpFloor, (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 2. K scale: VecMulMat — lsUb[i,j] *= kScale[j]
        // K scale 在 antiqKScaleUb 中，按列广播
        if (kvScaleBlockCount > 1) {
            LoadAntiquantScalePages(antiqKScaleUb, keyAntiqScaleGm);
            ApplyScaleRange(sUbOffset, rowNumCurLoop, columnNumRound, 0, kvScaleBlockCount * kvScaleBlockSize,
                            antiqKScaleUb);
            return;
        }
        for (uint32_t vidx = 0; vidx < columnNum / FLOAT_VECTOR_SIZE; ++vidx) {
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + vidx * FLOAT_VECTOR_SIZE],
                                       antiqKScaleUb[vidx * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset + vidx * FLOAT_VECTOR_SIZE], (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE, 0,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + (columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       antiqKScaleUb[(columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset + (columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE, 0,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 伪量化：应用 V scale (VecMulMat) 到 softmax 结果 P，在 DownCastPAntiquant 之前调用
    __aicore__ inline void ApplyVScaleToP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNum,
                                          uint32_t columnNumRound)
    {
        if constexpr (!ANTIQUANT) {
            return;
        }
        // V scale: VecMulMat — lsUb[i,j] *= vScale[j]
        // V scale 在 antiqVScaleUb 中，按列广播
        if (kvScaleBlockCount > 1) {
            LoadAntiquantScalePages(antiqVScaleUb, valueAntiqScaleGm);
            ApplyScaleRange(sUbOffset, rowNumCurLoop, columnNumRound, 0, kvScaleBlockCount * kvScaleBlockSize,
                            antiqVScaleUb);
            return;
        }
        for (uint32_t vidx = 0; vidx < columnNum / FLOAT_VECTOR_SIZE; ++vidx) {
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + vidx * FLOAT_VECTOR_SIZE],
                                       antiqVScaleUb[vidx * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset + vidx * FLOAT_VECTOR_SIZE], (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE, 0,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Mul<float, false>(lsUbTensor[sUbOffset + (columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       antiqVScaleUb[(columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       lsUbTensor[sUbOffset + (columnNum / FLOAT_VECTOR_SIZE) * FLOAT_VECTOR_SIZE],
                                       (uint64_t)0, rowNumCurLoop,
                                       AscendC::BinaryRepeatParams(1, 1, 1, columnNumRound / FLOAT_BLOCK_SIZE, 0,
                                                                   columnNumRound / FLOAT_BLOCK_SIZE));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset,
                                       uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(gOutput, lpUbTensor[sUbOffset],
                          AscendC::DataCopyParams(rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0,
                                                  (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    template <bool doTriUMask>
    __aicore__ inline void SubCoreCompute(AscendC::GlobalTensor<ElementOutput> gOutput,
                                          const LayoutOutput &layoutOutput, uint32_t rowOffset,
                                          uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
                                          uint32_t isFirstRowLoop, uint32_t isLastRowLoop, uint32_t columnNumRound,
                                          uint32_t pingpongFlag, uint32_t curStackTileMod,
                                          Arch::CrossCoreFlag softmaxFlag)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseMode::OUT_ONLY) {
            // wait for lse from ub to gm
            // In lse out-only mode, tv is used in the last stack tile to transport lse
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        UpdateGlobalRowMax(rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, dmUbOffsetCurCycle, rowOffset,
                           isFirstStackTile);

        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        if constexpr (!doTriUMask) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
        }

        if constexpr (ANTIQUANT) {
            curStackTileMod_ = curStackTileMod;
            // 伪量化：先计算 rowsum（在 V_scale 之前，基于原始 softmax 输出）
            CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
            // Int8 Cube consumes K in 32-element units.  Keep the padded P
            // lanes neutral before the aligned BMM2 reads them.
            ZeroColumnPadding(sUbOffset, rowNumCurLoop, columnNum, columnNumRound);
            // 应用 V scale (VecMulMat) 到 P
            ApplyVScaleToP(sUbOffset, rowNumCurLoop, columnNum, columnNumRound);
            // 计算 Amax_P_v，使用 127/Amax 量化，结果存入 antiqResInt8 + antiqAmaxBmm1
            DownCastPAntiquant(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        } else {
            DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);

        if constexpr (!ANTIQUANT) {
            CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        if constexpr (ANTIQUANT) {
            StoreAntiquantP(gOutput, sUbOffset, rowNumCurLoop, columnNumRound);
            // Store Amax_P_v to GM in Brcb format (8 floats per row).
            uint64_t amaxPOffset = static_cast<uint64_t>(curStackTileMod) * antiqGroupSize * FLOAT_BLOCK_SIZE +
                                   static_cast<uint64_t>(curRowOffsetIoGm_) * FLOAT_BLOCK_SIZE;
            AscendC::DataCopy(amaxPGm[amaxPOffset], antiqAmaxBmm1, AscendC::DataCopyParams(1, rowNumCurLoop, 0, 0));
        } else {
            CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        }
        if constexpr (!doTriUMask) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (isLastRowLoop) {
            NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxFlag);
        }
        UpdateGlobalRowSum(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset,
                           isFirstStackTile);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementOutput> gOutput,
                                      AscendC::GlobalTensor<ElementInput> gInput, const LayoutOutput &layoutOutput,
                                      const LayoutInput &layoutInput, GemmCoord actualBlockShape,
                                      uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile, uint32_t qSBlockSize,
                                      uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag softmaxFlag)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = ANTIQUANT ? RoundUp(columnNum, BLOCK_SIZE_IN_BYTE) : RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ? 0 :
                                  (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) :
                                                       qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ? (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;
        if (rowActualThisSubBlock == 0) {
            // When qNBlockSize==1, AIV1 owns the sole query row and AIV0 has
            // no consumer work.  Do not wait for the single QK token on the
            // idle antiquant sub-block: the token is consumed by the active
            // sub-block. Waiting here would leave the AIV/CUBE handshake
            // unbalanced for MHA (groupSize=1).
            // The two CUBE sub-blocks still consume one completion token each,
            // including when only one AIV sub-block owns the row.
            NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxFlag);
            return;
        }

        // 伪量化：加载 K/V per-token scale 到 UB（每 KV block 调用一次）
        LoadAntiquantScales(columnNum, columnNumRound);

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNum, columnNumRound,
                            columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                // 伪量化：应用 Amax_Q (RowMuls) + K scale (VecMulMat) 到 BMM1 结果
                ApplyAmaxQAndKScale((pingpongFlag * MAX_UB_S_ELEM_NUM), rowOffsetIoGm, rowNumCurLoop, columnNum,
                                    columnNumRound);
                if constexpr (!ANTIQUANT) {
                    ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                }
                curRowOffsetIoGm_ = rowOffsetIoGm;
                SubCoreCompute<false>(gOutputCurLoop, layoutOutputCurLoop, rowOffsetCurLoop, isFirstStackTile,
                                      isLastNoMaskStackTile, (delayedRowLoopIdx == 0),
                                      (delayedRowLoopIdx == rowLoopNum - 1), columnNumRound, pingpongFlag,
                                      curStackTileMod, softmaxFlag);
            }
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementOutput> gOutput,
                                      AscendC::GlobalTensor<ElementInput> gInput,
                                      AscendC::GlobalTensor<ElementMask> gMask, const LayoutOutput &layoutOutput,
                                      const LayoutInput &layoutInput, const LayoutInput &layoutMask,
                                      GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
                                      uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady,
                                      uint32_t triUp, uint32_t triDown, uint32_t kvSStartIdx, uint32_t kvSEndIdx)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ? 0 :
                                  (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) :
                                                       qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ? (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ? rowOffsetThisSubBlock : 0;

        // calc mask shift in gm
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRoundDown = RoundDown(triUp, BLOCK_SIZE_IN_BYTE);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0;
            maskColumn = kvSEndIdx - triUpRoundDown;
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;
        } else {
            gmOffsetMaskRow = 0;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = columnNum;
            addMaskUbOffset = 0;
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, BLOCK_SIZE_IN_BYTE);

        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(gMaskThisSubBlock, maskColumn, maskColumnRound, maskStride,
                                   tokenNumPerHeadThisSubBlock, proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNum, columnNumRound,
                            columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
                UpCastMask<float, half>(maskUbTensor32, maskUbTensor16, rowNumCurLoop, columnNumRound);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                ApplyMask((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, maskColumnRound,
                          addMaskUbOffset);
                // next loop mask load
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(gMaskThisSubBlock, maskColumn, maskColumnRound, maskStride,
                                   tokenNumPerHeadThisSubBlock, proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                // online softmax vectorized compute
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(gOutputCurLoop, layoutOutputCurLoop, rowOffsetCurLoop, isFirstStackTile, 0,
                                     (delayedRowLoopIdx == 0), (delayedRowLoopIdx == rowLoopNum - 1), columnNumRound,
                                     pingpongFlag, curStackTileMod);
            }
        }
    }

private:
    float scaleValue;
    AscendC::LocalTensor<float> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<ElementMask> maskUbTensor;
    AscendC::LocalTensor<half> maskUbTensor16;
    AscendC::LocalTensor<float> maskUbTensor32;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> glUbTensor;

    // Antiquant buffers (only used when ANTIQUANT=true)
    AscendC::LocalTensor<int32_t> antiqTmpInt32;
    AscendC::LocalTensor<float> antiqTmpFloat;
    AscendC::LocalTensor<float> antiqTmpFloor;
    AscendC::LocalTensor<half> antiqResHalf;
    AscendC::LocalTensor<int8_t> antiqResInt8;
    AscendC::LocalTensor<float> antiqTmpRowMax;
    AscendC::LocalTensor<float> antiqAmaxBmm1;
    AscendC::LocalTensor<float> antiqKScaleUb; // K per-token scale
    AscendC::LocalTensor<float> antiqVScaleUb; // V per-token scale
    AscendC::LocalTensor<float> antiqAmaxQUb;  // Amax_Q per-row (written by QueryPreProcess)
    AscendC::LocalTensor<half> antiqResHalfPp; // ping-pong int8 P staging (2 slots)
    uint32_t antiqStoreParity = 0;             // current ping-pong slot (per sub-block)
    uint32_t antiqStoreUseCount = 0;           // >=2 => slot reuse needs MTE3_V wait
    uint32_t msdIterNum = 2;
    uint32_t pMsdIterNum = 2;
    uint32_t antiqGroupSize = 1;
    bool fuseTwoDigitStore = false;

    // Antiquant GM tensors (set via SetAntiquantScaleGm)
    AscendC::GlobalTensor<float> keyAntiqScaleGm;
    AscendC::GlobalTensor<float> valueAntiqScaleGm;
    AscendC::GlobalTensor<float> amaxQGm;
    AscendC::GlobalTensor<float> amaxPGm;
    static constexpr uint32_t MAX_SCALE_BLOCKS = 16;
    uint64_t kvScaleOffset = 0; // offset in scale GM for current KV block
    uint64_t kvScalePageOffsets[MAX_SCALE_BLOCKS];
    uint32_t kvScaleBlockCount = 1;
    uint32_t kvScaleBlockSize = 0;
    uint32_t curRowOffsetIoGm_ = 0; // absolute row offset for GM indexing
    uint32_t curStackTileMod_ = 0;  // stage slot for amaxP GM indexing
};
} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_HPP
