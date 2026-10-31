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
 * \file sta_epilogue.h
 * \brief Vector-core epilogues of SlidingTileAttention for Atlas A2: online softmax and output rescaling.
 *
 * Copied from the BlockSparseAttention A2 fp32 epilogues (attn_infra/epilogue/block/
 * bsa_block_epilogue_online_softmax.hpp and bsa_block_epilogue_rescale_o.hpp) without the attention-mask path.
 */

#ifndef STA_EPILOGUE_H
#define STA_EPILOGUE_H

#include "../../block_sparse_attention/op_kernel/attn_infra/epilogue/block/bsa_block_epilogue.hpp"

namespace NpuArch::Epilogue {

struct StaOnlineSoftmax : public EpilogueAtlasA2OnlineSoftmax<LseMode::NONE, float> {};
struct StaRescaleO : public EpilogueAtlasA2RescaleO<LseMode::NONE, float> {};

// UB layout shared by the two epilogues (byte offsets; the UB holds 192 KiB):
//   [0, 96K)      LS  score buffers of two row loops (the next loop is loaded while the current one computes); an
//                     8192-score loop (padded heads) casts its probabilities into the last 16 KiB of its buffer
//   [96K, 120K)   LP  probabilities of a 12288-score loop (unpadded heads), so score loads never wait for stores
//   [120K, 144K)  LO  P*V of the current KV stack (192 rows x 32 floats)
//   [144K, 168K)  GO  accumulated output
//   [168K, 176K)  TV  temporaries
//   [176K, 188K)  per-row LM, HM, GM, LL (1 KiB each), GL and DM (4 KiB each: one row set per KV-stack slot)
//   [188K, 191K)  QW  scratch of the kernel's query read-ahead (one C0 per L2 line)
namespace StaUbLayout {
constexpr uint32_t KIB = 1024;
constexpr uint32_t LS = 0;
constexpr uint32_t LP = 96 * KIB;
constexpr uint32_t LO = 120 * KIB;
constexpr uint32_t GO = 144 * KIB;
constexpr uint32_t TV = 168 * KIB;
constexpr uint32_t LM = 176 * KIB;
constexpr uint32_t HM = 177 * KIB;
constexpr uint32_t GM = 178 * KIB;
constexpr uint32_t LL = 179 * KIB;
constexpr uint32_t GL = 180 * KIB;
// Most KV-stack slots of the kernel (largest stage-2 delay + 1): per-stack row values are kept per slot until stage 2
// has used them, because stage 1 of later stacks, possibly of the next task, runs before it.
constexpr uint32_t STACK_SLOT_MAX = 4;
constexpr uint32_t DM = GL + STACK_SLOT_MAX * KIB;
constexpr uint32_t QW = DM + STACK_SLOT_MAX * KIB;
constexpr uint32_t QW_C0_NUM = 96;
static_assert(QW + QW_C0_NUM * 32 <= 192 * KIB, "the UB holds 192 KiB");
// Float elements of LO and GO.
constexpr uint32_t O_ELEM_NUM = 6144;
} // namespace StaUbLayout

} // namespace NpuArch::Epilogue

namespace NpuArch::Epilogue::Block {

template <class OutputType_, class InputType_, class MaskType_>
class BlockEpilogue<StaOnlineSoftmax, OutputType_, InputType_, MaskType_> {
public:
    using DispatchPolicy = StaOnlineSoftmax;
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
    // Scores per row loop: 24 rows of a 512-key stack. The loop count, not the element count, dominates the
    // vector-side cost of unpadded heads, so their loops are as large as the UB layout allows. Padded heads skip
    // the scaling and row sums and measured faster with 16-row loops.
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 12288;
    static constexpr uint32_t PADDED_S_ELEM_NUM = 8192;

    // First-stage outputs of the row reductions (rows * 64 floats at most) end below this float index.
    static constexpr uint32_t REDUCE_UB_SIZE = 1536;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_16 = 16;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;
    // MTE3_V event (set once by the kernel before the first stack): the previous loop's probabilities in the
    // separate buffer LP have been stored, so the buffer may be overwritten.
    static constexpr uint32_t SEPARATE_P_STORED_EVENT = EVENT_ID0;

    // skipRowSum: the cube's P*V provides the row sums (see BlockEpilogue<StaRescaleO>), so the running normalizer
    // is not tracked here. slotNum: KV-stack slots in use.
    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, float scaleValue_, bool skipRowSum_,
                                    uint32_t slotNum_)
    {
        slotNum = slotNum_;
        skipRowSum = skipRowSum_;
        sElemNumPerLoop = skipRowSum_ ? PADDED_S_ELEM_NUM : MAX_UB_S_ELEM_NUM;
        separateP = sElemNumPerLoop == MAX_UB_S_ELEM_NUM;
        scaleValue = scaleValue_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::LS);
        lpUbTensor =
            resource.ubBuf.template GetBufferByByte<ElementOutput>(separateP ? StaUbLayout::LP : StaUbLayout::LS);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::LM);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::HM);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::GM);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::DM);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::LL);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::TV);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::GL);
    }

    __aicore__ inline ~BlockEpilogue() {}

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

    __aicore__ inline void RowsumSPECTILE1024(const AscendC::LocalTensor<float>& srcUb,
                                              const AscendC::LocalTensor<float>& rowsumUb,
                                              const AscendC::LocalTensor<float>& tvUbTensor, uint32_t numRowsRound,
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

    __aicore__ inline void RowsumSPECTILE512(const AscendC::LocalTensor<float>& srcUb,
                                             const AscendC::LocalTensor<float>& rowsumUb,
                                             const AscendC::LocalTensor<float>& tvUbTensor, uint32_t numRowsRound,
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

    __aicore__ inline void RowsumSPECTILE256(const AscendC::LocalTensor<float>& srcUb,
                                             const AscendC::LocalTensor<float>& rowsumUb,
                                             const AscendC::LocalTensor<float>& tvUbTensor, uint32_t numRowsRound,
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

    __aicore__ inline void RowsumTAILTILE(const AscendC::LocalTensor<float>& srcUb,
                                          const AscendC::LocalTensor<float>& rowsumUb,
                                          const AscendC::LocalTensor<float>& tvUbTensor, uint32_t numRowsRound,
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

    __aicore__ inline void RowmaxSPECTILE1024(const AscendC::LocalTensor<float>& srcUb,
                                              const AscendC::LocalTensor<float>& rowmaxUb,
                                              const AscendC::LocalTensor<float>& tvUbTensor, uint32_t numRowsRound,
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

    __aicore__ inline void RowmaxSPECTILE512(const AscendC::LocalTensor<float>& srcUb,
                                             const AscendC::LocalTensor<float>& rowmaxUb,
                                             const AscendC::LocalTensor<float>& tvUbTensor, uint32_t numRowsRound,
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

    __aicore__ inline void RowmaxSPECTILE256(const AscendC::LocalTensor<float>& srcUb,
                                             const AscendC::LocalTensor<float>& rowmaxUb,
                                             const AscendC::LocalTensor<float>& tvUbTensor, uint32_t numRowsRound,
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

    __aicore__ inline void RowmaxTAILTILE(const AscendC::LocalTensor<float>& srcUb,
                                          const AscendC::LocalTensor<float>& rowmaxUb,
                                          const AscendC::LocalTensor<float>& tvUbTensor, uint32_t numRowsRound,
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
                                       uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(lsUbTensor[sUbOffset], gInput,
                          AscendC::DataCopyParams(rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE,
                                                  (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE, 0));
    }

    __aicore__ inline void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<float, false>(lsUbTensor[sUbOffset], lsUbTensor[sUbOffset], scaleValue, (uint64_t)0,
                                    CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
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

    // Row maxima of a later stack of a task: hm = max(lm, gm), gm being the maxima of the task's earlier stacks. The
    // first stack of a task uses lm directly. The other per-row values follow once per stack (FinishStack).
    __aicore__ inline void UpdateRowMax(uint32_t rowNumCurLoop, uint32_t rowOffset)
    {
        SetVecMask(rowNumCurLoop);
        // *** hm = vmax(lm, gm)
        AscendC::Max<float, false>(hmUbTensor[rowOffset], lmUbTensor[rowOffset], gmUbTensor[rowOffset], (uint64_t)0, 1,
                                   AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    // After the row loops of a stack, for all its rows at once: the rescaling factors dm = exp(gm - hm) and the running
    // normalizers gl = dm * gl_prev + ll of the stack's slot, and the new row maxima gm = hm. Every row belongs to one
    // row loop, so updating these per loop computes the same values.
    __aicore__ inline void FinishStack(uint32_t rowNum, uint32_t isFirstStackTile, uint32_t curStackTileMod)
    {
        const uint32_t slotOffset = curStackTileMod * MAX_ROW_NUM_SUB_CORE;
        const uint32_t prevSlotOffset = ((curStackTileMod + slotNum - 1) % slotNum) * MAX_ROW_NUM_SUB_CORE;
        const uint32_t rowC0Num = CeilDiv(rowNum, FLOAT_BLOCK_SIZE);
        const int32_t count = static_cast<int32_t>(rowNum);
        if (isFirstStackTile) {
            // *** gm = lm, gl = ll
            AscendC::DataCopy(gmUbTensor, lmUbTensor, AscendC::DataCopyParams(1, rowC0Num, 0, 0));
            if (!skipRowSum) {
                AscendC::DataCopy(glUbTensor[slotOffset], llUbTensor, AscendC::DataCopyParams(1, rowC0Num, 0, 0));
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        // *** dm = exp(gm - hm)
        AscendC::Sub(dmUbTensor[slotOffset], gmUbTensor, hmUbTensor, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(dmUbTensor[slotOffset], dmUbTensor[slotOffset], count);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm
        AscendC::DataCopy(gmUbTensor, hmUbTensor, AscendC::DataCopyParams(1, rowC0Num, 0, 0));
        if (!skipRowSum) {
            // *** gl = dm * gl_prev + ll
            AscendC::Mul(glUbTensor[slotOffset], dmUbTensor[slotOffset], glUbTensor[prevSlotOffset], count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(glUbTensor[slotOffset], glUbTensor[slotOffset], llUbTensor, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
                                   uint32_t columnNum, uint32_t columnNumRound,
                                   const AscendC::LocalTensor<float>& rowMax)
    {
        // *** hm_block = expand_to_block(hm), 存放于 tv
        AscendC::Brcb(tvUbTensor.template ReinterpretCast<uint32_t>(), rowMax.template ReinterpretCast<uint32_t>(),
                      rowNumCurLoopRound / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
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

    // Half-precision index in lpUbTensor of the probabilities of the scores at float index sUbOffset.
    __aicore__ inline uint32_t PUbOffset(uint32_t sUbOffset) const
    {
        return separateP ? 0 : 2 * (sUbOffset + PADDED_S_ELEM_NUM);
    }

    __aicore__ inline void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** lp = castfp32to16(ls)
        const uint32_t pOffset = PUbOffset(sUbOffset);
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[pOffset], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[pOffset], lsUbTensor[sUbOffset], AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    __aicore__ inline void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset,
                                       uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(gOutput, lpUbTensor[PUbOffset(sUbOffset)],
                          AscendC::DataCopyParams(rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0,
                                                  (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    template <bool doTriUMask>
    __aicore__ inline void SubCoreCompute(AscendC::GlobalTensor<ElementOutput> gOutput,
                                          const LayoutOutput& layoutOutput, uint32_t rowOffset,
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

        if constexpr (LSE_MODE == LseMode::OUT_ONLY) {
            // wait for lse from ub to gm
            // In lse out-only mode, tv is used in the last stack tile to transport lse
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        if (!isFirstStackTile) {
            UpdateRowMax(rowNumCurLoop, rowOffset);
        }

        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound,
                isFirstStackTile ? lmUbTensor[rowOffset] : hmUbTensor[rowOffset]);
        // The row sums read the float probabilities, which the cast below overwrites.
        if (!skipRowSum) {
            CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        }
        // The probability buffer is overwritten only after its previous probabilities have been stored.
        const uint32_t pStoredEvent = separateP ? SEPARATE_P_STORED_EVENT : pingpongFlag;
        if constexpr (!doTriUMask) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pStoredEvent);
        }

        DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        if constexpr (!doTriUMask) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pStoredEvent);
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
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementOutput> gOutput,
                                      AscendC::GlobalTensor<ElementInput> gInput, const LayoutOutput& layoutOutput,
                                      const LayoutInput& layoutInput, GemmCoord actualBlockShape,
                                      uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile, uint32_t qSBlockSize,
                                      uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag softmaxFlag)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
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
        uint32_t maxRowNumPerLoop = sElemNumPerLoop / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;
        if (rowActualThisSubBlock == 0) {
            NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxFlag);
            return;
        }

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
                CopySGmToUb(gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound,
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
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                SubCoreCompute<false>(gOutputCurLoop, layoutOutputCurLoop, rowOffsetCurLoop, isFirstStackTile,
                                      isLastNoMaskStackTile, (delayedRowLoopIdx == 0),
                                      (delayedRowLoopIdx == rowLoopNum - 1), columnNumRound, pingpongFlag,
                                      curStackTileMod, softmaxFlag);
            }
        }
        FinishStack(rowActualThisSubBlock, isFirstStackTile, curStackTileMod);
    }

private:
    float scaleValue;
    uint32_t slotNum = StaUbLayout::STACK_SLOT_MAX;
    bool skipRowSum = false;
    uint32_t sElemNumPerLoop = MAX_UB_S_ELEM_NUM;
    // true: 12288-score loops keep their probabilities in LP; false: 8192-score loops keep them behind their scores.
    bool separateP = true;
    AscendC::LocalTensor<float> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
};
template <class OutputType_, class InputType_, class UpdateType_, class LseType_>
class BlockEpilogue<StaRescaleO, OutputType_, InputType_, UpdateType_, LseType_> {
public:
    // Type aliases
    using DispatchPolicy = StaRescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementUpdate = typename UpdateType_::Element;
    using ElementLse = typename LseType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutUpdate = typename UpdateType_::Layout;
    using LayoutLse = typename LseType_::Layout;

    static constexpr LseMode LSE_MODE = DispatchPolicy::LSE_MODE;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;
    static constexpr uint32_t MULTIPLIER = 2;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t NUM4 = 4;
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = StaUbLayout::O_ELEM_NUM;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr uint32_t SIZE_OF_16BIT = 2;

    // sumColumn: non-zero when the value tensor carries ones in columns [sumColumn, embed); the accumulated output
    // then holds each row's probability sum there and replaces the normalizer tracked by the online softmax.
    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, uint32_t sumColumn_)
    {
        sumColumn = sumColumn_;
        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::LO);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::DM);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::GL);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::TV);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(StaUbLayout::GO);
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::GO);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::HM);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::GM);
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(StaUbLayout::GL);
    }

    __aicore__ inline ~BlockEpilogue() {}

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

    __aicore__ inline void CopyOToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx,
                                     uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
                                     uint32_t qSThisSubBlock, uint32_t embed, uint32_t oHiddenSize)
    {
        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(gOutput[innerOGmOffset + proTokenIdx * oHiddenSize], goUbTensor16[innerGOUbOffset],
                                 AscendC::DataCopyExtParams(proTokenNum, embed * SIZE_OF_16BIT, 0,
                                                            (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embed;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(gOutput[innerOGmOffset], goUbTensor16[innerGOUbOffset],
                                 AscendC::DataCopyExtParams(qSThisSubBlock, embed * SIZE_OF_16BIT, 0,
                                                            (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embed;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(gOutput[innerOGmOffset], goUbTensor16[innerGOUbOffset],
                                 AscendC::DataCopyExtParams(epiTokenNum, embed * SIZE_OF_16BIT, 0,
                                                            (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
        }
    }

    __aicore__ inline void SubCoreCompute(AscendC::GlobalTensor<ElementOutput> gOutput,
                                          AscendC::GlobalTensor<ElementInput> gInput,
                                          AscendC::GlobalTensor<ElementUpdate> gUpdate,
                                          AscendC::GlobalTensor<ElementLse> gLse, const LayoutOutput& layoutOutput,
                                          const LayoutInput& layoutInput, const LayoutUpdate& layoutUpdate,
                                          const LayoutLse& layoutLse, uint32_t qNThisSubBlock, uint32_t qSThisSubBlock,
                                          uint32_t totalRowNum, uint32_t isFirstStackTile, uint32_t isLastStackTile,
                                          uint32_t curStackTileMod, uint32_t needRowLoop, uint32_t isLastRowLoop,
                                          uint32_t rowOffsetLoop, uint32_t proTokenIdx, uint32_t proTokenNum,
                                          uint32_t epiTokenNum, uint32_t integralHeadNum)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t stride = layoutLse.shape(1); // stride for lse copy out
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        if (!isFirstStackTile) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::DataCopy(loUbTensor, gInput,
                              AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                          dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(),
                          curRowNumRound / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                AscendC::DataCopy(goUbTensor32, gUpdate,
                                  AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // *** go = go * dm_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, false>(goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                                           goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0,
                                           curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Mul<float, false>(goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                           goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], tvUbTensor,
                                           (uint64_t)0, curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go
            AscendC::Add<float, false>(goUbTensor32, goUbTensor32, loUbTensor, (uint64_t)0,
                                       (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // *** go = lo
            AscendC::DataCopy(goUbTensor32, gInput,
                              AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            if (sumColumn != 0U) {
                // The C0 at sumColumn repeats the row's probability sum: exactly the per-row block layout below.
                AscendC::DataCopy(tvUbTensor, goUbTensor32[sumColumn],
                                  AscendC::DataCopyParams(curRowNum, 1, embedRound / FLOAT_BLOCK_SIZE - 1, 0));
            } else {
                // *** gl_block = expand_to_block(gl), 存放于 tv
                AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                              glUbTensor.ReinterpretCast<uint32_t>()[dmUbOffsetCurStackTile],
                              curRowNumRound / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
            }
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<float, false>(goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                                           goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE], tvUbTensor, (uint64_t)0,
                                           curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Div<float, false>(goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                           goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], tvUbTensor,
                                           (uint64_t)0, curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // *** go = castfp32to16(go)
            if (std::is_same<ElementOutput, bfloat16_t>::value) {
                AscendC::Cast<ElementOutput, float, false>(
                    goUbTensor16, goUbTensor32, AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                    (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
            } else {
                AscendC::Cast<ElementOutput, float, false>(
                    goUbTensor16, goUbTensor32, AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                    (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // ***move O to GM
            CopyOToGm(gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed,
                      oHiddenSize);
            if constexpr (LSE_MODE == LseMode::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln<float, false>(lse32_ubuf_tensor, glUbTensor, (uint64_t)0,
                                              CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                                              AscendC::UnaryRepeatParams(1, 1, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float, false>(lse32_ubuf_tensor, lse32_ubuf_tensor, gmUbTensor, (uint64_t)0,
                                               CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                                               AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(), lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                                  CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE), AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    if (qNThisSubBlock == 0U) {
                        AscendC::DataCopyPad(
                            gLse, tvUbTensor,
                            AscendC::DataCopyExtParams(totalRowNum, sizeof(float), 0, (stride - 1) * sizeof(float), 0));
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            AscendC::DataCopyPad(gLse[qNIdx], tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                                 AscendC::DataCopyExtParams(qSBlockSize, sizeof(float), 0,
                                                                            (stride - 1) * sizeof(float), 0));
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                }
            }
        } else if (needRowLoop) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(gUpdate, goUbTensor32,
                              AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementOutput> gOutput,
                                      AscendC::GlobalTensor<ElementInput> gInput,
                                      AscendC::GlobalTensor<ElementUpdate> gUpdate,
                                      AscendC::GlobalTensor<ElementLse> gLse, const LayoutOutput& layoutOutput,
                                      const LayoutInput& layoutInput, const LayoutUpdate& layoutUpdate,
                                      const LayoutLse& layoutLse, GemmCoord actualBlockShape, uint32_t qSBlockSize,
                                      uint32_t qNBlockSize, uint32_t isFirstStackTile, uint32_t isLastStackTile,
                                      uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0 :
                                  (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) :
                                                        qNSplitSubBlock;
        uint32_t inRowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embed;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock;
        int64_t offsetLse = layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];

        if (inRowActualThisSubBlock > 0U) {
            uint32_t rowLoop = CeilDiv(inRowActualThisSubBlock, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;

            // The rows of each cycle consist of multiple heads with several tokens.
            // There are several integral heads, one prologue head, one epilogue head.
            uint32_t proTokenIdx = 0;     // the token idx of the start token of the prologue part
            uint32_t proTokenIdxPre = 0;  // the token idx of the start token of the pre prologue part
            uint32_t proTokenNum = 0;     // the token num of the prologue part
            uint32_t epiTokenNum = 0;     // the token num of the epilogue part
            uint32_t integralHeadNum = 0; // the number of integral heads within a cycle
            uint32_t qSRemian = qSThisSubBlock;
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                int64_t offsetOutput =
                    static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + outOffsetSubBlock;
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput;
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gInputCurLoop = gInput[offsetInput];
                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                SubCoreCompute(gOutputCurLoop, gInputCurLoop, gUpdateCurLoop, gLseThisSubBlock, layoutOutputCurLoop,
                               layoutInputCurLoop, layoutUpdateCurLoop, layoutLse, qNThisSubBlock, qSThisSubBlock,
                               inRowActualThisSubBlock, isFirstStackTile, isLastStackTile, curStackTileMod, needRowLoop,
                               (rowLoopIdx == rowLoop - 1U), rowOffsetLoop, proTokenIdx, proTokenNum, epiTokenNum,
                               integralHeadNum);
            }
        }
    }

private:
    uint32_t sumColumn = 0;
    AscendC::LocalTensor<float> loUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<ElementOutput> goUbTensor16;
    AscendC::LocalTensor<float> goUbTensor32;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> lse32_ubuf_tensor;
};
} // namespace NpuArch::Epilogue::Block

#endif // STA_EPILOGUE_H
