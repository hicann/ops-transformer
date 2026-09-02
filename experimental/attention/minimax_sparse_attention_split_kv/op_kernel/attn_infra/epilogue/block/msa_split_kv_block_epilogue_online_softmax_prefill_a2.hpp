/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_MSA_SPLIT_KV_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_A2_HPP
#define EPILOGUE_BLOCK_MSA_SPLIT_KV_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_A2_HPP

#include "../../../attn_infra/msa_split_kv_base_defs.hpp"
#include "../../../attn_infra/arch/msa_split_kv_cross_core_sync.hpp"
#include "../../../attn_infra/arch/msa_split_kv_resource.hpp"
#include "../../../attn_infra/epilogue/msa_split_kv_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/msa_split_kv_gemm_coord.hpp"
#include "../../../attn_infra/msa_split_kv_matrix_coord.hpp"
#include "../../../tla/msa_split_kv_tla_layout.hpp"
#include "../../../tla/msa_split_kv_tla_tensor.hpp"

#ifndef KERNEL_DUMP
#define KERNEL_DUMP 0
#endif

#ifndef KERNEL_DUMP_SCORE
#define KERNEL_DUMP_SCORE 0
#endif

#ifndef KERNEL_DUMP_P
#define KERNEL_DUMP_P 0
#endif

#ifndef KERNEL_DUMP_PHASE1_ROWSUM
#define KERNEL_DUMP_PHASE1_ROWSUM 0
#endif

#ifndef KERNEL_DUMP_CORE
#define KERNEL_DUMP_CORE 1U
#endif

#ifndef KERNEL_DUMP_SUBBLOCK
#define KERNEL_DUMP_SUBBLOCK 1U
#endif

#ifndef KERNEL_DUMP_BI
#define KERNEL_DUMP_BI 0U
#endif

namespace NpuArch::Epilogue::Block {

template <class OutputType_, class ElementInput_, class LayoutS_>
class BlockEpilogue<EpilogueOnlineSoftmaxPrefillA2, OutputType_, Gemm::GemmType<ElementInput_, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxPrefillA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = ElementInput_;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    // === A2 compact constants (fit 192KB UB) ===
    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    // Static-tensor programming reserves event IDs 6/7.  V_MTE2 IDs 0/1
    // protect the two S stages, leaving ID2 for this cross-pipe bridge.
    static constexpr uint32_t QK_READY_TO_MTE2_EVENT_ID = EVENT_ID2;

    // A batch may contain eight query groups.  With the two AIV sub-blocks
    // each owning four groups and groupSize=16, one sub-block needs 64 stats
    // elements.  The former 64-element total was split into two 32-element
    // regions, so the second half of each stats tile overwrote the adjacent
    // stage for high-GQA configurations.
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 128;
    static constexpr uint32_t SM_UB_STAGES = 2;
    static constexpr uint32_t STATS_UB_STAGE_BYTES = SM_ROW_MAX_ELEM_NUM * sizeof(float);

    static constexpr float MIN_VALUE_FP32 = -3.4028235e38f;

    // GM S workspace (A2: CUBE fixpipes L0C→GM, VEC does DataCopyPad GM→UB).
    // Set via SetGmSWorkspace before the Phase1 loop.
    AscendC::GlobalTensor<ElementInput> gmSWorkspace_;
    uint64_t gmSStageElems_ = 0;

    // GM P workspace (A2: VEC writes P to GM via DataCopyPad, CUBE reads P from GM via Nd2Nz).
    // A2 does NOT support VEC UB→L1 direct write (A5-only feature).
    AscendC::GlobalTensor<ElementOutput> gmPWorkspace_;
    uint64_t gmPStageElems_ = 0;

    __aicore__ inline void SetGmSWorkspace(AscendC::GlobalTensor<ElementInput> &gmSWorkspace, uint64_t gmSStageElems)
    {
        gmSWorkspace_ = gmSWorkspace;
        gmSStageElems_ = gmSStageElems;
    }

    __aicore__ inline void SetGmPWorkspace(AscendC::GlobalTensor<ElementOutput> &gmPWorkspace, uint64_t gmPStageElems)
    {
        gmPWorkspace_ = gmPWorkspace;
        gmPStageElems_ = gmPStageElems;
    }

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // UB layout (must match kernel's SM_UB_GM_OFFSET / SM_UB_GL_OFFSET):
        //   LS  (bf16 S input):  offset 0,                  2 stages * 16384B = 32768B
        //   LP  (bf16 P output): offset 2*16384 = 32768,    2 stages * 16384B = 32768B
        //   FS  (float S work):  offset 4*16384 = 65536,    1 stage  * 32768B = 32768B
        //   TV  (float temp):    offset 6*16384 = 98304,    ~16KB (reduce scratch)
        //   GM  (rowMax stats):  offset 7*16384 = 114688,   2 stages * 256B
        //   GL  (rowSum stats):  offset 114688 + 512 = 115200, 2 stages * 256B
        // SubBlock partitioning: TV, GM, GL are partitioned by subBlock in the
        // constructor (offset depends only on subBlockIdx_). LS, LP, FS are
        // partitioned in operator() (offset depends on mCopyOffset * nRound).
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t FS_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 2 * STATS_UB_STAGE_BYTES;

        subBlockIdx_ = AscendC::GetSubBlockIdx();
        scaleValue = scaleValue_;

        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        fsUbTensor = resource.ubBuf.template GetBufferByByte<float>(FS_UB_TENSOR_OFFSET);
        // Partition TV by subBlock: each gets REDUCE_UB_SIZE*2 floats (scratch + result).
        // Total: 2 subBlocks * 2 * 1024 * 4B = 16384B, fits in ~16KB TV region.
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET +
                                                                    subBlockIdx_ * REDUCE_UB_SIZE * 2U * sizeof(float));
        // Partition stats (GM/GL) by subBlock: each gets 64 floats, enough for
        // four groupSize=16 query groups in a full 128-row batch.
        constexpr uint32_t SUBBLOCK_STATS_ELEMS = SM_ROW_MAX_ELEM_NUM / 2U;
        for (uint32_t i = 0; i < SM_UB_STAGES; i++) {
            gmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                GM_UB_TENSOR_OFFSET + i * STATS_UB_STAGE_BYTES + subBlockIdx_ * SUBBLOCK_STATS_ELEMS * sizeof(float));
            glUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                GL_UB_TENSOR_OFFSET + i * STATS_UB_STAGE_BYTES + subBlockIdx_ * SUBBLOCK_STATS_ELEMS * sizeof(float));
        }
    }

    __aicore__ inline ~BlockEpilogue() {}

    // === A2: MODE_2 (0x2) cross-core sync — proven A2 pattern from infer code ===
    // SET: CrossCoreSetFlag<0x2, PIPE_xxx> (PIPE_V for mm1ToSmFlag, PIPE_MTE3 for smToMm2Flag)
    // WAIT: CrossCoreWaitFlag (non-templated, MODE=0/PIPE_S — generic sync pipe)
    // MODE_4 with PIPE_FIX↔PIPE_V pairing does NOT work on A2 hardware.
    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 0x2U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 0x2U) {
            // A2: must use MODE=0x2 to match SET. Arch::CrossCoreWaitFlag defaults
            // to MODE=0/PIPE_S — MODE mismatch causes Wait to never see the SET.
            AscendC::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag.id);
        }
    }

    // === Vector mask helpers (from A2 infer / A5 prefill patterns) ===
    __aicore__ inline void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < static_cast<int64_t>(temp); i++) {
            mask |= one << i;
        }
        if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, static_cast<uint64_t>(-1));
        } else if (len > 0) {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        } else {
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    __aicore__ inline void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            return;
        }
        uint64_t subMask = (static_cast<uint64_t>(1) << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    // === Cast bf16 S → float FS (for standard vector compute) ===
    __aicore__ inline void CastSToFloat(AscendC::LocalTensor<float> &fsUb,
                                        AscendC::LocalTensor<ElementInput> const &lsUb, uint32_t m, uint32_t nRound)
    {
        if constexpr (!AscendC::IsSameType<ElementInput, float>::value) {
            AscendC::Cast<float, ElementInput, false>(fsUb, lsUb, AscendC::RoundMode::CAST_NONE,
                                                      static_cast<uint64_t>(0), CeilDiv(m * nRound, FLOAT_VECTOR_SIZE),
                                                      AscendC::UnaryRepeatParams(1, 1, 8, 4));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // === Scale: FS = FS * scaleValue ===
    __aicore__ inline void ScaleS(AscendC::LocalTensor<float> &fsUb, uint32_t m, uint32_t nRound)
    {
        AscendC::Muls<float, false>(fsUb, fsUb, scaleValue, static_cast<uint64_t>(0),
                                    CeilDiv(m * nRound, FLOAT_VECTOR_SIZE), AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // === Zero out [tailN, nRound) for each row (causal mask after Exp) ===
    // Called AFTER SubtractRowMaxAndExp so we zero the exp result directly.
    // Handles unaligned tailN: A2 VEC instructions require 32-byte (8-float)
    // aligned UB addresses. When tailN % 8 != 0, the unaligned part is zeroed
    // via a masked Duplicate at the aligned address tailNAlignedDown.
    __aicore__ inline void ZeroOutCausalTail(AscendC::LocalTensor<float> &fsUb, uint32_t rows, uint32_t nRound,
                                             uint32_t tailN)
    {
        if (tailN >= nRound) {
            return;
        }
        uint32_t tailNAlignedDown = (tailN / FLOAT_BLOCK_SIZE) * FLOAT_BLOCK_SIZE;
        uint32_t tailNRem = tailN % FLOAT_BLOCK_SIZE; // 0..7

        for (uint32_t row = 0; row < rows; row++) {
            uint32_t rowOff = row * nRound;

            // Overwrite instead of multiplying by zero: stale NaNs in padded
            // L0C lanes survive NaN * 0 and would poison the subsequent RowSum.
            if (tailNRem != 0U) {
                // Build mask: bits [tailNRem, FLOAT_BLOCK_SIZE) set
                uint64_t mask = 0;
                for (uint32_t i = tailNRem; i < FLOAT_BLOCK_SIZE; i++) {
                    mask |= (1ULL << i);
                }
                AscendC::SetVectorMask<int8_t>(0x0, mask);
                AscendC::Duplicate<float, false>(fsUb[rowOff + tailNAlignedDown], 0.0f, static_cast<uint64_t>(0), 1, 1,
                                                 8);
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }

            // Aligned part: zero [alignedStart, nRound) via Duplicate(0)
            uint32_t alignedStart = tailNAlignedDown + (tailNRem != 0U ? FLOAT_BLOCK_SIZE : 0U);
            if (alignedStart < nRound) {
                for (uint32_t col = alignedStart; col < nRound; col += FLOAT_VECTOR_SIZE) {
                    uint32_t fillLen = (col + FLOAT_VECTOR_SIZE <= nRound) ? FLOAT_VECTOR_SIZE : (nRound - col);
                    AscendC::Duplicate<float>(fsUb[rowOff + col], 0.0f, static_cast<int32_t>(fillLen));
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // === RowMax: two-stage BlockReduceMax (from A2 infer RowmaxTAILTILE pattern) ===
    __aicore__ inline void CalcRowMax(AscendC::LocalTensor<float> const &srcUb, AscendC::LocalTensor<float> &rowmaxUb,
                                      uint32_t numRows, uint32_t numRowsRound, uint32_t numElems,
                                      uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb, numRows, 0, 1, 1,
                                                  numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb, tvUbTensor, CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t colIdx = 1; colIdx < static_cast<uint64_t>(numElems) / FLOAT_VECTOR_SIZE; ++colIdx) {
                AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb[colIdx * FLOAT_VECTOR_SIZE], numRows, 0, 1, 1,
                                                      numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                                      CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1,
                                                      1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Max<float, false>(rowmaxUb, rowmaxUb, tvUbTensor[REDUCE_UB_SIZE], static_cast<uint64_t>(0), 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::BlockReduceMax<float, false>(tvUbTensor, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                  numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(static_cast<int32_t>(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE)));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceMax<float, false>(
                    rowmaxUb, tvUbTensor, CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            } else {
                AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                                      CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1,
                                                      1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Max<float, false>(rowmaxUb, rowmaxUb, tvUbTensor[REDUCE_UB_SIZE], static_cast<uint64_t>(0), 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    // === Subtract rowMax and Exp (from A2 infer CalcExp pattern) ===
    __aicore__ inline void SubtractRowMaxAndExp(AscendC::LocalTensor<float> &srcUb,
                                                AscendC::LocalTensor<float> const &rowmaxUb, uint32_t numRows,
                                                uint32_t numRowsRound, uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::Brcb(tvUbTensor.template ReinterpretCast<uint32_t>(), rowmaxUb.template ReinterpretCast<uint32_t>(),
                      numRowsRound / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t colBlocks = numElemsAligned / FLOAT_BLOCK_SIZE;
        for (uint32_t colIdx = 0; colIdx < numElems / FLOAT_VECTOR_SIZE; ++colIdx) {
            AscendC::Sub<float, false>(srcUb[colIdx * FLOAT_VECTOR_SIZE], srcUb[colIdx * FLOAT_VECTOR_SIZE], tvUbTensor,
                                       static_cast<uint64_t>(0), numRows,
                                       AscendC::BinaryRepeatParams(1, 1, 0, colBlocks, colBlocks, 1));
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::Sub<float, false>(srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                       srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], tvUbTensor,
                                       static_cast<uint64_t>(0), numRows,
                                       AscendC::BinaryRepeatParams(1, 1, 0, colBlocks, colBlocks, 1));
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp<float, false>(srcUb, srcUb, static_cast<uint64_t>(0),
                                   CeilDiv(numRows * numElemsAligned, FLOAT_VECTOR_SIZE),
                                   AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // === RowSum: two-stage BlockReduceSum (from A2 infer RowsumTAILTILE pattern) ===
    __aicore__ inline void CalcRowSum(AscendC::LocalTensor<float> const &srcUb, AscendC::LocalTensor<float> &rowsumUb,
                                      uint32_t numRows, uint32_t numRowsRound, uint32_t numElems,
                                      uint32_t numElemsAligned)
    {
        // General two-stage reduction for larger row tiles.
        uint32_t reduceScratchElems = numRowsRound * FLOAT_BLOCK_SIZE;
        AscendC::Duplicate<float>(tvUbTensor, 0.0f, reduceScratchElems);
        AscendC::Duplicate<float>(rowsumUb, 0.0f, numRowsRound);
        AscendC::PipeBarrier<PIPE_V>();
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb, numRows, 0, 1, 1,
                                                  numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb, tvUbTensor, CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t colIdx = 1; colIdx < static_cast<uint64_t>(numElems) / FLOAT_VECTOR_SIZE; ++colIdx) {
                AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb[colIdx * FLOAT_VECTOR_SIZE], numRows, 0, 1, 1,
                                                      numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                                      CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1,
                                                      1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Add<float, false>(rowsumUb, rowsumUb, tvUbTensor[REDUCE_UB_SIZE], static_cast<uint64_t>(0), 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::BlockReduceSum<float, false>(tvUbTensor, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                  numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(static_cast<int32_t>(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE)));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb, tvUbTensor, CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1, 1, 8);
            } else {
                AscendC::BlockReduceSum<float, false>(tvUbTensor[REDUCE_UB_SIZE], tvUbTensor,
                                                      CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE), 0, 1,
                                                      1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Add<float, false>(rowsumUb, rowsumUb, tvUbTensor[REDUCE_UB_SIZE], static_cast<uint64_t>(0), 1,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    // === DownCast float→16-bit P (from A2 infer DownCastP pattern) ===
    __aicore__ inline void DownCastP(AscendC::LocalTensor<ElementOutput> &lpUb, AscendC::LocalTensor<float> const &fsUb,
                                     uint32_t m, uint32_t nRound)
    {
        if constexpr (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(lpUb, fsUb, AscendC::RoundMode::CAST_RINT,
                                                       static_cast<uint64_t>(0), CeilDiv(m * nRound, FLOAT_VECTOR_SIZE),
                                                       AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(lpUb, fsUb, AscendC::RoundMode::CAST_ROUND,
                                                       static_cast<uint64_t>(0), CeilDiv(m * nRound, FLOAT_VECTOR_SIZE),
                                                       AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // === Copy RowMajor UB P → GM P workspace (A2 path) ===
    // A2 does NOT support VEC UB→L1 DataCopy (A5-only feature, see cross_platform_migration_guide).
    // Instead, VEC writes P to GM via DataCopyPad (MTE3, valid on VEC), then CUBE reads P
    // from GM to L1 via Nd2Nz DataCopy (MTE2, valid on CUBE).
    // Double-buffered by bi%2; sub-block offset ensures both AIVs write to correct rows.
    __aicore__ inline void CopyPUbToGm(AscendC::LocalTensor<ElementOutput> const &ubPTensor, uint32_t m,
                                       uint32_t nRound, uint32_t bi, uint32_t mCopyOffset)
    {
        uint32_t gmStageOffset = static_cast<uint32_t>((bi % 3U) * gmPStageElems_);
        uint32_t gmSubOffset = (subBlockIdx_ == 0) ? gmStageOffset : (gmStageOffset + mCopyOffset * nRound);
#if KERNEL_DUMP
        if (AscendC::GetBlockIdx() == 1U && bi == 0U) {
            AscendC::DumpTensor(gmSWorkspace_[gmSubOffset], 992, m * nRound);
        }
#endif
        AscendC::DataCopyExtParams copyParams(static_cast<uint16_t>(m), nRound * sizeof(ElementOutput), 0, 0, 0);
        AscendC::DataCopyPad(gmPWorkspace_[gmSubOffset], ubPTensor, copyParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // === operator() — A2: adds bi param for GM workspace stage offset ===
    template <class TensorP>
    __aicore__ inline void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape, uint32_t ubSBufId,
                                      uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag,
                                      Arch::CrossCoreFlag smToMm2Flag, const uint32_t *causalValidLens,
                                      uint32_t groupCount, uint32_t groupRows, uint32_t bi, bool dumpPhase1Rowsum,
                                      bool extraToSubBlock0)
    {
        (void)l1PTensorTla; // A2: VEC writes P to GM, not L1. Kept for API compatibility.
        (void)l1PBufId;
        uint32_t M = actualBlockShape.m();
        // Split by group so each AIV owns whole query groups.
        // Keep complete query groups on one AIV, but alternate the owner of the
        // odd group.  This preserves the exact per-row math while preventing
        // every one-group batch from running entirely on subBlock 0.
        uint32_t group0 = groupCount / 2U;
        if ((groupCount & 1U) != 0U && extraToSubBlock0) {
            ++group0;
        }
        uint32_t mCopyOffset = group0 * groupRows;
        uint32_t mHalf = mCopyOffset < M ? mCopyOffset : M;
        uint32_t m = subBlockIdx_ == 0U ? mHalf : (M - mHalf);
        if (m == 0) {
            // MODE_2 cross-core sync requires BOTH AIV subBlocks to participate.
            // AIC's counter only increments when BOTH AIVs SET; if one subBlock
            // skips, AIC's WAIT hangs forever. SubBlock 1 (m=0, no data) must
            // still execute all 4 cross-core ops to maintain the protocol.
            // Pipes MUST match the main path exactly: mm1ToSm WAIT+SET on
            // PIPE_V (matching line 686), smToMm2 WAIT on PIPE_MTE3 (matching
            // line 687), smToMm2 SET on PIPE_MTE3 (matching line 693).
            WaitCrossCoreSync<0x2U, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<0x2U, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<0x2U, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<0x2U, PIPE_MTE3>(smToMm2Flag);
            return;
        }

        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t startRow = subBlockIdx_ * mCopyOffset;
        uint32_t endRow = startRow + m;

        // UB buffer bases for this stage — partitioned by subBlock.
        // Each subBlock gets its own region within the stage buffer to avoid
        // data corruption when both subBlocks have m > 0 (e.g. groupSize=1,
        // groupCount>=2). SubBlock 0 writes [0, mHalf*nRound), subBlock 1 writes
        // [mCopyOffset*nRound, M*nRound) — no overlap.
        // Limitation: requires m*nRound <= MAX_UB_S_ELEM_NUM/2 per subBlock.
        uint32_t ubSubOffset = subBlockIdx_ * mCopyOffset * nRound;
        AscendC::LocalTensor<ElementInput> lsStage;
        if constexpr (!AscendC::IsSameType<ElementInput, float>::value) {
            lsStage = lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM + ubSubOffset];
        }
        auto lpStage = lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM + ubSubOffset];
        auto fsStage = fsUbTensor[ubSubOffset];
        auto nowMax = gmUbTensor[ubSBufId];
        auto nowSum = glUbTensor[ubSBufId];

        // === A2: DataCopyPad GM→UB to load S from GM workspace ===
        // CUBE fixpiped L0C→GM and signaled via MODE_2 cross-core flag.
        // The cross-core wait runs on V, while the following GM->UB transfer
        // runs on MTE2. Bridge the dependency before issuing that transfer.
        WaitCrossCoreSync<0x2U, PIPE_V>(mm1ToSmFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(QK_READY_TO_MTE2_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(QK_READY_TO_MTE2_EVENT_ID);

        // GM->UB is an MTE2 transfer. The stage event protects UB reuse;
        // QK_READY_TO_MTE2_EVENT_ID above protects GM S visibility.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubSBufId);
        uint32_t gmStageOffset = (bi % 2U) * gmSStageElems_;
        // QK's A2 fixpipe keeps a fixed physical row stride of
        // RoundUp(blockSize, 16) elements in GM, even when the current KV
        // block is shorter.  The UB tile is compact, so describe the gap
        // between copied rows explicitly instead of treating GM as compact.
        constexpr uint32_t GM_S_STAGE_ROWS = 64U;
        uint32_t gmStageRowStride = static_cast<uint32_t>(gmSStageElems_ / GM_S_STAGE_ROWS);
        uint32_t gmSubOffset = (subBlockIdx_ == 0) ? gmStageOffset : (gmStageOffset + mCopyOffset * gmStageRowStride);
        AscendC::DataCopyExtParams copyParams(static_cast<uint16_t>(m), nRound * sizeof(ElementInput),
                                              (gmStageRowStride - nRound) * sizeof(ElementInput), 0, 0);
        AscendC::DataCopyPadExtParams<ElementInput> padParams(false, 0, 0, 0);
        if constexpr (AscendC::IsSameType<ElementInput, float>::value) {
            AscendC::DataCopyPad(fsStage, gmSWorkspace_[gmSubOffset], copyParams, padParams);
        } else {
            AscendC::DataCopyPad(lsStage, gmSWorkspace_[gmSubOffset], copyParams, padParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubSBufId);
#if KERNEL_DUMP || KERNEL_DUMP_P || KERNEL_DUMP_SCORE
        const bool dumpThisBatch =
            AscendC::GetBlockIdx() == KERNEL_DUMP_CORE && subBlockIdx_ == KERNEL_DUMP_SUBBLOCK && bi == KERNEL_DUMP_BI;
#if KERNEL_DUMP || KERNEL_DUMP_SCORE
        if (dumpThisBatch) {
            // S is the unscaled QK result read back from GM staging.
            if constexpr (AscendC::IsSameType<ElementInput, float>::value) {
                AscendC::DumpTensor(fsStage, 900, m * nRound);
            } else {
                AscendC::DumpTensor(lsStage, 900, m * nRound);
            }
        }
#endif
#endif

        // Wait for previous iteration's UB→L1 copy (MTE3) to complete before
        // V can safely overwrite lpStage. Pre-primed in InitSyncFlags for first iteration.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2U);
        // gmUb/glUb share this two-stage ring with the asynchronous stats
        // scatter. Do not overwrite a stage until its prior MTE3 copy is done.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 4U);

        // V compute: fp32 score is already resident in fsStage; bf16 score is cast first.
        CastSToFloat(fsStage, lsStage, m, nRound);

        // Step 2: Scale FS = FS * scaleValue.
        ScaleS(fsStage, m, nRound);
#if KERNEL_DUMP
        if (dumpThisBatch) {
            // This corresponds to golden's scores_S tensor.
            AscendC::DumpTensor(fsStage, 910, m * nRound);
        }
#endif

        // Step 3: Per-group compute.
        uint32_t gStart = startRow / groupRows;
        uint32_t gEnd = (endRow == 0) ? 0 : (endRow - 1) / groupRows;
        for (uint32_t g = gStart; g <= gEnd; g++) {
            uint32_t grpLo = g * groupRows;
            uint32_t grpHi = grpLo + groupRows;
            uint32_t lo = grpLo > startRow ? grpLo : startRow;
            uint32_t hi = grpHi < endRow ? grpHi : endRow;
            uint32_t rows = hi - lo;
            if (rows == 0) {
                continue;
            }

            uint32_t localOff = lo - startRow;
            uint32_t grpStride = RoundUp(groupRows, 8U);
            uint32_t statsOff = (g - gStart) * grpStride;
            uint32_t tailN = causalValidLens[g];

            // FS slice for this group's rows: [rows, nRound] RowMajor float.
            auto fsSlice = fsStage[localOff * nRound];
            auto maxSlice = nowMax[statsOff];
            auto sumSlice = nowSum[statsOff];

            uint32_t rowsRound = RoundUp(rows, FLOAT_BLOCK_SIZE);

            // 3a: RowMax (over [0, tailN) only — CalcRowMax already limits elements).
            if (tailN == 0U) {
                // Keep the synchronization/writeback path alive for an empty
                // causal row; only the numerical contribution is zeroed.
                AscendC::Duplicate<float>(fsSlice, 0.0f, rows * nRound);
                AscendC::Duplicate<float>(maxSlice, -3.402823466e+38F, rowsRound);
                AscendC::Duplicate<float>(sumSlice, 0.0f, rowsRound);
            } else {
                CalcRowMax(fsSlice, maxSlice, rows, rowsRound, tailN, nRound);

                // 3b: Subtract rowMax and Exp (over ALL nRound elements).
                SubtractRowMaxAndExp(fsSlice, maxSlice, rows, rowsRound, nRound, nRound);

                // 3c: Zero out [tailN, nRound) — causal mask applied AFTER Exp.
                // A2 requires 32-byte aligned UB for Duplicate; unaligned tailN
                // is handled via SetVecMask + Muls(0) in ZeroOutCausalTail.
                ZeroOutCausalTail(fsSlice, rows, nRound, tailN);

                // 3d: RowSum (over ALL nRound — [tailN, nRound) is now 0).
                CalcRowSum(fsSlice, sumSlice, rows, rowsRound, nRound, nRound);
            }
        }

#if KERNEL_DUMP_PHASE1_ROWSUM
        if (dumpPhase1Rowsum && subBlockIdx_ == 0U && bi == 0U) {
            uint32_t statsCount = (m / groupRows) * RoundUp(groupRows, 8U);
            AscendC::DumpTensor(nowSum, 931, statsCount);
        }
#endif

#if KERNEL_DUMP
        if (dumpThisBatch) {
            // Each group's stats are padded to an eight-float stride in UB.
            uint32_t statsCount = (m / groupRows) * RoundUp(groupRows, 8U);
            AscendC::DumpTensor(nowMax, 920, statsCount);
            AscendC::DumpTensor(nowSum, 930, statsCount);
            // Keep the pre-scatter values distinct from the Phase2 GM readback
            // (desc 930 in CopyLseIn) so a bad value can be attributed to the
            // softmax reduction rather than the UB->GM transfer.
            AscendC::DumpTensor(nowSum, 931, statsCount);
        }
#endif

        // Step 4: DownCast float FS → 16-bit LP (RowMajor, entire tile, once).
        DownCastP(lpStage, fsStage, m, nRound);
#if KERNEL_DUMP_P
        if (dumpThisBatch) {
            AscendC::DumpTensor(lpStage, 940, m * nRound);
        }
#endif
        // Step 5: V→MTE3 sync for P, then signal CUBE that softmax is done.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
#if KERNEL_DUMP
        if (dumpThisBatch) {
            AscendC::DumpTensor(lpStage, 941, 1);
        }
#endif
        SetCrossCoreSync<0x2U, PIPE_V>(mm1ToSmFlag);
#if KERNEL_DUMP_SYNC_TRACE
        if (AscendC::GetBlockIdx() == 0U && subBlockIdx_ == 0U && bi == 0U) {
            AscendC::DumpTensor(nowSum, 932, 1);
        }
#endif
        WaitCrossCoreSync<0x2U, PIPE_MTE3>(smToMm2Flag);
#if KERNEL_DUMP_SYNC_TRACE
        if (AscendC::GetBlockIdx() == 0U && subBlockIdx_ == 0U && bi == 0U) {
            AscendC::DumpTensor(nowSum, 933, 1);
        }
#endif
        // A2: Write P to GM workspace (UB→GM DataCopyPad, MTE3).
        // CUBE will read P from GM→L1 via Nd2Nz (MTE2) after receiving smToMm2Flag.
        CopyPUbToGm(lpStage, m, nRound, bi, mCopyOffset);
#if KERNEL_DUMP_SYNC_TRACE
        if (AscendC::GetBlockIdx() == 0U && subBlockIdx_ == 0U && bi == 0U) {
            AscendC::DumpTensor(nowSum, 934, 1);
        }
#endif
        // Signal MTE3→V for next iteration (consumed by WaitFlag above).
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2U);
        SetCrossCoreSync<0x2U, PIPE_MTE3>(smToMm2Flag);
#if KERNEL_DUMP_SYNC_TRACE
        if (AscendC::GetBlockIdx() == 0U && subBlockIdx_ == 0U && bi == 0U) {
            AscendC::DumpTensor(nowSum, 935, 1);
        }
#endif
        // Re-prime the ping-pong MTE2 gate after this stage's S tile is
        // fully consumed; the same UB stage is reused after two batches.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubSBufId);
    }

private:
    float scaleValue;
    uint32_t subBlockIdx_;

    AscendC::LocalTensor<ElementInput> lsUbTensor;        // bf16 S input
    AscendC::LocalTensor<ElementOutput> lpUbTensor;       // bf16 P output
    AscendC::LocalTensor<float> fsUbTensor;               // float S work buffer
    AscendC::LocalTensor<float> tvUbTensor;               // temp for reduce
    AscendC::LocalTensor<float> gmUbTensor[SM_UB_STAGES]; // rowMax stats
    AscendC::LocalTensor<float> glUbTensor[SM_UB_STAGES]; // rowSum stats
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_MSA_SPLIT_KV_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_A2_HPP
