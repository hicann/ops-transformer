/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * innerPrecise==0 (ALL_HIGH): fp32 S from QK NoQuant fixpipe, fp32 softmax, P cast to bf16 zN.
 * Per-block local softmax (no cross-block UpdateMax); Phase2 FlashDecode combines blocks.
 * operator() signature matches the bf16 regbase softmax so the kernel call site is shared.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/gemm/gemm_type.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <class OutputType_, class LayoutS_>
class BlockEpilogue<EpilogueOnlineSoftmaxBsa, OutputType_, Gemm::GemmType<float, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = float;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_UB_STAGES = 2;
    static constexpr float NEG_INF = -3.402823466e+38f;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // fp32 S is 2x the bf16 footprint: 2 stages * 16384 * 4B = 128KB at offset 0.
        // bf16 P zN sits after S; stats stay at 7*32KB so kernel SM_UB_GM_OFFSET matches.
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;  // 131072
        constexpr uint32_t TMP_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE; // 196608, 32KB scratch
        constexpr uint32_t STATS_UB_STAGE_BYTES = SM_ROW_MAX_ELEM_NUM * sizeof(float);
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE; // 229376
        constexpr uint32_t GL_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 2 * STATS_UB_STAGE_BYTES;

        subBlockIdx_ = AscendC::GetSubBlockIdx();
        scaleValue = scaleValue_;

        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        tmpUbTensor = resource.ubBuf.template GetBufferByByte<float>(TMP_UB_TENSOR_OFFSET);
        ndPUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(TMP_UB_TENSOR_OFFSET);
        for (uint32_t i = 0; i < SM_UB_STAGES; i++) {
            gmUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET + i * STATS_UB_STAGE_BYTES);
            glUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET + i * STATS_UB_STAGE_BYTES);
        }
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t m)
    {
        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        AscendC::DataCopyParams repeatParams;
        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = m;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - m;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - m;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorP>
    __aicore__ inline void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape, uint32_t ubSBufId,
                                      uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag,
                                      Arch::CrossCoreFlag smToMm2Flag, const uint32_t *causalValidLens,
                                      uint32_t groupCount, uint32_t groupRows)
    {
        (void)l1PBufId;
        uint32_t M = actualBlockShape.m();
        // AIV0-only: NoQuant Fixpipe cannot split to AIV1 (subBlockId=1 MTE exception,
        // dualDstCtl=1 CrossCore deadlock). AIV1 only keeps the flag handshake.
        uint32_t m = (subBlockIdx_ == 0) ? M : 0U;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }
        uint32_t n = actualBlockShape.n();
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t startRow = 0;
        uint32_t endRow = m;

        auto sBase = lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM];
        auto pZnBase = lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM];
        auto nowMax = gmUbTensor[ubSBufId];
        auto nowSum = glUbTensor[ubSBufId];

        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);

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
            auto sSlice = sBase[localOff * nRound];
            auto maxSlice = nowMax[statsOff];
            auto sumSlice = nowSum[statsOff];
            ComputeLocalSoftmax(sSlice, maxSlice, sumSlice, rows, nRound, tailN);
        }

        // fp32 P sits in S; cast to bf16 ND in scratch, then ND->zN into lpUb.
        AscendC::Cast(ndPUbTensor, sBase, AscendC::RoundMode::CAST_RINT, m * nRound);
        AscendC::PipeBarrier<PIPE_V>();
        // A5 cannot store a 64-row zN from AIV0: VF DATA_BLOCK_COPY with
        // blockStride=64 traps (271), and 1D UB2UB into mRound=64 leaves a
        // hole at M-row 36 (token 2 / head 4 on the example). innerPrecise=4
        // is correct because each AIV writes a 32-row zN (blockStride=32)
        // and CopyPUbToPL1 scatters it into the full L1 zN (dstStride =
        // l1MRound - 32). Replicate that geometry here: packed 32-row zN
        // tiles in UB, then one UB->L1 copy per tile.
        constexpr uint32_t P_ZN_CHUNK = 32U;
        uint32_t znPacked = 0;
        for (uint32_t row0 = 0; row0 < m; row0 += P_ZN_CHUNK) {
            uint32_t rows = (m - row0 > P_ZN_CHUNK) ? P_ZN_CHUNK : (m - row0);
            uint16_t mR = RoundUp(rows, C0_NUM_PER_FRACTAL);
            CopyPNdToNz(pZnBase[znPacked], ndPUbTensor[row0 * nRound], rows, mR, nRound);
            znPacked += static_cast<uint32_t>(mR) * nRound;
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        znPacked = 0;
        for (uint32_t row0 = 0; row0 < m; row0 += P_ZN_CHUNK) {
            uint32_t rows = (m - row0 > P_ZN_CHUNK) ? P_ZN_CHUNK : (m - row0);
            uint16_t mR = RoundUp(rows, C0_NUM_PER_FRACTAL);
            auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mR, nRound);
            auto ubPTensorTla = tla::MakeTensor(pZnBase[znPacked], ubPLayoutTla, Arch::PositionUB{});
            auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(rows, n));
            auto l1PTensorTlaTile = GetTile(l1PTensorTla, tla::MakeCoord(row0, 0), tla::MakeShape(rows, n));
            CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, rows);
            znPacked += static_cast<uint32_t>(mR) * nRound;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    float scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<float> tmpUbTensor;
    AscendC::LocalTensor<ElementOutput> ndPUbTensor;
    AscendC::LocalTensor<float> gmUbTensor[SM_UB_STAGES];
    AscendC::LocalTensor<float> glUbTensor[SM_UB_STAGES];
    uint32_t subBlockIdx_;

    __aicore__ inline void CopyPNdToNz(const AscendC::LocalTensor<ElementOutput> &zn,
                                       const AscendC::LocalTensor<ElementOutput> &nd, uint32_t m, uint32_t mRound,
                                       uint32_t nRound)
    {
        // 1D UB2UB copies only (16 bf16 = 32B). A5 traps on 2D UB gather
        // (blockCount=m, srcStride=nC0-1: drops 2 rows) and on 2D UB scatter
        // (dstStride=mRound-1: scalar UB OOB, 507015). Also traps: vector
        // Adds TransND2NZ (dstRepStride=1 over M) and VF DATA_BLOCK_COPY with
        // blockStride=mRound=64 (errcode 271). zN: C0 k of row i sits at
        // zn[k * mRound * C0 + i * C0], matching VF DATA_BLOCK_COPY geometry.
        const uint32_t nC0 = nRound / ELE_NUM_PER_C0;
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t k = 0; k < nC0; ++k) {
                AscendC::DataCopy(zn[k * mRound * ELE_NUM_PER_C0 + i * ELE_NUM_PER_C0],
                                  nd[i * nRound + k * ELE_NUM_PER_C0], ELE_NUM_PER_C0);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void MaskCausalTail(const AscendC::LocalTensor<float> &sLocal, uint32_t rows, uint32_t nRound,
                                          uint32_t tailN)
    {
        if (tailN >= nRound) {
            return;
        }
        auto maskRow = tmpUbTensor;
        AscendC::Duplicate(maskRow, 0.0f, nRound);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t aligned = RoundUp(tailN, FLOAT_BLOCK_SIZE);
        if (aligned > nRound) {
            aligned = nRound;
        }
        if (tailN < aligned) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (uint32_t c = tailN; c < aligned; ++c) {
                maskRow.SetValue(c, NEG_INF);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        }
        if (aligned < nRound) {
            AscendC::Duplicate(maskRow[aligned], NEG_INF, nRound - aligned);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::BinaryRepeatParams addRp;
        addRp.dstBlkStride = 1;
        addRp.src0BlkStride = 1;
        addRp.src1BlkStride = 1;
        addRp.dstRepStride = nRound / FLOAT_BLOCK_SIZE;
        addRp.src0RepStride = nRound / FLOAT_BLOCK_SIZE;
        addRp.src1RepStride = 0;
        uint32_t col = 0;
        while (col + FLOAT_VECTOR_SIZE <= nRound) {
            AscendC::Add(sLocal[col], sLocal[col], maskRow[col], FLOAT_VECTOR_SIZE, rows, addRp);
            col += FLOAT_VECTOR_SIZE;
        }
        if (col < nRound) {
            AscendC::Add(sLocal[col], sLocal[col], maskRow[col], nRound - col, rows, addRp);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowReduceMax(const AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> src,
                                        uint32_t rows, uint32_t columnCount)
    {
        uint32_t dtypeMask = FLOAT_VECTOR_SIZE;
        uint32_t blockCount = columnCount / dtypeMask;
        uint32_t remain = columnCount % dtypeMask;
        AscendC::BinaryRepeatParams rp;
        rp.src0BlkStride = 1;
        rp.src1BlkStride = 1;
        rp.dstBlkStride = 1;
        rp.src0RepStride = columnCount / FLOAT_BLOCK_SIZE;
        rp.src1RepStride = columnCount / FLOAT_BLOCK_SIZE;
        rp.dstRepStride = columnCount / FLOAT_BLOCK_SIZE;
        if (blockCount > 0 && remain > 0) {
            AscendC::Max(src, src, src[blockCount * dtypeMask], remain, rows, rp);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t loopCount = blockCount / 2U; loopCount > 0; loopCount = blockCount / 2U) {
            blockCount = (blockCount + 1U) / 2U;
            for (uint32_t j = 0; j < loopCount; j++) {
                AscendC::Max(src[j * dtypeMask], src[j * dtypeMask], src[(j + blockCount) * dtypeMask], dtypeMask, rows,
                             rp);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::WholeReduceMax(dst, src, (columnCount < dtypeMask) ? columnCount : dtypeMask, rows, 1, 1,
                                columnCount / FLOAT_BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowReduceSum(const AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> src,
                                        uint32_t rows, uint32_t columnCount)
    {
        uint32_t dtypeMask = FLOAT_VECTOR_SIZE;
        uint32_t blockCount = columnCount / dtypeMask;
        uint32_t remain = columnCount % dtypeMask;
        AscendC::BinaryRepeatParams rp;
        rp.src0BlkStride = 1;
        rp.src1BlkStride = 1;
        rp.dstBlkStride = 1;
        rp.src0RepStride = columnCount / FLOAT_BLOCK_SIZE;
        rp.src1RepStride = columnCount / FLOAT_BLOCK_SIZE;
        rp.dstRepStride = columnCount / FLOAT_BLOCK_SIZE;
        if (blockCount > 0 && remain > 0) {
            AscendC::Add(src, src, src[blockCount * dtypeMask], remain, rows, rp);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t loopCount = blockCount / 2U; loopCount > 0; loopCount = blockCount / 2U) {
            blockCount = (blockCount + 1U) / 2U;
            for (uint32_t j = 0; j < loopCount; j++) {
                AscendC::Add(src[j * dtypeMask], src[j * dtypeMask], src[(j + blockCount) * dtypeMask], dtypeMask, rows,
                             rp);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::WholeReduceSum(dst, src, (columnCount < dtypeMask) ? columnCount : dtypeMask, rows, 1, 1,
                                columnCount / FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SubBroadcastMax(const AscendC::LocalTensor<float> &sLocal,
                                           const AscendC::LocalTensor<float> &maxLocal, uint32_t rows, uint32_t nRound)
    {
        AscendC::Brcb(tmpUbTensor, maxLocal, static_cast<uint8_t>((rows + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE),
                      {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams subRp;
        subRp.dstBlkStride = 1;
        subRp.src0BlkStride = 1;
        subRp.src1BlkStride = 0;
        subRp.dstRepStride = nRound / FLOAT_BLOCK_SIZE;
        subRp.src0RepStride = nRound / FLOAT_BLOCK_SIZE;
        subRp.src1RepStride = 1;
        uint32_t col = 0;
        while (col + FLOAT_VECTOR_SIZE <= nRound) {
            AscendC::Sub(sLocal[col], sLocal[col], tmpUbTensor, FLOAT_VECTOR_SIZE, rows, subRp);
            col += FLOAT_VECTOR_SIZE;
        }
        if (col < nRound) {
            AscendC::Sub(sLocal[col], sLocal[col], tmpUbTensor, nRound - col, rows, subRp);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeLocalSoftmax(const AscendC::LocalTensor<float> &sLocal,
                                               const AscendC::LocalTensor<float> &maxLocal,
                                               const AscendC::LocalTensor<float> &sumLocal, uint32_t rows,
                                               uint32_t nRound, uint32_t tailN)
    {
        uint32_t elemNum = rows * nRound;
        uint32_t statsAlign = RoundUp(rows, FLOAT_BLOCK_SIZE);
        if (tailN == 0U) {
            AscendC::Duplicate(sLocal, 0.0f, elemNum);
            AscendC::Duplicate(maxLocal, NEG_INF, statsAlign);
            AscendC::Duplicate(sumLocal, 0.0f, statsAlign);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        AscendC::Muls(sLocal, sLocal, scaleValue, elemNum);
        AscendC::PipeBarrier<PIPE_V>();
        MaskCausalTail(sLocal, rows, nRound, tailN);

        AscendC::DataCopy(tmpUbTensor, sLocal, elemNum);
        AscendC::PipeBarrier<PIPE_V>();
        RowReduceMax(maxLocal, tmpUbTensor, rows, nRound);

        SubBroadcastMax(sLocal, maxLocal, rows, nRound);
        AscendC::Exp(sLocal, sLocal, elemNum);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::DataCopy(tmpUbTensor, sLocal, elemNum);
        AscendC::PipeBarrier<PIPE_V>();
        RowReduceSum(sumLocal, tmpUbTensor, rows, nRound);
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_HPP
