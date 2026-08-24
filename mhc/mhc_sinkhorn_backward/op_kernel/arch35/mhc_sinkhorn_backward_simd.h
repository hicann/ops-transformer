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
 * \file mhc_sinkhorn_backward_simd.h
 * \brief mhc_sinkhorn_backward
 */

#ifndef ASCENDC_MHC_SINKHORN_BACKWARD_SIMD_H
#define ASCENDC_MHC_SINKHORN_BACKWARD_SIMD_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "mhc_sinkhorn_backward_tiling_data.h"
#include "mhc_sinkhorn_backward_tiling_key.h"

namespace MhcSinkhornBackward {
using namespace AscendC;

constexpr int64_t MASK_BUFFER_SIZE = 64;
constexpr int64_t TEMP_BUFFER_SIZE = 256;
constexpr uint32_t MASK_4 = 0b00000000000000001111111111111111;
constexpr uint32_t MASK_6 = 0b00000000111111111111111111111111;
constexpr uint32_t MASK_8 = 0b11111111111111111111111111111111;
constexpr int64_t MASK_NUM = 8;
constexpr int64_t INDEX_BLOCK_LEN = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

class MhcSinkhornBackwardSimd {
public:
    __aicore__ inline MhcSinkhornBackwardSimd(TPipe &pipe, const MhcSinkhornBackwardTilingData &tilingData)
        : pipe_(pipe),
          tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR norm, GM_ADDR sum, GM_ADDR gradInput, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcSoftmaxBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                               __local_mem__ float *gradInputAddr, __local_mem__ uint32_t *maskAddr,
                                               __local_mem__ float *tempAddr, uint32_t n, uint16_t repeatTimes,
                                               uint32_t perloopSize, uint32_t repeatSize);
    __aicore__ inline void CalcColBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ uint32_t *maskAddr,
                                           __local_mem__ float *tempAddr, uint32_t n, uint16_t repeatTimes,
                                           uint32_t perloopSize, uint32_t repeatSize);
    __aicore__ inline void CalcRowBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ uint32_t *maskAddr,
                                           __local_mem__ float *tempAddr, uint32_t n, uint16_t repeatTimes,
                                           uint32_t perloopSize, uint32_t perloopSizePad, uint32_t repeatSize);

    __aicore__ inline void CopyInNormAndSumPad(const uint64_t normOffset, const uint32_t normProcessNum,
                                               const uint64_t sumOffset, const uint32_t sumProcessNum,
                                               uint32_t perloopSize);
    __aicore__ inline void CopyInNormAndSum(const uint64_t normOffset, const uint32_t normProcessNum,
                                            const uint64_t sumOffset, const uint32_t sumProcessNum);
    __aicore__ inline void CopyInGradY(const uint64_t gradYOffset, const uint32_t gradYProcessNum);
    __aicore__ inline void CopyInNorm(const uint64_t normOffset, const uint32_t normProcessNum);
    __aicore__ inline void CopyOut(const uint64_t gradInputOffset, const uint32_t gradInputProcessNum);

    __aicore__ inline void CalcCol(uint32_t repeatSize, uint16_t repeatTimes, int64_t colOffset, int64_t loopIdx,
                                   uint32_t perloopSize);
    __aicore__ inline void CalcRow(uint32_t repeatSize, uint16_t repeatTimes, int64_t rowOffset, int64_t loopIdx,
                                   uint32_t perloopSize);
    __aicore__ inline void CalcSoftmax(uint32_t repeatSize, uint16_t repeatTimes, int64_t loopIdx,
                                       uint32_t perloopSize);
    __aicore__ inline void CreateGatherIndex(uint32_t n);

    GlobalTensor<float> gradY_;
    GlobalTensor<float> norm_;
    GlobalTensor<float> sum_;
    GlobalTensor<float> gradInput_;
    TPipe &pipe_;
    const MhcSinkhornBackwardTilingData &tilingData_;
    int64_t blockIdx_;      // 核id
    int64_t loop_;          // 循环次数,单次处理normCorePerLoopTSize个矩阵，需要loop_次
    int64_t loopTSize_;     // 整核/尾核 整块处理的矩阵数量
    int64_t tailLoopTSize_; // 整核/尾核 尾块处理的矩阵数量
    int64_t nSize_;
    int64_t nAlignSize_;
    int64_t num_iters;
    TQue<QuePosition::VECIN, 1> inputGradYQue_;
    TQue<QuePosition::VECIN, 1> inputNormQue_;
    TQue<QuePosition::VECIN, 1> inputSumQue_;
    TQue<QuePosition::VECOUT, 1> outputQue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> tempBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> broadcastBuffer_;
};

__aicore__ inline void MhcSinkhornBackwardSimd::Init(GM_ADDR gradY, GM_ADDR norm, GM_ADDR sum, GM_ADDR gradInput,
                                                     GM_ADDR tiling)
{
    blockIdx_ = GetBlockIdx();
    gradY_.SetGlobalBuffer((__gm__ float *)gradY);
    norm_.SetGlobalBuffer((__gm__ float *)norm);
    sum_.SetGlobalBuffer((__gm__ float *)sum);
    gradInput_.SetGlobalBuffer((__gm__ float *)gradInput);

    num_iters = tilingData_.numIters;
    // num_iters = 1;
    nSize_ = tilingData_.nSize;
    nAlignSize_ = tilingData_.nAlignSize;

    // 如果是主核的话
    loop_ = tilingData_.normCoreTLoops;                 // 循环次数
    loopTSize_ = tilingData_.normCorePerLoopTSize;      // 主块T
    tailLoopTSize_ = tilingData_.normCoreLastLoopTSize; // 尾块T
    // 如果是尾核的话
    if (blockIdx_ == tilingData_.needCoreNum - 1) {
        loop_ = tilingData_.tailCoreLoops;
        loopTSize_ = tilingData_.tailCorePerLoopTSize;
        tailLoopTSize_ = tilingData_.tailCoreLastLoopTSize;
    }

    pipe_.InitBuffer(maskBuffer_, MASK_BUFFER_SIZE);
    pipe_.InitBuffer(tempBuffer_, TEMP_BUFFER_SIZE);
    pipe_.InitBuffer(indexBuffer_, TEMP_BUFFER_SIZE);
    pipe_.InitBuffer(broadcastBuffer_, TEMP_BUFFER_SIZE);
    pipe_.InitBuffer(inputGradYQue_, DOUBLE_BUFFER, tilingData_.normCorePerLoopTSize * nSize_ * nSize_ * sizeof(float));
    pipe_.InitBuffer(inputNormQue_, DOUBLE_BUFFER,
                     tilingData_.normCorePerLoopTSize * nSize_ * nAlignSize_ * sizeof(float));
    int64_t sumFlatSize = tilingData_.normCorePerLoopTSize * nAlignSize_;
    int64_t sumPadSize = nSize_ * Ceil(tilingData_.normCorePerLoopTSize, INDEX_BLOCK_LEN) * INDEX_BLOCK_LEN;
    int64_t sumBufSize = sumFlatSize > sumPadSize ? sumFlatSize : sumPadSize;
    pipe_.InitBuffer(inputSumQue_, DOUBLE_BUFFER, sumBufSize * sizeof(float));
    pipe_.InitBuffer(outputQue_, DOUBLE_BUFFER, tilingData_.normCorePerLoopTSize * nSize_ * nSize_ * sizeof(float));
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcRowBackward(
    __local_mem__ float *gradYAddr, __local_mem__ float *normAddr, __local_mem__ float *sumAddr,
    __local_mem__ uint32_t *maskAddr, __local_mem__ float *tempAddr, uint32_t n, uint16_t repeatTimes,
    uint32_t perloopSize, uint32_t perloopSizePad, uint32_t repeatSize)
{
    LocalTensor<int32_t> indexLocal = indexBuffer_.Get<int32_t>();
    __local_mem__ int32_t *indexAddr = (__local_mem__ int32_t *)indexLocal.GetPhyAddr();
    LocalTensor<int32_t> broadcastLocal = broadcastBuffer_.Get<int32_t>();
    __local_mem__ int32_t *broadcastAddr = (__local_mem__ int32_t *)broadcastLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg allMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::LoadAlign(maskReg, maskAddr); // VL

        MicroAPI::RegTensor<float> gradYReg;
        MicroAPI::RegTensor<float> normReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<float> dotProdReg;
        MicroAPI::RegTensor<float> sumGradYReg;
        MicroAPI::RegTensor<float> sumBrcbReg;
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> broadcastReg;

        MicroAPI::LoadAlign(indexReg, indexAddr);         // VL
        MicroAPI::LoadAlign(broadcastReg, broadcastAddr); // VL
        for (uint16_t j = 0; j < repeatTimes; j++) {
            uint32_t perT = min(repeatSize, perloopSize - (repeatSize * j));
            uint32_t dataLen = perT * static_cast<uint32_t>(nAlignSize_);
            MicroAPI::MaskReg CalcMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen); // dataLen
            MicroAPI::And(CalcMaskReg, CalcMaskReg, maskReg, allMask);
            MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(perT);
            for (uint16_t k = 0; k < static_cast<uint16_t>(nSize_); k++) {
                // 根据索引位置将输入gradYAddr对应元素搬入gradYReg
                MicroAPI::DataCopyGather(gradYReg, gradYAddr + j * repeatSize * nSize_ * nSize_ + k * nSize_,
                                         (MicroAPI::RegTensor<uint32_t> &)(indexReg), CalcMaskReg);
                MicroAPI::DataCopy(normReg, normAddr + k * perloopSize * nAlignSize_ + j * repeatSize * nAlignSize_);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(
                    sumReg, sumAddr + k * perloopSizePad + j * repeatSize);

                MicroAPI::Mul(dotProdReg, gradYReg, normReg, CalcMaskReg);
                MicroAPI::ReduceSumWithDataBlock(sumGradYReg, dotProdReg, CalcMaskReg);
                MicroAPI::Gather(sumBrcbReg, sumGradYReg, (MicroAPI::RegTensor<uint32_t> &)(broadcastReg));
                MicroAPI::Sub(dotProdReg, gradYReg, sumBrcbReg, CalcMaskReg);
                MicroAPI::Div(dotProdReg, dotProdReg, sumReg, CalcMaskReg);

                MicroAPI::DataCopyScatter(gradYAddr + j * repeatSize * nSize_ * nSize_ + k * nSize_, dotProdReg,
                                          (MicroAPI::RegTensor<uint32_t> &)(indexReg), CalcMaskReg);
            }
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcColBackward(
    __local_mem__ float *gradYAddr, __local_mem__ float *normAddr, __local_mem__ float *sumAddr,
    __local_mem__ uint32_t *maskAddr, __local_mem__ float *tempAddr, uint32_t n, uint16_t repeatTimes,
    uint32_t perloopSize, uint32_t repeatSize)
{
    LocalTensor<int32_t> indexLocal = indexBuffer_.Get<int32_t>();
    __local_mem__ int32_t *indexAddr = (__local_mem__ int32_t *)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg allMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::LoadAlign(maskReg, maskAddr); // VL

        MicroAPI::RegTensor<float> gradYReg;
        MicroAPI::RegTensor<float> gradInputReg;
        MicroAPI::RegTensor<float> normReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<float> mulResultReg;
        MicroAPI::RegTensor<float> sumColReg;
        MicroAPI::RegTensor<int32_t> indexReg;

        MicroAPI::LoadAlign(indexReg, indexAddr); // VL

        for (uint16_t j = 0; j < repeatTimes; j++) {
            uint32_t perT = min(repeatSize, perloopSize - (repeatSize * j));
            uint32_t dataLen = perT * static_cast<uint32_t>(nAlignSize_);

            MicroAPI::MaskReg CalcMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen); // dataLen
            MicroAPI::And(CalcMaskReg, CalcMaskReg, maskReg, allMask);
            MicroAPI::Duplicate(sumColReg, static_cast<float>(0));
            MicroAPI::DataCopy(sumReg, sumAddr + j * repeatSize * nAlignSize_);
            for (uint16_t i = 0; i < static_cast<uint16_t>(n); i++) {
                MicroAPI::DataCopyGather(gradYReg, gradYAddr + j * repeatSize * nSize_ * nSize_ + i * n,
                                         (MicroAPI::RegTensor<uint32_t> &)(indexReg), CalcMaskReg);
                MicroAPI::DataCopy(normReg, normAddr + j * repeatSize * nAlignSize_ + i * nAlignSize_ * perloopSize);
                MicroAPI::Mul(mulResultReg, gradYReg, normReg, CalcMaskReg);
                MicroAPI::Add(sumColReg, sumColReg, mulResultReg, CalcMaskReg);
            }
            for (uint16_t i = 0; i < static_cast<uint16_t>(n); i++) {
                MicroAPI::DataCopyGather(gradYReg, gradYAddr + j * repeatSize * nSize_ * nSize_ + i * n,
                                         (MicroAPI::RegTensor<uint32_t> &)(indexReg), CalcMaskReg);
                MicroAPI::Sub(gradInputReg, gradYReg, sumColReg, CalcMaskReg);
                MicroAPI::Div(gradInputReg, gradInputReg, sumReg, CalcMaskReg);
                MicroAPI::DataCopyScatter(gradYAddr + j * repeatSize * nSize_ * nSize_ + i * n, gradInputReg,
                                          (MicroAPI::RegTensor<uint32_t> &)(indexReg), CalcMaskReg);
            }
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcSoftmaxBackward(
    __local_mem__ float *gradYAddr, __local_mem__ float *normAddr, __local_mem__ float *gradInputAddr,
    __local_mem__ uint32_t *maskAddr, __local_mem__ float *tempAddr, uint32_t n, uint16_t repeatTimes,
    uint32_t perloopSize, uint32_t repeatSize)
{
    LocalTensor<int32_t> indexLocal = indexBuffer_.Get<int32_t>();
    __local_mem__ int32_t *indexAddr = (__local_mem__ int32_t *)indexLocal.GetPhyAddr();
    LocalTensor<int32_t> broadcastLocal = broadcastBuffer_.Get<int32_t>();
    __local_mem__ int32_t *broadcastAddr = (__local_mem__ int32_t *)broadcastLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg allMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::LoadAlign(maskReg, maskAddr); // VL

        MicroAPI::RegTensor<float> gradYReg;
        MicroAPI::RegTensor<float> normReg;
        MicroAPI::RegTensor<float> mulResultReg;
        MicroAPI::RegTensor<float> dotProdReg;
        MicroAPI::RegTensor<float> dotProdBrcbReg;
        MicroAPI::RegTensor<float> gradInputReg;
        MicroAPI::RegTensor<int32_t> broadcastReg;

        MicroAPI::RegTensor<int32_t> indexReg;

        MicroAPI::LoadAlign(indexReg, indexAddr);         // VL
        MicroAPI::LoadAlign(broadcastReg, broadcastAddr); // VL

        for (uint16_t k = 0; k < static_cast<uint16_t>(nSize_); k++) {
            for (uint16_t j = 0; j < repeatTimes; j++) {
                uint32_t perT = min(repeatSize, perloopSize - (repeatSize * j));
                uint32_t dataLen = perT * static_cast<uint32_t>(nAlignSize_);

                MicroAPI::MaskReg CalcMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen); // dataLen
                MicroAPI::And(CalcMaskReg, CalcMaskReg, maskReg, allMask);
                MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(perT);

                // 按照indeReg中存储索引的顺序将数据搬到gradYReg,此时gradYReg [n, t, align_n]
                MicroAPI::DataCopyGather(gradYReg, gradYAddr + j * repeatSize * nSize_ * nSize_ + k * nSize_,
                                         (MicroAPI::RegTensor<uint32_t> &)(indexReg), CalcMaskReg);

                MicroAPI::DataCopy(normReg, normAddr + k * perloopSize * nAlignSize_ + j * repeatSize * nAlignSize_);

                MicroAPI::Mul(mulResultReg, gradYReg, normReg, CalcMaskReg);
                MicroAPI::ReduceSumWithDataBlock(dotProdReg, mulResultReg, CalcMaskReg);
                MicroAPI::Gather(dotProdBrcbReg, dotProdReg, (MicroAPI::RegTensor<uint32_t> &)(broadcastReg));
                MicroAPI::Sub(gradInputReg, gradYReg, dotProdBrcbReg, CalcMaskReg);
                MicroAPI::Mul(gradInputReg, gradInputReg, normReg, CalcMaskReg);

                MicroAPI::DataCopyScatter(gradInputAddr + +j * repeatSize * nSize_ * nSize_ + k * nSize_, gradInputReg,
                                          (MicroAPI::RegTensor<uint32_t> &)(indexReg), CalcMaskReg);
            }
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::CopyInNormAndSumPad(const uint64_t normOffset,
                                                                    const uint32_t normProcessNum,
                                                                    const uint64_t sumOffset,
                                                                    const uint32_t sumProcessNum, uint32_t perloopSize)
{
    LocalTensor<float> normLocal = inputNormQue_.AllocTensor<float>();
    LocalTensor<float> sumLocal = inputSumQue_.AllocTensor<float>();

    DataCopyPadExtParams<float> normPadExtParams{false, 0, 0, 0};
    DataCopyExtParams normParams{1, static_cast<uint32_t>(normProcessNum * sizeof(float)), 0, 0, 0};
    uint8_t padNum = static_cast<uint8_t>(INDEX_BLOCK_LEN - perloopSize % INDEX_BLOCK_LEN);
    DataCopyPadExtParams<float> sumPadExtParams{1, 0, padNum, 0};
    DataCopyExtParams sumParams{static_cast<uint16_t>(nSize_), static_cast<uint32_t>(perloopSize * sizeof(float)), 0, 0,
                                0};

    DataCopyPad(normLocal, norm_[normOffset], normParams, normPadExtParams);
    DataCopyPad(sumLocal, sum_[sumOffset], sumParams, sumPadExtParams);

    inputNormQue_.EnQue(normLocal);
    inputSumQue_.EnQue(sumLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::CopyInNormAndSum(const uint64_t normOffset,
                                                                 const uint32_t normProcessNum,
                                                                 const uint64_t sumOffset, const uint32_t sumProcessNum)
{
    LocalTensor<float> normLocal = inputNormQue_.AllocTensor<float>();
    LocalTensor<float> sumLocal = inputSumQue_.AllocTensor<float>();

    DataCopyPadExtParams<float> PadExtParams{false, 0, 0, 0};
    DataCopyExtParams normParams{1, static_cast<uint32_t>(normProcessNum * sizeof(float)), 0, 0, 0};
    DataCopyExtParams sumParams{1, static_cast<uint32_t>(sumProcessNum * sizeof(float)), 0, 0, 0};

    DataCopyPad(normLocal, norm_[normOffset], normParams, PadExtParams);
    DataCopyPad(sumLocal, sum_[sumOffset], sumParams, PadExtParams);

    inputNormQue_.EnQue(normLocal);
    inputSumQue_.EnQue(sumLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::CopyInGradY(const uint64_t gradYOffset, const uint32_t gradYProcessNum)
{
    LocalTensor<float> gradYLocal = inputGradYQue_.AllocTensor<float>();
    DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
    DataCopyExtParams gradYParams{1, static_cast<uint32_t>(gradYProcessNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(gradYLocal, gradY_[gradYOffset], gradYParams, dataCopyPadExtParams);
    inputGradYQue_.EnQue(gradYLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::CopyInNorm(const uint64_t normOffset, const uint32_t normProcessNum)
{
    LocalTensor<float> normLocal = inputNormQue_.AllocTensor<float>();
    DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
    DataCopyExtParams normParams = {1, static_cast<uint32_t>(normProcessNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(normLocal, norm_[normOffset], normParams, dataCopyPadExtParams);
    inputNormQue_.EnQue(normLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::CopyOut(const uint64_t gradInputOffset,
                                                        const uint32_t gradInputProcessNum)
{
    LocalTensor<float> gradInputLocal = outputQue_.DeQue<float>();
    DataCopyExtParams gradInputParams{1, static_cast<uint32_t>(gradInputProcessNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(gradInput_[gradInputOffset], gradInputLocal, gradInputParams);
    outputQue_.FreeTensor<float>(gradInputLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcCol(uint32_t repeatSize, uint16_t repeatTimes, int64_t colOffset,
                                                        int64_t loopIdx, uint32_t perloopSize)
{
    uint64_t normOffset = blockIdx_ * nSize_ * nAlignSize_ * num_iters * 2 * tilingData_.coreTSize +
                          loopIdx * loopTSize_ * nSize_ * nAlignSize_ * num_iters * 2 +
                          colOffset * nSize_ * nAlignSize_ * perloopSize;
    uint64_t sumOffset = blockIdx_ * nAlignSize_ * num_iters * 2 * tilingData_.coreTSize +
                         loopIdx * loopTSize_ * nAlignSize_ * num_iters * 2 + colOffset * nAlignSize_ * perloopSize;
    uint32_t normSize = perloopSize * nSize_ * nAlignSize_;
    uint32_t sumSize = perloopSize * nAlignSize_;
    CopyInNormAndSum(normOffset, normSize, sumOffset, sumSize);

    LocalTensor<float> gradYLocal = inputGradYQue_.DeQue<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    LocalTensor<uint32_t> tempLocal = tempBuffer_.Get<uint32_t>();
    __local_mem__ float *gradYAddr = (__local_mem__ float *)gradYLocal.GetPhyAddr();
    __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();
    __local_mem__ float *tempAddr = (__local_mem__ float *)tempLocal.GetPhyAddr();

    LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
    LocalTensor<float> sumLocal = inputSumQue_.DeQue<float>();
    __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
    __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();

    CalcColBackward(gradYAddr, normAddr, sumAddr, maskAddr, tempAddr, static_cast<uint32_t>(nSize_), repeatTimes,
                    perloopSize, repeatSize);

    inputGradYQue_.EnQue(gradYLocal);
    inputNormQue_.FreeTensor<float>(normLocal);
    inputSumQue_.FreeTensor<float>(sumLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcRow(uint32_t repeatSize, uint16_t repeatTimes, int64_t rowOffset,
                                                        int64_t loopIdx, uint32_t perloopSize)
{
    uint64_t normOffset = blockIdx_ * nSize_ * nAlignSize_ * num_iters * 2 * tilingData_.coreTSize +
                          loopIdx * loopTSize_ * nSize_ * nAlignSize_ * num_iters * 2 +
                          rowOffset * nSize_ * nAlignSize_ * perloopSize;
    uint64_t sumOffset = blockIdx_ * nAlignSize_ * num_iters * 2 * tilingData_.coreTSize +
                         loopIdx * loopTSize_ * nAlignSize_ * num_iters * 2 + rowOffset * nAlignSize_ * perloopSize;
    uint32_t normSize = perloopSize * nSize_ * nAlignSize_;
    uint32_t sumSize = perloopSize * nAlignSize_;
    uint32_t perloopSizePad = perloopSize;
    if (perloopSize % 8 == 0) {
        CopyInNormAndSum(normOffset, normSize, sumOffset, sumSize);
    } else {
        CopyInNormAndSumPad(normOffset, normSize, sumOffset, sumSize, perloopSize);
        perloopSizePad = (perloopSize + 7) / 8 * 8;
    }

    LocalTensor<float> gradYLocal = inputGradYQue_.DeQue<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    LocalTensor<uint32_t> tempLocal = tempBuffer_.Get<uint32_t>();
    __local_mem__ float *gradYAddr = (__local_mem__ float *)gradYLocal.GetPhyAddr();
    __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();
    __local_mem__ float *tempAddr = (__local_mem__ float *)tempLocal.GetPhyAddr();

    LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
    LocalTensor<float> sumLocal = inputSumQue_.DeQue<float>();
    __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
    __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();

    CalcRowBackward(gradYAddr, normAddr, sumAddr, maskAddr, tempAddr, static_cast<uint32_t>(nSize_), repeatTimes,
                    perloopSize, perloopSizePad, repeatSize);

    inputGradYQue_.EnQue(gradYLocal);
    inputNormQue_.FreeTensor<float>(normLocal);
    inputSumQue_.FreeTensor<float>(sumLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcSoftmax(uint32_t repeatSize, uint16_t repeatTimes, int64_t loopIdx,
                                                            uint32_t perloopSize)
{
    uint64_t normOffset = blockIdx_ * nSize_ * nAlignSize_ * num_iters * 2 * tilingData_.coreTSize +
                          loopIdx * loopTSize_ * nSize_ * nAlignSize_ * num_iters * 2;
    uint32_t normSize = perloopSize * nSize_ * nAlignSize_;
    CopyInNorm(normOffset, normSize);

    LocalTensor<float> gradYLocal = inputGradYQue_.DeQue<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    LocalTensor<uint32_t> tempLocal = tempBuffer_.Get<uint32_t>();
    LocalTensor<float> gradInputLocal = outputQue_.AllocTensor<float>();
    __local_mem__ float *gradYAddr = (__local_mem__ float *)gradYLocal.GetPhyAddr();
    __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();
    __local_mem__ float *tempAddr = (__local_mem__ float *)tempLocal.GetPhyAddr();
    __local_mem__ float *gradInputAddr = (__local_mem__ float *)gradInputLocal.GetPhyAddr();

    LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
    __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();

    CalcSoftmaxBackward(gradYAddr, normAddr, gradInputAddr, maskAddr, tempAddr, static_cast<uint32_t>(nSize_),
                        repeatTimes, perloopSize, repeatSize);

    outputQue_.EnQue(gradInputLocal);
    inputGradYQue_.FreeTensor<float>(gradYLocal);
    inputNormQue_.FreeTensor<float>(normLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::CreateGatherIndex(uint32_t n)
{
    LocalTensor<int32_t> indexLocal = indexBuffer_.Get<int32_t>();
    LocalTensor<int32_t> broadcastLocal = broadcastBuffer_.Get<int32_t>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    __local_mem__ int32_t *indexAddr = (__local_mem__ int32_t *)indexLocal.GetPhyAddr();
    __local_mem__ int32_t *broadcastAddr = (__local_mem__ int32_t *)broadcastLocal.GetPhyAddr();
    __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg allMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::LoadAlign(maskReg, maskAddr); // VL

        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> orderReg;
        MicroAPI::RegTensor<int32_t> duplicateReg1;
        MicroAPI::RegTensor<int32_t> duplicateReg2;
        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;

        MicroAPI::RegTensor<int32_t> duplicateBrcbReg;
        MicroAPI::RegTensor<int32_t> orderBrcbReg;
        MicroAPI::RegTensor<int32_t> tmpBrcbReg;

        // indexReg：使用DataCopyGather生成索引
        MicroAPI::Duplicate(duplicateReg1, static_cast<int32_t>(INDEX_BLOCK_LEN)); // [8,8,8,...,8]
        MicroAPI::Duplicate(duplicateReg2, static_cast<int32_t>(n * n));           // 填充n*n
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0));                       // [0,1,2,...,63]
        MicroAPI::Div(tmpReg1, orderReg, duplicateReg1, maskReg); // [0,0,...,0,1,1,...,1,...,7,7,...,7]
        MicroAPI::Mul(tmpReg2, tmpReg1, duplicateReg1, maskReg);  // [0,0,...,0,8,8,...,8,...,63,63,...,63]
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg);       // [0,1,2,...,7,0,1,2,...,7,...,0,1,2,...,7]
        MicroAPI::Mul(tmpReg3, tmpReg1, duplicateReg2, maskReg);  // [0,0,...,0,16,16,...,16,...,112,112,...,112]
        MicroAPI::Add(indexReg, tmpReg2, tmpReg3, maskReg); // [0,1,2,...,7,16,17,18,...,23,...,112,113,114,...,119]
        MicroAPI::DataCopy<int32_t, MicroAPI::StoreDist::DIST_NORM>(indexAddr, indexReg, allMask);

        MicroAPI::Duplicate(duplicateBrcbReg, static_cast<int32_t>(INDEX_BLOCK_LEN));
        MicroAPI::Arange(orderBrcbReg, static_cast<int32_t>(0));
        MicroAPI::Div(tmpBrcbReg, orderBrcbReg, duplicateBrcbReg, allMask);
        MicroAPI::DataCopy<int32_t, MicroAPI::StoreDist::DIST_NORM>(broadcastAddr, tmpBrcbReg, allMask);

        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::Process()
{
    if (blockIdx_ >= tilingData_.needCoreNum) {
        return;
    }

    uint32_t mask = 0;
    if (nSize_ == N_VALID_4) {
        mask = MASK_4;
    } else if (nSize_ == N_VALID_6) {
        mask = MASK_6;
    } else if (nSize_ == N_VALID_8) {
        mask = MASK_8;
    }
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    Duplicate(maskLocal, mask, MASK_NUM);
    CreateGatherIndex(nSize_);

    for (int64_t i = 0; i < loop_; i++) {
        uint32_t perloopSize = (i == loop_ - 1) ? tailLoopTSize_ : loopTSize_;
        uint64_t gradYOffset = blockIdx_ * tilingData_.coreTSize * nSize_ * nSize_ + i * loopTSize_ * nSize_ * nSize_;
        uint32_t gradYProcessNum = perloopSize * (nSize_ * nSize_); // grad_y从GM搬到UB上的元素个数
        CopyInGradY(gradYOffset, gradYProcessNum);
        uint32_t repeatSize = Ops::Base::GetVRegSize() / BLOCK_SIZE;
        uint16_t repeatTimes = Ceil(perloopSize, repeatSize);
        uint32_t normSize = perloopSize * nSize_ * nAlignSize_;
        uint32_t sumSize = perloopSize * nAlignSize_;
        for (int64_t iter = num_iters - 1; iter > 0; iter--) {
            int64_t colOffset = 2 * iter + 1;
            int64_t rowOffset = 2 * iter;
            // 列归一
            CalcCol(repeatSize, repeatTimes, colOffset, i, perloopSize);
            // 行归一
            CalcRow(repeatSize, repeatTimes, rowOffset, i, perloopSize);
        }

        // 列归一 最后一次迭代
        CalcCol(repeatSize, repeatTimes, 1, i, perloopSize);

        // 行归一 最后一次迭代Softmax
        CalcSoftmax(repeatSize, repeatTimes, i, perloopSize);

        int64_t gradInputOffset =
            blockIdx_ * tilingData_.coreTSize * nSize_ * nSize_ + i * loopTSize_ * nSize_ * nSize_;
        uint32_t gradInputdSize = perloopSize * (nSize_ * nSize_);
        CopyOut(gradInputOffset, gradInputdSize);
    }
}

} // namespace MhcSinkhornBackward

#endif // ASCENDC_MHC_SINKHORN_BACKWARD_SIMD_H
