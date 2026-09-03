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
 * \file mhc_sinkhorn_arch35.h
 * \brief mhc_sinkhorn
 */

#ifndef ASCENDC_MHC_SINKHORN_ARCH35_H
#define ASCENDC_MHC_SINKHORN_ARCH35_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "mhc_sinkhorn_struct.h"
#include "mhc_sinkhorn_tiling_key.h"

namespace MhcSinkhorn {
using namespace AscendC;

constexpr int64_t MASK_BUFFER_SIZE = 64;
constexpr int64_t MAX_BUFFER_SIZE = 256;
constexpr uint32_t MASK_4 = 0b00000000000000001111111111111111;
constexpr uint32_t MASK_6 = 0b00000000111111111111111111111111;
constexpr uint32_t MASK_8 = 0b11111111111111111111111111111111;
constexpr int64_t MASK_NUM = 8;
constexpr int64_t DOUBLE = 2;
constexpr int64_t INDEX_BLOCK_LEN = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

template <typename T, bool OUT_FLAG>
class MhcSinkhornSimd {
public:
    __aicore__ inline MhcSinkhornSimd(TPipe &pipe, const MhcSinkhornTilingData &tilingData)
        : pipe_(pipe),
          tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR h_res, GM_ADDR y, GM_ADDR norm_out, GM_ADDR sum_out, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcSoftmax(int32_t handleNum);
    template <bool ADDS_EPS>
    __aicore__ inline void CalcColNorm(int32_t handleNum);
    __aicore__ inline void CalcRowNorm(int32_t handleNum);
    __aicore__ inline void CopyIn(int32_t loopIdx, int32_t handleNum);
    __aicore__ inline void CopyOut(int32_t loopIdx, int32_t handleNum);
    __aicore__ inline void ScatterOutFromNorm(int32_t handleNum);

    GlobalTensor<float> hRes_;
    GlobalTensor<float> y_;
    GlobalTensor<float> normOut_;
    GlobalTensor<float> sumOut_;

    TPipe &pipe_;
    TQue<QuePosition::VECIN, 1> inputQue_;
    TQue<QuePosition::VECOUT, 1> outputQue_;
    TQue<QuePosition::VECOUT, 1> normOutQue_;
    TQue<QuePosition::VECOUT, 1> sumColQue_;
    TQue<QuePosition::VECOUT, 1> sumRowQue_;

    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> midBuffer_;
    TBuf<TPosition::VECCALC> sumBuffer_;

    const MhcSinkhornTilingData &tilingData_;
    int64_t blockIdx_{0};
    int64_t loop_{0};
    int64_t tailHandelSize_{0};
    int64_t nnSize_{0};
    int64_t nAlign_{0};
    int64_t nnAligSize_{0};
    int64_t iterCnt_{0};
    int64_t blockDataNum_{0};
};

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::Init(GM_ADDR h_res, GM_ADDR y, GM_ADDR norm_out, GM_ADDR sum_out,
                                                          GM_ADDR tiling)
{
    blockIdx_ = GetBlockIdx();
    blockDataNum_ = Ops::Base::GetUbBlockSize() / sizeof(float);
    iterCnt_ = tilingData_.num_iters * DOUBLE;
    nAlign_ = Ops::Base::CeilAlign(tilingData_.n, blockDataNum_);
    nnSize_ = tilingData_.n * tilingData_.n;
    nnAligSize_ = tilingData_.n * nAlign_;

    hRes_.SetGlobalBuffer((__gm__ float *)h_res + GetBlockIdx() * tilingData_.tNormCore * nnSize_);
    y_.SetGlobalBuffer((__gm__ float *)y + GetBlockIdx() * tilingData_.tNormCore * nnSize_);
    normOut_.SetGlobalBuffer((__gm__ float *)norm_out);
    sumOut_.SetGlobalBuffer((__gm__ float *)sum_out);

    pipe_.InitBuffer(maskBuffer_, MASK_BUFFER_SIZE);
    pipe_.InitBuffer(midBuffer_, MAX_BUFFER_SIZE);
    pipe_.InitBuffer(sumBuffer_, MAX_BUFFER_SIZE);

    auto inputNum = Ops::Base::CeilAlign(tilingData_.tUbFactor * nnSize_, blockDataNum_);
    pipe_.InitBuffer(inputQue_, DOUBLE_BUFFER, inputNum * sizeof(float));
    pipe_.InitBuffer(outputQue_, DOUBLE_BUFFER, inputNum * sizeof(float));
    pipe_.InitBuffer(normOutQue_, DOUBLE_BUFFER, tilingData_.tUbFactor * nnAligSize_ * sizeof(float));
    pipe_.InitBuffer(sumColQue_, DOUBLE_BUFFER, tilingData_.tUbFactor * nAlign_ * sizeof(float));
    if constexpr (OUT_FLAG) {
        pipe_.InitBuffer(sumRowQue_, DOUBLE_BUFFER, tilingData_.tUbFactor * nAlign_ * sizeof(float));
    } else {
        pipe_.InitBuffer(sumRowQue_, DOUBLE_BUFFER, blockDataNum_ * sizeof(float));
    }
    loop_ = tilingData_.tNormCoreLoop;
    tailHandelSize_ = tilingData_.tUbFactorTail;    // 尾循环处理的数量
    if (blockIdx_ == tilingData_.usedCoreNum - 1) { //  尾核
        loop_ = tilingData_.tTailCoreLoop;
        tailHandelSize_ = tilingData_.tUbTailTail;
    }
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CopyIn(int32_t loopIdx, int32_t handleNum)
{
    LocalTensor<float> inputLocal = inputQue_.AllocTensor<float>();
    int64_t inputOffset = loopIdx * tilingData_.tUbFactor * nnSize_;
    uint32_t handleSize = handleNum * nnSize_ * sizeof(float);
    DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyExtParams{1, handleSize, 0, 0, 0};
    DataCopyPad(inputLocal, hRes_[inputOffset], dataCopyExtParams, dataCopyPadExtParams);

    inputQue_.EnQue(inputLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CopyOut(int32_t loopIdx, int32_t handleNum)
{
    LocalTensor<float> outputLocal = outputQue_.DeQue<float>();
    int64_t outOffset = loopIdx * tilingData_.tUbFactor * nnSize_;
    uint32_t handleSize = handleNum * nnSize_ * sizeof(float);

    DataCopyExtParams dataCopyExtParams{1, handleSize, 0, 0, 0};
    DataCopyPad(y_[outOffset], outputLocal, dataCopyExtParams);
    outputQue_.FreeTensor<float>(outputLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CalcSoftmax(int32_t handleNum)
{
    LocalTensor<float> inputLocal = inputQue_.DeQue<float>();
    LocalTensor<float> midResLocal = midBuffer_.Get<float>();
    LocalTensor<float> sumResLocal = sumBuffer_.Get<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();

    __ubuf__ float *inputAddr = (__ubuf__ float *)inputLocal.GetPhyAddr();
    __ubuf__ float *midResAddr = (__ubuf__ float *)midResLocal.GetPhyAddr();
    __ubuf__ float *sumResAddr = (__ubuf__ float *)sumResLocal.GetPhyAddr();
    __ubuf__ uint32_t *maskAddr = (__ubuf__ uint32_t *)maskLocal.GetPhyAddr();

    LocalTensor<float> normOutLocal = normOutQue_.AllocTensor<float>();
    LocalTensor<float> sumOutLocal = sumColQue_.AllocTensor<float>();
    __ubuf__ float *normOutAddr = (__ubuf__ float *)normOutLocal.GetPhyAddr();
    __ubuf__ float *sumOutAddr = (__ubuf__ float *)sumOutLocal.GetPhyAddr();

    uint16_t vflen = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t blockCnt = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    uint32_t copyLen = handleNum * blockDataNum_;
    uint32_t handleSize = handleNum * blockDataNum_;
    uint32_t handleLen = handleSize;
    uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint16_t>(handleNum), blockCnt);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> inputReg;
        Reg::RegTensor<float> midReg;
        Reg::RegTensor<float> sumReg;
        Reg::RegTensor<float> sumoutReg;

        Reg::RegTensor<int32_t> orderReg;
        Reg::RegTensor<int32_t> tmpReg1;
        Reg::RegTensor<int32_t> tmpReg2;
        Reg::RegTensor<int32_t> tmpReg3;
        Reg::RegTensor<int32_t> dupReg1;
        Reg::RegTensor<int32_t> dupReg2;
        Reg::RegTensor<int32_t> rowIndexReg;

        Reg::AddrReg normAddrReg;
        Reg::AddrReg sumAddrReg;
        Reg::MaskReg maskRegCalc;
        Reg::MaskReg maskRegCopy;
        Reg::MaskReg maskRegFresh;

        Reg::MaskReg maskReg = Reg::CreateMask<uint32_t>();
        Reg::Duplicate(dupReg1, static_cast<int32_t>(INDEX_BLOCK_LEN));
        Reg::Duplicate(dupReg2, static_cast<int32_t>(nnSize_));
        Reg::Arange(orderReg, static_cast<int32_t>(0));
        Reg::Div(tmpReg1, orderReg, dupReg1, maskReg);
        Reg::Mul(tmpReg2, tmpReg1, dupReg1, maskReg);
        Reg::Sub(tmpReg2, orderReg, tmpReg2, maskReg);
        Reg::Mul(tmpReg3, tmpReg1, dupReg2, maskReg);
        Reg::Add(rowIndexReg, tmpReg2, tmpReg3, maskReg);

        for (uint16_t i = 0; i < loopSize; i++) {
            /* 32B按mask处理 */
            maskRegFresh = Reg::UpdateMask<uint32_t>(handleSize);
            Reg::LoadAlign(maskRegCalc, maskAddr);
            Reg::MaskAnd(maskRegCalc, maskRegCalc, maskRegFresh, maskReg);
            /* 32B全部处理 */
            maskRegCopy = Reg::UpdateMask<uint32_t>(copyLen);

            Reg::Duplicate(sumoutReg, static_cast<float>(0));
            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(tilingData_.n); nIdx++) {
                auto inputOfset = inputAddr + (i * blockCnt * nnSize_) + nIdx * tilingData_.n;
                Reg::DataCopyGather(inputReg, inputOfset, (Reg::RegTensor<uint32_t> &)(rowIndexReg), maskRegCalc);
                Reg::ReduceMaxWithDataBlock(midReg, inputReg, maskRegCalc);
                Reg::LocalMemBar<AscendC::Reg::MemType::VEC_LOAD, AscendC::Reg::MemType::VEC_STORE>();
                Reg::DataCopy(midResAddr, midReg, maskReg);
                Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                Reg::DataCopy<float, Reg::LoadDist::DIST_E2B_B32>(midReg, midResAddr);
                Reg::Sub(inputReg, inputReg, midReg, maskRegCalc);
                Reg::Exp(inputReg, inputReg, maskRegCalc);

                Reg::ReduceSumWithDataBlock(sumReg, inputReg, maskRegCalc);
                Reg::LocalMemBar<AscendC::Reg::MemType::VEC_LOAD, AscendC::Reg::MemType::VEC_STORE>();
                Reg::DataCopy(sumResAddr, sumReg, maskReg);
                Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                Reg::DataCopy<float, Reg::LoadDist::DIST_E2B_B32>(sumReg, sumResAddr);
                Reg::Div(inputReg, inputReg, sumReg, maskRegCalc);
                /* 搬出normout[0] */
                normAddrReg = Reg::CreateAddrReg<float>(i, vflen, nIdx, handleLen);
                Reg::DataCopy(normOutAddr, inputReg, normAddrReg, maskRegCopy);
                Reg::Adds(inputReg, inputReg, tilingData_.eps, maskRegCalc);
                Reg::Add(sumoutReg, sumoutReg, inputReg, maskRegCalc);
            }
            /* 搬出sumout[1] */
            sumAddrReg = Reg::CreateAddrReg<float>(i, vflen);
            Reg::DataCopy(sumOutAddr, sumoutReg, sumAddrReg, maskRegCopy);
        }
    }

    inputQue_.FreeTensor<float>(inputLocal);
    normOutQue_.EnQue(normOutLocal);
    sumColQue_.EnQue(sumOutLocal);
}

template <typename T, bool OUT_FLAG>
template <bool ADDS_EPS>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CalcColNorm(int32_t handleNum)
{
    LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
    LocalTensor<float> sumOutLocal = sumColQue_.DeQue<float>();
    __ubuf__ float *normOutAddr = (__ubuf__ float *)normOutLocal.GetPhyAddr();
    __ubuf__ float *sumOutAddr = (__ubuf__ float *)sumOutLocal.GetPhyAddr();

    LocalTensor<float> midResLocal = midBuffer_.Get<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    __ubuf__ float *midResAddr = (__ubuf__ float *)midResLocal.GetPhyAddr();
    __ubuf__ uint32_t *maskAddr = (__ubuf__ uint32_t *)maskLocal.GetPhyAddr();

    uint16_t vflen = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t blockCnt = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    uint32_t copyLen = handleNum * blockDataNum_;
    uint32_t handleSize = handleNum * blockDataNum_;
    uint32_t handleLen = handleSize;
    uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint16_t>(handleNum), blockCnt);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> normReg;
        Reg::RegTensor<float> sumOutReg;
        Reg::RegTensor<float> midReg;

        Reg::AddrReg normAddrReg;
        Reg::AddrReg sumAddrReg;
        Reg::MaskReg maskRegCalc;
        Reg::MaskReg maskRegCopy;
        Reg::MaskReg maskRegFresh;

        Reg::MaskReg maskReg = Reg::CreateMask<uint32_t>();
        for (uint16_t i = 0; i < loopSize; i++) {
            maskRegFresh = Reg::UpdateMask<uint32_t>(handleSize);
            Reg::LoadAlign(maskRegCalc, maskAddr);
            Reg::MaskAnd(maskRegCalc, maskRegCalc, maskRegFresh, maskReg);
            maskRegCopy = Reg::UpdateMask<uint32_t>(copyLen);

            sumAddrReg = Reg::CreateAddrReg<uint32_t>(i, vflen);
            Reg::DataCopy(sumOutReg, sumOutAddr, sumAddrReg);
            Reg::Adds(sumOutReg, sumOutReg, tilingData_.eps, maskRegCalc);
            if constexpr (OUT_FLAG) {
                Reg::DataCopy(sumOutAddr, sumOutReg, sumAddrReg, maskRegCopy);
            }

            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(tilingData_.n); nIdx++) {
                normAddrReg = Reg::CreateAddrReg<uint32_t>(i, vflen, nIdx, handleLen);
                Reg::DataCopy(normReg, normOutAddr, normAddrReg);
                if constexpr (ADDS_EPS) {
                    Reg::Adds(normReg, normReg, tilingData_.eps, maskRegCalc); // 这里有区别
                }
                Reg::Div(normReg, normReg, sumOutReg, maskRegCalc);
                Reg::DataCopy(normOutAddr, normReg, normAddrReg, maskRegCopy);
            }
        }
    }
    normOutQue_.EnQue(normOutLocal);
    sumColQue_.EnQue(sumOutLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CalcRowNorm(int32_t handleNum)
{
    LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
    LocalTensor<float> sumColLocal = sumColQue_.DeQue<float>();
    LocalTensor<float> sumRowLocal = sumRowQue_.AllocTensor<float>();

    __ubuf__ float *normOutAddr = (__ubuf__ float *)normOutLocal.GetPhyAddr();
    __ubuf__ float *sumRowOutAddr = (__ubuf__ float *)sumRowLocal.GetPhyAddr();
    __ubuf__ float *sumColOutAddr = (__ubuf__ float *)sumColLocal.GetPhyAddr();

    LocalTensor<float> midResLocal = midBuffer_.Get<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    __ubuf__ float *midResAddr = (__ubuf__ float *)midResLocal.GetPhyAddr();
    __ubuf__ uint32_t *maskAddr = (__ubuf__ uint32_t *)maskLocal.GetPhyAddr();

    uint16_t vflen = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t blockCnt = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    uint32_t copyLen = handleNum * blockDataNum_;
    uint32_t handleSize = handleNum * blockDataNum_;
    uint32_t handleLen = handleSize;
    uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint16_t>(handleNum), blockCnt);
    auto tailLoopHandleNum = handleNum - (loopSize - 1) * blockCnt;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> normReg;
        Reg::RegTensor<float> midReg;
        Reg::RegTensor<float> sumReg;
        Reg::RegTensor<float> sumoutReg;
        Reg::UnalignRegForStore uReg;

        Reg::AddrReg sumAddrReg;
        Reg::AddrReg normAddrReg;
        Reg::MaskReg maskRegCalc;
        Reg::MaskReg maskRegCopy;
        Reg::MaskReg maskRegFresh;
        Reg::MaskReg maskReg;

        maskReg = Reg::CreateMask<uint32_t>();
        for (uint16_t i = 0; i < loopSize; i++) {
            auto postUpdateStride = (i == loopSize - 1) ? tailLoopHandleNum : blockCnt;
            maskRegFresh = Reg::UpdateMask<uint32_t>(handleSize);
            Reg::LoadAlign(maskRegCalc, maskAddr);
            Reg::MaskAnd(maskRegCalc, maskRegCalc, maskRegFresh, maskReg);
            maskRegCopy = Reg::UpdateMask<uint32_t>(copyLen);

            Reg::Duplicate(sumoutReg, static_cast<float>(0));
            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(tilingData_.n); nIdx++) {
                normAddrReg = Reg::CreateAddrReg<uint32_t>(i, vflen, nIdx, handleLen);
                Reg::DataCopy(normReg, normOutAddr, normAddrReg);
                Reg::ReduceSumWithDataBlock(sumReg, normReg, maskRegCalc);
                Reg::DataCopy(midResAddr, sumReg, maskRegCopy);
                if constexpr (OUT_FLAG) {
                    /* 搬出sumout[2*i] */
                    auto sumOfset = sumRowOutAddr + i * blockCnt + nIdx * handleNum;
                    Reg::Adds(sumReg, sumReg, tilingData_.eps, maskReg);
                    /* 非对齐搬运，偏移为handleNum */
                    Reg::StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(sumOfset, sumReg, uReg,
                                                                                 postUpdateStride);
                    Reg::StoreUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(sumOfset, uReg, 0);
                }
                Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                Reg::DataCopy<float, Reg::LoadDist::DIST_E2B_B32>(sumReg, midResAddr);
                Reg::Adds(sumReg, sumReg, tilingData_.eps, maskRegCalc);
                Reg::Div(normReg, normReg, sumReg, maskRegCalc);
                Reg::DataCopy(normOutAddr, normReg, normAddrReg, maskRegCopy);
                Reg::Add(sumoutReg, sumoutReg, normReg, maskRegCalc);
            }
            /* 搬出sumout[2*i+1] */
            sumAddrReg = Reg::CreateAddrReg<uint32_t>(i, vflen);
            Reg::DataCopy(sumColOutAddr, sumoutReg, sumAddrReg, maskRegCopy);
        }
    }
    normOutQue_.EnQue(normOutLocal);
    sumColQue_.EnQue(sumColLocal);
    sumRowQue_.EnQue(sumRowLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::ScatterOutFromNorm(int32_t handleNum)
{
    LocalTensor<float> outputLocal = outputQue_.AllocTensor<float>();
    LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
    __ubuf__ float *normOutAddr = (__ubuf__ float *)normOutLocal.GetPhyAddr();
    __ubuf__ float *outputAddr = (__ubuf__ float *)outputLocal.GetPhyAddr();

    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    __ubuf__ uint32_t *maskAddr = (__ubuf__ uint32_t *)maskLocal.GetPhyAddr();

    uint16_t vflen = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t blockCnt = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    uint32_t handleSize = handleNum * blockDataNum_;
    uint32_t handleLen = handleSize;
    uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint16_t>(handleNum), blockCnt);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> normReg;
        Reg::RegTensor<int32_t> orderReg;
        Reg::RegTensor<int32_t> tmpReg1;
        Reg::RegTensor<int32_t> tmpReg2;
        Reg::RegTensor<int32_t> tmpReg3;
        Reg::RegTensor<int32_t> dupReg1;
        Reg::RegTensor<int32_t> dupReg2;
        Reg::RegTensor<int32_t> indexReg;

        Reg::AddrReg normAddrReg;
        Reg::MaskReg maskRegCalc;
        Reg::MaskReg maskRegFresh;

        Reg::MaskReg maskReg = Reg::CreateMask<uint32_t>();
        Reg::Duplicate(dupReg1, INDEX_BLOCK_LEN);
        Reg::Duplicate(dupReg2, static_cast<int32_t>(nnSize_));
        Reg::Arange(orderReg, static_cast<int32_t>(0));
        Reg::Div(tmpReg1, orderReg, dupReg1, maskReg);
        Reg::Mul(tmpReg2, tmpReg1, dupReg1, maskReg);
        Reg::Sub(tmpReg2, orderReg, tmpReg2, maskReg);
        Reg::Mul(tmpReg3, tmpReg1, dupReg2, maskReg);
        Reg::Add(indexReg, tmpReg2, tmpReg3, maskReg);
        for (uint16_t i = 0; i < loopSize; i++) {
            maskRegFresh = Reg::UpdateMask<uint32_t>(handleSize);
            Reg::LoadAlign(maskRegCalc, maskAddr);
            Reg::MaskAnd(maskRegCalc, maskRegCalc, maskRegFresh, maskReg);
            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(tilingData_.n); nIdx++) {
                auto outOfset = outputAddr + (i * blockCnt * nnSize_) + nIdx * tilingData_.n;
                normAddrReg = Reg::CreateAddrReg<uint32_t>(i, vflen, nIdx, handleLen);
                Reg::DataCopy(normReg, normOutAddr, normAddrReg);
                Reg::DataCopyScatter(outOfset, normReg, (Reg::RegTensor<uint32_t> &)(indexReg), maskRegCalc);
            }
        }
    }
    outputQue_.EnQue(outputLocal);
    normOutQue_.FreeTensor(normOutLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    uint32_t mask = 0;
    if (tilingData_.n == N_VALID_4) {
        mask = MASK_4;
    } else if (tilingData_.n == N_VALID_6) {
        mask = MASK_6;
    } else if (tilingData_.n == N_VALID_8) {
        mask = MASK_8;
    }
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    Duplicate(maskLocal, mask, MASK_NUM);

    auto normStartOfset = GetBlockIdx() * iterCnt_ * tilingData_.tNormCore * nnAligSize_;
    auto sumStartOfset = GetBlockIdx() * iterCnt_ * tilingData_.tNormCore * nAlign_;
    auto normStride = iterCnt_ * tilingData_.tUbFactor * nnAligSize_;
    auto sumStride = iterCnt_ * tilingData_.tUbFactor * nAlign_;

    for (int64_t loopIdx = 0; loopIdx < loop_; loopIdx++) {
        auto handleNum = (loopIdx == loop_ - 1) ? tailHandelSize_ : tilingData_.tUbFactor;
        auto curNormSize = handleNum * tilingData_.n * nAlign_;
        auto curSumSize = handleNum * nAlign_;
        auto normLoopStride = loopIdx * normStride;
        auto sumLoopStride = loopIdx * sumStride;

        CopyIn(loopIdx, handleNum);
        CalcSoftmax(handleNum);
        if constexpr (OUT_FLAG) {
            int64_t normOutOfset = normStartOfset + normLoopStride;
            uint32_t normOutLen = curNormSize * sizeof(float);
            DataCopyExtParams normCopyExtParams{1, normOutLen, 0, 0, 0};
            LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
            DataCopyPad(normOut_[normOutOfset], normOutLocal, normCopyExtParams);
            normOutQue_.EnQue(normOutLocal);
        }
        event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);

        CalcColNorm<true>(handleNum);
        if constexpr (OUT_FLAG) {
            int64_t normOutOfset = normStartOfset + normLoopStride + curNormSize;
            uint32_t normOutLen = curNormSize * sizeof(float);
            DataCopyExtParams normCopyExtParams{1, normOutLen, 0, 0, 0};
            LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
            DataCopyPad(normOut_[normOutOfset], normOutLocal, normCopyExtParams);
            normOutQue_.EnQue(normOutLocal);

            int64_t sumOutOfset = sumStartOfset + sumLoopStride + curSumSize;
            uint32_t sumOutLen = curSumSize * sizeof(float);
            DataCopyExtParams sumCopyExtParams{1, sumOutLen, 0, 0, 0};
            LocalTensor<float> sumOutLocal = sumColQue_.DeQue<float>();
            DataCopyPad(sumOut_[sumOutOfset], sumOutLocal, sumCopyExtParams);
            sumColQue_.EnQue(sumOutLocal);
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        for (int64_t iter = 1; iter < tilingData_.num_iters; iter++) {
            event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);

            CalcRowNorm(handleNum);
            if constexpr (OUT_FLAG) {
                int64_t normOutOfset = normStartOfset + normLoopStride + (2 * iter * curNormSize);
                uint32_t normOutLen = curNormSize * sizeof(float);
                DataCopyExtParams normCopyExtParams{1, normOutLen, 0, 0, 0};
                LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
                DataCopyPad(normOut_[normOutOfset], normOutLocal, normCopyExtParams);
                normOutQue_.EnQue(normOutLocal);

                int64_t sumOutOfset = sumStartOfset + sumLoopStride + (2 * iter * curSumSize);
                uint32_t sumOutLen = curSumSize * sizeof(float);
                DataCopyExtParams sumCopyExtParams{1, sumOutLen, 0, 0, 0};
                LocalTensor<float> sumRowLocal = sumRowQue_.DeQue<float>();
                DataCopyPad(sumOut_[sumOutOfset], sumRowLocal, sumCopyExtParams);
                sumRowQue_.EnQue(sumRowLocal);
            }
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);

            CalcColNorm<false>(handleNum);
            if constexpr (OUT_FLAG) {
                int64_t normOutOfset = normStartOfset + normLoopStride + (2 * iter + 1) * curNormSize;
                uint32_t normOutLen = curNormSize * sizeof(float);
                DataCopyExtParams normCopyExtParams{1, normOutLen, 0, 0, 0};
                LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
                DataCopyPad(normOut_[normOutOfset], normOutLocal, normCopyExtParams);
                normOutQue_.EnQue(normOutLocal);

                int64_t sumOutOfset = sumStartOfset + sumLoopStride + (2 * iter + 1) * curSumSize;
                uint32_t sumOutLen = curSumSize * sizeof(float);
                DataCopyExtParams sumCopyExtParams{1, sumOutLen, 0, 0, 0};
                LocalTensor<float> sumColLocal = sumColQue_.DeQue<float>();
                DataCopyPad(sumOut_[sumOutOfset], sumColLocal, sumCopyExtParams);
                sumColQue_.EnQue(sumColLocal);
            }
            LocalTensor<float> sumRowLocal = sumRowQue_.DeQue<float>();
            sumRowQue_.FreeTensor(sumRowLocal);
        }
        LocalTensor<float> sumColLocal = sumColQue_.DeQue<float>();
        sumColQue_.FreeTensor(sumColLocal);

        ScatterOutFromNorm(handleNum);
        CopyOut(loopIdx, handleNum);
    }
}
} // namespace MhcSinkhorn

#endif
