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
 * \file mhc_post_regbase.h
 * \brief MhcPost kernel implementation
 * Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 *          where: (H_{l}^{res})^{T} * x_l represents matrix multiplication with transposed h_res
 *                h_{l}^{out} * H_{t}^{post} represents element-wise multiplication and broadcasting
 */

#ifndef ASCENDC_MHC_POST_REGBASE_H
#define ASCENDC_MHC_POST_REGBASE_H

#include "kernel_operator.h"
#include "basic_api/kernel_operator_utils_intf.h"
#include "op_kernel/math_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_post_tiling_data.h"
#include "mhc_post_tiling_key.h"

namespace MhcPost {
using namespace AscendC;

#define REGBASE_TEMPLATE_DECLARE template <typename T>
#define REGBASE_TEMPLATE_ARGS T

REGBASE_TEMPLATE_DECLARE
class MhcPostRegbase {
public:
    __aicore__ inline MhcPostRegbase(TPipe *tPipe, const MhcPostRegbaseTilingData *__restrict tilingData)
        : pipe_(tPipe),
          tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR output,
                                GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInHOut(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void CopyInX(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void CopyInHPost(int64_t bsIdx, int64_t nIdx, int64_t nNum);
    __aicore__ inline void CopyInHRes(int64_t bsIdx, int64_t nIdx, int64_t nNum);
    __aicore__ inline void DoMulAndAdd(LocalTensor<float> hPostUb, LocalTensor<float> hResUb, int64_t nNum,
                                       int64_t dNum);
    __aicore__ inline void DoMulAndAdd_N4(LocalTensor<float> hPostUb, LocalTensor<float> hResUb, int64_t nNum,
                                          int64_t dNum);
    __aicore__ inline void CopyOutY(int64_t bsIdx, int64_t nIdx, int64_t nNum, int64_t dIdx, int64_t dNum);

private:
    TPipe *pipe_;
    const MhcPostRegbaseTilingData *tilingData_;

    // Input queues - Double Buffer enabled (queue depth = DOUBLE_BUFFER_DEPTH)
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> hOutTileQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> xTileQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> hResTileQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> hPostTileQueue_;

    // Output queues - Double Buffer enabled (queue depth = DOUBLE_BUFFER_DEPTH)
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_DEPTH> yTileQueue_;
    TBuf<QuePosition::VECCALC> yTileBuf_;

    // Global memory tensors - inputs (bf16/fp16)
    GlobalTensor<T> xGm_;
    GlobalTensor<T> hOutGm_;
    // Global memory tensors - inputs (float32)
    GlobalTensor<float> hResGm_;
    GlobalTensor<float> hPostGm_;

    // Global memory tensors - outputs (bf16/fp16)
    GlobalTensor<T> outputGm_;
    static constexpr uint16_t VL_FP32 = 256 / sizeof(float);

    int64_t curBS_;
    uint32_t blockIdx_;
    constexpr static AscendC::MicroAPI::CastTrait castB16ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
};

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
                                                                   GM_ADDR output, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    // Calculate work distribution with remainder handling
    if (blockIdx_ < tilingData_->usedCoreNum - 1) {
        curBS_ = tilingData_->bsInner;
    } else {
        curBS_ = tilingData_->bsTail;
    }

    // Set global memory buffers
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    hOutGm_.SetGlobalBuffer((__gm__ T *)hOut);
    hResGm_.SetGlobalBuffer((__gm__ float *)hRes);
    hPostGm_.SetGlobalBuffer((__gm__ float *)hPost);
    outputGm_.SetGlobalBuffer((__gm__ T *)output);

    // Initialize output queues - Double Buffer with depth=DOUBLE_BUFFER_DEPTH
    pipe_->InitBuffer(yTileBuf_, tilingData_->nInner * tilingData_->dInner * sizeof(float));
    pipe_->InitBuffer(yTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->nInner * tilingData_->dInner * sizeof(T));

    // Initialize input queues - Double Buffer with depth=DOUBLE_BUFFER_DEPTH for data tiles
    int64_t nInnerBuf = Ops::Base::CeilAlign(tilingData_->nInner, static_cast<int64_t>(REG_ALIGN_N));
    pipe_->InitBuffer(hResTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->n * nInnerBuf * sizeof(float));
    pipe_->InitBuffer(hPostTileQueue_, DOUBLE_BUFFER_DEPTH, nInnerBuf * sizeof(float));

    pipe_->InitBuffer(hOutTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->dInner * sizeof(T));
    pipe_->InitBuffer(xTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->n * tilingData_->dInner * sizeof(T));
}

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::DoMulAndAdd(LocalTensor<float> hPostUb,
                                                                          LocalTensor<float> hResUb, int64_t nNum,
                                                                          int64_t dNum)
{
    if (tilingData_->n == 4) {
        DoMulAndAdd_N4(hPostUb, hResUb, nNum, dNum);
        return;
    }

    uint16_t nTimes = static_cast<uint16_t>(nNum);
    uint16_t nTotal = static_cast<uint16_t>(tilingData_->n);
    uint16_t dRepeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(dNum, static_cast<int64_t>(VL_FP32)));
    uint16_t dAlign = static_cast<uint16_t>(Ops::Base::CeilAlign(dNum, static_cast<int64_t>(REG_ALIGN_D)));
    uint16_t nAlign = static_cast<uint16_t>(Ops::Base::CeilAlign(nNum, static_cast<int64_t>(REG_ALIGN_N)));
    LocalTensor<T> xUb = xTileQueue_.DeQue<T>();
    LocalTensor<T> hOutUb = hOutTileQueue_.DeQue<T>();

    LocalTensor<float> yTileBuf = yTileBuf_.Get<float>();

    auto xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    auto hResAddr = (__ubuf__ float *)hResUb.GetPhyAddr();
    auto hPostAddr = (__ubuf__ float *)hPostUb.GetPhyAddr();
    auto hOutAddr = (__ubuf__ T *)hOutUb.GetPhyAddr();
    auto yAddr = (__ubuf__ float *)yTileBuf.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xReg;
        AscendC::MicroAPI::RegTensor<T> hOutReg;
        AscendC::MicroAPI::RegTensor<float> xRegFloat;
        AscendC::MicroAPI::RegTensor<float> hOutRegFloat;
        AscendC::MicroAPI::RegTensor<float> hResReg;
        AscendC::MicroAPI::RegTensor<float> hPostReg;
        AscendC::MicroAPI::RegTensor<float> outRegFloat;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t nIndex = 0; nIndex < nTimes; nIndex++) {
            uint32_t dNumU32 = static_cast<uint32_t>(dAlign);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(hPostReg, hPostAddr + nIndex);
            for (uint16_t j = 0; j < dRepeatTimes; j++) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(dNumU32);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(hOutReg,
                                                                                             hOutAddr + j * VL_FP32);
                AscendC::MicroAPI::Cast<float, T, castB16ToB32>(hOutRegFloat, hOutReg, pMask);
                AscendC::MicroAPI::Mul(outRegFloat, hOutRegFloat, hPostReg, pMask);
                for (uint16_t i = 0; i < nTotal; i++) {
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        xReg, xAddr + i * dAlign + j * VL_FP32);
                    AscendC::MicroAPI::Cast<float, T, castB16ToB32>(xRegFloat, xReg, pMask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        hResReg, hResAddr + i * nAlign + nIndex);
                    AscendC::MicroAPI::MulAddDst(outRegFloat, xRegFloat, hResReg, pMask);
                }
                AscendC::MicroAPI::DataCopy(yAddr + nIndex * dAlign + j * VL_FP32, outRegFloat, pMask);
            }
        }
    }
    hOutTileQueue_.FreeTensor(hOutUb);
    xTileQueue_.FreeTensor(xUb);

    LocalTensor<T> yTileLocal = yTileQueue_.AllocTensor<T>();
    AscendC::Cast(yTileLocal, yTileBuf, AscendC::RoundMode::CAST_RINT, nTimes * dAlign);
    yTileQueue_.EnQue(yTileLocal);
}

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::DoMulAndAdd_N4(LocalTensor<float> hPostUb,
                                                                             LocalTensor<float> hResUb, int64_t nNum,
                                                                             int64_t dNum)
{
    uint16_t nTimes = static_cast<uint16_t>(nNum);
    uint16_t dAlign = static_cast<uint16_t>(Ops::Base::CeilAlign(dNum, static_cast<int64_t>(REG_ALIGN_D)));
    uint16_t nAlign = static_cast<uint16_t>(Ops::Base::CeilAlign(nNum, static_cast<int64_t>(REG_ALIGN_N)));
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(dNum, static_cast<int64_t>(VL_FP32)));

    LocalTensor<T> xUb = xTileQueue_.DeQue<T>();
    LocalTensor<T> hOutUb = hOutTileQueue_.DeQue<T>();
    LocalTensor<float> yTileBuf = yTileBuf_.Get<float>();

    auto xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    auto hResAddr = (__ubuf__ float *)hResUb.GetPhyAddr();
    auto hPostAddr = (__ubuf__ float *)hPostUb.GetPhyAddr();
    auto hOutAddr = (__ubuf__ T *)hOutUb.GetPhyAddr();
    auto yAddr = (__ubuf__ float *)yTileBuf.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xReg0, xReg1, xReg2, xReg3;
        AscendC::MicroAPI::RegTensor<float> xRegFloat0, xRegFloat1, xRegFloat2, xRegFloat3;
        AscendC::MicroAPI::RegTensor<float> hResReg0, hResReg1, hResReg2, hResReg3;
        AscendC::MicroAPI::RegTensor<T> hOutReg;
        AscendC::MicroAPI::RegTensor<float> hOutRegFloat;
        AscendC::MicroAPI::RegTensor<float> hPostReg;
        AscendC::MicroAPI::RegTensor<float> outRegFloat;
        AscendC::MicroAPI::MaskReg pMask;

        for (uint16_t nIndex = 0; nIndex < nTimes; nIndex++) {
            uint32_t dNumU32 = static_cast<uint32_t>(dAlign);

            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                hResReg0, hResAddr + 0 * nAlign + nIndex);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                hResReg1, hResAddr + 1 * nAlign + nIndex);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                hResReg2, hResAddr + 2 * nAlign + nIndex);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                hResReg3, hResAddr + 3 * nAlign + nIndex);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(hPostReg, hPostAddr + nIndex);

            for (uint16_t j = 0; j < repeatTimes; j++) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(dNumU32);

                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(hOutReg,
                                                                                             hOutAddr + j * VL_FP32);
                AscendC::MicroAPI::Cast<float, T, castB16ToB32>(hOutRegFloat, hOutReg, pMask);

                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xReg0, xAddr + 0 * dAlign + j * VL_FP32);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xReg1, xAddr + 1 * dAlign + j * VL_FP32);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xReg2, xAddr + 2 * dAlign + j * VL_FP32);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xReg3, xAddr + 3 * dAlign + j * VL_FP32);

                AscendC::MicroAPI::Cast<float, T, castB16ToB32>(xRegFloat0, xReg0, pMask);
                AscendC::MicroAPI::Cast<float, T, castB16ToB32>(xRegFloat1, xReg1, pMask);
                AscendC::MicroAPI::Cast<float, T, castB16ToB32>(xRegFloat2, xReg2, pMask);
                AscendC::MicroAPI::Cast<float, T, castB16ToB32>(xRegFloat3, xReg3, pMask);

                AscendC::MicroAPI::Mul(outRegFloat, hOutRegFloat, hPostReg, pMask);
                AscendC::MicroAPI::MulAddDst(outRegFloat, xRegFloat0, hResReg0, pMask);
                AscendC::MicroAPI::MulAddDst(outRegFloat, xRegFloat1, hResReg1, pMask);
                AscendC::MicroAPI::MulAddDst(outRegFloat, xRegFloat2, hResReg2, pMask);
                AscendC::MicroAPI::MulAddDst(outRegFloat, xRegFloat3, hResReg3, pMask);

                AscendC::MicroAPI::DataCopy(yAddr + nIndex * dAlign + j * VL_FP32, outRegFloat, pMask);
            }
        }
    }

    hOutTileQueue_.FreeTensor(hOutUb);
    xTileQueue_.FreeTensor(xUb);

    LocalTensor<T> yTileLocal = yTileQueue_.AllocTensor<T>();
    AscendC::Cast(yTileLocal, yTileBuf, AscendC::RoundMode::CAST_RINT, nTimes * dAlign);
    yTileQueue_.EnQue(yTileLocal);
}

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t nLoopTime = tilingData_->nOuter;
    int64_t dLoopTime = tilingData_->dOuter;
    int64_t bsStart = blockIdx_ * tilingData_->bsInner;
    for (int64_t itemIdx = 0; itemIdx < curBS_; itemIdx++) {
        int64_t bsIdx = bsStart + itemIdx;
        for (int64_t nIdx = 0; nIdx < nLoopTime; nIdx++) {
            int64_t nNum = (nIdx < nLoopTime - 1) ? tilingData_->nInner : tilingData_->nTail;
            CopyInHRes(bsIdx, nIdx, nNum);
            CopyInHPost(bsIdx, nIdx, nNum);
            LocalTensor<float> hPostUb = hPostTileQueue_.DeQue<float>();
            LocalTensor<float> hResUb = hResTileQueue_.DeQue<float>();
            for (int64_t dIdx = 0; dIdx < dLoopTime; dIdx++) {
                int64_t dNum = (dIdx < dLoopTime - 1) ? tilingData_->dInner : tilingData_->dTail;
                CopyInHOut(bsIdx, dIdx, dNum);
                CopyInX(bsIdx, dIdx, dNum);
                DoMulAndAdd(hPostUb, hResUb, nNum, dNum);

                CopyOutY(bsIdx, nIdx, nNum, dIdx, dNum);
            }
            hPostTileQueue_.FreeTensor(hPostUb);
            hResTileQueue_.FreeTensor(hResUb);
        }
    }
}

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::CopyInHOut(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hOutOffset = bsIdx * tilingData_->d + dIdx * tilingData_->dInner;
    LocalTensor<T> hOutTileLocal = hOutTileQueue_.AllocTensor<T>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(hOutTileLocal, hOutGm_[hOutOffset], copyParams, {false, 0, 0, 0});

    hOutTileQueue_.EnQue<T>(hOutTileLocal);
}

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::CopyInHPost(int64_t bsIdx, int64_t nIdx, int64_t nNum)
{
    int64_t hPostOffset = bsIdx * tilingData_->n + nIdx * tilingData_->nInner;
    LocalTensor<float> hPostTileLocal = hPostTileQueue_.AllocTensor<float>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(nNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(hPostTileLocal, hPostGm_[hPostOffset], copyParams, {false, 0, 0, 0});
    hPostTileQueue_.EnQue<float>(hPostTileLocal);
}

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::CopyInHRes(int64_t bsIdx, int64_t nIdx, int64_t nNum)
{
    int64_t hResBase = bsIdx * tilingData_->n * tilingData_->n;
    int64_t nStart = nIdx * tilingData_->nInner;
    int64_t hResOffset = hResBase + nStart;
    LocalTensor<float> hResTileLocal = hResTileQueue_.AllocTensor<float>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(tilingData_->n), static_cast<uint32_t>(nNum * sizeof(float)),
                                    static_cast<uint32_t>((tilingData_->n - nNum) * sizeof(float)), 0, 0};

    DataCopyPad(hResTileLocal, hResGm_[hResOffset], copyParams, {false, 0, 0, 0});

    hResTileQueue_.EnQue<float>(hResTileLocal);
}

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::CopyInX(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t dStart = dIdx * tilingData_->dInner;
    int64_t xBase = bsIdx * tilingData_->n * tilingData_->d;
    int64_t xOffset = xBase + dStart;
    LocalTensor<T> xTileLocal = xTileQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(tilingData_->n), static_cast<uint32_t>(dNum * sizeof(T)),
                                    static_cast<uint32_t>((tilingData_->d - dNum) * sizeof(T)), 0, 0};
    DataCopyPad(xTileLocal, xGm_[xOffset], copyParams, {false, 0, 0, 0});

    xTileQueue_.EnQue<T>(xTileLocal);
}

REGBASE_TEMPLATE_DECLARE
__aicore__ inline void MhcPostRegbase<REGBASE_TEMPLATE_ARGS>::CopyOutY(int64_t bsIdx, int64_t nIdx, int64_t nNum,
                                                                       int64_t dIdx, int64_t dNum)
{
    int64_t dStart = dIdx * tilingData_->dInner;
    int64_t nStart = nIdx * tilingData_->nInner * tilingData_->d;
    int64_t yBase = bsIdx * tilingData_->n * tilingData_->d;
    int64_t yOffset = yBase + nStart + dStart;
    LocalTensor<T> yTileLocal = yTileQueue_.DeQue<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(nNum), static_cast<uint32_t>(dNum * sizeof(T)), 0,
                                    static_cast<uint32_t>((tilingData_->d - dNum) * sizeof(T)), 0};

    DataCopyPad(outputGm_[yOffset], yTileLocal, copyParams);

    yTileQueue_.FreeTensor(yTileLocal);
}

} // namespace MhcPost

#endif // ASCENDC_MHC_POST_REGBASE_H
