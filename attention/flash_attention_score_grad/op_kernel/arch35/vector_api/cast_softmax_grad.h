/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_softmax_grad.h
 */
#ifndef CAST_SOFTMAX_GRAD_
#define CAST_SOFTMAX_GRAD_
#include "../flash_attention_score_grad_common.h"
#include "./vf_softmax_grad_front_cast.h"
#include "./vf_anti_quant_softmax_grad_front_cast.h"

using namespace commondef;

template <typename T1, typename T2, const uint32_t VECTOR_BASEM = 64, const uint32_t HEAD_DIM_ALIGN = 128,
          const bool IS_D_NO_EQUAL = false>
__aicore__ inline void CopyInSoftmaxGrad(FagConstInfo &constInfo, FagRunInfo &runInfo, int32_t loopIdx,
                                         int32_t curLoopSize, int32_t loopSize, TQue<QuePosition::VECIN, 1> &yInQue,
                                         TQue<QuePosition::VECIN, 1> &dxInQue, GlobalTensor<T1> &dxGm,
                                         GlobalTensor<T1> &yGm)
{
    int64_t srcGmOffset = 0;
    int64_t transpose_stride = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    if (constInfo.commonConstInfo.layoutType == BNGSD) {
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gS1Dv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.s1Dv;
        s1Offset = runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO * constInfo.commonConstInfo.dSizeV +
                   runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.dSizeV +
                   loopIdx * loopSize * constInfo.commonConstInfo.dSizeV;
        transpose_stride = 0;
    } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
        s1Offset = runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO * constInfo.commonConstInfo.bN2GDv +
                   runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.bN2GDv +
                   loopIdx * loopSize * constInfo.commonConstInfo.bN2GDv;
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GDv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
        transpose_stride = (constInfo.commonConstInfo.bN2GDv - constInfo.commonConstInfo.dSizeV) * sizeof(T1);
    } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dv;
        s1Offset = runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO * constInfo.commonConstInfo.n2GDv +
                   runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2GDv +
                   loopIdx * loopSize * constInfo.commonConstInfo.n2GDv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
        transpose_stride = (constInfo.commonConstInfo.n2GDv - constInfo.commonConstInfo.dSizeV) * sizeof(T1);
    } else if (constInfo.commonConstInfo.layoutType == TND) {
        int64_t tndS1PrefixSum = (runInfo.commonRunInfo.boIdx == 0 ?
                                      0 :
                                      ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
        bOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * constInfo.commonConstInfo.dSizeV;
        s1Offset = runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO * constInfo.commonConstInfo.n2GDv +
                   runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2GDv +
                   loopIdx * loopSize * constInfo.commonConstInfo.n2GDv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        transpose_stride = (constInfo.commonConstInfo.n2GDv - constInfo.commonConstInfo.dSizeV) * sizeof(T1);
    }
    srcGmOffset = bOffset + n2Offset + gOffset + s1Offset;

    LocalTensor<T1> yTensor = yInQue.AllocTensor<T1>();
    LocalTensor<T1> dxTensor = dxInQue.AllocTensor<T1>();

    if constexpr (IS_D_NO_EQUAL) {
        Duplicate<T1>(yTensor, 0, curLoopSize * HEAD_DIM_ALIGN);
        Duplicate<T1>(dxTensor, 0, curLoopSize * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)));
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)));
    }

    uint32_t dstBlockStride = (HEAD_DIM_ALIGN - constInfo.dAlignToBlock) * sizeof(T1) / 32;
    DataCopyPad(
        dxTensor, dxGm[srcGmOffset],
        {static_cast<uint16_t>(curLoopSize), static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(T1)),
         static_cast<uint32_t>(transpose_stride), dstBlockStride, 0},
        {true, 0, static_cast<uint8_t>((constInfo.dAlignToBlock - constInfo.commonConstInfo.dSizeV)), 0});
    DataCopyPad(
        yTensor, yGm[srcGmOffset],
        {static_cast<uint16_t>(curLoopSize), static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(T1)),
         static_cast<uint32_t>(transpose_stride), dstBlockStride, 0},
        {true, 0, static_cast<uint8_t>((constInfo.dAlignToBlock - constInfo.commonConstInfo.dSizeV)), 0});

    yInQue.EnQue(yTensor);
    dxInQue.EnQue(dxTensor);
}

template <typename T1, typename T2, const uint32_t VECTOR_BASEM = 64, const uint32_t HEAD_DIM_ALIGN = 128>
__aicore__ inline void CalculateCastSoftmaxGrad(FagConstInfo &constInfo, int32_t curLoopSize,
                                                TQue<QuePosition::VECIN, 1> &yInQue,
                                                TQue<QuePosition::VECIN, 1> &dxInQue, const LocalTensor<T2> &dstTensor)
{
    LocalTensor<T1> yTensor = yInQue.DeQue<T1>();
    LocalTensor<T1> dxTensor = dxInQue.DeQue<T1>();
    if constexpr (HEAD_DIM_ALIGN <= 256) {
        AscendC::MySoftmaxGradFrontCast<T1, T2, HEAD_DIM_ALIGN, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor,
                                                                                curLoopSize, constInfo.dAlignToBlock);
    } else {
        if (constInfo.dAlignToBlock <= 384 && !IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 384, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock <= 320 && IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 320, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 320 && constInfo.dAlignToBlock <= 384 && IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 384, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 384 && constInfo.dAlignToBlock <= 448 && IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 448, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 448 && constInfo.dAlignToBlock <= 512 && IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 512, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock <= 512 && !IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 512, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 512 && constInfo.dAlignToBlock <= 576 && IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 576, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 576 && constInfo.dAlignToBlock <= 640 && IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 640, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock <= 640 && !IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 640, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 640 && constInfo.dAlignToBlock <= 704 && IsSameType<T1, float>::value) {
            AscendC::MySoftmaxGradFrontCast<T1, T2, 704, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                         constInfo.dAlignToBlock);
        } else {
            AscendC::MySoftmaxGradFrontCast<T1, T2, HEAD_DIM_ALIGN, HEAD_DIM_ALIGN>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        }
    }
    yInQue.FreeTensor(yTensor);
    dxInQue.FreeTensor(dxTensor);
}

#endif
