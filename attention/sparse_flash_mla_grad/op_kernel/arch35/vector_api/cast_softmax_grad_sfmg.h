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
 * \file cast_softmax_grad_sfmg.h
 */
#ifndef CAST_SOFTMAX_GRAD__SFMG
#define CAST_SOFTMAX_GRAD__SFMG
#include "../sparse_flash_mla_grad_arch35_common.h"
#include "./vf_softmax_grad_front_cast_sfmg.h"

using namespace commondef;

template <typename T1, typename T2, typename OUTDTYPE, const uint32_t VECTOR_BASEM = 64,
          const uint32_t HEAD_DIM_ALIGN = 128, const bool IS_D_NO_EQUAL = false>
__aicore__ inline void CopyInSoftmaxGrad(FagConstInfo &constInfo, FagRunInfo &runInfo, int32_t loopIdx,
                                         int32_t curLoopSize, int32_t loopSize, TQue<QuePosition::VECIN, 1> &yInQue,
                                         TQue<QuePosition::VECIN, 1> &dxInQue, GlobalTensor<T1> &dxGm,
                                         GlobalTensor<OUTDTYPE> &yGm)
{
    constexpr static bool IS_FP8_INPUT = IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value;
    int64_t srcGmOffset = 0;
    int64_t transpose_stride = 0;
    int64_t transpose_stride_for_out = 0;
    srcGmOffset = runInfo.t1Index * constInfo.commonConstInfo.n2GDv +
                  runInfo.firstHalfGRealSize * GetSubBlockIdx() * constInfo.commonConstInfo.dSizeV +
                  loopIdx * loopSize * constInfo.commonConstInfo.dSizeV;

    LocalTensor<OUTDTYPE> yTensor = yInQue.AllocTensor<OUTDTYPE>();
    LocalTensor<T1> dxTensor = dxInQue.AllocTensor<T1>();

    Duplicate<OUTDTYPE>(yTensor, 0, curLoopSize * HEAD_DIM_ALIGN);
    Duplicate<T1>(dxTensor, 0, curLoopSize * HEAD_DIM_ALIGN);
    SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)));
    WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)));

    uint32_t dstBlockStride = (HEAD_DIM_ALIGN - constInfo.dAlignToBlock) * sizeof(T1) / 32;
    uint32_t dstBlockStrideForOut = (HEAD_DIM_ALIGN - constInfo.dAlignToBlockForFp8) * sizeof(OUTDTYPE) / 32;

    DataCopyPad(
        dxTensor, dxGm[srcGmOffset],
        {static_cast<uint16_t>(curLoopSize), static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(T1)),
         static_cast<uint32_t>(transpose_stride), dstBlockStride, 0},
        {true, 0, static_cast<uint8_t>((constInfo.dAlignToBlock - constInfo.commonConstInfo.dSizeV)), 0});
    DataCopyPad(
        yTensor, yGm[srcGmOffset],
        {static_cast<uint16_t>(curLoopSize), static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(OUTDTYPE)),
         static_cast<uint32_t>(transpose_stride_for_out), dstBlockStrideForOut, 0},
        {true, 0, static_cast<uint8_t>((constInfo.dAlignToBlockForFp8 - constInfo.commonConstInfo.dSizeV)), 0});

    yInQue.EnQue(yTensor);
    dxInQue.EnQue(dxTensor);
}

template <typename T1, typename T2, typename OUTDTYPE, const uint32_t VECTOR_BASEM = 64,
          const uint32_t HEAD_DIM_ALIGN = 128>
__aicore__ inline void CalculateCastSoftmaxGrad(FagConstInfo &constInfo, int32_t curLoopSize,
                                                TQue<QuePosition::VECIN, 1> &yInQue,
                                                TQue<QuePosition::VECIN, 1> &dxInQue, const LocalTensor<T2> &dstTensor,
                                                float deqScaleDyValue = 1.0)
{
    LocalTensor<OUTDTYPE> yTensor = yInQue.DeQue<OUTDTYPE>();
    LocalTensor<T1> dxTensor = dxInQue.DeQue<T1>();
    AscendC::MySoftmaxGradFrontCast<T1, T2, 512, HEAD_DIM_ALIGN>(dstTensor, yTensor, dxTensor, curLoopSize,
                                                                 constInfo.dAlignToBlock);
    yInQue.FreeTensor(yTensor);
    dxInQue.FreeTensor(dxTensor);
}

#endif
