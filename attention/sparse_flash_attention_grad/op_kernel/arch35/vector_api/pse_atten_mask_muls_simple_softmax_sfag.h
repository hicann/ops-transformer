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
 * \file pse_atten_mask_muls_simple_softmax_sfag.h
 */

#ifndef PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX__SFAG
#define PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX__SFAG
#include "../common.h"
#include "../../../../common/op_kernel/arch35/pse_arch35.h"
#include "../../../../common/op_kernel/arch35/attenmask_arch35.h"
#include "vf_muls_sel_simple_softmax_sfag.h"

using namespace commondef;

template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline void CopyInMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                    TQue<QuePosition::VECIN, 1> &maxSumInQue, GlobalTensor<T2> &maxGm,
                                    GlobalTensor<T2> &sumGm)
{
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = 0;
    maxSumGmOffset = runInfo.t1Index * constInfo.commonConstInfo.gSize + runInfo.firstHalfGRealSize * GetSubBlockIdx();
    LocalTensor<T2> maxSumTensor = maxSumInQue.AllocTensor<T2>();

    DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                {static_cast<uint16_t>(runInfo.halfGRealSize), static_cast<uint16_t>(sizeof(float)), 0, 0}, {false, 0, 0, 0});
    DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                {static_cast<uint16_t>(runInfo.halfGRealSize), static_cast<uint16_t>(sizeof(float)), 0, 0}, {false, 0, 0, 0});
    maxSumInQue.EnQue(maxSumTensor);
}

/*************************
Function： VF计算函数，实现Pse + AttenMask + Muls + SimpleSoftmax计算
baseParams：循环定参，入参
runInfo: 循环变参，入参
attenMaskInfo：attenMask相关参数，入参
maxSumInQue：maxSum分配Que，入参
attenMaskInQue：attenMask分配Que，入参
pseInQue：pse分配Que，入参
dstTensor：返回计算结果，出参
srcTensor：VF输入，入参
*************************/
template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 0, const uint32_t IS_DETER_OLD = 0,
          const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void
CalculatePseMulsSelSimpleSoftMax(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo& pseInfo, AttenMaskInfo &attenMaskInfo,
                                 TQue<QuePosition::VECIN, 1> &maxSumInQue, TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                 TQue<QuePosition::VECIN, 1> &pseInQue, LocalTensor<T2> &dstTensor, LocalTensor<T2> &srcTensor, 
                                 __gm__ uint8_t *pseSlope)
{
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<T1> pseTensor;
    // Compute
    LocalTensor<T2> maxSumTensor = maxSumInQue.DeQue<T2>();

    if (runInfo.commonRunInfo.s2RealSize > 64) {
        AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
        dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor, 
        attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.halfGRealSize,
        runInfo.commonRunInfo.s2RealSize);
    } else {
        AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
        dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor, pseTensor, 
        attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.halfGRealSize,
        runInfo.commonRunInfo.s2RealSize);
    }
    maxSumInQue.FreeTensor(maxSumTensor);
}

#endif