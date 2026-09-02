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
 * \file pse_atten_mask_muls_simple_softmax_sfmg.h
 */

#ifndef PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX__SFMG
#define PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX__SFMG
#include "../sparse_flash_mla_grad_arch35_common.h"
#include "../../../../../../../../attention/common/op_kernel/arch35/pse_arch35.h"
#include "../../../../../../../../attention/common/op_kernel/arch35/attenmask_arch35.h"
#include "vf_muls_sel_simple_softmax_sfmg.h"

using namespace commondef;

template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline void CopyInSink(FagConstInfo &constInfo, TQue<QuePosition::VECIN, 1> &sinkQue,
                                  GlobalTensor<T2> &sinkGm)
{
    int64_t sinkGmOffset = 0;
    sinkGmOffset = constInfo.firstHalfGRealSize * GetSubBlockIdx();
    LocalTensor<T2> sinkTensor = sinkQue.AllocTensor<T2>();
    DataCopyPad(sinkTensor, sinkGm[sinkGmOffset],
                {1, static_cast<uint16_t>(constInfo.halfGRealSize * sizeof(T2)), 0, 0}, {false, 0, 0, 0});
    sinkQue.EnQue(sinkTensor);
    sinkQue.DeQue<T2>();
    sinkQue.FreeTensor(sinkTensor);
}

template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline void CopyInLse(FagConstInfo &constInfo, FagRunInfo &runInfo, TQue<QuePosition::VECIN, 1> &lseInQue,
                                 GlobalTensor<T2> &lseGm)
{
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    int64_t lseGmOffset = 0;
    lseGmOffset = runInfo.t1Index * constInfo.commonConstInfo.gSize + runInfo.firstHalfGRealSize * GetSubBlockIdx();
    LocalTensor<T2> lseTensor = lseInQue.AllocTensor<T2>();
    DataCopyPad(lseTensor, lseGm[lseGmOffset], {1, static_cast<uint16_t>(runInfo.halfGRealSize * sizeof(T2)), 0, 0},
                {false, 0, 0, 0});
    lseInQue.EnQue(lseTensor);
    lseInQue.DeQue<T2>();
    lseInQue.FreeTensor(lseTensor);
}

/*************************
Function： VF计算函数，实现Pse + AttenMask + Muls + SimpleSoftmax计算
baseParams：循环定参，入参
runInfo: 循环变参，入参
attenMaskInfo：attenMask相关参数，入参
lseInQue：lse分配Que，入参
attenMaskInQue：attenMask分配Que，入参
pseInQue：pse分配Que，入参
dstTensor：返回计算结果，出参
srcTensor：VF输入，入参
*************************/
template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 0,
          const uint32_t IS_DETER_OLD = 0, const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128,
          const uint32_t IsSparseIndicesExist = 0>
__aicore__ inline void CalculatePseMulsSelSimpleSoftMax(
    FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo &pseInfo, AttenMaskInfo &attenMaskInfo,
    TQue<QuePosition::VECIN, 1> &lseInQue, TQue<QuePosition::VECIN, 1> &attenMaskInQue,
    TQue<QuePosition::VECIN, 1> &pseInQue, LocalTensor<T2> &dstTensor, LocalTensor<T2> &srcTensor,
    LocalTensor<T2> &softmaxL1Tensor, __gm__ uint8_t *pseSlope, float pScalar)
{
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<T1> pseTensor;
    // Compute
    LocalTensor<T2> lseTensor = lseInQue.AllocTensor<T2>();
    lseInQue.EnQue(lseTensor);
    lseInQue.DeQue<T2>();

    if (runInfo.commonRunInfo.s2RealSize > 64) {
        AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD, IsSparseIndicesExist>(
            dstTensor, lseTensor, lseTensor, srcTensor, softmaxL1Tensor, pseTensor, attenMaskTensor,
            constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.halfGRealSize, runInfo.commonRunInfo.s2RealSize,
            pScalar);
    } else {
        AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD, IsSparseIndicesExist>(
            dstTensor, lseTensor, lseTensor, srcTensor, softmaxL1Tensor, pseTensor, attenMaskTensor,
            constInfo.scaleValue, constInfo.attenMaskMinValue, runInfo.halfGRealSize, runInfo.commonRunInfo.s2RealSize,
            pScalar);
    }
    lseInQue.FreeTensor(lseTensor);
}

#endif
