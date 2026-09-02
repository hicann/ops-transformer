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
 * \file pse_atten_mask_muls_simple_softmax.h
 */

#ifndef PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX_
#define PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX_
#include "../flash_attention_score_grad_common.h"
#include "../../../../common/op_kernel/arch35/pse_arch35.h"
#include "../../../../common/op_kernel/arch35/attenmask_arch35.h"
#include "vf_muls_sel_simple_softmax.h"
#include "vf_muls_sel_simple_softmax_aligned256.h"

using namespace commondef;

template <typename T1, typename T2, const uint32_t IS_PSE = 0>
__aicore__ inline void CopyInPse(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo &pseInfo,
                                 TQue<QuePosition::VECIN, 1> &pseInQue, GlobalTensor<T1> &pseGm)
{
    // 调用公共函数，传入LocalTensor
    if constexpr (IS_PSE) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0 || pseInfo.pseType == 2 || pseInfo.pseType == 3) {
            return;
        }
        LocalTensor<T1> pseTensor = pseInQue.AllocTensor<T1>();
        // compute offset + copy
        PseCopyIn<T2, T1, IS_PSE>(pseTensor, pseGm, runInfo.commonRunInfo, constInfo.commonConstInfo, pseInfo);
        pseInQue.EnQue(pseTensor);
    }
}

template <typename T1, const uint32_t IS_PSE = 0>
__aicore__ inline void GenPse(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo &pseInfo,
                              TQue<QuePosition::VECIN, 1> &pseInQue)
{
    // 调用公共函数
    if constexpr (IS_PSE) {
        LocalTensor<T1> pseTensor = pseInQue.AllocTensor<T1>();
        // gen pse
        pseInQue.EnQue(pseTensor);
    }
}

template <const uint32_t IS_ATTEN_MASK = 0, const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void CopyInAttenMask(FagConstInfo &constInfo, FagRunInfo &runInfo, AttenMaskInfo &attenMaskInfo,
                                       TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                       TQue<QuePosition::VECIN, 1> &attenMaskInQuePre,
                                       GlobalTensor<uint8_t> &attenMaskGm)
{
    if constexpr (IS_ATTEN_MASK) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0) {
            return;
        }
        AttenMaskCopyIn<IS_ATTEN_MASK>(attenMaskInQue, attenMaskInQuePre, attenMaskGm, runInfo.commonRunInfo,
                                       constInfo.commonConstInfo, attenMaskInfo);
    }
}

template <typename T2, const uint32_t VECTOR_BASEM = 64, const uint32_t IS_ROPE = 0>
__aicore__ inline void CopyInMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                    TQue<QuePosition::VECIN, 1> &maxSumInQue, GlobalTensor<T2> &maxGm,
                                    GlobalTensor<T2> &sumGm)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = 0;
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) { // TND + TND
        int64_t dSize = constInfo.commonConstInfo.dSize;
        if constexpr (IS_ROPE) {
            dSize = constInfo.commonConstInfo.dSizeV;
        }
        maxSumGmOffset =
            (runInfo.commonRunInfo.queryOffset / dSize +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2G) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    } else if (constInfo.commonConstInfo.layoutType == TND) { // TND + BNS8
        int64_t tndS1PrefixSum = (runInfo.commonRunInfo.boIdx == 0 ?
                                      0 :
                                      ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
        int64_t actualS1Len = 0;
        maxSumGmOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
            actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[0];
        } else {
            actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                          ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
        }
        maxSumGmOffset +=
            ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) *
                 actualS1Len +
             runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    } else { // BNS8
        maxSumGmOffset = (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                               constInfo.commonConstInfo.gSize +
                           runInfo.commonRunInfo.goIdx) *
                              constInfo.commonConstInfo.s1Size +
                          runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
                          runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) *
                         MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    }
    LocalTensor<T2> maxSumTensor = maxSumInQue.AllocTensor<T2>();
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) {
        uint32_t srcStride = constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
    } else {
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
    }
    maxSumInQue.EnQue(maxSumTensor);
}

template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline int64_t GetBNSOffset(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    int64_t offset = 0;
    int64_t tndS1PrefixSum =
        (runInfo.commonRunInfo.boIdx == 0 ? 0 :
                                            ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
    int64_t actualS1Len = 0;
    offset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
        actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[0];
    } else {
        actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                      ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
    }
    offset +=
        ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) * actualS1Len +
         runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
         runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) *
        MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    return offset;
}

template <typename T2, const uint32_t VECTOR_BASEM = 64, const uint32_t IS_ROPE = 0>
__aicore__ inline void CopyInMaxSumWithSfmg(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                            TQue<QuePosition::VECIN, 1> &maxSumInQue, GlobalTensor<T2> &maxGm,
                                            GlobalTensor<T2> &sumGm, GlobalTensor<T2> &sfmgWorkspaceGm)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = 0;
    int64_t sfmgGmOffset = 0;
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) { // TND + TND
        int64_t dSize = constInfo.commonConstInfo.dSize;
        if constexpr (IS_ROPE) {
            dSize = constInfo.commonConstInfo.dSizeV;
        }
        maxSumGmOffset =
            (runInfo.commonRunInfo.queryOffset / dSize +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2G) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        sfmgGmOffset = GetBNSOffset<T2, VECTOR_BASEM>(constInfo, runInfo);
    } else if (constInfo.commonConstInfo.layoutType == TND) { // TND + BNS8
        maxSumGmOffset = GetBNSOffset<T2, VECTOR_BASEM>(constInfo, runInfo);
        sfmgGmOffset = maxSumGmOffset;
    } else { // BNS8
        maxSumGmOffset = (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                               constInfo.commonConstInfo.gSize +
                           runInfo.commonRunInfo.goIdx) *
                              constInfo.commonConstInfo.s1Size +
                          runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
                          runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) *
                         MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        sfmgGmOffset = maxSumGmOffset;
    }
    LocalTensor<T2> maxSumTensor = maxSumInQue.AllocTensor<T2>();
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) {
        uint32_t srcStride = constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
    } else {
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
    }
    DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2) << 1],
                sfmgWorkspaceGm[sfmgGmOffset / 8], {1, static_cast<uint16_t>(VECTOR_BASEM * sizeof(T2)), 0, 0},
                {false, 0, 0, 0});
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
template <typename T1, typename T2, const bool IS_FP8_INPUT = false, const uint32_t IS_ATTEN_MASK = 0,
          const uint32_t IS_PSE = 0, const uint32_t IS_DETER_OLD = 0, const uint32_t VECTOR_BASEM = 64,
          const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void CalculatePseMulsSelSimpleSoftMax(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo &pseInfo,
                                                        AttenMaskInfo &attenMaskInfo,
                                                        TQue<QuePosition::VECIN, 1> &maxSumInQue,
                                                        TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                                        TQue<QuePosition::VECIN, 1> &pseInQue,
                                                        LocalTensor<T2> &dstTensor, LocalTensor<T2> &srcTensor,
                                                        __gm__ uint8_t *pseSlope)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<T1> pseTensor;
    if constexpr (IS_ATTEN_MASK) {
        attenMaskTensor = attenMaskInQue.DeQue<uint8_t>();
    }
    // Compute
    LocalTensor<T2> maxSumTensor = maxSumInQue.DeQue<T2>();
    constexpr uint16_t CONVERT_VECTOR_BASEN = static_cast<uint16_t>(VECTOR_BASEN);
    if constexpr (IS_PSE) {
        float posShift;
        float slopes;
        // sparse mode 8非batch = 0下不需要加偏移
        if ((pseInfo.pseType == 2 || pseInfo.pseType == 3) && constInfo.commonConstInfo.layoutType == TND &&
            attenMaskInfo.compressMode == static_cast<uint8_t>(BAND_LEFT_UP_CASUAL) &&
            runInfo.commonRunInfo.boIdx != 0) {
            pseInfo.qStartIdx = 0;
            pseInfo.kvStartIdx = 0;
        }
        ComputeInnerPseOffset<T2, T1, IS_PSE>(slopes, posShift, runInfo.commonRunInfo, constInfo.commonConstInfo,
                                              pseInfo, pseSlope);
        pseTensor = (pseInfo.pseType == 0 || pseInfo.pseType == 1) ? pseInQue.DeQue<T1>() : pseInQue.AllocTensor<T1>();
        if (IS_FP8_INPUT) {
            AscendC::MulsSelSimpleSoftMaxAligned256<T1, T2, CONVERT_VECTOR_BASEN, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor,
                pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize, pseInfo.pseType,
                pseInfo.pseLayoutType, posShift, slopes);
        } else if (runInfo.commonRunInfo.s2RealSize > 64) {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor,
                pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize, pseInfo.pseType,
                pseInfo.pseLayoutType, posShift, slopes);
        } else {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor,
                pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize, pseInfo.pseType,
                pseInfo.pseLayoutType, posShift, slopes);
        }
        pseInQue.FreeTensor(pseTensor);
    } else {
        if (IS_FP8_INPUT) {
            AscendC::MulsSelSimpleSoftMaxAligned256<T1, T2, CONVERT_VECTOR_BASEN, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor,
                pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        } else if (runInfo.commonRunInfo.s2RealSize > 64) {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 128, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor,
                pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        } else {
            AscendC::MulsSelSimpleSoftMax<T1, T2, 64, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD>(
                dstTensor, maxSumTensor, maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], srcTensor,
                pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        }
    }
    // FreeTensor
    if constexpr (IS_ATTEN_MASK) {
        attenMaskInQue.FreeTensor(attenMaskTensor);
    }
    maxSumInQue.FreeTensor(maxSumTensor);
}

#endif
