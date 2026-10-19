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
 * \file pse_atten_mask_muls_simple_softmax_david.h
 */

#ifndef PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX_DAVID_
#define PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX_DAVID_
#include "../flash_attention_score_grad_common_david.h"
#include "david/pse_david.h"
#include "david/attenmask_david.h"
#include "vf_muls_sel_simple_softmax_david.h"

using namespace commondef;

template <typename T1, typename T2, const uint32_t IS_PSE = 0>
__aicore__ inline void CopyInPseReuseBuf(FagConstInfo& constInfo, FagRunInfo& runInfo, PseInfo& pseInfo,
                                         TBuf<>& reusedBuf0, GlobalTensor<T1>& pseGm)
{
    // 调用公共函数，传入LocalTensor
    if constexpr (IS_PSE) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0 || pseInfo.pseType == 2 || pseInfo.pseType == 3) {
            return;
        }
        LocalTensor<T1> pseTensor = reusedBuf0.Get<T1>();
        // compute offset + copy
        PseCopyIn<T2, T1, IS_PSE>(pseTensor, pseGm, runInfo.commonRunInfo, constInfo.commonConstInfo, pseInfo);
    }
}

template <typename T1, typename T2, const uint32_t IS_PSE = 0>
__aicore__ inline void CopyInPse(FagConstInfo& constInfo, FagRunInfo& runInfo, PseInfo& pseInfo,
                                 TQue<QuePosition::VECIN, 1>& pseInQue, GlobalTensor<T1>& pseGm)
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
__aicore__ inline void GenPse(FagConstInfo& constInfo, FagRunInfo& runInfo, PseInfo& pseInfo,
                              TQue<QuePosition::VECIN, 1>& pseInQue)
{
    // 调用公共函数
    if constexpr (IS_PSE) {
        LocalTensor<T1> pseTensor = pseInQue.AllocTensor<T1>();
        // gen pse
        pseInQue.EnQue(pseTensor);
    }
}

template <const uint32_t IS_ATTEN_MASK = 0, const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void CopyInAttenMask(FagConstInfo& constInfo, FagRunInfo& runInfo, AttenMaskInfo& attenMaskInfo,
                                       TBuf<>& attenMaskBuf, GlobalTensor<uint8_t>& attenMaskGm)
{
    if constexpr (IS_ATTEN_MASK) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0) {
            return;
        }
        AttenMaskCopyIn(attenMaskBuf, attenMaskGm, runInfo.commonRunInfo, constInfo.commonConstInfo, attenMaskInfo);
    }
}

template <const uint32_t IS_ATTEN_MASK = 0, const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void CopyInAttenMaskDn(FagConstInfo& constInfo, FagRunInfo& runInfo, AttenMaskInfo& attenMaskInfo,
                                         TBuf<>& attenMaskBuf, GlobalTensor<uint8_t>& attenMaskGm)
{
    if constexpr (IS_ATTEN_MASK) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0) {
            return;
        }
        AttenMaskCopyIn(attenMaskBuf, attenMaskGm, runInfo.commonRunInfo, constInfo.commonConstInfo, attenMaskInfo,
                        true);
        SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    }
}

template <const uint32_t IS_ATTEN_MASK = 0, const uint32_t VECTOR_BASEM = 64, const uint32_t VECTOR_BASEN = 128>
__aicore__ inline void CopyInAttenMask(FagConstInfo& constInfo, FagRunInfo& runInfo, AttenMaskInfo& attenMaskInfo,
                                       TQue<QuePosition::VECIN, 1>& attenMaskInQue,
                                       TQue<QuePosition::VECIN, 1>& attenMaskInQuePre,
                                       GlobalTensor<uint8_t>& attenMaskGm)
{
    if constexpr (IS_ATTEN_MASK) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0) {
            return;
        }
        AttenMaskCopyIn<IS_ATTEN_MASK>(attenMaskInQue, attenMaskInQuePre, attenMaskGm, runInfo.commonRunInfo,
                                       constInfo.commonConstInfo, attenMaskInfo);
    }
}

template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline void CopyInMaxSum(FagConstInfo& constInfo, FagRunInfo& runInfo,
                                    TQue<QuePosition::VECIN, 1>& maxSumInQue, GlobalTensor<T2>& maxGm,
                                    GlobalTensor<T2>& sumGm)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = 0;
    if (constInfo.commonConstInfo.layoutType == TND) {
        int64_t actualS1Len = 0;
        for (int64_t bIdx = 0; bIdx < runInfo.commonRunInfo.boIdx; bIdx++) {
            if (unlikely(bIdx == 0)) {
                actualS1Len = ((__gm__ int64_t*)constInfo.seqS1_addr)[0];
            } else {
                actualS1Len =
                    ((__gm__ int64_t*)constInfo.seqS1_addr)[bIdx] - ((__gm__ int64_t*)constInfo.seqS1_addr)[bIdx - 1];
            }
            maxSumGmOffset += actualS1Len * constInfo.n2Size * constInfo.commonConstInfo.gSize *
                              MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        }
        if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
            actualS1Len = ((__gm__ int64_t*)constInfo.seqS1_addr)[0];
        } else {
            actualS1Len = ((__gm__ int64_t*)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                          ((__gm__ int64_t*)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
        }
        maxSumGmOffset +=
            ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) *
                 actualS1Len +
             runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
             runInfo.commonRunInfo.firstHalfS1RealSize * get_subblockid()) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    } else {
        // todo: 展开乘减少scalar
        maxSumGmOffset = (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                               constInfo.commonConstInfo.gSize +
                           runInfo.commonRunInfo.goIdx) *
                              constInfo.commonConstInfo.s1Size +
                          runInfo.commonRunInfo.s1oIdx * VECTOR_BASEM * CV_CORE_RATIO +
                          runInfo.commonRunInfo.firstHalfS1RealSize * get_subblockid()) *
                         MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    }
    LocalTensor<T2> maxSumTensor = maxSumInQue.AllocTensor<T2>();
    DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                {1, static_cast<uint16_t>(VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0}, {false, 0, 0, 0});
    DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2)], maxGm[maxSumGmOffset],
                {1, static_cast<uint16_t>(VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0}, {false, 0, 0, 0});
    maxSumInQue.EnQue(maxSumTensor);
}

template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline int64_t GetBNSOffset(FagConstInfo& constInfo, FagRunInfo& runInfo)
{
    int64_t offset = 0;
    int64_t tndS1PrefixSum =
        (runInfo.commonRunInfo.boIdx == 0 ? 0 :
                                            ((__gm__ int64_t*)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
    int64_t actualS1Len = 0;
    offset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
        actualS1Len = ((__gm__ int64_t*)constInfo.seqS1_addr)[0];
    } else {
        actualS1Len = ((__gm__ int64_t*)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                      ((__gm__ int64_t*)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
    }
    offset +=
        ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) * actualS1Len +
         runInfo.commonRunInfo.s1oIdx * constInfo.commonConstInfo.s1BaseSize + runInfo.commonRunInfo.vecS1Offset) *
        MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    return offset;
}

template <typename T2, const uint32_t VECTOR_BASEM = 64>
__aicore__ inline void CopyInMaxSumWithTbuf(FagConstInfo& constInfo, FagRunInfo& runInfo, TBuf<>& maxSumBuf,
                                            GlobalTensor<T2>& maxGm, GlobalTensor<T2>& sumGm,
                                            GlobalTensor<T2>& sfmgWorkspaceGm)
{
    uint32_t maxSumTensorOffset = 0;
    uint32_t maxOffset = VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    uint32_t sfmgOffset = VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2) * 2;
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = 0;
    int64_t sfmgGmOffset = 0;
    uint16_t srcStride = 0;

    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) {
        maxSumGmOffset = (runInfo.queryGmOffset / constInfo.commonConstInfo.dSize +
                          runInfo.commonRunInfo.vecS1Offset * constInfo.commonConstInfo.n2G) *
                         MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        sfmgGmOffset = GetBNSOffset<T2, VECTOR_BASEM>(constInfo, runInfo);
    } else if (constInfo.commonConstInfo.layoutType == TND) {
        maxSumGmOffset = GetBNSOffset<T2, VECTOR_BASEM>(constInfo, runInfo);
        sfmgGmOffset = maxSumGmOffset;
    } else {
        // todo: 展开乘减少scalar
        maxSumGmOffset =
            (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                  constInfo.commonConstInfo.gSize +
              runInfo.commonRunInfo.goIdx) *
                 constInfo.commonConstInfo.s1Size +
             runInfo.commonRunInfo.s1oIdx * constInfo.commonConstInfo.s1BaseSize + runInfo.commonRunInfo.vecS1Offset) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        sfmgGmOffset = maxSumGmOffset;
    }
    LocalTensor<T2> maxSumTensor = maxSumBuf.Get<T2>();
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) {
        srcStride = constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
        DataCopyPad(maxSumTensor[maxSumTensorOffset], sumGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint16_t>(MAX_SUM_REDUCE_AXIS_SIZE), srcStride, 0},
                    {false, 0, 0, 0}, /*pre_alloc*/ true);
        DataCopyPad(maxSumTensor[maxSumTensorOffset + maxOffset], maxGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint16_t>(MAX_SUM_REDUCE_AXIS_SIZE), srcStride, 0},
                    {false, 0, 0, 0}, /*pre_alloc*/ true);
    } else {
        DataCopyPad(maxSumTensor[maxSumTensorOffset], sumGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE), srcStride, 0}, {false, 0, 0, 0},
                    /*pre_alloc*/ true);
        DataCopyPad(maxSumTensor[maxSumTensorOffset + maxOffset], maxGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE), srcStride, 0}, {false, 0, 0, 0},
                    /*pre_alloc*/ true);
    }
    DataCopyPad(maxSumTensor[maxSumTensorOffset + sfmgOffset], sfmgWorkspaceGm[sfmgGmOffset / 8],
                {1, static_cast<uint16_t>(VECTOR_BASEM * sizeof(T2)), 0, 0}, {false, 0, 0, 0}, /*pre_alloc*/ true);
}

template <typename T2, const uint32_t VECTOR_BASEN = 64>
__aicore__ inline void CopyInMaxSumWithTbufDn(FagConstInfo& constInfo, FagRunInfo& runInfo, TBuf<>& maxSumBuf,
                                              GlobalTensor<T2>& maxGm, GlobalTensor<T2>& sumGm,
                                              GlobalTensor<T2>& sfmgWorkspaceGm)
{
    uint32_t maxSumTensorOffset = 0;
    uint32_t maxOffset = VECTOR_BASEN;
    uint32_t sfmgOffset = VECTOR_BASEN * 2;
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = 0;
    if (constInfo.commonConstInfo.layoutType == TND) {
        int64_t actualS1Len = 0;
        for (int64_t bIdx = 0; bIdx < runInfo.commonRunInfo.boIdx; bIdx++) {
            if (unlikely(bIdx == 0)) {
                actualS1Len = ((__gm__ int64_t*)constInfo.seqS1_addr)[0];
            } else {
                actualS1Len =
                    ((__gm__ int64_t*)constInfo.seqS1_addr)[bIdx] - ((__gm__ int64_t*)constInfo.seqS1_addr)[bIdx - 1];
            }
            maxSumGmOffset += actualS1Len * constInfo.n2Size * constInfo.commonConstInfo.gSize *
                              MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
        }
        if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
            actualS1Len = ((__gm__ int64_t*)constInfo.seqS1_addr)[0];
        } else {
            actualS1Len = ((__gm__ int64_t*)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                          ((__gm__ int64_t*)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
        }
        maxSumGmOffset +=
            ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) *
                 actualS1Len +
             runInfo.commonRunInfo.s1oIdx * constInfo.commonConstInfo.s1BaseSize + runInfo.commonRunInfo.vecS1Offset) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    } else {
        // todo: 展开乘减少scalar
        maxSumGmOffset = (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                               constInfo.commonConstInfo.gSize +
                           runInfo.commonRunInfo.goIdx) *
                              constInfo.commonConstInfo.s1Size +
                          runInfo.commonRunInfo.s1oIdx * constInfo.commonConstInfo.s1BaseSize);
    }
    LocalTensor<T2> maxSumTensor = maxSumBuf.Get<T2>();
    DataCopyPad(maxSumTensor[maxSumTensorOffset], sumGm[maxSumGmOffset],
                {1, static_cast<uint16_t>(VECTOR_BASEN * sizeof(T2)), 0, 0}, {false, 0, 0, 0}, /*pre_alloc*/ true);
    DataCopyPad(maxSumTensor[maxSumTensorOffset + maxOffset], maxGm[maxSumGmOffset],
                {1, static_cast<uint16_t>(VECTOR_BASEN * sizeof(T2)), 0, 0}, {false, 0, 0, 0}, /*pre_alloc*/ true);
    DataCopyPad(maxSumTensor[maxSumTensorOffset + sfmgOffset], sfmgWorkspaceGm[maxSumGmOffset],
                {1, static_cast<uint16_t>(VECTOR_BASEN * sizeof(T2)), 0, 0}, {false, 0, 0, 0}, /*pre_alloc*/ true);
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
template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 0,
          const uint32_t IS_DETER = 0, const uint32_t VECTOR_BASEM = 64, const uint32_t CUBE_BASEN = 256,
          const bool useDn = false>
__aicore__ inline void CalculatePseMulsSelSimpleSoftMaxWithTbuf(
    FagConstInfo& constInfo, FagRunInfo& runInfo, PseInfo& pseInfo, AttenMaskInfo& attenMaskInfo, TBuf<>& maxSumBuf,
    TBuf<>& attenMaskBuf, LocalTensor<T2>& dstPTensor, LocalTensor<T2>& srcPTensor, LocalTensor<T2>& dstDsTensor,
    LocalTensor<T2>& srcDsTensor, __gm__ uint8_t* pseSlope, TBuf<>& reusedBuf0, float deqScaleQK, float deqScaleDyV)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<T1> pseTensor;
    if constexpr (IS_ATTEN_MASK) {
        attenMaskTensor = attenMaskBuf.Get<uint8_t>();
    }
    // Compute
    LocalTensor<T2> maxSumTensor = maxSumBuf.Get<T2>();
    uint32_t maxSumTensorOffset = 0;
    uint32_t maxOffset = VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    if constexpr (useDn) {
        maxSumTensorOffset = runInfo.commonRunInfo.vecS1Offset;
        maxOffset = CUBE_BASEN; // CUBE_BASEN
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
            pseTensor = reusedBuf0.Get<T1>();
            ComputeInnerPseOffset<T2, T1, IS_PSE>(slopes, posShift, runInfo.commonRunInfo, constInfo.commonConstInfo,
                                                  pseInfo, pseSlope);
            if (runInfo.commonRunInfo.s2RealSize > 64) {
                AscendC::ComputeSimpleSoftMaxAndSubMulDn<T1, T2, 128, CUBE_BASEN, IS_ATTEN_MASK, IS_PSE, IS_DETER>(
                    dstPTensor, dstDsTensor, maxSumTensor[maxSumTensorOffset],
                    maxSumTensor[maxSumTensorOffset + maxOffset], maxSumTensor[maxSumTensorOffset + maxOffset * 2],
                    srcPTensor, srcDsTensor, pseTensor, attenMaskTensor, constInfo.scaleValue * deqScaleQK, deqScaleDyV,
                    constInfo.attenMaskMinValue, runInfo.commonRunInfo.s2VecSize, runInfo.commonRunInfo.halfS1RealSize,
                    pseInfo.pseType, pseInfo.pseLayoutType, posShift, slopes);
            } else {
                AscendC::ComputeSimpleSoftMaxAndSubMulDn<T1, T2, 64, CUBE_BASEN, IS_ATTEN_MASK, IS_PSE, IS_DETER>(
                    dstPTensor, dstDsTensor, maxSumTensor[maxSumTensorOffset],
                    maxSumTensor[maxSumTensorOffset + maxOffset], maxSumTensor[maxSumTensorOffset + maxOffset * 2],
                    srcPTensor, srcDsTensor, pseTensor, attenMaskTensor, constInfo.scaleValue * deqScaleQK, deqScaleDyV,
                    constInfo.attenMaskMinValue, runInfo.commonRunInfo.s2VecSize, runInfo.commonRunInfo.halfS1RealSize,
                    pseInfo.pseType, pseInfo.pseLayoutType, posShift, slopes);
            }
        } else {
            if (runInfo.commonRunInfo.halfS1RealSize > 64) {
                AscendC::ComputeSimpleSoftMaxAndSubMulDn<T1, T2, 128, CUBE_BASEN, IS_ATTEN_MASK, IS_PSE, IS_DETER>(
                    dstPTensor, dstDsTensor, maxSumTensor[maxSumTensorOffset],
                    maxSumTensor[maxSumTensorOffset + maxOffset], maxSumTensor[maxSumTensorOffset + maxOffset * 2],
                    srcPTensor, srcDsTensor, pseTensor, attenMaskTensor, constInfo.scaleValue * deqScaleQK, deqScaleDyV,
                    constInfo.attenMaskMinValue, runInfo.commonRunInfo.s2VecSize, runInfo.commonRunInfo.halfS1RealSize);
            } else {
                AscendC::ComputeSimpleSoftMaxAndSubMulDn<T1, T2, 64, CUBE_BASEN, IS_ATTEN_MASK, IS_PSE, IS_DETER>(
                    dstPTensor, dstDsTensor, maxSumTensor[maxSumTensorOffset],
                    maxSumTensor[maxSumTensorOffset + maxOffset], maxSumTensor[maxSumTensorOffset + maxOffset * 2],
                    srcPTensor, srcDsTensor, pseTensor, attenMaskTensor, constInfo.scaleValue * deqScaleQK, deqScaleDyV,
                    constInfo.attenMaskMinValue, runInfo.commonRunInfo.s2VecSize, runInfo.commonRunInfo.halfS1RealSize);
            }
        }
    } else {
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
            pseTensor = reusedBuf0.Get<T1>();
            ComputeInnerPseOffset<T2, T1, IS_PSE>(slopes, posShift, runInfo.commonRunInfo, constInfo.commonConstInfo,
                                                  pseInfo, pseSlope);
            if (runInfo.commonRunInfo.s2RealSize > 64) {
                AscendC::ComputeSimpleSoftMaxAndSubMul<T1, T2, 128, VECTOR_BASEM, IS_ATTEN_MASK, IS_PSE, IS_DETER>(
                    dstPTensor, dstDsTensor, maxSumTensor[maxSumTensorOffset],
                    maxSumTensor[maxSumTensorOffset + maxOffset], srcPTensor, srcDsTensor, pseTensor, attenMaskTensor,
                    constInfo.scaleValue * deqScaleQK, deqScaleDyV, constInfo.attenMaskMinValue,
                    runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize, pseInfo.pseType,
                    pseInfo.pseLayoutType, posShift, slopes);
            } else {
                AscendC::ComputeSimpleSoftMaxAndSubMul<T1, T2, 64, VECTOR_BASEM, IS_ATTEN_MASK, IS_PSE, IS_DETER>(
                    dstPTensor, dstDsTensor, maxSumTensor[maxSumTensorOffset],
                    maxSumTensor[maxSumTensorOffset + maxOffset], srcPTensor, srcDsTensor, pseTensor, attenMaskTensor,
                    constInfo.scaleValue * deqScaleQK, deqScaleDyV, constInfo.attenMaskMinValue,
                    runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize, pseInfo.pseType,
                    pseInfo.pseLayoutType, posShift, slopes);
            }
        } else {
            if (runInfo.commonRunInfo.s2RealSize > 64) {
                AscendC::ComputeSimpleSoftMaxAndSubMul<T1, T2, 128, VECTOR_BASEM, IS_ATTEN_MASK, IS_PSE, IS_DETER>(
                    dstPTensor, dstDsTensor, maxSumTensor[maxSumTensorOffset],
                    maxSumTensor[maxSumTensorOffset + maxOffset], srcPTensor, srcDsTensor, pseTensor, attenMaskTensor,
                    constInfo.scaleValue * deqScaleQK, deqScaleDyV, constInfo.attenMaskMinValue,
                    runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
            } else {
                AscendC::ComputeSimpleSoftMaxAndSubMul<T1, T2, 64, VECTOR_BASEM, IS_ATTEN_MASK, IS_PSE, IS_DETER>(
                    dstPTensor, dstDsTensor, maxSumTensor[maxSumTensorOffset],
                    maxSumTensor[maxSumTensorOffset + maxOffset], srcPTensor, srcDsTensor, pseTensor, attenMaskTensor,
                    constInfo.scaleValue * deqScaleQK, deqScaleDyV, constInfo.attenMaskMinValue,
                    runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
            }
        }
    }
}

#endif
