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
 * \file weight_quant_vcv_basic_block.h
 * \brief
 */
#ifndef GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
#define GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_H

#include "../common/basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_tool.h"
#include "gmm_fr_weight_quant_cube_compute.h"
#include "gmm_fr_weight_quant_vec_compute.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;

namespace WeightQuantBatchMatmulV2::Arch35 {
#define GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM \
    template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, \
              typename perTokenScaleType, typename biasType, typename yType, typename sharedInputDType, \
              typename logitsType, typename rowIndexType, bool hasBias, const WqmmConfig &wqmmConfig, \
              const VecAntiQuantConfig &vecConfig>

#define GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS \
    GMMFRWeightQuantVcvBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, \
                                  sharedInputDType, logitsType, rowIndexType, hasBias, wqmmConfig, vecConfig>
GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
class GMMFRWeightQuantVcvBasicBlock {
public:
    __aicore__ inline GMMFRWeightQuantVcvBasicBlock(){};
    __aicore__ inline void Init(uint64_t antiQuantGroupSize, __gm__ yType *y, float sharedInputWeight);
    __aicore__ inline void InitAtomicGm(uint64_t initSize, uint64_t sharedInputStartSize, uint64_t sharedInputSize,
                                        __gm__ sharedInputDType *shareInputAddr);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight,
                                            __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
                                            __gm__ perTokenScaleType *perTokenScale, __gm__ biasType *bias,
                                            __gm__ yType *y, __gm__ logitsType *logits, __gm__ rowIndexType *rowIndex,
                                            const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                             const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void End(const BasicBlockOffsetParam &curOffsetParam);

protected:
    __aicore__ inline void InitGmZeroWithIterate(uint64_t &yGmStartOffset, uint64_t yGmEndOffset,
                                                 uint64_t sharedInputStartSize, uint64_t sharedInputSize);
    __aicore__ inline void IterateNzNkWithAiv(const BasicBlockOffsetParam &curOffsetParam,
                                              const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void VecComputeNzNkWithStartLimit(uint64_t &kMte2Offset, uint64_t kMte2Limit,
                                                        const BasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void IterateNzNkWithKAic(const BasicBlockOffsetParam &curOffsetParam,
                                               const BasicBlockOffsetParam &lastOffsetParam);

    template <const pipe_t pipe>
    __aicore__ inline void SetAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void SetAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    GMMFRWeightQuantVecCompute<xType, wType, biasType, yType, sharedInputDType, logitsType, rowIndexType, hasBias,
                               wqmmConfig, vecConfig>
        vecCompute_;
    GMMFRWeightQuantCubeCompute<xType, biasType, antiQuantScaleType, perTokenScaleType, float, hasBias, wqmmConfig>
        cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
    uint64_t weightL1DbOffset_ = 0;
    uint64_t biasL1DbOffset_ = 0;
    uint64_t rlLoopIdx_ = 0;

    LocalTensor<xType> weightL1_;
    LocalTensor<biasType> biasL1_;
    LocalTensor<float> ubOutputF32Buffer_;
};

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::Init(uint64_t antiQuantGroupSize, __gm__ yType *y,
                                                                       float sharedInputWeight)
{
    biasL1DbOffset_ = 0;
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256; // weight单块大小的标准shape是256*256
    static constexpr uint64_t MX_BIAS_L1_SIZE = BIAS_L1_SIZE * GetKBUnit<biasType>() * sizeof(biasType);
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;
    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;
    biasL1_ = LocalTensor<biasType>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(biasType));
    if constexpr (hasBias) {
        biasL1DbOffset_ = (l1RemainSize - MX_BIAS_L1_SIZE) / sizeof(biasType);
        l1RemainSize -= DOUBLE_BUFFER_NUM * MX_BIAS_L1_SIZE;
        l1StartSize += MX_BIAS_L1_SIZE;
    }

    constexpr uint64_t ubOffset =
        WeightQuantBatchMatmulV2::Arch35::GetGmmFRMxA8W4BufferInfo<vecConfig>().weightLowbitTotalSize;
    constexpr uint64_t highBitSize =
        WeightQuantBatchMatmulV2::Arch35::GetGmmFRMxA8W4BufferInfo<vecConfig>().weightLowbitTotalSize;
    ubOutputF32Buffer_ = LocalTensor<float>(TPosition::LCM, ubOffset, highBitSize);

    if ASCEND_IS_AIC {
        cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize, biasL1DbOffset_, biasL1_);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    } else {
        vecCompute_.Init(sharedInputWeight, y);
    }
    cvLoopIdx_ = 0;
}

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::InitAtomicGm(uint64_t initSize,
                                                                               uint64_t sharedInputStartSize,
                                                                               uint64_t sharedInputSize,
                                                                               __gm__ sharedInputDType *shareInputAddr)
{
    constexpr uint64_t initZeroBufferSize =
        GetGmmFRMxA8W4BufferInfo<vecConfig>().weightHighBitSingleBufferSize / sizeof(float);
    uint64_t yGmOffset = GetBlockIdx() * initZeroBufferSize;
    // 一次直接将GM上的result上需要初始化为0的部分全部清零
    InitGmZeroWithIterate(yGmOffset, initSize, sharedInputStartSize, sharedInputSize);

    // shared input mte3写出
    // 初始化GM上的result上shareInput部分
    constexpr uint64_t mte2BufferSize =
        WeightQuantBatchMatmulV2::Arch35::GetGmmFRMxA8W4BufferInfo<vecConfig>().weightLowBitSingleBufferSize /
        sizeof(sharedInputDType);
    for (uint64_t sharedInputGmOffset = AscendC::GetBlockIdx() * mte2BufferSize; sharedInputGmOffset < sharedInputSize;
         sharedInputGmOffset += AscendC::GetBlockNum() * mte2BufferSize) {
        uint64_t initSharedInputRealSize = sharedInputGmOffset + mte2BufferSize > sharedInputSize ?
                                               sharedInputSize - sharedInputGmOffset :
                                               mte2BufferSize;
        vecCompute_.WaitVToMTE2();
        vecCompute_.CopyShareInputGmToUb(sharedInputGmOffset, initSharedInputRealSize, shareInputAddr);
        vecCompute_.CopyShareInputUbToGm(sharedInputGmOffset + sharedInputStartSize, initSharedInputRealSize);
        vecCompute_.SetVToMTE2();
    }
}

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::InitGmZeroWithIterate(uint64_t &yGmStartOffset,
                                                                                        uint64_t yGmEndOffset,
                                                                                        uint64_t sharedInputStartSize,
                                                                                        uint64_t sharedInputSize)
{
    constexpr uint64_t initZeroBufferSize =
        GetGmmFRMxA8W4BufferInfo<vecConfig>().weightHighBitSingleBufferSize / sizeof(float);
    uint64_t bufferSize = initZeroBufferSize;
    for (; yGmStartOffset < yGmEndOffset; yGmStartOffset += AscendC::GetBlockNum() * bufferSize) {
        uint64_t initZeroRealSize =
            yGmStartOffset + bufferSize > yGmEndOffset ? yGmEndOffset - yGmStartOffset : bufferSize;
        if (yGmStartOffset >= sharedInputStartSize) {
            vecCompute_.InitGmToZero(yGmStartOffset + sharedInputSize, initZeroRealSize);
        } else if (yGmStartOffset + bufferSize <= sharedInputStartSize) {
            vecCompute_.InitGmToZero(yGmStartOffset, initZeroRealSize);
        } else {
            uint64_t initZeroFirstTail = sharedInputStartSize - yGmStartOffset;
            vecCompute_.InitGmToZero(yGmStartOffset, initZeroFirstTail);
            uint64_t initZeroSecondTail = initZeroRealSize - initZeroFirstTail;
            vecCompute_.InitGmToZero(sharedInputSize + sharedInputStartSize, initZeroSecondTail);
        }
    }
}

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::UpdateGlobalAddr(
    __gm__ xType *x, __gm__ wType *weight, __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
    __gm__ perTokenScaleType *perTokenScale, __gm__ biasType *bias, __gm__ yType *y, __gm__ logitsType *logits,
    __gm__ rowIndexType *rowIndex, const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(x, y, reinterpret_cast<__gm__ biasType *>(bias), antiquantScale, nullptr,
                                      perTokenScale);
    } else {
        // For MX A8W4: scale is perChannelScale (float*), perTokenScale is also float*
        vecCompute_.UpdateGlobalAddr(weight, bias, logits, rowIndex, weightL2Cacheable);
    }
}

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::ComputeBasicBlock(
    const BasicBlockOffsetParam &curOffsetParam, const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        IterateNzNkWithAiv(curOffsetParam, lastOffsetParam);
    } else {
        IterateNzNkWithKAic(curOffsetParam, lastOffsetParam);
    }
}

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithAiv(
    const BasicBlockOffsetParam &curOffsetParam, const BasicBlockOffsetParam &lastOffsetParam)
{
    uint64_t kMte2Offset = 0;
    uint64_t curCvLoopIdx = cvLoopIdx_;

    VecComputeNzNkWithStartLimit(kMte2Offset, Min(curOffsetParam.kSize, DOUBLE_BUFFER_NUM * curOffsetParam.kbL1Size),
                                 curOffsetParam);

    vecCompute_.WaitSToMTE2(rlLoopIdx_);
    vecCompute_.WaitFrVToMTE2(rlLoopIdx_);
    uint64_t mSingleVec = CeilDivide(curOffsetParam.mL1Size, 2UL);
    vecCompute_.CopyRowIndexLogitsGmToUb(curOffsetParam.mOffset + GetSubBlockIdx() * mSingleVec,
                                         GetSubBlockIdx() == 0 ? mSingleVec : curOffsetParam.mL1Size - mSingleVec,
                                         rlLoopIdx_);
    vecCompute_.SetMte2ToS(rlLoopIdx_);
    rlLoopIdx_++;
    if (curCvLoopIdx > 0) {
        // y反量化在vec上两core切m
        uint64_t lastBasicBlockMSize = lastOffsetParam.mL1Size - (lastOffsetParam.mL1Size >> 1);
        lastBasicBlockMSize = GetSubBlockIdx() == 0 ? lastBasicBlockMSize : (lastOffsetParam.mL1Size >> 1);

        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

        vecCompute_.MulLogits(ubOutputF32Buffer_, lastBasicBlockMSize, lastOffsetParam, rlLoopIdx_);
        vecCompute_.SetFrToMTE2(rlLoopIdx_);
        vecCompute_.WaitMte2ToS(rlLoopIdx_);
        vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_);
        vecCompute_.SetSToMTE2(rlLoopIdx_);
    }
    // todo 先不搞preload mte2, 收益有限
    VecComputeNzNkWithStartLimit(kMte2Offset, curOffsetParam.kSize, curOffsetParam);
}

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::VecComputeNzNkWithStartLimit(
    uint64_t &kMte2Offset, uint64_t kMte2Limit, const BasicBlockOffsetParam &curOffsetParam)
{
    uint64_t kMte2BaseSize = curOffsetParam.kbL1Size / DOUBLE_BUFFER_NUM;

    for (; kMte2Offset < kMte2Limit; kMte2Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        vecCompute_.WaitVToMTE2();
        uint64_t biasL1Offset = GetSubBlockIdx() * MX_BIAS_SINGLE_VECTOR_SIZE;
        uint64_t biasGmOffset = curOffsetParam.nOffset + biasL1Offset;
        uint64_t biasRealSize = biasL1Offset >= curOffsetParam.nL1Size ? 0 :
                                biasGmOffset + MX_BIAS_SINGLE_VECTOR_SIZE >= curOffsetParam.nSize ?
                                                                         curOffsetParam.nSize - biasGmOffset :
                                                                         MX_BIAS_SINGLE_VECTOR_SIZE;
        uint64_t kL1Offset = GetSubBlockIdx() * kMte2BaseSize;
        uint64_t mte2RealK = kMte2Offset + kL1Offset >= kMte2Limit ? 0 :
                             kMte2Offset + kL1Offset + kMte2BaseSize >= kMte2Limit ?
                                                                     kMte2Limit - kMte2Offset - kL1Offset :
                                                                     kMte2BaseSize;

        vecCompute_.CopyGmToUb(mte2RealK, kMte2Offset, kL1Offset, biasGmOffset, biasRealSize, curOffsetParam);

        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

        vecCompute_.WeightAntiQuantComputeNzNk(
            mte2RealK, biasRealSize, kMte2Offset,
            weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_ +
                      CeilAlign(curOffsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset],
            biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_ + biasL1Offset], curOffsetParam);
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        vecCompute_.SetVToMTE2();
    }
}

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithKAic(
    const BasicBlockOffsetParam &curOffsetParam, const BasicBlockOffsetParam &lastOffsetParam)
{
    if (cvLoopIdx_ > 0) {
        WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        cubeCompute_.GetTensorC(ubOutputF32Buffer_, lastOffsetParam);
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }

    for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize;
         kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbL1Offset + curOffsetParam.kbL1Size) >= curOffsetParam.kSize ?
                                    curOffsetParam.kSize - kbL1Offset :
                                    curOffsetParam.kbL1Size;
        cubeCompute_.WaitScaleMTE1ToMTE2(kbL1Offset);
        cubeCompute_.CopyMxScaleGmToL1(curOffsetParam, kbL1Offset);
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.CopyAAndBiasGmToL1(curOffsetParam, kbL1Offset, cvLoopIdx_);
        cubeCompute_.WaitWeightMTE1ToMTE2();
        WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
        cubeCompute_.PadL1IfNotAlign(weightL1_, weightL1DbOffset_, cvLoopIdx_, kbL1RealSize, curOffsetParam.nL1Size);
        cubeCompute_.LaunchMatmul(weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_], kbL1Offset, kbL1RealSize, cvLoopIdx_,
                                  curOffsetParam); // mte1 mmad fixp流水
        cubeCompute_.SetWeightMTE1ToMTE2();
        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.SetScaleMTE1ToMTE2(kbL1Offset, curOffsetParam);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}

GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_VCV_BASIC_BLOCK_CLASS::End(const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIC {
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(ubOutputF32Buffer_, lastOffsetParam);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync();
    } else {
        if (cvLoopIdx_ > 0) {
            uint64_t lastBasicBlockMSize = lastOffsetParam.mL1Size - (lastOffsetParam.mL1Size >> 1);
            lastBasicBlockMSize = GetSubBlockIdx() == 0 ? lastBasicBlockMSize : (lastOffsetParam.mL1Size >> 1);

            SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

            vecCompute_.MulLogits(ubOutputF32Buffer_, lastBasicBlockMSize, lastOffsetParam, rlLoopIdx_ - 1);

            vecCompute_.WaitMte2ToS(rlLoopIdx_ - 1);
            vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_ - 1);
            vecCompute_.SetSToMTE2(rlLoopIdx_ - 1);
        }
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        vecCompute_.End();
    }
}
} // namespace WeightQuantBatchMatmulV2::Arch35

#endif // GROUPED_MATMUL_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
