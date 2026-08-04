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
 * \file moe_gating_top_k_backward_regbase.h
 * \brief arch35 RegBase implementation.
 */
#ifndef MOE_GATING_TOP_K_BACKWARD_REGBASE_H
#define MOE_GATING_TOP_K_BACKWARD_REGBASE_H
#include "kernel_operator.h"
#include "../common.h"
#include "moe_gating_top_k_backward_tiling_def.h"
#include "basic_api/kernel_operator_utils_intf.h"
namespace MoeGatingTopKBackward {
using namespace AscendC;

constexpr uint32_t VL_FLOAT_SIZE = VECTOR_REG_WIDTH / sizeof(float);

constexpr MicroAPI::CastTrait CAST_TRAIT_B32_TO_B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void StoreRegTensorForDtype(__local_mem__ T *output, MicroAPI::RegTensor<float> &src,
                                              MicroAPI::MaskReg &mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> outFp16;
        MicroAPI::Cast<half, float, CAST_TRAIT_B32_TO_B16>(outFp16, src, mask);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(((__local_mem__ half *)output + offset), outFp16,
                                                                     mask);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> outBf16;
        MicroAPI::Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outBf16, src, mask);
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(
            ((__local_mem__ bfloat16_t *)output + offset), outBf16, mask);
    } else {
        MicroAPI::DataCopy(((__local_mem__ float *)output + offset), src, mask);
    }
}

template <typename T>
__aicore__ inline void DenseSigmoidGradRegbase(LocalTensor<float> xNormTensor, LocalTensor<float> gradNormTensor,
                                               LocalTensor<T> outTensor, uint32_t count)
{
    __local_mem__ float *xNorm = (__local_mem__ float *)xNormTensor.GetPhyAddr();
    __local_mem__ float *gradNorm = (__local_mem__ float *)gradNormTensor.GetPhyAddr();
    __local_mem__ T *out = (__local_mem__ T *)outTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> xReg;
        MicroAPI::RegTensor<float> oneReg;
        MicroAPI::RegTensor<float> oneMinusReg;
        MicroAPI::RegTensor<float> gradNormReg;
        MicroAPI::RegTensor<float> outReg;
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<float>();
        const uint16_t loopNum = static_cast<uint16_t>(CeilDiv(count, VL_FLOAT_SIZE));

        for (uint16_t i = 0; i < loopNum; ++i) {
            uint32_t remain = count - static_cast<uint32_t>(i) * VL_FLOAT_SIZE;
            if (remain > VL_FLOAT_SIZE) {
                remain = VL_FLOAT_SIZE;
            }
            mask = MicroAPI::UpdateMask<float>(remain);
            const uint32_t offset = static_cast<uint32_t>(i) * VL_FLOAT_SIZE;
            MicroAPI::DataCopy(xReg, xNorm + offset);
            MicroAPI::Duplicate(oneReg, 1.0f, mask);
            MicroAPI::Sub(oneMinusReg, oneReg, xReg, mask);
            MicroAPI::Mul(outReg, xReg, oneMinusReg, mask);
            MicroAPI::DataCopy(gradNormReg, gradNorm + offset);
            MicroAPI::Mul(outReg, outReg, gradNormReg, mask);
            StoreRegTensorForDtype<T>(out, outReg, mask, offset);
        }
    }
}

template <typename T>
class MoeGatingTopKBackwardRegbase {
public:
    __aicore__ inline MoeGatingTopKBackwardRegbase(){};
    __aicore__ inline void Init(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx, GM_ADDR gradX, GM_ADDR workspace,
                                const MoeGatingTopKBackwardRegbaseTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GetGatherOffsetIndex();
    __aicore__ inline void CopyInGradY(int64_t loopIdx);
    __aicore__ inline void CopyInXNorm(int64_t loopIdx);
    __aicore__ inline void CopyInExpertIdx(int64_t loopIdx);
    __aicore__ inline void Sigmoid();
    __aicore__ inline void GetGradXNorm();
    __aicore__ inline void GetGradX();
    __aicore__ inline void CopyOut(int64_t loopIdx);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> gradYQue_;
    TQue<QuePosition::VECIN, 1> indicesQue_;
    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    TBuf<TPosition::VECCALC> bufk4Mask_;
    TBuf<TPosition::VECCALC> bufk0_;
    TBuf<TPosition::VECCALC> bufk1_;
    TBuf<TPosition::VECCALC> bufn2_;
    TBuf<TPosition::VECCALC> bufn3_;
    TBuf<TPosition::VECCALC> bufs_;
    TBuf<TPosition::VECCALC> bufk4Add_;
    TBuf<TPosition::VECCALC> bufk4Index_;
    TBuf<TPosition::VECCALC> bufk4RecipSumW_;

    GlobalTensor<float> xNormGm_;
    GlobalTensor<T> gradYGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> gradXGm_;

    LocalTensor<float> xNormLocal_;
    LocalTensor<T> gradYLocal_;
    LocalTensor<int32_t> expertIdxLocal_;
    LocalTensor<uint32_t> expertIdxLocalUint32_;
    LocalTensor<T> gradXOutTensor_;

    int64_t blockIdx_ = 0;
    int64_t loopTimes_ = 0;
    int64_t tailRows_ = 0;
    int64_t curRows_ = 0;

    int64_t elementCountPerLoop_ = 0;
    int64_t curRowsByKAlign_ = 0;
    int64_t kAlign_ = 0;   // fp32的k 32字节对齐后的元素个数
    int64_t kAlign16_ = 0; // 16位的k 32字节对齐后的元素个数
    const MoeGatingTopKBackwardRegbaseTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void
MoeGatingTopKBackwardRegbase<T>::Init(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx, GM_ADDR gradX, GM_ADDR workspace,
                                      const MoeGatingTopKBackwardRegbaseTilingData *tilingData, TPipe *tPipe)
{
    (void)workspace;
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    kAlign_ = Align(tilingData_->k, SIZE_OF_INT32);
    kAlign16_ = Align(tilingData_->k, NUM_TWO);
    if (blockIdx_ == tilingData_->needCoreNum - 1) {
        loopTimes_ = tilingData_->lastLoopTimes;
        tailRows_ = tilingData_->lastTailRows;
    } else {
        loopTimes_ = tilingData_->perLoopTimes;
        tailRows_ = tilingData_->perTailRows;
    }

    // init gm buf
    xNormGm_.SetGlobalBuffer((__gm__ float *)xNorm + tilingData_->perCoreRows * tilingData_->expertCount * blockIdx_);
    gradYGm_.SetGlobalBuffer((__gm__ T *)gradY + tilingData_->perCoreRows * tilingData_->k * blockIdx_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + tilingData_->perCoreRows * tilingData_->k * blockIdx_);
    gradXGm_.SetGlobalBuffer((__gm__ T *)gradX + tilingData_->perCoreRows * tilingData_->expertCount * blockIdx_);

    // init que
    pipe_->InitBuffer(gradYQue_, NUM_TWO,
                      tilingData_->baseRows * AlignBytes(tilingData_->k, tilingData_->gradYDtypeSize));
    pipe_->InitBuffer(indicesQue_, NUM_TWO, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_INT32));
    pipe_->InitBuffer(xQue_, NUM_TWO, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(outQue_, NUM_ONE,
                      AlignBytes(tilingData_->baseRows * tilingData_->expertCount, tilingData_->gradYDtypeSize));

    pipe_->InitBuffer(bufk4Mask_, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk0_, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk1_, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk4Add_, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk4Index_, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk4RecipSumW_, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufn2_, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufn3_, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufs_, tilingData_->baseRows * AlignBytes(tilingData_->k, SIZE_OF_FLOAT32));
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::Process()
{
    GetGatherOffsetIndex();
    for (int64_t loopIdx = 0; loopIdx < loopTimes_; loopIdx++) {
        curRows_ = loopIdx == loopTimes_ - 1 ? tailRows_ : tilingData_->baseRows;
        elementCountPerLoop_ = curRows_ * tilingData_->expertCount;
        curRowsByKAlign_ = curRows_ * kAlign_;
        CopyInGradY(loopIdx);
        CopyInXNorm(loopIdx);
        CopyInExpertIdx(loopIdx);
        Sigmoid();
        GetGradXNorm();
        GetGradX();
        CopyOut(loopIdx);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::GetGatherOffsetIndex()
{
    LocalTensor<float> bufk1Fp32 = bufk1_.Get<float>();
    LocalTensor<float> bufk4AddFp32 = bufk4Add_.Get<float>();
    LocalTensor<float> bufk4Mask = bufk4Mask_.Get<float>();
    LocalTensor<float> bufk4Tmp = bufk4Index_.Get<float>(); // 临时存储中间变量

    Duplicate(bufk4Tmp, (float)0.0, static_cast<int32_t>(kAlign_));
    PipeBarrier<PIPE_V>();
    Duplicate(bufk4Tmp, (float)1.0, static_cast<int32_t>(tilingData_->k));
    PipeBarrier<PIPE_V>();
    uint32_t dstShape[NUM_TWO] = {static_cast<uint32_t>(tilingData_->baseRows), static_cast<uint32_t>(kAlign_)};
    uint32_t srcShape[NUM_TWO] = {1, static_cast<uint32_t>(kAlign_)};
    BroadCast<float, NUM_TWO, 0>(bufk4Mask, bufk4Tmp, dstShape, srcShape);

    Arange(bufk1Fp32, static_cast<float>(0), static_cast<float>(tilingData_->expertCount),
           static_cast<int32_t>(tilingData_->baseRows));
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V); // Arange中存在S和V，不同count结束的操作流不一样
    PipeBarrier<PIPE_V>();
    uint32_t srcShapeAdd[NUM_TWO] = {static_cast<uint32_t>(tilingData_->baseRows), 1};
    BroadCast<float, NUM_TWO, 1>(bufk4AddFp32, bufk1Fp32, dstShape,
                                 srcShapeAdd); // broadcast不支持int32类型，因此先生成fp32再转换为int32
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> bufk4Add = bufk4AddFp32.ReinterpretCast<int32_t>();
    Cast(bufk4Add, bufk4AddFp32, RoundMode::CAST_RINT, tilingData_->baseRows * kAlign_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::CopyInGradY(int64_t loopIdx)
{
    gradYLocal_ = gradYQue_.AllocTensor<T>();
    LocalTensor<float> bufk0 = bufk0_.Get<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curRows_);
    copyParams.blockLen = static_cast<uint32_t>(tilingData_->k) * static_cast<uint32_t>(sizeof(T));
    copyParams.srcStride = static_cast<uint32_t>(0);
    copyParams.dstStride = static_cast<uint32_t>(0);
    int64_t kAlign = tilingData_->gradYDtypeSize == NUM_TWO ? kAlign16_ : kAlign_;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(kAlign - tilingData_->k), 0};
    DataCopyPad(gradYLocal_, gradYGm_[loopIdx * tilingData_->baseRows * tilingData_->k], copyParams, padParams);
    gradYQue_.EnQue(gradYLocal_);
    gradYLocal_ = gradYQue_.DeQue<T>();
    if constexpr (IsSameType<T, float>::value) {
        Muls(bufk0, gradYLocal_, tilingData_->routedScalingFactor, curRowsByKAlign_);
        PipeBarrier<PIPE_V>();
    } else {
        if (kAlign16_ == kAlign_) {
            Cast(bufk0, gradYLocal_, RoundMode::CAST_NONE, curRowsByKAlign_);
        } else {
            for (int64_t i = 0; i < curRows_; i++) {
                Cast(bufk0[i * kAlign_], gradYLocal_[i * kAlign16_], RoundMode::CAST_NONE, kAlign_);
            }
        }
        PipeBarrier<PIPE_V>();
        Muls(bufk0, bufk0, tilingData_->routedScalingFactor, curRowsByKAlign_);
        PipeBarrier<PIPE_V>();
    }
    gradYQue_.FreeTensor(gradYLocal_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::CopyInXNorm(int64_t loopIdx)
{
    xNormLocal_ = xQue_.AllocTensor<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = static_cast<uint32_t>(curRows_ * tilingData_->expertCount * SIZE_OF_FLOAT32);
    copyParams.srcStride = static_cast<uint32_t>(0);
    copyParams.dstStride = static_cast<uint32_t>(0);
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(xNormLocal_, xNormGm_[loopIdx * tilingData_->baseRows * tilingData_->expertCount], copyParams,
                padParams);

    xQue_.EnQue(xNormLocal_);
    xNormLocal_ = xQue_.DeQue<float>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::CopyInExpertIdx(int64_t loopIdx)
{
    expertIdxLocal_ = indicesQue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curRows_);
    copyParams.blockLen = static_cast<uint32_t>(tilingData_->k * SIZE_OF_INT32);
    copyParams.srcStride = static_cast<uint32_t>(0);
    copyParams.dstStride = static_cast<uint32_t>(0);
    DataCopyPadExtParams<int32_t> padParams{true, 0, static_cast<uint8_t>(kAlign_ - tilingData_->k), 0};
    DataCopyPad(expertIdxLocal_, expertIdxGm_[loopIdx * tilingData_->baseRows * tilingData_->k], copyParams, padParams);

    indicesQue_.EnQue(expertIdxLocal_);
    expertIdxLocal_ = indicesQue_.DeQue<int32_t>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::Sigmoid()
{
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> bufk4Index = bufk4Index_.Get<int32_t>();
    LocalTensor<int32_t> bufk4Add = bufk4Add_.Get<int32_t>();
    LocalTensor<float> bufk1 = bufk1_.Get<float>();
    LocalTensor<float> bufs = bufs_.Get<float>();
    LocalTensor<float> bufk4RecipSumW = bufk4RecipSumW_.Get<float>();
    LocalTensor<float> bufk4Mask = bufk4Mask_.Get<float>();

    Add(bufk4Index, expertIdxLocal_, bufk4Add, curRowsByKAlign_);
    indicesQue_.FreeTensor(expertIdxLocal_);
    PipeBarrier<PIPE_V>();
    Muls(bufk4Index, bufk4Index, static_cast<int32_t>(SIZE_OF_INT32), curRowsByKAlign_);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> bufk4IndexUint32 = bufk4Index.ReinterpretCast<uint32_t>();
    Gather(bufk1, xNormLocal_, bufk4IndexUint32, (uint32_t)0, curRowsByKAlign_);
    PipeBarrier<PIPE_V>();

    Mul(bufk1, bufk1, bufk4Mask, curRowsByKAlign_);
    PipeBarrier<PIPE_V>();
    uint32_t srcShape[NUM_TWO] = {static_cast<uint32_t>(curRows_), static_cast<uint32_t>(kAlign_)};
    ReduceSum<float, Pattern::Reduce::AR, false>(bufs, bufk1, srcShape, true);
    PipeBarrier<PIPE_V>();

    Adds(bufs, bufs, tilingData_->eps, curRows_);
    PipeBarrier<PIPE_V>();
    uint32_t dstShapeSumW[NUM_TWO] = {static_cast<uint32_t>(curRows_), static_cast<uint32_t>(kAlign_)};
    uint32_t srcShapeSumW[NUM_TWO] = {static_cast<uint32_t>(curRows_), 1};
    BroadCast<float, NUM_TWO, 1>(bufk4RecipSumW, bufs, dstShapeSumW, srcShapeSumW);
    PipeBarrier<PIPE_V>();
    Div(bufk1, bufk1, bufk4RecipSumW, curRowsByKAlign_);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::GetGradXNorm()
{
    LocalTensor<float> bufk1 = bufk1_.Get<float>();
    LocalTensor<float> bufk0 = bufk0_.Get<float>();
    Mul(bufk1, bufk0, bufk1, curRowsByKAlign_);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> bufs = bufs_.Get<float>();
    LocalTensor<float> bufk4Mask = bufk4Mask_.Get<float>();
    Mul(bufk1, bufk1, bufk4Mask, curRowsByKAlign_);
    PipeBarrier<PIPE_V>();

    uint32_t reduceSumShape[NUM_TWO] = {static_cast<uint32_t>(curRows_), static_cast<uint32_t>(kAlign_)};
    ReduceSum<float, Pattern::Reduce::AR, true>(bufs, bufk1, reduceSumShape, true);

    uint32_t dstShape[NUM_TWO] = {static_cast<uint32_t>(curRows_), static_cast<uint32_t>(kAlign_)};
    uint32_t srcShape[NUM_TWO] = {static_cast<uint32_t>(curRows_), 1};
    Broadcast<float, NUM_TWO, 1>(bufk1, bufs, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
    Sub(bufk0, bufk0, bufk1, curRowsByKAlign_);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> bufk4RecipSumW = bufk4RecipSumW_.Get<float>();
    Div(bufk0, bufk0, bufk4RecipSumW, curRowsByKAlign_);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> bufn3 = bufn3_.Get<float>();
    Duplicate<float>(bufn3, 0.0f, elementCountPerLoop_);

    LocalTensor<int32_t> bufk4Index = bufk4Index_.Get<int32_t>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    for (int64_t i = 0; i < curRowsByKAlign_; ++i) {
        if (i % kAlign_ < tilingData_->k) {
            uint32_t index = static_cast<uint32_t>(bufk4Index.GetValue(i) / SIZE_OF_INT32);
            float value = bufk0.GetValue(i);
            bufn3.SetValue(index, value);
        }
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::GetGradX()
{
    LocalTensor<float> bufn3 = bufn3_.Get<float>();

    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    gradXOutTensor_ = outQue_.AllocTensor<T>();
    DenseSigmoidGradRegbase<T>(xNormLocal_, bufn3, gradXOutTensor_, static_cast<uint32_t>(elementCountPerLoop_));
    xQue_.FreeTensor(xNormLocal_);
    outQue_.EnQue(gradXOutTensor_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardRegbase<T>::CopyOut(int64_t loopIdx)
{
    gradXOutTensor_ = outQue_.DeQue<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(elementCountPerLoop_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(gradXGm_[loopIdx * tilingData_->baseRows * tilingData_->expertCount], gradXOutTensor_, dataCopyParams);
    outQue_.FreeTensor(gradXOutTensor_);
}

} // namespace MoeGatingTopKBackward
#endif // MOE_GATING_TOP_K_BACKWARD_REGBASE_H
