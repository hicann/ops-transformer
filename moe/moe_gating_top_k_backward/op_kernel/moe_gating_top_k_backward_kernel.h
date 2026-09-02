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
 * \file moe_gating_top_k_backward_kernel.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_BACKWARD_KERNEL_H
#define MOE_GATING_TOP_K_BACKWARD_KERNEL_H
#include "kernel_operator.h"
#include "moe_gating_top_k_backward_common.h"
#include "basic_api/kernel_operator_utils_intf.h"
namespace MoeGatingTopKBackward {
using namespace AscendC;

template <typename T>
class MoeGatingTopKBackwardKernel {
public:
    __aicore__ inline MoeGatingTopKBackwardKernel(){};
    __aicore__ inline void Init(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx, GM_ADDR gradX,
                                const MoeGatingTopKBackwardTilingData *tilingData, TPipe *tPipe);
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
    const MoeGatingTopKBackwardTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::Init(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx,
                                                            GM_ADDR gradX,
                                                            const MoeGatingTopKBackwardTilingData *tilingData,
                                                            TPipe *tPipe)
{
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
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::Process()
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
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::GetGatherOffsetIndex()
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
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::CopyInGradY(int64_t loopIdx)
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
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::CopyInXNorm(int64_t loopIdx)
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
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::CopyInExpertIdx(int64_t loopIdx)
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
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::Sigmoid()
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
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::GetGradXNorm()
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
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::GetGradX()
{
    LocalTensor<float> bufn2 = bufn2_.Get<float>();
    Duplicate<float>(bufn2, 1.0f, elementCountPerLoop_);
    PipeBarrier<PIPE_V>();
    Sub(bufn2, bufn2, xNormLocal_, elementCountPerLoop_);
    PipeBarrier<PIPE_V>();
    Mul(bufn2, bufn2, xNormLocal_, elementCountPerLoop_);
    xQue_.FreeTensor(xNormLocal_);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> bufn3 = bufn3_.Get<float>();

    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    gradXOutTensor_ = outQue_.AllocTensor<T>();
    if constexpr (!IsSameType<T, float>::value) {
        Mul(bufn2, bufn2, bufn3, elementCountPerLoop_);
        PipeBarrier<PIPE_V>();
        Cast(gradXOutTensor_, bufn2, RoundMode::CAST_RINT, elementCountPerLoop_);
    } else {
        Mul(gradXOutTensor_, bufn2, bufn3, elementCountPerLoop_);
    }
    outQue_.EnQue(gradXOutTensor_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::CopyOut(int64_t loopIdx)
{
    gradXOutTensor_ = outQue_.DeQue<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(elementCountPerLoop_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(gradXGm_[loopIdx * tilingData_->baseRows * tilingData_->expertCount], gradXOutTensor_, dataCopyParams);
    outQue_.FreeTensor(gradXOutTensor_);
}

} // namespace MoeGatingTopKBackward
#endif // MOE_GATING_TOP_K_BACKWARD_KERNEL_H
