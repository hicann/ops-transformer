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
 * \file attention_worker_combine.h
 * \brief
 */

#ifndef OP_KERNEL_ATTENTION_WORKER_COMBINE_SPLIT_K_H_
#define OP_KERNEL_ATTENTION_WORKER_COMBINE_SPLIT_K_H_

#include "kernel_operator.h"
#include "attention_worker_combine_common_utils.h"

using namespace AscendC;
template <typename T>
class KernelAttentionWorkerCombineSplitK {
public:
    __aicore__ inline KernelAttentionWorkerCombineSplitK() {}

    __aicore__ inline KernelAttentionWorkerCombineSplitK(TPipe *pipe, const AttentionWorkerCombineTilingData *tiling)
        : pipe_(pipe),
          tl_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR expert_scales, GM_ADDR layer_id, GM_ADDR y,
                                GM_ADDR next_layer_id);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t offset, int64_t kSize);
    __aicore__ inline void CopyInTokenInfo(int64_t offset, int64_t kSize);
    __aicore__ inline void ScanTokenInfo(int64_t offset, int64_t kSize);
    __aicore__ inline void ClearTokenInfo(int64_t offset, int64_t kSize);
    __aicore__ inline void Compute(int64_t offset, int64_t kSize);
    __aicore__ inline void ComputeTail(int64_t offset, int64_t kSize);
    __aicore__ inline void CopyOut(int64_t offset);

    TPipe *pipe_ = nullptr;
    const AttentionWorkerCombineTilingData *tl_;
    constexpr static int64_t blockSize = 32;
    GlobalTensor<uint32_t> contextGm0;
    GlobalTensor<uint64_t> contextGm1;
    GlobalTensor<T> srcTokenGm;
    GlobalTensor<int32_t> srcTokenInfoGm;
    GlobalTensor<float> srcScalesGm;
    GlobalTensor<int32_t> srcLayerIdGm;
    GlobalTensor<T> dstGm;
    GlobalTensor<int32_t> dstNextLayerIdGm;

    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> tmpBuf;
    int64_t bsLoopNum;
    uint32_t micro_batch_id;
};

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::Init(GM_ADDR schedule_context, GM_ADDR expert_scales,
                                                                   GM_ADDR layer_id, GM_ADDR y, GM_ADDR next_layer_id)
{
    contextGm0.SetGlobalBuffer((__gm__ uint32_t *)schedule_context);
    contextGm1.SetGlobalBuffer((__gm__ uint64_t *)schedule_context);

    if (tl_->needSchedule == 1) {
        uint32_t micro_batch_num = contextGm0(GET_OFFSET_B32(ScheduleContext, common.micro_batch_num));
        micro_batch_id = (contextGm0(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)) + 1) % micro_batch_num;
    } else {
        micro_batch_id = contextGm0(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id));
    }
    uint64_t token_data_addr = contextGm1(GET_OFFSET_B64(ScheduleContext, attention.token_data_buf));
    uint64_t token_info_addr = contextGm1(GET_OFFSET_B64(ScheduleContext, attention.token_info_buf));

    srcTokenGm.SetGlobalBuffer(
        (__gm__ T *)(token_data_addr + micro_batch_id * tl_->BS * (tl_->K + 1) * tl_->H * sizeof(T)));
    srcTokenInfoGm.SetGlobalBuffer(
        (__gm__ int32_t *)(token_info_addr + micro_batch_id * tl_->BS * (tl_->K + 1) * sizeof(int32_t)));
    srcScalesGm.SetGlobalBuffer((__gm__ float *)expert_scales);
    srcLayerIdGm.SetGlobalBuffer((__gm__ int32_t *)layer_id);
    dstGm.SetGlobalBuffer((__gm__ T *)y);
    dstNextLayerIdGm.SetGlobalBuffer((__gm__ int32_t *)next_layer_id);

    this->pipe_->InitBuffer(inQue, 2, tl_->KSplitFactor * tl_->HSplitFactor * sizeof(T));
    this->pipe_->InitBuffer(outQue, 2, tl_->HSplitFactor * sizeof(T));
    this->pipe_->InitBuffer(tmpBuf, tl_->HSplitFactor * sizeof(float) * 2);
}

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::Process()
{
    int64_t blockId = GetBlockIdx();
    if (blockId == tl_->usedCoreNum - 1) {
        bsLoopNum = tl_->tailCoreBsLoopNum;
    } else {
        bsLoopNum = tl_->mainCoreBsLoopNum;
    }

    if (blockId == 0) {
        dstNextLayerIdGm(0) = srcLayerIdGm(0) + 1;
    }
    int64_t tokenSrcKInOffset = tl_->KSplitFactor * tl_->H;
    int64_t tokenSrcBsOffset = (tl_->K + 1) * (tl_->H);
    int64_t tokenSrcStart = blockId * tl_->mainCoreBsLoopNum * tokenSrcBsOffset;
    int64_t tokenInfoStart = blockId * tl_->mainCoreBsLoopNum * (tl_->K + 1);
    int64_t tokenDstStart = blockId * tl_->mainCoreBsLoopNum * tl_->H;
    int64_t scaleStart = blockId * tl_->mainCoreBsLoopNum * tl_->K;
    int32_t sum = 0;
    event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    if (tl_->needSchedule == 1) {
        int32_t sumRes = 0;
        LocalTensor<int32_t> sumLocal = tmpBuf.Get<int32_t>();
        while (sumRes != bsLoopNum * (tl_->K + 1)) {
            ScanTokenInfo(tokenInfoStart, bsLoopNum * (tl_->K + 1));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            sumRes = sumLocal.GetValue(0);
            SetFlag<HardEvent::S_V>(eventIDSToV);
            WaitFlag<HardEvent::S_V>(eventIDSToV);
        }
    }

    for (int64_t bsLoopId = 0; bsLoopId < bsLoopNum; bsLoopId++) {
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>()[tl_->HSplitFactor];
        Duplicate(tmpLocal, static_cast<float>(0), tl_->HSplitFactor);
        for (int64_t kSplitLoopId = 0; kSplitLoopId < tl_->KSplitLoopNum; kSplitLoopId++) {
            CopyIn(tokenSrcStart + bsLoopId * tokenSrcBsOffset + kSplitLoopId * tokenSrcKInOffset, tl_->KSplitFactor);
            Compute(scaleStart + bsLoopId * tl_->K + kSplitLoopId * tl_->KSplitFactor, tl_->KSplitFactor);
        }
        CopyIn(tokenSrcStart + bsLoopId * tokenSrcBsOffset + tl_->KSplitLoopNum * tokenSrcKInOffset,
               tl_->KSplitTailFactor + 1);
        ComputeTail(scaleStart + bsLoopId * tl_->K + tl_->KSplitLoopNum * tl_->KSplitFactor, tl_->KSplitTailFactor);
        CopyOut(tokenDstStart + bsLoopId * tl_->H);
    }
    if (tl_->needSchedule == 1) {
        ClearTokenInfo(tokenInfoStart, bsLoopNum * (tl_->K + 1));
        SyncAll();
        if (blockId == 0) {
            contextGm0(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)) = micro_batch_id;
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(
                contextGm0[GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)]);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::CopyIn(int64_t offset, int64_t kSize)
{
    LocalTensor<T> srcLocal = inQue.AllocTensor<T>();
    int64_t hAlign32B = AlignUp(tl_->H * sizeof(T), blockSize);
    DataCopyExtParams copyParams(kSize, tl_->H * sizeof(T), 0, (tl_->HSplitFactor * sizeof(T) - hAlign32B) / blockSize,
                                 0);
    DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
    DataCopyPad(srcLocal, srcTokenGm[offset], copyParams, padParams);
    inQue.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::CopyInTokenInfo(int64_t offset, int64_t kSize)
{
    LocalTensor<int32_t> srcLocal = inQue.AllocTensor<int32_t>();
    DataCopyExtParams copyParams(1, kSize * sizeof(int32_t), 0, 0, 0);
    DataCopyPadExtParams<int32_t> padParams(false, 0, 0, 0);
    DataCopyPad(srcLocal, srcTokenInfoGm[offset], copyParams, padParams);
    inQue.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::ScanTokenInfo(int64_t offset, int64_t kSize)
{
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    CopyInTokenInfo(offset, kSize);
    LocalTensor<int32_t> srcLocal = inQue.DeQue<int32_t>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<int32_t> resLocal = tmpLocal.template ReinterpretCast<int32_t>();
    float tmp1 = tmpLocal.GetValue(0);
    int32_t tmp2 = resLocal.GetValue(0);
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
    Cast(tmpLocal, srcLocal, AscendC::RoundMode::CAST_NONE, kSize);
    PipeBarrier<PIPE_V>();
    ReduceSum(tmpLocal, tmpLocal, tmpLocal[tl_->HSplitFactor], kSize);
    PipeBarrier<PIPE_V>();
    Cast(resLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, kSize);
    PipeBarrier<PIPE_V>();
    inQue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::ClearTokenInfo(int64_t offset, int64_t kSize)
{
    LocalTensor<int32_t> dstLocal = outQue.AllocTensor<int32_t>();
    Duplicate(dstLocal, static_cast<int32_t>(0), kSize);
    outQue.EnQue(dstLocal);
    dstLocal = outQue.DeQue<int32_t>();
    DataCopyExtParams copyParams(1, kSize * sizeof(int32_t), 0, 0, 0);
    DataCopyPad(srcTokenInfoGm[offset], dstLocal, copyParams);
    outQue.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::Compute(int64_t offset, int64_t kSize)
{
    LocalTensor<T> srcLocal = inQue.DeQue<T>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<T> tmpCastLocal = tmpLocal.template ReinterpretCast<T>();
    for (int64_t i = 0; i < kSize; i++) {
        Cast(tmpLocal, srcLocal[i * tl_->HSplitFactor], AscendC::RoundMode::CAST_NONE, tl_->HSplitFactor);
        PipeBarrier<PIPE_V>();
        Muls(tmpLocal, tmpLocal, srcScalesGm.GetValue(offset + i), tl_->HSplitFactor);
        PipeBarrier<PIPE_V>();
        Add(tmpLocal[tl_->HSplitFactor], tmpLocal[tl_->HSplitFactor], tmpLocal, tl_->HSplitFactor);
        PipeBarrier<PIPE_V>();
    }
    inQue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::ComputeTail(int64_t offset, int64_t kSize)
{
    LocalTensor<T> srcLocal = inQue.DeQue<T>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<T> tmpCastLocal = tmpLocal.template ReinterpretCast<T>();
    LocalTensor<T> dstLocal = outQue.AllocTensor<T>();
    for (int64_t i = 0; i < kSize; i++) {
        Cast(tmpLocal, srcLocal[i * tl_->HSplitFactor], AscendC::RoundMode::CAST_NONE, tl_->HSplitFactor);
        PipeBarrier<PIPE_V>();
        Muls(tmpLocal, tmpLocal, srcScalesGm.GetValue(offset + i), tl_->HSplitFactor);
        PipeBarrier<PIPE_V>();
        Add(tmpLocal[tl_->HSplitFactor], tmpLocal[tl_->HSplitFactor], tmpLocal, tl_->HSplitFactor);
        PipeBarrier<PIPE_V>();
    }
    Cast(tmpLocal, srcLocal[kSize * tl_->HSplitFactor], AscendC::RoundMode::CAST_NONE, tl_->HSplitFactor);
    PipeBarrier<PIPE_V>();
    Add(tmpLocal, tmpLocal, tmpLocal[tl_->HSplitFactor], tl_->HSplitFactor);
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(dstLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, tl_->HSplitFactor);
    } else if constexpr (std::is_same<T, half>::value) {
        Cast(dstLocal, tmpLocal, AscendC::RoundMode::CAST_NONE, tl_->HSplitFactor);
    }
    outQue.EnQue(dstLocal);
    inQue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void KernelAttentionWorkerCombineSplitK<T>::CopyOut(int64_t offset)
{
    LocalTensor<T> dstLocal = outQue.DeQue<T>();
    int64_t hAlign32B = AlignUp(tl_->H * sizeof(T), blockSize);
    DataCopyExtParams copyParams(1, tl_->H * sizeof(T), (tl_->HSplitFactor * sizeof(T) - hAlign32B) / blockSize, 0, 0);
    DataCopyPad(dstGm[offset], dstLocal, copyParams);
    outQue.FreeTensor(dstLocal);
}

#endif // OP_KERNEL_ATTENTION_WORKER_COMBINE_SPLIT_K_H_
