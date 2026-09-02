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
 * \file moe_permute_prob.h
 * \brief
 */
#ifndef MOE_PERMUTE_PROB_H
#define MOE_PERMUTE_PROB_H

#include "moe_token_permute_with_routing_map_common.h"

namespace MoeTokenPermute {
using namespace AscendC;
constexpr int32_t BUFFER_LEN = 1024;
constexpr int32_t BUFFER_NUM_DB = 2;
constexpr int32_t BUFFER_LEN_DB = 1024 * 2;

template <typename T>
class MoePermuteProb {
public:
    __aicore__ inline MoePermuteProb(){};
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR indices, GM_ADDR dst,
                                const MoeTokenPermuteWithRoutingMapTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void copyIndice();

    __aicore__ inline void AllocEvent();
    __aicore__ inline void ReleaseEvent();

private:
    TPipe *pipe;

    event_t eventIdICopyin;
    event_t eventIdPCopyin;
    event_t eventIdPCopyOut;
    event_t eventIdGetI;
    event_t eventIdGetP;
    event_t eventIdSetP;
    event_t eventIdVToMTE3K;
    event_t eventIdMTE3ToMTE2K;
    event_t eventIdMTE2ToMTE3V;
    event_t eventIdMTE3ToMTE2V;
    event_t eventIdVToMTE3V;
    event_t eventIdMTE3ToVV;
    TBuf<TPosition::VECCALC> inQue;
    TBuf<TPosition::VECCALC> indiceQue;
    TBuf<TPosition::VECCALC> outQue;

    GlobalTensor<T> srcGm;
    GlobalTensor<T> dstGm;

    GlobalTensor<int32_t> indicesGm;
    LocalTensor<int32_t> indicesLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;

    const PermuteVBSComputeRMTilingData *vbsTilingData;

    int64_t coreTaskNum;
    int64_t coreTaskNumFront;
    int64_t tailcoreTask;
    int64_t coreNum;
    int64_t taskId = 0;
    int64_t blockIdx;
    int64_t srcWsIndex = 0;
    int64_t blockFactor;
    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;
    int64_t capacity;
    int64_t tokenNum;
    int64_t coreProblen;
    int64_t nowTask = 0;
    int64_t capacityI = 0;
    int64_t nowProbI = 0;
    int64_t copyIntokenId = 0;
    int64_t IntokenId = 0;
    int64_t LastIntokenId = -BUFFER_LEN_DB;
    int64_t allProblen;
    int64_t loaclId;
    DataCopyExtParams indicesCopyParams{1, static_cast<uint32_t>(1), 0, 0, 0};
    DataCopyExtParams indicesCopyLastParams;
    DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(1), 0, 0, 0};
    DataCopyExtParams tokenCopyLastParams;
    DataCopyExtParams tokenCopyOutParams;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<int32_t> indicesPadParams{false, 0, 0, 0};
};

template <typename T>
__aicore__ inline void MoePermuteProb<T>::AllocEvent()
{
    eventIdICopyin = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE2>());
    eventIdPCopyin = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE2>());
    eventIdPCopyOut = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE3>());
    eventIdGetI = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    eventIdGetP = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    eventIdSetP = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>());
}
template <typename T>
__aicore__ inline void MoePermuteProb<T>::ReleaseEvent()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE2>(eventIdICopyin);
    GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE2>(eventIdPCopyin);
    GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE3>(eventIdPCopyOut);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdGetI);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdGetP);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_S>(eventIdSetP);
}

template <typename T>
__aicore__ inline void MoePermuteProb<T>::CopyIn()
{
    tokenCopyParams.blockLen =
        ((tokenNum - IntokenId) < BUFFER_LEN) ? (tokenNum - IntokenId) * sizeof(T) : BUFFER_LEN * sizeof(T);
    DataCopyPadCustom(xLocal, srcGm[nowProbI / capacity * tokenNum + IntokenId], tokenCopyParams, padParams);
}

template <typename T>
__aicore__ inline void MoePermuteProb<T>::copyIndice()
{
    indicesCopyParams.blockLen = ((coreProblen - nowProbI) < BUFFER_LEN) ? (coreProblen - nowProbI) * sizeof(int32_t) :
                                                                           BUFFER_LEN * sizeof(int32_t);
    DataCopyPadCustom(indicesLocal, indicesGm[nowProbI], indicesCopyParams, indicesPadParams);
}

template <typename T>
__aicore__ inline void MoePermuteProb<T>::CopyOut()
{
    tokenCopyParams.blockLen = (loaclId + 1) * sizeof(T);
    DataCopyPadCustom(dstGm[nowProbI - loaclId], yLocal, tokenCopyParams);
}

template <typename T>
__aicore__ inline void MoePermuteProb<T>::Init(GM_ADDR dst, GM_ADDR src, GM_ADDR indices,
                                               const MoeTokenPermuteWithRoutingMapTilingData *tilingData, TPipe *tPipe)
{
    this->tokenNum = tilingData->n;
    this->coreNum = tilingData->vbsComputeParamsOp.needCoreNum;
    this->capacity = tilingData->capacity;
    this->vbsTilingData = &(tilingData->vbsComputeParamsOp);

    tailcoreTask = this->vbsTilingData->tailcoreTask;
    this->blockIdx = GetBlockIdx();
    if (this->blockIdx > tilingData->vbsComputeParamsOp.frontCoreNum - 1) {
        coreTaskNumFront = tilingData->vbsComputeParamsOp.frontCoreNum;
        this->coreTaskNum = this->vbsTilingData->tailcoreTask;
    } else {
        coreTaskNumFront = this->blockIdx * (this->vbsTilingData->frontcoreTask - this->vbsTilingData->tailcoreTask);
        this->coreTaskNum = this->vbsTilingData->frontcoreTask;
    }
    coreProblen = coreTaskNum * capacity;
    allProblen = coreTaskNum * tokenNum;

    this->pipe = tPipe;
    srcGm.SetGlobalBuffer((__gm__ T *)src + (this->blockIdx * tailcoreTask + coreTaskNumFront) * tokenNum);
    dstGm.SetGlobalBuffer((__gm__ T *)dst + (this->blockIdx * tailcoreTask + coreTaskNumFront) * capacity);
    indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices +
                              (this->blockIdx * tailcoreTask + coreTaskNumFront) * capacity);

    pipe->InitBuffer(inQue, BUFFER_LEN * sizeof(T));
    pipe->InitBuffer(indiceQue, BUFFER_LEN * sizeof(int32_t));
    pipe->InitBuffer(outQue, BUFFER_LEN * sizeof(T));
}

template <typename T>
__aicore__ inline void MoePermuteProb<T>::Process()
{
    if (this->blockIdx < coreNum) {
        indicesLocal = indiceQue.Get<int32_t>();
        xLocal = inQue.Get<T>();
        yLocal = outQue.Get<T>();
        int64_t endI = coreProblen;
        for (int64_t startI = 0; startI < endI; startI++) {
            auto indexINInidice = nowProbI % BUFFER_LEN;
            loaclId = nowProbI % BUFFER_LEN;
            if (loaclId == 0) {
                copyIndice();
            }
            IntokenId = indicesLocal.GetValue(loaclId);
            T p;

            if ((LastIntokenId + BUFFER_LEN - 1) < IntokenId || LastIntokenId > IntokenId) {
                CopyIn();
                LastIntokenId = IntokenId;
                p = xLocal.GetValue(0);
            } else {
                p = xLocal.GetValue(IntokenId - LastIntokenId);
            }
            yLocal.SetValue(loaclId, p);
            if (loaclId == (BUFFER_LEN - 1) || nowProbI == (coreProblen - 1)) {
                CopyOut();
            }
            if (nowProbI % capacity == (capacity - 1)) {
                LastIntokenId = -BUFFER_LEN_DB;
            }
            nowProbI += 1;
        }
    }
}

} // namespace MoeTokenPermute
#endif // MOE_PERMUTE_PROB_H
