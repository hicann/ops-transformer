/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file sparse_lightning_indexer_kl_loss_grad_vector_block.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_BLOCK_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_BLOCK_H
#include "sparse_lightning_indexer_kl_loss_grad_regbase_common.h"
#include "c_api/cache_ctrl/cache_ctrl.h"
#include "c_api/vector_datamove/vector_datamove.h"
#include "vf/vf_process_vec0_kl_loss_grad.h"
#include "vf/vf_process_vec1_kl_loss_grad.h"
#include "vf/vf_process_vec2_kl_loss_grad.h"
#include "vf/vf_process_vec4_kl_loss_grad.h"
#include "vf/vf_process_vec5_kl_loss_grad.h"
#include "vf/vf_process_vec6_kl_loss_grad.h"
#include "vf/vf_pair_permute_kl_loss_grad.h"
#include "vf/vf_cast_dup_kl_loss_grad.h"
#include "vf/vf_reduce_sum_kl_loss_grad.h"

namespace Slig {
template <bool Deterministic>
struct SligGatherMutexState {};
template <>
struct SligGatherMutexState<false> {
    MutexID mutexIds_[2];
    uint32_t gatherHalfIdx_;
    MutexID weightMutexId_;
    MutexID reduceSumMutexId_;
    MutexID reluGradMutexId_;
};
template <bool Enabled>
struct SligPairPipelineState {};
template <>
struct SligPairPipelineState<true> {
    LocalTensor<int32_t> pairIndexUb_;
    LocalTensor<float> sumRelayL1_;
    __gm__ float* pairScatterGm_;
    uint8_t pairScatterCacheMode_;
    bool pairRawAddressSafe_;
    bool pairByteCacheEn_;
    event_t eventPairIndexSToMte2_;
    event_t eventPairIndexVToS_;
};
TEMPLATES_DEF
class SligBlockVec : private SligGatherMutexState<Deterministic>, private SligPairPipelineState<ResidentPipeline> {
public:
    using Shape = SligShapeTraits<ResidentPipeline>;
    __aicore__ inline SligBlockVec(){};
    __aicore__ inline void InitParams(const SLIGradConstInfo& vecConstInfo,
                                      const optiling::SparseLightningIndexerGradRegBaseTilingData* tiling);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR attnSoftmaxL1,
                                            GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR softmaxOut,
                                            GlobalTensor<INPUT_T>& gatherSYRes, GlobalTensor<T>& relu,
                                            GlobalTensor<T>& reduceSumRes, GlobalTensor<T>& scatterAddRes,
                                            GlobalTensor<T>& scatterAddResPong, GlobalTensor<T>& bmm3Res,
                                            GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK,
                                            GM_ADDR cmpResidualK);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void SetGatherAllL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& buf)
    {
        this->gatherAllL1Tensor = buf.GetTensor<INPUT_T>();
    }
    __aicore__ inline void SetSumRelayL1(const LocalTensor<float>& tensor)
    {
        if constexpr (ResidentPipeline) {
            this->sumRelayL1_ = tensor;
        }
    }
    __aicore__ inline void ProcessVector0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
                                          SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo);
    __aicore__ inline void CopyInWeight(SLIGradRunInfo& runInfo);
    __aicore__ inline void CopyInAttnNorm(SLIGradRunInfo& runInfo);
    __aicore__ inline void ProcessVector1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf,
                                          SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo);
    __aicore__ inline void ProcessVector2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf,
                                          SLIGradRunInfo& runInfo);
    __aicore__ inline void PrefetchFirstRelu(SLIGradRunInfo& runInfo);
    __aicore__ inline void ProcessVector6(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
                                          SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo);
    __aicore__ inline void ProcessVector7(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm3ResBuf,
                                          SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo);
    template <bool rawAddressSafe, bool bytePairCached>
    __aicore__ inline void ProcessVector7PairSorted(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm3ResBuf,
                                                    SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo);
    __aicore__ inline void DeterScatterAdd(int32_t vRealKSize, GlobalTensor<T>& srcGm,
                                           Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& tmpBuf,
                                           int64_t coreKOffset, int64_t keyRowBaseIn, int64_t topkBaseIn, int32_t idx,
                                           GlobalTensor<T>& scatterAddGm, event_t (&eventIdArr)[MODE_NUM_2],
                                           int32_t& scatterAddPingpong);
    __aicore__ inline void ProcessVector7Deter(SLIGradRunInfo& runInfo,
                                               Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& tmpBuf);
    __aicore__ inline void ReInitBuffers(TPipe* pipe);
    __aicore__ inline void ProcessVector8();

    // gm addr
    GlobalTensor<INPUT_T> keyIndexGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<int32_t> topKGm;
    GlobalTensor<T> attnSoftmaxL1Gm;

    // workspace
    GlobalTensor<INPUT_T> gatherSYResGm;
    GlobalTensor<T> reduceSumResGm;
    GlobalTensor<T> reluGm;
    GlobalTensor<T> scatterAddResGm;
    GlobalTensor<T> scatterAddResGmPong;
    GlobalTensor<T> mm3DeterResGm;

    GlobalTensor<T> softmaxOutGm;
    GlobalTensor<T> dWeightGm;
    GlobalTensor<OUT_T> dKeyIndexGm;

    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR usedSeqQlenAddr;
    GM_ADDR usedSeqKvlenAddr;
    GM_ADDR cmpResidualKAddr;

    const optiling::SparseLightningIndexerGradRegBaseTilingData* tilingData;
    TPipe* pipe;

private:
    static constexpr int64_t UB_ROW_SIZE = 64;
    static constexpr int32_t GATHER_ROW_NUM =
        Deterministic ? 64 : (ResidentPipeline ? (VEC_SY_BASESIZE - COOP_AIC_GATHER_ROWS) / 2 : 32);
    static constexpr int32_t GATHER_BUFFER_NUM = 2;
    static constexpr uint32_t GATHER_UB_BYTES = Deterministic ?
                                                    SLIGradConstInfo::BUFFER_SIZE_BYTE_32K :
                                                    GATHER_BUFFER_NUM * GATHER_ROW_NUM * 128U * sizeof(INPUT_T);
    static constexpr int32_t SPARSE_BLOCK_SIZE = 1;
    static constexpr event_t EVENT_ID0 = (event_t)4;
    static constexpr event_t EVENT_ID2 = (event_t)7;
    static constexpr uint32_t PAIR_EXPORT_UB_OFFSET = 4096;
    static constexpr uint32_t PAIR_INDEX_UB_OFFSET = 6144;

    __aicore__ inline void IssueGatherRows(GlobalTensor<INPUT_T>& srcTensor, DataCopyExtParams& intriParams,
                                           DataCopyPadExtParams<INPUT_T>& padParams, uint32_t ubElemOffset,
                                           int64_t gmRowStart, uint32_t dValue, uint16_t blockCount,
                                           uint32_t srcStrideBytes);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const SLIGradRunInfo& runInfo, int64_t s2IdLimit);
    __aicore__ inline void SkipAicGatherRows(const SLIGradKRunInfo& kRunInfo);
    template <event_t IdStart, bool syGmEn, bool gatherL1EnT>
    __aicore__ inline void CopyOutMrgeResult(int32_t mte2Size, int32_t mte3Size, int32_t s2GmStartOffset,
                                             int32_t mergeMte3Idx, uint32_t dValue, uint32_t ubElemBase,
                                             int64_t wsElemBase);
    template <event_t IdStart, bool syGmEn, bool gatherL1EnT>
    __aicore__ inline void MergeKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
                                   const SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo,
                                   GlobalTensor<INPUT_T> srcTensor);
    __aicore__ inline bool GetBS1Index(int64_t usedT1Index, int64_t& bIdx, int64_t& s1Idx, int64_t& bS1Index,
                                       int64_t& accumS1Len, int64_t& accumS2Len, int32_t& actualSeqLensQ,
                                       int32_t& actualSeqLensK, int64_t taskId);
    __aicore__ inline void GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx, SLIGradRunInfo& runInfo,
                                      int64_t accumS1Len, int64_t accumS2Len, int32_t actualSeqLensQ,
                                      int32_t actualSeqLensK);
    __aicore__ inline int32_t GetS2SparseLen(int32_t bIdx, int32_t s1Idx, int32_t actualSeqLensQ,
                                             int32_t actualSeqLensK);
    SLIGradConstInfo constInfo;
    uint32_t gRealSizeAlignTo16;
    int32_t deterKSplitSize;
    TBuf<> gatherTbuf;
    TBuf<> attnNormTbuf;
    using ReduceSumOutputBuffer = typename std::conditional<Deterministic, TQue<QuePosition::VECOUT, 1>, TBuf<>>::type;
    ReduceSumOutputBuffer reduceSumOutQue;
    TQue<QuePosition::VECOUT, 1> softmaxReduceSumOutQue;
    using WeightInputBuffer = typename std::conditional<Deterministic, TQue<QuePosition::VECIN, 1>, TBuf<>>::type;
    WeightInputBuffer weightInQue;
    TQue<QuePosition::VECIN, 1> reluQue;
    using ReluGradOutputBuffer = typename std::conditional<Deterministic, TQue<QuePosition::VECOUT, 1>, TBuf<>>::type;
    ReluGradOutputBuffer reluGradQue;
    TQue<QuePosition::VECOUT, 1> dwQue;
    TQue<QuePosition::VECOUT, 1> dwOutQue;
    using GatherOutputBuffer = typename std::conditional<Deterministic, TQue<QuePosition::VECOUT, 1>, TBuf<>>::type;
    GatherOutputBuffer gatherOutQue;
    TQue<QuePosition::VECIN, 1> scatterAddInQue[2];
    TQue<QuePosition::VECIN, 1> scatterAddInQue2[2];
    TQue<QuePosition::VECOUT, 1> dKeyOutQue[2];

    LocalTensor<INPUT_T> kvMergUb_;
    LocalTensor<INPUT_T> mm1AL1Tensor;
    LocalTensor<INPUT_T> gatherAllL1Tensor;
    LocalTensor<float> weightInUb_;
    LocalTensor<T> reduceSumUb_;
    LocalTensor<T> softmaxReduceSumUb_;
    LocalTensor<T> reluUb_;
    event_t eventReluMte2ToV_;
    event_t eventReluVToMte2_;
    event_t eventMte2ToVShared_;
    LocalTensor<T> dwUb_;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::InitParams(
    const SLIGradConstInfo& vecConstInfo, const optiling::SparseLightningIndexerGradRegBaseTilingData* tiling)
{
    this->constInfo = vecConstInfo;
    tilingData = tiling;
    this->gRealSizeAlignTo16 = (Shape::GSize(vecConstInfo) + 15) / 16 * 16;
    this->deterKSplitSize = static_cast<int32_t>(SLIGradConstInfo::BUFFER_SIZE_BYTE_32K /
                                                 (MODE_NUM_2 * sizeof(T) * vecConstInfo.dSizeQueryIndex));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR key, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR attnSoftmaxL1, GM_ADDR dKey, GM_ADDR dWeight,
    GM_ADDR softmaxOut, GlobalTensor<INPUT_T>& gatherSYRes, GlobalTensor<T>& relu, GlobalTensor<T>& reduceSumRes,
    GlobalTensor<T>& scatterAddRes, GlobalTensor<T>& scatterAddResPong, GlobalTensor<T>& bmm3Res, GM_ADDR cuSeqlensQ,
    GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK)
{
    keyIndexGm.SetGlobalBuffer((__gm__ INPUT_T*)key);
    weightGm.SetGlobalBuffer((__gm__ T*)weight);
    topKGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
    attnSoftmaxL1Gm.SetGlobalBuffer((__gm__ T*)attnSoftmaxL1);

    dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T*)dKey);
    dWeightGm.SetGlobalBuffer((__gm__ T*)dWeight);
    softmaxOutGm.SetGlobalBuffer((__gm__ T*)softmaxOut);

    this->gatherSYResGm = gatherSYRes;
    this->reluGm = relu;
    this->reduceSumResGm = reduceSumRes;
    this->scatterAddResGm = scatterAddRes;

    actualSeqQlenAddr = cuSeqlensQ;
    actualSeqKvlenAddr = cuSeqlensK;
    usedSeqQlenAddr = sequsedQ;
    usedSeqKvlenAddr = sequsedK;
    cmpResidualKAddr = cmpResidualK;

    if constexpr (Deterministic) {
        this->mm3DeterResGm = bmm3Res;
        this->scatterAddResGmPong = scatterAddResPong;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(this->gatherTbuf, GATHER_UB_BYTES);
    if constexpr (Deterministic) {
        pipe->InitBuffer(this->gatherOutQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    } else {
        pipe->InitBuffer(this->gatherOutQue, GATHER_UB_BYTES);
    }
    if constexpr (Deterministic) {
        pipe->InitBuffer(this->reduceSumOutQue, 1,
                         SLIGradConstInfo::BUFFER_SIZE_BYTE_8K + SLIGradConstInfo::BUFFER_SIZE_BYTE_2K);
    } else {
        pipe->InitBuffer(this->reduceSumOutQue,
                         SLIGradConstInfo::BUFFER_SIZE_BYTE_8K + SLIGradConstInfo::BUFFER_SIZE_BYTE_2K);
    }
    if constexpr (Deterministic) {
        pipe->InitBuffer(this->weightInQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_256);
    } else {
        pipe->InitBuffer(this->weightInQue, SLIGradConstInfo::BUFFER_SIZE_BYTE_256);
    }
    pipe->InitBuffer(this->reluQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    if constexpr (Deterministic) {
        pipe->InitBuffer(this->reluGradQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_16K);
    } else {
        pipe->InitBuffer(this->reluGradQue, SLIGradConstInfo::BUFFER_SIZE_BYTE_16K);
    }
    pipe->InitBuffer(this->dwQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->dwOutQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->softmaxReduceSumOutQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_64);
    if constexpr (!Deterministic) {
        pipe->InitBuffer(this->attnNormTbuf, 2048U * sizeof(T));
    }
    this->kvMergUb_ = gatherTbuf.Get<INPUT_T>();
    auto dwUb = this->dwQue.template AllocTensor<T>();
    Duplicate(dwUb, 0.0f, 64);
    this->dwUb_ = dwUb;
    this->dwQue.template FreeTensor(dwUb);
    if constexpr (Deterministic) {
        auto weightCacheUb = this->weightInQue.template AllocTensor<float>();
        this->weightInUb_ = weightCacheUb;
        this->weightInQue.template FreeTensor(weightCacheUb);
    } else {
        this->weightInUb_ = this->weightInQue.template Get<float>();
    }
    if constexpr (Deterministic) {
        auto reduceSumCacheUb = this->reduceSumOutQue.template AllocTensor<T>();
        this->reduceSumUb_ = reduceSumCacheUb;
        this->reduceSumOutQue.template FreeTensor(reduceSumCacheUb);
    } else {
        this->reduceSumUb_ = this->reduceSumOutQue.template Get<T>();
    }
    auto softmaxReduceSumCacheUb = this->softmaxReduceSumOutQue.template AllocTensor<T>();
    this->softmaxReduceSumUb_ = softmaxReduceSumCacheUb;
    this->softmaxReduceSumOutQue.template FreeTensor(softmaxReduceSumCacheUb);
    auto reluCacheUb = this->reluQue.template AllocTensor<T>();
    this->reluUb_ = reluCacheUb;
    this->reluQue.template FreeTensor(reluCacheUb);
    if constexpr (ResidentPipeline) {
        this->pairIndexUb_ = this->reduceSumOutQue.template Get<int32_t>()[PAIR_INDEX_UB_OFFSET / sizeof(int32_t)];
        this->pairScatterGm_ = reinterpret_cast<__gm__ float*>(const_cast<__gm__ T*>(scatterAddResGm.GetPhyAddr()));
        this->pairScatterCacheMode_ = AscendC::ExtractCacheMode(scatterAddResGm);
        const uint64_t scatterBase = reinterpret_cast<uint64_t>(this->pairScatterGm_);
        this->pairRawAddressSafe_ = scatterBase <= AscendC::L2_CACHE_OFFSET_MASK &&
                                    static_cast<uint64_t>(constInfo.totalCost) <=
                                        (AscendC::L2_CACHE_OFFSET_MASK - scatterBase) / (128 * sizeof(T));
        this->eventPairIndexSToMte2_ = static_cast<event_t>(pipe->template AllocEventID<HardEvent::S_MTE2>());
        this->eventPairIndexVToS_ = static_cast<event_t>(pipe->template AllocEventID<HardEvent::V_S>());
    }
    this->eventReluMte2ToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    this->eventReluVToMte2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    this->eventMte2ToVShared_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::V_MTE2>(this->eventReluVToMte2_);
    if constexpr (!Deterministic) {
        this->mutexIds_[0] = AllocMutexID();
        this->mutexIds_[1] = AllocMutexID();
        this->gatherHalfIdx_ = 0;
        this->weightMutexId_ = AllocMutexID();
        this->reduceSumMutexId_ = AllocMutexID();
        this->reluGradMutexId_ = AllocMutexID();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf, SLIGradRunInfo& runInfo,
    SLIGradKRunInfo& kRunInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[kRunInfo.kTaskIdMod2]);
    if constexpr (ResidentPipeline) {
        if (kRunInfo.kTaskId == 0 && kRunInfo.s2RealBaseSize == VEC_SY_BASESIZE) {
            CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_COOP_GATHER_REUSE_FLAG);
        }
    }
    this->mm1AL1Tensor = outputBuf.GetTensor<INPUT_T>();

    if constexpr (Deterministic) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if constexpr (Deterministic) {
        MergeKv<EVENT_ID0, true, false>(outputBuf, runInfo, kRunInfo, keyIndexGm);
    } else if (Shape::GatherResident(constInfo)) {
        MergeKv<EVENT_ID0, true, true>(outputBuf, runInfo, kRunInfo, keyIndexGm);
    } else {
        MergeKv<EVENT_ID0, true, false>(outputBuf, runInfo, kRunInfo, keyIndexGm);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[kRunInfo.kTaskIdMod2]);
    if constexpr (Deterministic) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::IssueGatherRows(
    GlobalTensor<INPUT_T>& srcTensor, DataCopyExtParams& intriParams, DataCopyPadExtParams<INPUT_T>& padParams,
    uint32_t ubElemOffset, int64_t gmRowStart, uint32_t dValue, uint16_t blockCount, uint32_t srcStrideBytes)
{
    intriParams.blockCount = blockCount;
    intriParams.srcStride = srcStrideBytes;
    DataCopyPad(kvMergUb_[ubElemOffset], srcTensor[gmRowStart * dValue], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SligBlockVec<TEMPLATE_ARGS>::GetKeyGmOffset(int64_t realS2Idx, const SLIGradRunInfo& runInfo,
                                                                      int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = runInfo.accumS2Idx + realS2Idx;
    return realKeyGmOffset;
}

TEMPLATES_DEF_NO_DEFAULT
template <event_t IdStart, bool syGmEn, bool gatherL1EnT>
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::CopyOutMrgeResult(int32_t mte2Size, int32_t mte3Size,
                                                                      int32_t s2GmStartOffset, int32_t mergeMte3Idx,
                                                                      uint32_t dValue, uint32_t ubElemBase,
                                                                      int64_t wsElemBase)
{
    if constexpr (Deterministic) {
        if (mte2Size <= mte3Size) {
            return;
        }
        SetFlag<HardEvent::MTE2_MTE3>(mergeMte3Idx + IdStart);
        WaitFlag<HardEvent::MTE2_MTE3>(mergeMte3Idx + IdStart);
        constexpr uint32_t dstNzC0Stride = VEC_SY_BASESIZE;
        SetFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
        LocalTensor<INPUT_T> kvMergUbCp_ = this->gatherOutQue.template AllocTensor<INPUT_T>();
        Ub2UbNd2Nz<INPUT_T>(kvMergUbCp_[ubElemBase], kvMergUb_[ubElemBase], GATHER_ROW_NUM, dValue);
        this->gatherOutQue.template EnQue(kvMergUbCp_);
        this->gatherOutQue.template DeQue<INPUT_T>();
        uint16_t valid_row = static_cast<uint16_t>(mte2Size - mte3Size);
        DataCopy(mm1AL1Tensor[(s2GmStartOffset + mte3Size) << 4], kvMergUbCp_[ubElemBase],
                 {static_cast<uint16_t>(dValue >> 4), valid_row, static_cast<uint16_t>(GATHER_ROW_NUM - valid_row),
                  static_cast<uint16_t>(dstNzC0Stride - valid_row)});
        this->gatherOutQue.template FreeTensor(kvMergUbCp_);
        if constexpr (syGmEn) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = static_cast<uint16_t>(mte2Size - mte3Size);
            dataCopyParams.blockLen = dValue * sizeof(INPUT_T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(gatherSYResGm[wsElemBase + static_cast<int64_t>(s2GmStartOffset + mte3Size) * dValue],
                        kvMergUb_[ubElemBase], dataCopyParams);
        }
    } else {
        const MutexID mutexId = this->mutexIds_[mergeMte3Idx];
        Mutex::Unlock<PIPE_MTE2>(mutexId);
        if (mte2Size <= mte3Size) {
            return;
        }
        constexpr uint32_t dstNzC0Stride = VEC_SY_BASESIZE;
        Mutex::Lock<PIPE_V>(mutexId);
        LocalTensor<INPUT_T> kvMergUbCp_ = this->gatherOutQue.template Get<INPUT_T>();
        Ub2UbNd2Nz<INPUT_T>(kvMergUbCp_[ubElemBase], kvMergUb_[ubElemBase], GATHER_ROW_NUM, dValue);
        Mutex::Unlock<PIPE_V>(mutexId);
        Mutex::Lock<PIPE_MTE3>(mutexId);
        uint16_t valid_row = static_cast<uint16_t>(mte2Size - mte3Size);
        if constexpr (gatherL1EnT) {
            const uint32_t gAllOff =
                static_cast<uint32_t>(wsElemBase) + (static_cast<uint32_t>(s2GmStartOffset + mte3Size) << 4);
            DataCopy(gatherAllL1Tensor[gAllOff], kvMergUbCp_[ubElemBase],
                     {static_cast<uint16_t>(dValue >> 4), valid_row, static_cast<uint16_t>(GATHER_ROW_NUM - valid_row),
                      static_cast<uint16_t>(dstNzC0Stride - valid_row)});
        } else {
            DataCopy(mm1AL1Tensor[(s2GmStartOffset + mte3Size) << 4], kvMergUbCp_[ubElemBase],
                     {static_cast<uint16_t>(dValue >> 4), valid_row, static_cast<uint16_t>(GATHER_ROW_NUM - valid_row),
                      static_cast<uint16_t>(dstNzC0Stride - valid_row)});
            if constexpr (syGmEn) {
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = static_cast<uint16_t>(mte2Size - mte3Size);
                dataCopyParams.blockLen = dValue * sizeof(INPUT_T);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                DataCopyPad(gatherSYResGm[wsElemBase + static_cast<int64_t>(s2GmStartOffset + mte3Size) * dValue],
                            kvMergUb_[ubElemBase], dataCopyParams);
            }
        }
        Mutex::Unlock<PIPE_MTE3>(mutexId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::SkipAicGatherRows(const SLIGradKRunInfo& kRunInfo)
{
    const int32_t rows = kRunInfo.s2RealBaseSize;
    const int32_t halfSplit = CeilDiv(CeilDiv(rows, 2), 2) * 2;
    const int32_t begin = (GetSubBlockIdx() == 0) ? 0 : halfSplit;
    const int32_t endRow = (GetSubBlockIdx() == 0) ? Min(halfSplit, rows) : rows;
    uint32_t half = this->gatherHalfIdx_;
    for (int32_t row = begin; row < endRow; row += GATHER_ROW_NUM) {
        const MutexID mutexId = this->mutexIds_[half];
        Mutex::Lock<PIPE_MTE2>(mutexId);
        Mutex::Unlock<PIPE_MTE2>(mutexId);
        Mutex::Lock<PIPE_V>(mutexId);
        Mutex::Unlock<PIPE_V>(mutexId);
        Mutex::Lock<PIPE_MTE3>(mutexId);
        Mutex::Unlock<PIPE_MTE3>(mutexId);
        half = GATHER_BUFFER_NUM == 1 ? 0U : 1U - half;
    }
    this->gatherHalfIdx_ = half;
}

TEMPLATES_DEF_NO_DEFAULT
template <event_t IdStart, bool syGmEn, bool gatherL1EnT>
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::MergeKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf, const SLIGradRunInfo& runInfo,
    SLIGradKRunInfo& kRunInfo, GlobalTensor<INPUT_T> srcTensor)
{
    if constexpr (Deterministic) {
        constexpr uint32_t dValue = 128;
        const uint32_t dValueBytes = dValue * static_cast<uint32_t>(sizeof(INPUT_T));
        const int32_t s2IdLimit = static_cast<int32_t>(runInfo.s2SparseLen);
        const int32_t gapOverflowAt =
            static_cast<int32_t>((static_cast<int64_t>(INT32_MAX) + dValueBytes - 1) / dValueBytes + SPARSE_BLOCK_SIZE);
        const int64_t keyRowBase = Shape::AccumS2Idx(runInfo);
        const int64_t topkBase = runInfo.topkGmBaseOffset + static_cast<int64_t>(kRunInfo.s2Idx) * runInfo.s2BaseSize;
        const int64_t wsElemBase = static_cast<int64_t>(kRunInfo.s2Idx) * runInfo.s2BaseSize * dValue;

        const int32_t s2ProcessSize = kRunInfo.s2RealBaseSize;
        const int32_t halfSplit = CeilDiv(CeilDiv(s2ProcessSize, 2), 2) * 2;
        const int32_t s2GmStartOffset = (GetSubBlockIdx() == 0) ? 0 : halfSplit;
        const int32_t s2GmLimit = (GetSubBlockIdx() == 0) ? Min(halfSplit, s2ProcessSize) : s2ProcessSize;

        DataCopyExtParams intriParams;
        intriParams.blockLen = static_cast<uint32_t>(SPARSE_BLOCK_SIZE) * dValueBytes;
        intriParams.blockCount = 1;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> padParams;

        int64_t mte2Size = 0;
        int64_t mte3Size = 0;
        uint32_t ubFill = 0;
        if (s2GmStartOffset < s2GmLimit) {
            WaitFlag<HardEvent::MTE3_MTE2>(IdStart);
        }
        int32_t s2GmOffsetArray = s2GmStartOffset;
        bool terminated = false;
        const __gm__ int32_t* topkPtr = topKGm.GetPhyAddr();
        while (!terminated && s2GmOffsetArray < s2GmLimit) {
            int32_t k = static_cast<int32_t>((GATHER_ROW_NUM - 2 - (mte2Size - mte3Size)) / 2);
            if (k < 1) {
                k = 1;
            }
            int32_t batchEnd = s2GmOffsetArray + 2 * k;
            if (batchEnd > s2GmLimit) {
                batchEnd = s2GmLimit;
            }
            for (; s2GmOffsetArray < batchEnd; s2GmOffsetArray += 2) {
                const int32_t s2IdxArray0 = topkPtr[topkBase + s2GmOffsetArray];
                if (!likely(s2IdxArray0 >= 0)) {
                    CopyOutMrgeResult<IdStart, syGmEn, gatherL1EnT>(static_cast<int32_t>(mte2Size),
                                                                    static_cast<int32_t>(mte3Size), s2GmStartOffset, 0,
                                                                    dValue, 0, wsElemBase);
                    mte3Size = mte2Size;
                    SetFlag<HardEvent::MTE3_MTE2>(IdStart);
                    terminated = true;
                    break;
                }
                const int32_t s2IdxArray1 = (s2GmOffsetArray + SPARSE_BLOCK_SIZE < s2GmLimit) ?
                                                topkPtr[topkBase + s2GmOffsetArray + SPARSE_BLOCK_SIZE] :
                                                -1;
                if constexpr (Deterministic) {
                    const bool valid0 = (s2IdxArray0 < s2IdLimit);
                    const bool valid1 = (static_cast<uint32_t>(s2IdxArray1) < static_cast<uint32_t>(s2IdLimit));
                    if (likely(valid0 && valid1)) {
                        const int64_t gapRows = static_cast<int64_t>(s2IdxArray1) - s2IdxArray0;
                        if (likely(gapRows > 0 && gapRows < gapOverflowAt)) {
                            const int64_t strideBytes = static_cast<int64_t>(gapRows - SPARSE_BLOCK_SIZE) * dValueBytes;
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue,
                                            2, static_cast<uint32_t>(strideBytes));
                        } else {
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue,
                                            1, 0);
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill + dValue,
                                            keyRowBase + s2IdxArray1, dValue, 1, 0);
                        }
                        mte2Size += 2;
                        ubFill += dValue + dValue;
                    } else if (valid0) {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue, 1,
                                        0);
                        mte2Size += 1;
                        ubFill += dValue;
                    } else if (valid1) {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray1, dValue, 1,
                                        0);
                        mte2Size += 1;
                        ubFill += dValue;
                    }
                } else {
                    if (likely(s2IdxArray1 >= 0)) {
                        const int64_t gapRows = static_cast<int64_t>(s2IdxArray1) - s2IdxArray0;
                        if (likely(gapRows > 0 && gapRows < gapOverflowAt)) {
                            const int64_t strideBytes = static_cast<int64_t>(gapRows - SPARSE_BLOCK_SIZE) * dValueBytes;
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue,
                                            2, static_cast<uint32_t>(strideBytes));
                        } else {
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue,
                                            1, 0);
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill + dValue,
                                            keyRowBase + s2IdxArray1, dValue, 1, 0);
                        }
                        mte2Size += 2;
                        ubFill += dValue + dValue;
                    } else {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue, 1,
                                        0);
                        mte2Size += 1;
                        ubFill += dValue;
                    }
                }
            }
            if (mte2Size - mte3Size + 2 > GATHER_ROW_NUM) {
                CopyOutMrgeResult<IdStart, syGmEn, gatherL1EnT>(static_cast<int32_t>(mte2Size),
                                                                static_cast<int32_t>(mte3Size), s2GmStartOffset, 0,
                                                                dValue, 0, wsElemBase);
                mte3Size = mte2Size;
                SetFlag<HardEvent::MTE3_MTE2>(IdStart);
            }
        }
        if (mte2Size > mte3Size) {
            CopyOutMrgeResult<IdStart, syGmEn, gatherL1EnT>(static_cast<int32_t>(mte2Size),
                                                            static_cast<int32_t>(mte3Size), s2GmStartOffset, 0, dValue,
                                                            0, wsElemBase);
            SetFlag<HardEvent::MTE3_MTE2>(IdStart);
        }
        return;
    } else {
        if constexpr (gatherL1EnT && !ResidentPipeline) {
            if (IsAicGatherTile(static_cast<uint32_t>(kRunInfo.kTaskId), runInfo.s2LoopTimes)) {
                SkipAicGatherRows(kRunInfo);
                return;
            }
        }
        constexpr uint32_t dValue = 128;
        constexpr uint32_t ubHalfStride = GATHER_ROW_NUM * dValue;
        const bool pairSortedTile = Shape::PairSortedTile(kRunInfo);
        const uint32_t dValueBytes = dValue * static_cast<uint32_t>(sizeof(INPUT_T));
        const int32_t gapOverflowAt =
            static_cast<int32_t>((static_cast<int64_t>(INT32_MAX) + dValueBytes - 1) / dValueBytes + SPARSE_BLOCK_SIZE);
        const int64_t keyRowBase = Shape::AccumS2Idx(runInfo);
        const int64_t topkBase = runInfo.topkGmBaseOffset + static_cast<int64_t>(kRunInfo.s2Idx) * runInfo.s2BaseSize;
        const int64_t wsElemBase = static_cast<int64_t>(kRunInfo.s2Idx) * runInfo.s2BaseSize * dValue;

        const int32_t s2ProcessSize = kRunInfo.s2RealBaseSize;
        const int32_t halfSplit = CeilDiv(CeilDiv(s2ProcessSize, 2), 2) * 2;
        int32_t s2GmStartOffset = (GetSubBlockIdx() == 0) ? 0 : halfSplit;
        int32_t s2GmLimit = (GetSubBlockIdx() == 0) ? Min(halfSplit, s2ProcessSize) : s2ProcessSize;
        if constexpr (gatherL1EnT && ResidentPipeline) {
            if (s2ProcessSize == VEC_SY_BASESIZE) {
                constexpr int32_t rowsPerAiv = (VEC_SY_BASESIZE - COOP_AIC_GATHER_ROWS) / 2;
                s2GmStartOffset = COOP_AIC_GATHER_ROWS + GetSubBlockIdx() * rowsPerAiv;
                s2GmLimit = s2GmStartOffset + rowsPerAiv;
#if (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1) && (!defined(ASCENDC_OOM) || ASCENDC_OOM != 1) && \
    !defined(ASCENDC_TIME_STAMP_ON)
                __gm__ uint16_t* rawKey =
                    reinterpret_cast<__gm__ uint16_t*>(const_cast<__gm__ INPUT_T*>(srcTensor.GetPhyAddr()));
                const uint64_t keyAddress = reinterpret_cast<uint64_t>(rawKey);
                const bool rawAddressSafe = keyAddress <= AscendC::L2_CACHE_OFFSET_MASK &&
                                            static_cast<uint64_t>(constInfo.totalCost) <=
                                                (AscendC::L2_CACHE_OFFSET_MASK - keyAddress) / dValueBytes;
                if (pairSortedTile && rawAddressSafe) {
                    const uint32_t half = this->gatherHalfIdx_;
                    const uint32_t rawUbBase = half * ubHalfStride;
                    __ubuf__ uint16_t* rawUb = reinterpret_cast<__ubuf__ uint16_t*>(kvMergUb_.GetPhyAddr()) + rawUbBase;
                    rawKey += keyRowBase * dValue;
                    const __gm__ int32_t* indices = topKGm.GetPhyAddr() + topkBase + s2GmStartOffset;
                    const __gm__ int32_t* tileIndices = topKGm.GetPhyAddr() + topkBase;
                    if ((reinterpret_cast<uint64_t>(tileIndices) & 63U) == 0 &&
                        (kRunInfo.kTaskId + 2) * VEC_SY_BASESIZE <= runInfo.s2RealSize) {
                        __gm__ uint64_t* indexCache =
                            reinterpret_cast<__gm__ uint64_t*>(const_cast<__gm__ int32_t*>(tileIndices));
                        const uint32_t firstLine = (static_cast<uint32_t>(s2GmStartOffset) * 4U) & ~63U;
                        constexpr uint32_t cacheLines = (rowsPerAiv * sizeof(int32_t) + 63U) / 64U;
#pragma unroll
                        for (uint32_t line = 0; line < cacheLines; ++line) {
                            asc_datacache_preload(indexCache,
                                                  static_cast<int64_t>(VEC_SY_BASESIZE * 4U + firstLine + line * 64U));
                        }
                    }
                    const auto cache = static_cast<asc_load_l2_cache_mode>(AscendC::ExtractCacheMode(srcTensor));
                    Mutex::Lock<PIPE_MTE2>(this->mutexIds_[half]);
                    uint32_t row = 0;
                    for (; row + 8 <= rowsPerAiv; row += 8) {
                        int32_t index0 = indices[row + 0];
                        int32_t index1 = indices[row + 1];
                        int32_t index2 = indices[row + 2];
                        int32_t index3 = indices[row + 3];
                        int32_t index4 = indices[row + 4];
                        int32_t index5 = indices[row + 5];
                        int32_t index6 = indices[row + 6];
                        int32_t index7 = indices[row + 7];
                        SortSligPair(index0, index1);
                        SortSligPair(index2, index3);
                        SortSligPair(index4, index5);
                        SortSligPair(index6, index7);
                        ::asc_copy_gm2ub_align(rawUb + (row + 0) * dValue,
                                               rawKey + static_cast<int64_t>(index0) * dValue, uint32_t(2), dValueBytes,
                                               0, 0, false, cache, static_cast<uint64_t>(index1 - index0) * dValueBytes,
                                               dValueBytes);
                        ::asc_copy_gm2ub_align(rawUb + (row + 2) * dValue,
                                               rawKey + static_cast<int64_t>(index2) * dValue, uint32_t(2), dValueBytes,
                                               0, 0, false, cache, static_cast<uint64_t>(index3 - index2) * dValueBytes,
                                               dValueBytes);
                        ::asc_copy_gm2ub_align(rawUb + (row + 4) * dValue,
                                               rawKey + static_cast<int64_t>(index4) * dValue, uint32_t(2), dValueBytes,
                                               0, 0, false, cache, static_cast<uint64_t>(index5 - index4) * dValueBytes,
                                               dValueBytes);
                        ::asc_copy_gm2ub_align(rawUb + (row + 6) * dValue,
                                               rawKey + static_cast<int64_t>(index6) * dValue, uint32_t(2), dValueBytes,
                                               0, 0, false, cache, static_cast<uint64_t>(index7 - index6) * dValueBytes,
                                               dValueBytes);
                    }
                    for (; row < rowsPerAiv; row += 2) {
                        int32_t index0 = indices[row];
                        int32_t index1 = indices[row + 1];
                        SortSligPair(index0, index1);
                        ::asc_copy_gm2ub_align(rawUb + row * dValue, rawKey + static_cast<int64_t>(index0) * dValue,
                                               uint32_t(2), dValueBytes, 0, 0, false, cache,
                                               static_cast<uint64_t>(index1 - index0) * dValueBytes, dValueBytes);
                    }
                    CopyOutMrgeResult<IdStart, syGmEn, gatherL1EnT>(rowsPerAiv, 0, s2GmStartOffset, half, dValue,
                                                                    rawUbBase, wsElemBase);
                    this->gatherHalfIdx_ = GATHER_BUFFER_NUM == 1 ? 0U : 1U - half;
                    return;
                }
#endif
            }
        }

        DataCopyExtParams intriParams;
        intriParams.blockLen = static_cast<uint32_t>(SPARSE_BLOCK_SIZE) * dValueBytes;
        intriParams.blockCount = 1;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> padParams;

        int64_t mte2Size = 0;
        int64_t mte3Size = 0;
        int32_t mergeMte3Idx = static_cast<int32_t>(this->gatherHalfIdx_);
        uint32_t ubBase = mergeMte3Idx * ubHalfStride;
        uint32_t ubFill = ubBase;
        bool needLockMte2 = true;
        int32_t s2GmOffsetArray = s2GmStartOffset;
        const __gm__ int32_t* topkPtr = topKGm.GetPhyAddr();
        if ((s2ProcessSize & 1) == 0) {
            while (s2GmOffsetArray < s2GmLimit) {
                if (needLockMte2) {
                    Mutex::Lock<PIPE_MTE2>(this->mutexIds_[mergeMte3Idx]);
                    needLockMte2 = false;
                }
                int32_t k = static_cast<int32_t>((GATHER_ROW_NUM - 2 - (mte2Size - mte3Size)) / 2);
                if (k < 1) {
                    k = 1;
                }
                int32_t batchEnd = s2GmOffsetArray + 2 * k;
                if (batchEnd > s2GmLimit) {
                    batchEnd = s2GmLimit;
                }
                const int32_t unrollEnd = s2GmOffsetArray + ((batchEnd - s2GmOffsetArray) / 8) * 8;
                for (; s2GmOffsetArray < unrollEnd; s2GmOffsetArray += 8) {
                    int32_t s2IdxArray0 = topkPtr[topkBase + s2GmOffsetArray];
                    int32_t s2IdxArray1 = topkPtr[topkBase + s2GmOffsetArray + 1];
                    int32_t s2IdxArray2 = topkPtr[topkBase + s2GmOffsetArray + 2];
                    int32_t s2IdxArray3 = topkPtr[topkBase + s2GmOffsetArray + 3];
                    int32_t s2IdxArray4 = topkPtr[topkBase + s2GmOffsetArray + 4];
                    int32_t s2IdxArray5 = topkPtr[topkBase + s2GmOffsetArray + 5];
                    int32_t s2IdxArray6 = topkPtr[topkBase + s2GmOffsetArray + 6];
                    int32_t s2IdxArray7 = topkPtr[topkBase + s2GmOffsetArray + 7];
                    if constexpr (ResidentPipeline) {
                        if (pairSortedTile) {
                            SortSligPair(s2IdxArray0, s2IdxArray1);
                            SortSligPair(s2IdxArray2, s2IdxArray3);
                            SortSligPair(s2IdxArray4, s2IdxArray5);
                            SortSligPair(s2IdxArray6, s2IdxArray7);
                        }
                    }
                    const int64_t keyOffset0 = keyRowBase + s2IdxArray0;
                    const int64_t keyOffset1 = keyRowBase + s2IdxArray1;
                    const int64_t keyOffset2 = keyRowBase + s2IdxArray2;
                    const int64_t keyOffset3 = keyRowBase + s2IdxArray3;
                    const int64_t keyOffset4 = keyRowBase + s2IdxArray4;
                    const int64_t keyOffset5 = keyRowBase + s2IdxArray5;
                    const int64_t keyOffset6 = keyRowBase + s2IdxArray6;
                    const int64_t keyOffset7 = keyRowBase + s2IdxArray7;
                    const int64_t gapRows01 = static_cast<int64_t>(s2IdxArray1) - s2IdxArray0;
                    const int64_t gapRows23 = static_cast<int64_t>(s2IdxArray3) - s2IdxArray2;
                    const int64_t gapRows45 = static_cast<int64_t>(s2IdxArray5) - s2IdxArray4;
                    const int64_t gapRows67 = static_cast<int64_t>(s2IdxArray7) - s2IdxArray6;
                    if (likely(gapRows01 > 0 && gapRows01 < gapOverflowAt)) {
                        const int64_t strideBytes = (gapRows01 - SPARSE_BLOCK_SIZE) * dValueBytes;
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyOffset0, dValue, 2,
                                        static_cast<uint32_t>(strideBytes));
                    } else {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyOffset0, dValue, 1, 0);
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + dValue, keyOffset1, dValue, 1, 0);
                    }
                    if (likely(gapRows23 > 0 && gapRows23 < gapOverflowAt)) {
                        const int64_t strideBytes = (gapRows23 - SPARSE_BLOCK_SIZE) * dValueBytes;
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 2 * dValue, keyOffset2, dValue, 2,
                                        static_cast<uint32_t>(strideBytes));
                    } else {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 2 * dValue, keyOffset2, dValue, 1,
                                        0);
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 3 * dValue, keyOffset3, dValue, 1,
                                        0);
                    }
                    if (likely(gapRows45 > 0 && gapRows45 < gapOverflowAt)) {
                        const int64_t strideBytes = (gapRows45 - SPARSE_BLOCK_SIZE) * dValueBytes;
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 4 * dValue, keyOffset4, dValue, 2,
                                        static_cast<uint32_t>(strideBytes));
                    } else {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 4 * dValue, keyOffset4, dValue, 1,
                                        0);
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 5 * dValue, keyOffset5, dValue, 1,
                                        0);
                    }
                    if (likely(gapRows67 > 0 && gapRows67 < gapOverflowAt)) {
                        const int64_t strideBytes = (gapRows67 - SPARSE_BLOCK_SIZE) * dValueBytes;
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 6 * dValue, keyOffset6, dValue, 2,
                                        static_cast<uint32_t>(strideBytes));
                    } else {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 6 * dValue, keyOffset6, dValue, 1,
                                        0);
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + 7 * dValue, keyOffset7, dValue, 1,
                                        0);
                    }
                    mte2Size += 8;
                    ubFill += 8 * dValue;
                }
                for (; s2GmOffsetArray < batchEnd; s2GmOffsetArray += 2) {
                    int32_t s2IdxArray0 = topkPtr[topkBase + s2GmOffsetArray];
                    int32_t s2IdxArray1 = topkPtr[topkBase + s2GmOffsetArray + SPARSE_BLOCK_SIZE];
                    if constexpr (ResidentPipeline) {
                        if (pairSortedTile) {
                            SortSligPair(s2IdxArray0, s2IdxArray1);
                        }
                    }
                    const int64_t gapRows = static_cast<int64_t>(s2IdxArray1) - s2IdxArray0;
                    if (likely(gapRows > 0 && gapRows < gapOverflowAt)) {
                        const int64_t strideBytes = static_cast<int64_t>(gapRows - SPARSE_BLOCK_SIZE) * dValueBytes;
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue, 2,
                                        static_cast<uint32_t>(strideBytes));
                    } else {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue, 1,
                                        0);
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill + dValue, keyRowBase + s2IdxArray1,
                                        dValue, 1, 0);
                    }
                    mte2Size += 2;
                    ubFill += dValue + dValue;
                }
                if (mte2Size - mte3Size + 2 > GATHER_ROW_NUM) {
                    CopyOutMrgeResult<IdStart, syGmEn, gatherL1EnT>(static_cast<int32_t>(mte2Size),
                                                                    static_cast<int32_t>(mte3Size), s2GmStartOffset,
                                                                    mergeMte3Idx, dValue, ubBase, wsElemBase);
                    mte3Size = mte2Size;
                    mergeMte3Idx = GATHER_BUFFER_NUM == 1 ? 0 : 1 - mergeMte3Idx;
                    ubBase = GATHER_BUFFER_NUM == 1 ? 0U : ubHalfStride - ubBase;
                    ubFill = ubBase;
                    needLockMte2 = true;
                }
            }
        } else {
            bool terminated = false;
            while (!terminated && s2GmOffsetArray < s2GmLimit) {
                if (needLockMte2) {
                    Mutex::Lock<PIPE_MTE2>(this->mutexIds_[mergeMte3Idx]);
                    needLockMte2 = false;
                }
                int32_t k = static_cast<int32_t>((GATHER_ROW_NUM - 2 - (mte2Size - mte3Size)) / 2);
                if (k < 1) {
                    k = 1;
                }
                int32_t batchEnd = s2GmOffsetArray + 2 * k;
                if (batchEnd > s2GmLimit) {
                    batchEnd = s2GmLimit;
                }
                for (; s2GmOffsetArray < batchEnd; s2GmOffsetArray += 2) {
                    const int32_t s2IdxArray0 = topkPtr[topkBase + s2GmOffsetArray];
                    if (!likely(s2IdxArray0 >= 0)) {
                        CopyOutMrgeResult<IdStart, syGmEn, gatherL1EnT>(static_cast<int32_t>(mte2Size),
                                                                        static_cast<int32_t>(mte3Size), s2GmStartOffset,
                                                                        mergeMte3Idx, dValue, ubBase, wsElemBase);
                        if (mte2Size > mte3Size) {
                            mergeMte3Idx = GATHER_BUFFER_NUM == 1 ? 0 : 1 - mergeMte3Idx;
                            ubBase = GATHER_BUFFER_NUM == 1 ? 0U : ubHalfStride - ubBase;
                        }
                        mte3Size = mte2Size;
                        terminated = true;
                        break;
                    }
                    const int32_t s2IdxArray1 = (s2GmOffsetArray + SPARSE_BLOCK_SIZE < s2GmLimit) ?
                                                    topkPtr[topkBase + s2GmOffsetArray + SPARSE_BLOCK_SIZE] :
                                                    -1;
                    if (likely(s2IdxArray1 >= 0)) {
                        const int64_t gapRows = static_cast<int64_t>(s2IdxArray1) - s2IdxArray0;
                        if (likely(gapRows > 0 && gapRows < gapOverflowAt)) {
                            const int64_t strideBytes = static_cast<int64_t>(gapRows - SPARSE_BLOCK_SIZE) * dValueBytes;
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue,
                                            2, static_cast<uint32_t>(strideBytes));
                        } else {
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue,
                                            1, 0);
                            IssueGatherRows(srcTensor, intriParams, padParams, ubFill + dValue,
                                            keyRowBase + s2IdxArray1, dValue, 1, 0);
                        }
                        mte2Size += 2;
                        ubFill += dValue + dValue;
                    } else {
                        IssueGatherRows(srcTensor, intriParams, padParams, ubFill, keyRowBase + s2IdxArray0, dValue, 1,
                                        0);
                        mte2Size += 1;
                        ubFill += dValue;
                    }
                }
                if (mte2Size - mte3Size + 2 > GATHER_ROW_NUM) {
                    CopyOutMrgeResult<IdStart, syGmEn, gatherL1EnT>(static_cast<int32_t>(mte2Size),
                                                                    static_cast<int32_t>(mte3Size), s2GmStartOffset,
                                                                    mergeMte3Idx, dValue, ubBase, wsElemBase);
                    mte3Size = mte2Size;
                    mergeMte3Idx = GATHER_BUFFER_NUM == 1 ? 0 : 1 - mergeMte3Idx;
                    ubBase = GATHER_BUFFER_NUM == 1 ? 0U : ubHalfStride - ubBase;
                    ubFill = ubBase;
                    needLockMte2 = true;
                }
            }
        }
        if (mte2Size > mte3Size) {
            CopyOutMrgeResult<IdStart, syGmEn, gatherL1EnT>(static_cast<int32_t>(mte2Size),
                                                            static_cast<int32_t>(mte3Size), s2GmStartOffset,
                                                            mergeMte3Idx, dValue, ubBase, wsElemBase);
            mergeMte3Idx = GATHER_BUFFER_NUM == 1 ? 0 : 1 - mergeMte3Idx;
            ubBase = GATHER_BUFFER_NUM == 1 ? 0U : ubHalfStride - ubBase;
        }
        this->gatherHalfIdx_ = static_cast<uint32_t>(mergeMte3Idx);
        return;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool SligBlockVec<TEMPLATE_ARGS>::GetBS1Index(int64_t usedT1Index, int64_t& bIdx, int64_t& s1Idx,
                                                                int64_t& bS1Index, int64_t& accumS1Len,
                                                                int64_t& accumS2Len, int32_t& actualSeqLensQ,
                                                                int32_t& actualSeqLensK, int64_t taskId)
{
    int64_t t1Offset = 0;
    int64_t t2Offset = 0;
    int64_t bIndex = HasSequsedQ ? -1 : 0;
    int64_t s1Index = 0;
    int64_t nextUsedT1Offset = 0;
    int64_t usedT1Offset = 0;
    int64_t curT1 = 0;
    int64_t curT2 = 0;
    int64_t curS1 = constInfo.s1Size;
    int64_t curS2 = constInfo.s2Size;
    int64_t usedS1 = 0;
    int64_t usedS2 = 0;
    if (usedT1Index >= constInfo.totalNum) {
        return false;
    }
    if constexpr (!HasSequsedQ) {
        if constexpr (Layout_Q == SLILayout::TND) {
            curT1 = ((__gm__ int32_t*)actualSeqQlenAddr)[bIndex];
            while (usedT1Index >= curT1) {
                curT1 = ((__gm__ int32_t*)actualSeqQlenAddr)[++bIndex];
            }
            bIndex = bIndex - 1;
            t1Offset = ((__gm__ int32_t*)actualSeqQlenAddr)[bIndex];
            s1Index = usedT1Index - t1Offset;
        } else {
            bIndex = usedT1Index / constInfo.s1Size;
            s1Index = usedT1Index % constInfo.s1Size;
            t1Offset = usedT1Index - s1Index;
        }
        bS1Index = usedT1Index;
    } else {
        while (usedT1Index >= nextUsedT1Offset) {
            usedT1Offset = nextUsedT1Offset;
            nextUsedT1Offset = ((__gm__ int32_t*)usedSeqQlenAddr)[++bIndex] + usedT1Offset;
        }
        s1Index = usedT1Index - usedT1Offset;
        if constexpr (Layout_Q == SLILayout::TND) {
            t1Offset = ((__gm__ int32_t*)actualSeqQlenAddr)[bIndex];
        } else {
            t1Offset = bIndex * constInfo.s1Size;
        }
        bS1Index = t1Offset + s1Index;
    }
    if constexpr (Layout_Q == SLILayout::TND) {
        if (unlikely(bIndex == 0)) {
            t2Offset = 0;
            curS1 = ((__gm__ int32_t*)actualSeqQlenAddr)[bIndex + 1];
            curS2 = ((__gm__ int32_t*)actualSeqKvlenAddr)[bIndex + 1];
        } else {
            curS1 = ((__gm__ int32_t*)actualSeqQlenAddr)[bIndex + 1] - ((__gm__ int32_t*)actualSeqQlenAddr)[bIndex];
            curS2 = ((__gm__ int32_t*)actualSeqKvlenAddr)[bIndex + 1] - ((__gm__ int32_t*)actualSeqKvlenAddr)[bIndex];
            t2Offset = ((__gm__ int32_t*)actualSeqKvlenAddr)[bIndex];
        }
    } else {
        t2Offset = bIndex * constInfo.s2Size;
    }

    if constexpr (HasSequsedQ) {
        usedS1 = ((__gm__ int32_t*)usedSeqQlenAddr)[bIndex];
    } else {
        usedS1 = curS1;
    }
    if constexpr (HasSequsedK) {
        usedS2 = ((__gm__ int32_t*)usedSeqKvlenAddr)[bIndex];
    } else {
        usedS2 = curS2;
    }
    bIdx = bIndex;
    s1Idx = s1Index;
    actualSeqLensQ = usedS1;
    actualSeqLensK = usedS2;
    accumS1Len = t1Offset;
    accumS2Len = t2Offset;
    return true;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx,
                                                               SLIGradRunInfo& runInfo, int64_t accumS1Len,
                                                               int64_t accumS2Len, int32_t actualSeqLensQ,
                                                               int32_t actualSeqLensK)
{
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;

    runInfo.bIdx = bIdx;
    runInfo.s1Idx = s1Idx;
    runInfo.actS1Size = actualSeqLensQ;
    runInfo.actS2Size = actualSeqLensK;
    runInfo.accumS1Idx = accumS1Len + s1Idx;
    runInfo.accumS2Idx = accumS2Len;
    runInfo.s2SparseLen = GetS2SparseLen(runInfo.bIdx, runInfo.s1Idx, runInfo.actS1Size, runInfo.actS2Size);
    runInfo.s2RealSize = Min(constInfo.kSize, runInfo.s2SparseLen);

    runInfo.kRealSize = runInfo.s2RealSize;

    runInfo.topkGmBaseOffset = runInfo.accumS1Idx * constInfo.kSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t SligBlockVec<TEMPLATE_ARGS>::GetS2SparseLen(int32_t bIdx, int32_t s1Idx,
                                                                      int32_t actualSeqLensQ, int32_t actualSeqLensK)
{
    if (constInfo.sparseMode == SLISparseMode::RightDown) {
        if (constInfo.cmpRatio == 1) {
            return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
        }
        int32_t cmpResidualK = 0;
        if constexpr (HasCmpResidualK) {
            cmpResidualK = ((__gm__ uint32_t*)cmpResidualKAddr)[bIdx];
        }
        return Max(
            ((actualSeqLensK * constInfo.cmpRatio + cmpResidualK) - actualSeqLensQ + s1Idx + 1) / constInfo.cmpRatio,
            0);
    } else if (constInfo.sparseMode == SLISparseMode::DefaultMask) {
        return actualSeqLensK;
    } else {
        return 0;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::CopyInWeight(SLIGradRunInfo& runInfo)
{
    if constexpr (Deterministic) {
        LocalTensor<float> weightInUb = weightInQue.template AllocTensor<float>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = runInfo.nIndexSize * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams;
        DataCopyPad(weightInUb, weightGm[runInfo.weightOffset], dataCopyParams, padParams);
        this->weightInQue.template EnQue(weightInUb);
        this->weightInQue.template DeQue<float>();
        this->weightInQue.template FreeTensor(weightInUb);
    } else {
        Mutex::Lock<PIPE_MTE2>(this->weightMutexId_);
        if constexpr (ResidentPipeline) {
            this->pairByteCacheEn_ = BytePairCache && runInfo.actS2Size <= PAIR_BYTE_INDEX_MAX_ROWS;
            if (runInfo.s2RealSize >= VEC_P_BASESIZE) {
                SetFlag<HardEvent::S_MTE2>(this->eventPairIndexSToMte2_);
                WaitFlag<HardEvent::S_MTE2>(this->eventPairIndexSToMte2_);
                DataCopyExtParams indexCopyParams;
                indexCopyParams.blockCount = 1;
                indexCopyParams.blockLen = (runInfo.s2RealSize / VEC_P_BASESIZE) * VEC_P_BASESIZE * sizeof(int32_t);
                indexCopyParams.srcStride = 0;
                indexCopyParams.dstStride = 0;
                DataCopyPadExtParams<int32_t> indexPadParams;
                DataCopyPad(this->pairIndexUb_, topKGm[runInfo.topkGmBaseOffset], indexCopyParams, indexPadParams);
            }
        }
        LocalTensor<float> weightInUb = weightInQue.template Get<float>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = Shape::NIndexSize(runInfo) * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams;
        DataCopyPad(weightInUb, weightGm[runInfo.weightOffset], dataCopyParams, padParams);
        Mutex::Unlock<PIPE_MTE2>(this->weightMutexId_);
        if constexpr (!ResidentPipeline) {
            Mutex::Lock<PIPE_V>(this->weightMutexId_);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf, SLIGradRunInfo& runInfo,
    SLIGradKRunInfo& kRunInfo)
{
    if constexpr (Deterministic) {
        if (kRunInfo.s2SingleIdx == 0) {
            CrossCoreWaitFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
        }
        auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        ProcessVec1Vf<T, float>(reduceSumTensor[runInfo.s2CurSize], mmRes[kRunInfo.s2SingleCurSize], weightInUb_,
                                runInfo.nIndexSize, constInfo.syKBaseSize);
        if (kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1) {
            CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
        }
        if (kRunInfo.isS2end) {
            this->reduceSumOutQue.template EnQue(reduceSumTensor);
            this->reduceSumOutQue.template DeQue<T>();
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = runInfo.s2RealSize * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(reduceSumResGm[constInfo.subBlockIdx * constInfo.kSizeAlign128], reduceSumTensor,
                        dataCopyParams);
            CrossCoreSetFlag<1, PIPE_MTE3>(EVENT_ID2);
        }
        this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
    } else {
        CrossCoreWaitFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
        if (kRunInfo.kTaskId == 0) {
            if constexpr (ResidentPipeline) {
                Mutex::Lock<PIPE_V>(this->weightMutexId_);
            }
            Mutex::Lock<PIPE_V>(this->reduceSumMutexId_);
        }
        auto reduceSumTensor = this->reduceSumOutQue.template Get<T>();

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        ProcessVec1Vf<T, float>(reduceSumTensor[runInfo.s2CurSize], mmRes, weightInUb_, Shape::NIndexSize(runInfo),
                                VEC_SY_BASESIZE);
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
        if (kRunInfo.isS2end) {
            Mutex::Unlock<PIPE_V>(this->reduceSumMutexId_);
            Mutex::Lock<PIPE_MTE3>(this->reduceSumMutexId_);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = runInfo.s2RealSize * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            if constexpr (ResidentPipeline) {
                const uint32_t copyRows = (static_cast<uint32_t>(runInfo.s2RealSize) + 7U) & ~7U;
                DataCopy(this->sumRelayL1_[constInfo.subBlockIdx * constInfo.kSizeAlign128], reduceSumTensor, copyRows);
                Mutex::Unlock<PIPE_MTE3>(this->reduceSumMutexId_);
                CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_SUM_L1_RELAY_FLAG);
            } else {
                DataCopyPad(reduceSumResGm[constInfo.subBlockIdx * constInfo.kSizeAlign128], reduceSumTensor,
                            dataCopyParams);
                Mutex::Unlock<PIPE_MTE3>(this->reduceSumMutexId_);
                CrossCoreSetFlag<1, PIPE_MTE3>(EVENT_ID2);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf, SLIGradRunInfo& runInfo)
{
    if constexpr (Deterministic) {
        auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();
        LocalTensor<T> reduceOtherSumTensor = bmm1ResBuf.template GetTensor<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = runInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams;
        CrossCoreWaitFlag<1, PIPE_MTE2>(EVENT_ID2);
        DataCopyPad(reduceOtherSumTensor, reduceSumResGm[(1 - constInfo.subBlockIdx) * constInfo.kSizeAlign128],
                    dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
        if (runInfo.s2TailSize == VEC_SY_BASESIZE) {
            ProcessVec2Vf<T, true>(reduceSumTensor, reduceSumTensor, reduceOtherSumTensor, runInfo.s2RealSize);
        } else {
            ProcessVec2Vf<T, false>(reduceSumTensor, reduceSumTensor, reduceOtherSumTensor, runInfo.s2RealSize);
        }
        this->reduceSumOutQue.template EnQue(reduceSumTensor);
        this->reduceSumOutQue.template DeQue<T>();
        int64_t softmaxOutOffset = runInfo.t1Index * constInfo.kSize;
        dataCopyParams.dstStride = constInfo.kSize;
        DataCopyPad(softmaxOutGm[softmaxOutOffset], reduceSumTensor, dataCopyParams);
        this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
    } else {
        Mutex::Lock<PIPE_V>(this->reduceSumMutexId_);
        auto reduceSumTensor = this->reduceSumOutQue.template Get<T>();
        LocalTensor<T> reduceOtherSumTensor = bmm1ResBuf.template GetTensor<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = runInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams;
        if constexpr (ResidentPipeline) {
            CrossCoreWaitFlag<2, PIPE_V>(SYNC_SUM_L1_RELAY_FLAG);
        } else {
            CrossCoreWaitFlag<1, PIPE_MTE2>(EVENT_ID2);
            DataCopyPad(reduceOtherSumTensor, reduceSumResGm[(1 - constInfo.subBlockIdx) * constInfo.kSizeAlign128],
                        dataCopyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
            WaitFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
        }
        if (runInfo.s2TailSize == VEC_SY_BASESIZE) {
            ProcessVec2Vf<T, true>(reduceSumTensor, reduceSumTensor, reduceOtherSumTensor, runInfo.s2RealSize);
        } else {
            ProcessVec2Vf<T, false>(reduceSumTensor, reduceSumTensor, reduceOtherSumTensor, runInfo.s2RealSize);
        }
        if constexpr (ResidentPipeline) {
            if (runInfo.s2RealSize >= VEC_SY_BASESIZE) {
                const uint32_t realRows = static_cast<uint32_t>(runInfo.s2RealSize);
                const uint32_t fullRows = (realRows / VEC_SY_BASESIZE) * VEC_SY_BASESIZE;
                const uint32_t firstRows = Min(realRows, ((realRows + 127U) / 128U) * 64U);
                const uint32_t exportStart = constInfo.subBlockIdx == 0 ? 0 : firstRows;
                const uint32_t exportRows = constInfo.subBlockIdx == 0 ? firstRows : realRows - firstRows;
                const uint32_t sortedRows = exportStart < fullRows ? Min(exportRows, fullRows - exportStart) : 0U;
                LocalTensor<T> exportTensor = reduceSumTensor[PAIR_EXPORT_UB_OFFSET / sizeof(T)];
                PairPermuteFloatVf(exportTensor, reduceSumTensor[exportStart], this->pairIndexUb_[exportStart],
                                   exportRows, sortedRows);
                Mutex::Unlock<PIPE_V>(this->reduceSumMutexId_);
                Mutex::Lock<PIPE_MTE3>(this->reduceSumMutexId_);
                dataCopyParams.blockLen = exportRows * sizeof(T);
                dataCopyParams.dstStride = 0;
                DataCopyPad(softmaxOutGm[runInfo.t1Index * constInfo.kSize + exportStart], exportTensor,
                            dataCopyParams);
                Mutex::Unlock<PIPE_MTE3>(this->reduceSumMutexId_);
                return;
            }
        }
        Mutex::Unlock<PIPE_V>(this->reduceSumMutexId_);
        Mutex::Lock<PIPE_MTE3>(this->reduceSumMutexId_);
        int64_t softmaxOutOffset = runInfo.t1Index * constInfo.kSize;
        dataCopyParams.dstStride = constInfo.kSize;
        DataCopyPad(softmaxOutGm[softmaxOutOffset], reduceSumTensor, dataCopyParams);
        Mutex::Unlock<PIPE_MTE3>(this->reduceSumMutexId_);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::CopyInAttnNorm(SLIGradRunInfo& runInfo)
{
    if constexpr (ResidentPipeline) {
        DataCopyExtParams copy;
        DataCopyPadExtParams<T> pad;
        copy.blockCount = 1;
        copy.blockLen = constInfo.kSize * sizeof(T);
        copy.srcStride = 0;
        copy.dstStride = 0;
        DataCopyPad(attnNormTbuf.Get<T>(), attnSoftmaxL1Gm[runInfo.t1Index * constInfo.kSize], copy, pad);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::PrefetchFirstRelu(SLIGradRunInfo& runInfo)
{
    if constexpr (ResidentPipeline) {
        WaitFlag<HardEvent::V_MTE2>(eventReluVToMte2_);
        CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_FIRST_RELU_TO_V6_FLAG);
        DataCopyExtParams copy;
        DataCopyPadExtParams<T> pad;
        copy.blockCount = Shape::NIndexSize(runInfo);
        copy.blockLen = VEC_P_BASESIZE * sizeof(T);
        copy.srcStride = (constInfo.kSizeAlign128 - VEC_P_BASESIZE) * sizeof(T);
        copy.dstStride = 0;
        DataCopyPad(reluUb_, reluGm[Shape::FirstNIndexSize(runInfo) * constInfo.subBlockIdx * constInfo.kSizeAlign128],
                    copy, pad);
        SetFlag<HardEvent::MTE2_V>(eventReluMte2ToV_);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector6(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf, SLIGradRunInfo& runInfo,
    SLIGradKRunInfo& kRunInfo)
{
    if constexpr (Deterministic) {
        if (kRunInfo.s2SingleIdx == 0) {
            CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
        }
        auto reluGradUb = this->reluGradQue.template AllocTensor<INPUT_T>();
        LocalTensor<T> weightInUb = weightInQue.template AllocTensor<T>();
        LocalTensor<T> softmaxTensor = gatherTbuf.Get<T>();
        DataCopyExtParams dataCopyParams;
        DataCopyPadExtParams<T> padParams;
        if (kRunInfo.kTaskId == 0) {
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = constInfo.kSize * sizeof(T);
            dataCopyParams.srcStride = (2048 - constInfo.kSize) * sizeof(T);
            dataCopyParams.dstStride = 0;
            DataCopyPad(softmaxTensor, attnSoftmaxL1Gm[runInfo.t1Index * constInfo.kSize], dataCopyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
            WaitFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
            ReduceSumVf(softmaxReduceSumUb_, softmaxTensor, constInfo.kSize);
        }
        dataCopyParams.blockCount = runInfo.nIndexSize;
        dataCopyParams.blockLen = VEC_P_BASESIZE * sizeof(T);
        dataCopyParams.srcStride = (constInfo.kSizeAlign128 - VEC_P_BASESIZE) * sizeof(T);
        dataCopyParams.dstStride = 0;
        WaitFlag<HardEvent::V_MTE2>(eventReluVToMte2_);
        DataCopyPad(
            reluUb_,
            reluGm[runInfo.s2CurSize + runInfo.firstNIndexSize * constInfo.subBlockIdx * constInfo.kSizeAlign128],
            dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventReluMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventReluMte2ToV_);

        if (kRunInfo.isAlign64) {
            ProcessVec6Vf<T, INPUT_T, true>(reluGradUb[kRunInfo.s2SingleCurSize * runInfo.nIndexSize], dwUb_,
                                            softmaxTensor[runInfo.s2CurSize], softmaxReduceSumUb_, weightInUb,
                                            reduceSumUb_[runInfo.s2CurSize], reluUb_, kRunInfo.s2RealBaseSize,
                                            runInfo.nIndexSize, VEC_P_BASESIZE);
        } else {
            ProcessVec6Vf<T, INPUT_T, false>(reluGradUb[kRunInfo.s2SingleCurSize * runInfo.nIndexSize], dwUb_,
                                             softmaxTensor[runInfo.s2CurSize], softmaxReduceSumUb_, weightInUb,
                                             reduceSumUb_[runInfo.s2CurSize], reluUb_, kRunInfo.s2RealBaseSize,
                                             runInfo.nIndexSize, VEC_P_BASESIZE);
        }
        SetFlag<HardEvent::V_MTE2>(eventReluVToMte2_);
        if (kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1) {
            this->reluGradQue.template EnQue(reluGradUb);
            this->reluGradQue.template DeQue<INPUT_T>();
            LocalTensor<INPUT_T> mm3AL1Tensor = outputBuf.GetTensor<INPUT_T>();
            DataCopy(mm3AL1Tensor[constInfo.subBlockIdx * runInfo.firstNIndexSize * BLOCK_SINGLE_LEN], reluGradUb,
                     {static_cast<uint16_t>(constInfo.pKBaseSize / BLOCK_SINGLE_LEN),
                      static_cast<uint16_t>(runInfo.nIndexSize), 0,
                      static_cast<uint16_t>(gRealSizeAlignTo16 - runInfo.nIndexSize)});
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
        }
        if (kRunInfo.isS2end) {
            LocalTensor<T> dwOutTensor = dwOutQue.template AllocTensor<T>();
            CastDupVf<T, T>(dwOutTensor, dwUb_, runInfo.nIndexSize);
            this->dwOutQue.template EnQue(dwOutTensor);
            this->dwOutQue.template DeQue<T>();
            DataCopyExtParams dataCopyParams1;
            dataCopyParams1.blockCount = 1;
            dataCopyParams1.blockLen = runInfo.nIndexSize * sizeof(T);
            dataCopyParams1.srcStride = 0;
            dataCopyParams1.dstStride = 0;
            DataCopyPad(dWeightGm[runInfo.weightOffset], dwOutTensor, dataCopyParams1);
            this->dwOutQue.template FreeTensor(dwOutTensor);
        }
        this->reluGradQue.template FreeTensor(reluGradUb);
        this->weightInQue.template FreeTensor(weightInUb);
    } else {
        if constexpr (!ResidentPipeline) {
            if (Shape::GatherResident(constInfo) &&
                IsAicGatherTile(static_cast<uint32_t>(kRunInfo.kTaskId), runInfo.s2LoopTimes)) {
                const int64_t indexOffset = runInfo.topkGmBaseOffset + runInfo.s2CurSize +
                                            static_cast<int64_t>(constInfo.subBlockIdx) * (VEC_P_BASESIZE / 2);
                __gm__ int32_t* indexPtr = const_cast<__gm__ int32_t*>(topKGm.GetPhyAddr()) + indexOffset;
                if ((reinterpret_cast<uint64_t>(indexPtr) & 63U) == 0) {
                    __gm__ uint64_t* cachePtr = reinterpret_cast<__gm__ uint64_t*>(indexPtr);
                    asc_datacache_preload(cachePtr, int64_t(0));
                    asc_datacache_preload(cachePtr, int64_t(64));
                    asc_datacache_preload(cachePtr, int64_t(128));
                    asc_datacache_preload(cachePtr, int64_t(192));
                }
            }
        }
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
        Mutex::Lock<PIPE_V>(this->reluGradMutexId_);
        const int32_t nIndexSize = Shape::NIndexSize(runInfo);
        const int32_t firstNIndexSize = Shape::FirstNIndexSize(runInfo);
        const uint32_t gAlign16 = Shape::GAlign16(gRealSizeAlignTo16);
        auto reluGradUb = this->reluGradQue.template Get<INPUT_T>();
        LocalTensor<T> weightInUb = weightInQue.template Get<T>();
        LocalTensor<T> softmaxTensor = attnNormTbuf.Get<T>();
        DataCopyExtParams dataCopyParams;
        DataCopyPadExtParams<T> padParams;
        if (kRunInfo.kTaskId == 0) {
            if constexpr (!ResidentPipeline) {
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = constInfo.kSize * sizeof(T);
                dataCopyParams.srcStride = (2048 - constInfo.kSize) * sizeof(T);
                dataCopyParams.dstStride = 0;
                DataCopyPad(softmaxTensor, attnSoftmaxL1Gm[runInfo.t1Index * constInfo.kSize], dataCopyParams,
                            padParams);
                SetFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
                WaitFlag<HardEvent::MTE2_V>(eventMte2ToVShared_);
            } else {
                WaitFlag<HardEvent::MTE2_V>(eventReluMte2ToV_);
            }
            ReduceSumVf(softmaxReduceSumUb_, softmaxTensor, constInfo.kSize);
            if constexpr (ResidentPipeline) {
                const uint32_t fullRows = (static_cast<uint32_t>(runInfo.s2RealSize) / VEC_P_BASESIZE) * VEC_P_BASESIZE;
                if (fullRows != 0) {
                    if (this->pairByteCacheEn_) {
                        PairPermuteFloatVf<T, true, true>(softmaxTensor, softmaxTensor, this->pairIndexUb_, fullRows,
                                                          fullRows);
                    } else {
                        PairPermuteFloatVf<T, true, false>(softmaxTensor, softmaxTensor, this->pairIndexUb_, fullRows,
                                                           fullRows);
                    }
                    SetFlag<HardEvent::V_S>(this->eventPairIndexVToS_);
                }
            }
        }
        if constexpr (ResidentPipeline) {
            if (kRunInfo.kTaskId != 0) {
                WaitFlag<HardEvent::MTE2_V>(eventReluMte2ToV_);
            }
            if (runInfo.s2LoopTimes == 1) {
                CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_RELU_TO_V6_FLAG);
            }
        }
        if constexpr (!ResidentPipeline) {
            dataCopyParams.blockCount = nIndexSize;
            dataCopyParams.blockLen = VEC_P_BASESIZE * sizeof(T);
            dataCopyParams.srcStride = (constInfo.kSizeAlign128 - VEC_P_BASESIZE) * sizeof(T);
            dataCopyParams.dstStride = 0;
            WaitFlag<HardEvent::V_MTE2>(eventReluVToMte2_);
            DataCopyPad(reluUb_,
                        reluGm[runInfo.s2CurSize + firstNIndexSize * constInfo.subBlockIdx * constInfo.kSizeAlign128],
                        dataCopyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(eventReluMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(eventReluMte2ToV_);
        }

        if (kRunInfo.isAlign64) {
            ProcessVec6Vf<T, INPUT_T, true, false>(reluGradUb, dwUb_, softmaxTensor[runInfo.s2CurSize],
                                                   softmaxReduceSumUb_, weightInUb, reduceSumUb_[runInfo.s2CurSize],
                                                   reluUb_, kRunInfo.s2RealBaseSize, nIndexSize, VEC_P_BASESIZE);
        } else {
            ProcessVec6Vf<T, INPUT_T, false, false>(reluGradUb, dwUb_, softmaxTensor[runInfo.s2CurSize],
                                                    softmaxReduceSumUb_, weightInUb, reduceSumUb_[runInfo.s2CurSize],
                                                    reluUb_, kRunInfo.s2RealBaseSize, nIndexSize, VEC_P_BASESIZE);
        }
        SetFlag<HardEvent::V_MTE2>(eventReluVToMte2_);
        if constexpr (ResidentPipeline) {
            if (!kRunInfo.isS2end) {
                WaitFlag<HardEvent::V_MTE2>(eventReluVToMte2_);
                if (kRunInfo.kTaskId + 2 == runInfo.s2LoopTimes) {
                    CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_RELU_TO_V6_FLAG);
                }
                dataCopyParams.blockCount = nIndexSize;
                dataCopyParams.blockLen = VEC_P_BASESIZE * sizeof(T);
                dataCopyParams.srcStride = (constInfo.kSizeAlign128 - VEC_P_BASESIZE) * sizeof(T);
                dataCopyParams.dstStride = 0;
                DataCopyPad(reluUb_,
                            reluGm[runInfo.s2CurSize + VEC_P_BASESIZE +
                                   firstNIndexSize * constInfo.subBlockIdx * constInfo.kSizeAlign128],
                            dataCopyParams, padParams);
                SetFlag<HardEvent::MTE2_V>(eventReluMte2ToV_);
            }
        }
        Mutex::Unlock<PIPE_V>(this->reluGradMutexId_);
        Mutex::Lock<PIPE_MTE3>(this->reluGradMutexId_);
        LocalTensor<INPUT_T> mm3AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        DataCopy(mm3AL1Tensor[constInfo.subBlockIdx * firstNIndexSize * BLOCK_SINGLE_LEN], reluGradUb,
                 {static_cast<uint16_t>(VEC_P_BASESIZE / BLOCK_SINGLE_LEN), static_cast<uint16_t>(nIndexSize), 0,
                  static_cast<uint16_t>(gAlign16 - nIndexSize)});
        Mutex::Unlock<PIPE_MTE3>(this->reluGradMutexId_);
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
        if (kRunInfo.isS2end) {
            LocalTensor<T> dwOutTensor = dwOutQue.template AllocTensor<T>();
            CastDupVf<T, T>(dwOutTensor, dwUb_, nIndexSize);
            this->dwOutQue.template EnQue(dwOutTensor);
            this->dwOutQue.template DeQue<T>();
            DataCopyExtParams dataCopyParams1;
            dataCopyParams1.blockCount = 1;
            dataCopyParams1.blockLen = nIndexSize * sizeof(T);
            dataCopyParams1.srcStride = 0;
            dataCopyParams1.dstStride = 0;
            DataCopyPad(dWeightGm[runInfo.weightOffset], dwOutTensor, dataCopyParams1);
            this->dwOutQue.template FreeTensor(dwOutTensor);
        }
        if (kRunInfo.isS2end) {
            Mutex::Unlock<PIPE_V>(this->weightMutexId_);
        }
    }
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void CopyScatterRowsRaw(__gm__ T* dst, __ubuf__ T* src, uint16_t blockCount,
                                                               int64_t dstStrideBytes, uint8_t cacheMode)
{
    ::asc_copy_ub2gm_align(dst, src, static_cast<uint32_t>(blockCount), 512,
                           static_cast<asc_store_l2_cache_mode>(cacheMode), static_cast<uint64_t>(dstStrideBytes) + 512,
                           512);
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void CopyScatterPairHeadStepRaw(__gm__ T* dst, __ubuf__ T* src,
                                                                       uint64_t dstHeadStepBytes, uint8_t cacheMode)
{
    ::asc_copy_ub2gm_align(dst, src, uint32_t(2), 512, static_cast<asc_store_l2_cache_mode>(cacheMode),
                           dstHeadStepBytes, 512);
}

TEMPLATES_DEF_NO_DEFAULT
template <bool rawAddressSafe, bool bytePairCached>
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector7PairSorted(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm3ResBuf, SLIGradRunInfo& runInfo,
    SLIGradKRunInfo& kRunInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
    if (kRunInfo.s2Idx == 0) {
        WaitFlag<HardEvent::V_S>(this->eventPairIndexVToS_);
    }
    LocalTensor<T> scatterAddTmpUb = bmm3ResBuf.template GetTensor<T>();
    __ubuf__ T* scatterUb = reinterpret_cast<__ubuf__ T*>(scatterAddTmpUb.GetPhyAddr());
    __ubuf__ uint32_t* indexUb = reinterpret_cast<__ubuf__ uint32_t*>(this->pairIndexUb_.GetPhyAddr()) +
                                 kRunInfo.s2Idx * VEC_P_BASESIZE + constInfo.subBlockIdx * 64;
    __gm__ T* scatterGm = this->pairScatterGm_;
    const uint8_t cacheMode = this->pairScatterCacheMode_;
    const int64_t keyRowBase = Shape::AccumS2Idx(runInfo);
    __gm__ uint8_t* scatterBatchBytes = reinterpret_cast<__gm__ uint8_t*>(scatterGm + keyRowBase * 128);
    DataCopyExtParams copyParams;
    copyParams.blockCount = 2;
    copyParams.blockLen = 512;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    SetAtomicAdd<T>();
    for (uint32_t row = 0; row < 64; row += 8) {
        const uint32_t low0 = indexUb[row];
        const uint32_t delta0 = indexUb[row + 1];
        const uint32_t low1 = indexUb[row + 2];
        const uint32_t delta1 = indexUb[row + 3];
        const uint32_t low2 = indexUb[row + 4];
        const uint32_t delta2 = indexUb[row + 5];
        const uint32_t low3 = indexUb[row + 6];
        const uint32_t delta3 = indexUb[row + 7];
        const int64_t dst0 =
            bytePairCached ? keyRowBase * 128 + static_cast<int64_t>(low0 >> 2) : (keyRowBase + low0) * 128;
        const int64_t dst1 =
            bytePairCached ? keyRowBase * 128 + static_cast<int64_t>(low1 >> 2) : (keyRowBase + low1) * 128;
        const int64_t dst2 =
            bytePairCached ? keyRowBase * 128 + static_cast<int64_t>(low2 >> 2) : (keyRowBase + low2) * 128;
        const int64_t dst3 =
            bytePairCached ? keyRowBase * 128 + static_cast<int64_t>(low3 >> 2) : (keyRowBase + low3) * 128;
        const int64_t gap0 =
            bytePairCached ? static_cast<int64_t>(delta0) - 512 : (static_cast<int64_t>(delta0) - 1) * 512;
        const int64_t gap1 =
            bytePairCached ? static_cast<int64_t>(delta1) - 512 : (static_cast<int64_t>(delta1) - 1) * 512;
        const int64_t gap2 =
            bytePairCached ? static_cast<int64_t>(delta2) - 512 : (static_cast<int64_t>(delta2) - 1) * 512;
        const int64_t gap3 =
            bytePairCached ? static_cast<int64_t>(delta3) - 512 : (static_cast<int64_t>(delta3) - 1) * 512;
        if constexpr (rawAddressSafe) {
            __gm__ T* dstPtr0 =
                bytePairCached ? reinterpret_cast<__gm__ T*>(scatterBatchBytes + low0) : scatterGm + dst0;
            __gm__ T* dstPtr1 =
                bytePairCached ? reinterpret_cast<__gm__ T*>(scatterBatchBytes + low1) : scatterGm + dst1;
            __gm__ T* dstPtr2 =
                bytePairCached ? reinterpret_cast<__gm__ T*>(scatterBatchBytes + low2) : scatterGm + dst2;
            __gm__ T* dstPtr3 =
                bytePairCached ? reinterpret_cast<__gm__ T*>(scatterBatchBytes + low3) : scatterGm + dst3;
            if (delta0 == 0) {
                CopyScatterRowsRaw(dstPtr0, scatterUb + row * 128, 1, 0, cacheMode);
                CopyScatterRowsRaw(dstPtr0, scatterUb + (row + 1) * 128, 1, 0, cacheMode);
            } else {
                CopyScatterPairHeadStepRaw(
                    dstPtr0, scatterUb + row * 128,
                    bytePairCached ? static_cast<uint64_t>(delta0) : static_cast<uint64_t>(delta0) * 512U, cacheMode);
            }
            if (delta1 == 0) {
                CopyScatterRowsRaw(dstPtr1, scatterUb + (row + 2) * 128, 1, 0, cacheMode);
                CopyScatterRowsRaw(dstPtr1, scatterUb + (row + 3) * 128, 1, 0, cacheMode);
            } else {
                CopyScatterPairHeadStepRaw(
                    dstPtr1, scatterUb + (row + 2) * 128,
                    bytePairCached ? static_cast<uint64_t>(delta1) : static_cast<uint64_t>(delta1) * 512U, cacheMode);
            }
            if (delta2 == 0) {
                CopyScatterRowsRaw(dstPtr2, scatterUb + (row + 4) * 128, 1, 0, cacheMode);
                CopyScatterRowsRaw(dstPtr2, scatterUb + (row + 5) * 128, 1, 0, cacheMode);
            } else {
                CopyScatterPairHeadStepRaw(
                    dstPtr2, scatterUb + (row + 4) * 128,
                    bytePairCached ? static_cast<uint64_t>(delta2) : static_cast<uint64_t>(delta2) * 512U, cacheMode);
            }
            if (delta3 == 0) {
                CopyScatterRowsRaw(dstPtr3, scatterUb + (row + 6) * 128, 1, 0, cacheMode);
                CopyScatterRowsRaw(dstPtr3, scatterUb + (row + 7) * 128, 1, 0, cacheMode);
            } else {
                CopyScatterPairHeadStepRaw(
                    dstPtr3, scatterUb + (row + 6) * 128,
                    bytePairCached ? static_cast<uint64_t>(delta3) : static_cast<uint64_t>(delta3) * 512U, cacheMode);
            }
        } else {
            if (delta0 == 0) {
                copyParams.blockCount = 1;
                copyParams.dstStride = 0;
                DataCopyPad(scatterAddResGm[dst0], scatterAddTmpUb[row * 128], copyParams);
                DataCopyPad(scatterAddResGm[dst0], scatterAddTmpUb[(row + 1) * 128], copyParams);
                copyParams.blockCount = 2;
            } else {
                copyParams.dstStride = gap0;
                DataCopyPad(scatterAddResGm[dst0], scatterAddTmpUb[row * 128], copyParams);
            }
            if (delta1 == 0) {
                copyParams.blockCount = 1;
                copyParams.dstStride = 0;
                DataCopyPad(scatterAddResGm[dst1], scatterAddTmpUb[(row + 2) * 128], copyParams);
                DataCopyPad(scatterAddResGm[dst1], scatterAddTmpUb[(row + 3) * 128], copyParams);
                copyParams.blockCount = 2;
            } else {
                copyParams.dstStride = gap1;
                DataCopyPad(scatterAddResGm[dst1], scatterAddTmpUb[(row + 2) * 128], copyParams);
            }
            if (delta2 == 0) {
                copyParams.blockCount = 1;
                copyParams.dstStride = 0;
                DataCopyPad(scatterAddResGm[dst2], scatterAddTmpUb[(row + 4) * 128], copyParams);
                DataCopyPad(scatterAddResGm[dst2], scatterAddTmpUb[(row + 5) * 128], copyParams);
                copyParams.blockCount = 2;
            } else {
                copyParams.dstStride = gap2;
                DataCopyPad(scatterAddResGm[dst2], scatterAddTmpUb[(row + 4) * 128], copyParams);
            }
            if (delta3 == 0) {
                copyParams.blockCount = 1;
                copyParams.dstStride = 0;
                DataCopyPad(scatterAddResGm[dst3], scatterAddTmpUb[(row + 6) * 128], copyParams);
                DataCopyPad(scatterAddResGm[dst3], scatterAddTmpUb[(row + 7) * 128], copyParams);
                copyParams.blockCount = 2;
            } else {
                copyParams.dstStride = gap3;
                DataCopyPad(scatterAddResGm[dst3], scatterAddTmpUb[(row + 6) * 128], copyParams);
            }
        }
    }
    SetAtomicNone();
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector7(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm3ResBuf, SLIGradRunInfo& runInfo,
    SLIGradKRunInfo& kRunInfo)
{
    if constexpr (ResidentPipeline) {
        if (Shape::PairSortedTile(kRunInfo)) {
            if (this->pairByteCacheEn_) {
                if (this->pairRawAddressSafe_) {
                    ProcessVector7PairSorted<true, true>(bmm3ResBuf, runInfo, kRunInfo);
                } else {
                    ProcessVector7PairSorted<false, true>(bmm3ResBuf, runInfo, kRunInfo);
                }
            } else {
                if (this->pairRawAddressSafe_) {
                    ProcessVector7PairSorted<true, false>(bmm3ResBuf, runInfo, kRunInfo);
                } else {
                    ProcessVector7PairSorted<false, false>(bmm3ResBuf, runInfo, kRunInfo);
                }
            }
            return;
        }
    }
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
    int32_t vRealKSize;
    int64_t coreKOffset;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
        coreKOffset = 0;
    } else {
        vRealKSize = kRunInfo.s2RealBaseSize - AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
        coreKOffset = AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
    }
    if (vRealKSize <= 0) {
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
        return;
    }

    LocalTensor<T> scatterAddTmpUb = bmm3ResBuf.template GetTensor<T>();
    SetAtomicAdd<T>();
    constexpr int32_t topKSplitSize = 2;
    const int32_t topKLoopTimes = CeilDiv(vRealKSize, topKSplitSize);

    constexpr uint32_t dSize = 128;
    const int64_t dSizeBytes = static_cast<int64_t>(dSize) * sizeof(T);
    const int64_t absDiffOverflowAt =
        (static_cast<int64_t>(INT32_MAX) + dSizeBytes - 1) / dSizeBytes + SPARSE_BLOCK_SIZE;
    const int64_t keyRowBase = Shape::AccumS2Idx(runInfo);
    const int64_t topkBase = runInfo.topkGmBaseOffset + static_cast<int64_t>(kRunInfo.s2Idx) * runInfo.s2BaseSize;
    const bool hasOddTail = ((vRealKSize & 1) != 0);
    const int32_t pairLoops = vRealKSize / 2;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = static_cast<uint32_t>(dSizeBytes);
    dataCopyParams.srcStride = 0;
    dataCopyParams.blockCount = 1;
    dataCopyParams.dstStride = 0;

    const int64_t ubRowStep = static_cast<int64_t>(topKSplitSize) * dSize;
    int64_t s2GmOffset = coreKOffset;
    int64_t ub1Offset = 0;
    const int64_t s2GmOffsetEnd = coreKOffset + static_cast<int64_t>(pairLoops) * topKSplitSize;
    const __gm__ int32_t* topkPtr = topKGm.GetPhyAddr();
    __gm__ T* scatterGmPtr = const_cast<__gm__ T*>(scatterAddResGm.GetPhyAddr());
    __ubuf__ T* scatterUbPtr = reinterpret_cast<__ubuf__ T*>(scatterAddTmpUb.GetPhyAddr());
    const uint8_t scatterCacheMode = AscendC::ExtractCacheMode(scatterAddResGm);
    const uint64_t scatterGmAddress = reinterpret_cast<uint64_t>(scatterGmPtr);
    const bool rawScatterAddressSafe = static_cast<uint64_t>(constInfo.totalCost) <=
                                       (AscendC::L2_CACHE_OFFSET_MASK - scatterGmAddress) / (128 * sizeof(T));
    const int64_t unrollEnd =
        rawScatterAddressSafe ? coreKOffset + static_cast<int64_t>(pairLoops / 4) * 8 : coreKOffset;
    for (; s2GmOffset < unrollEnd; s2GmOffset += 4 * topKSplitSize, ub1Offset += 4 * ubRowStep) {
        const int64_t realS2Idx1 = topkPtr[topkBase + s2GmOffset];
        const int64_t realS2Idx2 = topkPtr[topkBase + s2GmOffset + 1];
        const int64_t realS2Idx3 = topkPtr[topkBase + s2GmOffset + 2];
        const int64_t realS2Idx4 = topkPtr[topkBase + s2GmOffset + 3];
        const int64_t realS2Idx5 = topkPtr[topkBase + s2GmOffset + 4];
        const int64_t realS2Idx6 = topkPtr[topkBase + s2GmOffset + 5];
        const int64_t realS2Idx7 = topkPtr[topkBase + s2GmOffset + 6];
        const int64_t realS2Idx8 = topkPtr[topkBase + s2GmOffset + 7];
        const int64_t keyOffset1 = keyRowBase + realS2Idx1;
        const int64_t keyOffset2 = keyRowBase + realS2Idx2;
        const int64_t keyOffset3 = keyRowBase + realS2Idx3;
        const int64_t keyOffset4 = keyRowBase + realS2Idx4;
        const int64_t keyOffset5 = keyRowBase + realS2Idx5;
        const int64_t keyOffset6 = keyRowBase + realS2Idx6;
        const int64_t keyOffset7 = keyRowBase + realS2Idx7;
        const int64_t keyOffset8 = keyRowBase + realS2Idx8;
        const int64_t ub2Offset = ub1Offset + dSize;
        const int64_t ub3Offset = ub1Offset + ubRowStep;
        const int64_t ub4Offset = ub3Offset + dSize;
        const int64_t ub5Offset = ub1Offset + 4 * dSize;
        const int64_t ub6Offset = ub1Offset + 5 * dSize;
        const int64_t ub7Offset = ub1Offset + 6 * dSize;
        const int64_t ub8Offset = ub1Offset + 7 * dSize;

        __gm__ T* dst1 = scatterGmPtr + keyOffset1 * dSize;
        __gm__ T* dst2 = scatterGmPtr + keyOffset2 * dSize;
        __gm__ T* dst3 = scatterGmPtr + keyOffset3 * dSize;
        __gm__ T* dst4 = scatterGmPtr + keyOffset4 * dSize;
        __gm__ T* dst5 = scatterGmPtr + keyOffset5 * dSize;
        __gm__ T* dst6 = scatterGmPtr + keyOffset6 * dSize;
        __gm__ T* dst7 = scatterGmPtr + keyOffset7 * dSize;
        __gm__ T* dst8 = scatterGmPtr + keyOffset8 * dSize;
        __ubuf__ T* src1 = scatterUbPtr + ub1Offset;
        __ubuf__ T* src2 = scatterUbPtr + ub2Offset;
        __ubuf__ T* src3 = scatterUbPtr + ub3Offset;
        __ubuf__ T* src4 = scatterUbPtr + ub4Offset;
        __ubuf__ T* src5 = scatterUbPtr + ub5Offset;
        __ubuf__ T* src6 = scatterUbPtr + ub6Offset;
        __ubuf__ T* src7 = scatterUbPtr + ub7Offset;
        __ubuf__ T* src8 = scatterUbPtr + ub8Offset;

        if (realS2Idx1 > realS2Idx2) {
            CopyScatterRowsRaw(dst1, src1, 1, 0, scatterCacheMode);
            CopyScatterRowsRaw(dst2, src2, 1, 0, scatterCacheMode);
        } else {
            const int64_t dstStrideBytes = (keyOffset2 - keyOffset1 - 1) * dSizeBytes;
            CopyScatterRowsRaw(dst1, src1, 2, dstStrideBytes, scatterCacheMode);
        }
        if (realS2Idx3 > realS2Idx4) {
            CopyScatterRowsRaw(dst3, src3, 1, 0, scatterCacheMode);
            CopyScatterRowsRaw(dst4, src4, 1, 0, scatterCacheMode);
        } else {
            const int64_t dstStrideBytes = (keyOffset4 - keyOffset3 - 1) * dSizeBytes;
            CopyScatterRowsRaw(dst3, src3, 2, dstStrideBytes, scatterCacheMode);
        }
        if (realS2Idx5 > realS2Idx6) {
            CopyScatterRowsRaw(dst5, src5, 1, 0, scatterCacheMode);
            CopyScatterRowsRaw(dst6, src6, 1, 0, scatterCacheMode);
        } else {
            const int64_t dstStrideBytes = (keyOffset6 - keyOffset5 - 1) * dSizeBytes;
            CopyScatterRowsRaw(dst5, src5, 2, dstStrideBytes, scatterCacheMode);
        }
        if (realS2Idx7 > realS2Idx8) {
            CopyScatterRowsRaw(dst7, src7, 1, 0, scatterCacheMode);
            CopyScatterRowsRaw(dst8, src8, 1, 0, scatterCacheMode);
        } else {
            const int64_t dstStrideBytes = (keyOffset8 - keyOffset7 - 1) * dSizeBytes;
            CopyScatterRowsRaw(dst7, src7, 2, dstStrideBytes, scatterCacheMode);
        }
    }
    for (; s2GmOffset < s2GmOffsetEnd; s2GmOffset += topKSplitSize, ub1Offset += ubRowStep) {
        const int64_t realS2Idx1 = topkPtr[topkBase + s2GmOffset];
        const int64_t realS2Idx2 = static_cast<int64_t>(topkPtr[topkBase + s2GmOffset + 1]);

        const int64_t keyOffset1 = keyRowBase + realS2Idx1;
        const int64_t keyOffset2 = keyRowBase + realS2Idx2;
        const int64_t ub2Offset = ub1Offset + dSize;

        if (realS2Idx1 > realS2Idx2) {
            DataCopy(scatterAddResGm[keyOffset1 * dSize], scatterAddTmpUb[ub1Offset], dSize);
            DataCopy(scatterAddResGm[keyOffset2 * dSize], scatterAddTmpUb[ub2Offset], dSize);
        } else {
            const int64_t absDiff = keyOffset2 - keyOffset1;
            if (unlikely(absDiff >= absDiffOverflowAt)) {
                DataCopy(scatterAddResGm[keyOffset1 * dSize], scatterAddTmpUb[ub1Offset], dSize);
                DataCopy(scatterAddResGm[keyOffset2 * dSize], scatterAddTmpUb[ub2Offset], dSize);
            } else {
                dataCopyParams.blockCount = 2;
                dataCopyParams.dstStride = static_cast<uint32_t>((absDiff - SPARSE_BLOCK_SIZE) * dSizeBytes);
                DataCopyPad(scatterAddResGm[keyOffset1 * dSize], scatterAddTmpUb[ub1Offset], dataCopyParams);
            }
        }
    }
    if (hasOddTail && s2GmOffset == s2GmOffsetEnd) {
        const int64_t realS2IdxTail = topKGm.GetValue(topkBase + s2GmOffset);
        if (realS2IdxTail >= 0) {
            DataCopy(scatterAddResGm[(keyRowBase + realS2IdxTail) * dSize], scatterAddTmpUb[ub1Offset], dSize);
        }
    }
    SetAtomicNone();
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector7Deter(
    SLIGradRunInfo& runInfo, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& tmpBuf)
{
    CrossCoreWaitFlag<MODE_NUM_2, PIPE_MTE2>(SYNC_C3_TO_V7_FLAG[runInfo.taskIdMod2]);
    CrossCoreSetFlag<0, PIPE_MTE2>(SYNC_C3_TO_V7_DETER_MTE2_FLAG[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<0, PIPE_MTE2>(SYNC_C3_TO_V7_DETER_MTE2_FLAG[runInfo.taskIdMod2]);

    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
    event_t eventIdPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    event_t eventIdPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    event_t eventIdArr[MODE_NUM_2] = {eventIdPing, eventIdPong};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdPing);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdPong);
    int32_t scatterAddPingpong = 0;
    SetAtomicAdd<T>();
    int32_t coreNum = GetBlockNum();
    int32_t memoKRealSize = -1;
    int32_t memoPerCoreKSize = 0;
    int32_t memoUsedCoreNum = 0;
    constexpr bool kFastBS1 = (Layout_Q == SLILayout::BSND) && !HasSequsedQ && !HasSequsedK;
    int64_t fastBIdx = 0;
    int64_t fastS1Idx = 0;
    int64_t fastAccumS1 = 0;
    int64_t fastAccumS2 = 0;
    bool fastInit = false;
    int64_t preBIdx = -1;
    int64_t bIdx;
    int64_t s1Idx;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int32_t actualSeqLensQ = 0;
    int32_t actualSeqLensK = 0;
    GlobalTensor<T> scatterAddGm[2] = {scatterAddResGm, scatterAddResGmPong};
    int64_t usedBS1Index = -1;
    int64_t bS1Index = -1;
    int64_t usedBS1IndexEnd = constInfo.totalNum - 1;

    int64_t satBIdx = -1;
    bool satCoreIdle = true;
    int32_t satVRealKSize = 0;
    int64_t satVCoreKOffset = 0;
    if constexpr (kFastBS1) {
        const int32_t kFix = static_cast<int32_t>(constInfo.kSize);
        const int32_t perCoreFix = CeilDiv(kFix, coreNum);
        const int32_t usedCoreFix = CeilDiv(kFix, perCoreFix);
        if (constInfo.aicIdx < usedCoreFix) {
            const int32_t tailFix = kFix - (usedCoreFix - 1) * perCoreFix;
            const int32_t curFix = constInfo.aicIdx == usedCoreFix - 1 ? tailFix : perCoreFix;
            int32_t v0Fix = CeilDiv(curFix, MODE_NUM_2);
            v0Fix = Min(AlignTo(v0Fix, static_cast<int32_t>(MODE_NUM_2)), curFix);
            const int64_t coreKOffFix = static_cast<int64_t>(constInfo.aicIdx) * perCoreFix;
            if (constInfo.subBlockIdx == 0) {
                satVRealKSize = v0Fix;
                satVCoreKOffset = coreKOffFix;
            } else {
                satVRealKSize = curFix - v0Fix;
                satVCoreKOffset = coreKOffFix + v0Fix;
            }
        }
        satCoreIdle = (satVRealKSize <= 0);
    }

    int32_t idx = 0;
    while (idx < coreNum) {
        for (; idx < coreNum; idx++) {
            usedBS1Index = runInfo.taskId * coreNum + idx;
            // 重新获取b和S1的值
            if constexpr (kFastBS1) {
                if (usedBS1Index >= constInfo.totalNum) {
                    continue;
                }
                if (unlikely(!fastInit)) {
                    fastInit = true;
                    fastBIdx = usedBS1Index / constInfo.s1Size;
                    fastS1Idx = usedBS1Index - fastBIdx * constInfo.s1Size;
                    fastAccumS1 = fastBIdx * constInfo.s1Size;
                    fastAccumS2 = fastBIdx * constInfo.s2Size;
                }
                bIdx = fastBIdx;
                s1Idx = fastS1Idx;
                bS1Index = usedBS1Index;
                accumS1Len = fastAccumS1;
                accumS2Len = fastAccumS2;
                actualSeqLensQ = static_cast<int32_t>(constInfo.s1Size);
                actualSeqLensK = static_cast<int32_t>(constInfo.s2Size);
                if (++fastS1Idx == static_cast<int64_t>(constInfo.s1Size)) {
                    fastS1Idx = 0;
                    ++fastBIdx;
                    fastAccumS1 += constInfo.s1Size;
                    fastAccumS2 += constInfo.s2Size;
                }
            } else {
                if (!GetBS1Index(usedBS1Index, bIdx, s1Idx, bS1Index, accumS1Len, accumS2Len, actualSeqLensQ,
                                 actualSeqLensK, runInfo.taskId)) {
                    continue;
                }
            }

            int64_t topkBaseCur;
            int64_t keyRowBaseCur;
            int32_t vRealKSize;
            int64_t vCoreKOffset;
            bool useSatGeom = false;
            if constexpr (kFastBS1) {
                if (bIdx == satBIdx) {
                    useSatGeom = true;
                } else {
                    if (GetS2SparseLen(static_cast<int32_t>(bIdx), static_cast<int32_t>(s1Idx), actualSeqLensQ,
                                       actualSeqLensK) >= static_cast<int32_t>(constInfo.kSize)) {
                        satBIdx = bIdx;
                        useSatGeom = true;
                    }
                }
            }
            if (useSatGeom) {
                topkBaseCur = (accumS1Len + s1Idx) * constInfo.kSize;
                keyRowBaseCur = accumS2Len;
                if (satCoreIdle) {
                    if ((idx & 1) == 0) {
                        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
                    }
                    if ((idx & 1) == 1 || usedBS1Index == usedBS1IndexEnd) {
                        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
                    }
                    continue;
                }
                vRealKSize = satVRealKSize;
                vCoreKOffset = satVCoreKOffset;
            } else {
                GetRunInfo(runInfo.taskId, bIdx, s1Idx, runInfo, accumS1Len, accumS2Len, actualSeqLensQ,
                           actualSeqLensK);
                topkBaseCur = runInfo.topkGmBaseOffset;
                keyRowBaseCur = runInfo.accumS2Idx;
                int32_t v0RealKSize;
                int32_t v1RealKSize;
                int32_t perCoreKSize;
                int32_t tailCoreKSize;
                int32_t curCoreKSize;
                int64_t coreKOffset;
                if (runInfo.kRealSize != memoKRealSize) {
                    memoKRealSize = runInfo.kRealSize;
                    memoPerCoreKSize = CeilDiv(runInfo.kRealSize, coreNum);
                    memoUsedCoreNum = CeilDiv(runInfo.kRealSize, memoPerCoreKSize);
                }
                perCoreKSize = memoPerCoreKSize;
                int32_t usedCoreNum = memoUsedCoreNum;
                if (constInfo.aicIdx >= usedCoreNum) {
                    if (idx % MODE_NUM_2 == 0) {
                        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
                    }
                    if (idx % MODE_NUM_2 == 1 || usedBS1Index == usedBS1IndexEnd) {
                        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
                    }
                    continue;
                }
                tailCoreKSize = runInfo.kRealSize - (usedCoreNum - 1) * perCoreKSize;
                curCoreKSize = constInfo.aicIdx == usedCoreNum - 1 ? tailCoreKSize : perCoreKSize;
                v0RealKSize = CeilDiv(curCoreKSize, MODE_NUM_2);
                v0RealKSize = Min(AlignTo(v0RealKSize, static_cast<int32_t>(MODE_NUM_2)), curCoreKSize);
                v1RealKSize = curCoreKSize - v0RealKSize;
                coreKOffset = constInfo.aicIdx * perCoreKSize;
                if (constInfo.subBlockIdx == 0) {
                    vRealKSize = v0RealKSize;
                    vCoreKOffset = coreKOffset;
                } else {
                    vRealKSize = v1RealKSize;
                    vCoreKOffset = coreKOffset + v0RealKSize;
                }
                if (vRealKSize <= 0) {
                    if (idx % MODE_NUM_2 == 0) {
                        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
                    }
                    if (idx % MODE_NUM_2 == 1 || usedBS1Index == usedBS1IndexEnd) {
                        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
                    }
                    continue;
                }
            }

            int64_t mm3GmBaseOffset = idx * constInfo.kSize * constInfo.dSizeQueryIndex * MODE_NUM_2;
            GlobalTensor<T> srcGm =
                mm3DeterResGm[mm3GmBaseOffset +
                              (runInfo.taskIdMod2 * constInfo.kSize + vCoreKOffset) * constInfo.dSizeQueryIndex];
            DeterScatterAdd(vRealKSize, srcGm, tmpBuf, vCoreKOffset, keyRowBaseCur, topkBaseCur, idx,
                            scatterAddGm[idx % MODE_NUM_2], eventIdArr, scatterAddPingpong);
            if (idx % MODE_NUM_2 == 1 || usedBS1Index == usedBS1IndexEnd) {
                CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
            }
            if (useSatGeom && !satCoreIdle) {
                ++idx;
                break;
            }
        }
        if constexpr (kFastBS1) {
            const int32_t vRealKSize = satVRealKSize;
            const int64_t vCoreKOffset = satVCoreKOffset;
            for (; idx < coreNum; idx++) {
                usedBS1Index = runInfo.taskId * coreNum + idx;
                if (usedBS1Index >= constInfo.totalNum) {
                    continue;
                }
                if (unlikely(fastBIdx != satBIdx)) {
                    break;
                }
                const int64_t topkBaseCur = (fastAccumS1 + fastS1Idx) * constInfo.kSize;
                const int64_t keyRowBaseCur = fastAccumS2;
                if (++fastS1Idx == static_cast<int64_t>(constInfo.s1Size)) {
                    fastS1Idx = 0;
                    ++fastBIdx;
                    fastAccumS1 += constInfo.s1Size;
                    fastAccumS2 += constInfo.s2Size;
                }
                int64_t mm3GmBaseOffset = idx * constInfo.kSize * constInfo.dSizeQueryIndex * MODE_NUM_2;
                GlobalTensor<T> srcGm =
                    mm3DeterResGm[mm3GmBaseOffset +
                                  (runInfo.taskIdMod2 * constInfo.kSize + vCoreKOffset) * constInfo.dSizeQueryIndex];
                DeterScatterAdd(vRealKSize, srcGm, tmpBuf, vCoreKOffset, keyRowBaseCur, topkBaseCur, idx,
                                scatterAddGm[idx % MODE_NUM_2], eventIdArr, scatterAddPingpong);
                if (idx % MODE_NUM_2 == 1 || usedBS1Index == usedBS1IndexEnd) {
                    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
                }
            }
        }
    }
    SetAtomicNone();
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdPing);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdPong);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(eventIdPing);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(eventIdPong);
    CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::DeterScatterAdd(
    int32_t vRealKSize, GlobalTensor<T>& srcGm, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& tmpBuf,
    int64_t coreKOffset, int64_t keyRowBaseIn, int64_t topkBaseIn, int32_t idx, GlobalTensor<T>& scatterAddGm,
    event_t (&eventIdArr)[MODE_NUM_2], int32_t& scatterAddPingpong)
{
    constexpr int32_t topKSplitSize = MODE_NUM_2;
    LocalTensor<T> scatterAddTmpUb = tmpBuf.template GetTensor<T>();
    const int32_t kSplitSize = this->deterKSplitSize;
    int32_t kLoopTimes;
    int32_t kTailSize;
    if (likely(vRealKSize <= kSplitSize)) {
        kLoopTimes = 1;
        kTailSize = vRealKSize;
    } else {
        kLoopTimes = CeilDiv(vRealKSize, kSplitSize);
        kTailSize = vRealKSize - (kLoopTimes - 1) * kSplitSize;
    }
    int32_t kProcessSize = kSplitSize;
    if (idx % MODE_NUM_2 == 0) {
        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
    }

    const uint32_t dSize = constInfo.dSizeQueryIndex;
    constexpr int64_t dSizeBytes = static_cast<int64_t>(128) * sizeof(T);
    const int64_t absDiffOverflowAt =
        (static_cast<int64_t>(INT32_MAX) + dSizeBytes - 1) / dSizeBytes + SPARSE_BLOCK_SIZE;
    const int64_t keyRowBase = keyRowBaseIn;
    const int64_t topkBase = topkBaseIn;
    const int64_t ubRowStep = static_cast<int64_t>(topKSplitSize) * dSize;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = static_cast<uint32_t>(dSizeBytes);
    dataCopyParams.srcStride = 0;
    dataCopyParams.blockCount = 1;
    dataCopyParams.dstStride = 0;

    for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
        if (kLoopIdx == kLoopTimes - 1) {
            kProcessSize = kTailSize;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        const int64_t ubChunkBase = static_cast<int64_t>(scatterAddPingpong) * kSplitSize * dSize;
        LocalTensor<T> scatterUb = scatterAddTmpUb[ubChunkBase];
        DataCopy(scatterUb, srcGm[static_cast<int64_t>(kLoopIdx) * kSplitSize * dSize], kProcessSize * dSize);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);

        const bool hasOddTail = ((kProcessSize & 1) != 0);
        const int64_t chunkKBase = coreKOffset + static_cast<int64_t>(kLoopIdx) * kSplitSize;
        const int64_t s2GmOffsetEnd = chunkKBase + static_cast<int64_t>(kProcessSize / 2) * topKSplitSize;
        int64_t s2GmOffset = chunkKBase;
        int64_t ub1Offset = 0;
        for (; s2GmOffset < s2GmOffsetEnd; s2GmOffset += topKSplitSize, ub1Offset += ubRowStep) {
            const int64_t realS2Idx1 = topKGm.GetValue(topkBase + s2GmOffset);
            const int64_t realS2Idx2 = topKGm.GetValue(topkBase + s2GmOffset + 1);
            const int64_t keyOffset1 = keyRowBase + realS2Idx1;
            const int64_t keyOffset2 = keyRowBase + realS2Idx2;
            const int64_t ub2Offset = ub1Offset + dSize;

            if (likely(realS2Idx2 >= 0)) {
                if (realS2Idx1 > realS2Idx2) {
                    DataCopy(scatterAddGm[keyOffset1 * dSize], scatterUb[ub1Offset], 128);
                    DataCopy(scatterAddGm[keyOffset2 * dSize], scatterUb[ub2Offset], 128);
                } else {
                    const int64_t absDiff = keyOffset2 - keyOffset1;
                    if (unlikely(absDiff >= absDiffOverflowAt)) {
                        DataCopy(scatterAddGm[keyOffset1 * dSize], scatterUb[ub1Offset], dSize);
                        DataCopy(scatterAddGm[keyOffset2 * dSize], scatterUb[ub2Offset], dSize);
                    } else {
                        dataCopyParams.blockCount = 2;
                        dataCopyParams.dstStride = static_cast<uint32_t>((absDiff - SPARSE_BLOCK_SIZE) * dSizeBytes);
                        DataCopyPad(scatterAddGm[keyOffset1 * dSize], scatterUb[ub1Offset], dataCopyParams);
                    }
                }
            } else {
                if (likely(realS2Idx1 >= 0)) {
                    DataCopy(scatterAddGm[keyOffset1 * dSize], scatterUb[ub1Offset], dSize);
                }
                break;
            }
        }
        if (hasOddTail && s2GmOffset == s2GmOffsetEnd) {
            const int64_t realS2IdxTail = topKGm.GetValue(topkBase + s2GmOffset);
            if (realS2IdxTail >= 0) {
                DataCopy(scatterAddGm[(keyRowBase + realS2IdxTail) * dSize], scatterUb[ub1Offset], dSize);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddPingpong = 1 - scatterAddPingpong;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ReInitBuffers(TPipe* pipe)
{
    WaitFlag<HardEvent::V_MTE2>(this->eventReluVToMte2_);
    if constexpr (ResidentPipeline) {
        pipe->template ReleaseEventID<HardEvent::S_MTE2>(this->eventPairIndexSToMte2_);
        pipe->template ReleaseEventID<HardEvent::V_S>(this->eventPairIndexVToS_);
    }
    if constexpr (!Deterministic) {
        ReleaseMutexID(this->mutexIds_[0]);
        ReleaseMutexID(this->mutexIds_[1]);
        ReleaseMutexID(this->weightMutexId_);
        ReleaseMutexID(this->reduceSumMutexId_);
        ReleaseMutexID(this->reluGradMutexId_);
    }
    pipe->Reset();
    pipe->InitBuffer(this->scatterAddInQue[0], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->scatterAddInQue[1], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->scatterAddInQue2[0], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->scatterAddInQue2[1], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->dKeyOutQue[0], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->dKeyOutQue[1], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_16K);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector8()
{
    int64_t avgCost = CeilDiv(constInfo.totalCost, GetBlockNum() * GetTaskRation());
    int32_t t2Start = Min(constInfo.aivIdx * avgCost, constInfo.totalCost);
    int32_t t2End = Min(t2Start + avgCost, constInfo.totalCost);

    int32_t t2ProcessSize = UB_ROW_SIZE;
    int32_t t2TailSize = ((t2End - t2Start) % UB_ROW_SIZE == 0) ? UB_ROW_SIZE : ((t2End - t2Start) % UB_ROW_SIZE);
    int32_t pingPongIdx = 0;

    LocalTensor<T> copyInUb;
    LocalTensor<T> copyInUbPong;
    LocalTensor<OUT_T> copyOutUb;
    for (int32_t t2Idx = t2Start; t2Idx < t2End; t2Idx += UB_ROW_SIZE) {
        if (t2Idx + UB_ROW_SIZE >= t2End) {
            t2ProcessSize = t2TailSize;
        }
        if constexpr (Deterministic) {
            copyInUb = this->scatterAddInQue[pingPongIdx].template AllocTensor<T>();
            copyInUbPong = this->scatterAddInQue2[pingPongIdx].template AllocTensor<T>();
            DataCopy(copyInUb, scatterAddResGm[t2Idx * constInfo.dSizeQueryIndex],
                     t2ProcessSize * constInfo.dSizeQueryIndex);
            DataCopy(copyInUbPong, scatterAddResGmPong[t2Idx * constInfo.dSizeQueryIndex],
                     t2ProcessSize * constInfo.dSizeQueryIndex);
            this->scatterAddInQue[pingPongIdx].template EnQue(copyInUb);
            this->scatterAddInQue[pingPongIdx].template DeQue<T>();
            this->scatterAddInQue2[pingPongIdx].template EnQue(copyInUbPong);
            this->scatterAddInQue2[pingPongIdx].template DeQue<T>();
            Add(copyInUb, copyInUb, copyInUbPong, t2ProcessSize * constInfo.dSizeQueryIndex);
            this->scatterAddInQue2[pingPongIdx].template FreeTensor(copyInUbPong);
        } else {
            copyInUb = this->scatterAddInQue[pingPongIdx].template AllocTensor<T>();
            DataCopy(copyInUb, scatterAddResGm[t2Idx * constInfo.dSizeQueryIndex],
                     t2ProcessSize * constInfo.dSizeQueryIndex);
            this->scatterAddInQue[pingPongIdx].template EnQue(copyInUb);
            this->scatterAddInQue[pingPongIdx].template DeQue<T>();
        }
        copyOutUb = this->dKeyOutQue[pingPongIdx].template AllocTensor<OUT_T>();
        Cast(copyOutUb, copyInUb, RoundMode::CAST_ROUND, t2ProcessSize * constInfo.dSizeQueryIndex);
        this->dKeyOutQue[pingPongIdx].template EnQue(copyOutUb);
        this->dKeyOutQue[pingPongIdx].template DeQue<OUT_T>();
        DataCopy(dKeyIndexGm[t2Idx * constInfo.dSizeQueryIndex], copyOutUb, t2ProcessSize * constInfo.dSizeQueryIndex);
        this->scatterAddInQue[pingPongIdx].template FreeTensor(copyInUb);
        this->dKeyOutQue[pingPongIdx].template FreeTensor(copyOutUb);
        pingPongIdx = 1 - pingPongIdx;
    }
}

TEMPLATES_DEF
class SligBlockVecDummy {
public:
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR attnSoftmaxL1,
                                            GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR softmaxOut,
                                            GlobalTensor<INPUT_T>& gatherSYRes, GlobalTensor<T>& relu,
                                            GlobalTensor<T>& reduceSumRes, GlobalTensor<T>& scatterAddRes,
                                            GlobalTensor<T>& scatterAddResPong, GlobalTensor<T>& bmm3Res,
                                            GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK,
                                            GM_ADDR cmpResidualK) {};
    __aicore__ inline void InitBuffers(TPipe* pipe) {};
    __aicore__ inline void ProcessVector0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
                                          SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo) {};
    __aicore__ inline void CopyInWeight(SLIGradRunInfo& runInfo) {};
    __aicore__ inline void CopyInAttnNorm(SLIGradRunInfo& runInfo) {};
    __aicore__ inline void ProcessVector1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf,
                                          SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo) {};
    __aicore__ inline void ProcessVector2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf,
                                          SLIGradRunInfo& runInfo) {};
    __aicore__ inline void PrefetchFirstRelu(SLIGradRunInfo& runInfo) {};
    __aicore__ inline void ProcessVector6(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
                                          SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo) {};
    __aicore__ inline void ProcessVector7(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm3ResBuf,
                                          SLIGradRunInfo& runInfo, SLIGradKRunInfo& kRunInfo) {};
    __aicore__ inline void DeterScatterAdd(int32_t vRealKSize, GlobalTensor<T>& srcGm,
                                           Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& tmpBuf,
                                           int64_t coreKOffset, int64_t keyRowBaseIn, int64_t topkBaseIn, int32_t idx,
                                           GlobalTensor<T>& scatterAddGm, event_t (&eventIdArr)[MODE_NUM_2],
                                           int32_t& scatterAddPingpong) {};
    __aicore__ inline void ProcessVector7Deter(SLIGradRunInfo& runInfo,
                                               Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& tmpBuf) {};
    __aicore__ inline void ReInitBuffers(TPipe* pipe) {};
    __aicore__ inline void ProcessVector8() {};
};
} // namespace Slig
#endif
