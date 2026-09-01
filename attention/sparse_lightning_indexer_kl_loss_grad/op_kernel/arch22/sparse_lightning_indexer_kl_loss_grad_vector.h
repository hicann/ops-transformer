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
 * \file sparse_lightning_indexer_kl_loss_grad_vector.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_kl_loss_grad_common.h"
#include "sparse_lightning_indexer_kl_loss_grad_metadata_arch22.h"
#include "sparse_lightning_indexer_kl_loss_grad_tiling.h"

using namespace AscendC;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SLIT>
class SLIKLLossVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SLIT::inputQT;
    using KV_T = typename SLIT::inputKT;
    using OUT_T = typename SLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using MM5_OUT_T = T;

    static constexpr uint32_t topKSize = static_cast<uint32_t>(SLIT::topKRange);
    static constexpr bool isTopkLess2k = (topKSize <= SLIKLLossGradConstInfo::BUFFER_SIZE_BYTE_2K);
    static constexpr uint32_t PSY_SYNC_BLOCK_FLOAT_NUM = 32 / sizeof(T);
    static constexpr uint32_t PSY_SYNC_SUM_P_OFFSET = topKSize * 2;
    static constexpr uint32_t PSY_SYNC_STRIDE = PSY_SYNC_SUM_P_OFFSET + PSY_SYNC_BLOCK_FLOAT_NUM;
    static constexpr int64_t P_OWNER_SY_ROW_BIAS = 1;
    static constexpr int64_t SY_LB_MIN_G_SIZE = 64;
    static constexpr int64_t SY_LB_MIN_SERIAL_WORK = 4096;
    static constexpr SLILayout LAYOUT_T = SLIT::inputQLayout;
    static constexpr SLILayout KV_LAYOUT_T = SLIT::inputKLayout;
    static constexpr bool hasSequsedQ = SLIT::hasSequsedQ;
    static constexpr bool hasSequsedK = SLIT::hasSequsedK;
    static constexpr UBAllocPolicy<isTopkLess2k> ubAllocPolicy;
    static constexpr bool deterministic = SLIT::deterministic;
    static constexpr bool privateScatter = SLIT::privateScatter;

    using scatterAddGmType = typename std::conditional<deterministic, GlobalTensor<T>, int8_t>::type;

    __aicore__ inline SLIKLLossVectorService(){};
    __aicore__ inline void InitParams(const SLIKLLossGradConstInfo &vecConstInfo,
                                      const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tilingData,
                                      GlobalTensor<int32_t> metadataGm, GlobalTensor<int32_t> cmpResidualKeyGm,
                                      bool hasMetadata);
    __aicore__ inline void InitSeqUsedGM(GlobalTensor<int32_t> &seqUsedQ, GlobalTensor<int32_t> &seqUsedK);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    // =============== vector 0 functions ==============
    __aicore__ inline void InitVector0GM(GlobalTensor<KV_T> &key, GlobalTensor<KV_T> &keyRope,
                                         GlobalTensor<KV_T> &keyIndex, GlobalTensor<int32_t> &topK,
                                         GlobalTensor<int32_t> &actualSeqLengthsQ,
                                         GlobalTensor<int32_t> &actualSeqLengthsKV, GlobalTensor<KV_T> &gatherPRes,
                                         GlobalTensor<KV_T> &gatherSYRes);
    __aicore__ inline void ProcessVector0(SLIKLLossGradRunInfo &runInfo);

    // =============== vector 1 functions ==============
    __aicore__ inline void InitVector1GM(const GlobalTensor<T> &attnSoftmaxL1Norm,
                                         const GlobalTensor<MM12_OUT_T> &bmm2Res, const GlobalTensor<T> &weight,
                                         const GlobalTensor<T> psySync, GlobalTensor<T> &softmaxOut,
                                         GlobalTensor<T> &dWeight, GlobalTensor<T> &reluGm,
                                         GlobalTensor<KV_T> &reluGradRes,
                                         GlobalTensor<int32_t> &actualSeqLengthsQueryGm,
                                         GlobalTensor<int32_t> &actualSeqLengthsKeyGm);
    __aicore__ inline void ProcessVector1(SLIKLLossGradRunInfo &runInfo);

    // =============== vector 2 functions ==============
    __aicore__ inline void ProcessVector2(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline void InitVector2GM(const GlobalTensor<MM5_OUT_T> &bmm5Res, const GlobalTensor<int32_t> &topK,
                                         GlobalTensor<T> &scatterAddRes,
                                         scatterAddGmType scatterAddBanks[SLI_DETER_SCATTER_BANK_NUM]);
    __aicore__ inline void ProcessDeterVector2(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline void ProcessPrivateScatterVector2(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline void SetCachedValidTotalSize(int64_t validTotalSize);

private:
    // =============== vector 0 functions ==============
    template <event_t IdStart, bool gatherRope>
    __aicore__ inline void MergeKv(const SLIKLLossGradRunInfo &runInfo, GlobalTensor<KV_T> kvMergeGm_,
                                   GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor);
    template <bool gatherRope>
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const SLIKLLossGradRunInfo &runInfo,
                                          GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor);
    __aicore__ inline int64_t GetKeyRopeGmOffset(int64_t realS2Idx, const SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const SLIKLLossGradRunInfo &runInfo, int64_t s2IdLimit,
                                             int32_t dValue);
    template <bool gatherRope>
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const SLIKLLossGradRunInfo &runInfo,
                                    GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor);
    template <event_t IdStart, bool gatherRope>
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset,
                                             int64_t s2IdxOffset, int64_t mergeMte3Idx,
                                             const SLIKLLossGradRunInfo &runInfo, GlobalTensor<KV_T> kvMergeGm_,
                                             GlobalTensor<KV_T> srcTensor, GlobalTensor<KV_T> srcRopeTensor);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t s2IdxOffset, int64_t &realS2Idx,
                                        const SLIKLLossGradRunInfo &runInfo);

    // =============== vector 1 functions ==============
    // Legacy Q@K/P path is intentionally isolated in this op. P is supplied by attn_softmax_l1_norm.
#if 0
    __aicore__ inline void VectorP(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline void VectorPWithOutSoftMaxSum(SLIKLLossGradRunInfo &runInfo);
#endif
    __aicore__ inline void CopyPFromInput(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline void PreloadWeight(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline bool ShouldEnableSyLoadBalance(const SLIKLLossGradRunInfo &runInfo) const;
    __aicore__ inline void VectorSy(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline void VectorSyLoadBalance(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline void FinalizeVectorSy(SLIKLLossGradRunInfo &runInfo);
    __aicore__ inline void ReLUGrad(LocalTensor<KV_T> &reluGradOutTensor, LocalTensor<T> &subResTensor,
                                    LocalTensor<T> &maskUb, int32_t kRealSizeAlign);
    template <uint32_t range>
    __aicore__ inline void VectorDwDqDk(SLIKLLossGradRunInfo &runInfo, int32_t kLoopIdx);
    __aicore__ inline void VectorDwDqDkMoreThan2k(SLIKLLossGradRunInfo &runInfo, int32_t kLoopIdx);
    __aicore__ inline void VectorDwDqDkLess2k(SLIKLLossGradRunInfo &runInfo);
    // =============== vector 2 functions ==============
    __aicore__ inline void ScatterAddCopyOutSingle(const LocalTensor<MM5_OUT_T> &srcUb, int64_t keyBNBOffset,
                                                   GlobalTensor<T> &scatterAddGm);
    __aicore__ inline int32_t GetActualSeqLens(int32_t bIdx, int32_t defaultLens,
                                               GlobalTensor<int32_t> &actualSeqLensGm, SLILayout layout,
                                               int64_t &accumLen);
    __aicore__ inline int32_t GetUsedSeqLens(int32_t bIdx, int32_t defaultLens, GlobalTensor<int32_t> &seqUsedGm);
    __aicore__ inline int32_t GetCmpResidualK(int32_t bIdx);
    __aicore__ inline int64_t GetPreCompressS2Len(int32_t bIdx, int32_t actualSeqLensK);
    __aicore__ inline int32_t GetS2SparseLen(int32_t bIdx, int32_t s1Idx, int32_t actualSeqLensQ,
                                             int32_t actualSeqLensK, int64_t cmpRatio, SLISparseMode sparseMode);
    __aicore__ inline int64_t GetInvalidS1Size(int64_t bIdx, int64_t actualSeqLensQ, int64_t actualSeqLensK);
    __aicore__ inline void ResetValidIdxMap();
    __aicore__ inline bool LoadValidBatchAt(int64_t curB);
    __aicore__ inline bool MapValidIdxToBS1(int64_t validIdx, int64_t &bIdx, int64_t &s1Idx, int64_t &accumS1Len,
                                            int64_t &accumS2Len, int32_t &actualSeqLensQ, int32_t &actualSeqLensK);
    __aicore__ inline void GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx, SLIKLLossGradRunInfo &runInfo,
                                      int64_t accumS1Len, int64_t accumS2Len, int32_t actualSeqLensQ,
                                      int32_t actualSeqLensK);
    __aicore__ inline void Vector2ScatterAdd(int32_t vRealKSize, GlobalTensor<MM5_OUT_T> &srcGm, int64_t coreKOffset,
                                             SLIKLLossGradRunInfo &runInfo, GlobalTensor<T> &scatterAddGm);
    __aicore__ inline int64_t GetMetadataTotalSize();
    __aicore__ inline int64_t GetMetadataBS1Index(uint32_t coreIdx);

    // =============== vector common variable ==============
    SLIKLLossGradConstInfo constInfo;
    const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tilingData;
    bool hasMetadata = false;
    GlobalTensor<int32_t> metadataGm;
    GlobalTensor<int32_t> cmpResidualKeyGm;

    int64_t cachedValidTotalSize_ = -1;
    int64_t mapCurB_ = 0;
    int64_t mapPrefixBeforeCurB_ = 0;
    int64_t mapValidCntCurB_ = 0;
    int64_t mapInvalidS1CurB_ = 0;
    int64_t mapAccumS1CurB_ = 0;
    int64_t mapAccumS2CurB_ = 0;
    int32_t mapSeqQCurB_ = 0;
    int32_t mapSeqKCurB_ = 0;
    bool mapBatchReady_ = false;

    // global tensor
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyIndexGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<MM12_OUT_T> bmm1ResGm;
    GlobalTensor<T> attnSoftmaxL1NormGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<MM12_OUT_T> bmm2ResGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> psySyncGm, psySyncOtherGm;
    GlobalTensor<T> tempGm;
    GlobalTensor<T> softmaxOutGm;
    GlobalTensor<T> dWeightGm;
    GlobalTensor<T> reluGm;
    GlobalTensor<KV_T> reluGradResGm;

    GlobalTensor<MM5_OUT_T> bmm5ResGm;

    GlobalTensor<int32_t> actualSeqLengthsQueryGm;
    GlobalTensor<int32_t> actualSeqLengthsKeyGm;
    GlobalTensor<int32_t> seqUsedQueryGm;
    GlobalTensor<int32_t> seqUsedKeyGm;

    // local tensor
    TBuf<> mm1Tbuf;
    TBuf<> mm2TBuf; // 复用 -> mm4 scatterAdd reluGrad
    TBuf<> sharedTBuf;
    TBuf<> resPSYTBuf;
    TBuf<> reduceSumDwTBuf;
    TBuf<> v1TmpTBuf;

    TBuf<> weightTBuf;
    TBuf<> weightInTBuf;
    TBuf<> dwTBuf;
    TBuf<> maskBuf;
    TBuf<> scatterAddBuf;

    // =============== vector0 variable ==============
    GlobalTensor<KV_T> gatherPResGm;
    GlobalTensor<KV_T> gatherSYResGm;

    TBuf<> gatherTbuf;

    LocalTensor<KV_T> gatherKeyUb;
    LocalTensor<KV_T> gatherKeyIndexUb;
    static constexpr int32_t blockBytes = 32;

    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<KV_T> ropeMergUb_;

    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> srcTensor;
    GlobalTensor<KV_T> srcRopeTensor;

    GatherParams gatherParams;

    static constexpr int64_t UB_ROW_SIZE = 8;
    // =============== vector1 variable ==============
    LocalTensor<T> weightUb;
    LocalTensor<T> weightInUb;

    LocalTensor<T> reluResUb;
    LocalTensor<T> reduceSumYResTmpBuffer;
    LocalTensor<T> reduceSumYResUb;
    LocalTensor<T> reduceSumPResUb;
    LocalTensor<uint8_t> reduceSumTmpBuffer;
    LocalTensor<uint8_t> softmaxTmpBuffer;
    LocalTensor<T> reduceSumDwResUb;

    // =============== vector2 variable ==============
    GlobalTensor<T> scatterAddResGm;
    scatterAddGmType scatterAddResGmBanks[SLI_DETER_SCATTER_BANK_NUM];
    LocalTensor<T> scatterAddUb;
    int32_t scatterAddPingpong = 0;

    // dwdqdk计算
    LocalTensor<T> v1TmpUb;
    LocalTensor<T> pingBuf;
    LocalTensor<T> pongBuf;
    LocalTensor<KV_T> reluGradpingBuf;
    LocalTensor<KV_T> reluGradpongBuf;
    LocalTensor<T> weightTensor;
    LocalTensor<uint8_t> tmpUb;
    LocalTensor<T> dwOutTensor;

    // =============== Event_id variable ==============
    static constexpr event_t EVENT_ID0 = (event_t)0;
    static constexpr event_t EVENT_ID1 = (event_t)1;
    static constexpr event_t EVENT_ID2 = (event_t)2;
    static constexpr event_t EVENT_ID3 = (event_t)3;
    static constexpr event_t EVENT_ID4 = (event_t)4;
    static constexpr event_t EVENT_ID5 = (event_t)5;
    static constexpr event_t EVENT_ID6 = (event_t)6;
    static constexpr event_t EVENT_ID7 = (event_t)7;

    // MTE2_V
    event_t eventIdMte2ToV4SY;
    event_t eventIdMte2ToVSyPingPong[2];
    event_t eventIdMte2ToVInnerVecP;
    event_t eventIdMte2ToVInnerPreW;
    event_t eventIdMte2ToVInnerDwDqDk;
    // MTE2_S
    event_t eventIdMte2ToSPsum;
    // S_V
    event_t eventIdSToVPsum;
    // V_MTE2
    event_t eventIdVToMte24SY;
    event_t eventIdVToMte2SyPingPong[2];
    event_t eventIdVToMte2P[2];
    event_t eventIdVToMte2Weight[2];
    event_t eventIdvToMte2DwDqDkPingPong[2];
    // MTE3_V
    event_t eventIdMte3ToVTmp;
    event_t eventIdMte3ToVDwDqDkPingPong[2];
    // V_MTE3
    event_t eventVToMte3InnerVecP;
    event_t eventVToMte3InnerSy;
    event_t eventIdVToMte3DwDqDk;
    // MTE3_MTE2
    event_t eventIdVec0Inner2;
    event_t eventIdVec0Inner3;
    event_t eventIdmte3ToMte2DwDqDkPingPong[2];
    // event_t eventIdVToV4SY; // 加一行性能会变好
    event_t eventIdScatterAdd;
    event_t eventIdScatterAddPong;
    // MTE2_MTE3
    event_t eventIdMergeRes;
};

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitParams(
    const SLIKLLossGradConstInfo &vecConstInfo,
    const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tilingData, GlobalTensor<int32_t> metadataGm,
    GlobalTensor<int32_t> cmpResidualKeyGm, bool hasMetadata)
{
    this->constInfo = vecConstInfo;
    this->tilingData = tilingData;
    this->metadataGm = metadataGm;
    this->cmpResidualKeyGm = cmpResidualKeyGm;
    this->hasMetadata = hasMetadata;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitSeqUsedGM(GlobalTensor<int32_t> &seqUsedQ,
                                                                   GlobalTensor<int32_t> &seqUsedK)
{
    this->seqUsedQueryGm = seqUsedQ;
    this->seqUsedKeyGm = seqUsedK;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector0GM(
    GlobalTensor<KV_T> &key, GlobalTensor<KV_T> &keyRope, GlobalTensor<KV_T> &keyIndex, GlobalTensor<int32_t> &topK,
    GlobalTensor<int32_t> &actualSeqLengthsQ, GlobalTensor<int32_t> &actualSeqLengthsKV, GlobalTensor<KV_T> &gatherPRes,
    GlobalTensor<KV_T> &gatherSYRes)
{
    this->keyGm = key;
    this->keyRopeGm = keyRope;
    this->keyIndexGm = keyIndex;
    this->topKGm = topK;

    this->gatherPResGm = gatherPRes;
    this->gatherSYResGm = gatherSYRes;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector1GM(
    const GlobalTensor<T> &attnSoftmaxL1Norm, const GlobalTensor<MM12_OUT_T> &bmm2Res, const GlobalTensor<T> &weight,
    const GlobalTensor<T> psySync, GlobalTensor<T> &softmaxOut, GlobalTensor<T> &dWeight, GlobalTensor<T> &reluGm,
    GlobalTensor<KV_T> &reluGradRes, GlobalTensor<int32_t> &actualSeqLengthsQueryGm,
    GlobalTensor<int32_t> &actualSeqLengthsKeyGm)
{
    this->attnSoftmaxL1NormGm = attnSoftmaxL1Norm;
    this->bmm2ResGm = bmm2Res;
    this->weightGm = weight;
    this->psySyncGm = psySync[constInfo.subBlockIdx * PSY_SYNC_STRIDE];
    this->psySyncOtherGm = psySync[!constInfo.subBlockIdx * PSY_SYNC_STRIDE];

    this->softmaxOutGm = softmaxOut;
    this->dWeightGm = dWeight;
    this->reluGm = reluGm;
    this->reluGradResGm = reluGradRes;
    this->actualSeqLengthsQueryGm = actualSeqLengthsQueryGm;
    this->actualSeqLengthsKeyGm = actualSeqLengthsKeyGm;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitVector2GM(
    const GlobalTensor<MM5_OUT_T> &bmm5Res, const GlobalTensor<int32_t> &topK, GlobalTensor<T> &scatterAddRes,
    scatterAddGmType scatterAddBanks[SLI_DETER_SCATTER_BANK_NUM])
{
    this->bmm5ResGm = bmm5Res;
    this->topKGm = topK;
    this->scatterAddResGm = scatterAddRes;
    if constexpr (deterministic) {
        for (int32_t bank = 0; bank < SLI_DETER_SCATTER_BANK_NUM; ++bank) {
            this->scatterAddResGmBanks[bank] = scatterAddBanks[bank];
        }
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::InitBuffers(TPipe *pipe)
{
    // 可选输入的buffer空间，保持和stage1处理的size一致
    pipe->InitBuffer(this->gatherTbuf, ubAllocPolicy.gatherUbSize);
    pipe->InitBuffer(this->mm1Tbuf, ubAllocPolicy.mm1UbSize);
    pipe->InitBuffer(this->mm2TBuf, ubAllocPolicy.mm2UbSize);
    pipe->InitBuffer(this->sharedTBuf, ubAllocPolicy.sharedUbSize);
    pipe->InitBuffer(this->resPSYTBuf, ubAllocPolicy.resPSYUbSize);
    pipe->InitBuffer(this->reduceSumDwTBuf, ubAllocPolicy.reduceSumDwUbSize);
    pipe->InitBuffer(this->v1TmpTBuf, ubAllocPolicy.v1TmpUbSize);
    pipe->InitBuffer(this->weightTBuf, ubAllocPolicy.weightUbSize);
    pipe->InitBuffer(this->weightInTBuf, ubAllocPolicy.weightInUbSize);
    pipe->InitBuffer(this->dwTBuf, ubAllocPolicy.dwUbSize);
    pipe->InitBuffer(this->maskBuf, ubAllocPolicy.maskUbSize);
    pipe->InitBuffer(this->scatterAddBuf, ubAllocPolicy.scatterAddUbSize);

    gatherKeyUb = gatherTbuf.Get<KV_T>();
    gatherKeyIndexUb = gatherKeyUb[SLIKLLossGradConstInfo::BUFFER_SIZE_BYTE_9K * 2];

    weightUb = weightTBuf.Get<T>();
    weightInUb = weightInTBuf.Get<T>();
    reluResUb = mm2TBuf.Get<T>();
    reduceSumTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(8 * 1024, 8 * 1024);
    reduceSumYResTmpBuffer = sharedTBuf.GetWithOffset<T>(topKSize, 2 * 8 * 1024);
    reduceSumYResUb = resPSYTBuf.Get<T>();
    reduceSumPResUb = resPSYTBuf.template Get<T>();
    reduceSumDwResUb = reduceSumDwTBuf.Get<T>();
    softmaxTmpBuffer = sharedTBuf.Get<uint8_t>();
    scatterAddUb = scatterAddBuf.Get<T>();

    v1TmpUb = v1TmpTBuf.template Get<T>();
    pingBuf = mm1Tbuf.template Get<T>();
    pongBuf = mm2TBuf.template Get<T>();
    reluGradpingBuf = mm1Tbuf.template Get<KV_T>();
    reluGradpongBuf = mm2TBuf.template Get<KV_T>();
    weightTensor = weightTBuf.template Get<T>();
    tmpUb = sharedTBuf.template Get<uint8_t>();
    dwOutTensor = this->dwTBuf.template Get<T>();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::AllocEventID()
{
    // MTE2_To_V
    eventIdMte2ToV4SY = EVENT_ID0;
    eventIdMte2ToVSyPingPong[0] = EVENT_ID0;
    eventIdMte2ToVSyPingPong[1] = EVENT_ID1;
    eventIdMte2ToVInnerVecP = EVENT_ID1;
    eventIdMte2ToVInnerPreW = EVENT_ID2;
    eventIdMte2ToVInnerDwDqDk = EVENT_ID3;
    // MTE2_To_S
    eventIdMte2ToSPsum = EVENT_ID0;
    // S_To_V
    eventIdSToVPsum = EVENT_ID0;
    // V_To_MTE2
    eventIdVToMte2Weight[0] = EVENT_ID0;
    eventIdVToMte2Weight[1] = EVENT_ID1;
    eventIdVToMte2P[0] = EVENT_ID3;
    eventIdVToMte2P[1] = EVENT_ID4;
    eventIdVToMte24SY = EVENT_ID5;
    eventIdVToMte2SyPingPong[0] = EVENT_ID5;
    eventIdVToMte2SyPingPong[1] = EVENT_ID2;
    eventIdvToMte2DwDqDkPingPong[0] = EVENT_ID6;
    eventIdvToMte2DwDqDkPingPong[1] = EVENT_ID7;
    // MTE3_To_V
    eventIdMte3ToVTmp = EVENT_ID0;
    eventIdMte3ToVDwDqDkPingPong[0] = EVENT_ID1;
    eventIdMte3ToVDwDqDkPingPong[1] = EVENT_ID2;
    // V_To_MTE3
    eventVToMte3InnerVecP = EVENT_ID0;
    eventVToMte3InnerSy = EVENT_ID1;
    eventIdVToMte3DwDqDk = EVENT_ID2;
    // MTE3_MTE2
    eventIdVec0Inner2 = EVENT_ID2;
    eventIdVec0Inner3 = EVENT_ID3;
    eventIdmte3ToMte2DwDqDkPingPong[0] = EVENT_ID4;
    eventIdmte3ToMte2DwDqDkPingPong[1] = EVENT_ID5;
    eventIdScatterAdd = EVENT_ID6;
    eventIdScatterAddPong = EVENT_ID7;
    // MTE2_MTE3
    eventIdMergeRes = EVENT_ID0;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAdd);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAddPong);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[1]);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::FreeEventID()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAdd);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdScatterAddPong);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[1]);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector0(SLIKLLossGradRunInfo &runInfo)
{
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner2);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner3);

    gatherParams.dValue = constInfo.dSizeQueryIndex;
    gatherParams.dRopeValue = 0;
    gatherParams.gatherResGmEleSize = constInfo.gatherKeyIndexSize;
    this->kvMergUb_ = gatherKeyIndexUb;

    for (int32_t i = 0; i < runInfo.s2LoopTimes; ++i) {
        runInfo.s2Idx = i;
        gatherParams.s2ProcessSize = (i >= runInfo.s2LoopTimes - 1) ? runInfo.s2TailSize : constInfo.s2BaseSize;
        MergeKv<EVENT_ID2, false>(runInfo, gatherSYResGm, keyIndexGm, keyRopeGm);
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V0_TO_C1_SY_FLAG[i & 1]);
    }

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner2);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdVec0Inner3);
}

template <typename SLIT>
template <bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size,
                                                                    int64_t mergeMte3Idx, int64_t realS2Idx,
                                                                    int64_t keyBNBOffset, int64_t s2IdLimit,
                                                                    const SLIKLLossGradRunInfo &runInfo,
                                                                    GlobalTensor<KV_T> srcTensor,
                                                                    GlobalTensor<KV_T> srcRopeTensor)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams(1, validS2Count * gatherParams.dValue * sizeof(KV_T), 0, 0, 0);
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(
        kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue + (mte2Size - mte3Size) * gatherParams.dValue],
        srcTensor[keyBNBOffset * gatherParams.dValue], intriParams, padParams);

    if constexpr (gatherRope) {
        intriParams.blockLen = validS2Count * gatherParams.dRopeValue * sizeof(KV_T);
        DataCopyPad(ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue +
                                (mte2Size - mte3Size) * gatherParams.dRopeValue],
                    srcRopeTensor[keyBNBOffset * gatherParams.dRopeValue], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetKeyRopeGmOffset(int64_t realS2Idx,
                                                                           const SLIKLLossGradRunInfo &runInfo)
{
    int64_t tensorBRopeOffset = runInfo.accumS2Idx * gatherParams.dRopeValue;
    int64_t realKeyRopeGmOffset = (tensorBRopeOffset + realS2Idx * gatherParams.dRopeValue) / gatherParams.dRopeValue;
    return realKeyRopeGmOffset;
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetKeyGmOffset(int64_t realS2Idx,
                                                                       const SLIKLLossGradRunInfo &runInfo,
                                                                       int64_t s2IdLimit, int32_t dValue)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t tensorBOffset = runInfo.accumS2Idx * dValue;
    int64_t realKeyGmOffset = (tensorBOffset + realS2Idx * dValue) / dValue;
    return realKeyGmOffset;
}

template <typename SLIT>
template <bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                              int64_t realS2Idx1, int64_t realS2Idx2,
                                                              const SLIKLLossGradRunInfo &runInfo,
                                                              GlobalTensor<KV_T> srcTensor,
                                                              GlobalTensor<KV_T> srcRopeTensor)
{
    int64_t s2IdLimit = runInfo.s2SparseLen;

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, gatherParams.dValue);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, gatherParams.dValue);
    bool keyOffset1Pass = (keyOffset1 >= 0);
    bool keyOffset2Pass = (keyOffset2 >= 0);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                    constInfo.sparseBlockSize) *
                   gatherParams.dValue * sizeof(KV_T);
    if constexpr (gatherRope) {
        int64_t keyRopeOffset1 = GetKeyRopeGmOffset(realS2Idx1, runInfo);
        int64_t keyRopeOffset2 = GetKeyRopeGmOffset(realS2Idx2, runInfo);
        keyRopeSrcStride =
            ((keyRopeOffset1 > keyRopeOffset2 ? (keyRopeOffset1 - keyRopeOffset2) : (keyRopeOffset2 - keyRopeOffset1)) -
             constInfo.sparseBlockSize) *
            gatherParams.dRopeValue * sizeof(KV_T);
    }

    bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);
    bool strideInvalid =
        (keySrcStride >= INT32_MAX) || (keySrcStride < 0) || (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0);
    bool copyOutOfRange =
        (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit || realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);

    if (key1LessThankey2 || strideInvalid || copyOutOfRange) {
        // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
        CopyInSingleKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, runInfo,
                                   srcTensor, srcRopeTensor);
        CopyInSingleKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, runInfo,
                                   srcTensor, srcRopeTensor);
    } else {
        DataCopyExtParams intriParams(keyOffset1Pass + keyOffset2Pass,
                                      constInfo.sparseBlockSize * gatherParams.dValue * sizeof(KV_T), keySrcStride, 0,
                                      0);
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1Pass ? keyOffset1 : keyOffset2;
        if (keyOffset2Pass && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(
            kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue + (mte2Size - mte3Size) * gatherParams.dValue],
            srcTensor[startGmOffset * gatherParams.dValue], intriParams, padParams);

        if constexpr (gatherRope) {
            intriParams.blockLen = constInfo.sparseBlockSize * gatherParams.dRopeValue * sizeof(KV_T);
            intriParams.srcStride = keyRopeSrcStride;
            DataCopyPad(ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue +
                                    (mte2Size - mte3Size) * gatherParams.dRopeValue],
                        srcRopeTensor[startGmOffset * gatherParams.dRopeValue], intriParams, padParams);
        }

        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename SLIT>
template <event_t IdStart, bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::CopyOutMrgeResult(
    int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset, int64_t s2IdxOffset, int64_t mergeMte3Idx,
    const SLIKLLossGradRunInfo &runInfo, GlobalTensor<KV_T> kvMergeGm_, GlobalTensor<KV_T> srcTensor,
    GlobalTensor<KV_T> srcRopeTensor)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMergeRes);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMergeRes);

    DataCopyExtParams dataCopyParams(mte2Size - mte3Size, gatherParams.dValue * sizeof(KV_T), 0, 0, 0);

    int64_t gmStartOffset = runInfo.taskIdMod2 * gatherParams.gatherResGmEleSize +
                            s2IdxOffset * (gatherParams.dValue + gatherParams.dRopeValue);

    DataCopyPad(kvMergeGm_[gmStartOffset + (s2GmStartOffset + mte3Size) * gatherParams.dValue],
                kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dValue], dataCopyParams);

    if constexpr (gatherRope) {
        dataCopyParams.blockLen = gatherParams.dRopeValue * sizeof(KV_T);
        DataCopyPad(kvMergeGm_[gmStartOffset + constInfo.s2BaseSize * gatherParams.dValue +
                               (s2GmStartOffset + mte3Size) * gatherParams.dRopeValue],
                    ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * gatherParams.dRopeValue], dataCopyParams);
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::GetRealS2Idx(int64_t s2GmOffset, int64_t s2IdxOffset,
                                                                  int64_t &realS2Idx,
                                                                  const SLIKLLossGradRunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + s2IdxOffset) / constInfo.sparseBlockSize;
    realS2Idx =
        topKGm.GetValue(runInfo.topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
        static_cast<int64_t>((s2GmOffset + s2IdxOffset) % constInfo.sparseBlockSize);
}

template <typename SLIT>
template <event_t IdStart, bool gatherRope>
__aicore__ inline void SLIKLLossVectorService<SLIT>::MergeKv(const SLIKLLossGradRunInfo &runInfo,
                                                             GlobalTensor<KV_T> kvMergeGm_,
                                                             GlobalTensor<KV_T> srcTensor,
                                                             GlobalTensor<KV_T> srcRopeTensor)
{
    int64_t s2Pair = CeilDiv(gatherParams.s2ProcessSize, 2L * constInfo.sparseBlockSize);

    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    bool subBlockZero = (GetSubBlockIdx() == 0);
    int64_t s2GmStartOffset = subBlockZero ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = subBlockZero ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize : gatherParams.s2ProcessSize;
    if (s2GmLimit > gatherParams.s2ProcessSize) {
        s2GmLimit = gatherParams.s2ProcessSize;
    }
    int64_t s2IdxOffset = static_cast<int64_t>(runInfo.s2Idx) * constInfo.s2BaseSize;
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxOffset, s2IdxArray0, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult<IdStart, gatherRope>(mte2Size, mte3Size, s2GmStartOffset, s2IdxOffset, mergeMte3Idx,
                                                   runInfo, kvMergeGm_, srcTensor, srcRopeTensor);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxOffset, s2IdxArray1, runInfo);
        CopyInKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo, srcTensor,
                             srcRopeTensor);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > UB_ROW_SIZE) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult<IdStart, gatherRope>(mte2Size, mte3Size, s2GmStartOffset, s2IdxOffset, mergeMte3Idx,
                                                   runInfo, kvMergeGm_, srcTensor, srcRopeTensor);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector1(SLIKLLossGradRunInfo &runInfo)
{
    CrossCoreWaitFlag(SYNC_C1_TO_V1_P_FLAG[runInfo.taskIdMod2]);
    CrossCoreWaitFlag(SYNC_C1_TO_V1_SY_FLAG[runInfo.taskIdMod2]);
    // 空任务只配对 V1→C2，避免 kRealSize==0 的 DataCopy/Sum
    if (runInfo.kLoopTimes <= 0 || runInfo.kRealSize <= 0) {
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_TO_C2_DW_FLAG[runInfo.taskIdMod2]);
        return;
    }
    PreloadWeight(runInfo);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    bool useSyLoadBalance = ShouldEnableSyLoadBalance(runInfo);
    if (!useSyLoadBalance) {
        if (runInfo.calcP) {
            CopyPFromInput(runInfo);
        } else {
            VectorSy(runInfo);
        }
    } else {
        if (runInfo.calcP) {
            CopyPFromInput(runInfo);
        }
        VectorSyLoadBalance(runInfo);
        AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
        AscendC::CrossCoreWaitFlag<0x1, PIPE_MTE2>(0x8);
        if (!runInfo.calcP) {
            FinalizeVectorSy(runInfo);
        }
    }
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag<0x1, PIPE_MTE2>(0x8);
    for (int32_t kLoopIdx = 0; kLoopIdx < runInfo.kLoopTimes; ++kLoopIdx) {
        VectorDwDqDk<topKSize>(runInfo, kLoopIdx); // Sub + Mul + Mul + ReluGrad
        if (kLoopIdx >= runInfo.kLoopTimes - 1) {
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_TO_C2_DW_FLAG[runInfo.taskIdMod2]);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ScatterAddCopyOutSingle(const LocalTensor<MM5_OUT_T> &srcUb,
                                                                             int64_t keyBNBOffset,
                                                                             GlobalTensor<T> &scatterAddGm)
{
    if (keyBNBOffset < 0) {
        return;
    }
    LocalTensor<T> srcTmpUb = srcUb.template ReinterpretCast<T>();
    DataCopy(scatterAddGm[keyBNBOffset * constInfo.dSizeQueryIndex], srcTmpUb, constInfo.dSizeQueryIndex);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::CopyPFromInput(SLIKLLossGradRunInfo &runInfo)
{
    int64_t softmaxOffset = 0;
    if constexpr (LAYOUT_T == SLILayout::BSND) {
        softmaxOffset = (runInfo.bIdx * constInfo.s1Size + runInfo.s1Idx) * constInfo.n2Size * topKSize;
    } else if constexpr (LAYOUT_T == SLILayout::TND) {
        softmaxOffset = runInfo.accumS1Idx * constInfo.n2Size * topKSize;
    }
    int64_t realKSizeAlign8 = CeilDiv(runInfo.kRealSize, 8) * 8;
    DataCopyExtParams copyParams(1, static_cast<uint32_t>(runInfo.kRealSize * sizeof(T)), 0, 0, 0);
    DataCopyPadExtParams<T> copyPadParams(true, 0, static_cast<uint8_t>(realKSizeAlign8 - runInfo.kRealSize), 0.0f);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
    AscendC::DataCopyPad(reduceSumPResUb, attnSoftmaxL1NormGm[softmaxOffset], copyParams, copyPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
    PipeBarrier<PIPE_V>();

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::DataCopyPad(psySyncGm, reduceSumPResUb, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);

    Duplicate(reduceSumDwResUb, static_cast<T>(0.0), PSY_SYNC_BLOCK_FLOAT_NUM);
    SumParams sumPParams = {static_cast<uint32_t>(1), static_cast<uint32_t>(realKSizeAlign8),
                            static_cast<uint32_t>(runInfo.kRealSize)};
    PipeBarrier<PIPE_V>();
    AscendC::Sum(reduceSumDwResUb, reduceSumPResUb, reduceSumTmpBuffer, sumPParams);
    PipeBarrier<PIPE_V>();

    DataCopyExtParams pSumCopyParams(1, PSY_SYNC_BLOCK_FLOAT_NUM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::DataCopyPad(psySyncGm[PSY_SYNC_SUM_P_OFFSET], reduceSumDwResUb, pSumCopyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToVTmp);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[0]);
}

template <typename SLIT>
__aicore__ inline int32_t SLIKLLossVectorService<SLIT>::GetActualSeqLens(int32_t bIdx, int32_t defaultLens,
                                                                         GlobalTensor<int32_t> &actualSeqLensGm,
                                                                         SLILayout layout, int64_t &accumLen)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (layout == SLILayout::TND) {
        accumLen = actualSeqLensGm.GetValue(bIdx);
        return actualSeqLensGm.GetValue(bIdx + 1) - accumLen;
    } else {
        return 0;
    }
}

template <typename SLIT>
__aicore__ inline int32_t SLIKLLossVectorService<SLIT>::GetUsedSeqLens(int32_t bIdx, int32_t defaultLens,
                                                                       GlobalTensor<int32_t> &seqUsedGm)
{
    if (seqUsedGm.GetSize() <= 0) {
        return defaultLens;
    }
    return seqUsedGm.GetValue(bIdx);
}

template <typename SLIT>
__aicore__ inline int32_t SLIKLLossVectorService<SLIT>::GetCmpResidualK(int32_t bIdx)
{
    if (cmpResidualKeyGm.GetSize() <= 0) {
        return 0;
    }
    return cmpResidualKeyGm.GetValue(bIdx);
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetPreCompressS2Len(int32_t bIdx, int32_t actualSeqLensK)
{
    int64_t preCompressS2Len = static_cast<int64_t>(actualSeqLensK) * constInfo.cmpRatio + GetCmpResidualK(bIdx);
    return Max(preCompressS2Len, static_cast<int64_t>(0));
}

template <typename SLIT>
__aicore__ inline int32_t SLIKLLossVectorService<SLIT>::GetS2SparseLen(int32_t bIdx, int32_t s1Idx,
                                                                       int32_t actualSeqLensQ, int32_t actualSeqLensK,
                                                                       int64_t cmpRatio, SLISparseMode sparseMode)
{
    if (sparseMode == SLISparseMode::RightDown) {
        if (cmpRatio != 0) {
            int64_t preCompressS2Len = GetPreCompressS2Len(bIdx, actualSeqLensK);
            return static_cast<int32_t>((preCompressS2Len - actualSeqLensQ + s1Idx + 1) / constInfo.cmpRatio);
        } else {
            return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
        }
    } else if (sparseMode == SLISparseMode::NoMask) {
        return actualSeqLensK;
    } else {
        return 0;
    }
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetInvalidS1Size(int64_t bIdx, int64_t actualSeqLensQ,
                                                                         int64_t actualSeqLensK)
{
    int64_t invalidS1Size = 0;
    if (constInfo.sparseMode == SLISparseMode::RightDown) {
        if (constInfo.cmpRatio != 0) {
            invalidS1Size = actualSeqLensQ - GetPreCompressS2Len(bIdx, actualSeqLensK) + constInfo.cmpRatio - 1;
        } else {
            invalidS1Size = actualSeqLensQ - actualSeqLensK;
        }
    } else if (constInfo.sparseMode == SLISparseMode::NoMask && actualSeqLensK <= 0) {
        invalidS1Size = actualSeqLensQ;
    }
    return Min(Max(invalidS1Size, static_cast<int64_t>(0)), actualSeqLensQ);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::SetCachedValidTotalSize(int64_t validTotalSize)
{
    cachedValidTotalSize_ = validTotalSize;
    ResetValidIdxMap();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ResetValidIdxMap()
{
    mapCurB_ = 0;
    mapPrefixBeforeCurB_ = 0;
    mapValidCntCurB_ = 0;
    mapInvalidS1CurB_ = 0;
    mapAccumS1CurB_ = 0;
    mapAccumS2CurB_ = 0;
    mapSeqQCurB_ = 0;
    mapSeqKCurB_ = 0;
    mapBatchReady_ = false;
}

template <typename SLIT>
__aicore__ inline bool SLIKLLossVectorService<SLIT>::LoadValidBatchAt(int64_t curB)
{
    if (curB < 0 || curB >= constInfo.bSize) {
        mapBatchReady_ = false;
        return false;
    }
    if constexpr (LAYOUT_T == SLILayout::TND) {
        int64_t curAccumS1 = 0;
        int64_t curAccumS2 = 0;
        int32_t seqQ = GetActualSeqLens(curB, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, curAccumS1);
        int32_t seqK = 0;
        if constexpr (KV_LAYOUT_T == SLILayout::TND) {
            seqK = GetActualSeqLens(curB, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, curAccumS2);
        } else {
            seqK = constInfo.s2Size;
            curAccumS2 = curB * constInfo.s2Size;
        }
        mapInvalidS1CurB_ = GetInvalidS1Size(curB, seqQ, seqK);
        mapValidCntCurB_ = Max(static_cast<int64_t>(seqQ) - mapInvalidS1CurB_, static_cast<int64_t>(0));
        mapAccumS1CurB_ = curAccumS1;
        mapAccumS2CurB_ = curAccumS2;
        mapSeqQCurB_ = seqQ;
        mapSeqKCurB_ = seqK;
    } else {
        mapInvalidS1CurB_ = GetInvalidS1Size(curB, constInfo.s1Size, constInfo.s2Size);
        mapValidCntCurB_ = Max(constInfo.s1Size - mapInvalidS1CurB_, static_cast<int64_t>(0));
        mapAccumS1CurB_ = 0;
        mapAccumS2CurB_ = 0;
        mapSeqQCurB_ = constInfo.s1Size;
        mapSeqKCurB_ = constInfo.s2Size;
    }
    mapCurB_ = curB;
    mapBatchReady_ = true;
    return true;
}

template <typename SLIT>
__aicore__ inline bool SLIKLLossVectorService<SLIT>::MapValidIdxToBS1(int64_t validIdx, int64_t &bIdx, int64_t &s1Idx,
                                                                      int64_t &accumS1Len, int64_t &accumS2Len,
                                                                      int32_t &actualSeqLensQ, int32_t &actualSeqLensK)
{
    if (validIdx < 0) {
        return false;
    }
    if (cachedValidTotalSize_ < 0 || validIdx >= cachedValidTotalSize_) {
        return false;
    }

    if (mapBatchReady_ && validIdx >= mapPrefixBeforeCurB_ && validIdx < mapPrefixBeforeCurB_ + mapValidCntCurB_) {
        int64_t remain = validIdx - mapPrefixBeforeCurB_;
        bIdx = mapCurB_;
        s1Idx = mapInvalidS1CurB_ + remain;
        accumS1Len = mapAccumS1CurB_;
        accumS2Len = mapAccumS2CurB_;
        actualSeqLensQ = mapSeqQCurB_;
        actualSeqLensK = mapSeqKCurB_;
        return true;
    }

    if (!mapBatchReady_ || validIdx < mapPrefixBeforeCurB_) {
        ResetValidIdxMap();
        if (!LoadValidBatchAt(0)) {
            return false;
        }
    }

    while (mapBatchReady_ && validIdx >= mapPrefixBeforeCurB_ + mapValidCntCurB_) {
        mapPrefixBeforeCurB_ += mapValidCntCurB_;
        int64_t nextB = mapCurB_ + 1;
        if (!LoadValidBatchAt(nextB)) {
            return false;
        }
    }

    if (!mapBatchReady_ || mapValidCntCurB_ <= 0) {
        return false;
    }

    int64_t remain = validIdx - mapPrefixBeforeCurB_;
    bIdx = mapCurB_;
    s1Idx = mapInvalidS1CurB_ + remain;
    accumS1Len = mapAccumS1CurB_;
    accumS2Len = mapAccumS2CurB_;
    actualSeqLensQ = mapSeqQCurB_;
    actualSeqLensK = mapSeqKCurB_;
    return true;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx,
                                                                SLIKLLossGradRunInfo &runInfo, int64_t accumS1Len,
                                                                int64_t accumS2Len, int32_t actualSeqLensQ,
                                                                int32_t actualSeqLensK)
{
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;

    runInfo.bIdx = bIdx;
    runInfo.s1Idx = s1Idx;
    if constexpr (LAYOUT_T == SLILayout::TND) {
        runInfo.actS1Size = actualSeqLensQ;
        runInfo.actS2Size = actualSeqLensK;
        runInfo.accumS1Idx = accumS1Len + s1Idx;
        runInfo.accumS2Idx = accumS2Len;
    } else if constexpr (LAYOUT_T == SLILayout::BSND) {
        runInfo.actS1Size =
            hasSequsedQ ? GetUsedSeqLens(runInfo.bIdx, constInfo.s1Size, seqUsedQueryGm) : constInfo.s1Size;
        runInfo.actS2Size =
            hasSequsedK ? GetUsedSeqLens(runInfo.bIdx, constInfo.s2Size, seqUsedKeyGm) : constInfo.s2Size;
        runInfo.accumS1Idx = bIdx * constInfo.s1Size + s1Idx;
        runInfo.accumS2Idx = bIdx * constInfo.s2Size;
    }
    runInfo.s2SparseLen = GetS2SparseLen(runInfo.bIdx, runInfo.s1Idx, runInfo.actS1Size, runInfo.actS2Size,
                                         constInfo.cmpRatio, constInfo.sparseMode);
    if (runInfo.s2SparseLen <= 0) {
        runInfo.s2RealSize = 0;
        runInfo.kRealSize = 0;
        runInfo.kRealSizeAlign8 = 0;
        return;
    }
    runInfo.s2RealSize = Min(topKSize, runInfo.s2SparseLen);

    if (constInfo.cmpRatio != 0) {
        runInfo.s2RealSize = Max(1, runInfo.s2RealSize);
    }

    runInfo.kRealSize = runInfo.s2RealSize;
    runInfo.kRealSizeAlign8 = (runInfo.kRealSize + 7) >> 3 << 3;

    if constexpr (LAYOUT_T == SLILayout::TND) {
        runInfo.topkGmBaseOffset = runInfo.accumS1Idx * topKSize;
    } else {
        runInfo.topkGmBaseOffset = (runInfo.bIdx * constInfo.s1Size + runInfo.s1Idx) * topKSize;
    }
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetMetadataTotalSize()
{
    return hasMetadata ? metadataGm.GetValue(optiling::SLI_META_TOTAL_SIZE_INDEX) :
                         tilingData->multiCoreParams.totalSize;
}

template <typename SLIT>
__aicore__ inline int64_t SLIKLLossVectorService<SLIT>::GetMetadataBS1Index(uint32_t coreIdx)
{
    if (hasMetadata) {
        return metadataGm.GetValue(optiling::GetSliMetaBS1IndexAttr(coreIdx));
    }
    return tilingData->multiCoreParams.bS1Index[coreIdx];
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::Vector2ScatterAdd(int32_t vRealKSize,
                                                                       GlobalTensor<MM5_OUT_T> &srcGm,
                                                                       int64_t coreKOffset,
                                                                       SLIKLLossGradRunInfo &runInfo,
                                                                       GlobalTensor<T> &scatterAddGm)
{
    LocalTensor<MM5_OUT_T> scatterAddTmpUb;
    int32_t kSplitSize = ubAllocPolicy.scatterAddUbSize / (2 * sizeof(T) * constInfo.dSizeQueryIndex);
    int32_t tailSize = vRealKSize % kSplitSize;
    int32_t kTailSize = tailSize == 0 ? kSplitSize : tailSize;
    int32_t kProcessSize = kSplitSize;
    int32_t kLoopTimes = CeilDiv(vRealKSize, kSplitSize);
    event_t eventIdArr[2] = {eventIdScatterAdd, eventIdScatterAddPong};
    runInfo.s2Idx = 0;
    int64_t s2IdxOffset = runInfo.s2Idx * constInfo.s2BaseSize;

    SetAtomicAdd<T>();
    for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
        if (kLoopIdx == kLoopTimes - 1) {
            kProcessSize = kTailSize;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddTmpUb = scatterAddUb[scatterAddPingpong * kSplitSize * constInfo.dSizeQueryIndex]
                              .template ReinterpretCast<MM5_OUT_T>();
        DataCopy(scatterAddTmpUb, srcGm[kLoopIdx * kSplitSize * constInfo.dSizeQueryIndex],
                 kProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);

        constexpr int32_t topKSplitSize = 2;
        int32_t topKTail = kProcessSize % topKSplitSize;
        int32_t topKTailSize = topKTail == 0 ? topKSplitSize : topKTail;
        int32_t topKProcessSize = topKSplitSize;
        int32_t topKLoopTimes = CeilDiv(kProcessSize, topKSplitSize);
        for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
            if (topKIdx == topKLoopTimes - 1) {
                topKProcessSize = topKTailSize;
            }
            int64_t s2GmOffset = coreKOffset + kLoopIdx * kSplitSize + topKIdx * topKSplitSize;
            int64_t realS2Idx1 = -1;
            int64_t realS2Idx2 = -1;
            GetRealS2Idx(s2GmOffset, s2IdxOffset, realS2Idx1, runInfo);
            if (topKProcessSize > 1) {
                GetRealS2Idx(s2GmOffset + 1, s2IdxOffset, realS2Idx2, runInfo);
            }

            int64_t s2IdLimit = runInfo.s2SparseLen;
            int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
            int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
            bool keyOffset1Pass = keyOffset1 >= 0;
            bool keyOffset2Pass = keyOffset2 >= 0;
            if (!keyOffset1Pass && !keyOffset2Pass) {
                continue;
            }

            int64_t keySrcStride = ((keyOffset1 > keyOffset2 ? keyOffset1 - keyOffset2 : keyOffset2 - keyOffset1) -
                                    static_cast<int64_t>(constInfo.sparseBlockSize)) *
                                   static_cast<int64_t>(constInfo.dSizeQueryIndex) * sizeof(T);
            bool strideInvalid = keySrcStride >= INT32_MAX || keySrcStride < 0;
            bool copyOutOfRange = realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                                  realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit;
            bool key1LessThanKey2 = realS2Idx1 > realS2Idx2;
            int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
            int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;

            if (!keyOffset1Pass || !keyOffset2Pass || strideInvalid || copyOutOfRange || key1LessThanKey2) {
                // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
                if (keyOffset1Pass) {
                    ScatterAddCopyOutSingle(scatterAddTmpUb[ub1Offset], keyOffset1, scatterAddGm);
                }
                if (keyOffset2Pass) {
                    ScatterAddCopyOutSingle(scatterAddTmpUb[ub2Offset], keyOffset2, scatterAddGm);
                }
            } else {
                DataCopyExtParams dataCopyParams(keyOffset1Pass + keyOffset2Pass, constInfo.dSizeQueryIndex * sizeof(T),
                                                 0, keySrcStride, 0);
                int64_t keyStartOffset = keyOffset1Pass ? keyOffset1 : keyOffset2;
                int64_t ubStartOffset = keyOffset1Pass ? ub1Offset : ub2Offset;
                LocalTensor<T> srcTmpUb = scatterAddTmpUb.template ReinterpretCast<T>();
                DataCopyPad(scatterAddGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset],
                            dataCopyParams);
            }
        }

        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddPingpong = 1 - scatterAddPingpong;
    }
    SetAtomicNone();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessDeterVector2(SLIKLLossGradRunInfo &runInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C2_TO_V2_SA_FLAG[runInfo.taskIdMod2]);
    CrossCoreSetFlag<0, PIPE_MTE2>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
    CrossCoreWaitFlag<0, PIPE_MTE2>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);

    int32_t coreNum = GetBlockNum();
    int32_t peerNums = 0;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int32_t actualSeqLensQ = constInfo.s1Size;
    int32_t actualSeqLensK = constInfo.s2Size;

    for (int32_t idx = 0; idx < coreNum; ++idx) {
        int64_t bS1Index = runInfo.taskId * coreNum + idx;
        int64_t bIdx = 0;
        int64_t s1Idx = 0;
        if (!MapValidIdxToBS1(bS1Index, bIdx, s1Idx, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK)) {
            continue;
        }
        GetRunInfo(runInfo.taskId, bIdx, s1Idx, runInfo, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK);
        if (runInfo.kRealSize <= 0) {
            continue;
        }

        int32_t perCoreKSize = CeilDiv(static_cast<int32_t>(runInfo.kRealSize), coreNum);
        perCoreKSize = SLIGAlign(perCoreKSize, 4);
        int32_t usedCoreNum = CeilDiv(static_cast<int32_t>(runInfo.kRealSize), perCoreKSize);

        int32_t vRealKSize = 0;
        int64_t vCoreKOffset = 0;
        if (constInfo.aicIdx < static_cast<uint32_t>(usedCoreNum)) {
            int32_t tailCoreKSize = runInfo.kRealSize - (usedCoreNum - 1) * perCoreKSize;
            int32_t curCoreKSize =
                constInfo.aicIdx == static_cast<uint32_t>(usedCoreNum - 1) ? tailCoreKSize : perCoreKSize;
            int32_t v0RealKSize = CeilDiv(curCoreKSize, 2);
            v0RealKSize = Min(SLIGAlign(v0RealKSize, 2), curCoreKSize);
            int32_t v1RealKSize = curCoreKSize - v0RealKSize;
            int64_t coreKOffset = static_cast<int64_t>(constInfo.aicIdx) * perCoreKSize;
            if (constInfo.subBlockIdx == 0) {
                vRealKSize = v0RealKSize;
                vCoreKOffset = coreKOffset;
            } else {
                vRealKSize = v1RealKSize;
                vCoreKOffset = coreKOffset + v0RealKSize;
            }
        }

        int32_t bankIdx = peerNums % SLI_DETER_SCATTER_BANK_NUM;
        if (bankIdx == 0 && peerNums > 0) {
            CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
        }
        if (vRealKSize > 0) {
            int64_t srcOffset = idx * topKSize * constInfo.dSizeQueryIndex * 2;
            GlobalTensor<MM5_OUT_T> srcGm =
                bmm5ResGm[srcOffset + (runInfo.taskIdMod2 * topKSize + vCoreKOffset) * constInfo.dSizeQueryIndex];
            Vector2ScatterAdd(vRealKSize, srcGm, vCoreKOffset, runInfo, scatterAddResGmBanks[bankIdx]);
        }
        ++peerNums;
        if ((peerNums % SLI_DETER_SCATTER_BANK_NUM) == 0) {
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
        }
    }

    if (peerNums <= 0) {
        return;
    }
    if ((peerNums % SLI_DETER_SCATTER_BANK_NUM) != 0) {
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
    }
    CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessVector2(SLIKLLossGradRunInfo &runInfo)
{
    CrossCoreWaitFlag(SYNC_C2_TO_V2_SA_FLAG[runInfo.taskIdMod2]);

    int32_t v0RealKSize, v1RealKSize, vRealKSize;
    int64_t coreKOffset;
    v0RealKSize = CeilDiv(runInfo.kRealSize, 2);
    v0RealKSize = Min(SLIGAlign(v0RealKSize, 2), runInfo.kRealSize);
    v1RealKSize = runInfo.kRealSize - v0RealKSize;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = v0RealKSize;
        coreKOffset = 0;
    } else {
        vRealKSize = v1RealKSize;
        coreKOffset = v0RealKSize;
    }
    if (vRealKSize <= 0) {
        return;
    }

    int srcOffset = constInfo.aicIdx * topKSize * constInfo.dSizeQueryIndex * 2;
    GlobalTensor<MM5_OUT_T> srcGm =
        bmm5ResGm[srcOffset + (runInfo.taskIdMod2 * topKSize + coreKOffset) * constInfo.dSizeQueryIndex];
    LocalTensor<MM5_OUT_T> scatterAddTmpUb;
    int32_t kSplitSize = ubAllocPolicy.scatterAddUbSize / (2 * sizeof(T) * constInfo.dSizeQueryIndex);
    int32_t tailSize = vRealKSize % kSplitSize;
    int32_t kTailSize = (!tailSize) ? kSplitSize : tailSize;
    int32_t kProcessSize = kSplitSize;
    int32_t kLoopTimes = CeilDiv(vRealKSize, kSplitSize);
    int64_t realS2Idx1, realS2Idx2, s2GmOffset;
    event_t eventIdArr[2] = {eventIdScatterAdd, eventIdScatterAddPong};
    runInfo.s2Idx = 0;
    int64_t s2IdxOffset = static_cast<int64_t>(runInfo.s2Idx) * constInfo.s2BaseSize;

    SetAtomicAdd<T>();
    for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
        if (kLoopIdx >= kLoopTimes - 1) {
            kProcessSize = kTailSize;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddTmpUb = scatterAddUb[scatterAddPingpong * kSplitSize * constInfo.dSizeQueryIndex]
                              .template ReinterpretCast<MM5_OUT_T>();
        DataCopy(scatterAddTmpUb, srcGm[kLoopIdx * kSplitSize * constInfo.dSizeQueryIndex],
                 kProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);

        constexpr int32_t topKSplitSize = 2;
        int32_t tailSize = kProcessSize % topKSplitSize;
        int32_t topKTailSize = (!tailSize) ? topKSplitSize : tailSize;
        int32_t topKProcessSize = topKSplitSize;
        int32_t topKLoopTimes = CeilDiv(kProcessSize, topKSplitSize);
        for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
            if (topKIdx >= topKLoopTimes - 1) {
                topKProcessSize = topKTailSize;
            }
            s2GmOffset = coreKOffset + kLoopIdx * kSplitSize + topKIdx * topKSplitSize;
            GetRealS2Idx(s2GmOffset, s2IdxOffset, realS2Idx1, runInfo);
            realS2Idx2 = -1;
            if (topKProcessSize > 1) {
                GetRealS2Idx(s2GmOffset + 1, s2IdxOffset, realS2Idx2, runInfo);
            }

            int64_t s2IdLimit = runInfo.s2SparseLen;
            int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
            int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, constInfo.dSizeQueryIndex);
            bool keyOffset1Pass = (keyOffset1 >= 0);
            bool keyOffset2Pass = (keyOffset2 >= 0);
            if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
                break;
            }

            int64_t keySrcStride = 0;
            keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                            constInfo.sparseBlockSize) *
                           constInfo.dSizeQueryIndex * sizeof(T);

            bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0);
            bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                                   realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);
            bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);

            int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
            int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;

            if (strideInvalid || copyOutOfRange || key1LessThankey2) {
                // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
                ScatterAddCopyOutSingle(scatterAddTmpUb[ub1Offset], keyOffset1, scatterAddResGm);
                ScatterAddCopyOutSingle(scatterAddTmpUb[ub2Offset], keyOffset2, scatterAddResGm);
            } else {
                DataCopyExtParams dataCopyParams(keyOffset1Pass + keyOffset2Pass, constInfo.dSizeQueryIndex * sizeof(T),
                                                 0, keySrcStride, 0);
                int64_t keyStartOffset = (keyOffset1Pass) ? keyOffset1 : keyOffset2;
                int64_t ubStartOffset = (keyOffset1Pass) ? ub1Offset : ub2Offset;
                LocalTensor<T> srcTmpUb = scatterAddTmpUb.template ReinterpretCast<T>();
                DataCopyPad(scatterAddResGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset],
                            dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddPingpong = 1 - scatterAddPingpong;
    }
    SetAtomicNone();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ProcessPrivateScatterVector2(SLIKLLossGradRunInfo &runInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C2_TO_V2_SA_FLAG[runInfo.taskIdMod2]);
    if (!runInfo.isValid) {
        return;
    }

    int32_t v0RealKSize = CeilDiv(runInfo.kRealSize, 2);
    v0RealKSize = Min(SLIGAlign(v0RealKSize, 2), runInfo.kRealSize);
    int32_t v1RealKSize = runInfo.kRealSize - v0RealKSize;
    int32_t vRealKSize = 0;
    int64_t coreKOffset = 0;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = v0RealKSize;
        coreKOffset = 0;
    } else {
        vRealKSize = v1RealKSize;
        coreKOffset = v0RealKSize;
    }
    if (vRealKSize <= 0) {
        return;
    }

    int srcOffset = constInfo.aicIdx * topKSize * constInfo.dSizeQueryIndex * 2;
    GlobalTensor<MM5_OUT_T> srcGm =
        bmm5ResGm[srcOffset + (runInfo.taskIdMod2 * topKSize + coreKOffset) * constInfo.dSizeQueryIndex];
    Vector2ScatterAdd(vRealKSize, srcGm, coreKOffset, runInfo, scatterAddResGm);
}

#if 0
template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorP(SLIKLLossGradRunInfo &runInfo)
{
    int64_t bmm1Offset = runInfo.taskIdMod2 * constInfo.gSizeQuery * topKSize;
    int64_t nLoops = CeilDiv(constInfo.gSizeQuery, runInfo.nBaseSizeP);
    int64_t kLoops = CeilDiv(runInfo.kRealSize, K_BASE_SIZE);

    LocalTensor<T> mulsResUb[2] = {mm1Tbuf.Get<T>(), mm2TBuf.Get<T>()};
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(16 * 1024, 0);
    LocalTensor<T> reduceSumPTmpBuffer = sharedTBuf.GetWithOffset<T>(K_BASE_SIZE, 16 * 1024);

    int64_t softmaxInputSize = CeilDiv(runInfo.nBaseSizeP, 8) * 8;
    LocalTensor<T> softmaxMaxTmp[2] = {sharedTBuf.GetWithOffset<T>(softmaxInputSize, 0), sharedTBuf.GetWithOffset<T>(softmaxInputSize, softmaxInputSize * sizeof(T))};
    LocalTensor<T> softmaxSumTmp[2] = {sharedTBuf.GetWithOffset<T>(softmaxInputSize, 2 * softmaxInputSize * sizeof(T)), sharedTBuf.GetWithOffset<T>(softmaxInputSize, 3 * softmaxInputSize * sizeof(T))};
    LocalTensor<T> softmaxMaxUb = sharedTBuf.GetWithOffset<T>(softmaxInputSize * 8, 1024);
    LocalTensor<T> softmaxSumUb = sharedTBuf.GetWithOffset<T>(softmaxInputSize * 8, 2048);

    uint8_t pingpong = 0;
    for (int64_t n = 0; n < nLoops; n++) { // nLoops=4
        int64_t curNSize = (n == nLoops - 1) ? constInfo.gSizeQuery - (runInfo.nBaseSizeP * n) : runInfo.nBaseSizeP;
        for (int64_t k = 0; k < kLoops; k++) {
            int64_t curKOffset = K_BASE_SIZE * k;
            int64_t curKSize = (k == kLoops - 1) ? runInfo.kRealSize - curKOffset : K_BASE_SIZE;
            int64_t curKSizeAlign8 = CeilDiv(curKSize, 8) * 8;
            int64_t bmm1NOffset = n * runInfo.nBaseSizeP * topKSize + k * K_BASE_SIZE;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            DataCopyExtParams copyParams(   static_cast<uint16_t>(curNSize),
                                            static_cast<uint32_t>(curKSize * sizeof(T)),
                                            static_cast<uint32_t>((topKSize - curKSize) * sizeof(T)), 0, 0);
            DataCopyPadExtParams<T> copyPadParams(false, 0, 0, 0);
            AscendC::DataCopyPad(mulsResUb[pingpong], bmm1ResGm[bmm1Offset + bmm1NOffset], copyParams, copyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::Muls(mulsResUb[pingpong], mulsResUb[pingpong], constInfo.scaleValue, curNSize * curKSizeAlign8);
            PipeBarrier<PIPE_V>();

            DataCopyExtParams softmaxCopyParams(    static_cast<uint16_t>(1),
                                                    static_cast<uint32_t>(runInfo.nBaseSizeP * sizeof(T)), // 8 * 4 = 32Bytes
                                                    static_cast<uint32_t>(0), 0, 0);
            DataCopyPadExtParams<T> softmaxCopyPadParams(false, 0, 0, 0);

            int64_t softmaxInputOffset = 0;
            if constexpr (LAYOUT_T == SLILayout::BSND) {
                softmaxInputOffset = runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQuery) +
                    runInfo.s1Idx * constInfo.gSizeQuery + n * runInfo.nBaseSizeP;
            } else if constexpr (LAYOUT_T == SLILayout::TND) {
                softmaxInputOffset = runInfo.accumS1Idx * constInfo.gSizeQuery + n * runInfo.nBaseSizeP;
            }
            AscendC::DataCopyPad(softmaxMaxTmp[pingpong], softmaxMaxGm[softmaxInputOffset], softmaxCopyParams, softmaxCopyPadParams);
            AscendC::DataCopyPad(softmaxSumTmp[pingpong], softmaxSumGm[softmaxInputOffset], softmaxCopyParams, softmaxCopyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::Brcb(softmaxMaxUb, softmaxMaxTmp[pingpong], curNSize, {1, 8});
            AscendC::Brcb(softmaxSumUb, softmaxSumTmp[pingpong], curNSize, {1, 8});
            PipeBarrier<PIPE_V>();
            LocalTensor<uint8_t> softmaxTmpUb = sharedTBuf.GetWithOffset<uint8_t>(8*1024, 0);
            AscendC::SimpleSoftMax<T>(mulsResUb[pingpong], softmaxSumUb, softmaxMaxUb, mulsResUb[pingpong],
                                      softmaxTmpUb, constInfo.simpleSoftMaxTilingInfo,
                                      {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8),
                                      static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSize)});
            PipeBarrier<PIPE_V>();
            uint32_t reduceShape[] = {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8)};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(reduceSumPTmpBuffer, mulsResUb[pingpong], sharedTmpBuffer, reduceShape, true);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            if (n == 0) {
                AscendC::DataCopy(reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            } else {
                AscendC::Add(reduceSumPResUb[curKOffset], reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            }
            PipeBarrier<PIPE_V>();
            pingpong = 1 - pingpong;
        }
    }
    PipeBarrier<PIPE_V>();
    float gRec = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.gSizeQuery));
    AscendC::Muls(reduceSumPResUb, reduceSumPResUb, gRec, topKSize);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::DataCopy(psySyncGm, reduceSumPResUb, topKSize);
}
#endif

#if 0
template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorPWithOutSoftMaxSum(SLIKLLossGradRunInfo &runInfo)
{
    int64_t bmm1Offset = runInfo.taskIdMod2 * constInfo.gSizeQuery * topKSize;
    int64_t nLoops = CeilDiv(constInfo.gSizeQuery, runInfo.nBaseSizeP);
    int64_t kLoops = CeilDiv(runInfo.kRealSize, K_BASE_SIZE);

    LocalTensor<T> mulsResUb[2] = {mm1Tbuf.Get<T>(), mm2TBuf.Get<T>()};
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTBuf.GetWithOffset<uint8_t>(16 * 1024, 0);
    LocalTensor<T> reduceSumPTmpBuffer = sharedTBuf.GetWithOffset<T>(K_BASE_SIZE, 16 * 1024);

    uint8_t pingpong = 0;
    for (int64_t n = 0; n < nLoops; n++) {
        int64_t curNSize = (n == nLoops - 1) ? constInfo.gSizeQuery - (runInfo.nBaseSizeP * n) : runInfo.nBaseSizeP;
        for (int64_t k = 0; k < kLoops; k++) {
            int64_t curKOffset = K_BASE_SIZE * k;
            int64_t curKSize = (k == kLoops - 1) ? runInfo.kRealSize - curKOffset : K_BASE_SIZE;
            int64_t curKSizeAlign8 = CeilDiv(curKSize, 8) * 8;
            int64_t bmm1NOffset = n * runInfo.nBaseSizeP * topKSize + k * K_BASE_SIZE;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            DataCopyExtParams copyParams(   static_cast<uint16_t>(curNSize),
                                            static_cast<uint32_t>(curKSize * sizeof(T)),
                                            static_cast<uint32_t>((topKSize - curKSize) * sizeof(T)), 0, 0);
            DataCopyPadExtParams<T> copyPadParams(false, 0, 0, 0);
            AscendC::DataCopyPad(mulsResUb[pingpong], bmm1ResGm[bmm1Offset + bmm1NOffset], copyParams, copyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerVecP);
            AscendC::Muls(mulsResUb[pingpong], mulsResUb[pingpong], constInfo.scaleValue, curNSize * curKSizeAlign8);
            PipeBarrier<PIPE_V>();

            LocalTensor<uint8_t> softmaxTmpUb = sharedTBuf.GetWithOffset<uint8_t>(8*1024, 0);
            AscendC::SoftMax<T>(mulsResUb[pingpong], mulsResUb[pingpong], softmaxTmpUb, constInfo.simpleSoftMaxTilingInfo,
                                {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8),
                                 static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSize)});

            PipeBarrier<PIPE_V>();
            uint32_t reduceShape[] = {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(curKSizeAlign8)};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(reduceSumPTmpBuffer, mulsResUb[pingpong], sharedTmpBuffer, reduceShape, true);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2P[pingpong]);
            if (n == 0) {
                AscendC::DataCopy(reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            } else {
                AscendC::Add(reduceSumPResUb[curKOffset], reduceSumPResUb[curKOffset], reduceSumPTmpBuffer, curKSizeAlign8);
            }
            PipeBarrier<PIPE_V>();
            pingpong = 1 - pingpong;
        }
    }
    PipeBarrier<PIPE_V>();
    float gRec = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.gSizeQuery));
    AscendC::Muls(reduceSumPResUb, reduceSumPResUb, gRec, topKSize);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerVecP);
    AscendC::DataCopy(psySyncGm, reduceSumPResUb, topKSize);
}
#endif

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::PreloadWeight(SLIKLLossGradRunInfo &runInfo)
{
    int64_t weightOffset = 0;
    if constexpr (LAYOUT_T == SLILayout::BSND) {
        weightOffset = (runInfo.bIdx * constInfo.s1Size + runInfo.s1Idx) * constInfo.gSizeQueryIndex;
    } else if constexpr (LAYOUT_T == SLILayout::TND) {
        weightOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex;
    }
    int64_t weightDBOffset = runInfo.taskIdMod2 * constInfo.gSizeQueryIndexAlign16;
    // weight 可以常驻, 所以直接搬运, 减少搬运切片

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[runInfo.taskIdMod2]);
    AscendC::DataCopy(weightInUb[weightDBOffset], weightGm[weightOffset], constInfo.gSizeQueryIndexAlign16);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerPreW);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVInnerPreW);
    AscendC::Adds(weightUb[weightDBOffset], weightInUb[weightDBOffset], static_cast<T>(0.0),
                  constInfo.gSizeQueryIndexAlign16);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[runInfo.taskIdMod2]);
    PipeBarrier<PIPE_V>();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorSy(SLIKLLossGradRunInfo &runInfo)
{
    {
        // weight UB cast To weight.GetValue and release eventID
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    }
    // 核内保留一份BS1Idx
    int64_t realKSize = runInfo.s2RealSize; // int64_t &realKSize = runInfo.s2RealSize;
    int64_t realKSizeAlign = CeilDiv(realKSize, 8) * 8;

    // 切B, S1做循环, bmm2 单次存储 [N1_INDEX, TopK] 大小的数据块
    // 为了softmax K轴尽量不切分, 只能在N1方向上切
    int64_t bmm2ResSize = constInfo.gSizeQueryIndex * topKSize; //  sizeof(float)
    int64_t bmm2ResOffset = runInfo.taskIdMod2 * bmm2ResSize;
    // relu无法复用LocalTensor 所以再 8/ 2 = 4行
    int64_t nSplitSize = ubAllocPolicy.mm2UbSize / (topKSize * sizeof(float));
    // 一个Cube 做 G 行, 分给两个Vector, 每个做 G / 2 的大小
    int64_t nLoopSize = CeilDiv(constInfo.gSizeQueryIndex, nSplitSize);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    for (int64_t n = 0; n < nLoopSize; n++) {
        int64_t curNSize = Min(nSplitSize, static_cast<int64_t>(constInfo.gSizeQueryIndex) - n * nSplitSize);
        int64_t bmm2Offset = n * nSplitSize * topKSize; // 4 * 2048
        DataCopyExtParams copyBmm2Params(static_cast<uint16_t>(curNSize), static_cast<uint32_t>(realKSize * sizeof(T)),
                                         static_cast<uint32_t>((topKSize - realKSize) * sizeof(T)), 0, 0);
        DataCopyPadExtParams<T> copyBmm2PadParams(true, 0, (uint8_t)(realKSizeAlign - realKSize), 0.0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
        AscendC::DataCopyPad<T>(reluResUb, bmm2ResGm[bmm2ResOffset + bmm2Offset], copyBmm2Params, copyBmm2PadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
        int64_t weightOffset = !constInfo.subBlockIdx * constInfo.gSizeQueryIndexAlign16;
        for (int32_t i = 0; i < curNSize; i++) {
            float weightValue = weightUb[weightOffset].GetValue(n * nSplitSize + i);
            int64_t mulOffset = i * realKSizeAlign;
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::Muls<T>(reluResUb[mulOffset], reluResUb[mulOffset], weightValue, realKSizeAlign);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<T> reduceUb = nSplitSize > 1 ? reduceSumYResTmpBuffer : reluResUb;
        if (nSplitSize > 1) {
            // 使用高阶API做自选维度的reduce
            // 不支持源操作数与目的操作数地址重叠。不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
            uint32_t reduceShape[] = {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(realKSizeAlign)};
            constexpr bool isReuse = true;
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, isReuse>(reduceUb, reluResUb, reduceSumTmpBuffer,
                                                                         reduceShape, true);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
            PipeBarrier<PIPE_V>();
        }

        if (n == 0) {
            AscendC::DataCopy(reduceSumYResUb, reduceUb, realKSizeAlign);
        } else {
            AscendC::Add(reduceSumYResUb, reduceSumYResUb, reduceUb, realKSizeAlign);
        }

        if (nSplitSize == 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
        }
        PipeBarrier<PIPE_V>();
    }

    SoftMaxShapeInfo softmaxShapeInfoData = {
        static_cast<uint32_t>(1),
        static_cast<uint32_t>(realKSizeAlign),
        static_cast<uint32_t>(1),
        static_cast<uint32_t>(realKSize),
    };
    AscendC::SoftMax<T>(reduceSumYResUb, reduceSumYResUb, softmaxTmpBuffer, constInfo.tilingInfo,
                        {static_cast<uint32_t>(1), static_cast<uint32_t>(realKSizeAlign), static_cast<uint32_t>(1),
                         static_cast<uint32_t>(realKSize)});
    PipeBarrier<PIPE_V>();
    DataCopyExtParams copyParams(1, static_cast<uint32_t>(realKSize * sizeof(float)), 0, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerSy);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerSy);
    AscendC::DataCopyPad(psySyncGm[topKSize], reduceSumYResUb, copyParams);
    int64_t softmaxOutOffset = 0;
    if constexpr (LAYOUT_T == SLILayout::BSND) {
        softmaxOutOffset = (runInfo.bIdx * constInfo.s1Size + runInfo.s1Idx) * constInfo.n2Size * topKSize;
    } else if constexpr (LAYOUT_T == SLILayout::TND) {
        softmaxOutOffset = runInfo.accumS1Idx * constInfo.n2Size * topKSize;
    }
    AscendC::DataCopyPad(softmaxOutGm[softmaxOutOffset], reduceSumYResUb, copyParams);
    if (realKSize < topKSize) {
        AscendC::InitOutput(softmaxOutGm[softmaxOutOffset + realKSize], topKSize - realKSize, static_cast<T>(0));
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
}

template <typename SLIT>
__aicore__ inline bool SLIKLLossVectorService<SLIT>::ShouldEnableSyLoadBalance(
    const SLIKLLossGradRunInfo &runInfo) const
{
    int64_t gSize = static_cast<int64_t>(constInfo.gSizeQueryIndex);
    if (gSize < SY_LB_MIN_G_SIZE) {
        return false;
    }
    int64_t rowsPerTile = ubAllocPolicy.mm2UbSize / (topKSize * sizeof(T));
    int64_t serialLoops = CeilDiv(gSize, rowsPerTile);
    int64_t serialWork = serialLoops * static_cast<int64_t>(runInfo.s2RealSize);
    return serialWork >= SY_LB_MIN_SERIAL_WORK;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorSyLoadBalance(SLIKLLossGradRunInfo &runInfo)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    int64_t realKSize = runInfo.s2RealSize;
    int64_t realKSizeAlign = CeilDiv(realKSize, 8) * 8;
    int64_t gHalfSize = constInfo.gSizeQueryIndex / 2;
    int64_t pOwnerGSize = Max(static_cast<int64_t>(0), gHalfSize - P_OWNER_SY_ROW_BIAS);
    int64_t gStart = runInfo.calcP ? 0 : pOwnerGSize;
    int64_t gProcessSize = runInfo.calcP ? pOwnerGSize : constInfo.gSizeQueryIndex - pOwnerGSize;
    int64_t bmm2ResSize = constInfo.gSizeQueryIndex * topKSize;
    int64_t bmm2ResOffset = runInfo.taskIdMod2 * bmm2ResSize;
    int64_t nSplitSize = ubAllocPolicy.mm2UbSize / (topKSize * sizeof(float));
    int64_t nLoopSize = CeilDiv(gProcessSize, nSplitSize);
    int64_t weightOffset = runInfo.taskIdMod2 * constInfo.gSizeQueryIndexAlign16 + gStart;
    LocalTensor<T> syInputUb[2] = {pingBuf, pongBuf};
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2SyPingPong[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2SyPingPong[1]);
    if (gProcessSize == 0) {
        AscendC::Duplicate(reduceSumYResUb, static_cast<T>(0), realKSizeAlign);
        PipeBarrier<PIPE_V>();
    }
    if (nLoopSize > 0) {
        int64_t firstNSize = Min(nSplitSize, gProcessSize);
        DataCopyExtParams firstCopyParams(static_cast<uint16_t>(firstNSize),
                                          static_cast<uint32_t>(realKSize * sizeof(T)),
                                          static_cast<uint32_t>((topKSize - realKSize) * sizeof(T)), 0, 0);
        DataCopyPadExtParams<T> firstCopyPadParams(true, 0, static_cast<uint8_t>(realKSizeAlign - realKSize), 0.0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2SyPingPong[0]);
        AscendC::DataCopyPad<T>(syInputUb[0], bmm2ResGm[bmm2ResOffset + gStart * topKSize], firstCopyParams,
                                firstCopyPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVSyPingPong[0]);
    }
    for (int64_t n = 0; n < nLoopSize; n++) {
        int32_t pingPong = n & 1;
        int64_t curNSize = Min(nSplitSize, gProcessSize - n * nSplitSize);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVSyPingPong[pingPong]);
        if (n + 1 < nLoopSize) {
            int32_t nextPingPong = 1 - pingPong;
            int64_t nextN = n + 1;
            int64_t nextNSize = Min(nSplitSize, gProcessSize - nextN * nSplitSize);
            DataCopyExtParams nextCopyParams(static_cast<uint16_t>(nextNSize),
                                             static_cast<uint32_t>(realKSize * sizeof(T)),
                                             static_cast<uint32_t>((topKSize - realKSize) * sizeof(T)), 0, 0);
            DataCopyPadExtParams<T> nextCopyPadParams(true, 0, static_cast<uint8_t>(realKSizeAlign - realKSize), 0.0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2SyPingPong[nextPingPong]);
            AscendC::DataCopyPad<T>(syInputUb[nextPingPong],
                                    bmm2ResGm[bmm2ResOffset + (gStart + nextN * nSplitSize) * topKSize], nextCopyParams,
                                    nextCopyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToVSyPingPong[nextPingPong]);
        }
        for (int32_t i = 0; i < curNSize; i++) {
            float weightValue = weightUb[weightOffset].GetValue(n * nSplitSize + i);
            int64_t mulOffset = i * realKSizeAlign;
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::Muls<T>(syInputUb[pingPong][mulOffset], syInputUb[pingPong][mulOffset], weightValue,
                             realKSizeAlign);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<T> reduceUb = nSplitSize > 1 ? reduceSumYResTmpBuffer : syInputUb[pingPong];
        if (nSplitSize > 1) {
            uint32_t reduceShape[] = {static_cast<uint32_t>(curNSize), static_cast<uint32_t>(realKSizeAlign)};
            constexpr bool isReuse = true;
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, isReuse>(reduceUb, syInputUb[pingPong],
                                                                         reduceSumTmpBuffer, reduceShape, true);
            PipeBarrier<PIPE_V>();
        }
        if (n == 0) {
            AscendC::DataCopy(reduceSumYResUb, reduceUb, realKSizeAlign);
        } else {
            AscendC::Add(reduceSumYResUb, reduceSumYResUb, reduceUb, realKSizeAlign);
        }
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2SyPingPong[pingPong]);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2SyPingPong[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2SyPingPong[1]);
    DataCopyExtParams copyParams(1, static_cast<uint32_t>(realKSize * sizeof(T)), 0, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerSy);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerSy);
    AscendC::DataCopyPad(psySyncGm[topKSize], reduceSumYResUb, copyParams);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::FinalizeVectorSy(SLIKLLossGradRunInfo &runInfo)
{
    int64_t realKSize = runInfo.s2RealSize;
    int64_t realKSizeAlign = CeilDiv(realKSize, 8) * 8;
    LocalTensor<T> peerYUb = mm2TBuf.Get<T>();
    DataCopyExtParams copyParams(1, static_cast<uint32_t>(realKSize * sizeof(T)), 0, 0, 0);
    DataCopyPadExtParams<T> copyPadParams(true, 0, static_cast<uint8_t>(realKSizeAlign - realKSize), 0.0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    AscendC::DataCopyPad(peerYUb, psySyncOtherGm[topKSize], copyParams, copyPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
    AscendC::Add(reduceSumYResUb, reduceSumYResUb, peerYUb, realKSizeAlign);
    PipeBarrier<PIPE_V>();
    AscendC::SoftMax<T>(reduceSumYResUb, reduceSumYResUb, softmaxTmpBuffer, constInfo.tilingInfo,
                        {static_cast<uint32_t>(1), static_cast<uint32_t>(realKSizeAlign), static_cast<uint32_t>(1),
                         static_cast<uint32_t>(realKSize)});
    PipeBarrier<PIPE_V>();
    int64_t softmaxOutOffset = 0;
    if constexpr (LAYOUT_T == SLILayout::BSND) {
        softmaxOutOffset = (runInfo.bIdx * constInfo.s1Size + runInfo.s1Idx) * constInfo.n2Size * topKSize;
    } else if constexpr (LAYOUT_T == SLILayout::TND) {
        softmaxOutOffset = runInfo.accumS1Idx * constInfo.n2Size * topKSize;
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerSy);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3InnerSy);
    AscendC::DataCopyPad(psySyncGm[topKSize], reduceSumYResUb, copyParams);
    AscendC::DataCopyPad(softmaxOutGm[softmaxOutOffset], reduceSumYResUb, copyParams);
    if (realKSize < topKSize) {
        AscendC::InitOutput(softmaxOutGm[softmaxOutOffset + realKSize], topKSize - realKSize, static_cast<T>(0));
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::ReLUGrad(LocalTensor<KV_T> &reluGradOutTensor,
                                                              LocalTensor<T> &subResTensor, LocalTensor<T> &maskUb,
                                                              int32_t kRealSizeAlign)
{
    // 根据mask选择subResTensor中可用数据
    Select(subResTensor, maskUb, subResTensor, static_cast<T>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
           kRealSizeAlign);
    PipeBarrier<PIPE_V>();
    Cast(reluGradOutTensor, subResTensor, AscendC::RoundMode::CAST_ROUND, kRealSizeAlign);
    PipeBarrier<PIPE_V>();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorDwDqDkLess2k(SLIKLLossGradRunInfo &runInfo)
{
    LocalTensor<T> reluResUb[2] = {pingBuf, pongBuf};
    LocalTensor<T> mulLeftUb[2] = {pingBuf, pongBuf};
    LocalTensor<T> reduceSumPUbSingleK, reduceSumYUbSingleK;
    LocalTensor<KV_T> reluGradUb[2] = {reluGradpingBuf, reluGradpongBuf};

    // maskTensor需要创建 目前一次传输32k数据对比需要8k个数，需要8k/8=1k大小空间（按bit存储）
    LocalTensor<T> maskUb = this->dwTBuf.template Get<T>()[256];
    // dwTBuf理论上最大只需要512bytes，一个元素4bytes，一共最大64个元素，开DB一共256*2=512
    LocalTensor<T> reduceSumResTensor = this->dwTBuf.template Get<T>();

    int32_t kRealSize = runInfo.kTailSize;
    int32_t kRealSizeAlign16 = BlockAlign<KV_T>(kRealSize); // 按照KV_T类型对齐，datacopy才不会有问题

    SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    DataCopyExtParams pSumCopyParams(1, PSY_SYNC_BLOCK_FLOAT_NUM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
    DataCopyPadExtParams<T> pSumPadParams(false, 0, 0, 0.0f);
    if (!ShouldEnableSyLoadBalance(runInfo)) {
        if (runInfo.calcP) {
            reduceSumPUbSingleK = resPSYTBuf.Get<T>();
            reduceSumYUbSingleK = mm2TBuf.template Get<T>()[6144];
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
            DataCopy(reduceSumYUbSingleK, psySyncOtherGm[topKSize], kRealSizeAlign16);
            DataCopyPad(reduceSumDwResUb, psySyncGm[PSY_SYNC_SUM_P_OFFSET], pSumCopyParams, pSumPadParams);
        } else {
            reduceSumPUbSingleK = mm2TBuf.template Get<T>()[6144];
            reduceSumYUbSingleK = resPSYTBuf.Get<T>();
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
            DataCopy(reduceSumPUbSingleK, psySyncOtherGm, kRealSizeAlign16);
            DataCopyPad(reduceSumDwResUb, psySyncOtherGm[PSY_SYNC_SUM_P_OFFSET], pSumCopyParams, pSumPadParams);
        }
    } else {
        reduceSumPUbSingleK = mm2TBuf.template Get<T>()[6144];
        reduceSumYUbSingleK = resPSYTBuf.Get<T>();
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        GlobalTensor<T> &pOwnerSyncGm = runInfo.calcP ? psySyncGm : psySyncOtherGm;
        DataCopy(reduceSumPUbSingleK, pOwnerSyncGm, kRealSizeAlign16);
        if (runInfo.calcP) {
            DataCopy(reduceSumYUbSingleK, psySyncOtherGm[topKSize], kRealSizeAlign16);
        }
        DataCopyPad(reduceSumDwResUb, pOwnerSyncGm[PSY_SYNC_SUM_P_OFFSET], pSumCopyParams, pSumPadParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);

    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToSPsum);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToSPsum);
    T pSumValue = reduceSumDwResUb.GetValue(0);
    SetFlag<HardEvent::S_V>(eventIdSToVPsum);
    WaitFlag<HardEvent::S_V>(eventIdSToVPsum);
    Muls(reduceSumYUbSingleK, reduceSumYUbSingleK, pSumValue, kRealSizeAlign16);
    PipeBarrier<PIPE_V>();

    Sub(v1TmpUb, reduceSumYUbSingleK, reduceSumPUbSingleK, kRealSize);
    PipeBarrier<PIPE_V>();

    // k方向单次处理最大长度为2048， g方向为4，故一次性为 4*2048*4 = 32k
    uint32_t gSizePerVec = constInfo.gSizeQueryIndex / 2;
    constexpr uint32_t gSizeInner = 4;
    int32_t gLoopTimes = gSizePerVec / gSizeInner;

    SumParams sumParams = {static_cast<uint32_t>(gSizeInner), static_cast<uint32_t>(kRealSizeAlign16),
                           static_cast<uint32_t>(kRealSize)};
    DataCopyParams dataCopyReluResParams(gSizeInner, kRealSizeAlign16 * sizeof(T),
                                         (topKSize - kRealSizeAlign16) * sizeof(T), 0);
    DataCopyPadParams dataCopyReluResPadParams(false, 0, 0, 0);
    DataCopyParams dataCopyReluGradParams(gSizeInner, kRealSize * sizeof(KV_T), 0,
                                          (topKSize - kRealSize) * sizeof(KV_T));

    int64_t subBlockGQueryIndexOffset = constInfo.subBlockIdx * gSizePerVec;
    int64_t reluGmOffset = (runInfo.taskIdMod2 * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset) * topKSize;
    int64_t weightOffset = runInfo.taskIdMod2 * constInfo.gSizeQueryIndexAlign16 + subBlockGQueryIndexOffset;

    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[0]);
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[1]);
    SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[1]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[0]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[1]);

    for (size_t nIdx = 0; nIdx < gLoopTimes; nIdx++) {
        uint8_t pingPong = nIdx & 1;
        maskUb = maskUb[pingPong * 256]; // 相差256个数正好1k
        WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[pingPong]);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[pingPong]);

        // 获取输入的relu数据和weights数据偏差每gSizePerVec中gSizeInner个
        int64_t reluResOffset = reluGmOffset + nIdx * gSizeInner * topKSize;
        // 每次搬运gSizeInner个kRealSizeAlign16 | reluResUb shape-> (gSizeInner, kRealSizeAlign16)
        DataCopyPad(reluResUb[pingPong], reluGm[reluResOffset], dataCopyReluResParams, dataCopyReluResPadParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);

        CompareScalar(maskUb, reluResUb[pingPong], static_cast<T>(0.0), AscendC::CMPMODE::GT,
                      gSizeInner * kRealSizeAlign16);
        PipeBarrier<PIPE_V>();

        for (size_t gInnerIdx = 0; gInnerIdx < gSizeInner; gInnerIdx++) {
            PipeBarrier<PIPE_V>();
            Mul(mulLeftUb[pingPong][gInnerIdx * kRealSizeAlign16], v1TmpUb,
                reluResUb[pingPong][gInnerIdx * kRealSizeAlign16], kRealSizeAlign16);
        }
        PipeBarrier<PIPE_V>();

        // 每一块相同g轴方向的k需要Add相加，第一次可直接输出到最后的蔬菜，后面需要叠加到第一块输出上
        // 只有一次计算直接输出 Sum每一次得到gSizeInner个数
        AscendC::Sum(reduceSumResTensor[nIdx * gSizeInner], mulLeftUb[pingPong], tmpUb, sumParams);

        for (size_t gInnerIdx = 0; gInnerIdx < gSizeInner; gInnerIdx++) {
            PipeBarrier<PIPE_V>();
            Muls(mulLeftUb[pingPong][gInnerIdx * kRealSizeAlign16], v1TmpUb,
                 weightTensor.GetValue(nIdx * gSizeInner + gInnerIdx + weightOffset), kRealSizeAlign16);
        }
        PipeBarrier<PIPE_V>();

        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[pingPong]);
        ReLUGrad(reluGradUb[pingPong], mulLeftUb[pingPong], maskUb, gSizeInner * kRealSizeAlign16);
        SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[pingPong]);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        DataCopyPad(reluGradResGm[reluResOffset], reluGradUb[pingPong], dataCopyReluGradParams);

        SetFlag<HardEvent::MTE3_MTE2>(
            eventIdmte3ToMte2DwDqDkPingPong
                [pingPong]); // reluGradUb、reluResUb用的同一块buf，reluResUb获取Gm数据前需等待这里mte3搬运完
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[pingPong]);
    }

    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[0]);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[1]);
    WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[1]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[1]);

    // 只有一轮计算，计算完直接可以拷贝到GM输出
    PipeBarrier<PIPE_V>();

    if (runInfo.s2SparseLen <= 0) {
        Duplicate(dwOutTensor, static_cast<T>(0.0), gSizePerVec);
    } else {
        Adds(dwOutTensor, reduceSumResTensor, static_cast<T>(0.0), gSizePerVec);
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);

    // 拷出到GM
    DataCopyParams dataCopyParams(1, gSizePerVec * sizeof(T), 0, 1);
    if constexpr (LAYOUT_T == SLILayout::BSND) {
        DataCopyPad(dWeightGm[runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQueryIndex) +
                              runInfo.s1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset],
                    dwOutTensor, dataCopyParams);
    } else if constexpr (LAYOUT_T == SLILayout::TND) {
        DataCopyPad(dWeightGm[runInfo.accumS1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset], dwOutTensor,
                    dataCopyParams);
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorDwDqDkMoreThan2k(SLIKLLossGradRunInfo &runInfo,
                                                                            int32_t kLoopIdx)
{
    LocalTensor<T> reluResUb[2] = {pingBuf, pongBuf};
    LocalTensor<T> mulLeftUb[2] = {pingBuf, pongBuf};
    LocalTensor<T> reduceSumPUbSingleK, reduceSumYUbSingleK;
    LocalTensor<KV_T> reluGradUb[2] = {reluGradpingBuf, reluGradpongBuf};

    // maskTensor需要创建 目前一次传输32k数据对比需要8k个数，需要8k/8=1k大小空间（按bit存储）
    LocalTensor<T> maskUb = this->dwTBuf.template Get<T>()[256];
    // dwTBuf理论上最大只需要512bytes，一个元素4bytes，一共最大64个元素，开DB一共256*2=512
    LocalTensor<T> reduceSumResTensor = this->dwTBuf.template Get<T>();
    int64_t kLoopOffset = kLoopIdx * K_BASE_SIZE;
    int32_t kRealSize = (kLoopIdx == runInfo.kLoopTimes - 1) ? runInfo.kTailSize : K_BASE_SIZE;
    int32_t kRealSizeAlign16 = BlockAlign<KV_T>(kRealSize); // 按照KV_T类型对齐，datacopy才不会有问题

    SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    DataCopyExtParams pSumCopyParams(1, PSY_SYNC_BLOCK_FLOAT_NUM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
    DataCopyPadExtParams<T> pSumPadParams(false, 0, 0, 0.0f);
    if (!ShouldEnableSyLoadBalance(runInfo)) {
        if (runInfo.calcP) {
            reduceSumPUbSingleK = resPSYTBuf.Get<T>()[kLoopOffset];
            reduceSumYUbSingleK = mm2TBuf.template Get<T>()[6144];
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
            DataCopy(reduceSumYUbSingleK, psySyncOtherGm[topKSize + kLoopOffset], kRealSizeAlign16);
            DataCopyPad(reduceSumDwResUb, psySyncGm[PSY_SYNC_SUM_P_OFFSET], pSumCopyParams, pSumPadParams);
        } else {
            reduceSumPUbSingleK = mm2TBuf.template Get<T>()[6144];
            reduceSumYUbSingleK = resPSYTBuf.Get<T>()[kLoopOffset];
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
            DataCopy(reduceSumPUbSingleK, psySyncOtherGm[kLoopOffset], kRealSizeAlign16);
            DataCopyPad(reduceSumDwResUb, psySyncOtherGm[PSY_SYNC_SUM_P_OFFSET], pSumCopyParams, pSumPadParams);
        }
    } else {
        reduceSumPUbSingleK = mm2TBuf.template Get<T>()[6144];
        reduceSumYUbSingleK = resPSYTBuf.Get<T>()[kLoopOffset];
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte24SY);
        GlobalTensor<T> &pOwnerSyncGm = runInfo.calcP ? psySyncGm : psySyncOtherGm;
        DataCopy(reduceSumPUbSingleK, pOwnerSyncGm[kLoopOffset], kRealSizeAlign16);
        if (runInfo.calcP) {
            DataCopy(reduceSumYUbSingleK, psySyncOtherGm[topKSize + kLoopOffset], kRealSizeAlign16);
        }
        DataCopyPad(reduceSumDwResUb, pOwnerSyncGm[PSY_SYNC_SUM_P_OFFSET], pSumCopyParams, pSumPadParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);

    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToSPsum);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToSPsum);
    T pSumValue = reduceSumDwResUb.GetValue(0);
    SetFlag<HardEvent::S_V>(eventIdSToVPsum);
    WaitFlag<HardEvent::S_V>(eventIdSToVPsum);
    Muls(reduceSumYUbSingleK, reduceSumYUbSingleK, pSumValue, kRealSizeAlign16);
    PipeBarrier<PIPE_V>();

    Sub(v1TmpUb, reduceSumYUbSingleK, reduceSumPUbSingleK, kRealSize);
    PipeBarrier<PIPE_V>();

    // k方向单次处理最大长度为2048， g方向为4，故一次性为 4*2048*4 = 32k
    uint32_t gSizePerVec = constInfo.gSizeQueryIndex / 2;
    constexpr uint32_t gSizeInner = 4;
    int32_t gLoopTimes = gSizePerVec / gSizeInner;

    SumParams sumParams = {static_cast<uint32_t>(gSizeInner), static_cast<uint32_t>(kRealSizeAlign16),
                           static_cast<uint32_t>(kRealSize)};
    DataCopyParams dataCopyReluResParams(gSizeInner, kRealSizeAlign16 * sizeof(T),
                                         (topKSize - kRealSizeAlign16) * sizeof(T), 0);
    DataCopyPadParams dataCopyReluResPadParams(false, 0, 0, 0);

    DataCopyParams dataCopyReluGradParams(gSizeInner, kRealSize * sizeof(KV_T), 0,
                                          (topKSize - kRealSize) * sizeof(KV_T));

    int64_t subBlockGQueryIndexOffset = constInfo.subBlockIdx * gSizePerVec;
    int64_t reluGmOffset =
        (runInfo.taskIdMod2 * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset) * topKSize + kLoopOffset;
    int64_t weightOffset = runInfo.taskIdMod2 * constInfo.gSizeQueryIndexAlign16 + subBlockGQueryIndexOffset;

    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[0]);
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[1]);
    SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[1]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[0]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[1]);

    for (size_t nIdx = 0; nIdx < gLoopTimes; nIdx++) {
        uint8_t pingPong = nIdx & 1;
        maskUb = maskUb[pingPong * 256]; // 相差256个数正好1k
        WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[pingPong]);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[pingPong]);

        // 获取输入的relu数据和weights数据偏差每gSizePerVec中gSizeInner个
        int64_t reluResOffset = reluGmOffset + nIdx * gSizeInner * topKSize;
        // 每次搬运gSizeInner个kRealSizeAlign16 | reluResUb shape-> (gSizeInner, kRealSizeAlign16)
        DataCopyPad(reluResUb[pingPong], reluGm[reluResOffset], dataCopyReluResParams, dataCopyReluResPadParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInnerDwDqDk);

        CompareScalar(maskUb, reluResUb[pingPong], static_cast<T>(0.0), AscendC::CMPMODE::GT,
                      gSizeInner * kRealSizeAlign16);
        PipeBarrier<PIPE_V>();

        for (size_t gInnerIdx = 0; gInnerIdx < gSizeInner; gInnerIdx++) {
            PipeBarrier<PIPE_V>();
            Mul(mulLeftUb[pingPong][gInnerIdx * kRealSizeAlign16], v1TmpUb,
                reluResUb[pingPong][gInnerIdx * kRealSizeAlign16], kRealSizeAlign16);
        }
        PipeBarrier<PIPE_V>();

        // 每一块相同g轴方向的k需要Add相加，第一次可直接输出到最后的蔬菜，后面需要叠加到第一块输出上
        if (kLoopIdx == 0) {
            // 首轮直接输出 Sum每一次得到gSizeInner个数
            AscendC::Sum(reduceSumResTensor[nIdx * gSizeInner], mulLeftUb[pingPong], tmpUb, sumParams);
        } else {
            AscendC::Sum(reduceSumDwResUb[nIdx * gSizeInner], mulLeftUb[pingPong], tmpUb, sumParams);
        }

        for (size_t gInnerIdx = 0; gInnerIdx < gSizeInner; gInnerIdx++) {
            PipeBarrier<PIPE_V>();
            Muls(mulLeftUb[pingPong][gInnerIdx * kRealSizeAlign16], v1TmpUb,
                 weightTensor.GetValue(nIdx * gSizeInner + gInnerIdx + weightOffset), kRealSizeAlign16);
        }
        PipeBarrier<PIPE_V>();

        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[pingPong]);
        ReLUGrad(reluGradUb[pingPong], mulLeftUb[pingPong], maskUb, gSizeInner * kRealSizeAlign16);
        SetFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[pingPong]);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        DataCopyPad(reluGradResGm[reluResOffset], reluGradUb[pingPong], dataCopyReluGradParams);

        SetFlag<HardEvent::MTE3_MTE2>(
            eventIdmte3ToMte2DwDqDkPingPong
                [pingPong]); // reluGradUb、reluResUb用的同一块buf，reluResUb获取Gm数据前需等待这里mte3搬运完
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[pingPong]);
    }

    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[0]);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVDwDqDkPingPong[1]);
    WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdvToMte2DwDqDkPingPong[1]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdmte3ToMte2DwDqDkPingPong[1]);

    if (kLoopIdx > 0) {
        PipeBarrier<PIPE_V>();
        Add(reduceSumResTensor, reduceSumResTensor, reduceSumDwResUb, gSizePerVec);
    }

    if (kLoopIdx == runInfo.kLoopTimes - 1) {
        PipeBarrier<PIPE_V>();
        if (runInfo.s2SparseLen <= 0) {
            Duplicate(dwOutTensor, static_cast<T>(0.0), gSizePerVec);
        } else {
            Adds(dwOutTensor, reduceSumResTensor, static_cast<T>(0.0), gSizePerVec);
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3DwDqDk);

        // 拷出到GM
        DataCopyParams dataCopyParams(1, gSizePerVec * sizeof(T), 0, 1);
        if constexpr (LAYOUT_T == SLILayout::BSND) {
            DataCopyPad(dWeightGm[runInfo.bIdx * (constInfo.s1Size * constInfo.gSizeQueryIndex) +
                                  runInfo.s1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset],
                        dwOutTensor, dataCopyParams);
        } else if constexpr (LAYOUT_T == SLILayout::TND) {
            DataCopyPad(dWeightGm[runInfo.accumS1Idx * constInfo.gSizeQueryIndex + subBlockGQueryIndexOffset],
                        dwOutTensor, dataCopyParams);
        }
    }
}

template <typename SLIT>
template <uint32_t range>
__aicore__ inline void SLIKLLossVectorService<SLIT>::VectorDwDqDk(SLIKLLossGradRunInfo &runInfo, int32_t kLoopIdx)
{
    if constexpr (range <= SLIKLLossGradConstInfo::BUFFER_SIZE_BYTE_2K) {
        VectorDwDqDkLess2k(runInfo);
    } else {
        VectorDwDqDkMoreThan2k(runInfo, kLoopIdx);
    }
}

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_H
