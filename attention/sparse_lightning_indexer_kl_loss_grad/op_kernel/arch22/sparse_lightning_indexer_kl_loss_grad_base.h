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
 * \file sparse_lightning_indexer_kl_loss_grad_base.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_BASE_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_kl_loss_grad_common.h"
#include "sparse_lightning_indexer_kl_loss_grad_metadata_arch22.h"
#include "sparse_lightning_indexer_kl_loss_grad_tiling.h"
#include "sparse_lightning_indexer_kl_loss_grad_vector.h"
#include "sparse_lightning_indexer_kl_loss_grad_vector2.h"
#include "sparse_lightning_indexer_kl_loss_grad_service_cube.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SLIT>
class SparseLightningIndexerKLLossGradBase {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SLIT::inputQT;
    using KV_T = typename SLIT::inputKT;
    using OUT_T = typename SLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using MM3_OUT_T = T;

    static constexpr bool hasRope = SLIT::hasRope;
    static constexpr bool deterministic = SLIT::deterministic;
    static constexpr bool privateScatter = SLIT::privateScatter;
    static constexpr bool hasSequsedQ = SLIT::hasSequsedQ;
    static constexpr bool hasSequsedK = SLIT::hasSequsedK;
    static constexpr uint32_t topKSize = static_cast<uint32_t>(SLIT::topKRange);
    static constexpr SLILayout LAYOUT_T = SLIT::inputQLayout;
    static constexpr SLILayout KV_LAYOUT_T = SLIT::inputKLayout;

    __aicore__ inline SparseLightningIndexerKLLossGradBase(){};
    __aicore__ inline void Init(__gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w, __gm__ uint8_t *sparseIndices,
                                __gm__ uint8_t *attnSoftmaxL1Norm, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *metadata, __gm__ uint8_t *dq,
                                __gm__ uint8_t *dk, __gm__ uint8_t *dw, __gm__ uint8_t *softmaxOut,
                                __gm__ uint8_t *workspace,
                                const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void InitConstInfo();
    __aicore__ inline void InitBuffer(TPipe *pipe);
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline void Process();
    __aicore__ inline void MainProcess();
    __aicore__ inline void DeterProcess();
    __aicore__ inline void GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx, int64_t s1IdxEnd, int64_t accumS1Len,
                                      int64_t accumS2Len, int32_t actualSeqLensQ, int32_t actualSeqLensK,
                                      SLIKLLossGradRunInfo &runInfo);

private:
    __aicore__ inline int32_t GetActualSeqLens(int32_t bIdx, int32_t defaultLens,
                                               GlobalTensor<int32_t> &actualSeqLensGm, SLILayout layout,
                                               int64_t &accumLen);
    __aicore__ inline int32_t GetUsedSeqLens(int32_t bIdx, int32_t defaultLens, GlobalTensor<int32_t> &seqUsedGm);
    __aicore__ inline int32_t GetCmpResidualK(int32_t bIdx);
    __aicore__ inline int64_t GetPreCompressS2Len(int32_t bIdx, int32_t actualSeqLensK);
    __aicore__ inline int32_t GetS2SparseLen(int32_t bIdx, int32_t s1Idx, int32_t actualSeqLensQ,
                                             int32_t actualSeqLensK, SLISparseMode sparseMode);
    __aicore__ inline int64_t GetInvalidS1Size(int64_t bIdx, int64_t actualSeqLensQ, int64_t actualSeqLensK);
    __aicore__ inline void CalcCoreClearRange(int64_t totalSize, int64_t totalCoreNum, int64_t &clearStart,
                                              int64_t &clearEnd);
    template <typename CLEAR_T>
    __aicore__ inline void ClearInvalidS1Output(GlobalTensor<CLEAR_T> &outputGm, int64_t clearStart, int64_t clearEnd,
                                                int64_t invalidS1Base, int64_t invalidS1Size, int64_t gmS1Base,
                                                int64_t rowSize);
    __aicore__ inline void InitInvalidS1Outputs();
    __aicore__ inline int64_t FindBIndex(int64_t bIndex, int64_t curIndex, int64_t &accumulateLen);
    __aicore__ inline int64_t FindBIndexBySeqUsed(int64_t bIndex, int64_t curIndex, int64_t &accumulateLen);
    __aicore__ inline int64_t GetEndS1(int64_t bIdx);
    __aicore__ inline int64_t GetMetadataTotalSize();
    __aicore__ inline int64_t GetMetadataBS1Index(uint32_t coreIdx);
    __aicore__ inline int64_t GetEndS1Etx(int32_t bIdx, int32_t defaultLens, GlobalTensor<int32_t> &actualSeqLensGm,
                                          SLILayout layout);
    __aicore__ inline void CalcMultiCoreOffset(int64_t &bStartIdx, int64_t &s1StartIdx, int64_t &bEndIdx,
                                               int64_t &s1EndIdx);
    __aicore__ inline int64_t CalcBS1Loop();
    // 确定性：按有效 S1 压缩调度（不改 tiling totalSize）
    __aicore__ inline int64_t CalcValidTotalSize();
    __aicore__ inline void ResetValidIdxMap();
    __aicore__ inline bool LoadValidBatchAt(int64_t curB);
    __aicore__ inline bool MapValidIdxToBS1(int64_t validIdx, int64_t &bIdx, int64_t &s1Idx, int64_t &accumS1Len,
                                            int64_t &accumS2Len, int32_t &actualSeqLensQ, int32_t &actualSeqLensK);

    TPipe *pipe = nullptr;
    const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tilingData = nullptr;
    bool hasMetadata = false;
    SLIKLLossGradConstInfo constInfo;
    SLIKLLossGradRunInfo runInfos[3];

    // validIdx -> (b,s1) 游标缓存
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

    // vector and cube class
    SLIKLLossVectorService<SLIT> vectorService;
    SLIKLLossVector2Service<SLIT> vector2Service;
    SLITMatmulService<SLIT> matmulService;

    // input GM
    GlobalTensor<Q_T> queryGm, queryIndexGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<KV_T> keyGm, keyIndexGm;
    GlobalTensor<Q_ROPE_T> queryRopeGm;
    GlobalTensor<K_ROPE_T> keyRopeGm;
    GlobalTensor<T> attnSoftmaxL1NormGm;
    GlobalTensor<int32_t> topKIndexGm;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm, actualSeqLengthsKeyGm;
    GlobalTensor<int32_t> seqUsedQueryGm, seqUsedKeyGm;
    GlobalTensor<int32_t> cmpResidualKeyGm;
    GlobalTensor<int32_t> metadataGm;
    // output GM
    GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
    GlobalTensor<T> dWeightGm, softmaxOutGm;
    // workspace
    GlobalTensor<KV_T> gatherPRes, gatherSYRes;
    GlobalTensor<MM12_OUT_T> bmm1Res, bmm2Res;
    GlobalTensor<T> psySyncGm;
    GlobalTensor<KV_T> reluGradRes;
    GlobalTensor<T> scatterAddRes;
    using scatterAddGmType = typename std::conditional<deterministic, GlobalTensor<T>, int8_t>::type;
    scatterAddGmType scatterAddResBanks[SLI_DETER_SCATTER_BANK_NUM];
    using privateScatterGmType = typename std::conditional<privateScatter, GlobalTensor<T>, int8_t>::type;
    privateScatterGmType scatterAddBase;
    GlobalTensor<MM3_OUT_T> bmm3Res;
    GlobalTensor<T> reluGm;
    // local tensor
    TBuf<> gatherTbuf;
    TBuf<> mm1Tbuf;
    TBuf<> mm2TBuf; // 复用 -> mm4 scatterAdd reluGrad
};

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::Init(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *attnSoftmaxL1Norm, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *metadata, __gm__ uint8_t *dq,
    __gm__ uint8_t *dk, __gm__ uint8_t *dw, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
    const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tiling, TPipe *tPipe)
{
    // init tiling data
    pipe = tPipe;
    tilingData = tiling;

    InitConstInfo();

    // init input global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)q);
    keyGm.SetGlobalBuffer((__gm__ KV_T *)k);
    queryIndexGm.SetGlobalBuffer((__gm__ Q_T *)q);
    keyIndexGm.SetGlobalBuffer((__gm__ KV_T *)k);
    weightGm.SetGlobalBuffer((__gm__ T *)w);
    topKIndexGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    attnSoftmaxL1NormGm.SetGlobalBuffer((__gm__ T *)attnSoftmaxL1Norm);
    if constexpr (SLIT::hasRope) {
        queryRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T *)q);
        keyRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T *)k);
    }
    if (cuSeqlensQ != nullptr) {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ, constInfo.bSize + 1);
    } else {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ, 0);
    }
    if (cuSeqlensK != nullptr) {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensK, constInfo.bSize + 1);
    } else {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensK, 0);
    }
    if (cmpResidualK != nullptr) {
        cmpResidualKeyGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualK, constInfo.bSize);
    } else {
        cmpResidualKeyGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualK, 0);
    }
    if (sequsedQ != nullptr) {
        seqUsedQueryGm.SetGlobalBuffer((__gm__ int32_t *)sequsedQ, constInfo.bSize);
    } else {
        seqUsedQueryGm.SetGlobalBuffer((__gm__ int32_t *)sequsedQ, 0);
    }
    if (sequsedK != nullptr) {
        seqUsedKeyGm.SetGlobalBuffer((__gm__ int32_t *)sequsedK, constInfo.bSize);
    } else {
        seqUsedKeyGm.SetGlobalBuffer((__gm__ int32_t *)sequsedK, 0);
    }
    hasMetadata = metadata != nullptr;
    if (hasMetadata) {
        metadataGm.SetGlobalBuffer((__gm__ int32_t *)metadata, optiling::SLI_METADATA_SIZE);
    }

    // init output global buffer
    dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dq);
    dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dk);
    dWeightGm.SetGlobalBuffer((__gm__ T *)dw);
    softmaxOutGm.SetGlobalBuffer((__gm__ T *)softmaxOut);
    InitWorkspace(workspace);
    InitBuffer(pipe);

    if ASCEND_IS_AIV {
        // InitVecOP
        vectorService.InitParams(constInfo, tilingData, metadataGm, cmpResidualKeyGm, hasMetadata);
        vectorService.InitSeqUsedGM(seqUsedQueryGm, seqUsedKeyGm);
        vectorService.InitVector0GM(keyGm, keyRopeGm, keyIndexGm, topKIndexGm, actualSeqLengthsQueryGm,
                                    actualSeqLengthsKeyGm, gatherPRes, gatherSYRes);
        vectorService.InitVector1GM(attnSoftmaxL1NormGm, bmm2Res, weightGm, psySyncGm, softmaxOutGm, dWeightGm, reluGm,
                                    reluGradRes, actualSeqLengthsQueryGm, actualSeqLengthsKeyGm);
        vectorService.InitVector2GM(bmm3Res, topKIndexGm, scatterAddRes, scatterAddResBanks);
    } else if ASCEND_IS_AIC {
        // initCubeOP
        matmulService.InitParams(constInfo);

        matmulService.InitMm1GlobalTensor(queryGm, gatherPRes, queryRopeGm, actualSeqLengthsQueryGm,
                                          actualSeqLengthsKeyGm, bmm1Res, dk);
        matmulService.InitMm2GlobalTensor(queryIndexGm, gatherSYRes, bmm2Res);
        matmulService.InitMm5GlobalTensor(reluGradRes, queryIndexGm, bmm3Res, topKIndexGm);
        matmulService.InitMm6GlobalTensor(reluGradRes, gatherSYRes, dQueryIndexGm);
    }
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::InitConstInfo()
{
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx(); // vec:0-47
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % 2;
    } else {
        constInfo.aicIdx = GetBlockIdx(); // cube:0-23
    }

    auto &baseInfo = tilingData->baseParams;
    constInfo.bSize = baseInfo.bSize;
    constInfo.n2Size = baseInfo.n2Size;
    constInfo.gSizeQuery = baseInfo.gSizeQuery;
    constInfo.gSizeQueryIndex = baseInfo.gSizeQueryIndex;
    constInfo.s1Size = baseInfo.s1Size;
    constInfo.s2Size = baseInfo.s2Size;
    constInfo.kSize = baseInfo.kSize;

    constInfo.dSizeQuery = baseInfo.dSizeQuery;
    constInfo.dSizeQueryIndex = baseInfo.dSizeQueryIndex;
    constInfo.gSizeQueryIndexAlign16 = ((constInfo.gSizeQueryIndex + 15) / 16) * 16;
    constInfo.sparseMode = static_cast<SLISparseMode>(baseInfo.sparseMode);
    constInfo.scaleValue = baseInfo.scaleValue;
    constInfo.cmpRatio = baseInfo.cmpRatio;
    constInfo.hasSoftmaxInput = baseInfo.hasSoftmaxInput;
    constInfo.gatherKeySize = topKSize * (constInfo.dSizeQuery + constInfo.dSizeRope);
    constInfo.gatherKeyIndexSize = topKSize * constInfo.dSizeQueryIndex;
    if constexpr (!SLIT::hasRope) {
        constInfo.dSizeQueryRope = 0;
        constInfo.gatherKeySize = topKSize * (constInfo.dSizeQuery);
    }
    constInfo.tilingInfo = tilingData->vectorParams.softmaxYTilingData;
    constInfo.simpleSoftMaxTilingInfo = tilingData->vectorParams.simpleSoftmaxPTilingData;
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::InitWorkspace(__gm__ uint8_t *workspace)
{
    int64_t pOffset = topKSize * (constInfo.dSizeQuery + constInfo.dSizeQueryRope) * sizeof(KV_T); // * 2;
    int64_t syOffset = topKSize * constInfo.dSizeQueryIndex * sizeof(KV_T);                        // * 2;
    int64_t bmm1Offset = constInfo.gSizeQuery * topKSize * sizeof(float);                          // * 2;
    int64_t psySyncSize = (topKSize * 2 + 32 / sizeof(float)) * sizeof(float);
    int64_t bmm2Offset = constInfo.gSizeQueryIndex * topKSize * sizeof(float);     // * 2;
    int64_t reluGradOffset = constInfo.gSizeQueryIndex * topKSize * sizeof(float); // * 2;
    int64_t bmm3Offset = topKSize * constInfo.dSizeQueryIndex * sizeof(float);     // * 2;
    int64_t scatterAddOffset;
    int64_t bS2Len = 0;
    if constexpr (LAYOUT_T == SLILayout::TND) {
        bS2Len = actualSeqLengthsKeyGm.GetValue(constInfo.bSize);
    } else {
        bS2Len = constInfo.bSize * constInfo.s2Size;
    }
    scatterAddOffset = bS2Len * constInfo.dSizeQueryIndex * sizeof(T);
    int64_t scatterAddBankElems = bS2Len * constInfo.dSizeQueryIndex;

    int64_t coreTotalOffset =
        constInfo.aicIdx * (pOffset * constInfo.gatherKeyDbNum + syOffset * constInfo.gatherKeyIndexDbNum +
                            bmm1Offset * 2 + bmm2Offset * 2 + reluGradOffset * 2 + psySyncSize * 2);

    int64_t totalOffset =
        GetBlockNum() * (pOffset * constInfo.gatherKeyDbNum + syOffset * constInfo.gatherKeyIndexDbNum +
                         bmm1Offset * 2 + bmm2Offset * 2 + reluGradOffset * 2 + psySyncSize * 2);

    uint64_t offset = 0;
    // workspace 按核分, 每个核内不同workspace相邻
    gatherPRes.SetGlobalBuffer((__gm__ KV_T *)(workspace + offset + coreTotalOffset));
    offset += pOffset * constInfo.gatherKeyDbNum;

    gatherSYRes.SetGlobalBuffer((__gm__ KV_T *)(workspace + offset + coreTotalOffset));
    offset += syOffset * constInfo.gatherKeyIndexDbNum;

    bmm1Res.SetGlobalBuffer((__gm__ MM12_OUT_T *)(workspace + offset + coreTotalOffset));
    offset += bmm1Offset * 2;

    psySyncGm.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += psySyncSize * 2;

    bmm2Res.SetGlobalBuffer((__gm__ MM12_OUT_T *)(workspace + offset + coreTotalOffset));
    reluGm.SetGlobalBuffer((__gm__ T *)(bmm2Res.GetPhyAddr()));
    offset += bmm2Offset * 2;

    reluGradRes.SetGlobalBuffer((__gm__ OUT_T *)(workspace + offset + coreTotalOffset));
    offset += reluGradOffset * 2;

    bmm3Res.SetGlobalBuffer((__gm__ MM3_OUT_T *)(workspace + totalOffset));
    totalOffset += bmm3Offset * GetBlockNum() * 2;

    if constexpr (deterministic && privateScatter) {
        __gm__ uint8_t *scatterBasePtr = workspace + totalOffset;
        int32_t scatterBankNum = static_cast<int32_t>(GetBlockNum());
        if (scatterBankNum <= 0) {
            scatterBankNum = 1;
        }
        scatterAddBase.SetGlobalBuffer((__gm__ T *)scatterBasePtr);
        int64_t myBank = 0;
        if ASCEND_IS_AIV {
            myBank = static_cast<int64_t>(constInfo.aicIdx);
            if (myBank >= scatterBankNum) {
                myBank = 0;
            }
        }
        scatterAddRes.SetGlobalBuffer((__gm__ T *)(scatterBasePtr + static_cast<uint64_t>(myBank) * scatterAddOffset));
        totalOffset += scatterAddOffset * scatterBankNum;
    } else if constexpr (deterministic) {
        __gm__ uint8_t *scatterBase = workspace + totalOffset;
        for (int32_t bank = 0; bank < SLI_DETER_SCATTER_BANK_NUM; ++bank) {
            scatterAddResBanks[bank].SetGlobalBuffer(
                (__gm__ T *)(scatterBase + static_cast<uint64_t>(bank) * scatterAddOffset));
        }
        totalOffset += scatterAddOffset * SLI_DETER_SCATTER_BANK_NUM;
        scatterAddRes.SetGlobalBuffer((__gm__ T *)scatterBase);
    } else {
        scatterAddRes.SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
    }
    if ASCEND_IS_AIV {
        int64_t totalCost = 0;
        int64_t totalCostQ = 0;

        if constexpr (KV_LAYOUT_T == SLILayout::TND) {
            totalCostQ = actualSeqLengthsQueryGm.GetValue(constInfo.bSize);
            totalCost = actualSeqLengthsKeyGm.GetValue(constInfo.bSize);
        } else {
            totalCostQ = constInfo.bSize * constInfo.s1Size;
            totalCost = constInfo.bSize * constInfo.s2Size;
        }

        int64_t totalCoreNum = GetBlockNum() * GetTaskRation();
        int64_t avgCost = CeilDiv(totalCost, totalCoreNum);
        int64_t avgCostQ = CeilDiv(totalCostQ, totalCoreNum);
        int32_t t2Start = Min(constInfo.aivIdx * avgCost, totalCost);
        int32_t t2End = Min(t2Start + avgCost, totalCost);
        int32_t t2StartQ = Min(constInfo.aivIdx * avgCostQ, totalCostQ);
        int32_t t2EndQ = Min(t2StartQ + avgCostQ, totalCostQ);
        int64_t qBaseOffset = constInfo.gSizeQuery * constInfo.dSizeQuery;

        if constexpr (hasSequsedQ || hasSequsedK) {
            int64_t dwRowSize = static_cast<int64_t>(constInfo.n2Size) * constInfo.gSizeQueryIndex;
            AscendC::InitOutput(dQueryIndexGm[t2StartQ * qBaseOffset], qBaseOffset * (t2EndQ - t2StartQ),
                                static_cast<OUT_T>(0));
            AscendC::InitOutput(dWeightGm[t2StartQ * dwRowSize], dwRowSize * (t2EndQ - t2StartQ), static_cast<T>(0));
            AscendC::InitOutput(softmaxOutGm[t2StartQ * constInfo.n2Size * topKSize],
                                constInfo.n2Size * topKSize * (t2EndQ - t2StartQ), static_cast<T>(0));
        }
        if constexpr (deterministic && privateScatter) {
            int32_t scatterBankNum = static_cast<int32_t>(GetBlockNum());
            if (scatterBankNum <= 0) {
                scatterBankNum = 1;
            }
            int64_t totalElems = static_cast<int64_t>(scatterBankNum) * scatterAddBankElems;
            scatterAddBase.SetGlobalBuffer((__gm__ T *)(workspace + totalOffset - scatterAddOffset * scatterBankNum),
                                           totalElems);
            int64_t avgElems = CeilDiv(totalElems, totalCoreNum);
            int64_t elemStart = Min(static_cast<int64_t>(constInfo.aivIdx) * avgElems, totalElems);
            int64_t elemCnt = Min(avgElems, totalElems - elemStart);
            if (elemCnt > 0) {
                AscendC::InitOutput(scatterAddBase[elemStart], elemCnt, static_cast<T>(0));
            }
        } else {
            AscendC::InitOutput(scatterAddRes[t2Start * constInfo.dSizeQueryIndex],
                                constInfo.dSizeQueryIndex * (t2End - t2Start), static_cast<T>(0));
            if constexpr (deterministic) {
                for (int32_t bank = 1; bank < SLI_DETER_SCATTER_BANK_NUM; ++bank) {
                    AscendC::InitOutput(scatterAddResBanks[bank][t2Start * constInfo.dSizeQueryIndex],
                                        constInfo.dSizeQueryIndex * (t2End - t2Start), static_cast<T>(0));
                }
            }
        }
    }
    SyncAll();
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetInvalidS1Size(int64_t bIdx,
                                                                                       int64_t actualSeqLensQ,
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
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::CalcValidTotalSize()
{
    if (cachedValidTotalSize_ >= 0) {
        return cachedValidTotalSize_;
    }
    int64_t validTotalSize = 0;
    if constexpr (LAYOUT_T == SLILayout::TND) {
        for (int64_t bIdx = 0; bIdx < constInfo.bSize; ++bIdx) {
            int64_t accumS1Len = 0;
            int64_t accumS2Len = 0;
            int32_t actualSeqLensQ =
                GetActualSeqLens(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, accumS1Len);
            int32_t actualSeqLensK = 0;
            if constexpr (KV_LAYOUT_T == SLILayout::TND) {
                actualSeqLensK =
                    GetActualSeqLens(bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, accumS2Len);
            } else {
                actualSeqLensK = constInfo.s2Size;
            }
            int64_t invalidS1Size = GetInvalidS1Size(bIdx, actualSeqLensQ, actualSeqLensK);
            validTotalSize += Max(static_cast<int64_t>(actualSeqLensQ) - invalidS1Size, static_cast<int64_t>(0));
        }
    } else {
        for (int64_t bIdx = 0; bIdx < constInfo.bSize; ++bIdx) {
            int64_t invalidS1Size = GetInvalidS1Size(bIdx, constInfo.s1Size, constInfo.s2Size);
            validTotalSize += Max(constInfo.s1Size - invalidS1Size, static_cast<int64_t>(0));
        }
    }
    cachedValidTotalSize_ = validTotalSize;
    return validTotalSize;
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::ResetValidIdxMap()
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
__aicore__ inline bool SparseLightningIndexerKLLossGradBase<SLIT>::LoadValidBatchAt(int64_t curB)
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
__aicore__ inline bool SparseLightningIndexerKLLossGradBase<SLIT>::MapValidIdxToBS1(int64_t validIdx, int64_t &bIdx,
                                                                                    int64_t &s1Idx, int64_t &accumS1Len,
                                                                                    int64_t &accumS2Len,
                                                                                    int32_t &actualSeqLensQ,
                                                                                    int32_t &actualSeqLensK)
{
    if (validIdx < 0) {
        return false;
    }
    if (cachedValidTotalSize_ < 0) {
        CalcValidTotalSize();
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
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::CalcCoreClearRange(int64_t totalSize,
                                                                                      int64_t totalCoreNum,
                                                                                      int64_t &clearStart,
                                                                                      int64_t &clearEnd)
{
    int64_t singleCoreSize = CeilDiv(totalSize, totalCoreNum);
    clearStart = Min(static_cast<int64_t>(constInfo.aivIdx) * singleCoreSize, totalSize);
    clearEnd = Min(clearStart + singleCoreSize, totalSize);
}

template <typename SLIT>
template <typename CLEAR_T>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::ClearInvalidS1Output(
    GlobalTensor<CLEAR_T> &outputGm, int64_t clearStart, int64_t clearEnd, int64_t invalidS1Base, int64_t invalidS1Size,
    int64_t gmS1Base, int64_t rowSize)
{
    int64_t segmentStart = invalidS1Base * rowSize;
    int64_t segmentEnd = segmentStart + invalidS1Size * rowSize;
    int64_t localStart = Max(clearStart, segmentStart);
    int64_t localEnd = Min(clearEnd, segmentEnd);
    if (localStart < localEnd) {
        int64_t gmOffset = gmS1Base * rowSize + localStart - segmentStart;
        AscendC::InitOutput(outputGm[gmOffset], localEnd - localStart, static_cast<CLEAR_T>(0));
    }
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::InitInvalidS1Outputs()
{
    if constexpr (deterministic) {
        int64_t totalCoreNum = GetBlockNum() * GetTaskRation();
        int64_t qRowSize = constInfo.gSizeQuery * constInfo.dSizeQuery;
        int64_t dwRowSize = constInfo.gSizeQueryIndex;
        int64_t softmaxRowSize = constInfo.n2Size * topKSize;
        int64_t validTotalSize = 0;

        if constexpr (LAYOUT_T == SLILayout::TND) {
            for (int64_t bIdx = 0; bIdx < constInfo.bSize; ++bIdx) {
                int64_t accumS1Len = 0;
                int64_t accumS2Len = 0;
                int64_t actualSeqLensQ =
                    GetActualSeqLens(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, accumS1Len);
                int64_t actualSeqLensK = 0;
                if constexpr (KV_LAYOUT_T == SLILayout::TND) {
                    actualSeqLensK =
                        GetActualSeqLens(bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, accumS2Len);
                } else {
                    actualSeqLensK = constInfo.s2Size;
                }
                int64_t invalidS1Size = GetInvalidS1Size(bIdx, actualSeqLensQ, actualSeqLensK);
                validTotalSize += Max(actualSeqLensQ - invalidS1Size, static_cast<int64_t>(0));
                if ASCEND_IS_AIV {
                    int64_t dqClearStart = 0;
                    int64_t dqClearEnd = 0;
                    int64_t dwClearStart = 0;
                    int64_t dwClearEnd = 0;
                    int64_t softmaxClearStart = 0;
                    int64_t softmaxClearEnd = 0;
                    CalcCoreClearRange(invalidS1Size * qRowSize, totalCoreNum, dqClearStart, dqClearEnd);
                    CalcCoreClearRange(invalidS1Size * dwRowSize, totalCoreNum, dwClearStart, dwClearEnd);
                    CalcCoreClearRange(invalidS1Size * softmaxRowSize, totalCoreNum, softmaxClearStart,
                                       softmaxClearEnd);
                    ClearInvalidS1Output(dQueryIndexGm, dqClearStart, dqClearEnd, 0, invalidS1Size, accumS1Len,
                                         qRowSize);
                    ClearInvalidS1Output(dWeightGm, dwClearStart, dwClearEnd, 0, invalidS1Size, accumS1Len, dwRowSize);
                    ClearInvalidS1Output(softmaxOutGm, softmaxClearStart, softmaxClearEnd, 0, invalidS1Size, accumS1Len,
                                         softmaxRowSize);
                }
            }
        } else {
            for (int64_t bIdx = 0; bIdx < constInfo.bSize; ++bIdx) {
                int64_t invalidS1Size = GetInvalidS1Size(bIdx, constInfo.s1Size, constInfo.s2Size);
                validTotalSize += Max(constInfo.s1Size - invalidS1Size, static_cast<int64_t>(0));
                if ASCEND_IS_AIV {
                    int64_t batchS1Base = bIdx * constInfo.s1Size;
                    int64_t dqClearStart = 0;
                    int64_t dqClearEnd = 0;
                    int64_t dwClearStart = 0;
                    int64_t dwClearEnd = 0;
                    int64_t softmaxClearStart = 0;
                    int64_t softmaxClearEnd = 0;
                    CalcCoreClearRange(invalidS1Size * qRowSize, totalCoreNum, dqClearStart, dqClearEnd);
                    CalcCoreClearRange(invalidS1Size * dwRowSize, totalCoreNum, dwClearStart, dwClearEnd);
                    CalcCoreClearRange(invalidS1Size * softmaxRowSize, totalCoreNum, softmaxClearStart,
                                       softmaxClearEnd);
                    ClearInvalidS1Output(dQueryIndexGm, dqClearStart, dqClearEnd, 0, invalidS1Size, batchS1Base,
                                         qRowSize);
                    ClearInvalidS1Output(dWeightGm, dwClearStart, dwClearEnd, 0, invalidS1Size, batchS1Base, dwRowSize);
                    ClearInvalidS1Output(softmaxOutGm, softmaxClearStart, softmaxClearEnd, 0, invalidS1Size,
                                         batchS1Base, softmaxRowSize);
                }
            }
        }
        cachedValidTotalSize_ = validTotalSize;
    } else if ASCEND_IS_AIV {
        int64_t totalCoreNum = GetBlockNum() * GetTaskRation();
        int64_t qRowSize = constInfo.gSizeQuery * constInfo.dSizeQuery;
        int64_t dwRowSize = static_cast<int64_t>(constInfo.n2Size) * constInfo.gSizeQueryIndex;
        int64_t softmaxRowSize = constInfo.n2Size * topKSize;
        int64_t totalInvalidS1Size = 0;

        if constexpr (LAYOUT_T == SLILayout::TND) {
            for (int64_t bIdx = 0; bIdx < constInfo.bSize; ++bIdx) {
                int64_t accumS1Len = 0;
                int64_t accumS2Len = 0;
                int64_t actualSeqLensQ =
                    GetActualSeqLens(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, accumS1Len);
                int64_t actualSeqLensK = 0;
                if constexpr (KV_LAYOUT_T == SLILayout::TND) {
                    actualSeqLensK =
                        GetActualSeqLens(bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, accumS2Len);
                } else {
                    actualSeqLensK = constInfo.s2Size;
                }
                totalInvalidS1Size += GetInvalidS1Size(bIdx, actualSeqLensQ, actualSeqLensK);
            }
        } else if constexpr (LAYOUT_T == SLILayout::BSND) {
            for (int64_t bIdx = 0; bIdx < constInfo.bSize; ++bIdx) {
                totalInvalidS1Size += GetInvalidS1Size(bIdx, constInfo.s1Size, constInfo.s2Size);
            }
        }

        int64_t dqClearStart = 0;
        int64_t dqClearEnd = 0;
        int64_t dwClearStart = 0;
        int64_t dwClearEnd = 0;
        int64_t softmaxClearStart = 0;
        int64_t softmaxClearEnd = 0;
        CalcCoreClearRange(totalInvalidS1Size * qRowSize, totalCoreNum, dqClearStart, dqClearEnd);
        CalcCoreClearRange(totalInvalidS1Size * dwRowSize, totalCoreNum, dwClearStart, dwClearEnd);
        CalcCoreClearRange(totalInvalidS1Size * softmaxRowSize, totalCoreNum, softmaxClearStart, softmaxClearEnd);

        if constexpr (LAYOUT_T == SLILayout::TND) {
            int64_t invalidS1Base = 0;
            for (int64_t bIdx = 0; bIdx < constInfo.bSize && invalidS1Base < totalInvalidS1Size; ++bIdx) {
                int64_t accumS1Len = 0;
                int64_t accumS2Len = 0;
                int64_t actualSeqLensQ =
                    GetActualSeqLens(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, accumS1Len);
                int64_t actualSeqLensK = 0;
                if constexpr (KV_LAYOUT_T == SLILayout::TND) {
                    actualSeqLensK =
                        GetActualSeqLens(bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, accumS2Len);
                } else {
                    actualSeqLensK = constInfo.s2Size;
                }
                int64_t invalidS1Size = GetInvalidS1Size(bIdx, actualSeqLensQ, actualSeqLensK);
                ClearInvalidS1Output(dQueryIndexGm, dqClearStart, dqClearEnd, invalidS1Base, invalidS1Size, accumS1Len,
                                     qRowSize);
                ClearInvalidS1Output(dWeightGm, dwClearStart, dwClearEnd, invalidS1Base, invalidS1Size, accumS1Len,
                                     dwRowSize);
                ClearInvalidS1Output(softmaxOutGm, softmaxClearStart, softmaxClearEnd, invalidS1Base, invalidS1Size,
                                     accumS1Len, softmaxRowSize);
                invalidS1Base += invalidS1Size;
            }
        } else if constexpr (LAYOUT_T == SLILayout::BSND) {
            int64_t invalidS1Base = 0;
            for (int64_t bIdx = 0; bIdx < constInfo.bSize && invalidS1Base < totalInvalidS1Size; ++bIdx) {
                int64_t invalidS1Size = GetInvalidS1Size(bIdx, constInfo.s1Size, constInfo.s2Size);
                int64_t batchS1Base = bIdx * constInfo.s1Size;

                ClearInvalidS1Output(dQueryIndexGm, dqClearStart, dqClearEnd, invalidS1Base, invalidS1Size, batchS1Base,
                                     qRowSize);
                ClearInvalidS1Output(dWeightGm, dwClearStart, dwClearEnd, invalidS1Base, invalidS1Size, batchS1Base,
                                     dwRowSize);
                ClearInvalidS1Output(softmaxOutGm, softmaxClearStart, softmaxClearEnd, invalidS1Base, invalidS1Size,
                                     batchS1Base, softmaxRowSize);
                invalidS1Base += invalidS1Size;
            }
        }
    }
    SyncAll();
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::InitBuffer(TPipe *pipe)
{
    if ASCEND_IS_AIC {
        matmulService.InitBuffers(pipe);
    } else {
        vectorService.InitBuffers(pipe);
    }
}
template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::FindBIndex(int64_t bIndex, int64_t curIndex,
                                                                                 int64_t &accumulateLen)
{
    for (int index = bIndex; index < constInfo.bSize; index++) {
        int64_t actualLen = this->actualSeqLengthsQueryGm.GetValue(index + 1);
        if (curIndex < actualLen) {
            return index;
        }
        accumulateLen = actualLen;
    }
    return GetMetadataTotalSize() >= curIndex ? constInfo.bSize : -1;
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::FindBIndexBySeqUsed(int64_t bIndex,
                                                                                          int64_t curIndex,
                                                                                          int64_t &accumulateLen)
{
    int64_t usedAccum = accumulateLen;
    for (int index = bIndex; index < constInfo.bSize; index++) {
        int64_t usedLen = GetUsedSeqLens(index, constInfo.s1Size, seqUsedQueryGm);
        if constexpr (LAYOUT_T == SLILayout::TND) {
            int64_t allocAccum = 0;
            usedLen = GetUsedSeqLens(
                index, GetActualSeqLens(index, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, allocAccum),
                seqUsedQueryGm);
        }
        if (curIndex < usedAccum + usedLen) {
            accumulateLen = usedAccum;
            return index;
        }
        usedAccum += usedLen;
    }
    accumulateLen = usedAccum;
    return GetMetadataTotalSize() > curIndex ? constInfo.bSize : -1;
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetEndS1(int64_t bIdx)
{
    int64_t end =
        constInfo.aicIdx + 1 < optiling::MAX_CORE_NUM ? GetMetadataBS1Index(bIdx + 1) : GetMetadataTotalSize();
    return end - GetMetadataBS1Index(bIdx);
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetMetadataTotalSize()
{
    return hasMetadata ? metadataGm.GetValue(optiling::SLI_META_TOTAL_SIZE_INDEX) :
                         tilingData->multiCoreParams.totalSize;
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetMetadataBS1Index(uint32_t coreIdx)
{
    if (hasMetadata) {
        return metadataGm.GetValue(optiling::GetSliMetaBS1IndexAttr(coreIdx));
    }
    return tilingData->multiCoreParams.bS1Index[coreIdx];
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetEndS1Etx(
    int32_t bIdx, int32_t defaultLens, GlobalTensor<int32_t> &actualSeqLensGm, SLILayout layout)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (layout == SLILayout::TND) {
        return actualSeqLensGm.GetValue(bIdx + 1) - actualSeqLensGm.GetValue(bIdx);
    } else {
        assert(false, "do not support current layout!\n");
        return 0;
    }
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::CalcMultiCoreOffset(int64_t &bStartIdx,
                                                                                       int64_t &s1StartIdx,
                                                                                       int64_t &bEndIdx,
                                                                                       int64_t &s1EndIdx)
{
    int64_t actualSum = 0;
    int64_t bS1Index = GetMetadataBS1Index(constInfo.aicIdx);
    int64_t bS1EndIndex = constInfo.aicIdx + 1 < optiling::MAX_CORE_NUM ? GetMetadataBS1Index(constInfo.aicIdx + 1) :
                                                                          GetMetadataTotalSize();
    if (bS1Index >= bS1EndIndex || bS1Index >= GetMetadataTotalSize()) {
        bStartIdx = 1;
        bEndIdx = 0;
        s1StartIdx = 0;
        s1EndIdx = 0;
        return;
    }
    if constexpr (LAYOUT_T == SLILayout::TND) {
        if constexpr (hasSequsedQ) {
            bStartIdx = FindBIndexBySeqUsed(0, bS1Index, actualSum);
            s1StartIdx = bS1Index - actualSum;
            bEndIdx = FindBIndexBySeqUsed(bStartIdx, bS1EndIndex - 1, actualSum);
            s1EndIdx = bS1EndIndex - actualSum;
        } else {
            bStartIdx = FindBIndex(0, bS1Index, actualSum);
            s1StartIdx = bS1Index - actualSum;
            bEndIdx = FindBIndex(bStartIdx, bS1EndIndex - 1, actualSum);
            s1EndIdx = bS1EndIndex - actualSum;
        }
    } else {
        if constexpr (hasSequsedQ) {
            bStartIdx = FindBIndexBySeqUsed(0, bS1Index, actualSum);
            s1StartIdx = bS1Index - actualSum;
            bEndIdx = FindBIndexBySeqUsed(bStartIdx, bS1EndIndex - 1, actualSum);
            s1EndIdx = bS1EndIndex - actualSum;
        } else {
            bStartIdx = bS1Index / constInfo.s1Size;
            bEndIdx = (bS1EndIndex - 1) / constInfo.s1Size;
            s1StartIdx = bS1Index - bStartIdx * constInfo.s1Size;
            s1EndIdx = bS1EndIndex - bEndIdx * constInfo.s1Size;
        }
    }
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::CalcBS1Loop()
{
    int64_t maxLoop = 0;
    int32_t coreNum = GetBlockNum();
    int64_t bS1Index, bS1EndIndex;
    for (int32_t aicIdx = 0; aicIdx < coreNum; aicIdx++) {
        bS1Index = GetMetadataBS1Index(aicIdx);
        bS1EndIndex = aicIdx + 1 < optiling::MAX_CORE_NUM ? GetMetadataBS1Index(aicIdx + 1) : GetMetadataTotalSize();
        maxLoop = Max(maxLoop, bS1EndIndex - bS1Index);
    }
    return maxLoop;
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::Process()
{
    if constexpr (deterministic) {
        DeterProcess();
    } else {
        MainProcess();
    }
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::DeterProcess()
{
    InitInvalidS1Outputs();

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }
    int64_t coreNum = GetBlockNum();
    int64_t bS1TotalSize = CalcValidTotalSize();
    if ASCEND_IS_AIV {
        vectorService.SetCachedValidTotalSize(bS1TotalSize);
    }
    ResetValidIdxMap();
    int64_t extraLoopTimes = 2;
    int64_t bIdx = 0;
    int64_t s1Idx = 0;
    int64_t taskId = 0;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int32_t actualSeqLensQ = 0;
    int32_t actualSeqLensK = 0;

    if ASCEND_IS_AIV {
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC_DETER_CREDIT_FLAG);
    }

    const int64_t loopLimit = bS1TotalSize + extraLoopTimes * coreNum;
    for (int64_t bS1Idx = constInfo.aicIdx; bS1Idx < loopLimit; bS1Idx += coreNum) {
        SLIKLLossGradRunInfo &runInfoNeg2 = runInfos[(taskId + 1) % 3]; // 上2轮
        SLIKLLossGradRunInfo &runInfoNeg1 = runInfos[(taskId + 2) % 3]; // 上1轮
        SLIKLLossGradRunInfo &runInfo0 = runInfos[taskId % 3];          // 当前轮
        const bool hasNextRound = (bS1Idx + coreNum < loopLimit);

        if (MapValidIdxToBS1(bS1Idx, bIdx, s1Idx, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK)) {
            GetRunInfo(taskId, bIdx, s1Idx, s1Idx + 1, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK,
                       runInfo0);
        } else {
            runInfo0.isValid = false;
        }

        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC_DETER_CREDIT_FLAG);
        }

        if (taskId >= 2) {
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_AIC_TO_AIV_DETER_BARRIER_FLAG);
            }
        }

        if (runInfo0.isValid) {
            if ASCEND_IS_AIV {
                vectorService.ProcessVector0(runInfo0); // V0
            }
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_TO_AIC_DETER_CREDIT_FLAG);
        }

        if (runInfoNeg1.isValid) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_TO_V1_P_FLAG[runInfoNeg1.taskIdMod2]);
                matmulService.ComputeMm2(runInfoNeg1); // C1
            }

            if ASCEND_IS_AIV {
                vectorService.ProcessVector1(runInfoNeg1); // V1
            }
            if ASCEND_IS_AIC {
                matmulService.ComputeMm5(runInfoNeg1); // C2
                matmulService.ComputeMm6(runInfoNeg1); // C2
            }
        } else if (taskId >= 1) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_TO_V2_SA_FLAG[(taskId - 1) & 1]);
            }
        }

        if (taskId >= 1 && hasNextRound) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIC_TO_AIV_DETER_BARRIER_FLAG);
            }
        }

        if (taskId >= 2) {
            if ASCEND_IS_AIV {
                runInfoNeg2.taskId = taskId - 2;
                runInfoNeg2.taskIdMod2 = runInfoNeg2.taskId & 1;
                if constexpr (privateScatter) {
                    vectorService.ProcessPrivateScatterVector2(runInfoNeg2);
                } else {
                    vectorService.ProcessDeterVector2(runInfoNeg2);
                }
                runInfoNeg2.isValid = false;
            }
        }
        taskId++;
    }

    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC_DETER_CREDIT_FLAG);
    }

    if (constInfo.aicIdx + 1 > bS1TotalSize % coreNum) {
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_TO_V2_SA_FLAG[(taskId - extraLoopTimes) & 1]);
        }
        if ASCEND_IS_AIV {
            SLIKLLossGradRunInfo runInfo;
            runInfo.taskId = taskId - extraLoopTimes;
            runInfo.taskIdMod2 = runInfo.taskId & 1;
            if constexpr (privateScatter) {
                vectorService.ProcessPrivateScatterVector2(runInfo);
            } else {
                vectorService.ProcessDeterVector2(runInfo);
            }
        }
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
    }

    if ASCEND_IS_AIV {
        vector2Service.InitParams(constInfo, tilingData);
        if constexpr (privateScatter) {
            vector2Service.InitVector2GM(scatterAddBase, topKIndexGm, dKeyIndexGm, actualSeqLengthsQueryGm,
                                         actualSeqLengthsKeyGm, scatterAddResBanks);
        } else {
            vector2Service.InitVector2GM(scatterAddRes, topKIndexGm, dKeyIndexGm, actualSeqLengthsQueryGm,
                                         actualSeqLengthsKeyGm, scatterAddResBanks);
        }
        vector2Service.InitBuffers(pipe);
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        vector2Service.AllocEventID();
        vector2Service.ProcessVector2();
        vector2Service.FreeEventID();
    }
}

template <typename SLIT>
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::MainProcess()
{
    InitInvalidS1Outputs();

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    int64_t bStartIdx, s1StartIdx, bEndIdx, s1EndIdx;
    CalcMultiCoreOffset(bStartIdx, s1StartIdx, bEndIdx, s1EndIdx);

    int64_t taskId = 0;
    int64_t extraLoopTimes = 0;
    for (int64_t bIdx = bStartIdx; bIdx <= bEndIdx; bIdx++) {
        bool lastB = (bIdx == bEndIdx);
        int64_t s1StartIdxThisBatch = 0;
        int64_t s1EndIdxThisBatch = 0;

        int64_t accumS1Len = 0;
        int64_t accumS2Len = 0;
        int32_t actualSeqLensQ = 0;
        int32_t actualSeqLensK = 0;
        if constexpr (LAYOUT_T == SLILayout::TND) {
            int32_t allocatedSeqLensQ =
                GetActualSeqLens(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, accumS1Len);
            actualSeqLensQ = hasSequsedQ ? GetUsedSeqLens(bIdx, allocatedSeqLensQ, seqUsedQueryGm) : allocatedSeqLensQ;
            if constexpr (KV_LAYOUT_T == SLILayout::TND) {
                int32_t allocatedSeqLensK =
                    GetActualSeqLens(bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, accumS2Len);
                actualSeqLensK =
                    hasSequsedK ? GetUsedSeqLens(bIdx, allocatedSeqLensK, seqUsedKeyGm) : allocatedSeqLensK;
            } else {
                accumS2Len = bIdx * constInfo.s2Size;
                actualSeqLensK = hasSequsedK ? GetUsedSeqLens(bIdx, constInfo.s2Size, seqUsedKeyGm) : constInfo.s2Size;
            }
            s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
            s1EndIdxThisBatch = (!lastB) ? actualSeqLensQ : s1EndIdx;
        } else if constexpr (LAYOUT_T == SLILayout::BSND) {
            s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
            actualSeqLensQ = hasSequsedQ ? GetUsedSeqLens(bIdx, constInfo.s1Size, seqUsedQueryGm) : constInfo.s1Size;
            actualSeqLensK = hasSequsedK ? GetUsedSeqLens(bIdx, constInfo.s2Size, seqUsedKeyGm) : constInfo.s2Size;
            s1EndIdxThisBatch = (!lastB) ? actualSeqLensQ : s1EndIdx;
        }
        int64_t invalidS1Size = GetInvalidS1Size(bIdx, actualSeqLensQ, actualSeqLensK);
        s1StartIdxThisBatch = Min(Max(s1StartIdxThisBatch, invalidS1Size), s1EndIdxThisBatch);
        if (lastB) {
            extraLoopTimes = 2; // 最后一个Batch需要额外循环两次，因为preload方式会产生尾巴
        }

        for (int64_t s1Idx = s1StartIdxThisBatch; s1Idx < s1EndIdxThisBatch + extraLoopTimes; s1Idx++) {
            SLIKLLossGradRunInfo &runInfoNeg2 = runInfos[(taskId + 1) % 3]; // 上2轮
            SLIKLLossGradRunInfo &runInfoNeg1 = runInfos[(taskId + 2) % 3]; // 上1轮
            SLIKLLossGradRunInfo &runInfo0 = runInfos[taskId % 3];          // 当前轮

            GetRunInfo(taskId, bIdx, s1Idx, s1EndIdxThisBatch, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK,
                       runInfo0);
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<2, PIPE_MTE3>(14);
            } else {
                CrossCoreSetFlag<2, PIPE_MTE2>(14);
            }

            if (runInfo0.isValid) {
                if ASCEND_IS_AIV {
                    vectorService.ProcessVector0(runInfo0); // V0
                }
            }

            if (runInfoNeg1.isValid) {
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_TO_V1_P_FLAG[runInfoNeg1.taskIdMod2]);
                    matmulService.ComputeMm2(runInfoNeg1); // C1
                }

                if ASCEND_IS_AIV {
                    vectorService.ProcessVector1(runInfoNeg1); // V1
                }
                if ASCEND_IS_AIC {
                    matmulService.ComputeMm5(runInfoNeg1); // C2
                    matmulService.ComputeMm6(runInfoNeg1); // C2
                }
            }

            if (runInfoNeg2.isValid) {
                if ASCEND_IS_AIV {
                    vectorService.ProcessVector2(runInfoNeg2); // V2 ScatterAdd
                    runInfoNeg2.isValid = false;
                }
            }

            taskId++;
        }
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
    }
    if ASCEND_IS_AIV {
        vector2Service.InitParams(constInfo, tilingData);
        vector2Service.InitVector2GM(scatterAddRes, topKIndexGm, dKeyIndexGm, actualSeqLengthsQueryGm,
                                     actualSeqLengthsKeyGm, scatterAddResBanks);
        vector2Service.InitBuffers(pipe);
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        vector2Service.AllocEventID();
        vector2Service.ProcessVector2();
        vector2Service.FreeEventID();
    }
}

template <typename SLIT>
__aicore__ inline int32_t SparseLightningIndexerKLLossGradBase<SLIT>::GetActualSeqLens(
    int32_t bIdx, int32_t defaultLens, GlobalTensor<int32_t> &actualSeqLensGm, SLILayout layout, int64_t &accumLen)
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
__aicore__ inline int32_t SparseLightningIndexerKLLossGradBase<SLIT>::GetUsedSeqLens(int32_t bIdx, int32_t defaultLens,
                                                                                     GlobalTensor<int32_t> &seqUsedGm)
{
    if (seqUsedGm.GetSize() <= 0) {
        return defaultLens;
    }
    return seqUsedGm.GetValue(bIdx);
}

template <typename SLIT>
__aicore__ inline int32_t SparseLightningIndexerKLLossGradBase<SLIT>::GetCmpResidualK(int32_t bIdx)
{
    if (cmpResidualKeyGm.GetSize() <= 0) {
        return 0;
    }
    return cmpResidualKeyGm.GetValue(bIdx);
}

template <typename SLIT>
__aicore__ inline int64_t SparseLightningIndexerKLLossGradBase<SLIT>::GetPreCompressS2Len(int32_t bIdx,
                                                                                          int32_t actualSeqLensK)
{
    int64_t preCompressS2Len = static_cast<int64_t>(actualSeqLensK) * constInfo.cmpRatio + GetCmpResidualK(bIdx);
    return Max(preCompressS2Len, static_cast<int64_t>(0));
}

template <typename SLIT>
__aicore__ inline int32_t SparseLightningIndexerKLLossGradBase<SLIT>::GetS2SparseLen(int32_t bIdx, int32_t s1Idx,
                                                                                     int32_t actualSeqLensQ,
                                                                                     int32_t actualSeqLensK,
                                                                                     SLISparseMode sparseMode)
{
    if (sparseMode == SLISparseMode::RightDown) {
        if (constInfo.cmpRatio != 0) {
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
__aicore__ inline void SparseLightningIndexerKLLossGradBase<SLIT>::GetRunInfo(
    int64_t taskId, int64_t bIdx, int64_t s1Idx, int64_t s1IdxEnd, int64_t accumS1Len, int64_t accumS2Len,
    int32_t actualSeqLensQ, int32_t actualSeqLensK, SLIKLLossGradRunInfo &runInfo)
{
    if (s1Idx >= s1IdxEnd) { // extra循环阶段，不生产任务
        runInfo.isValid = false;
        return;
    }

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
        runInfo.actS1Size = actualSeqLensQ;
        runInfo.actS2Size = actualSeqLensK;
        runInfo.accumS1Idx = bIdx * constInfo.s1Size + s1Idx;
        runInfo.accumS2Idx = bIdx * constInfo.s2Size;
    }

    runInfo.s2SparseLen =
        GetS2SparseLen(runInfo.bIdx, runInfo.s1Idx, runInfo.actS1Size, runInfo.actS2Size, constInfo.sparseMode);
    if (runInfo.s2SparseLen <= 0) {
        runInfo.isValid = false;
        return;
    }

    runInfo.s2RealSize = Min(topKSize, runInfo.s2SparseLen);
    if (constInfo.cmpRatio != 0) {
        runInfo.s2RealSize = Max(1, runInfo.s2RealSize);
    }

    runInfo.kRealSize = runInfo.s2RealSize;
    runInfo.kRealSizeAlign8 = (runInfo.kRealSize + 7) >> 3 << 3;
    runInfo.s2LoopTimes = CeilDiv(runInfo.s2RealSize, constInfo.s2BaseSize);
    runInfo.s2TailSize = (runInfo.s2RealSize % constInfo.s2BaseSize == 0) ? constInfo.s2BaseSize :
                                                                            (runInfo.s2RealSize % constInfo.s2BaseSize);

    runInfo.kLoopTimes = CeilDiv(runInfo.kRealSize, runInfo.kBaseSize);
    runInfo.kTailSize =
        (runInfo.kRealSize % runInfo.kBaseSize == 0) ? runInfo.kBaseSize : (runInfo.kRealSize % runInfo.kBaseSize);

    if constexpr (LAYOUT_T == SLILayout::TND) {
        runInfo.queryTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQuery);
        runInfo.queryRopeTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQueryRope);
        runInfo.queryIndexTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex * constInfo.dSizeQueryIndex;
    } else if constexpr (LAYOUT_T == SLILayout::BSND) {
        runInfo.queryTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQuery);
        runInfo.queryRopeTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQuery * (constInfo.dSizeQueryRope);
        runInfo.queryIndexTensorOffset = runInfo.accumS1Idx * constInfo.gSizeQueryIndex * constInfo.dSizeQueryIndex;
    }

    if constexpr (LAYOUT_T == SLILayout::TND) {
        runInfo.topkGmBaseOffset = runInfo.accumS1Idx * topKSize;
    } else {
        runInfo.topkGmBaseOffset = runInfo.bIdx * constInfo.s1Size * topKSize + runInfo.s1Idx * topKSize;
    }

    runInfo.calcP = ((runInfo.taskIdMod2 == 0 && constInfo.subBlockIdx == 0) ||
                     (runInfo.taskIdMod2 != 0 && constInfo.subBlockIdx != 0));

    runInfo.isValid = true;
}

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_BASE_H
