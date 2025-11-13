/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_mat_mul_reduce_scatter_allto_all.h
 * \brief
 */
#ifndef BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_H
#define BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if __has_include("../../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h")
#include "../../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h"
#else
#include "../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h"
#endif

using namespace AscendC;
using namespace matmul;

struct Tile {
    uint64_t tileLen;
    uint64_t tileCnt;
    uint64_t tailLen;
    uint64_t tailCnt;
};

template <typename BMMRSATAT>  // Batch_Mat_Mul_Reduce_Scatter_All_To_All_Type缩写BMMRSATAT
class BatchMatMulReduceScatterAlltoAll {
public:
    __aicore__ inline BatchMatMulReduceScatterAlltoAll(){};
    __aicore__ inline void Init(__gm__ uint8_t* xGM, __gm__ uint8_t* weightGM, __gm__ uint8_t* biasGM,
                                __gm__ uint8_t* yGM, __gm__ uint8_t* workspaceGM,
                                const BatchMatMulReduceScatterAlltoAllTilingData* tiling, TPipe* tPipe);
    __aicore__ inline void Process();

    using X_T = typename BMMRSATAT::xType;
    using BIAS_T = typename BMMRSATAT::biasType;
    static constexpr bool NEED_BIAS = BMMRSATAT::needBias;
    static constexpr int64_t Y_SHARD_TYPE = BMMRSATAT::yShardType;
    static constexpr bool IS_TRANS = BMMRSATAT::transposeWeight;
    static constexpr bool IS_LITE = BMMRSATAT::isLite;
    
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, IS_TRANS>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BIAS_T, false>;
    BatchMatMulCommonKernel<aType, bType, cType, biasType> bmmV3;

private:
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitOffset();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void NonLocalCommunicationAndCal();
    __aicore__ inline void LocalCommunicationAndCal();
    __aicore__ inline void BmmNonLocalCal(uint64_t aCalcAddr, uint64_t bCalcAddr, uint64_t cCalcAddr, bool isCTail);
    __aicore__ inline void BmmLocalCal(uint64_t aCalcAddr, uint64_t bCalcAddr, uint64_t cCalcAddr, bool isCTail);
    __aicore__ inline void OnNonLocalReduceScatterFinished();
    __aicore__ inline void OnLocalReduceScatterFinished();
    __aicore__ inline void AddTransposeBeforeAlltoAll(bool isLocal, bool isCTail, uint64_t biasOffset, uint64_t rsOutGM,
                                                      uint64_t transOutOffset);
    __aicore__ inline void AddTranspose(uint64_t rsOutOffset, uint64_t biasOffset, uint64_t transOutOffset,
                                        uint64_t dataCnt, bool isLocal);
    __aicore__ inline void AlltoAll(uint64_t alltoallInOffset, uint64_t alltoallOutOffset, bool isCTail, uint64_t loop);
    __aicore__ inline void WaitAllAlltoAll();
    __aicore__ inline void TransposeAroundBMM(uint64_t dataCopyInOffset, bool isLocal, bool isCTail, bool isBefore);
    __aicore__ inline void AllocBuf();

    __aicore__ inline uint64_t MIN(uint64_t x, uint64_t y)
    {
        return (x < y) ? x : y;
    };

    template <typename T>
    __aicore__ inline uint64_t MAX(T x, T y)
    {
        return (x > y) ? static_cast<uint64_t>(x) : static_cast<uint64_t>(y);
    };

    static constexpr uint32_t MAX_HCCL_HANDLE = 32U;
    static constexpr uint32_t MAX_TP_SIZE = 32U;
    static constexpr uint64_t SIXTEEN_ALIGN = 16U;
    static constexpr uint64_t UB_ADDR_ALIGN = 32U;

    TPipe* pipe = nullptr;
    const BatchMatMulReduceScatterAlltoAllTilingData* tilingData = nullptr;
    BatchMatmulTilingData localTileTiling;
    BatchMatmulTilingData localTailTiling;
    BatchMatmulTilingData nonLocalTileTiling;
    BatchMatmulTilingData nonLocalTailTiling;

    uint64_t tmpBlockIdx = 0U;
    uint64_t E_ep = 0U;
    uint64_t C_tp = 0U;
    uint64_t H = 0U;
    uint64_t alignedH = 0U;
    uint64_t M_tp = 0U;
    uint64_t epRankSize = 0U;
    uint64_t tpRankSize = 0U;
    uint64_t epRankId = 0U; // To-do
    uint64_t rsDoneCnt = 0U;
    uint64_t allToAllDoneCnt = 0U;
    uint64_t aivNum = 0U;
    uint64_t dataLen = 0U;
    uint64_t biasLen = 0U;
    uint64_t ubCapacity = 0U; // 数据个数
    uint64_t ubSize = 0U;
    uint64_t localCCnt = 0U;
    uint64_t nonLocalCCnt = 0U;
    uint64_t nonLocalRSoffset = 0U;
    
    uint64_t aTpBlock = 0U; // C_tp * M/tp
    uint64_t aEpBlock = 0U; // tp * C_tp * M/tp
    uint64_t aTileEBlock = 0U; // ep * tp * C_tp * M/tp
    uint64_t bTileEBlock = 0U; // H * M/tp
    uint64_t aLocalECntBlock = 0U; // localE.tileLen * ep * tp * C_tp * M/tp
    uint64_t aLocalCCntBlock = 0U; // localC.tileLen * M/tp
    uint64_t aLocalCTileOffset = 0U; // (localC.tileCnt * localC.tileLen) * M/tp
    uint64_t aLocalCTailBlock = 0U; // localC.tailLen * M/tp
    uint64_t bLocalECntBlock = 0U; // localE.tileLen * H * M/tp
    uint64_t cLocalECntBlock = 0U; // tp * localE.tileLen * C_tp * H
    uint64_t cLocalCCntBlock = 0U; // tp * localE.tileLen * localC.tileLen * H
    uint64_t cLocalCTileOffset = 0U; // tp * localE.tileLen * (localC.tileCnt * localC.tileLen) * H
    uint64_t cLocalCTailBlock = 0U; // tp * localE.tileLen * localC.tailLen * H
    uint64_t cLocalTileEBlock = 0U; // localC.tileLen * H
    uint64_t cLocalTileECTailBlock = 0U; // localC.tailLen * H
    uint64_t cLocalTpBlock = 0U; // localE.tileLen * localC.tileLen * H
    uint64_t cLocalTpCTailBlock = 0U; // localE.tileLen * localC.tailLen * H

    uint64_t aNonLocalECntBlock = 0U; // nonLocalE.tileLen * ep * tp * C_tp * M/tp
    uint64_t aNonLocalCCntBlock = 0U; // nonLocalC.tileLen * M/tp
    uint64_t aNonLocalCTileOffset = 0U; // (nonLocalC.tileCnt * nonLocalC.tileLen) * M/tp
    uint64_t aNonLocalCTailBlock = 0U; // nonLocalC.tailLen * M/tp
    uint64_t bNonLocalECntBlock = 0U; // nonLocalE.tileLen * H * M/tp
    uint64_t cNonLocalECntBlock = 0U; // tp * (ep-1) * nonLocalE.tileLen * C_tp * H
    uint64_t cNonLocalCCntBlock = 0U; // tp * (ep-1) * nonLocalE.tileLen * nonLocalC.tileLen * H
    uint64_t cNonLocalCTileOffset = 0U; // tp * (ep-1) * nonLocalE.tileLen * (nonLocalC.tileCnt * nonLocalC.tileLen) * H
    uint64_t cNonLocalCTailBlock = 0U; // tp * (ep-1) * nonLocalE.tileLen * nonLocalC.tailLen * H
    uint64_t cNonLocalTileEBlock = 0U; // nonLocalC.tileLen * H
    uint64_t cNonLocalTileECTailBlock = 0U; // nonLocalC.tailLen * H
    uint64_t cNonLocalEpBlock = 0U; // nonLocalE.tileLen * nonLocalC.tileLen * H
    uint64_t cNonLocalEpCTailBlock = 0U; // nonLocalE.tileLen * nonLocalC.tailLen * H
    uint64_t cNonLocalTpBlock = 0U; // (ep-1) * nonLocalE.tileLen * nonLocalC.tileLen * H
    uint64_t cNonLocalTpCTailBlock = 0U; // (ep-1) * nonLocalE.tileLen * nonLocalC.tailLen * H
    
    uint64_t LocalRSRecvCount = 0U; // localE.tileLen * localC.tileLen * H
    uint64_t LocalRSRecvTailCount = 0U; // localE.tileLen * localC.tailLen * H
    uint64_t localRSECntBlock = 0U; // localE.tileLen * C_tp * H
    uint64_t localRSCCntBlock = 0U; // localE.tileLen * localC.tileLen * H
    uint64_t localRSCTailBlock = 0U; // localE.tileLen * localC.tailLen * H
    uint64_t localRSCTileOffset = 0U; // localE.tileLen * (localC.tileCnt * localC.tileLen) * H
    uint64_t nonLocalRSRecvCount = 0U; // (ep-1) * nonLocalE.tileLen * nonLocalC.tileLen * H
    uint64_t nonLocalRSRecvTailCount = 0U; // (ep-1) * nonLocalE.tileLen * nonLocalC.tailLen * H
    uint64_t nonLocalRSECntBlock = 0U; // (ep-1) * nonLocalE.tileLen * C_tp * H
    uint64_t nonLocalRSCCntBlock = 0U; // (ep-1) * nonLocalE.tileLen * nonLocalC.tileLen * H
    uint64_t nonLocalRSCTailBlock = 0U; // (ep-1) * nonLocalE.tileLen * nonLocalC.tailLen * H
    uint64_t nonLocalRSCTileOffset = 0U; // (ep-1) * nonLocalE.tileLen * (nonLocalC.tileCnt * nonLocalC.tileLen) * H

    uint64_t nonLocalTransECntBlock = 0U; // ep * nonLocalE.tileLen * C_tp * H
    uint64_t nonLocalTransCCntBlock = 0U; // ep * nonLocalE.tileLen * nonLocalC.tileLen * H
    uint64_t nonLocalTransCTailBlock = 0U; // ep * nonLocalE.tileLen * nonLocalC.tailLen * H
    uint64_t nonLocalTransCTileOffset = 0U; // ep * nonLocalE.tileLen * (nonLocalC.tileCnt * nonLocalC.tileLen) * H

    uint64_t nonLocalBiasECntBlock = 0U; // nonLocalE.tileLen * H
    uint64_t nonLocalYOutECntBlock = 0U; // nonLocalE.tileLen * C_tp * H
    uint64_t nonLocalYOutCCntBlock = 0U; // nonLocalC.tileLen * H
    uint64_t nonLocalYOutCTailBlock = 0U; // nonLocalC.tailLen * H
    uint64_t nonLocalYOutCTileOffset = 0U; // (nonLocalC.tileCnt * nonLocalC.tileLen) * H
    uint64_t localBiasECntBlock = 0U; // localE.tileLen * H
    uint64_t localYOutEpOffset = 0U; // epRankId * E_ep * C_tp * H
    uint64_t localYOutTileCOffset = 0U; // (localC.tileCnt * localC.tileLen) * H

    uint64_t startNonLocalTileRow = 0U; // coreIdx * RowsPerCore
    uint64_t startNonLocalTailRow = 0U;
    uint64_t startLocalTileRow = 0U;
    uint64_t startLocalTailRow = 0U;
    uint64_t startNonLocalTileOffset = 0U; // coreIdx * RowsPerCore * H
    uint64_t startNonLocalTailOffset = 0U;
    uint64_t startLocalTileOffset = 0U;
    uint64_t startLocalTailOffset = 0U;
    uint64_t nLocalTileEndOffset = 0U; // Min(((coreIdx + 1) * RowsPerCore * H), totalLen)
    uint64_t nLocalTailEndOffset = 0U;
    uint64_t localTileEndOffset = 0U;
    uint64_t localTailEndOffset = 0U;
    uint64_t rowsPerLoop = 0U; // ub / H

    uint64_t alltoallSendCount = 0U; // nonLocalC.tileLen * H
    uint64_t alltoallSendTailCount = 0U; // nonLocalC.tailLen * H
    uint64_t alltoallInTileEBlock = 0U; // nonLocalC.tileLen * H
    uint64_t alltoallInTileECTailBlock = 0U; // nonLocalC.tailLen * H
    uint64_t alltoallSendStride = 0U; // nonLocalE.tileLen * nonLocalC.tileLen * H
    uint64_t alltoallSendTailStride = 0U; // nonLocalE.tileLen * nonLocalC.tailLen * H
    uint64_t alltoallRecvStride = 0U; // nonLocalE.tileCnt * nonLocalE.tileLen * C_tp * H
    uint64_t alltoallOutTileEBlock = 0U; // C_tp * H

    GlobalTensor<X_T> xGM;
    GlobalTensor<X_T> weightGM;
    GlobalTensor<BIAS_T> biasGM;
    GlobalTensor<X_T> yGM;
    GlobalTensor<X_T> nonLocalMMOutGM;
    GlobalTensor<X_T> localMMOutGM;
    GlobalTensor<X_T> reduceScatterNonLocalOut;
    GlobalTensor<X_T> reduceScatterLocalOut;
    GlobalTensor<X_T> transposeOutGM;

    // for lite
    GlobalTensor<X_T> transBeforeBMMGM;
    GlobalTensor<X_T> sharedBMMOutGM;

    Hccl<HCCL_SERVER_TYPE_AICPU> hcclAlltoAll;
    Hccl<HCCL_SERVER_TYPE_AICPU> hcclReduceScatter;
    HcclHandle reduceScatterHandleList[MAX_HCCL_HANDLE];
    HcclHandle alltoallList[MAX_HCCL_HANDLE];
    HcclDataType hcclDataType;
    uint64_t alltoallVSendCount[MAX_TP_SIZE];
    uint64_t alltoallVSendStride[MAX_TP_SIZE];
    uint64_t alltoallVRecvStride[MAX_TP_SIZE];

    struct Tile localE; // 无tail
    struct Tile localC; // 有tail
    struct Tile nonLocalE; // 无tail
    struct Tile nonLocalC; // 有tail

    TQue<TPosition::VECIN, 1> rsOutInQue;
    TQue<TPosition::VECIN, 1> biasInQue;
    TQue<TPosition::VECOUT, 1> transposeOutQue;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> transposeQue;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> bmmTransQue;

    TBuf<> rsOutCastBuf;

    LocalTensor<X_T> rsOutTmp;
    LocalTensor<float> rsOutCastTmp;
    LocalTensor<BIAS_T> biasTmp;
    LocalTensor<X_T> transposeOutTmp;
};

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::Init(__gm__ uint8_t* x, __gm__ uint8_t* weight,
    __gm__ uint8_t* bias, __gm__ uint8_t* y, __gm__ uint8_t* workspace,
    const BatchMatMulReduceScatterAlltoAllTilingData* tiling, TPipe* tPipe)
{
    tmpBlockIdx = GetBlockIdx(); // 用于vector分核

    pipe = tPipe;
    tilingData = tiling;
    InitTilingData();

    // HCCL初始化 --- aic需要获取epRankId
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    auto contextGM1 = AscendC::GetHcclContext<1>();
    hcclAlltoAll.Init(contextGM0);
    hcclReduceScatter.Init(contextGM1);
    epRankId = hcclAlltoAll.GetRankId();
    InitOffset();

    xGM.SetGlobalBuffer((__gm__ X_T*)x);
    weightGM.SetGlobalBuffer((__gm__ X_T*)weight);
    yGM.SetGlobalBuffer((__gm__ X_T*)y);
    if constexpr (NEED_BIAS) {
        biasGM.SetGlobalBuffer((__gm__ BIAS_T*)bias);
    }

    // localMMOutGM/nonLocalMMOutGM/reduceScatterLocalOut/reduceScatterNonLocalOut/transposeOutGM
    if constexpr (IS_LITE) {
        uint64_t localMaxCBlock = MAX(localC.tileLen, localC.tailLen) * localE.tileLen * tpRankSize;
        uint64_t nonLocalMaxCBlock = MAX(nonLocalC.tileLen, nonLocalC.tailLen) * nonLocalE.tileLen * tpRankSize *
                                     (epRankSize - 1U);
        uint64_t localTransBefore = localMaxCBlock * M_tp;
        uint64_t nonLocalTransBefore = nonLocalMaxCBlock * M_tp;
        uint64_t localTransAfter = localMaxCBlock * H;
        uint64_t nonLocalTransAfter = nonLocalMaxCBlock * H;

        transBeforeBMMGM.SetGlobalBuffer((__gm__ X_T*)workspace);
        sharedBMMOutGM.SetGlobalBuffer((__gm__ X_T*)transBeforeBMMGM.GetPhyAddr() +
                                       MAX(localTransBefore, nonLocalTransBefore));
        localMMOutGM.SetGlobalBuffer((__gm__ X_T*)sharedBMMOutGM.GetPhyAddr() +
                                     MAX(localTransAfter, nonLocalTransAfter));
    } else {
        localMMOutGM.SetGlobalBuffer((__gm__ X_T*)workspace);
    }
    nonLocalMMOutGM.SetGlobalBuffer((__gm__ X_T*)localMMOutGM.GetPhyAddr() + (tpRankSize * E_ep * C_tp * H));
    reduceScatterLocalOut.SetGlobalBuffer((__gm__ X_T*)nonLocalMMOutGM.GetPhyAddr() +
                                          (tpRankSize * (epRankSize - 1) * E_ep * C_tp * H));
    reduceScatterNonLocalOut.SetGlobalBuffer((__gm__ X_T*)reduceScatterLocalOut.GetPhyAddr() + (E_ep * C_tp * H));
    transposeOutGM.SetGlobalBuffer((__gm__ X_T*)reduceScatterNonLocalOut.GetPhyAddr() +
                                   ((epRankSize - 1) * E_ep * C_tp * H));

    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::InitTilingData()
{
    E_ep = tilingData->commonTiling.EOverEp;
    C_tp = tilingData->commonTiling.COverTp;
    H = tilingData->commonTiling.H;
    alignedH = (H + SIXTEEN_ALIGN - 1U) / SIXTEEN_ALIGN * SIXTEEN_ALIGN;
    M_tp = tilingData->commonTiling.MOverTp;
    epRankSize = tilingData->commonTiling.epGroupSize;
    tpRankSize = tilingData->commonTiling.tpGroupSize;
    aivNum = tilingData->commonTiling.aivCoreNum;
    ubCapacity = tilingData->commonTiling.ubCapacityForAdd;
    localTileTiling = tilingData->localTiling.bmmTilingData;
    localTailTiling = tilingData->localTailTiling.bmmTilingData;
    nonLocalTileTiling = tilingData->domesticTiling.bmmTilingData;
    nonLocalTailTiling = tilingData->domesticTailTiling.bmmTilingData;
    ubSize = tilingData->commonTiling.totalUbSize;
    
    localE.tileCnt = tilingData->commonTiling.localTileE.tileCnt;
    localE.tileLen = tilingData->commonTiling.localTileE.tileLen;
    localC.tileCnt = tilingData->commonTiling.localTileC.tileCnt;
    localC.tileLen = tilingData->commonTiling.localTileC.tileLen;
    localC.tailCnt = tilingData->commonTiling.localTileC.tailCnt;
    localC.tailLen = tilingData->commonTiling.localTileC.tailLen;
    nonLocalE.tileCnt = tilingData->commonTiling.domesticTileE.tileCnt;
    nonLocalE.tileLen = tilingData->commonTiling.domesticTileE.tileLen;
    nonLocalC.tileCnt = tilingData->commonTiling.domesticTileC.tileCnt;
    nonLocalC.tileLen = tilingData->commonTiling.domesticTileC.tileLen;
    nonLocalC.tailCnt = tilingData->commonTiling.domesticTileC.tailCnt;
    nonLocalC.tailLen = tilingData->commonTiling.domesticTileC.tailLen;

    dataLen = sizeof(X_T);
    biasLen = sizeof(BIAS_T);
    if (AscendC::IsSameType<X_T, bfloat16_t>::value) {
        hcclDataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    } else {
        hcclDataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::InitOffset()
{
    nonLocalCCnt = nonLocalC.tileCnt + nonLocalC.tailCnt;
    localCCnt = localC.tileCnt + localC.tailCnt;
    nonLocalRSoffset = nonLocalE.tileCnt * nonLocalCCnt;

    // Common offset for local and nonlocal
    aTpBlock = C_tp * M_tp;
    aEpBlock = tpRankSize * aTpBlock;
    aTileEBlock = epRankSize * aEpBlock;
    bTileEBlock = H * M_tp;
    // LocalMMOffset
    aLocalECntBlock = localE.tileLen * aTileEBlock;
    aLocalCCntBlock = localC.tileLen * M_tp;
    aLocalCTileOffset = localC.tileCnt * aLocalCCntBlock;
    aLocalCTailBlock = localC.tailLen * M_tp;
    bLocalECntBlock = localE.tileLen * bTileEBlock;
    uint64_t localRSCommonBlock = localE.tileLen * H;
    localRSECntBlock = C_tp * localRSCommonBlock;
    cLocalECntBlock = tpRankSize * localRSECntBlock;
    cLocalTileEBlock = localC.tileLen * H;
    cLocalTpBlock = localE.tileLen * cLocalTileEBlock;
    cLocalCCntBlock = tpRankSize * cLocalTpBlock;
    cLocalCTileOffset = localC.tileCnt * cLocalCCntBlock;
    cLocalTileECTailBlock = localC.tailLen * H;
    cLocalTpCTailBlock = localE.tileLen * cLocalTileECTailBlock;
    cLocalCTailBlock = tpRankSize * cLocalTpCTailBlock;
    // NonLocalMMOffset
    aNonLocalECntBlock = nonLocalE.tileLen * aTileEBlock;
    aNonLocalCCntBlock = nonLocalC.tileLen * M_tp;
    aNonLocalCTileOffset = nonLocalC.tileCnt * aNonLocalCCntBlock;
    aNonLocalCTailBlock = nonLocalC.tailLen * M_tp;
    bNonLocalECntBlock = nonLocalE.tileLen * bTileEBlock;
    uint64_t nonLocalRSCommonBlock = (epRankSize - 1) * nonLocalE.tileLen * H;
    nonLocalRSECntBlock = C_tp * nonLocalRSCommonBlock;
    cNonLocalECntBlock = tpRankSize * nonLocalRSECntBlock;
    cNonLocalTileEBlock = nonLocalC.tileLen * H;
    cNonLocalEpBlock = nonLocalE.tileLen * cNonLocalTileEBlock;
    cNonLocalTpBlock = (epRankSize - 1) * cNonLocalEpBlock;
    cNonLocalCCntBlock = tpRankSize * cNonLocalTpBlock;
    cNonLocalCTileOffset = nonLocalC.tileCnt * cNonLocalCCntBlock;
    cNonLocalTileECTailBlock = nonLocalC.tailLen * H;
    cNonLocalEpCTailBlock = nonLocalE.tileLen * cNonLocalTileECTailBlock;
    cNonLocalTpCTailBlock = (epRankSize - 1) * cNonLocalEpCTailBlock;
    cNonLocalCTailBlock = tpRankSize * cNonLocalTpCTailBlock;
    // ReduceScatter offset
    LocalRSRecvCount = cLocalTpBlock;
    LocalRSRecvTailCount = cLocalTpCTailBlock;
    nonLocalRSRecvCount = cNonLocalTpBlock;
    nonLocalRSRecvTailCount = cNonLocalTpCTailBlock;
    localRSCCntBlock = localC.tileLen * localRSCommonBlock;
    localRSCTileOffset = localC.tileCnt * localRSCCntBlock;
    localRSCTailBlock = localC.tailLen * localRSCommonBlock;
    nonLocalRSCCntBlock = nonLocalC.tileLen * nonLocalRSCommonBlock;
    nonLocalRSCTileOffset = nonLocalC.tileCnt * nonLocalRSCCntBlock;
    nonLocalRSCTailBlock = nonLocalC.tailLen * nonLocalRSCommonBlock;
    // Nonlocal transpose offset
    uint64_t nonLocalTransCommonBlock = epRankSize * nonLocalE.tileLen * H;
    nonLocalTransECntBlock = C_tp * nonLocalTransCommonBlock;
    nonLocalTransCCntBlock = nonLocalC.tileLen * nonLocalTransCommonBlock;
    nonLocalTransCTileOffset = nonLocalC.tileCnt * nonLocalTransCCntBlock;
    nonLocalTransCTailBlock = nonLocalC.tailLen * nonLocalTransCommonBlock;
    // bias and yOut offset
    nonLocalBiasECntBlock = nonLocalE.tileLen * H;
    nonLocalYOutECntBlock = C_tp * nonLocalBiasECntBlock;
    nonLocalYOutCCntBlock = nonLocalC.tileLen * H;
    nonLocalYOutCTailBlock = nonLocalC.tailLen * H;
    nonLocalYOutCTileOffset = nonLocalC.tileCnt * nonLocalYOutCCntBlock;
    localBiasECntBlock = localE.tileLen * H;
    localYOutEpOffset = epRankId * E_ep * C_tp * H;
    localYOutTileCOffset = localC.tileCnt * cLocalTileEBlock;
    // Nonlocal/Local tile/tile transpose split core calculation
    uint64_t maxNonLocalTileEndOffset = (epRankSize - 1) * nonLocalE.tileLen * nonLocalC.tileLen;
    uint64_t maxNonLocalTailEndOffset = (epRankSize - 1) * nonLocalE.tileLen * nonLocalC.tailLen;
    uint64_t maxLocalTileEndOffset = localE.tileLen * localC.tileLen;
    uint64_t maxLocalTailEndOffset = localE.tileLen * localC.tailLen;
    uint64_t nLocalTileRowsPerCore = (maxNonLocalTileEndOffset + aivNum - 1) / aivNum;
    uint64_t nLocalTailRowsPerCore = (maxNonLocalTailEndOffset + aivNum - 1) / aivNum;
    uint64_t localTileRowsPerCore = (maxLocalTileEndOffset + aivNum - 1) / aivNum;
    uint64_t localTailRowsPerCore = (maxLocalTailEndOffset + aivNum - 1) / aivNum;
    startNonLocalTileRow = tmpBlockIdx * nLocalTileRowsPerCore;
    startNonLocalTailRow = tmpBlockIdx * nLocalTailRowsPerCore;
    startLocalTileRow = tmpBlockIdx * localTileRowsPerCore;
    startLocalTailRow = tmpBlockIdx * localTailRowsPerCore;
    startNonLocalTileOffset = H * startNonLocalTileRow;
    startNonLocalTailOffset = H * startNonLocalTailRow;
    startLocalTileOffset = H * startLocalTileRow;
    startLocalTailOffset = H * startLocalTailRow;
    nLocalTileEndOffset = H * (tmpBlockIdx + 1) * nLocalTileRowsPerCore;
    nLocalTailEndOffset = H * (tmpBlockIdx + 1) * nLocalTailRowsPerCore;
    localTileEndOffset = H * (tmpBlockIdx + 1) * localTileRowsPerCore;
    localTailEndOffset = H * (tmpBlockIdx + 1) * localTailRowsPerCore;
    nLocalTileEndOffset = MIN(nLocalTileEndOffset, maxNonLocalTileEndOffset * H);
    nLocalTailEndOffset = MIN(nLocalTailEndOffset, maxNonLocalTailEndOffset * H);
    localTileEndOffset = MIN(localTileEndOffset, maxLocalTileEndOffset * H);
    localTailEndOffset = MIN(localTailEndOffset, maxLocalTailEndOffset * H);
    rowsPerLoop = ubCapacity / alignedH;
    // Nonlocal AlltoAll offset
    alltoallInTileEBlock = nonLocalC.tileLen * H;
    alltoallInTileECTailBlock = nonLocalC.tailLen * H;
    alltoallSendCount = alltoallInTileEBlock;
    alltoallSendTailCount = alltoallInTileECTailBlock;
    alltoallSendStride = nonLocalE.tileLen * alltoallInTileEBlock;
    alltoallSendTailStride = nonLocalE.tileLen * alltoallInTileECTailBlock;
    alltoallOutTileEBlock = C_tp * H;
    alltoallRecvStride = nonLocalE.tileCnt * nonLocalE.tileLen * alltoallOutTileEBlock;
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::InitBuffers()
{
    if constexpr (NEED_BIAS) {
        pipe->InitBuffer(rsOutInQue, 1, ubCapacity * dataLen);
        pipe->InitBuffer(biasInQue, 1, ubCapacity * sizeof(BIAS_T));
        pipe->InitBuffer(transposeOutQue, 1, ubCapacity * dataLen);
        if (AscendC::IsSameType<X_T, bfloat16_t>::value) {
            pipe->InitBuffer(rsOutCastBuf, ubCapacity * sizeof(float));
        }
    } else {
        pipe->InitBuffer(transposeQue, 1, ubCapacity * dataLen);
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::AllocBuf()
{
    pipe->Reset();
    pipe->InitBuffer(bmmTransQue, 1, ubSize);
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::Process()
{
    NonLocalCommunicationAndCal();
    LocalCommunicationAndCal();

    if ASCEND_IS_AIV {
        hcclReduceScatter.Finalize();
        hcclAlltoAll.Finalize();
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::NonLocalCommunicationAndCal()
{
    // 非本地块 主块计算
    for (uint64_t eIdx = 0U; eIdx < nonLocalE.tileCnt; eIdx++) {
        uint64_t aCalcOffset = eIdx * aNonLocalECntBlock;
        uint64_t bCalcOffset = eIdx * bNonLocalECntBlock;
        uint64_t cCalcOffset = eIdx * cNonLocalECntBlock;
        uint64_t reduceScatterOutOffset = eIdx * nonLocalRSECntBlock;
        uint64_t rsEIdx = eIdx * nonLocalCCnt;
        uint64_t transOutEOffset = eIdx * nonLocalTransECntBlock;
        uint64_t yOutEOffset = eIdx * nonLocalYOutECntBlock;
        for (uint64_t cIdx = 0U; cIdx < nonLocalCCnt; cIdx++) {
            // 尾块
            bool isCTail = (cIdx >= nonLocalC.tileCnt);
            uint64_t aAddrCOffset = aCalcOffset + cIdx * aNonLocalCCntBlock;
            uint64_t cAddrCOffset = cCalcOffset + cIdx * cNonLocalCCntBlock;
            uint64_t reduceScatterOutCOffset = reduceScatterOutOffset + cIdx * nonLocalRSCCntBlock;
            uint64_t transOutOffset = transOutEOffset + cIdx * nonLocalTransCCntBlock;
            uint64_t yOutOffset = yOutEOffset + cIdx * nonLocalYOutCCntBlock;
            if (isCTail) {
                uint64_t tailCIdx = cIdx - nonLocalC.tileCnt;
                aAddrCOffset = aCalcOffset + tailCIdx * aNonLocalCTailBlock + aNonLocalCTileOffset;
                cAddrCOffset = cCalcOffset + tailCIdx * cNonLocalCTailBlock + cNonLocalCTileOffset;
                reduceScatterOutCOffset = reduceScatterOutOffset + tailCIdx * nonLocalRSCTailBlock +
                                          nonLocalRSCTileOffset;
                transOutOffset = transOutEOffset + tailCIdx * nonLocalTransCTailBlock + nonLocalTransCTileOffset;
                yOutOffset = yOutEOffset + tailCIdx * nonLocalYOutCTailBlock + nonLocalYOutCTileOffset;
            }
            uint64_t reduceScatterInOffset = cAddrCOffset;

            // reduceScatter prepare
            if ASCEND_IS_AIV {
                uint64_t recvCount = (isCTail ? nonLocalRSRecvTailCount : nonLocalRSRecvCount);
                if ((eIdx + cIdx) == 0) {
                    reduceScatterHandleList[rsEIdx + cIdx] = hcclReduceScatter.ReduceScatter<false>(
                        (__gm__ uint8_t*)nonLocalMMOutGM[reduceScatterInOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)reduceScatterNonLocalOut[reduceScatterOutCOffset].GetPhyAddr(),
                        recvCount, hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
                } else {
                    hcclReduceScatter.InterHcclGroupSync(HCCL_GROUP_ID_0,
                        alltoallList[(rsDoneCnt + 1) * nonLocalE.tileLen - 1]);
                    reduceScatterHandleList[rsEIdx + cIdx] = hcclReduceScatter.ReduceScatter<true>(
                        (__gm__ uint8_t*)nonLocalMMOutGM[reduceScatterInOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)reduceScatterNonLocalOut[reduceScatterOutCOffset].GetPhyAddr(),
                        recvCount, hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
                }
            }

            // alltoall prepare
            if ASCEND_IS_AIV {
                AlltoAll(transOutOffset, yOutOffset, isCTail, rsEIdx + cIdx);
            }

            if constexpr (IS_LITE) {
                AllocBuf();
                if ASCEND_IS_AIV {
                    TransposeAroundBMM(aAddrCOffset, false, isCTail, true);
                }
                SyncAll<false>(); // c等v核trans结束
            }

            if ASCEND_IS_AIC {
                // matmul loop
                BmmNonLocalCal(aAddrCOffset, bCalcOffset, cAddrCOffset, isCTail);
            }
            SyncAll<false>(); // 等bmm计算结束

            if constexpr (IS_LITE) {
                AllocBuf();
                if ASCEND_IS_AIV {
                    TransposeAroundBMM(cAddrCOffset, false, isCTail, false);
                }
                SyncAll<false>();
            }

            pipe->Reset();
            InitBuffers();

            if ASCEND_IS_AIV {
                // reduce scatter下发
                if ((eIdx + cIdx) == 0) {
                    hcclReduceScatter.Commit(reduceScatterHandleList[rsEIdx + cIdx]);
                }
                // ADD + AlltoAll
                if ((eIdx + cIdx) != 0) {  // 下一轮等上一轮结果，第一轮不做
                    hcclReduceScatter.Wait(reduceScatterHandleList[rsDoneCnt]);
                    OnNonLocalReduceScatterFinished();
                    rsDoneCnt += 1;
                }
            }
        }
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::BmmNonLocalCal(uint64_t aCalcAddr,
    uint64_t bCalcAddr, uint64_t cCalcAddr, bool isCTail)
{
    pipe->Reset();
    if constexpr (IS_LITE) {
        bmmV3.Init((__gm__ uint8_t*)transBeforeBMMGM.GetPhyAddr(), (__gm__ uint8_t*)weightGM[bCalcAddr].GetPhyAddr(),
            (__gm__ uint8_t*)sharedBMMOutGM.GetPhyAddr(), nullptr, nullptr, nullptr,
            (isCTail ? &nonLocalTailTiling : &nonLocalTileTiling), pipe);
        bmmV3.Process();
    } else {
        bmmV3.Init((__gm__ uint8_t*)xGM.GetPhyAddr(), (__gm__ uint8_t*)weightGM.GetPhyAddr(),
            (__gm__ uint8_t*)nonLocalMMOutGM.GetPhyAddr(), nullptr, nullptr, nullptr,
            (isCTail ? &nonLocalTailTiling : &nonLocalTileTiling), pipe);
        for (uint64_t i = 0U; i < tpRankSize; i++) {
            uint64_t aAddrTpOffset = aCalcAddr + i * aTpBlock;
            uint64_t cAddrTpOffset = cCalcAddr + i * (isCTail ? cNonLocalTpCTailBlock : cNonLocalTpBlock);
            for (uint64_t j = 0U; j < epRankSize; j++) {
                if (j == epRankId) { // MatrixA跳过local块
                    continue;
                }
                uint64_t aAddrEpOffset = aAddrTpOffset + j * aEpBlock;
                uint64_t curEpIdx = (j > epRankId) ? (j - 1) : j;
                uint64_t cAddrEpOffset = cAddrTpOffset + curEpIdx * (isCTail ? cNonLocalEpCTailBlock :
                                                                               cNonLocalEpBlock);
                for (uint64_t k = 0U; k < nonLocalE.tileLen; k++) {
                    uint64_t aAddrOffset = aAddrEpOffset + k * aTileEBlock;
                    uint64_t bAddrOffset = bCalcAddr + k * bTileEBlock;
                    uint64_t cAddrOffset = cAddrEpOffset + k * (isCTail ? cNonLocalTileECTailBlock :
                                                                          cNonLocalTileEBlock);
                    bmmV3.UpdateGlobalTensor((__gm__ uint8_t*)xGM[aAddrOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)weightGM[bAddrOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)nonLocalMMOutGM[cAddrOffset].GetPhyAddr(),
                        nullptr, nullptr, nullptr);
                    bmmV3.Process();
                }
            }
        }
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::LocalCommunicationAndCal()
{
    // 主块处理
    uint64_t aCalcOffset = IS_LITE ? 0UL : epRankId * aEpBlock; // 偏到local块
    for (uint64_t eIdx = 0U; eIdx < localE.tileCnt; eIdx++) {
        uint64_t aAddrEOffset = aCalcOffset + eIdx * aLocalECntBlock;
        uint64_t bCalcOffset = eIdx * bLocalECntBlock;
        uint64_t cCalcOffset = eIdx * cLocalECntBlock;
        uint64_t reduceScatterOut = eIdx * localRSECntBlock;
        uint64_t rsEIdx = nonLocalRSoffset + eIdx * localCCnt;
        for (uint64_t cIdx = 0U; cIdx < localCCnt; cIdx++) {
            // 尾块
            bool isCTail = (cIdx >= localC.tileCnt);
            uint64_t aAddrCOffset = aAddrEOffset + cIdx * aLocalCCntBlock;
            uint64_t cAddrCOffset = cCalcOffset + cIdx * cLocalCCntBlock;
            uint64_t reduceScatterOutCOffset = reduceScatterOut + cIdx * localRSCCntBlock;
            if (unlikely(isCTail)) { // 暂时不切localC
                uint64_t tailCIdx = cIdx - localC.tileCnt;
                aAddrCOffset = aAddrEOffset + tailCIdx * aLocalCTailBlock + aLocalCTileOffset;
                cAddrCOffset = cCalcOffset + tailCIdx * cLocalCTailBlock + cLocalCTileOffset;
                reduceScatterOutCOffset = reduceScatterOut + cIdx * localRSCTailBlock + localRSCTileOffset;
            }
            uint64_t reduceScatterInOffset = cAddrCOffset;

            // reduceScatter prepare
            if ASCEND_IS_AIV {
                uint64_t recvCount = isCTail ? LocalRSRecvTailCount : LocalRSRecvCount;
                if ((eIdx + cIdx) == 0) {
                    hcclReduceScatter.InterHcclGroupSync(HCCL_GROUP_ID_0,
                        alltoallList[(rsDoneCnt + 1) * nonLocalE.tileLen - 1]);
                    reduceScatterHandleList[rsEIdx + cIdx] = hcclReduceScatter.ReduceScatter<true>(
                        (__gm__ uint8_t*)localMMOutGM[reduceScatterInOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)reduceScatterLocalOut[reduceScatterOutCOffset].GetPhyAddr(),
                        recvCount, hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
                } else {
                    reduceScatterHandleList[rsEIdx + cIdx] = hcclReduceScatter.ReduceScatter<false>(
                        (__gm__ uint8_t*)localMMOutGM[reduceScatterInOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)reduceScatterLocalOut[reduceScatterOutCOffset].GetPhyAddr(),
                        recvCount, hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
                }
            }

            if constexpr (IS_LITE) {
                AllocBuf();
                if ASCEND_IS_AIV {
                    TransposeAroundBMM(aAddrCOffset, true, isCTail, true);
                }
                SyncAll<false>(); // c等v核trans结束
            }

            if ASCEND_IS_AIC {
                // batch matmul
                BmmLocalCal(aAddrCOffset, bCalcOffset, cAddrCOffset, isCTail);
            }
            SyncAll<false>();

            if constexpr (IS_LITE) {
                AllocBuf();
                if ASCEND_IS_AIV {
                    TransposeAroundBMM(cAddrCOffset, true, isCTail, false);
                }
                SyncAll<false>();
            }

            pipe->Reset();
            InitBuffers();

            // reduce scatter
            if ASCEND_IS_AIV {
                if ((eIdx + cIdx) != 0) {
                    hcclReduceScatter.Commit(reduceScatterHandleList[rsEIdx + cIdx]); // 下发当前轮Reduce Scatter
                }
                hcclReduceScatter.Wait(reduceScatterHandleList[rsDoneCnt]); // 等待上一轮Reduce Scatter
                // ADD + AlltoAll
                if ((eIdx + cIdx) == 0) { // 最后一块Nonlocal ReduceScatter处理
                    OnNonLocalReduceScatterFinished();
                } else {
                    if ((eIdx + cIdx) == 1) { // 做第一块local前等完所有nonlocal的结果，再覆盖local结果
                        WaitAllAlltoAll();
                    }
                    OnLocalReduceScatterFinished();
                }
                rsDoneCnt += 1;
            }
        }
    }
    if ASCEND_IS_AIV {
        WaitAllAlltoAll(); // 如果前面只有一轮，这里要等所有nonlocal结果
        hcclReduceScatter.Wait(reduceScatterHandleList[rsDoneCnt]);
        OnLocalReduceScatterFinished(); // 最后一块local ReduceScatter处理
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::BmmLocalCal(uint64_t aCalcAddr, uint64_t bCalcAddr,
    uint64_t cCalcAddr, bool isCTail)
{
    // 归一化Matmul计算类，负责MC2的Matmul计算
    pipe->Reset();
    if constexpr (IS_LITE) {
        bmmV3.Init((__gm__ uint8_t*)transBeforeBMMGM.GetPhyAddr(), (__gm__ uint8_t *)weightGM[bCalcAddr].GetPhyAddr(),
            (__gm__ uint8_t*)sharedBMMOutGM.GetPhyAddr(), nullptr, nullptr, nullptr,
            (isCTail ? &localTailTiling : &localTileTiling), pipe);
        bmmV3.Process();
    } else {
        bmmV3.Init((__gm__ uint8_t*)xGM.GetPhyAddr(), (__gm__ uint8_t*)weightGM.GetPhyAddr(),
            (__gm__ uint8_t*)localMMOutGM.GetPhyAddr(), nullptr, nullptr, nullptr,
            (isCTail ? &localTailTiling : &localTileTiling), pipe);
        for (uint64_t i = 0U; i < tpRankSize; i++) {
            uint64_t aAddrTpOffset = aCalcAddr + i * aTpBlock;
            uint64_t cAddrTpOffset = cCalcAddr + i * (isCTail ? cLocalTpCTailBlock : cLocalTpBlock);
            for (uint64_t j = 0U; j < localE.tileLen; j++) {
                uint64_t aAddrOffset = aAddrTpOffset + j * aTileEBlock;
                uint64_t bAddrOffset = bCalcAddr + j * bTileEBlock;
                uint64_t cAddrOffset = cAddrTpOffset + j * (isCTail ? cLocalTileECTailBlock : cLocalTileEBlock);
                bmmV3.UpdateGlobalTensor((__gm__ uint8_t*)xGM[aAddrOffset].GetPhyAddr(),
                    (__gm__ uint8_t*)weightGM[bAddrOffset].GetPhyAddr(),
                    (__gm__ uint8_t*)localMMOutGM[cAddrOffset].GetPhyAddr(),
                    nullptr, nullptr, nullptr);
                bmmV3.Process();
            }
        }
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::OnNonLocalReduceScatterFinished()
{
    bool isCTail = false;
    uint64_t eIdx = rsDoneCnt / nonLocalCCnt;
    uint64_t cIdx = rsDoneCnt % nonLocalCCnt;
    uint64_t transInEOffset = eIdx * nonLocalRSECntBlock;
    uint64_t transOutEOffset = eIdx * nonLocalTransECntBlock;
    uint64_t yOutEOffset = eIdx * nonLocalYOutECntBlock;
    uint64_t transInOffset = transInEOffset + cIdx * nonLocalRSCCntBlock;
    uint64_t transOutOffset = transOutEOffset + cIdx * nonLocalTransCCntBlock;
    if (cIdx >= nonLocalC.tileCnt) {
        isCTail = true;
        uint64_t tailCIdx = cIdx - nonLocalC.tileCnt;
        transInOffset = transInEOffset + tailCIdx * nonLocalRSCTailBlock + nonLocalRSCTileOffset;
        transOutOffset = transOutEOffset + tailCIdx * nonLocalTransCTailBlock + nonLocalTransCTileOffset;
    }
    AddTransposeBeforeAlltoAll(false, isCTail, eIdx * nonLocalBiasECntBlock, transInOffset, transOutOffset);
    SyncAll<true>(); // V核同步
    for (uint64_t eIdx = 0U; eIdx < nonLocalE.tileLen; eIdx++) { // alltoall下发
        hcclAlltoAll.Commit(alltoallList[rsDoneCnt * nonLocalE.tileLen + eIdx]);
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::OnLocalReduceScatterFinished()
{
    bool isCTail = false;
    uint64_t eIdx = (rsDoneCnt - nonLocalRSoffset) / localCCnt;
    uint64_t cIdx = (rsDoneCnt - nonLocalRSoffset) % localCCnt;
    uint64_t transInEOffset = eIdx * localRSECntBlock;
    uint64_t transOutEOffset = eIdx * localRSECntBlock + localYOutEpOffset;
    uint64_t transInOffset = transInEOffset + cIdx * localRSCCntBlock;
    uint64_t transOutOffset = transOutEOffset + cIdx * cLocalTileEBlock;
    if (unlikely(cIdx >= localC.tileCnt)) { // 暂时不切localC
        isCTail = true;
        uint64_t tailCIdx = cIdx - localC.tileCnt;
        transInOffset = transInEOffset + tailCIdx * localRSCTailBlock + localRSCTileOffset;
        transOutOffset = transOutEOffset + tailCIdx * cLocalTileECTailBlock + localYOutTileCOffset;
    }
    AddTransposeBeforeAlltoAll(true, isCTail, eIdx * localBiasECntBlock, transInOffset, transOutOffset);
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::AddTransposeBeforeAlltoAll(bool isLocal,
    bool isCTail, uint64_t biasOffset, uint64_t rsOutGM, uint64_t transOutOffset)
{
    uint64_t cLength = 0U;
    uint64_t eLength = 0U;
    uint64_t startDataOffset = 0U;
    uint64_t endDataOffset = 0U;
    uint64_t startRow = 0U;
    if (isLocal) {
        cLength = isCTail ? localC.tailLen : localC.tileLen;
        eLength = localE.tileLen;
        startDataOffset = isCTail ? startLocalTailOffset : startLocalTileOffset;
        endDataOffset = isCTail ? localTailEndOffset : localTileEndOffset;
        startRow = isCTail ? startLocalTailRow : startLocalTileRow;
    } else {
        eLength = nonLocalE.tileLen;
        cLength = isCTail ? nonLocalC.tailLen : nonLocalC.tileLen;
        startDataOffset = isCTail ? startNonLocalTailOffset : startNonLocalTileOffset;
        endDataOffset = isCTail ? nLocalTailEndOffset : nLocalTileEndOffset;
        startRow = isCTail ? startNonLocalTailRow : startNonLocalTileRow;
    }
    uint64_t curDataOffset = startDataOffset;
    uint64_t curRow = startRow;
    uint64_t curBlockMaxRow = (curRow / cLength + 1) * cLength;
    uint64_t dataCnt;
    while (curDataOffset < endDataOffset) {
        uint64_t transOutTmpOffset = transOutOffset;
        uint64_t curEpIdx = curRow / (eLength * cLength);
        if (!isLocal && (curEpIdx >= epRankId)) { // 跳过local块
            transOutTmpOffset += cLength * nonLocalBiasECntBlock;
        }
        uint64_t curTileEIdx = (curRow % (eLength * cLength) / cLength);
        if (rowsPerLoop == 0) { // H超大，分片切列，一片处理最大数据量为ubCapacity
            if (((curRow + 1UL) * H - curDataOffset) > ubCapacity) {
                dataCnt = ubCapacity;
            } else {
                dataCnt = (curRow + 1UL) * H - curDataOffset;
                curRow += 1;
            }
        } else { // H小，分片切行，一片处理最大的数据量为rowsPerloop * H
            if ((curBlockMaxRow * H - curDataOffset) > (rowsPerLoop * H)) {
                dataCnt = rowsPerLoop * H;
                curRow += rowsPerLoop;
            } else { // 不跨e块
                dataCnt = curBlockMaxRow * H - curDataOffset;
                curBlockMaxRow += cLength;
                curRow += ((dataCnt + H - 1)/ H);
            }
        }
        AddTranspose(rsOutGM + curDataOffset, biasOffset + curTileEIdx * H + ((curDataOffset - startDataOffset) % H),
                     transOutTmpOffset + curDataOffset, dataCnt, isLocal);
        curDataOffset += dataCnt;
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::AddTranspose(uint64_t rsOutOffset,
    uint64_t biasOffset, uint64_t transOutOffset, uint64_t dataCnt, bool isLocal)
{
    GlobalTensor<X_T> rsOutStartGM = (isLocal ? reduceScatterLocalOut[rsOutOffset] :
                                                reduceScatterNonLocalOut[rsOutOffset]);
    GlobalTensor<X_T> transOutGM = (isLocal ? yGM[transOutOffset] : transposeOutGM[transOutOffset]);
    if constexpr (NEED_BIAS) {
        // Add left Copy-In
        rsOutTmp = rsOutInQue.AllocTensor<X_T>();
        if ((dataCnt > H) && ((H * dataLen) % UB_ADDR_ALIGN != 0)) {
            const DataCopyExtParams copyParams{static_cast<uint16_t>(dataCnt / H),
                                                static_cast<uint32_t>(H * dataLen), 0U, 0U, 0U};
            const DataCopyPadExtParams<X_T> copyPadParams{false, 0U, 0U, 0U};
            DataCopyPad(rsOutTmp, rsOutStartGM, copyParams, copyPadParams);
        } else {
            const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(dataCnt * dataLen), 0U, 0U, 0U};
            const DataCopyPadExtParams<X_T> copyPadParams{false, 0U, 0U, 0U};
            DataCopyPad(rsOutTmp, rsOutStartGM, copyParams, copyPadParams);
        }
        rsOutInQue.EnQue(rsOutTmp);
        rsOutInQue.DeQue();

        // Add right Copy-In
        biasTmp = biasInQue.AllocTensor<BIAS_T>();
        const DataCopyExtParams copyParamsBias{1U, static_cast<uint32_t>(MIN(dataCnt, H) * biasLen), 0U, 0U, 0U};
        const DataCopyPadExtParams<BIAS_T> copyPadParamsBias{false, 0U, 0U, 0U};
        DataCopyPad(biasTmp, biasGM[biasOffset], copyParamsBias, copyPadParamsBias);
        biasInQue.EnQue(biasTmp);
        biasInQue.DeQue();
        transposeOutTmp = transposeOutQue.AllocTensor<X_T>();
        // add
        if constexpr (AscendC::IsSameType<X_T, bfloat16_t>::value) {
            rsOutCastTmp = rsOutCastBuf.Get<float>();
            Cast(rsOutCastTmp, rsOutTmp, RoundMode::CAST_NONE, (dataCnt + H - 1U) / H * alignedH); // cast成float
            PipeBarrier<PIPE_V>();
            if (dataCnt > H) { // 多行add bias, dataCnt为H的倍数
                for (int i = 0; i < (dataCnt / H); i++) {
                    Add(rsOutCastTmp[i * alignedH], rsOutCastTmp[i * alignedH], biasTmp, H);
                }
            } else { // 单行add部分bias
                Add(rsOutCastTmp, rsOutCastTmp, biasTmp, dataCnt);
            }
            PipeBarrier<PIPE_V>();
            Cast(transposeOutTmp, rsOutCastTmp, RoundMode::CAST_ROUND, (dataCnt + H - 1U) / H * alignedH); // cast回bfloat16
        } else {
            if (dataCnt > H) { // 多行add bias
                for (int i = 0; i < (dataCnt / H); i++) {
                    Add(transposeOutTmp[i * alignedH], rsOutTmp[i * alignedH], biasTmp, H);
                }
            } else { // 单行add部分bias
                Add(transposeOutTmp, rsOutTmp, biasTmp, dataCnt);
            }
        }
        PipeBarrier<PIPE_V>();
        // Transpose
        transposeOutQue.EnQue(transposeOutTmp);
        transposeOutQue.DeQue();
        if ((dataCnt > H) && ((H * dataLen) % UB_ADDR_ALIGN != 0)) {
            const DataCopyExtParams copyOutParams{static_cast<uint16_t>(dataCnt / H),
                                                    static_cast<uint32_t>(H * dataLen), 0U, 0U, 0U};
            DataCopyPad(transOutGM, transposeOutTmp, copyOutParams);
        } else {
            const DataCopyExtParams copyOutParams{1U, static_cast<uint32_t>(dataCnt * dataLen), 0U, 0U, 0U};
            DataCopyPad(transOutGM, transposeOutTmp, copyOutParams);
        }
        rsOutInQue.FreeTensor(rsOutTmp);
        biasInQue.FreeTensor(biasTmp);
        transposeOutQue.FreeTensor(transposeOutTmp);
    } else {
        // Copy-In + transpose
        rsOutTmp = transposeQue.AllocTensor<X_T>();
        const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(dataCnt * dataLen), 0U, 0U, 0U};
        const DataCopyPadExtParams<X_T> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(rsOutTmp, rsOutStartGM, copyParams, copyPadParams);
        transposeQue.EnQue(rsOutTmp);
        transposeQue.DeQue();

        // Copy-Out
        DataCopyPad(transOutGM, rsOutTmp, copyParams);
        transposeQue.FreeTensor(rsOutTmp);
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::AlltoAll(
    uint64_t alltoallInOffset, uint64_t alltoallOutOffset, bool isCTail, uint64_t loop)
{
    uint64_t alltoallInTmp = alltoallInOffset;
    uint64_t alltoallOutTmp = alltoallOutOffset;
    // 总通信量为ep * tile/tail C_tp * H
    for (uint64_t i = 0U; i < epRankSize; i++) {
        alltoallVSendCount[i] = (isCTail ? alltoallSendTailCount : alltoallSendCount);
        alltoallVSendStride[i] = i * (isCTail ? alltoallSendTailStride : alltoallSendStride);
        alltoallVRecvStride[i] = i * alltoallRecvStride;
    }

    // 每一轮bmmReduceScatter会对应nonLocalE.tileLen次alltoall
    for (uint64_t eIdx = 0U; eIdx < nonLocalE.tileLen; eIdx++) {
        alltoallList[loop * nonLocalE.tileLen + eIdx] =
            hcclAlltoAll.AlltoAllV<false>((__gm__ uint8_t*)transposeOutGM[alltoallInTmp].GetPhyAddr(),
            alltoallVSendCount, alltoallVSendStride, hcclDataType, (__gm__ uint8_t*)yGM[alltoallOutTmp].GetPhyAddr(),
            alltoallVSendCount, alltoallVRecvStride, hcclDataType, 1);
        alltoallInTmp += (isCTail ? alltoallInTileECTailBlock : alltoallInTileEBlock);
        alltoallOutTmp += alltoallOutTileEBlock;
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::WaitAllAlltoAll()
{
    while (allToAllDoneCnt < nonLocalRSoffset * nonLocalE.tileLen) {
        hcclAlltoAll.Wait(alltoallList[allToAllDoneCnt]);
        allToAllDoneCnt += 1;
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAll<BMMRSATAT>::TransposeAroundBMM(uint64_t dataCopyInOffset,
    bool isLocal, bool isCTail, bool isBefore)
{
    DataCopyExtParams dataCopyInParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<X_T> dataCopyInPadParams = {false, 0U, 0U, 0U};

    uint64_t col = isBefore ? M_tp : H;
    uint64_t factorOut = 0U;
    uint64_t tileLenC = 0U;
    uint64_t tileLenE = 0U;
    if (isLocal) {
        tileLenC = isCTail ? localC.tailLen : localC.tileLen;
        tileLenE = localE.tileLen;
        factorOut = 1U;
    } else {
        tileLenC = isCTail ? nonLocalC.tailLen : nonLocalC.tileLen;
        tileLenE = nonLocalE.tileLen;
        factorOut = epRankSize - 1U;
    }
    uint64_t totalM = static_cast<uint64_t>(tpRankSize) * tileLenE * factorOut * tileLenC;
    uint64_t tileLen = (totalM + aivNum - 1) / aivNum;
    uint64_t curRow = tileLen * tmpBlockIdx;
    uint64_t curDataOffset = curRow * col;
    uint64_t endDataOffset = MIN(tileLen * col * (tmpBlockIdx + 1U), totalM * col);
    uint64_t endRow = MIN(tileLen * (tmpBlockIdx + 1U), totalM);
    uint64_t curBlockMaxRow = MIN(((curRow / tileLenC) + 1) * tileLenC, endRow);
    uint64_t transUbCapacity = ubSize / dataLen;
    uint64_t perMaxRow = transUbCapacity / col;
    uint64_t dataCnt = 0UL;

    while (curDataOffset < endDataOffset) {
        uint64_t k = curRow / (tpRankSize * factorOut * tileLenC);
        uint64_t t = (curRow % (tpRankSize * tileLenC)) / tileLenC;
        uint64_t blockInnerOffset = curDataOffset % (tileLenC * col);
        uint64_t eOut = (curRow % (tpRankSize * factorOut * tileLenC)) / (tpRankSize * tileLenC);
        uint64_t eIn = (eOut >= epRankId) ? (eOut + 1UL) : eOut; // eIn只有transBefore需要
        eIn = isLocal ? epRankId : eIn;
        if (!isBefore && isLocal) { // 只有transAfter local需要
            eOut = 0UL;
        }

        if (perMaxRow == 0UL) { // col超大，分片切列，一片处理的最大数据量为transUbCapacity
            if (((curRow + 1UL) * col - curDataOffset) > transUbCapacity) {
                dataCnt = transUbCapacity;
            } else {
                dataCnt = (curRow + 1UL) * col - curDataOffset;
                curRow += 1UL;
            }
        } else { // col较小，分片切行
            if ((curBlockMaxRow * col - curDataOffset) > (perMaxRow * col)) {
                dataCnt = perMaxRow * col;
            } else {
                dataCnt = curBlockMaxRow * col - curDataOffset;
                curBlockMaxRow += tileLenC;
                curBlockMaxRow = MIN(curBlockMaxRow, endRow);
            }
            curRow += dataCnt / col;
        }
        dataCopyInParams.blockLen = dataCnt * dataLen;
        dataCopyOutParams.blockLen = dataCnt * dataLen;
        GlobalTensor<X_T> dataCopyInGM = (isBefore ?
            xGM[dataCopyInOffset + k * aTileEBlock + eIn * aEpBlock + t * aTpBlock + blockInnerOffset] :
            sharedBMMOutGM[curDataOffset]);
        GlobalTensor<X_T> dataCopyOutGM = transBeforeBMMGM[curDataOffset];
        if (!isBefore) {
            uint64_t dataCopyOutOffset = dataCopyInOffset + t * (tileLenE * factorOut * tileLenC * H) +
                eOut * (tileLenE * tileLenC * H) + k * (tileLenC * H) + blockInnerOffset;
            dataCopyOutGM = isLocal ? localMMOutGM[dataCopyOutOffset] : nonLocalMMOutGM[dataCopyOutOffset];
        }
        LocalTensor<X_T> transposeTmp = bmmTransQue.AllocTensor<X_T>();
        DataCopyPad(transposeTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);
        bmmTransQue.EnQue(transposeTmp);
        bmmTransQue.DeQue();
        DataCopyPad(dataCopyOutGM, transposeTmp, dataCopyOutParams);
        bmmTransQue.FreeTensor(transposeTmp);

        curDataOffset += dataCnt;
    }
}

#endif  // BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_H
