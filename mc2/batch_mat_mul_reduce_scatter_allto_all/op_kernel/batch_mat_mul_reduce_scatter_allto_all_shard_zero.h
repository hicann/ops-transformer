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
 * \file batch_mat_mul_reduce_scatter_allto_all_shard_zero.h
 * \brief
 */
#ifndef BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_SHARD_ZERO_H
#define BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_SHARD_ZERO_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if __has_include("../../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h")
#include "../../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h"
#else
#include "../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h"
#endif

using namespace AscendC;
using namespace matmul;

template <typename BMMRSATAT>  // Batch_Mat_Mul_Reduce_Scatter_All_To_All_Type缩写BMMRSATAT
class BatchMatMulReduceScatterAlltoAllShard0 {
public:
    __aicore__ inline BatchMatMulReduceScatterAlltoAllShard0(){};
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
    __aicore__ inline void InitBuffers();
    __aicore__ inline void NonLocalCommunication();
    __aicore__ inline void NonLocalCalculation();
    __aicore__ inline void LocalCommunication();
    __aicore__ inline void LocalCalculation();
    __aicore__ inline void ReshapeAfterBmm(uint64_t validELen, uint64_t validCLen, uint64_t validEpIdx,
        uint64_t reshapeOutOffset, bool isLocal);
    __aicore__ inline void AddTranspose(uint64_t rsOutOffset, uint64_t biasOffset, uint64_t addTransposeOffset,
        uint64_t dataCnt, bool isLocal, bool isFirst, bool isLargeH);
    __aicore__ inline void AddTransposeAfterRS(uint64_t rsOutOffset, uint64_t biasOffset, uint64_t addTransposeOffset,
        uint64_t validELen, uint64_t validCLen, uint64_t validEpIdx, bool isLocal);
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
    static constexpr uint64_t FP32_DATASIZE = 4;
    static constexpr uint64_t MAX_BLOCK_CNT = 4095;

    TPipe* pipe = nullptr;
    const BatchMatMulReduceScatterAlltoAllTilingData* tilingData = nullptr;
    BatchMatmulTilingData localTileTiling;
    BatchMatmulTilingData localTailTiling;
    BatchMatmulTilingData nonLocalTileTiling;
    BatchMatmulTilingData nonLocalTailTiling;

    // Overall Loop sequence is: e_cnt, c_cnt, ep, e_len, c_len, H(H/tp)
    // LocalE - 有尾块， LocalC 无尾块
    // nonLocalE - 有尾块， nonLocalC 有尾块

    uint64_t tmpBlockIdx = 0U;
    uint64_t E_ep = 0U;
    uint64_t C = 0U;
    uint64_t H = 0U;
    uint64_t H_tp = 0U;
    uint64_t alignedH = 0U;
    uint64_t alignedH_tp = 0U;
    uint64_t M_tp = 0U;
    uint64_t epRankSize = 0U;
    uint64_t epMinusOne = 0U;
    uint64_t tpRankSize = 0U;
    uint64_t epRankId = 0U;
    uint64_t tpRankId = 0U;
    uint64_t rsDoneCnt = 0U;
    uint64_t allToAllDoneCnt = 0U;
    uint64_t aivNum = 0U;
    uint64_t dataLen = 0U;
    uint64_t biasLen = 0U;
    uint64_t ubCapacity = 0U; // 数据个数
    uint64_t ubSize = 0U;
    uint64_t localCCnt = 0U;
    uint64_t nonLocalCCnt = 0U;
    uint64_t localECnt = 0U;
    uint64_t nonLocalECnt = 0U;
    uint64_t prevReduceScatterOutCOffset = 0U;
    uint64_t prevBiasOffset = 0U;
    uint64_t prevAddTransposeCOffset = 0U;
    uint64_t prevValidELen = 0U;
    uint64_t prevValidCLen = 0U;
    uint64_t prevValidEpIdx = 0U;
    bool prevIsLocal = false;
    uint64_t nonLocalLastAddTransposeOffset = 0U;
    uint64_t nonLocalLastYGMOffset = 0U;

    GlobalTensor<X_T> xGM;
    GlobalTensor<X_T> weightGM;
    GlobalTensor<BIAS_T> biasGM;
    GlobalTensor<X_T> yGM;
    GlobalTensor<X_T> mmOutGM; // Local与NonLocal公用
    GlobalTensor<X_T> reshapeLocalOutGM;
    GlobalTensor<X_T> reshapeNonLocalOutGM;
    GlobalTensor<X_T> reduceScatterNonLocalOutGM;
    GlobalTensor<X_T> reduceScatterLocalOutGM;
    GlobalTensor<X_T> addTransposeOutGM; // 仅Nonlocal会使用

    Hccl<HCCL_SERVER_TYPE_AICPU> hcclAlltoAll;
    Hccl<HCCL_SERVER_TYPE_AICPU> hcclReduceScatter;
    HcclHandle reduceScatterHandleList[MAX_HCCL_HANDLE];
    HcclHandle alltoallList[MAX_HCCL_HANDLE];
    HcclDataType hcclDataType;
    uint64_t alltoallVSendCount[MAX_TP_SIZE];
    uint64_t alltoallVSendStride[MAX_TP_SIZE];
    uint64_t alltoallVRecvStride[MAX_TP_SIZE];

    TileInfo localE; // 有tail
    TileInfo localC; // 无tail
    TileInfo nonLocalE; // 有tail
    TileInfo nonLocalC; // 有tail

    TBuf<> tBuf;

    LocalTensor<X_T> bmmOutTmp; // 存放搬入的bmm的结果，准备做Reshape
    LocalTensor<X_T> rsOutTmp; // 存放搬入的RS结果，准备做Add Bias和Transpose。Add的左输入
    LocalTensor<float> rsOutCastTmp; // 若Bias类型为FP32，将搬入的RS结果Cast成FP32并存在此处，替代上面rsOutTmp成为Add的左输入
    LocalTensor<BIAS_T> biasTmp; // 存放搬入的Bias数据，Add Bias时使用。Add的右输入
    LocalTensor<X_T> addOutTmp; // 保存Add操作的结果，若存在Cast成FP32的场景，则保存将结果Cast回BF16后的结果
};

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::Init(__gm__ uint8_t* x,
    __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* y,
    __gm__ uint8_t* workspace, const BatchMatMulReduceScatterAlltoAllTilingData* tiling,
    TPipe* tPipe)
{
    tmpBlockIdx = GetBlockIdx(); // 用于vector分核

    this->pipe = tPipe;
    this->tilingData = tiling;
    InitTilingData();

    // HCCL初始化 --- aic需要获取epRankId
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    auto contextGM1 = AscendC::GetHcclContext<1>();
    hcclAlltoAll.Init(contextGM0);
    hcclReduceScatter.Init(contextGM1);
    epRankId = hcclAlltoAll.GetRankId();
    tpRankId = hcclReduceScatter.GetRankId();

    localCCnt = localC.tileCnt + localC.tailCnt;
    nonLocalCCnt = nonLocalC.tileCnt + nonLocalC.tailCnt;
    localECnt = localE.tileCnt + localE.tailCnt;
    nonLocalECnt = nonLocalE.tileCnt + nonLocalE.tailCnt;

    xGM.SetGlobalBuffer((__gm__ X_T*)x);
    weightGM.SetGlobalBuffer((__gm__ X_T*)weight);
    yGM.SetGlobalBuffer((__gm__ X_T*)y);
    if constexpr (NEED_BIAS) {
        biasGM.SetGlobalBuffer((__gm__ BIAS_T*)bias);
    }

    mmOutGM.SetGlobalBuffer((__gm__ X_T*)workspace);
    uint64_t localBmmOut = MAX(localE.tileLen, localE.tailLen) * MAX(localC.tileLen, localC.tailLen) * H;
    uint64_t nonLocalBmmOut = MAX(nonLocalE.tileLen, nonLocalE.tailLen) * MAX(nonLocalC.tileLen, nonLocalC.tailLen) * epMinusOne * H;
    reshapeLocalOutGM.SetGlobalBuffer((__gm__ X_T*)mmOutGM.GetPhyAddr() +
                                      MAX(localBmmOut, nonLocalBmmOut));
    reshapeNonLocalOutGM.SetGlobalBuffer((__gm__ X_T*)reshapeLocalOutGM.GetPhyAddr() +
                                         E_ep * C * H);
    reduceScatterLocalOutGM.SetGlobalBuffer((__gm__ X_T*)reshapeNonLocalOutGM.GetPhyAddr() +
                                            (epMinusOne * E_ep * C * H));
    reduceScatterNonLocalOutGM.SetGlobalBuffer((__gm__ X_T*)reduceScatterLocalOutGM.GetPhyAddr() + (E_ep * C * H_tp));
    addTransposeOutGM.SetGlobalBuffer((__gm__ X_T*)reduceScatterNonLocalOutGM.GetPhyAddr() +
                                      (epMinusOne * E_ep * C * H_tp));

    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::InitTilingData()
{
    E_ep = tilingData->commonTiling.EOverEp;
    C = tilingData->commonTiling.C;
    H = tilingData->commonTiling.H;
    H_tp = tilingData->commonTiling.HOverTp;
    alignedH = (H + SIXTEEN_ALIGN - 1U) / SIXTEEN_ALIGN * SIXTEEN_ALIGN;
    alignedH_tp = (H_tp + SIXTEEN_ALIGN - 1U) / SIXTEEN_ALIGN * SIXTEEN_ALIGN;
    M_tp = tilingData->commonTiling.MOverTp;
    epRankSize = tilingData->commonTiling.epGroupSize;
    epMinusOne = epRankSize - 1;
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
    localE.tailCnt = tilingData->commonTiling.localTileE.tailCnt;
    localE.tailLen = tilingData->commonTiling.localTileE.tailLen;
    localC.tileCnt = tilingData->commonTiling.localTileC.tileCnt;
    localC.tileLen = tilingData->commonTiling.localTileC.tileLen;
    localC.tailCnt = tilingData->commonTiling.localTileC.tailCnt;
    localC.tailLen = tilingData->commonTiling.localTileC.tailLen;
    nonLocalE.tileCnt = tilingData->commonTiling.domesticTileE.tileCnt;
    nonLocalE.tileLen = tilingData->commonTiling.domesticTileE.tileLen;
    nonLocalE.tailCnt = tilingData->commonTiling.domesticTileE.tailCnt;
    nonLocalE.tailLen = tilingData->commonTiling.domesticTileE.tailLen;
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
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::InitBuffers()
{
    pipe->InitBuffer(tBuf, ubSize);
    bmmOutTmp = tBuf.Get<X_T>();
    rsOutTmp = tBuf.Get<X_T>(ubCapacity);
    uint64_t offset = ubCapacity * dataLen;
    if constexpr (NEED_BIAS) {
        biasTmp = tBuf.Get<BIAS_T>()[offset / biasLen];
        offset += ubCapacity * biasLen;
        addOutTmp = tBuf.Get<X_T>()[offset / dataLen];
        offset += ubCapacity * dataLen;
        if (AscendC::IsSameType<X_T, bfloat16_t>::value) {
            rsOutCastTmp = tBuf.Get<float>()[offset / FP32_DATASIZE];
        }
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::Process()
{
    NonLocalCommunication();
    NonLocalCalculation();
    LocalCommunication();
    LocalCalculation();

    if ASCEND_IS_AIV {
        hcclReduceScatter.Finalize();
        hcclAlltoAll.Finalize();
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::NonLocalCommunication()
{
    bool firstFlag = true;
    bool lastFlag = false;
    uint64_t rsDoneCnt = 0U;
    uint64_t alltoallDoneCnt = 0U;
    // Loop over E_cnt
    for (uint64_t eIdx = 0U; eIdx < nonLocalECnt; eIdx++) {
        bool isETail = (eIdx >= nonLocalE.tileCnt);
        uint64_t validETileCnt = MIN(eIdx, nonLocalE.tileCnt);
        uint64_t validETailCnt = isETail ? eIdx - nonLocalE.tileCnt : 0;
        uint64_t validELen = isETail ? nonLocalE.tailLen : nonLocalE.tileLen;
        // Input offset for RS
        uint64_t reshapeOutOffset = validETileCnt * C * tpRankSize * epMinusOne * nonLocalE.tileLen +\
                    validETailCnt * C * tpRankSize * epMinusOne * nonLocalE.tailLen;
        // Output offset for RS
        uint64_t reduceScatterOutOffset = validETileCnt * C * epMinusOne * nonLocalE.tileLen +\
                    validETailCnt * C * epMinusOne * nonLocalE.tailLen;
        // Input offset for AlltoAll
        uint64_t addTransposeOffset = validETileCnt * C * epRankSize * nonLocalE.tileLen +\
                    validETailCnt * C * epRankSize * nonLocalE.tailLen;
        // Output offset for AlltoAll
        uint64_t yGMOffset = validETileCnt * C * nonLocalE.tileLen + validETailCnt *\
                    C * nonLocalE.tailLen;
        // Loop over C_cnt
        for (uint64_t cIdx = 0U; cIdx < nonLocalCCnt; cIdx++) {
            if ((eIdx == (nonLocalECnt - 1)) && (cIdx == (nonLocalCCnt - 1))) {
                lastFlag = true;
            }
            bool isCTail = (cIdx >= nonLocalC.tileCnt);
            uint64_t validCTileCnt = MIN(cIdx, nonLocalC.tileCnt);
            uint64_t validCTailCnt = isCTail ? cIdx - nonLocalC.tileCnt : 0;
            uint64_t validCLen = isCTail ? nonLocalC.tailLen : nonLocalC.tileLen;
            uint64_t reshapeOutCOffset = reshapeOutOffset + validCTileCnt * tpRankSize * epMinusOne *\
                        validELen * nonLocalC.tileLen + validCTailCnt * tpRankSize * epMinusOne *\
                        validELen * nonLocalC.tailLen;
            reshapeOutCOffset *= H_tp;
            uint64_t reduceScatterOutCOffset = reduceScatterOutOffset + validCTileCnt * epMinusOne *\
                        validELen * nonLocalC.tileLen + validCTailCnt * epMinusOne * validELen *\
                        nonLocalC.tailLen;
            reduceScatterOutCOffset *= H_tp;
            uint64_t addTransposeCOffset = addTransposeOffset + validCTileCnt * epRankSize *\
                        validELen * nonLocalC.tileLen + validCTailCnt * epRankSize * validELen *\
                        nonLocalC.tailLen;
            addTransposeCOffset *= H_tp;
            uint64_t yGMCOffset = yGMOffset + validCTileCnt * nonLocalC.tileLen + validCTailCnt *\
                        nonLocalC.tailLen;
            yGMCOffset *= H_tp;

            for (uint64_t i = 0U; i < epRankSize; i++) {
                alltoallVSendCount[i] = validCLen * validELen * H_tp;
                alltoallVSendStride[i] = i * alltoallVSendCount[i];
                alltoallVRecvStride[i] = i * E_ep * C * H_tp;
            }
            uint64_t rsDataCnt = epMinusOne * validELen * validCLen * H_tp;
            if (firstFlag) {
                // 如果是第一次，仅prepare RS的任务0
                if ASCEND_IS_AIV {
                    reduceScatterHandleList[rsDoneCnt++] = hcclReduceScatter.ReduceScatter<false>(
                        (__gm__ uint8_t*)reshapeNonLocalOutGM[reshapeOutCOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)reduceScatterNonLocalOutGM[reduceScatterOutCOffset].GetPhyAddr(),
                        rsDataCnt, hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
                }
            } else {
                // 如果不是第一次，让RS任务都依赖前一轮AlltoAll做完（上一轮循环结束），让RS任务下发
                // 此时因为设置的依赖条件，即使下发也不会执行
                if ASCEND_IS_AIV {
                    hcclReduceScatter.InterHcclGroupSync(HCCL_GROUP_ID_0, alltoallList[alltoallDoneCnt - 1]);
                    reduceScatterHandleList[rsDoneCnt++] = hcclReduceScatter.ReduceScatter<true>(
                        (__gm__ uint8_t*)reshapeNonLocalOutGM[reshapeOutCOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)reduceScatterNonLocalOutGM[reduceScatterOutCOffset].GetPhyAddr(),
                        rsDataCnt, hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
                }
            }

            if (!lastFlag) {
                if ASCEND_IS_AIV {
                    // 将所有AlltoAll任务prepare但不下发，因为要等前面的trans完成才能做AlltoAll
                    // 所以实际下发会等到计算部分的trans做完后手动下发
                    alltoallList[alltoallDoneCnt++] = hcclAlltoAll.AlltoAllV<false>(
                        (__gm__ uint8_t*)addTransposeOutGM[addTransposeCOffset].GetPhyAddr(),
                        alltoallVSendCount, alltoallVSendStride, hcclDataType,
                        (__gm__ uint8_t*)yGM[yGMCOffset].GetPhyAddr(),
                        alltoallVSendCount, alltoallVRecvStride, hcclDataType, 1);
                }
            } else {
                nonLocalLastAddTransposeOffset = addTransposeCOffset;
                nonLocalLastYGMOffset = yGMCOffset;
            }
            firstFlag = false;
        }
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::NonLocalCalculation()
{
    bool firstFlag = true;
    uint64_t rsDoneCnt = 0U;
    uint64_t alltoallDoneCnt = 0U;
    
    // Loop over E_cnt
    for (uint64_t eIdx = 0U; eIdx < nonLocalECnt; eIdx++) {
        bool isETail = (eIdx >= nonLocalE.tileCnt);
        uint64_t validETileCnt = MIN(eIdx, nonLocalE.tileCnt);
        uint64_t validETailCnt = isETail ? eIdx - nonLocalE.tileCnt : 0;
        uint64_t validELen = isETail ? nonLocalE.tailLen : nonLocalE.tileLen;
        uint64_t aCalcOffset = validETileCnt * C * epRankSize * nonLocalE.tileLen +\
                    validETailCnt * C * epRankSize * nonLocalE.tailLen;
        uint64_t bCalcOffset = validETileCnt * nonLocalE.tileLen * M_tp + validETailCnt *\
                    nonLocalE.tailLen * M_tp;
        uint64_t reshapeOutOffset = validETileCnt * C * tpRankSize * epMinusOne * nonLocalE.tileLen +\
                    validETailCnt * C * tpRankSize * epMinusOne * nonLocalE.tailLen;
        uint64_t reduceScatterOutOffset = validETileCnt * C * epMinusOne * nonLocalE.tileLen +\
                    validETailCnt * C * epMinusOne * nonLocalE.tailLen;
        uint64_t addTransposeOffset = validETileCnt * C * epRankSize * nonLocalE.tileLen +\
                    validETailCnt * C * epRankSize * nonLocalE.tailLen;
        uint64_t biasOffset = validETileCnt * nonLocalE.tileLen + validETailCnt * nonLocalE.tailLen;
        biasOffset *= H_tp;
        // Loop over C_cnt
        for (uint64_t cIdx = 0U; cIdx < nonLocalCCnt; cIdx++) {
            bool isCTail = (cIdx >= nonLocalC.tileCnt);
            uint64_t validCTileCnt = MIN(cIdx, nonLocalC.tileCnt);
            uint64_t validCTailCnt = isCTail ? cIdx - nonLocalC.tileCnt : 0;
            uint64_t validCLen = isCTail ? nonLocalC.tailLen : nonLocalC.tileLen;
            uint64_t aAddrCOffset = aCalcOffset + validCTileCnt * nonLocalC.tileLen +\
                        validCTailCnt * nonLocalC.tailLen;
            uint64_t cCalcOffset = 0U;
            uint64_t reshapeOutCOffset = reshapeOutOffset + validCTileCnt * tpRankSize * epMinusOne *\
                        validELen * nonLocalC.tileLen + validCTailCnt * tpRankSize * epMinusOne *\
                        validELen * nonLocalC.tailLen;
            reshapeOutCOffset *= H_tp;
            uint64_t reduceScatterOutCOffset = reduceScatterOutOffset + validCTileCnt * epMinusOne *\
                        validELen * nonLocalC.tileLen + validCTailCnt * epMinusOne * validELen *\
                        nonLocalC.tailLen;
            reduceScatterOutCOffset *= H_tp;
            uint64_t addTransposeCOffset = addTransposeOffset + validCTileCnt * epRankSize * validELen *\
                        nonLocalC.tileLen + validCTailCnt * epRankSize * validELen * nonLocalC.tailLen;
            addTransposeCOffset *= H_tp;
            if ASCEND_IS_AIC {
                // Loop over Ep
                for (uint64_t i = 0U; i < epRankSize; i++) {
                    if (i == epRankId) { // MatrixA跳过local块
                        continue;
                    }
                    uint64_t aAddrEpOffset = aAddrCOffset + i * C;
                    uint64_t curEpIdx = (i > epRankId) ? (i - 1) : i;
                    uint64_t cAddrEpOffset = cCalcOffset + curEpIdx * validCLen * validELen;
                    
                    // Loop over E_len
                    for (uint64_t j = 0U; j < validELen; j++) {
                        uint64_t aAddrOffset = aAddrEpOffset + j * epRankSize * C;
                        uint64_t bAddrOffset = bCalcOffset + j * M_tp;
                        uint64_t cAddrOffset = cAddrEpOffset + j * validCLen;
                        aAddrOffset *= M_tp;
                        bAddrOffset *= H;
                        cAddrOffset *= H;

                        pipe->Reset();
                        bmmV3.Init((__gm__ uint8_t*)xGM[aAddrOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)weightGM[bAddrOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)mmOutGM[cAddrOffset].GetPhyAddr(), nullptr, nullptr, nullptr,
                        (isCTail ? &nonLocalTailTiling : &nonLocalTileTiling), pipe);
                        bmmV3.Process();
                    }
                }
            }
            SyncAll<false>();

            pipe->Reset();
            InitBuffers();

            if ASCEND_IS_AIV {
                ReshapeAfterBmm(validELen, validCLen, epMinusOne, reshapeOutCOffset, false);
            }
            SyncAll<false>();

            if (firstFlag) {
                if ASCEND_IS_AIV {
                    SyncAll<true>();
                    hcclReduceScatter.Commit(reduceScatterHandleList[0]);
                }
                firstFlag = false;
            } else {
                if ASCEND_IS_AIV {
                    hcclReduceScatter.Wait(reduceScatterHandleList[rsDoneCnt++]);
                }
                SyncAll<false>();

                // 上一轮的AddTranspose，所有的偏移都用前一轮循环的
                if ASCEND_IS_AIV {
                    AddTransposeAfterRS(prevReduceScatterOutCOffset, prevBiasOffset, prevAddTransposeCOffset,
                        prevValidELen, prevValidCLen, prevValidEpIdx, prevIsLocal);
                }
                SyncAll<true>();
                // alltoall commit
                if ASCEND_IS_AIV {
                    hcclAlltoAll.Commit(alltoallList[alltoallDoneCnt++]);
                }
            }
            prevReduceScatterOutCOffset = reduceScatterOutCOffset;
            prevBiasOffset = biasOffset;
            prevAddTransposeCOffset = addTransposeCOffset;
            prevValidELen = validELen;
            prevValidCLen = validCLen;
            prevValidEpIdx = epMinusOne;
            prevIsLocal = false;
        }
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::LocalCommunication()
{
    bool firstFlag = true;
    uint64_t rsDoneCnt = nonLocalECnt * nonLocalCCnt;
    uint64_t alltoallDoneCnt = nonLocalECnt * nonLocalCCnt - 1;
    for (uint64_t eIdx = 0U; eIdx < localECnt; eIdx++) {
        bool isETail = (eIdx >= localE.tileCnt);
        uint64_t validETileCnt = MIN(eIdx, localE.tileCnt);
        uint64_t validETailCnt = isETail ? eIdx - localE.tileCnt : 0;
        uint64_t validELen = isETail ? localE.tailLen : localE.tileLen;
        uint64_t reshapeOutOffset = validETileCnt * C * tpRankSize * localE.tileLen +\
                    validETailCnt * C * tpRankSize * localE.tailLen;
        uint64_t reduceScatterOutOffset = validETileCnt * C * localE.tileLen +\
                    validETailCnt * C * localE.tailLen;
        for (uint64_t cIdx = 0U; cIdx < localCCnt; cIdx++) {
            bool isCTail = (cIdx >= localC.tileCnt);
            uint64_t validCTileCnt = MIN(cIdx, localC.tileCnt);
            uint64_t validCTailCnt = isCTail ? cIdx - localC.tileCnt : 0;
            uint64_t validCLen = isCTail ? localC.tailLen : localC.tileLen;
            uint64_t reshapeOutCOffset = reshapeOutOffset + validCTileCnt * tpRankSize *\
                        validELen * localC.tileLen + validCTailCnt * tpRankSize *\
                        validELen * localC.tailLen;
            reshapeOutCOffset *= H_tp;
            uint64_t reduceScatterOutCOffset = reduceScatterOutOffset + validCTileCnt *\
                        validELen * localC.tileLen + validCTailCnt * validELen *\
                        localC.tailLen;
            reduceScatterOutCOffset *= H_tp;
            uint64_t rsDataCnt = validELen * validCLen * H_tp;
            if (firstFlag) {
                if ASCEND_IS_AIV {
                    reduceScatterHandleList[rsDoneCnt++] = hcclReduceScatter.ReduceScatter<false>(
                        (__gm__ uint8_t*)reshapeLocalOutGM[reshapeOutCOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)reduceScatterLocalOutGM[reduceScatterOutCOffset].GetPhyAddr(),
                        rsDataCnt, hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
                    firstFlag = false;
                    alltoallList[alltoallDoneCnt++] = hcclAlltoAll.AlltoAllV<false>(
                        (__gm__ uint8_t*)addTransposeOutGM[nonLocalLastAddTransposeOffset].GetPhyAddr(),
                        alltoallVSendCount, alltoallVSendStride, hcclDataType,
                        (__gm__ uint8_t*)yGM[nonLocalLastYGMOffset].GetPhyAddr(),
                        alltoallVSendCount, alltoallVRecvStride, hcclDataType, 1);
                }
            } else {
                if ASCEND_IS_AIV {
                    reduceScatterHandleList[rsDoneCnt++] = hcclReduceScatter.ReduceScatter<false>(
                        (__gm__ uint8_t*)reshapeLocalOutGM[reshapeOutCOffset].GetPhyAddr(),
                        (__gm__ uint8_t*)reduceScatterLocalOutGM[reduceScatterOutCOffset].GetPhyAddr(),
                        rsDataCnt, hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
                }
            }
        }
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::LocalCalculation()
{
    bool firstFlag = true;
    uint64_t rsDoneCnt = nonLocalECnt * nonLocalCCnt;
    uint64_t alltoallDoneCnt = nonLocalECnt * nonLocalCCnt - 1;
    uint64_t aEpOffset = epRankId * C;
    uint64_t loopCnt = 0U;
    for (uint64_t eIdx = 0U; eIdx < localECnt; eIdx++) {
        bool isETail = (eIdx >= localE.tileCnt);
        uint64_t validETileCnt = MIN(eIdx, localE.tileCnt);
        uint64_t validETailCnt = isETail ? eIdx - localE.tileCnt : 0;
        uint64_t validELen = isETail ? localE.tailLen : localE.tileLen;
        uint64_t aCalcOffset = aEpOffset + validETileCnt * C * epRankSize * localE.tileLen +\
                    validETailCnt * C * epRankSize * localE.tailLen;
        uint64_t bCalcOffset = validETileCnt * localE.tileLen * M_tp + validETailCnt *\
                    localE.tailLen * M_tp;
        uint64_t reshapeOutOffset = validETileCnt * C * tpRankSize * localE.tileLen +\
                    validETailCnt * C * tpRankSize * localE.tailLen;
        uint64_t reduceScatterOutOffset = validETileCnt * C * localE.tileLen +\
                    validETailCnt * C * localE.tailLen;
        uint64_t yOutOffset = validETileCnt * C * localE.tileLen + validETailCnt * C *\
                    localE.tailLen;
        uint64_t biasOffset = validETileCnt * localE.tileLen + validETailCnt * localE.tailLen;
        biasOffset *= H_tp;
        for (uint64_t cIdx = 0U; cIdx < localCCnt; cIdx++) {
            loopCnt += 1;
            bool isCTail = (cIdx >= localC.tileCnt);
            uint64_t validCTileCnt = MIN(cIdx, localC.tileCnt);
            uint64_t validCTailCnt = isCTail ? cIdx - localC.tileCnt : 0;
            uint64_t validCLen = isCTail ? localC.tailLen : localC.tileLen;
            uint64_t aAddrCOffset = aCalcOffset + validCTileCnt * localC.tileLen +\
                        validCTailCnt * localC.tailLen;
            uint64_t cCalcOffset = 0U;
            uint64_t reshapeOutCOffset = reshapeOutOffset + validCTileCnt * tpRankSize *\
                        validELen * localC.tileLen + validCTailCnt * tpRankSize *\
                        validELen * localC.tailLen;
            reshapeOutCOffset *= H_tp;
            uint64_t reduceScatterOutCOffset = reduceScatterOutOffset + validCTileCnt *\
                        validELen * localC.tileLen + validCTailCnt * validELen *\
                        localC.tailLen;
            reduceScatterOutCOffset *= H_tp;
            uint64_t yOutCOffset = yOutOffset + validCTileCnt * localC.tileLen + validCTailCnt *\
                        localC.tailLen;
            yOutCOffset *= H_tp;
            yOutCOffset += epRankId * E_ep * C * H_tp;

            if ASCEND_IS_AIC {
                // Loop over E_len
                for (uint64_t i = 0U; i < validELen; i++) {
                    uint64_t aAddrOffset = aAddrCOffset + i * epRankSize * C;
                    aAddrOffset *= M_tp;
                    uint64_t bAddrOffset = bCalcOffset + i * M_tp;
                    bAddrOffset *= H;
                    uint64_t cAddrOffset = cCalcOffset + i * validCLen;
                    cAddrOffset *= H;
                    pipe->Reset();
                    bmmV3.Init((__gm__ uint8_t*)xGM[aAddrOffset].GetPhyAddr(),
                    (__gm__ uint8_t*)weightGM[bAddrOffset].GetPhyAddr(),
                    (__gm__ uint8_t*)mmOutGM[cAddrOffset].GetPhyAddr(), nullptr, nullptr, nullptr,
                    (isCTail ? &localTailTiling : &localTileTiling), pipe);
                    bmmV3.Process();
                }
            }
            SyncAll<false>();
            pipe->Reset();
            InitBuffers();

            if ASCEND_IS_AIV {
                ReshapeAfterBmm(validELen, validCLen, 1, reshapeOutCOffset, true);
            }
            SyncAll<false>();

            if ASCEND_IS_AIV {
                hcclReduceScatter.Commit(reduceScatterHandleList[rsDoneCnt]);
            }
            if ASCEND_IS_AIV {
                hcclReduceScatter.Wait(reduceScatterHandleList[rsDoneCnt - 1]);
            }
            SyncAll<false>();
            if (loopCnt == 2) {
                hcclAlltoAll.Wait(alltoallList[alltoallDoneCnt]);
            }
            SyncAll<false>();
            if ASCEND_IS_AIV {
                AddTransposeAfterRS(prevReduceScatterOutCOffset, prevBiasOffset, prevAddTransposeCOffset,
                    prevValidELen, prevValidCLen, prevValidEpIdx, prevIsLocal);
            }
            SyncAll<false>();
            if (firstFlag) {
                if ASCEND_IS_AIV {
                    hcclAlltoAll.Commit(alltoallList[alltoallDoneCnt]);
                }
            }
            SyncAll<false>();

            firstFlag = false;
            prevReduceScatterOutCOffset = reduceScatterOutCOffset;
            prevBiasOffset = biasOffset;
            prevAddTransposeCOffset = yOutCOffset;
            prevValidELen = validELen;
            prevValidCLen = validCLen;
            prevValidEpIdx = 1;
            prevIsLocal = true;
            rsDoneCnt++;
        }
    }
    if ASCEND_IS_AIV {
        hcclReduceScatter.Wait(reduceScatterHandleList[rsDoneCnt - 1]);
    }
    SyncAll<false>();
    if (loopCnt == 1) {
        hcclAlltoAll.Wait(alltoallList[alltoallDoneCnt]);
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        AddTransposeAfterRS(prevReduceScatterOutCOffset, prevBiasOffset, prevAddTransposeCOffset,
            prevValidELen, prevValidCLen, prevValidEpIdx, prevIsLocal);
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::ReshapeAfterBmm(uint64_t validELen,
    uint64_t validCLen, uint64_t validEpIdx, uint64_t reshapeOutOffset, bool isLocal)
{
    DataCopyExtParams dataCopyInParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<X_T> dataCopyInPadParams = {false, 0U, 0U, 0U};
    uint64_t singleM = validEpIdx * validELen * validCLen;
    uint64_t totalM = singleM * tpRankSize; // 以H_tp为单位计算的总行数
    uint64_t tileLen = (totalM + aivNum - 1) / aivNum;  // round up, 确保不会漏掉尾块行
    uint64_t curRow = tileLen * tmpBlockIdx;
    uint64_t endRow = MIN(tileLen * (tmpBlockIdx + 1U), totalM);
    uint64_t startSkipTp = curRow / singleM;
    uint64_t curTpMaxRow = (startSkipTp + 1UL) * singleM;
    uint64_t curDataOffset = curRow * H_tp;
    uint64_t endDataOffset = MIN(tileLen * H_tp * (tmpBlockIdx + 1U), totalM * H_tp);
    uint64_t ubCapacityForReshape = ubSize / dataLen;
    uint64_t perMaxRow = MIN(ubCapacityForReshape / alignedH_tp, MAX_BLOCK_CNT);
    uint64_t dataCnt = 0UL;
    uint64_t syncCnt = 0U;

    while (curDataOffset < endDataOffset) {
        syncCnt += 1;
        uint64_t oriCurRow = curRow % singleM;
        uint64_t startSkipTp = curRow / singleM;
        uint64_t inlineGap = curDataOffset - (curRow * H_tp);
        uint64_t phyCurDataOffset = oriCurRow * H + startSkipTp * H_tp + inlineGap;
        if (perMaxRow == 0UL) { // 行超大一次搬不完一行，分片切列，一片处理的最大数据量为 ubCapacityForReshape
            if ((curRow + 1UL) * H_tp - curDataOffset > ubCapacityForReshape) { // 一次最大搬运不跨行，进行最大搬运
                dataCnt = ubCapacityForReshape;
            } else {                                                  //  若跨行，只搬完当前行，不搬下一行的
                dataCnt = (curRow + 1UL) * H_tp - curDataOffset;
                curRow += 1UL;
                curTpMaxRow += ((curRow == curTpMaxRow) ? singleM : 0);
            }
            dataCopyInParams.blockLen = static_cast<uint32_t>(dataCnt * dataLen);
            dataCopyOutParams.blockLen = static_cast<uint32_t>(dataCnt * dataLen);
        } else {
            if ((MIN(curTpMaxRow * H_tp, endDataOffset) - curDataOffset) > (perMaxRow * H_tp)) { // 一次最大搬运不会跨Tp块且不会超过分配的任务，进行最大行数搬运
                dataCnt = perMaxRow * H_tp;
                curRow += perMaxRow;
            } else {                                                       // 否则，TP块和任务块，谁先结束就搬到谁为止
                dataCnt = MIN(curTpMaxRow * H_tp, endDataOffset) - curDataOffset;
                curRow = MIN(curTpMaxRow, endRow);
                curTpMaxRow += ((curRow == curTpMaxRow) ? singleM : 0);
            }
            dataCopyInParams.blockCount = static_cast<uint16_t>(dataCnt / H_tp);
            dataCopyInParams.blockLen = static_cast<uint32_t>(H_tp * dataLen);
            dataCopyInParams.srcStride = static_cast<uint32_t>((H - H_tp) * dataLen);

            dataCopyOutParams.blockCount = static_cast<uint16_t>(dataCnt / H_tp);
            dataCopyOutParams.blockLen = static_cast<uint32_t>(H_tp * dataLen);
        }

        GlobalTensor<X_T> dataCopyInGM = mmOutGM[phyCurDataOffset];
        GlobalTensor<X_T> dataCopyOutGM = (isLocal ? reshapeLocalOutGM[reshapeOutOffset + curDataOffset] :
            reshapeNonLocalOutGM[reshapeOutOffset + curDataOffset]);
        if (syncCnt > 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
        DataCopyPad(bmmOutTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        DataCopyPad(dataCopyOutGM, bmmOutTmp, dataCopyOutParams);

        curDataOffset += dataCnt;
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::AddTransposeAfterRS(uint64_t rsOutOffset,
    uint64_t biasOffset, uint64_t addTransposeOffset, uint64_t validELen, uint64_t validCLen,
    uint64_t validEpIdx, bool isLocal)
{
    uint64_t rsBlockLen = validELen * validCLen;
    uint64_t totalRows = validEpIdx * rsBlockLen;
    uint64_t rowsPerCore = (totalRows + aivNum - 1) / aivNum;
    uint64_t curRow = tmpBlockIdx * rowsPerCore;
    uint64_t endRow = MIN(rowsPerCore * (tmpBlockIdx + 1U), totalRows);
    uint64_t curDataOffset = curRow * H_tp;
    uint64_t endDataOffset = endRow * H_tp;
    uint64_t perMaxRow = ubCapacity / alignedH_tp;
    uint64_t curBlockMaxRow = ((curRow / validCLen) + 1) * validCLen;
    uint64_t dataCnt = 0UL;
    uint64_t curTileCBlock = 0U;
    uint64_t biasRowOffset = 0U;
    uint64_t biasInlineOffset = 0U;
    uint64_t phyBiasOffset = 0U;
    uint64_t skipLocalOffset = 0U;
    uint64_t curEpIdx = 0U;
    uint64_t phyRsOutOffset = 0U;
    bool isFirst = true;
    bool isLargeH = false;

    while (curDataOffset < endDataOffset) {
        phyRsOutOffset = rsOutOffset + curDataOffset;
        curTileCBlock = curRow / validCLen;
        biasRowOffset = curTileCBlock % validELen;
        biasInlineOffset = curDataOffset - (curRow * H_tp);
        phyBiasOffset = biasOffset + biasRowOffset * H_tp + biasInlineOffset;
        curEpIdx = curRow / rsBlockLen;
        skipLocalOffset = (!isLocal && (curEpIdx >= epRankId)) ? (rsBlockLen * H_tp) : 0;
        uint64_t phyAddTransposeOffset = addTransposeOffset + skipLocalOffset + curDataOffset;

        if (perMaxRow == 0UL) { // 行超大一次搬不完一行，分片切列，一片处理的最大数据量为 ubCapacity
            isLargeH = true;
            if ((curRow + 1UL) * H_tp - curDataOffset > ubCapacity) { // 一次最大搬运不跨行，进行最大搬运
                dataCnt = ubCapacity;
            } else {                                                  //  若跨行，只搬完当前行，不搬下一行的
                dataCnt = (curRow + 1UL) * H_tp - curDataOffset;
                curRow += 1UL;
                curBlockMaxRow += ((curRow == curBlockMaxRow) ? validCLen : 0);
            }
        } else {
            if ((curBlockMaxRow * H_tp - curDataOffset) > (perMaxRow * H_tp)) { // 一次最大搬运不会跨block块，进行最大行数搬运
                dataCnt = MIN(perMaxRow * H_tp, endDataOffset - curDataOffset);
                curRow += ((dataCnt + H_tp - 1) / H_tp);
            } else {                                                       // 若跨block块，只搬完当前block块
                dataCnt = MIN(curBlockMaxRow * H_tp, endDataOffset) - curDataOffset;
                curRow = MIN(endRow, curBlockMaxRow);
                curBlockMaxRow += ((curRow == curBlockMaxRow) ? validCLen : 0);
            }
        }

        if ASCEND_IS_AIV {
            AddTranspose(phyRsOutOffset, phyBiasOffset, phyAddTransposeOffset, dataCnt, isLocal, isFirst, isLargeH);
            isFirst = false;
        }
        curDataOffset += dataCnt;
    }
}

template <typename BMMRSATAT>
__aicore__ inline void BatchMatMulReduceScatterAlltoAllShard0<BMMRSATAT>::AddTranspose(uint64_t rsOutOffset,
    uint64_t biasOffset, uint64_t addTransposeOffset, uint64_t dataCnt, bool isLocal, bool isFirst, bool isLargeH)
{
    GlobalTensor<X_T> rsOutStartGM = (isLocal ? reduceScatterLocalOutGM[rsOutOffset] :
        reduceScatterNonLocalOutGM[rsOutOffset]);
    GlobalTensor<X_T> transOutGM = (isLocal ? yGM[addTransposeOffset] : addTransposeOutGM[addTransposeOffset]);
    if (!isFirst) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    }
    if constexpr (NEED_BIAS) {
        // Add: left operand Copy-In
        if ((dataCnt > H_tp) && ((H_tp * dataLen) % UB_ADDR_ALIGN != 0)) {
            const DataCopyExtParams copyInParams{static_cast<uint16_t>(dataCnt / H_tp),
                                                static_cast<uint32_t>(H_tp * dataLen), 0U, 0U, 0U};
            const DataCopyPadExtParams<X_T> copyInPadParams{false, 0U, 0U, 0U};
            DataCopyPad(rsOutTmp, rsOutStartGM, copyInParams, copyInPadParams);
        } else {
            const DataCopyExtParams copyInParams{1U, static_cast<uint32_t>(dataCnt * dataLen), 0U, 0U, 0U};
            const DataCopyPadExtParams<X_T> copyInPadParams{false, 0U, 0U, 0U};
            DataCopyPad(rsOutTmp, rsOutStartGM, copyInParams, copyInPadParams);
        }
        PipeBarrier<PIPE_MTE2>();
        // Add: right operand Copy-In
        const DataCopyExtParams copyInParamsBias{1U, static_cast<uint32_t>(MIN(dataCnt, H_tp) * biasLen), 0U, 0U, 0U};
        const DataCopyPadExtParams<BIAS_T> copyInPadParamsBias{false, 0U, 0U, 0U};
        DataCopyPad(biasTmp, biasGM[biasOffset], copyInParamsBias, copyInPadParamsBias);
        // Cast to FP32 if dtype is BF16
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        if constexpr (AscendC::IsSameType<X_T, bfloat16_t>::value) {
            // BF16 cast成 FP32
            if (isLargeH) {
                Cast(rsOutCastTmp, rsOutTmp, RoundMode::CAST_NONE, dataCnt);
            } else {
                Cast(rsOutCastTmp, rsOutTmp, RoundMode::CAST_NONE, (dataCnt + H_tp - 1U) / H_tp * alignedH_tp);
            }
            PipeBarrier<PIPE_V>();
            if (dataCnt > H_tp) { // 多行add bias, dataCnt为H_tp的倍数
                for (int i = 0; i < (dataCnt / H_tp); i++) {
                    Add(rsOutCastTmp[i * alignedH_tp], rsOutCastTmp[i * alignedH_tp], biasTmp, H_tp);
                }
            } else { // 单行以内, add部分bias
                Add(rsOutCastTmp, rsOutCastTmp, biasTmp, dataCnt);
            }
            PipeBarrier<PIPE_V>();
            // cast回BF16
            if (isLargeH) {
                Cast(addOutTmp, rsOutCastTmp, RoundMode::CAST_ROUND, dataCnt);
            } else {
                Cast(addOutTmp, rsOutCastTmp, RoundMode::CAST_ROUND, (dataCnt + H_tp - 1U) / H_tp * alignedH_tp);
            }
        } else {
            if (dataCnt > H_tp) { // 多行add bias
                for (int i = 0; i < (dataCnt / H_tp); i++) {
                    Add(addOutTmp[i * alignedH_tp], rsOutTmp[i * alignedH_tp], biasTmp, H_tp);
                }
            } else { // 单行以内, add部分bias
                Add(addOutTmp, rsOutTmp, biasTmp, dataCnt);
            }
        }
        PipeBarrier<PIPE_V>();
        // Transpose
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));

        if ((dataCnt > H_tp) && ((H_tp * dataLen) % UB_ADDR_ALIGN != 0)) {
            const DataCopyExtParams copyOutParams{static_cast<uint16_t>(dataCnt / H_tp),
                                                    static_cast<uint32_t>(H_tp * dataLen), 0U, 0U, 0U};
            DataCopyPad(transOutGM, addOutTmp, copyOutParams);
        } else {
            const DataCopyExtParams copyOutParams{1U, static_cast<uint32_t>(dataCnt * dataLen), 0U, 0U, 0U};
            DataCopyPad(transOutGM, addOutTmp, copyOutParams);
        }
    } else {
        // Copy-In + transpose
        const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(dataCnt * dataLen), 0U, 0U, 0U};
        const DataCopyPadExtParams<X_T> copyInPadParams{false, 0U, 0U, 0U};
        DataCopyPad(rsOutTmp, rsOutStartGM, copyParams, copyInPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        // Copy-Out
        DataCopyPad(transOutGM, rsOutTmp, copyParams);
    }
}

#endif  // BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_SHARD_ZERO_H