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
 * \file allto_all_all_gather_batch_mat_mul.h
 * \brief
 */

#ifndef MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_H
#define MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if __has_include("../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h")
#include "../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h"
#else
#include "../../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h"
#endif

constexpr uint32_t MAX_HCCL_HANDLE = 32U;
constexpr uint32_t SIZE_OF_FLOAT32 = 4U;
constexpr uint32_t SIZE_OF_FLOAT16 = 2U;
constexpr uint32_t MAX_EP_RANK_SIZE = 32U;

using namespace AscendC;
using namespace matmul;

enum class AlltoAllAllGatherBatchMatmulActType : uint32_t {
    ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE = 0,
    ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU = 1,
    ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU = 2,
    ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_RELU = 3,
    ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU = 4,
    ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GEGLU = 5,
    ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SWIGLU = 6,
    ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_REGLU = 7
};

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
class AlltoAllAllGatherBatchMatMul{
public:
    __aicore__ inline AlltoAllAllGatherBatchMatMul(){};
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM, GM_ADDR y2GM, GM_ADDR y3GM,
                                GM_ADDR workspaceGM, TPipe *pipe, AlltoAllAllGatherBatchMatMulTilingData* tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void GetTilingData(AlltoAllAllGatherBatchMatMulTilingData *tilingData);
    __aicore__ inline void InitTpipe();
    __aicore__ inline void LocalCommunication();
    __aicore__ inline void LocalCalculation();
    __aicore__ inline void NonLocalCommunication();
    __aicore__ inline void NonLocalCalculation();

    __aicore__ inline void TransposeBeforeBMM(bool isLocal, bool isCTail, GlobalTensor<DataType1> dataCopyInBaseGM,
                                              GlobalTensor<DataType1> y2BaseGM);
    __aicore__ inline void TransposeAfterBMM(bool isLocal, bool isCTail, GlobalTensor<DataType1> dataCopyOutBaseGM,
                                             GlobalTensor<DataType1> y3BaseGM, GlobalTensor<DataType2> biasBaseGM);

    __aicore__ inline void TransposeBeforeBMMShardH(bool isLocal, bool isCTail, GlobalTensor<DataType1>
                                                    dataCopyInBaseGM, GlobalTensor<DataType1> y2BaseGM);
    __aicore__ inline void TransposeAfterBMMShardH(bool isLocal, bool isCTail, GlobalTensor<DataType1>
                                                   dataCopyOutBaseGM, GlobalTensor<DataType1> y3BaseGM,
                                                   GlobalTensor<DataType2> biasBaseGM);
    __aicore__ inline void TransposeBeforeBMMShardC(bool isLocal, bool isCTail, GlobalTensor<DataType1>
                                                    dataCopyInBaseGM, GlobalTensor<DataType1> y2BaseGM);
    __aicore__ inline void TransposeAfterBMMShardC(bool isLocal, bool isCTail, GlobalTensor<DataType1>
                                                   dataCopyOutBaseGM, GlobalTensor<DataType1> y3BaseGM,
                                                   GlobalTensor<DataType2> biasBaseGM);
    __aicore__ inline void TransposeAllGatherInput(bool isCTail, GlobalTensor<DataType1>
                                                   dataCopyInBaseGM, GlobalTensor<DataType1> dataCopyOutBaseGM);

    AlltoAllAllGatherBatchMatmulActType actType;

    // shardType 0/1: 输入 E,C,H
    uint64_t E;
    uint64_t C; //shardType 0: C   shardType 1: C/tp
    uint64_t H; //shardType 0: H/tp   shardType 1: H
    uint64_t M; // M/tp
    uint64_t MAlign;
    uint64_t expertOverEp;

    uint32_t epRankSize;
    uint32_t tpRankSize;
    uint32_t epRankId;
    uint32_t tpRankId;

    uint64_t dataLen1;
    uint64_t dataLen2;
    uint64_t shareTmpSize;
    uint64_t ubCapacityForTrans;
    uint64_t ubCapacityForAct;
    uint64_t ubSize;
    uint32_t vecCoreNum;
    uint32_t shardFactor;

    uint64_t allgatherRecvBufOffset;
    uint64_t allgatherRecvBufOffsetTile;
    uint64_t allgatherRecvBufOffsetTail;
    uint64_t allgatherSendBufOffsetTile;
    uint64_t allgatherSendBufOffsetTail;
    uint32_t handleIdx;
    uint32_t handleOffset;
    bool isbias;

    GlobalTensor<DataType1> xGM;
    GlobalTensor<DataType1> weightGM;
    GlobalTensor<DataType2> biasGM;
    GlobalTensor<DataType1> y1GM;
    GlobalTensor<DataType1> y2GM;
    GlobalTensor<DataType1> y3GM;

    GlobalTensor<DataType1> allgatherLocalOutGM;
    GlobalTensor<DataType1> alltoallNonLocalOutGM;
    GlobalTensor<DataType1> allgatherNonLocalInGM;
    GlobalTensor<DataType1> allgatherNonLocalOutGM;
    GlobalTensor<DataType1> transposeOutGM; //每轮地址复用
    GlobalTensor<DataType1> bmmOutGM;
    Mc2MatmulTilingData *bmmLocalTiling;
    Mc2MatmulTilingData *bmmNonLocalTileTiling;
    Mc2MatmulTilingData *bmmNonLocalTailTiling;

    Hccl<HCCL_SERVER_TYPE_AICPU> hcclAlltoall;
    Hccl<HCCL_SERVER_TYPE_AICPU> hcclAllgather;
    HcclHandle allgatherHandleList[MAX_HCCL_HANDLE];
    HcclDataType hcclDataType;

    TileInfo localE; // 无尾块
    TileInfo localC;
    TileInfo nonLocalE; // 无尾块
    TileInfo nonLocalC;

    TPipe *tpipe;
    TBuf<> tBuf;
    LocalTensor<DataType1> transposeTmp;
    LocalTensor<DataType1> xTmp; // 类型转换之前的数据
    LocalTensor<float> xCastTmp; // 类型转换之后的数据
    LocalTensor<DataType2> biasTmp;
    LocalTensor<half> actOutTmp; // 类型转换之后的输出
    LocalTensor<float> actCastOutTmp; // 类型转换之后的输出
    LocalTensor<uint8_t> sharedTmp;
};

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM, GM_ADDR y2GM, GM_ADDR y3GM,
                            GM_ADDR workspaceGM, TPipe *pipe, AlltoAllAllGatherBatchMatMulTilingData* tilingData)
{
    this->tpipe = pipe;
    GetTilingData(tilingData);

    if constexpr(ShardType == 0) {
        shardFactor = 1;
    } else {
        shardFactor = tpRankSize;
    }

    InitTpipe();

    this->xGM.SetGlobalBuffer((__gm__ DataType1*)xGM);
    this->weightGM.SetGlobalBuffer((__gm__ DataType1*)weightGM);
    this->y1GM.SetGlobalBuffer((__gm__ DataType1*)y1GM);
    if constexpr(IsNeedBias) {
        this->biasGM.SetGlobalBuffer((__gm__ DataType2*)biasGM);
    }
    if constexpr(IsNeedY2) {
        this->y2GM.SetGlobalBuffer((__gm__ DataType1*)y2GM);
    }
    if constexpr(IsNeedY3) {
        this->y3GM.SetGlobalBuffer((__gm__ DataType1*)y3GM);
    }

    uint64_t maxLenNonLocalC = ((nonLocalC.tileLen > nonLocalC.tailLen) ? nonLocalC.tileLen : nonLocalC.tailLen);
    alltoallNonLocalOutGM.SetGlobalBuffer((__gm__ DataType1*)workspaceGM);
    allgatherLocalOutGM.SetGlobalBuffer((__gm__ DataType1*)alltoallNonLocalOutGM.GetPhyAddr() + E * C * H);
    allgatherNonLocalInGM.SetGlobalBuffer((__gm__ DataType1*)allgatherLocalOutGM.GetPhyAddr() + expertOverEp * C * H * tpRankSize);
    allgatherNonLocalOutGM.SetGlobalBuffer((__gm__ DataType1*)allgatherNonLocalInGM.GetPhyAddr() + expertOverEp * C * H * epRankSize);
    transposeOutGM.SetGlobalBuffer((__gm__ DataType1*)allgatherNonLocalOutGM.GetPhyAddr() + E * C * H * tpRankSize);
    bmmOutGM.SetGlobalBuffer((__gm__ DataType1*)transposeOutGM.GetPhyAddr() +
                             ((localE.tileLen * localC.tileLen * H * tpRankSize > (epRankSize - 1U) * nonLocalE.tileLen * maxLenNonLocalC * H * tpRankSize)?
                              localE.tileLen * localC.tileLen * H * tpRankSize : (epRankSize - 1U) * nonLocalE.tileLen * maxLenNonLocalC * H * tpRankSize));

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    auto contextGM1 = AscendC::GetHcclContext<1>();
    hcclAlltoall.Init(contextGM0);
    hcclAllgather.Init(contextGM1);

    if constexpr(AscendC::IsSameType<DataType1, bfloat16_t>::value) {
        hcclDataType = HCCL_DATA_TYPE_BFP16;
    } else {
        hcclDataType = HCCL_DATA_TYPE_FP16;
    }

    epRankId = hcclAlltoall.GetRankId();
    tpRankId = hcclAllgather.GetRankId();

    allgatherRecvBufOffset = localE.tileLen * C * H * tpRankSize * dataLen1;
    handleIdx = localE.tileCnt + localE.tailCnt;
    handleOffset = nonLocalC.tileCnt + nonLocalC.tailCnt;
}


template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::InitTpipe()
{
    tpipe->InitBuffer(tBuf, ubSize);
    transposeTmp = tBuf.Get<DataType1>();
    xTmp = tBuf.Get<DataType1>(ubCapacityForAct);
    uint32_t offset = ubCapacityForAct * dataLen1;
    if (IsNeedBias || (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE)) {
        if ((AscendC::IsSameType<DataType1, bfloat16_t>::value) || (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU)) {
            xCastTmp = tBuf.Get<float>()[offset / SIZE_OF_FLOAT32];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT32);
            actCastOutTmp = tBuf.Get<float>()[offset / SIZE_OF_FLOAT32];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT32);
        } else {
            actOutTmp = tBuf.Get<half>()[offset / SIZE_OF_FLOAT16];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT16);
        }

        if constexpr(IsNeedBias) {
            biasTmp = tBuf.Get<DataType2>()[offset / dataLen2];
            offset += (ubCapacityForAct * dataLen2);
        }

        if ((actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU) ||
            (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU)) {
            sharedTmp = tBuf.Get<uint8_t>()[offset];
        }
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::GetTilingData(AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    E = tilingData->commonTiling.expert;
    C = tilingData->commonTiling.COverTp;
    H = tilingData->commonTiling.H;
    M = tilingData->commonTiling.MOverTp;
    expertOverEp = tilingData->commonTiling.EOverEp;
    MAlign = (M % 16 == 0) ? M : ((M / 16 + 1) * 16); // 32B 对齐，每个元素是2字节，跨行搬运场景使用

    tpRankSize = tilingData->commonTiling.tpGroupSize;
    epRankSize = tilingData->commonTiling.epGroupSize;

    shareTmpSize = tilingData->commonTiling.fastGeluBuffer;
    ubCapacityForTrans = tilingData->commonTiling.ubCapacityForTrans;
    ubCapacityForAct = tilingData->commonTiling.ubCapacityForAddActivate;
    ubSize = tilingData->commonTiling.totalUbSize;
    ubSize = (ubSize / 32) * 32;
    vecCoreNum = tilingData->commonTiling.aivCoreNum;

    localE = tilingData->commonTiling.localTileE;
    localC = tilingData->commonTiling.localTileC;
    nonLocalE = tilingData->commonTiling.domesticTileE;
    nonLocalC = tilingData->commonTiling.domesticTileC;

    bmmLocalTiling = &tilingData->localTiling;
    bmmNonLocalTileTiling = &tilingData->domesticTiling;
    bmmNonLocalTailTiling = &tilingData->domesticTailTiling;

    dataLen1 = sizeof(DataType1);
    dataLen2 = sizeof(DataType2);

    actType = (enum AlltoAllAllGatherBatchMatmulActType)tilingData->commonTiling.activateType;
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::Process()
{
    LocalCommunication();
    NonLocalCommunication();
    LocalCalculation();
    NonLocalCalculation();
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::LocalCommunication()
{
    //E_ep切分无尾块；Local块不切C
    GM_ADDR allgatherSendBuf = (__gm__ uint8_t*)xGM.GetPhyAddr() + expertOverEp * epRankId * C * H * dataLen1;
    GM_ADDR allgatherRecvBuf = (__gm__ uint8_t*)allgatherLocalOutGM.GetPhyAddr();
    uint64_t allGatherSendCount = localE.tileLen * C * H;
    uint64_t allgatherSendBufOffset = localE.tileLen * C * H * dataLen1;
    for (uint32_t e_idx = 0U; e_idx < localE.tileCnt; e_idx++) {
        if ASCEND_IS_AIV {
            allgatherHandleList[e_idx] = hcclAllgather.AllGather<true>(allgatherSendBuf, allgatherRecvBuf, allGatherSendCount, hcclDataType, 0U);
        }
        allgatherSendBuf += allgatherSendBufOffset;
        allgatherRecvBuf += allgatherRecvBufOffset;
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::LocalCalculation()
{
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, IsTransposeWeight>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType2, false>;
    GM_ADDR wGM = (__gm__ uint8_t*)weightGM.GetPhyAddr();
    uint64_t weightOffset = localE.tileLen * H * M * dataLen1;
    GlobalTensor<DataType2> biasBuf;
    uint64_t biasOffset;
    if constexpr(IsNeedBias) {
        biasBuf = biasGM;
        biasOffset = localE.tileLen * M;
    }

    GlobalTensor<DataType1> y1 = y1GM;
    GlobalTensor<DataType1> y2;
    GlobalTensor<DataType1> y3;
    GlobalTensor<DataType1> allgatherRecvTensor = allgatherLocalOutGM;
    uint64_t y1Offset = localE.tileLen * epRankSize * C * M * shardFactor;
    uint64_t y2Offset;
    allgatherRecvBufOffset = localE.tileLen * C * H * tpRankSize;
    if constexpr(IsNeedY2) {
        y2 = y2GM[epRankId * C * H * shardFactor];
        y2Offset = localE.tileLen * epRankSize * C * H * shardFactor;
    }
    if constexpr(IsNeedY3) {
        y3 = y3GM;
    }
    for (uint32_t e_idx = 0U; e_idx < localE.tileCnt; e_idx++) {
        if ASCEND_IS_AIV {
            hcclAllgather.Wait(allgatherHandleList[e_idx]);
        }
        SyncAll<false>();

        if ASCEND_IS_AIV {
            if ((localE.tileLen != 1) || IsNeedY2) { // Local块tile_e==1，bmm前的transpose跳过
                TransposeBeforeBMM(true, false, allgatherRecvTensor, y2);
            }
        }
        SyncAll<false>();

        GM_ADDR bmmInGm;
        if ((localE.tileLen != 1) || IsNeedY2) {
            bmmInGm = (__gm__ uint8_t*)transposeOutGM.GetPhyAddr();
        } else {
            bmmInGm = (__gm__ uint8_t*)allgatherRecvTensor.GetPhyAddr();
        }
        allgatherRecvTensor = allgatherRecvTensor[allgatherRecvBufOffset];

        // bmm
        tpipe->Reset();
        BatchMatMulCommonKernel<aType, bType, cType, biasType> bmmv3;
        bmmv3.Init(bmmInGm, wGM, (__gm__ uint8_t*)bmmOutGM.GetPhyAddr(), nullptr, nullptr, nullptr, &bmmLocalTiling->bmmTilingData, tpipe);
        bmmv3.Process();
        SyncAll<false>();
        tpipe->Reset();
        InitTpipe();
        wGM += weightOffset;

        if ASCEND_IS_AIV {
            TransposeAfterBMM(true, false, y1, y3, biasBuf);
        }

        y1 = y1[y1Offset];
        if constexpr(IsNeedY3) {
            y3 = y3[y1Offset];
        }
        if constexpr(IsNeedY2) {
            y2 = y2[y2Offset];
        }
        if constexpr(IsNeedBias) {
            biasBuf = biasBuf[biasOffset];
        }
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::NonLocalCommunication()
{
    GM_ADDR alltoallSendBuf = (__gm__ uint8_t*)xGM.GetPhyAddr();
    GM_ADDR alltoallRecvBuf = (__gm__ uint8_t*)alltoallNonLocalOutGM.GetPhyAddr();
    uint64_t tempDataPiece = H * dataLen1;
    uint64_t alltoallSendBufOffsetBase = C * tempDataPiece;
    uint64_t alltoallSendBufOffsetTile = nonLocalC.tileLen * tempDataPiece;
    uint64_t alltoallSendBufOffsetTail = nonLocalC.tailLen * tempDataPiece;
    uint64_t alltoallRecvBufOffsetTile = epRankSize * nonLocalC.tileLen * tempDataPiece;
    uint64_t alltoallRecvBufOffsetTail = epRankSize * nonLocalC.tailLen * tempDataPiece;
    uint64_t alltoallDataCountsTile[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallDataCountsTail[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallSdispls[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallRdisplsTile[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallRdisplsTail[MAX_EP_RANK_SIZE] = {0U};

    uint64_t dataCountsTileTemp = nonLocalC.tileLen * H;
    uint64_t dataCountsTailTemp = nonLocalC.tailLen * H;
    for (uint32_t i = 0U; i < epRankSize; i++)
    {
        alltoallDataCountsTile[i] = dataCountsTileTemp;
        alltoallDataCountsTail[i] = dataCountsTailTemp;
        alltoallSdispls[i] = i * expertOverEp * C * H;
        alltoallRdisplsTile[i] = i * dataCountsTileTemp;
        alltoallRdisplsTail[i] = i * dataCountsTailTemp;
    }

    GM_ADDR allgatherRecvBuf = (__gm__ uint8_t*)allgatherNonLocalOutGM.GetPhyAddr();
    GM_ADDR allgatherSendBuf = (__gm__ uint8_t*)alltoallNonLocalOutGM.GetPhyAddr();
    uint64_t allgatherSendBufOffsetCalcTemp = nonLocalE.tileLen * epRankSize * H * dataLen1;
    allgatherRecvBufOffsetTile = allgatherSendBufOffsetCalcTemp * nonLocalC.tileLen * tpRankSize;
    allgatherRecvBufOffsetTail = allgatherSendBufOffsetCalcTemp * nonLocalC.tailLen * tpRankSize;
    allgatherSendBufOffsetTile = allgatherSendBufOffsetCalcTemp * nonLocalC.tileLen;
    allgatherSendBufOffsetTail = allgatherSendBufOffsetCalcTemp * nonLocalC.tailLen;
    uint64_t allGatherSendCountTile = nonLocalE.tileLen * epRankSize * nonLocalC.tileLen * H;
    uint64_t allGatherSendCountTail = nonLocalE.tileLen * epRankSize * nonLocalC.tailLen * H;
    if (epRankSize == 2U) {
        allgatherSendBuf = (__gm__ uint8_t*)allgatherNonLocalInGM.GetPhyAddr();
        allGatherSendCountTile = allGatherSendCountTile / epRankSize;
        allGatherSendCountTail = allGatherSendCountTail / epRankSize;
    }

    HcclHandle alltoallHandle;
    HcclHandle alltoallHandleList[MAX_HCCL_HANDLE];
    uint64_t alltoallHandleIndex = 0UL;
    for (uint32_t e_idx = 0U; e_idx < nonLocalE.tileCnt; e_idx++) {
        for (uint32_t c_idx = 0U; c_idx < nonLocalC.tileCnt; c_idx++) {
            alltoallSendBuf = (__gm__ uint8_t*)xGM.GetPhyAddr() + e_idx * nonLocalE.tileLen * alltoallSendBufOffsetBase + c_idx * alltoallSendBufOffsetTile;
            if ASCEND_IS_AIV {
                uint32_t allgatherHandleId = 0;
                // 第2轮 -> n E轮次首个alltoall依赖上一次handleidx的尾allgather
                // 切C场景，第 2-> tilecnt 轮以来切C轮次的allgather结果
                if (epRankSize != 2U) {
                    allgatherHandleId = allgatherHandleList[handleIdx -1];
                    hcclAlltoall.InterHcclGroupSync(1, allgatherHandleId);
                }
                for (uint32_t k = 0U; k < nonLocalE.tileLen; k++) {
                    alltoallHandle = hcclAlltoall.AlltoAllV<true>(alltoallSendBuf, alltoallDataCountsTile, alltoallSdispls, hcclDataType,
                                                                  alltoallRecvBuf, alltoallDataCountsTile, alltoallRdisplsTile,  hcclDataType);
                    alltoallSendBuf += alltoallSendBufOffsetBase;
                    alltoallRecvBuf += alltoallRecvBufOffsetTile;
                    alltoallHandleList[alltoallHandleIndex++] = alltoallHandle;
                }
                if (epRankSize == 2U) {
                    allgatherHandleList[handleIdx++] = hcclAllgather.AllGather<false>(allgatherSendBuf, allgatherRecvBuf, allGatherSendCountTile, hcclDataType, 0U);
                } else {
                    hcclAllgather.InterHcclGroupSync(HCCL_GROUP_ID_0, alltoallHandle);
                    allgatherHandleList[handleIdx++] = hcclAllgather.AllGather<true>(allgatherSendBuf, allgatherRecvBuf, allGatherSendCountTile, hcclDataType, 0U);
                }
            }
            allgatherSendBuf += allgatherSendBufOffsetTile;
            allgatherRecvBuf += allgatherRecvBufOffsetTile;
        }

        if(nonLocalC.tailCnt == 1U) { //处理C的尾块
            if ASCEND_IS_AIV {
                alltoallSendBuf = (__gm__ uint8_t*)xGM.GetPhyAddr() + e_idx * nonLocalE.tileLen * alltoallSendBufOffsetBase + nonLocalC.tileCnt * alltoallSendBufOffsetTile;
                if ASCEND_IS_AIV {
                    if (epRankSize != 2U) {
                        hcclAlltoall.InterHcclGroupSync(1, handleIdx - 1);
                    }
                    for (uint32_t k = 0U; k < nonLocalE.tileLen; k++) {
                        alltoallHandle = hcclAlltoall.AlltoAllV<true>(alltoallSendBuf, (void *)alltoallDataCountsTail, (void *)alltoallSdispls, hcclDataType, alltoallRecvBuf, (void *)alltoallDataCountsTail, (void *)alltoallRdisplsTail, hcclDataType, 1);
                        alltoallSendBuf += alltoallSendBufOffsetBase;
                        alltoallRecvBuf += alltoallRecvBufOffsetTail;
                        alltoallHandleList[alltoallHandleIndex++] = alltoallHandle;
                    }
                    if (epRankSize == 2U) {
                        allgatherHandleList[handleIdx++] = hcclAllgather.AllGather<false>(allgatherSendBuf, allgatherRecvBuf, allGatherSendCountTail, hcclDataType, 0U);
                    } else {
                        hcclAllgather.InterHcclGroupSync(HCCL_GROUP_ID_0, alltoallHandle);
                        allgatherHandleList[handleIdx++] = hcclAllgather.AllGather<true>(allgatherSendBuf, allgatherRecvBuf, allGatherSendCountTail, hcclDataType, 0U);
                    }
                }
                allgatherSendBuf += allgatherSendBufOffsetTail;
                allgatherRecvBuf += allgatherRecvBufOffsetTail;
            }
        }
    }

    if (epRankSize == 2U) {
        alltoallHandleIndex = 0UL;
        handleIdx = localE.tileCnt + localE.tailCnt;
        bool isCTail = false;
        allgatherSendBufOffsetTile = allgatherSendBufOffsetTile / dataLen1;
        allgatherSendBufOffsetTail = allgatherSendBufOffsetTail / dataLen1;
        GlobalTensor<DataType1> alltoallOut = alltoallNonLocalOutGM;
        GlobalTensor<DataType1> allgatherIn = allgatherNonLocalInGM;
        for (uint32_t e_idx = 0U; e_idx < nonLocalE.tileCnt; e_idx++) {
            for (uint32_t c_idx = 0U; c_idx < nonLocalC.tileCnt + nonLocalC.tailCnt; c_idx++) {
                if (c_idx == nonLocalC.tileCnt) {
                    isCTail = true;
                } else {
                    isCTail = false;
                }

                if ASCEND_IS_AIV {
                    for (uint32_t k = 0U; k < nonLocalE.tileLen; k++) {
                        hcclAlltoall.Wait(alltoallHandleList[alltoallHandleIndex++]);
                    }
                    SyncAll<true>();
                    TransposeAllGatherInput(isCTail, alltoallOut, allgatherIn);
                    SyncAll<true>();
                    hcclAllgather.Commit(allgatherHandleList[handleIdx++]);
                    if (c_idx == nonLocalC.tileCnt) {
                        alltoallOut = alltoallOut[allgatherSendBufOffsetTail];
                        allgatherIn = allgatherIn[allgatherSendBufOffsetTail];
                    } else {
                        alltoallOut = alltoallOut[allgatherSendBufOffsetTile];
                        allgatherIn = allgatherIn[allgatherSendBufOffsetTile];
                    }
                }
            }
        }
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::NonLocalCalculation()
{
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, IsTransposeWeight>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType2, false>;
    GM_ADDR wGM = (__gm__ uint8_t*)weightGM.GetPhyAddr();
    GlobalTensor<DataType2> biasBuf;
    uint64_t weightOffset = nonLocalE.tileLen * H * M * dataLen1;
    uint64_t biasOffset;
    if constexpr(IsNeedBias) {
        biasBuf = biasGM;
        biasOffset = nonLocalE.tileLen * M;
    }

    bool isCTail = false;
    handleIdx = localE.tileCnt + localE.tailCnt;

    GlobalTensor<DataType1> y1;
    GlobalTensor<DataType1> y2;
    GlobalTensor<DataType1> y3;
    GlobalTensor<DataType1> allgatherRecvTensor = allgatherNonLocalOutGM;
    uint64_t y1OffsetBase = nonLocalE.tileLen * epRankSize * C * M * shardFactor;
    uint64_t y1Offset = nonLocalC.tileLen * M;
    uint64_t y2OffsetBase, y2Offset;
    allgatherRecvBufOffsetTile = nonLocalE.tileLen * epRankSize * nonLocalC.tileLen * H * tpRankSize;
    allgatherRecvBufOffsetTail = nonLocalE.tileLen * epRankSize * nonLocalC.tailLen * H * tpRankSize;
    if constexpr(IsNeedY2) {
        y2OffsetBase = nonLocalE.tileLen * epRankSize * C * H * shardFactor;
        y2Offset = nonLocalC.tileLen * H;
    }
    for (uint32_t e_idx = 0U; e_idx < nonLocalE.tileCnt; e_idx++) {
        y1 = y1GM[e_idx * y1OffsetBase];
        if constexpr(IsNeedY3) {
            y3 = y3GM[e_idx * y1OffsetBase];
        }
        if constexpr(IsNeedY2) {
            y2 = y2GM[e_idx * y2OffsetBase];
        }
        for (uint32_t c_idx = 0U; c_idx < (nonLocalC.tileCnt + nonLocalC.tailCnt); c_idx++) {
            if (c_idx == nonLocalC.tileCnt) {
                isCTail = true;
            }

            if ASCEND_IS_AIV {
                hcclAllgather.Wait(allgatherHandleList[handleIdx + c_idx]);
            }
            SyncAll<false>();

            // transpose
            TransposeBeforeBMM(false, isCTail, allgatherRecvTensor, y2);
            SyncAll<false>();

            // bmm
            tpipe->Reset();
            BatchMatMulCommonKernel<aType, bType, cType, biasType> bmmv3;
            if (isCTail) {
                allgatherRecvTensor = allgatherRecvTensor[allgatherRecvBufOffsetTail];
                bmmv3.Init((__gm__ uint8_t*)transposeOutGM.GetPhyAddr(), wGM, (__gm__ uint8_t*)bmmOutGM.GetPhyAddr(), nullptr, nullptr, nullptr, &bmmNonLocalTailTiling->bmmTilingData, tpipe);
            } else {
                allgatherRecvTensor = allgatherRecvTensor[allgatherRecvBufOffsetTile];
                bmmv3.Init((__gm__ uint8_t*)transposeOutGM.GetPhyAddr(), wGM, (__gm__ uint8_t*)bmmOutGM.GetPhyAddr(), nullptr, nullptr, nullptr, &bmmNonLocalTileTiling->bmmTilingData, tpipe);
            }
            bmmv3.Process();
            SyncAll<false>();
            tpipe->Reset();
            InitTpipe();

            // act + transpose
            if ASCEND_IS_AIV {
                TransposeAfterBMM(false, isCTail, y1, y3, biasBuf);
            }
            y1 = y1[y1Offset];
            if constexpr(IsNeedY2) {
                y2 = y2[y2Offset];
            }
            if constexpr(IsNeedY3) {
                y3 = y3[y1Offset];
            }
        }
        handleIdx += handleOffset;
        wGM += weightOffset;
        if constexpr(IsNeedBias) {
            biasBuf = biasBuf[biasOffset];
        }
    }
    if ASCEND_IS_AIV {
        hcclAlltoall.Finalize();
        hcclAllgather.Finalize();
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::TransposeAllGatherInput(bool isCTail, GlobalTensor<DataType1> dataCopyInBaseGM, GlobalTensor<DataType1> dataCopyOutBaseGM)
{
    uint32_t factorIn, factorOut;
    uint32_t tileLenC, tileLenE;
    if (isCTail) {
        tileLenC = nonLocalC.tailLen;
    } else {
        tileLenC = nonLocalC.tileLen;
    }
    tileLenE = nonLocalE.tileLen;
    factorIn = epRankSize;
    factorOut = epRankSize - 1U;

    DataCopyExtParams dataCopyInParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<DataType1> dataCopyInPadParams = {false, 0U, 0U, 0U};

    uint64_t totalM = tileLenE * factorOut * tileLenC;
    uint64_t tileLen = totalM % vecCoreNum == 0 ? totalM / vecCoreNum : totalM / vecCoreNum + 1;
    uint32_t blockIdx = GetBlockIdx();
    uint64_t curRowOffset = tileLen * blockIdx;
    uint64_t curDataOffset = curRowOffset * H;
    uint64_t endDataOffset = tileLen * H * (blockIdx + 1U);
    endDataOffset = endDataOffset < totalM * H ? endDataOffset : totalM * H;
    uint64_t endRowOffset = endDataOffset / H;
    uint64_t curBlockMaxRow = ((curRowOffset / tileLenC) + 1) * tileLenC;
    curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
    uint64_t perMaxRow = ubCapacityForTrans / H;
    uint64_t dataCnt = 0UL;
    uint32_t eIn, eOut, k, blockInnerOffset;
    GlobalTensor<DataType1> dataCopyInGM, dataCopyOutGM;
    uint32_t i = 0U;

    while (curDataOffset < endDataOffset) {
        i += 1;
        k = curRowOffset / (factorOut * tileLenC);
        eOut = (curRowOffset % (factorOut * tileLenC)) / tileLenC;
        eIn = eOut >= epRankId? eOut + 1U : eOut;
        blockInnerOffset = curDataOffset % (tileLenC * H);
        if (perMaxRow == 0UL) { //H超大，分片切列，一片处理的最大数据量为ubCapacity
            if (curRowOffset * H - curDataOffset > ubCapacityForTrans) {
                dataCnt = ubCapacityForTrans;
            } else {
                dataCnt = curRowOffset * H - curDataOffset;
                curRowOffset += 1U;
            }
        } else { //H较小，分片切行
            if (curBlockMaxRow * H - curDataOffset > perMaxRow * H) {
                dataCnt = perMaxRow * H;
            } else {
                dataCnt = curBlockMaxRow * H - curDataOffset;
                curBlockMaxRow += tileLenC;
                curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
            }
            curRowOffset += dataCnt / H;
        }
        dataCopyInParams.blockLen = dataCnt * dataLen1;
        dataCopyOutParams.blockLen = dataCnt * dataLen1;
        dataCopyInGM = dataCopyInBaseGM[k * (factorIn * tileLenC * H) + eIn * (tileLenC * H) + blockInnerOffset];
        dataCopyOutGM = dataCopyOutBaseGM[k * (factorOut * tileLenC * H) + eOut * (tileLenC * H) + blockInnerOffset];
        if (i > 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
        DataCopyPad(transposeTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        DataCopyPad(dataCopyOutGM, transposeTmp, dataCopyOutParams);

        curDataOffset += dataCnt;
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::TransposeBeforeBMM(bool isLocal, bool isCTail, GlobalTensor<DataType1> dataCopyInBaseGM, GlobalTensor<DataType1> y2BaseGM)
{
    if constexpr(ShardType == 0) {
        TransposeBeforeBMMShardH(isLocal, isCTail, dataCopyInBaseGM, y2BaseGM);
    } else {
        TransposeBeforeBMMShardC(isLocal, isCTail, dataCopyInBaseGM, y2BaseGM);
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::TransposeAfterBMM(bool isLocal, bool isCTail, GlobalTensor<DataType1> dataCopyOutBaseGM, GlobalTensor<DataType1> y3BaseGM, GlobalTensor<DataType2> biasBaseGM)
{
    if constexpr(ShardType == 0) {
        TransposeAfterBMMShardH(isLocal, isCTail, dataCopyOutBaseGM, y3BaseGM, biasBaseGM);
    } else {
        TransposeAfterBMMShardC(isLocal, isCTail, dataCopyOutBaseGM, y3BaseGM, biasBaseGM);
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::TransposeBeforeBMMShardC(bool isLocal, bool isCTail, GlobalTensor<DataType1> dataCopyInBaseGM, GlobalTensor<DataType1> y2BaseGM)
{
    uint32_t factorIn, factorOut;
    uint32_t tileLenC, tileLenE;
    if (isLocal) {
        if (isCTail) {
            tileLenC = localC.tailLen;
        } else {
            tileLenC = localC.tileLen;
        }
        tileLenE = localE.tileLen;
        factorIn = 1U;
        factorOut = 1U;
    } else {
        if (isCTail) {
            tileLenC = nonLocalC.tailLen;
        } else {
            tileLenC = nonLocalC.tileLen;
        }
        tileLenE = nonLocalE.tileLen;
        factorIn = epRankSize;
        factorOut = epRankSize - 1U;
        if (epRankSize == 2U) {
            factorIn = epRankSize - 1U;
        }
    }

    DataCopyExtParams dataCopyInParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<DataType1> dataCopyInPadParams = {false, 0U, 0U, 0U};

    uint64_t totalM = tpRankSize * tileLenE * factorOut * tileLenC;
    uint64_t tileLen = totalM % vecCoreNum == 0 ? (totalM / vecCoreNum) : (totalM / vecCoreNum + 1);
    uint32_t blockIdx = GetBlockIdx();
    uint64_t curRowOffset = tileLen * blockIdx;
    uint64_t curDataOffset = curRowOffset * H;
    uint64_t endDataOffset = tileLen * H * (blockIdx + 1U);
    endDataOffset = endDataOffset < totalM * H ? endDataOffset : totalM * H;
    uint64_t endRowOffset = endDataOffset / H;
    uint64_t curBlockMaxRow = ((curRowOffset / tileLenC) + 1) * tileLenC;
    curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
    uint64_t perMaxRow = ubCapacityForTrans / H;
    uint64_t dataCnt = 0UL;
    uint32_t t, eIn, eOut, k, blockInnerOffset, eY2;
    GlobalTensor<DataType1> dataCopyInGM, dataCopyOutGM, y2OutGM;
    uint32_t i = 0U;

    while (curDataOffset < endDataOffset) {
        i += 1;
        t = curRowOffset / (tileLenE * factorOut * tileLenC);
        k = (curRowOffset % (tileLenE * factorOut * tileLenC)) / (factorOut * tileLenC);
        eOut = (curRowOffset % (factorOut * tileLenC)) / tileLenC;
        eIn = eOut >= epRankId? eOut + 1U : eOut;
        eY2 = eIn;
        blockInnerOffset = curDataOffset % (tileLenC * H);
        if (isLocal) {
            eOut = 0UL;
            eIn = 0UL;
            eY2 = 0UL;
        } else {
            if (epRankSize == 2U) {
                eIn = eOut;
            }
        }
        if (perMaxRow == 0UL) { //H超大，分片切列
            if (curRowOffset * H - curDataOffset > ubCapacityForTrans) {
                dataCnt = ubCapacityForTrans;
            } else {
                dataCnt = curRowOffset * H - curDataOffset;
                curRowOffset += 1U;
            }
        } else { //H较小，分片切行
            if (curBlockMaxRow * H - curDataOffset > perMaxRow * H) {
                dataCnt = perMaxRow * H;
            } else {
                dataCnt = curBlockMaxRow * H - curDataOffset;
                curBlockMaxRow += tileLenC;
                curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
            }
            curRowOffset += dataCnt / H;
        }
        dataCopyInParams.blockLen = dataCnt * dataLen1;
        dataCopyOutParams.blockLen = dataCnt * dataLen1;
        dataCopyInGM = dataCopyInBaseGM[t * (tileLenE * factorIn * tileLenC * H) + k * (factorIn * tileLenC * H) + eIn * (tileLenC * H) + blockInnerOffset];
        dataCopyOutGM = transposeOutGM[k * (factorOut * tpRankSize * tileLenC * H) + eOut * (tpRankSize * tileLenC * H) + t * (tileLenC * H) + blockInnerOffset];
        if (i > 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
        DataCopyPad(transposeTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        DataCopyPad(dataCopyOutGM, transposeTmp, dataCopyOutParams);

        if constexpr(IsNeedY2) {
            dataCopyOutParams.blockLen = dataCnt * dataLen1;
            y2OutGM = y2BaseGM[k * (epRankSize * tpRankSize * C * H) + eY2 * (tpRankSize * C * H) + t * (C * H) + blockInnerOffset];
            DataCopyPad(y2OutGM, transposeTmp, dataCopyOutParams);
        }

        curDataOffset += dataCnt;
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::TransposeAfterBMMShardC(bool isLocal, bool isCTail, GlobalTensor<DataType1> dataCopyOutBaseGM, GlobalTensor<DataType1> y3BaseGM, GlobalTensor<DataType2> biasBaseGM)
{
    uint32_t tileLenC, tileLenE;
    uint32_t factorIn, factorOut;
    if (isLocal) {
        tileLenE = localE.tileLen;
        if (isCTail) {
            tileLenC = localC.tailLen;
        } else {
            tileLenC = localC.tileLen;
        }
        factorIn = 1U;
        factorOut = 1U;
    } else {
        tileLenE = nonLocalE.tileLen;
        if (isCTail) {
            tileLenC = nonLocalC.tailLen;
        } else {
            tileLenC = nonLocalC.tileLen;
        }

        factorIn = epRankSize - 1U;
        factorOut = epRankSize;
    }

    DataCopyExtParams dataCopyInParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyInBiasParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<DataType1> dataCopyInPadParams = {false, 0U, 0U, 0U};
    DataCopyPadExtParams<DataType2> dataCopyInBiasPadParams = {false, 0U, 0U, 0U};

    uint64_t totalM = tpRankSize * tileLenE * factorIn * tileLenC;
    uint64_t tileLen = totalM % vecCoreNum == 0 ? totalM / vecCoreNum : totalM / vecCoreNum + 1;
    uint32_t blockIdx = GetBlockIdx();
    uint64_t curRowOffset = tileLen * blockIdx;
    uint64_t curDataOffset = curRowOffset * M;
    uint64_t endDataOffset = tileLen * M * (blockIdx + 1U);
    endDataOffset = endDataOffset < totalM * M ? endDataOffset : totalM * M;
    uint64_t endRowOffset = endDataOffset / M;
    uint64_t curBlockMaxRow = ((curRowOffset / tileLenC) + 1) * tileLenC;
    curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
    uint64_t perMaxRow = ubCapacityForAct / MAlign;
    uint64_t dataCnt = 0UL;
    uint64_t realDataCnt = 0UL;
    uint64_t biasCnt = 0UL;
    uint32_t t, eIn, eOut, k, c, blockInnerOffset;
    GlobalTensor<DataType1> dataCopyInGM, dataCopyOutGM, y3OutGM;
    uint32_t i = 0U;

    while (curDataOffset < endDataOffset) {
        i += 1;
        k = curRowOffset / (factorIn * tpRankSize * tileLenC);
        eIn = (curRowOffset % (factorIn * tpRankSize * tileLenC)) / (tpRankSize * tileLenC);
        t = (curRowOffset % (tpRankSize * tileLenC)) / tileLenC;
        c = curRowOffset % tileLenC;
        blockInnerOffset = curDataOffset % M;
        if (isLocal) {
            eOut = epRankId;
            eIn = 0UL;
        } else {
            eOut = eIn >= epRankId? eIn + 1U : eIn;
        }
        if (perMaxRow == 0UL) { //H超大，分片切列，一片处理的最大数据量为ubCapacity
            if ((curRowOffset + 1) * M - curDataOffset > ubCapacityForAct) {
                dataCnt = ubCapacityForAct;
            } else {
                dataCnt = (curRowOffset + 1) * M - curDataOffset;
                curRowOffset += 1U;
            }
            biasCnt = 1UL;
            dataCopyInParams.blockLen = dataCnt * dataLen1;
            dataCopyOutParams.blockLen = dataCnt * dataLen1;
            dataCopyInBiasParams.blockLen = dataCnt * dataLen2;
            realDataCnt = dataCnt;
        } else { // H较小，分片切行
            if (curBlockMaxRow - curRowOffset > perMaxRow) {
                dataCnt = perMaxRow * M;
            } else {
                dataCnt = curBlockMaxRow * M - curDataOffset;
                curBlockMaxRow += tileLenC;
                curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
            }
            biasCnt = dataCnt / M;
            curRowOffset += biasCnt;
            dataCopyInParams.blockLen = M * dataLen1;
            dataCopyInParams.blockCount = biasCnt;
            dataCopyOutParams.blockLen = M * dataLen1;
            dataCopyOutParams.blockCount = biasCnt;
            dataCopyInBiasParams.blockLen = M * dataLen2;
            realDataCnt = biasCnt * MAlign;
        }
        dataCopyInGM = bmmOutGM[k * (factorIn * tpRankSize * tileLenC * M) + eIn * (tpRankSize * tileLenC * M) + t * (tileLenC * M) + c * M + blockInnerOffset];
        dataCopyOutGM = dataCopyOutBaseGM[k * (epRankSize * tpRankSize * C * M) + eOut * (tpRankSize * C * M) + t * (C * M) + c * M + blockInnerOffset];
        if (i > 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
        DataCopyPad(xTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);
        if constexpr(IsNeedBias) {
            DataCopyPad(biasTmp, biasBaseGM[k * M + blockInnerOffset], dataCopyInBiasParams, dataCopyInBiasPadParams);
        }

        if ((!IsNeedBias) && (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            DataCopyPad(dataCopyOutGM, xTmp, dataCopyOutParams);
            curDataOffset += dataCnt;
            continue;
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        if constexpr(AscendC::IsSameType<DataType1, bfloat16_t>::value) {
            Cast(xCastTmp, xTmp, RoundMode::CAST_NONE, realDataCnt);
            PipeBarrier<PIPE_V>();

            if constexpr(IsNeedBias) {
                for (uint64_t index = 0UL; index < biasCnt; index++) {
                    Add(xCastTmp[index * MAlign], xCastTmp[index * MAlign], biasTmp, dataCopyInBiasParams.blockLen / dataLen2);
                    PipeBarrier<PIPE_V>();
                }
            }

            if constexpr(IsNeedY3) {
                Cast(xTmp, xCastTmp, RoundMode::CAST_ROUND, realDataCnt);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                y3OutGM = y3BaseGM[k * (epRankSize * tpRankSize * C * M) + eOut * (tpRankSize * C * M) + t * (C * M) + c * M + blockInnerOffset];
                DataCopyPad(y3OutGM, xTmp, dataCopyOutParams);
            }

            if (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) {
                switch (actType){
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU:
                        Gelu(actCastOutTmp, xCastTmp, sharedTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU:
                        Silu(actCastOutTmp, xCastTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_RELU:
                        Relu(actCastOutTmp, xCastTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU:
                        FasterGelu(actCastOutTmp, xCastTmp, sharedTmp, realDataCnt);
                        break;
                    default:
                        break;
                }
                PipeBarrier<PIPE_V>();
            }

            if (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) {
                Cast(xTmp, xCastTmp, RoundMode::CAST_ROUND, realDataCnt);
            } else {
                if constexpr(IsNeedY3) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                }
                Cast(xTmp, actCastOutTmp, RoundMode::CAST_ROUND, realDataCnt);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            DataCopyPad(dataCopyOutGM, xTmp, dataCopyOutParams);
        } else {
            if constexpr(IsNeedBias) {
                for (uint64_t index = 0UL; index < biasCnt; index++) {
                    Add(xTmp[index * MAlign], xTmp[index * MAlign], biasTmp, dataCopyInBiasParams.blockLen / dataLen2);
                    PipeBarrier<PIPE_V>();
                }
            }
            if constexpr(IsNeedY3) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                y3OutGM = y3BaseGM[k * (epRankSize * tpRankSize * C * M) + eOut * (tpRankSize * C * M) + t * (C * M) + c * M + blockInnerOffset];
                DataCopyPad(y3OutGM, xTmp, dataCopyOutParams);
            }
            if (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) {
                switch (actType){
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU:
                        Gelu(actOutTmp, xTmp, sharedTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU:
                        Cast(xCastTmp, xTmp, RoundMode::CAST_NONE, realDataCnt);
                        PipeBarrier<PIPE_V>();
                        Silu(actCastOutTmp, xCastTmp, realDataCnt);
                        PipeBarrier<PIPE_V>();
                        if constexpr(IsNeedY3) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                        }
                        Cast(xTmp, actCastOutTmp, RoundMode::CAST_ROUND, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_RELU:
                        Relu(actOutTmp, xTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU:
                        FasterGelu(actOutTmp, xTmp, sharedTmp, realDataCnt);
                        break;
                    default:
                        break;
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            if ((actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) ||
                (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU)) {
                DataCopyPad(dataCopyOutGM, xTmp, dataCopyOutParams);
            } else {
                DataCopyPad(dataCopyOutGM, actOutTmp, dataCopyOutParams);
            }
        }
        curDataOffset += dataCnt;
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::TransposeBeforeBMMShardH(bool isLocal, bool isCTail, GlobalTensor<DataType1> dataCopyInBaseGM, GlobalTensor<DataType1> y2BaseGM)
{
    return;
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMul<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::TransposeAfterBMMShardH(bool isLocal, bool isCTail, GlobalTensor<DataType1> dataCopyOutBaseGM, GlobalTensor<DataType1> y3BaseGM, GlobalTensor<DataType2> biasBaseGM)
{
    return;
}

#endif