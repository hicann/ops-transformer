/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_all_all_gather_batch_mat_mul_shard_h.h
 * \brief
 */

#ifndef MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_SHARD_H_H
#define MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_SHARD_H_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "adv_api/activation/silu.h"
#include "adv_api/activation/gelu.h"
#include "lib/matmul_intf.h"
#include "../../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3.h"
#include "allto_all_all_gather_batch_mat_mul_tiling_struct.h"

using namespace AscendC;
using namespace matmul;

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
class AlltoAllAllGatherBatchMatMulShardH {
public:
    __aicore__ inline AlltoAllAllGatherBatchMatMulShardH(){};
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM, GM_ADDR y2GM, GM_ADDR y3GM,
                                GM_ADDR workspaceGM, TPipe *pipe, AlltoAllAllGatherBatchMatMulTilingData *tilingData,
                                __gm__ void *hcclInitTiling, __gm__ void *allGatherCcTiling,
                                __gm__ void *alltoAllCcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTpipe();
    __aicore__ inline void GetTilingData(AlltoAllAllGatherBatchMatMulTilingData *tilingData);
    __aicore__ inline void LocalCommunication();
    __aicore__ inline void NonLocalCommunication();
    __aicore__ inline void LocalCalculation();
    __aicore__ inline void NonLocalCalculation();
    __aicore__ inline void TransposeBeforeBMM(bool isLocal, bool isCTail, uint64_t tileLenE,
                                              GlobalTensor<DataType1> dataCopyInBaseGM,
                                              GlobalTensor<DataType1> y2BaseGM);
    __aicore__ inline void TransposeAfterBMM(bool isLocal, bool isCTail, uint64_t tileLenE,
                                             GlobalTensor<DataType1> dataCopyOutBaseGM,
                                             GlobalTensor<DataType1> y3BaseGM, GlobalTensor<DataType2> biasGM);

    AlltoAllAllGatherBatchMatmulActType actType;
    // shardType 0/1: 输入 E,C,H
    uint64_t E;
    uint64_t C; // shardType 0: C
    uint64_t H; // shardType 0: H/tp
    uint64_t M; // M/tp
    uint64_t MAlign;
    uint64_t expertOverEp;
    uint64_t totalH;

    uint32_t epRankSize;
    uint32_t tpRankSize;
    uint32_t epRankId;
    uint32_t tpRankId;

    uint64_t xDataLen;
    uint64_t biasDataLen;
    uint64_t shareTmpSize;
    uint64_t ubCapacityForTrans;
    uint64_t ubCapacityForAct;
    uint64_t ubSize;
    uint32_t vecCoreNum;

    uint64_t allgatherRecvBufOffset;
    uint64_t allgatherRecvBufOffsetTile;
    uint64_t allgatherRecvBufOffsetTail;
    uint64_t allgatherSendBufOffsetTile;
    uint64_t allgatherSendBufOffsetTail;
    uint64_t alltoallHandleIdx;
    uint64_t allgatherHandleIdx;
    uint64_t allgatherHandleWaitIdx;
    bool isBias;

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
    GlobalTensor<DataType1> transposeOutGM; // 每轮地址复用
    GlobalTensor<DataType1> bmmOutGM;
    BMM_Mc2MatmulTilingData *bmmLocalTiling;
    BMM_Mc2MatmulTilingData *bmmLocalTailTiling;
    BMM_Mc2MatmulTilingData *bmmNonLocalTileTiling;
    BMM_Mc2MatmulTilingData *bmmNonLocalTailTiling;
    BMM_Mc2MatmulTilingData *bmmNonLocalTailETiling;

    Hccl<HCCL_SERVER_TYPE_AICPU> hcclAlltoall;
    Hccl<HCCL_SERVER_TYPE_AICPU> hcclAllgather;
    HcclHandle allgatherHandleList[MAX_HCCL_HANDLE];
    HcclDataType hcclDataType;

    BMM_TileInfo localE; // 有尾块
    BMM_TileInfo localC;
    BMM_TileInfo nonLocalE; // 有尾块
    BMM_TileInfo nonLocalC;

    TPipe *tpipe;
    TBuf<> tBuf;
    LocalTensor<DataType1> transposeTmp;
    LocalTensor<DataType1> xTmp;
    LocalTensor<float> xCastTmp;
    LocalTensor<DataType2> biasTmp;
    LocalTensor<half> actOutTmp;
    LocalTensor<float> actCastOutTmp;
    LocalTensor<uint8_t> sharedTmp;
};

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void
AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2,
                                   IsNeedY3>::Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                   GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, TPipe *pipe,
                                                   AlltoAllAllGatherBatchMatMulTilingData *tilingData,
                                                   __gm__ void *hcclInitTiling, __gm__ void *allGatherCcTiling,
                                                   __gm__ void *alltoAllCcTiling)
{
    this->tpipe = pipe;
    GetTilingData(tilingData);
    InitTpipe();

    this->xGM.SetGlobalBuffer((__gm__ DataType1 *)xGM);
    this->weightGM.SetGlobalBuffer((__gm__ DataType1 *)weightGM);
    this->y1GM.SetGlobalBuffer((__gm__ DataType1 *)y1GM);
    if constexpr (IsNeedBias) {
        this->biasGM.SetGlobalBuffer((__gm__ DataType2 *)biasGM);
    }
    if constexpr (IsNeedY2) {
        this->y2GM.SetGlobalBuffer((__gm__ DataType1 *)y2GM);
    }
    if constexpr (IsNeedY3) {
        this->y3GM.SetGlobalBuffer((__gm__ DataType1 *)y3GM);
    }

    uint64_t maxLenNonLocalC = ((nonLocalC.tileLen > nonLocalC.tailLen) ? nonLocalC.tileLen : nonLocalC.tailLen);
    uint64_t maxLenNonLocalE = ((nonLocalE.tileLen > nonLocalE.tailLen) ? nonLocalE.tileLen : nonLocalE.tailLen);
    uint64_t maxLenLocalE = ((localE.tileLen > localE.tailLen) ? localE.tileLen : localE.tailLen);
    alltoallNonLocalOutGM.SetGlobalBuffer((__gm__ DataType1 *)workspaceGM);
    allgatherLocalOutGM.SetGlobalBuffer((__gm__ DataType1 *)alltoallNonLocalOutGM.GetPhyAddr() + E * C * H);
    allgatherNonLocalOutGM.SetGlobalBuffer((__gm__ DataType1 *)allgatherLocalOutGM.GetPhyAddr() +
                                           expertOverEp * C * H * tpRankSize);
    transposeOutGM.SetGlobalBuffer((__gm__ DataType1 *)allgatherNonLocalOutGM.GetPhyAddr() + E * C * H * tpRankSize);
    bmmOutGM.SetGlobalBuffer((__gm__ DataType1 *)transposeOutGM.GetPhyAddr() +
                             ((maxLenLocalE * localC.tileLen * H * tpRankSize >
                               (epRankSize - 1U) * maxLenNonLocalE * maxLenNonLocalC * H * tpRankSize) ?
                                  maxLenLocalE * localC.tileLen * H * tpRankSize :
                                  (epRankSize - 1U) * maxLenNonLocalE * maxLenNonLocalC * H * tpRankSize));

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    auto contextGM1 = AscendC::GetHcclContext<1>();
    hcclAlltoall.Init(contextGM0, hcclInitTiling);
    hcclAlltoall.SetCcTiling(alltoAllCcTiling);
    hcclAllgather.Init(contextGM1, hcclInitTiling);
    hcclAllgather.SetCcTiling(allGatherCcTiling);

    if constexpr (AscendC::IsSameType<DataType1, bfloat16_t>::value) {
        hcclDataType = HCCL_DATA_TYPE_BFP16;
    } else {
        hcclDataType = HCCL_DATA_TYPE_FP16;
    }

    epRankId = hcclAlltoall.GetRankId();
    tpRankId = hcclAllgather.GetRankId();

    alltoallHandleIdx = 0UL;
    allgatherHandleIdx = 0UL;
    allgatherHandleWaitIdx = 0UL;
    allgatherRecvBufOffset = localE.tileLen * C * H * tpRankSize * xDataLen;
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight,
                                                          IsNeedBias, IsNeedY2, IsNeedY3>::InitTpipe()
{
    tpipe->InitBuffer(tBuf, ubSize);
    transposeTmp = tBuf.Get<DataType1>();
    xTmp = tBuf.Get<DataType1>(ubCapacityForAct);
    uint32_t offset = ubCapacityForAct * xDataLen;
    if (IsNeedBias || (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE)) {
        if (AscendC::IsSameType<DataType1, bfloat16_t>::value ||
            (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU)) {
            xCastTmp = tBuf.Get<float>()[offset / SIZE_OF_FLOAT32];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT32);
            actCastOutTmp = tBuf.Get<float>()[offset / SIZE_OF_FLOAT32];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT32);
        } else {
            actOutTmp = tBuf.Get<half>()[offset / SIZE_OF_FLOAT16];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT16);
        }

        if constexpr (IsNeedBias) {
            biasTmp = tBuf.Get<DataType2>()[offset / biasDataLen];
            offset += (ubCapacityForAct * biasDataLen);
        }

        if ((actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU) ||
            (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU)) {
            sharedTmp = tBuf.Get<uint8_t>()[offset];
        }
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void
AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2,
                                   IsNeedY3>::GetTilingData(AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    E = tilingData->commonTiling.expert;
    C = tilingData->commonTiling.C;
    H = tilingData->commonTiling.HOverTp;
    M = tilingData->commonTiling.MOverTp;
    expertOverEp = tilingData->commonTiling.EOverEp;
    totalH = tilingData->commonTiling.H;
    MAlign = (M % 16 == 0) ? M : ((M / 16 + 1) * 16); // 32B 对齐

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
    bmmLocalTailTiling = &tilingData->localTailTiling;
    bmmNonLocalTileTiling = &tilingData->domesticTiling;
    bmmNonLocalTailTiling = &tilingData->domesticTailTiling;
    bmmNonLocalTailETiling = &tilingData->domesticTailETiling;

    xDataLen = sizeof(DataType1);
    biasDataLen = sizeof(DataType2);

    actType = (enum AlltoAllAllGatherBatchMatmulActType)tilingData->commonTiling.activateType;
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight,
                                                          IsNeedBias, IsNeedY2, IsNeedY3>::Process()
{
    LocalCommunication();
    NonLocalCommunication();
    LocalCalculation();
    NonLocalCalculation();
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight,
                                                          IsNeedBias, IsNeedY2, IsNeedY3>::LocalCommunication()
{
    // E_ep切分有尾块；尾块长度为1；Local块不切C
    GM_ADDR allgatherSendBuf = (__gm__ uint8_t *)xGM.GetPhyAddr() + expertOverEp * epRankId * C * H * xDataLen;
    GM_ADDR allgatherRecvBuf = (__gm__ uint8_t *)allgatherLocalOutGM.GetPhyAddr();
    for (uint32_t eIdx = 0U; eIdx < localE.tileCnt + localE.tailCnt; eIdx++) {
        uint64_t localELen = eIdx < localE.tileCnt ? localE.tileLen : localE.tailLen;
        uint64_t allgatherSendBufOffset = localELen * C * H * xDataLen;
        uint64_t allGatherSendCount = localELen * C * H;
        allgatherRecvBufOffset = allgatherSendBufOffset * tpRankSize;
        if ASCEND_IS_AIV {
            allgatherHandleList[allgatherHandleIdx++] =
                hcclAllgather.AllGather<true>(allgatherSendBuf, allgatherRecvBuf, allGatherSendCount, hcclDataType, 0U);
        }
        allgatherSendBuf += allgatherSendBufOffset;
        allgatherRecvBuf += allgatherRecvBufOffset;
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight,
                                                          IsNeedBias, IsNeedY2, IsNeedY3>::NonLocalCommunication()
{
    GM_ADDR alltoallSendBuf = (__gm__ uint8_t *)xGM.GetPhyAddr();
    GM_ADDR alltoallRecvBuf = (__gm__ uint8_t *)alltoallNonLocalOutGM.GetPhyAddr();
    uint64_t tempDataPiece = H * xDataLen;
    uint64_t alltoallSendBufOffsetBase = C * tempDataPiece;
    uint64_t alltoallSendBufOffsetTile = nonLocalE.tileLen * nonLocalC.tileLen * tempDataPiece;
    uint64_t alltoallSendBufOffsetTail = 0;
    uint64_t alltoallRecvBufOffsetTile = epRankSize * nonLocalE.tileLen * nonLocalC.tileLen * tempDataPiece;
    uint64_t alltoallRecvBufOffsetTail = 0;
    uint64_t alltoallDataCountsTile[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallDataCountsTail[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallSdispls[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallRdisplsTile[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallRdisplsTail[MAX_EP_RANK_SIZE] = {0U};

    GM_ADDR allgatherRecvBuf = (__gm__ uint8_t *)allgatherNonLocalOutGM.GetPhyAddr();
    GM_ADDR allgatherSendBuf = (__gm__ uint8_t *)alltoallNonLocalOutGM.GetPhyAddr();
    allgatherSendBufOffsetTile = nonLocalE.tileLen * nonLocalC.tileLen * epRankSize * tempDataPiece;
    allgatherRecvBufOffsetTile = allgatherSendBufOffsetTile * tpRankSize;
    uint64_t allgatherSendCountTile = nonLocalE.tileLen * nonLocalC.tileLen * epRankSize * H;
    uint64_t allgatherSendCountTail = 0;

    uint64_t dataCountsTileTemp = nonLocalE.tileLen * nonLocalC.tileLen * H;
    uint64_t dataCountsTailTemp = 0;

    if (nonLocalC.tileLen < C) { // E_len = 1，切C
        dataCountsTailTemp = nonLocalC.tailLen * H;
        alltoallSendBufOffsetTail = nonLocalC.tailLen * tempDataPiece;
        alltoallRecvBufOffsetTail = epRankSize * nonLocalC.tailLen * tempDataPiece;
        allgatherSendBufOffsetTail = nonLocalC.tailLen * epRankSize * tempDataPiece;
        allgatherSendCountTail = nonLocalC.tailLen * epRankSize * H;
    } else { // 切E，不切C
        dataCountsTailTemp = nonLocalE.tailLen * nonLocalC.tileLen * H;
        alltoallSendBufOffsetTail = nonLocalE.tailLen * nonLocalC.tileLen * tempDataPiece;
        alltoallRecvBufOffsetTail = epRankSize * nonLocalE.tailLen * nonLocalC.tileLen * tempDataPiece;
        allgatherSendBufOffsetTail = nonLocalE.tailLen * nonLocalC.tileLen * epRankSize * tempDataPiece;
        allgatherSendCountTail = nonLocalE.tailLen * nonLocalC.tileLen * epRankSize * H;
    }
    allgatherRecvBufOffsetTail = allgatherSendBufOffsetTail * tpRankSize;

    for (uint32_t i = 0U; i < epRankSize; i++) {
        alltoallDataCountsTile[i] = dataCountsTileTemp; // 每次alltoall的数据量：头块
        alltoallDataCountsTail[i] = dataCountsTailTemp; // 每次alltoall的数据量：尾块
        alltoallSdispls[i] = i * expertOverEp * C * H; // 每次alltoallV发送ep块数据（来自不同的E/ep块），数据间的偏移
        alltoallRdisplsTile[i] = i * dataCountsTileTemp; // 接收侧的数据偏移
        alltoallRdisplsTail[i] = i * dataCountsTailTemp;
    }

    HcclHandle alltoallHandle;
    HcclHandle alltoallHandleList[MAX_HCCL_HANDLE];
    for (uint32_t eIdx = 0U; eIdx < nonLocalE.tileCnt; eIdx++) {
        for (uint32_t cIdx = 0U; cIdx < nonLocalC.tileCnt; cIdx++) {
            if ASCEND_IS_AIV {
                hcclAlltoall.InterHcclGroupSync(
                    1,
                    allgatherHandleList[allgatherHandleIdx - 1]); // 等待上一块的allgather完成
                alltoallHandle = hcclAlltoall.AlltoAllV<true>(alltoallSendBuf, alltoallDataCountsTile, alltoallSdispls,
                                                              hcclDataType, alltoallRecvBuf, alltoallDataCountsTile,
                                                              alltoallRdisplsTile, hcclDataType);
                alltoallSendBuf += alltoallSendBufOffsetTile;
                alltoallRecvBuf += alltoallRecvBufOffsetTile;
                alltoallHandleList[alltoallHandleIdx++] = alltoallHandle;
                hcclAllgather.InterHcclGroupSync(HCCL_GROUP_ID_0, alltoallHandle); // 等待本块的alltoall完成
                allgatherHandleList[allgatherHandleIdx++] = hcclAllgather.AllGather<true>(
                    allgatherSendBuf, allgatherRecvBuf, allgatherSendCountTile, hcclDataType, 0U);
                allgatherSendBuf += allgatherSendBufOffsetTile;
                allgatherRecvBuf += allgatherRecvBufOffsetTile;
            }
        }
        if (nonLocalC.tailCnt != 0U) { // 处理C的尾块，不切C的情况走不进来
            for (uint32_t cIdx = 0U; cIdx < nonLocalC.tailCnt; cIdx++) {
                if ASCEND_IS_AIV {
                    hcclAlltoall.InterHcclGroupSync(
                        1,
                        allgatherHandleList[allgatherHandleIdx - 1]); // 等待上一块的allgather完成
                    alltoallHandle = hcclAlltoall.AlltoAllV<true>(
                        alltoallSendBuf, alltoallDataCountsTail, alltoallSdispls, hcclDataType, alltoallRecvBuf,
                        alltoallDataCountsTail, alltoallRdisplsTail, hcclDataType);
                    alltoallSendBuf += alltoallSendBufOffsetTail;
                    alltoallRecvBuf += alltoallRecvBufOffsetTail;
                    alltoallHandleList[alltoallHandleIdx++] = alltoallHandle;
                    hcclAllgather.InterHcclGroupSync(HCCL_GROUP_ID_0, alltoallHandle); // 等待本块的alltoall完成
                    allgatherHandleList[allgatherHandleIdx++] = hcclAllgather.AllGather<true>(
                        allgatherSendBuf, allgatherRecvBuf, allgatherSendCountTail, hcclDataType, 0U);
                    allgatherSendBuf += allgatherSendBufOffsetTail;
                    allgatherRecvBuf += allgatherRecvBufOffsetTail;
                }
            }
        }
    }
    if (nonLocalE.tailCnt != 0U) { // 处理E的尾块，切C的情况走不进来
        for (uint32_t eIdx = 0U; eIdx < nonLocalE.tailCnt; eIdx++) {
            if ASCEND_IS_AIV {
                hcclAlltoall.InterHcclGroupSync(
                    1,
                    allgatherHandleList[allgatherHandleIdx - 1]); // 等待上一块的allgather完成
                alltoallHandle = hcclAlltoall.AlltoAllV<true>(alltoallSendBuf, alltoallDataCountsTail, alltoallSdispls,
                                                              hcclDataType, alltoallRecvBuf, alltoallDataCountsTail,
                                                              alltoallRdisplsTail, hcclDataType);
                alltoallSendBuf += alltoallSendBufOffsetTail;
                alltoallRecvBuf += alltoallRecvBufOffsetTail;
                alltoallHandleList[alltoallHandleIdx++] = alltoallHandle;
                hcclAllgather.InterHcclGroupSync(HCCL_GROUP_ID_0, alltoallHandle); // 等待本块的alltoall完成
                allgatherHandleList[allgatherHandleIdx++] = hcclAllgather.AllGather<true>(
                    allgatherSendBuf, allgatherRecvBuf, allgatherSendCountTail, hcclDataType, 0U);
                allgatherSendBuf += allgatherSendBufOffsetTail;
                allgatherRecvBuf += allgatherRecvBufOffsetTail;
            }
        }
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight,
                                                          IsNeedBias, IsNeedY2, IsNeedY3>::LocalCalculation()
{
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, IsTransposeWeight>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType2, false>;
    GM_ADDR wGM = (__gm__ uint8_t *)weightGM.GetPhyAddr();
    uint64_t biasOffset = 0UL;
    uint64_t y2Offset = 0UL;
    GlobalTensor<DataType1> y1 = y1GM;
    GlobalTensor<DataType1> y2;
    GlobalTensor<DataType1> y3;
    GlobalTensor<DataType1> allgatherRecvTensor = allgatherLocalOutGM;
    GlobalTensor<DataType2> biasBuf;
    if constexpr (IsNeedY3) {
        y3 = y3GM;
    }
    if constexpr (IsNeedY2) {
        y2 = y2GM[epRankId * C * totalH]; // y2起始地址
    }
    if constexpr (IsNeedBias) {
        biasBuf = biasGM;
    }

    for (uint32_t eIdx = 0U; eIdx < localE.tileCnt + localE.tailCnt; eIdx++) {
        uint64_t localELen = eIdx < localE.tileCnt ? localE.tileLen : localE.tailLen;
        uint64_t weightOffset = localELen * totalH * M * xDataLen;
        if constexpr (IsNeedBias) {
            biasOffset = localELen * M;
        }
        uint64_t y1Offset = localELen * epRankSize * C * M;
        allgatherRecvBufOffset = tpRankSize * localELen * C * H;
        if constexpr (IsNeedY2) {
            y2Offset = localELen * epRankSize * C * totalH; // 下一个E循环y2的起始位置
        }

        if ASCEND_IS_AIV {
            hcclAllgather.Wait(allgatherHandleList[allgatherHandleWaitIdx++]);
        }
        SyncAll<false>();

        if ASCEND_IS_AIV {
            TransposeBeforeBMM(true, false, localELen, allgatherRecvTensor, y2);
        }
        SyncAll<false>();

        GM_ADDR bmmInGm;
        bmmInGm = (__gm__ uint8_t *)transposeOutGM.GetPhyAddr();
        allgatherRecvTensor = allgatherRecvTensor[allgatherRecvBufOffset];

        // bmm
        tpipe->Reset();
        Mc2BatchMatMulCommonKernel<aType, bType, cType, biasType> bmmv3;
        if (eIdx < localE.tileCnt) {
            bmmv3.Init(bmmInGm, wGM, (__gm__ uint8_t *)bmmOutGM.GetPhyAddr(), nullptr, nullptr, nullptr,
                       &bmmLocalTiling->bmmTilingData, tpipe);
        } else {
            bmmv3.Init(bmmInGm, wGM, (__gm__ uint8_t *)bmmOutGM.GetPhyAddr(), nullptr, nullptr, nullptr,
                       &bmmLocalTailTiling->bmmTilingData, tpipe);
        }
        bmmv3.Process();
        SyncAll<false>();
        tpipe->Reset();
        InitTpipe();
        wGM += weightOffset;

        if ASCEND_IS_AIV {
            TransposeAfterBMM(true, false, localELen, y1, y3, biasBuf);
        }

        y1 = y1[y1Offset];
        if constexpr (IsNeedY3) {
            y3 = y3[y1Offset];
        }
        if constexpr (IsNeedY2) {
            y2 = y2[y2Offset];
        }
        if constexpr (IsNeedBias) {
            biasBuf = biasBuf[biasOffset];
        }
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight,
                                                          IsNeedBias, IsNeedY2, IsNeedY3>::NonLocalCalculation()
{
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, IsTransposeWeight>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType2, false>;
    GM_ADDR wGM = (__gm__ uint8_t *)weightGM.GetPhyAddr();
    GlobalTensor<DataType1> y1;
    GlobalTensor<DataType1> y2;
    GlobalTensor<DataType1> y3;
    GlobalTensor<DataType1> allgatherRecvTensor = allgatherNonLocalOutGM;
    GlobalTensor<DataType2> biasBuf;

    uint64_t biasOffset = 0UL;
    uint64_t y1OffsetBase = nonLocalE.tileLen * epRankSize * C * M;
    uint64_t y1offsetTailBase = nonLocalE.tailLen * epRankSize * C * M;
    uint64_t y1Offset;
    uint64_t weightOffset = 0UL;
    uint64_t y2OffsetBase, y2OffsetTailBase, y2Offset;
    allgatherRecvBufOffsetTile = nonLocalE.tileLen * epRankSize * nonLocalC.tileLen * H * tpRankSize;
    if (nonLocalC.tileLen < C) {
        allgatherRecvBufOffsetTail = nonLocalE.tileLen * epRankSize * nonLocalC.tailLen * H * tpRankSize;
    } else {
        allgatherRecvBufOffsetTail = nonLocalE.tailLen * epRankSize * C * H * tpRankSize;
    }

    if constexpr (IsNeedBias) {
        biasBuf = biasGM;
    }
    if constexpr (IsNeedY2) {
        y2OffsetBase = nonLocalE.tileLen * epRankSize * C * totalH;
        y2OffsetTailBase = nonLocalE.tailLen * epRankSize * C * totalH;
    }
    for (uint32_t eIdx = 0U; eIdx < nonLocalE.tileCnt; eIdx++) {
        bool isCTail = false;
        uint64_t nonLocalELen = nonLocalE.tileLen;
        if constexpr (IsNeedBias) {
            biasOffset = nonLocalELen * M;
        }
        weightOffset = nonLocalELen * totalH * M * xDataLen;
        y1 = y1GM[eIdx * y1OffsetBase];
        if constexpr (IsNeedY3) {
            y3 = y3GM[eIdx * y1OffsetBase];
        }
        if constexpr (IsNeedY2) {
            y2 = y2GM[eIdx * y2OffsetBase];
        }

        for (uint32_t cIdx = 0U; cIdx < (nonLocalC.tileCnt + nonLocalC.tailCnt); cIdx++) {
            uint64_t nonLocalCLen = nonLocalC.tileLen;
            if (cIdx == nonLocalC.tileCnt) {
                isCTail = true;
            }
            if (isCTail) {
                nonLocalCLen = nonLocalC.tailLen;
            }
            y1Offset = nonLocalCLen * M;
            if constexpr (IsNeedY2) {
                y2Offset = nonLocalCLen * totalH;
            }
            if ASCEND_IS_AIV {
                hcclAllgather.Wait(allgatherHandleList[allgatherHandleWaitIdx++]);
            }
            SyncAll<false>();

            // transpose
            if ASCEND_IS_AIV {
                TransposeBeforeBMM(false, isCTail, nonLocalELen, allgatherRecvTensor, y2);
            }
            SyncAll<false>();

            // bmm
            tpipe->Reset();
            Mc2BatchMatMulCommonKernel<aType, bType, cType, biasType> bmmv3;
            if (isCTail) {
                allgatherRecvTensor = allgatherRecvTensor[allgatherRecvBufOffsetTail];
                bmmv3.Init((__gm__ uint8_t *)transposeOutGM.GetPhyAddr(), wGM, (__gm__ uint8_t *)bmmOutGM.GetPhyAddr(),
                           nullptr, nullptr, nullptr, &bmmNonLocalTailTiling->bmmTilingData, tpipe);
            } else {
                allgatherRecvTensor = allgatherRecvTensor[allgatherRecvBufOffsetTile];
                bmmv3.Init((__gm__ uint8_t *)transposeOutGM.GetPhyAddr(), wGM, (__gm__ uint8_t *)bmmOutGM.GetPhyAddr(),
                           nullptr, nullptr, nullptr, &bmmNonLocalTileTiling->bmmTilingData, tpipe);
            }

            bmmv3.Process();
            SyncAll<false>();
            tpipe->Reset();
            InitTpipe();

            // act + transpose
            if ASCEND_IS_AIV {
                TransposeAfterBMM(false, isCTail, nonLocalELen, y1, y3, biasBuf);
            }
            y1 = y1[y1Offset];
            if constexpr (IsNeedY2) {
                y2 = y2[y2Offset];
            }
            if constexpr (IsNeedY3) {
                y3 = y3[y1Offset];
            }
        }
        wGM += weightOffset;
        if constexpr (IsNeedBias) {
            biasBuf = biasBuf[biasOffset];
        }
    }

    if (nonLocalE.tailCnt != 0U) { // 处理E尾块，切C的情况走不进来
        uint64_t nonLocalELen = nonLocalE.tailLen;
        if constexpr (IsNeedBias) {
            biasOffset = nonLocalELen * M;
        }
        weightOffset = nonLocalELen * totalH * M * xDataLen;
        for (uint32_t eIdx = 0U; eIdx < nonLocalE.tailCnt; eIdx++) {
            y1 = y1GM[nonLocalE.tileCnt * y1OffsetBase + eIdx * y1offsetTailBase];
            if constexpr (IsNeedY3) {
                y3 = y3GM[nonLocalE.tileCnt * y1OffsetBase + eIdx * y1offsetTailBase];
            }
            if constexpr (IsNeedY2) {
                y2 = y2GM[nonLocalE.tileCnt * y2OffsetBase + eIdx * y2OffsetTailBase];
            }

            if ASCEND_IS_AIV {
                hcclAllgather.Wait(allgatherHandleList[allgatherHandleWaitIdx++]);
            }
            SyncAll<false>();

            // transpose
            if ASCEND_IS_AIV {
                TransposeBeforeBMM(false, false, nonLocalELen, allgatherRecvTensor, y2);
            }
            SyncAll<false>();

            // bmm
            tpipe->Reset();
            Mc2BatchMatMulCommonKernel<aType, bType, cType, biasType> bmmv3;
            allgatherRecvTensor = allgatherRecvTensor[allgatherRecvBufOffsetTail];
            bmmv3.Init((__gm__ uint8_t *)transposeOutGM.GetPhyAddr(), wGM, (__gm__ uint8_t *)bmmOutGM.GetPhyAddr(),
                       nullptr, nullptr, nullptr, &bmmNonLocalTailETiling->bmmTilingData, tpipe);
            bmmv3.Process();
            SyncAll<false>();
            tpipe->Reset();
            InitTpipe();

            // act + transpose
            if ASCEND_IS_AIV {
                TransposeAfterBMM(false, false, nonLocalELen, y1, y3, biasBuf);
            }
        }
        wGM += weightOffset;
        if constexpr (IsNeedBias) {
            biasBuf = biasBuf[biasOffset];
        }
    }

    if ASCEND_IS_AIV {
        hcclAlltoall.Finalize();
        hcclAllgather.Finalize();
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void
AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2,
                                   IsNeedY3>::TransposeBeforeBMM(bool isLocal, bool isCTail, uint64_t tileLenE,
                                                                 GlobalTensor<DataType1> dataCopyInBaseGM,
                                                                 GlobalTensor<DataType1> y2BaseGM)
{
    uint32_t factorIn, factorOut;
    uint32_t tileLenC;
    if (isLocal) {
        factorIn = 1U;
        factorOut = 1U;
        tileLenC = localC.tileLen;
    } else {
        if (isCTail) {
            tileLenC = nonLocalC.tailLen;
        } else {
            tileLenC = nonLocalC.tileLen;
        }
        factorIn = epRankSize;
        factorOut = epRankSize - 1U;
    }

    DataCopyExtParams dataCopyInParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<DataType1> dataCopyInPadParams = {false, 0U, 0U, 0U};

    uint64_t totalM = tpRankSize * tileLenE * tileLenC * factorOut; // 待处理的行数，无local块
    uint64_t tileLen = totalM % vecCoreNum == 0 ?
                           (totalM / vecCoreNum) :
                           (totalM / vecCoreNum + 1); // 每个核处理的行数（头块），除不尽时候向上取整
    uint32_t blockIdx = GetBlockIdx();
    uint64_t curRowOffset = tileLen * blockIdx;             // 每个核取到起始数据的行偏移
    uint64_t curDataOffset = curRowOffset * H;              // 每个核取到起始数据的个数偏移
    uint64_t endDataOffset = tileLen * H * (blockIdx + 1U); // 每个核取到数据的结尾
    endDataOffset = endDataOffset < totalM * H ? endDataOffset : totalM * H; // 分核有尾块时候防止越界
    uint64_t endRowOffset = endDataOffset / H;
    uint64_t curBlockMaxRow = ((curRowOffset / tileLenC) + 1) * tileLenC; // 单核可以计算的行数，不能跨C
    curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset; // 单次计算的尾块, 防止越界
    uint64_t HAlign = (H % 256 == 0) ? H : ((H / 256 + 1) * 256);                   // 512字节对齐
    uint64_t perMaxRow = ubCapacityForTrans / HAlign;
    uint64_t dataCnt = 0UL; // 单次计算的数据量
    uint32_t t, eIn, eOut, k, c, blockInnerOffset, eY2;
    GlobalTensor<DataType1> dataCopyInGM, dataCopyOutGM, y2OutGM;
    uint32_t i = 0U;

    while (curDataOffset < endDataOffset) {
        i += 1;
        t = curRowOffset / (factorOut * tileLenE * tileLenC);                              // tp
        eOut = (curRowOffset % (factorOut * tileLenE * tileLenC)) / (tileLenE * tileLenC); // ep
        k = (curRowOffset % (tileLenE * tileLenC)) / tileLenC;                             // tile_e
        c = curRowOffset % tileLenC;                                                       // tile_c
        eIn = eOut >= epRankId ? eOut + 1U : eOut;
        eY2 = eIn;
        blockInnerOffset = curDataOffset % H;
        if (isLocal) {
            eOut = 0UL;
            eIn = 0UL;
            eY2 = 0UL;
        }
        if (perMaxRow == 0UL) { // H超大，分片切列
            if ((curRowOffset + 1) * H - curDataOffset > ubCapacityForTrans) {
                dataCnt = ubCapacityForTrans;
            } else {
                dataCnt = (curRowOffset + 1) * H - curDataOffset;
                curRowOffset += 1U;
            }
            dataCopyInParams.blockLen = dataCnt * xDataLen;
            dataCopyOutParams.blockLen = dataCnt * xDataLen;
        } else { // H较小，分片切行
            if (curBlockMaxRow - curRowOffset > perMaxRow) {
                dataCnt = perMaxRow * H;
            } else {
                dataCnt = curBlockMaxRow * H - curDataOffset;
                curBlockMaxRow += tileLenC;
                curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
            }
            dataCopyInParams.blockLen = H * xDataLen;
            dataCopyInParams.blockCount = dataCnt / H;
            dataCopyOutParams.blockLen = H * xDataLen;
            dataCopyOutParams.blockCount = dataCnt / H;
            dataCopyOutParams.dstStride = (H * tpRankSize - H) * xDataLen;
            curRowOffset += dataCnt / H;
        }
        uint64_t copyInOffset = t * (factorIn * tileLenE * tileLenC * H) + eIn * (tileLenE * tileLenC * H) +
                                k * (tileLenC * H) + c * H + blockInnerOffset;
        uint64_t copyOutOffset =
            k * (factorOut * tileLenC * totalH) + eOut * (tileLenC * totalH) + c * totalH + t * H + blockInnerOffset;
        dataCopyInGM = dataCopyInBaseGM[copyInOffset];
        dataCopyOutGM = transposeOutGM[copyOutOffset];
        if (i > 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
        DataCopyPad(transposeTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        DataCopyPad(dataCopyOutGM, transposeTmp, dataCopyOutParams);

        if constexpr (IsNeedY2) {
            y2OutGM = y2BaseGM[k * epRankSize * C * totalH + eY2 * C * totalH + c * totalH + t * H + blockInnerOffset];
            DataCopyPad(y2OutGM, transposeTmp, dataCopyOutParams);
        }

        curDataOffset += dataCnt;
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias,
          bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void
AlltoAllAllGatherBatchMatMulShardH<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2,
                                   IsNeedY3>::TransposeAfterBMM(bool isLocal, bool isCTail, uint64_t tileLenE,
                                                                GlobalTensor<DataType1> dataCopyOutBaseGM,
                                                                GlobalTensor<DataType1> y3BaseGM,
                                                                GlobalTensor<DataType2> biasBaseGM)
{
    uint32_t tileLenC;
    uint32_t factorIn, factorOut;
    if (isLocal) {
        if (isCTail) {
            tileLenC = localC.tailLen;
        } else {
            tileLenC = localC.tileLen;
        }
        factorIn = 1U;
        factorOut = 1U;
    } else {
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

    uint64_t totalM = tileLenE * factorIn * tileLenC;
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
    uint32_t k, eIn, eOut, c, blockInnerOffset;
    GlobalTensor<DataType1> dataCopyInGM, dataCopyOutGM, y3OutGM;
    uint32_t i = 0U;

    while (curDataOffset < endDataOffset) {
        i += 1;
        k = curRowOffset / (factorIn * tileLenC);
        eIn = (curRowOffset % (factorIn * tileLenC)) / tileLenC;
        c = curRowOffset % tileLenC;
        blockInnerOffset = curDataOffset % M;
        if (isLocal) {
            eOut = epRankId;
            eIn = 0UL;
        } else {
            eOut = eIn >= epRankId ? eIn + 1U : eIn;
        }
        if (perMaxRow == 0UL) { // M超大，分片切列，一片处理的最大数据量为ubCapacity
            if ((curRowOffset + 1) * M - curDataOffset > ubCapacityForAct) {
                dataCnt = ubCapacityForAct;
            } else {
                dataCnt = (curRowOffset + 1) * M - curDataOffset;
                curRowOffset += 1U;
            }
            biasCnt = 1UL;
            dataCopyInParams.blockLen = dataCnt * xDataLen;
            dataCopyOutParams.blockLen = dataCnt * xDataLen;
            dataCopyInBiasParams.blockLen = dataCnt * biasDataLen;
            realDataCnt = dataCnt;
        } else { // M较小，分片切行
            if (curBlockMaxRow - curRowOffset > perMaxRow) {
                dataCnt = perMaxRow * M;
            } else {
                dataCnt = curBlockMaxRow * M - curDataOffset;
                curBlockMaxRow += tileLenC;
                curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
            }
            biasCnt = dataCnt / M;
            curRowOffset += biasCnt;
            dataCopyInParams.blockLen = M * xDataLen;
            dataCopyInParams.blockCount = biasCnt;
            dataCopyOutParams.blockLen = M * xDataLen;
            dataCopyOutParams.blockCount = biasCnt;
            dataCopyInBiasParams.blockLen = M * biasDataLen;
            realDataCnt = biasCnt * MAlign;
        }
        dataCopyInGM = bmmOutGM[k * (factorIn * tileLenC * M) + eIn * (tileLenC * M) + c * M + blockInnerOffset];
        dataCopyOutGM = dataCopyOutBaseGM[k * (epRankSize * C * M) + eOut * (C * M) + c * M + blockInnerOffset];
        if (i > 1) { // 从第二次循环开始，等待上一次循环xTmp向dataCopyOutGM/y3GM的拷贝
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
        DataCopyPad(xTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);

        if ((!IsNeedBias) &&
            (actType ==
             AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE)) { // 无bias无激活,
                                                                                                   // 直接拷出
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            DataCopyPad(dataCopyOutGM, xTmp, dataCopyOutParams);
            curDataOffset += dataCnt;
            continue;
        }

        if constexpr (IsNeedBias) {
            DataCopyPad(biasTmp, biasBaseGM[k * M + blockInnerOffset], dataCopyInBiasParams, dataCopyInBiasPadParams);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        if constexpr (AscendC::IsSameType<DataType1, bfloat16_t>::value) {
            Cast(xCastTmp, xTmp, RoundMode::CAST_NONE, realDataCnt);
            PipeBarrier<PIPE_V>();

            if constexpr (IsNeedBias) {
                for (uint64_t index = 0UL; index < biasCnt; index++) {
                    Add(xCastTmp[index * MAlign], xCastTmp[index * MAlign], biasTmp,
                        dataCopyInBiasParams.blockLen / biasDataLen);
                    PipeBarrier<PIPE_V>();
                }
            }

            if constexpr (IsNeedY3) {
                Cast(xTmp, xCastTmp, RoundMode::CAST_ROUND, realDataCnt);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                    GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                    GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                y3OutGM = y3BaseGM[k * (epRankSize * C * M) + eOut * (C * M) + c * M + blockInnerOffset];
                DataCopyPad(y3OutGM, xTmp, dataCopyOutParams);
            }

            if (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) {
                switch (actType) {
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
                if constexpr (IsNeedY3) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                }
                Cast(xTmp, actCastOutTmp, RoundMode::CAST_ROUND, realDataCnt);
            } else {
                Cast(xTmp, xCastTmp, RoundMode::CAST_ROUND, realDataCnt);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            DataCopyPad(dataCopyOutGM, xTmp, dataCopyOutParams);
        } else {
            if constexpr (IsNeedBias) {
                for (uint64_t index = 0UL; index < biasCnt; index++) {
                    Add(xTmp[index * MAlign], xTmp[index * MAlign], biasTmp,
                        dataCopyInBiasParams.blockLen / biasDataLen);
                    PipeBarrier<PIPE_V>();
                }
            }
            if constexpr (IsNeedY3) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                y3OutGM = y3BaseGM[k * (epRankSize * C * M) + eOut * (C * M) + c * M + blockInnerOffset];
                DataCopyPad(y3OutGM, xTmp, dataCopyOutParams);
            }

            if (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) {
                switch (actType) {
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU:
                        Gelu(actOutTmp, xTmp, sharedTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU:
                        Cast(xCastTmp, xTmp, RoundMode::CAST_NONE, realDataCnt);
                        PipeBarrier<PIPE_V>();
                        Silu(actCastOutTmp, xCastTmp, realDataCnt);
                        PipeBarrier<PIPE_V>();
                        if constexpr (IsNeedY3) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                                GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                                GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
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
#endif
