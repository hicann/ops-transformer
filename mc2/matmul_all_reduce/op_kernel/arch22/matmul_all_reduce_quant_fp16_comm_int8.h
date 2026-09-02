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
 * \file matmul_all_reduce_quant_fp16_comm_int8.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_FP16_COMM_INT8_H
#define MATMUL_ALL_REDUCE_QUANT_FP16_COMM_INT8_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#endif
#include "../matmul_all_reduce_common.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.h"
#include "matmul_all_reduce_add_x3.h"
#include "matmul_all_reduce_quant_perchannel.h"
#include "matmul_all_reduce_quant_reduce_sum.h"
#include "matmul_all_reduce_dequant_perchannel.h"

constexpr uint32_t MAX_HANDLE_NUM = 16U;
constexpr uint32_t NUM_TWO = 2U;

namespace MatmulAllReduceImpl {
using namespace AscendC;
using namespace DequantBmm;
using namespace MatmulAllReduceQuantPerchannelImpl;
using namespace MatmulAllReduceDequantPerchannelImpl;
using namespace MatmulAllReduceReduceSumImpl;
template <typename aType, typename bType, typename biasType, typename cType, typename commType, bool aTrans,
          bool bTrans>
class MatmulAllReduceQuantFP16CommInt8 {
public:
    __aicore__ inline MatmulAllReduceQuantFP16CommInt8() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR dequantGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR cGM,
                                GM_ADDR workspaceGM, Mc2Tiling::QuantMatmulAllReduceTilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void InitScale(GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(uint32_t isAdd, uint32_t tileCnt, Mc2QuantBatchMatmulV3TilingData &quant_tiling,
                                        uint32_t padM, uint32_t prePadM, bool isTiletoTileFlag);
    __aicore__ inline void PrepareInit();
    __aicore__ inline uint32_t SendCountCheck(uint32_t prepareIndex);

private:
    Mc2Tiling::QuantMatmulAllReduceTilingData *tilingData_;
    TPipe *tPipe_;
    GM_ADDR cGM_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR addGM_;
    GM_ADDR dequantGM_;
    GM_ADDR commQuantScale1GM_;
    GM_ADDR commQuantScale2GM_;
    GM_ADDR tempBuffGM_;
    GM_ADDR workspaceGMTemp_;
    GM_ADDR tempBuffWinOrGM_;
    GM_ADDR tempBuffDequantGM_;
    GM_ADDR allToAllSendbuffGM_;
    GM_ADDR allToAllRecbuffGM_;
    GM_ADDR workspaceGM_;
    bool notifyFlag_{false};
    GM_ADDR outGM_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    uint32_t tilePadM_ = 0;
    uint32_t tailPadM_ = 0;
    uint32_t tilePadDataCnt_ = 0;
    uint32_t tailPadDataCnt_ = 0;

    // 仅在0核上使用
    AscendC::HcclHandle allToAllHandleId_[MAX_HANDLE_NUM];
    AscendC::HcclHandle allGatherHandleId_[MAX_HANDLE_NUM];
    GM_ADDR allToAllSendGM_[MAX_HANDLE_NUM];
    GM_ADDR allGatherSendGM_[MAX_HANDLE_NUM];
    GM_ADDR allRecvOutGM_[MAX_HANDLE_NUM];
    int alltoAllCommitIdx_ = 0;
    int allToAllWaitIdx_ = 0;
    int allGatherCommitIdx_ = 0;
    // 所有核
    bool isSendTileFlag_ = false;
};

template <typename aType, typename bType, typename biasType, typename cType, typename commType, bool aTrans,
          bool bTrans>
__aicore__ inline void MatmulAllReduceQuantFP16CommInt8<aType, bType, biasType, cType, commType, aTrans, bTrans>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR dequantGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::QuantMatmulAllReduceTilingData *tilingData, TPipe *tPipe)
{
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    OOMInit(context);
    hccl_.Init(GetHcclContext<0>());
    tilingData_ = tilingData;
    outGM_ = cGM;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    dequantGM_ = dequantGM;
    biasGM_ = biasGM;
    addGM_ = addGM;
    cGM_ = cGM;
    workspaceGM_ = workspaceGM;
    tempBuffGM_ = workspaceGM_ + tilingData_->param.commWorkSpaceSize;
    tempBuffDequantGM_ = workspaceGM_ + tilingData_->param.commWorkSpaceSize;
    if ((tilingData->msg).useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN) {
        tempBuffWinOrGM_ = hccl_.GetWindowsInAddr(hccl_.GetRankId());
    } else {
        tempBuffWinOrGM_ = tempBuffGM_;
    }

    if ((block_idx == 0) && (g_coreType == AscendC::AIV)) {
        notifyFlag_ = true;
    }
}

template <typename aType, typename bType, typename biasType, typename cType, typename commType, bool aTrans,
          bool bTrans>
__aicore__ inline void MatmulAllReduceQuantFP16CommInt8<aType, bType, biasType, cType, commType, aTrans,
                                                        bTrans>::InitScale(GM_ADDR commQuantScale1GM,
                                                                           GM_ADDR commQuantScale2GM)
{
    commQuantScale1GM_ = commQuantScale1GM;
    commQuantScale2GM_ = commQuantScale2GM;
}

template <typename aType, typename bType, typename biasType, typename cType, typename commType, bool aTrans,
          bool bTrans>
__aicore__ inline uint32_t MatmulAllReduceQuantFP16CommInt8<aType, bType, biasType, cType, commType, aTrans,
                                                            bTrans>::SendCountCheck(uint32_t prepareIndex)
{
    uint32_t sendCount = tilePadDataCnt_ / tilingData_->param.rankDim;
    if (prepareIndex >= tilingData_->param.tileCnt) {
        sendCount = tailPadDataCnt_ / tilingData_->param.rankDim;
    }
    return sendCount;
}

template <typename aType, typename bType, typename biasType, typename cType, typename commType, bool aTrans,
          bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantFP16CommInt8<aType, bType, biasType, cType, commType, aTrans, bTrans>::PrepareInit()
{
    auto &&mc2Tiling = tilingData_->param;
    uint32_t rankNum = mc2Tiling.rankDim;
    uint32_t tileM = tilingData_->tilematmulTiling.matmulTiling.M; // 头块 pad 大小更新
    tilePadM_ = tileM;
    if (tileM % rankNum != 0) { // 按照 M 来 pad
        tilePadM_ += rankNum - (tileM % rankNum);
    }
    uint32_t tailM = tilingData_->tailmatmulTiling.matmulTiling.M;
    tailPadM_ = tailM;
    if (tailM % rankNum != 0) {
        tailPadM_ += rankNum - (tailM % rankNum);
    }

    tilePadDataCnt_ = tilePadM_ * tilingData_->tilematmulTiling.matmulTiling.N;
    tailPadDataCnt_ = tailPadM_ * tilingData_->tailmatmulTiling.matmulTiling.N;
    const int64_t tempBufOffsetTilePad = tilePadDataCnt_ * sizeof(int8_t); // 偏移计算
    const int64_t tempBufOffsetTailPad = tailPadDataCnt_ * sizeof(int8_t);
    const int64_t tempBufOffsetTilePadSingle = tempBufOffsetTilePad / rankNum;
    const int64_t tempBufOffsetTailPadSingle = tempBufOffsetTailPad / rankNum;

    for (uint32_t i = 0U; i < mc2Tiling.tileCnt; ++i) { // 头块偏移
        const int64_t indexOffsetTile = i * tempBufOffsetTilePad;
        const int64_t indexGatherOffsetTile = tempBufOffsetTilePadSingle * hccl_.GetRankId();
        allToAllSendGM_[i] = tempBuffWinOrGM_ + indexOffsetTile;
        allRecvOutGM_[i] = tempBuffGM_ + indexOffsetTile;
        allGatherSendGM_[i] = tempBuffGM_ + indexOffsetTile + indexGatherOffsetTile;
    }
    for (uint32_t i = 0U; i < mc2Tiling.tailCnt; ++i) { // 尾块偏移
        const int64_t indexOffsetTail = mc2Tiling.tileCnt * tempBufOffsetTilePad + i * tempBufOffsetTailPad;
        const int64_t indexGatherOffsetTail = tempBufOffsetTailPadSingle * hccl_.GetRankId();
        allToAllSendGM_[mc2Tiling.tileCnt + i] = tempBuffWinOrGM_ + indexOffsetTail;
        allRecvOutGM_[mc2Tiling.tileCnt + i] = tempBuffGM_ + indexOffsetTail;
        allGatherSendGM_[mc2Tiling.tileCnt + i] = tempBuffGM_ + indexOffsetTail + indexGatherOffsetTail;
    }

    if (GetBlockIdx() == 0) {
        int nowAll2AllIdx = 0;
        int nowAllGatherIdx = 0;
        int N = (mc2Tiling.tileCnt + mc2Tiling.tailCnt) / NUM_TWO;
        int ReN = (mc2Tiling.tileCnt + mc2Tiling.tailCnt) % NUM_TWO;
        for (uint32_t i = 0U; i < N; ++i) { // 按总核数下发
            allToAllHandleId_[nowAll2AllIdx] =
                hccl_.AlltoAll<false>(allToAllSendGM_[nowAll2AllIdx], allRecvOutGM_[nowAll2AllIdx],
                                      SendCountCheck(nowAll2AllIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAll2AllIdx++;
            allToAllHandleId_[nowAll2AllIdx] =
                hccl_.AlltoAll<false>(allToAllSendGM_[nowAll2AllIdx], allRecvOutGM_[nowAll2AllIdx],
                                      SendCountCheck(nowAll2AllIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAll2AllIdx++;

            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.AllGather<false>(allGatherSendGM_[nowAllGatherIdx], allRecvOutGM_[nowAllGatherIdx],
                                       SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.AllGather<false>(allGatherSendGM_[nowAllGatherIdx], allRecvOutGM_[nowAllGatherIdx],
                                       SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }

        if (ReN != 0) { // 余数下发
            allToAllHandleId_[nowAll2AllIdx] =
                hccl_.AlltoAll<false>(allToAllSendGM_[nowAll2AllIdx], allRecvOutGM_[nowAll2AllIdx],
                                      SendCountCheck(nowAll2AllIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAll2AllIdx++;
            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.AllGather<false>(allGatherSendGM_[nowAllGatherIdx], allRecvOutGM_[nowAllGatherIdx],
                                       SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }
    }
}

template <typename aType, typename bType, typename biasType, typename cType, typename commType, bool aTrans,
          bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantFP16CommInt8<aType, bType, biasType, cType, commType, aTrans, bTrans>::Process()
{
    auto &&cfg = tilingData_->param;
    if (g_coreType == AscendC::AIV) {
        PrepareInit();
    }
    // 调用InnerProcess,matmul计算以及通讯任务下发
    InnerProcess(cfg.isAdd, cfg.tileCnt, tilingData_->tilematmulTiling, tilePadM_, tilePadM_, false);
    if (cfg.tailM != 0U) {
        InnerProcess(cfg.isAdd, cfg.tailCnt, tilingData_->tailmatmulTiling, tailPadM_, tilePadM_, true);
    }

    // 最后一次的all2all任务没有wait， 以及最后一次的allgather任务没有下发
    if (g_coreType == AscendC::AIV) {
        if (notifyFlag_) { // 从第二块all2all任务commit开始，等待前一块all2all任务做完
            hccl_.Wait(allToAllHandleId_[allToAllWaitIdx_]);
        }
        SyncAll();
        uint32_t padM = tilePadM_;
        uint32_t lastN = tilingData_->tilematmulTiling.matmulTiling.N;
        if (cfg.tailM != 0U) { // 最后一块，没尾块取头块pad大小，有尾块拿尾块的大小
            padM = tailPadM_;
            lastN = tilingData_->tailmatmulTiling.matmulTiling.N;
        }
        MatmulAllReduceReduceSumInt8<cType>(tempBuffGM_, commQuantScale1GM_, commQuantScale2GM_, padM, lastN, tPipe_,
                                            hccl_);
        SyncAll();
        if (notifyFlag_) { // 下发allgather任务
            hccl_.Commit(allGatherHandleId_[allToAllWaitIdx_]);
            allToAllWaitIdx_++;
        }
    }

    // 等待所有的allgather任务
    if (g_coreType == AscendC::AIV) {
        // wait所有的allGather任务 + dequant
        const uint64_t outGmTileOffset = static_cast<uint64_t>(tilingData_->tilematmulTiling.matmulTiling.M) *
                                         static_cast<uint64_t>(tilingData_->tilematmulTiling.matmulTiling.N) *
                                         sizeof(cType);
        const uint64_t outGmTailOffset = static_cast<uint64_t>(tilingData_->tailmatmulTiling.matmulTiling.M) *
                                         static_cast<uint64_t>(tilingData_->tailmatmulTiling.matmulTiling.N) *
                                         sizeof(cType);
        for (uint32_t i = 0; i < cfg.tileCnt + cfg.tailCnt; i++) {
            if (block_idx == 0) {
                hccl_.Wait(allGatherHandleId_[i]);
            }
            SyncAll();
            if (i < cfg.tileCnt) { // dequant只传有效大小
                MatmulAllReduceDequantPerchannelCommInt8<cType>(tempBuffDequantGM_, commQuantScale2GM_, outGM_, tPipe_,
                                                                tilingData_->tilematmulTiling.matmulTiling.N,
                                                                tilingData_->tilematmulTiling.matmulTiling.M);
                tempBuffDequantGM_ += tilePadDataCnt_ * sizeof(int8_t); // 偏移pad后的大小
                outGM_ += outGmTileOffset;                              // 偏移原始块大小
                SyncAll();
            } else {
                MatmulAllReduceDequantPerchannelCommInt8<cType>(tempBuffDequantGM_, commQuantScale2GM_, outGM_, tPipe_,
                                                                tilingData_->tailmatmulTiling.matmulTiling.N,
                                                                tilingData_->tailmatmulTiling.matmulTiling.M);
                tempBuffDequantGM_ += tailPadDataCnt_ * sizeof(int8_t);
                outGM_ += outGmTailOffset;
                SyncAll();
            }
        }
        // 通信Finalize
        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }
}

template <typename aType, typename bType, typename biasType, typename cType, typename commType, bool aTrans,
          bool bTrans>
__aicore__ inline void
MatmulAllReduceQuantFP16CommInt8<aType, bType, biasType, cType, commType, aTrans, bTrans>::InnerProcess(
    uint32_t isAdd, uint32_t tileCnt, Mc2QuantBatchMatmulV3TilingData &quant_tiling, uint32_t padM, uint32_t prePadM,
    bool isTiletoTailFlag)
{
    const int64_t aOffset = static_cast<int64_t>(quant_tiling.matmulTiling.M) *
                            static_cast<int64_t>(quant_tiling.matmulTiling.Ka) * sizeof(aType);
    const int64_t cOffset = static_cast<int64_t>(quant_tiling.matmulTiling.M) *
                            static_cast<int64_t>(quant_tiling.matmulTiling.N) * sizeof(cType);
    int64_t tempOffset = CalcShapeOffset(sizeof(int8_t), padM, quant_tiling.matmulTiling.N);
    BmmDequant<aType, bType, FORMAT_X1, FORMAT_X2, biasType, uint64_t, cType, aTrans, bTrans> op;
    for (uint32_t i = 0; i < tileCnt; i++) {
        // mm + add + quant
        tPipe_->Reset();
        op.Init(aGM_, bGM_, biasGM_, dequantGM_, cGM_, workspaceGM_, &quant_tiling, tPipe_);
        op.Process();
        SyncAll<false>();
        if (isAdd != 0) {
            Matmul_All_Reduce_Add_X3<cType>(cGM_, addGM_, cOffset / sizeof(cType), tilingData_->param.addX3UbCnt,
                                            tPipe_);
            SyncAll<false>();
        }
        MatmulAllReduceQuantPerchannelCommInt8<cType>(cGM_, commQuantScale1GM_, tempBuffWinOrGM_, tPipe_,
                                                      quant_tiling.matmulTiling.N, quant_tiling.matmulTiling.M);
        SyncAll<false>();
        // commit
        if (notifyFlag_) {
            hccl_.Commit(allToAllHandleId_[alltoAllCommitIdx_]);
            alltoAllCommitIdx_++;
        }
        if (g_coreType == AscendC::AIV && isSendTileFlag_) { // 从第二块all2all任务commit开始，等待前一块all2all任务做完
            if (notifyFlag_) {
                hccl_.Wait(allToAllHandleId_[allToAllWaitIdx_]);
            }
            SyncAll();
            if (isTiletoTailFlag && i == 0) {
                MatmulAllReduceReduceSumInt8<cType>(tempBuffGM_, commQuantScale1GM_, commQuantScale2GM_, prePadM,
                                                    quant_tiling.matmulTiling.N, tPipe_, hccl_);
                tempBuffGM_ += prePadM * quant_tiling.matmulTiling.N;
            } else {
                MatmulAllReduceReduceSumInt8<cType>(tempBuffGM_, commQuantScale1GM_, commQuantScale2GM_, padM,
                                                    quant_tiling.matmulTiling.N, tPipe_, hccl_);
                tempBuffGM_ += padM * quant_tiling.matmulTiling.N;
            }
            SyncAll();
            if (notifyFlag_) { // 下发allgather任务
                hccl_.Commit(allGatherHandleId_[allToAllWaitIdx_]);
                allToAllWaitIdx_++;
            }
        }
        isSendTileFlag_ = true;
        aGM_ += aOffset;
        cGM_ += cOffset;
        addGM_ += cOffset;
        tempBuffWinOrGM_ += tempOffset; // 偏移padM*N大小
    }
}
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_FP16_COMM_INT8_H
