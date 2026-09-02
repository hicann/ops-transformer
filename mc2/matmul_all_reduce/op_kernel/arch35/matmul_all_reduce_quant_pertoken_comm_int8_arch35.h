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
 * \file matmul_all_reduce_quant_pertoken_comm_int8_arch35.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_ARCH35_H
#define MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"

#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h"
#include "matmul_all_reduce_add_x3_arch35.h"
#include "matmul_all_reduce_quant_perchannel_arch35.h"
#include "matmul_all_reduce_quant_reduce_sum_arch35.h"
#include "matmul_all_reduce_dequant_perchannel_arch35.h"

namespace MatmulAllReduceImpl {
constexpr uint32_t MAX_HANDLE_ID_NUM = 16;
constexpr uint32_t NUM_TWO_PERTOKEN = 2;

using namespace AscendC;
using namespace Mc2Kernel;
using namespace MatmulAllReduceQuantPerchannelImpl;
using namespace MatmulAllReduceDequantPerchannelImpl;
template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
class MatmulAllReduceQuantPertokenCommInt8 {
public:
    __aicore__ inline MatmulAllReduceQuantPertokenCommInt8() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM,
                                GM_ADDR pertokenScaleGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM,
                                GM_ADDR cGM, GM_ADDR workspaceGM,
                                Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(MmType &mmOp, uint32_t tileCnt,
                                        DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling, uint32_t isAdd,
                                        uint32_t needUbBuffer, uint32_t padM, bool isTailFlag);
    __aicore__ inline void PrepareInit();
    __aicore__ inline uint32_t SendCountCheck(uint32_t prepareIndex);

    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *tilingData_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR addGM_;
    GM_ADDR dequantScaleGM_;
    GM_ADDR pertokenScaleGM_;
    GM_ADDR commQuantScale1GM_;
    GM_ADDR commQuantScale2GM_;
    GM_ADDR cGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR outGM_;
    GM_ADDR all2allInGM_;
    GM_ADDR all2allOutGM_;
    GM_ADDR allGatherOutGM_;
    typename HcclTypeSelector<commMode>::type hccl_;
    bool notifyFlag_{false};

    // 仅在0核上使用
    AscendC::HcclHandle all2allHandleId_[MAX_HANDLE_ID_NUM] = {0};
    AscendC::HcclHandle allGatherHandleId_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allSendGM_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allGatherSendGM_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allRecvGM_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allGatherRecvGM_[MAX_HANDLE_ID_NUM] = {0};
    int all2allCommitIdx_ = 0;
    int all2allWaitIdx_ = 0;
    int allGatherCommitIdx_ = 0;
    int allGatherWaitIdx_ = 0;
    // 所有核
    bool isSendTileFlag_ = false;
    uint32_t tilePadM_ = 0;
    uint32_t tailPadM_ = 0;
    uint32_t tilePadDataCnt_ = 0;
    uint32_t tailPadDataCnt_ = 0;
};

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType, commMode>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM, GM_ADDR pertokenScaleGM,
    GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *tilingData, TPipe *tPipe)
{
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    OOMInit(context);
    hccl_.InitV2(GetHcclContext<0>(), tilingData);
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTiling));
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTilingComm));
    cGM_ = cGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    addGM_ = addGM;
    dequantScaleGM_ = dequantScaleGM;
    pertokenScaleGM_ = pertokenScaleGM;
    commQuantScale1GM_ = commQuantScale1GM;
    commQuantScale2GM_ = commQuantScale2GM;
    workspaceGM_ = workspaceGM;
    outGM_ = cGM;

    all2allInGM_ = workspaceGM_ + tilingData_->param.commWorkSpaceSize;     // all2all输入
    all2allOutGM_ = all2allInGM_ + tilingData_->param.commInt8WorkSpace;    // all2all输出及allGather输入
    allGatherOutGM_ = all2allOutGM_ + tilingData_->param.commInt8WorkSpace; // allGather输出

    if ((GetBlockIdx() == 0) && (g_coreType == AscendC::AIV)) {
        notifyFlag_ = true;
    }
}

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline uint32_t MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType,
                                                                commMode>::SendCountCheck(uint32_t prepareIndex)
{
    uint32_t sendCount = tilePadDataCnt_ / tilingData_->param.rankDim;
    if (prepareIndex >= tilingData_->param.tileCnt) {
        sendCount = tailPadDataCnt_ / tilingData_->param.rankDim;
    }
    return sendCount;
}

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void
MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType, commMode>::PrepareInit()
{
    auto &&mc2Tiling = tilingData_->param;
    uint32_t rankNum = mc2Tiling.rankDim;
    uint32_t tileM = tilingData_->tilematmulTiling.matmulTiling.M; // 头块 pad 大小更新
    tilePadM_ = tileM;
    if ((tileM % rankNum) != 0) { // 按照 M 来 pad
        tilePadM_ += rankNum - (tileM % rankNum);
    }
    uint32_t tailM = tilingData_->tailmatmulTiling.matmulTiling.M;
    tailPadM_ = tailM;
    if ((tailM % rankNum) != 0) {
        tailPadM_ += rankNum - (tailM % rankNum);
    }

    tilePadDataCnt_ = tilePadM_ * tilingData_->tilematmulTiling.matmulTiling.N; // 一个MM头块的数据个数
    tailPadDataCnt_ = tailPadM_ * tilingData_->tailmatmulTiling.matmulTiling.N; // 一个MM尾块的数据个数
    const int64_t offsetTilePad = tilePadDataCnt_ * sizeof(int8_t);             // 偏移计算
    const int64_t offsetTailPad = tailPadDataCnt_ * sizeof(int8_t);
    const int64_t offsetTilePadPerRank = offsetTilePad / rankNum;
    const int64_t offsetTailPadPerRank = offsetTailPad / rankNum;

    for (uint32_t i = 0U; i < mc2Tiling.tileCnt; ++i) { // 头块偏移
        const int64_t indexOffsetTile = i * offsetTilePad;
        const int64_t indexGatherOffsetTile = offsetTilePadPerRank * hccl_.GetRankId();
        all2allSendGM_[i] = all2allInGM_ + indexOffsetTile;
        all2allRecvGM_[i] = all2allOutGM_ + indexOffsetTile;
        allGatherSendGM_[i] = all2allOutGM_ + indexOffsetTile + indexGatherOffsetTile;
        allGatherRecvGM_[i] = allGatherOutGM_ + indexOffsetTile;
    }
    for (uint32_t i = 0U; i < mc2Tiling.tailCnt; ++i) { // 尾块偏移
        const int64_t indexOffsetTail = mc2Tiling.tileCnt * offsetTilePad + i * offsetTailPad;
        const int64_t indexGatherOffsetTail = offsetTailPadPerRank * hccl_.GetRankId();
        all2allSendGM_[mc2Tiling.tileCnt + i] = all2allInGM_ + indexOffsetTail;
        all2allRecvGM_[mc2Tiling.tileCnt + i] = all2allOutGM_ + indexOffsetTail;
        allGatherSendGM_[mc2Tiling.tileCnt + i] = all2allOutGM_ + indexOffsetTail + indexGatherOffsetTail;
        allGatherRecvGM_[mc2Tiling.tileCnt + i] = allGatherOutGM_ + indexOffsetTail;
    }

    if (notifyFlag_) {
        uint32_t nowAll2allIdx = 0U;
        uint32_t nowAllGatherIdx = 0U;
        uint32_t numN = (mc2Tiling.tileCnt + mc2Tiling.tailCnt) / NUM_TWO_PERTOKEN;
        uint32_t numReN = (mc2Tiling.tileCnt + mc2Tiling.tailCnt) % NUM_TWO_PERTOKEN;
        for (uint32_t i = 0U; i < numN; ++i) { // 按总核数下发
            all2allHandleId_[nowAll2allIdx] =
                hccl_.template AlltoAll<false>(all2allSendGM_[nowAll2allIdx], all2allRecvGM_[nowAll2allIdx],
                                               SendCountCheck(nowAll2allIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAll2allIdx++;
            all2allHandleId_[nowAll2allIdx] =
                hccl_.template AlltoAll<false>(all2allSendGM_[nowAll2allIdx], all2allRecvGM_[nowAll2allIdx],
                                               SendCountCheck(nowAll2allIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAll2allIdx++;

            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.template AllGather<false>(allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx],
                                                SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.template AllGather<false>(allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx],
                                                SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }

        if (numReN != 0U) { // 余数下发
            all2allHandleId_[nowAll2allIdx] =
                hccl_.template AlltoAll<false>(all2allSendGM_[nowAll2allIdx], all2allRecvGM_[nowAll2allIdx],
                                               SendCountCheck(nowAll2allIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAll2allIdx++;

            allGatherHandleId_[nowAllGatherIdx] =
                hccl_.template AllGather<false>(allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx],
                                                SendCountCheck(nowAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }
    }
}

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void
MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType, commMode>::InnerProcess(
    MmType &mmOp, uint32_t tileCnt, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling, uint32_t isAdd,
    uint32_t needUbBuffer, uint32_t padM, bool isTailFlag)
{
    const uint64_t aOffset = CalcShapeOffset(sizeof(xType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.Ka);
    const uint64_t cOffset = CalcShapeOffset(sizeof(YType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.N);
    const uint64_t tempOffset = CalcShapeOffset(sizeof(int8_t), padM, mmTiling->matmulTiling.N);
    const uint64_t pertokenOffset = sizeof(float) * mmTiling->matmulTiling.M;
    for (uint32_t i = 0U; i < tileCnt; ++i) {
        tPipe_->Reset();
        mmOp.Init(aGM_, bGM_, dequantScaleGM_, nullptr, biasGM_, pertokenScaleGM_, cGM_, workspaceGM_, mmTiling,
                  tPipe_);
        mmOp.Process();
        SyncAll<false>();
        if (isAdd) {
            MatmulAllReduceAddX3Kernel<YType>(cGM_, addGM_, cOffset / sizeof(YType), tilingData_->param.addX3UbCnt,
                                              tPipe_);
            addGM_ += cOffset;
            SyncAll<false>();
        }
        // 输出int8 存在 all2allInGM_
        MatmulAllReduceQuantPerchannelCommInt8<YType>(cGM_, commQuantScale1GM_, all2allInGM_, tPipe_,
                                                      mmTiling->matmulTiling.N, mmTiling->matmulTiling.M);
        SyncAll<false>();
        if (notifyFlag_) {
            hccl_.Commit(all2allHandleId_[all2allCommitIdx_]);
            all2allCommitIdx_++;
        }
        if (g_coreType == AscendC::AIV && isSendTileFlag_) { // 需要同步前一块
            if (notifyFlag_) {
                hccl_.Wait(all2allHandleId_[all2allWaitIdx_]);
            }
            SyncAll();
            if (isTailFlag && (i == 0U)) {
                MatmulAllReduceQuantReduceSumInt8<YType, commMode>(all2allOutGM_, commQuantScale1GM_,
                                                                   commQuantScale2GM_, tilePadM_,
                                                                   mmTiling->matmulTiling.N, tPipe_, hccl_);
                all2allOutGM_ += tilePadM_ * mmTiling->matmulTiling.N * sizeof(int8_t);
            } else {
                MatmulAllReduceQuantReduceSumInt8<YType, commMode>(all2allOutGM_, commQuantScale1GM_,
                                                                   commQuantScale2GM_, padM, mmTiling->matmulTiling.N,
                                                                   tPipe_, hccl_);
                all2allOutGM_ += padM * mmTiling->matmulTiling.N * sizeof(int8_t);
            }
            SyncAll();
            if (notifyFlag_) {
                hccl_.Commit(allGatherHandleId_[all2allWaitIdx_]);
                all2allWaitIdx_++;
            }
        }
        // 避免MatmulAllReduceQuantReduceSumInt8的vec计算和quant_batch_matmul_v3的vec计算发生数据覆盖，确保流水正常
        SyncAll<false>();
        isSendTileFlag_ = true;
        aGM_ += aOffset;
        cGM_ += cOffset;            // 偏原始大小
        all2allInGM_ += tempOffset; // 偏移 padM*N 大小
        pertokenScaleGM_ += pertokenOffset;
    }
}

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType, commMode>::Process()
{
    auto &&mc2Tiling = tilingData_->param;

    // all2all allgather Prepare + InnerProcess
    PrepareInit();
    MmType opTile;
    InnerProcess(opTile, mc2Tiling.tileCnt, &tilingData_->tilematmulTiling, mc2Tiling.isAdd, mc2Tiling.needUbBuffer,
                 tilePadM_, false);
    if (mc2Tiling.tailM != 0U) {
        MmType opTail;
        InnerProcess(opTail, mc2Tiling.tailCnt, &tilingData_->tailmatmulTiling, mc2Tiling.isAdd, mc2Tiling.needUbBuffer,
                     tailPadM_, true);
    }

    // 最后一次的 all2all 任务没有 wait，以及最后一次的 allgather 任务没有下发
    if (g_coreType == AscendC::AIV) {
        if (notifyFlag_) {
            hccl_.Wait(all2allHandleId_[all2allWaitIdx_]);
        }
        SyncAll();
        uint32_t padM = tilePadM_;
        uint32_t lastN = tilingData_->tilematmulTiling.matmulTiling.N;
        if (mc2Tiling.tailM != 0U) {
            padM = tailPadM_;
            lastN = tilingData_->tailmatmulTiling.matmulTiling.N;
        }
        MatmulAllReduceQuantReduceSumInt8<YType, commMode>(all2allOutGM_, commQuantScale1GM_, commQuantScale2GM_, padM,
                                                           lastN, tPipe_, hccl_);
        SyncAll();
        if (notifyFlag_) {
            hccl_.Commit(allGatherHandleId_[all2allWaitIdx_]);
            all2allWaitIdx_++;
        }

        // wait所有的allGather任务 + dequant
        const uint64_t outGmTileOff =
            tilingData_->tilematmulTiling.matmulTiling.M * tilingData_->tilematmulTiling.matmulTiling.N * sizeof(YType);
        const uint64_t outGmTailOff =
            tilingData_->tailmatmulTiling.matmulTiling.M * tilingData_->tailmatmulTiling.matmulTiling.N * sizeof(YType);
        for (uint32_t i = 0U; i < (mc2Tiling.tileCnt + mc2Tiling.tailCnt); ++i) { // 尾块偏移
            if (notifyFlag_) {
                hccl_.Wait(allGatherHandleId_[i]);
            }
            SyncAll();
            if (i < mc2Tiling.tileCnt) {
                MatmulAllReduceDequantPerchannelCommInt8<YType>(allGatherOutGM_, commQuantScale2GM_, outGM_, tPipe_,
                                                                tilingData_->tilematmulTiling.matmulTiling.N,
                                                                tilingData_->tilematmulTiling.matmulTiling.M);
                allGatherOutGM_ += tilePadDataCnt_ * sizeof(int8_t);
                outGM_ += outGmTileOff;
                SyncAll();
            } else {
                MatmulAllReduceDequantPerchannelCommInt8<YType>(allGatherOutGM_, commQuantScale2GM_, outGM_, tPipe_,
                                                                tilingData_->tailmatmulTiling.matmulTiling.N,
                                                                tilingData_->tailmatmulTiling.matmulTiling.M);
                allGatherOutGM_ += tailPadDataCnt_ * sizeof(int8_t);
                outGM_ += outGmTailOff;
                SyncAll();
            }
        }
        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }
}

#define INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(templateClass, coreType, commMode, scaleType, isATrans, \
                                                          isBTrans, ...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM}; \
        using OpType = templateClass<DTYPE_X1, DTYPE_X2, scaleType, DTYPE_BIAS, float, DTYPE_Y, X1_FORMAT, X2_FORMAT, \
                                     Y_FORMAT, isATrans, isBTrans, DTYPE_LOC_LOCAL, \
                                     Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>; \
        MatmulAllReduceQuantPertokenCommInt8<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType, commMode> op; \
        op.Init(aGM, bGM, biasGM, addGM, dequantGM, pertokenGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS, \
                &tilingData, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_ARCH35_H
