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
 * \file matmul_all_reduce_quant_comm_int8.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_QUANT_COMM_INT8_H
#define MATMUL_ALL_REDUCE_QUANT_COMM_INT8_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"

#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"
#include "matmul_all_reduce_add_x3_arch35.h"
#include "matmul_all_reduce_quant_perchannel_arch35.h"
#include "matmul_all_reduce_quant_reduce_sum_arch35.h"
#include "matmul_all_reduce_dequant_perchannel_arch35.h"

namespace MatmulAllReduceImpl {
constexpr uint32_t MAX_HANDLE_NUM = 16U;
constexpr uint32_t NUM_TWO = 2U;

using namespace AscendC;
using namespace Mc2Kernel;
using namespace MatmulAllReduceQuantPerchannelImpl;
using namespace MatmulAllReduceDequantPerchannelImpl;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
class MatmulAllReduceQuantCommInt8 {
public:
    __aicore__ inline MatmulAllReduceQuantCommInt8() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM,
                                GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
                                GM_ADDR workspaceGM, Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(MmType &mmOp, uint32_t tileCnt,
                                        DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling, uint32_t isAdd,
                                        uint32_t needUbBuffer, uint32_t curPadM, bool isTailFlag);
    __aicore__ inline void PrepareInit();
    __aicore__ inline uint32_t SendCountCheck(uint32_t prepareIndex);

    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *tilingData_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR addGM_;
    GM_ADDR dequantScaleGM_;
    GM_ADDR pertokenGM_;
    GM_ADDR commQuantScale1GM_;
    GM_ADDR commQuantScale2GM_;
    GM_ADDR workspaceGM_;
    GM_ADDR all2allInGM_;
    GM_ADDR all2allOutGM_;
    bool notifyFlag_{false};
    GM_ADDR allGatherOutGM_;
    typename HcclTypeSelector<commMode>::type hccl_;
    GM_ADDR outGM_;
    uint32_t tilePadM_ = 0U;
    uint32_t tailPadM_ = 0U;
    uint32_t tilePadDataCnt_ = 0U;
    uint32_t tailPadDataCnt_ = 0U;

    // 仅在0核上使用
    AscendC::HcclHandle all2allHandleId_[MAX_HANDLE_NUM] = {0};
    AscendC::HcclHandle allGatherHandleId_[MAX_HANDLE_NUM] = {0};
    GM_ADDR all2allSendGM_[MAX_HANDLE_NUM] = {0};
    GM_ADDR allGatherSendGM_[MAX_HANDLE_NUM] = {0};
    GM_ADDR all2allRecvGM_[MAX_HANDLE_NUM] = {0};
    GM_ADDR allGatherRecvGM_[MAX_HANDLE_NUM] = {0};
    int all2allCommitIdx_ = 0;
    int all2allWaitIdx_ = 0;
    int allGatherCommitIdx_ = 0;
    int allGatherWaitIdx_ = 0;
    // 所有核
    bool isSendTileFlag_ = false;
};

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantCommInt8<XType, WType, YType, MmType, CoreType, commMode>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM, GM_ADDR pertokenGM,
    GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *tilingData, TPipe *tPipe)
{
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    OOMInit(context);
    hccl_.InitV2(GetHcclContext<0>(), tilingData);
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTiling));
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTilingComm));
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    addGM_ = addGM;
    dequantScaleGM_ = dequantScaleGM;
    pertokenGM_ = pertokenGM;
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

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline uint32_t
MatmulAllReduceQuantCommInt8<XType, WType, YType, MmType, CoreType, commMode>::SendCountCheck(uint32_t prepareIndex)
{
    uint32_t sendCount = tilePadDataCnt_ / tilingData_->param.rankDim;
    if (prepareIndex >= tilingData_->param.tileCnt) {
        sendCount = tailPadDataCnt_ / tilingData_->param.rankDim;
    }
    return sendCount;
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantCommInt8<XType, WType, YType, MmType, CoreType, commMode>::PrepareInit()
{
    auto &&tilingParam = tilingData_->param;
    uint32_t rankNum = tilingParam.rankDim;
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
    const int64_t tempBufOffsetTilePad = tilePadDataCnt_ * sizeof(int8_t);      // 偏移计算
    const int64_t tempBufOffsetTailPad = tailPadDataCnt_ * sizeof(int8_t);
    const int64_t tempBufOffsetTilePadPerRank = tempBufOffsetTilePad / rankNum;
    const int64_t tempBufOffsetTailPadPerRank = tempBufOffsetTailPad / rankNum;

    for (uint32_t i = 0U; i < tilingParam.tileCnt; ++i) { // 头块偏移
        const int64_t indexOffsetTile = i * tempBufOffsetTilePad;
        const int64_t indexGatherOffsetTile = tempBufOffsetTilePadPerRank * hccl_.GetRankId();
        all2allSendGM_[i] = all2allInGM_ + indexOffsetTile;
        all2allRecvGM_[i] = all2allOutGM_ + indexOffsetTile;
        allGatherSendGM_[i] = all2allOutGM_ + indexOffsetTile + indexGatherOffsetTile;
        allGatherRecvGM_[i] = allGatherOutGM_ + indexOffsetTile;
    }
    for (uint32_t i = 0U; i < tilingParam.tailCnt; ++i) { // 尾块偏移
        const int64_t indexOffsetTail = tilingParam.tileCnt * tempBufOffsetTilePad + i * tempBufOffsetTailPad;
        const int64_t indexGatherOffsetTail = tempBufOffsetTailPadPerRank * hccl_.GetRankId();
        all2allSendGM_[tilingParam.tileCnt + i] = all2allInGM_ + indexOffsetTail;
        all2allRecvGM_[tilingParam.tileCnt + i] = all2allOutGM_ + indexOffsetTail;
        allGatherSendGM_[tilingParam.tileCnt + i] = all2allOutGM_ + indexOffsetTail + indexGatherOffsetTail;
        allGatherRecvGM_[tilingParam.tileCnt + i] = allGatherOutGM_ + indexOffsetTail;
    }

    if (notifyFlag_) {
        uint32_t curAll2allIdx = 0U;
        uint32_t curAllGatherIdx = 0U;
        uint32_t numN = (tilingParam.tileCnt + tilingParam.tailCnt) / NUM_TWO;
        uint32_t numReN = (tilingParam.tileCnt + tilingParam.tailCnt) % NUM_TWO;
        for (uint32_t i = 0U; i < numN; ++i) { // 按总核数下发
            all2allHandleId_[curAll2allIdx] =
                hccl_.template AlltoAll<false>(all2allSendGM_[curAll2allIdx], all2allRecvGM_[curAll2allIdx],
                                               SendCountCheck(curAll2allIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            curAll2allIdx++;
            all2allHandleId_[curAll2allIdx] =
                hccl_.template AlltoAll<false>(all2allSendGM_[curAll2allIdx], all2allRecvGM_[curAll2allIdx],
                                               SendCountCheck(curAll2allIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            curAll2allIdx++;

            allGatherHandleId_[curAllGatherIdx] =
                hccl_.template AllGather<false>(allGatherSendGM_[curAllGatherIdx], allGatherRecvGM_[curAllGatherIdx],
                                                SendCountCheck(curAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            curAllGatherIdx++;
            allGatherHandleId_[curAllGatherIdx] =
                hccl_.template AllGather<false>(allGatherSendGM_[curAllGatherIdx], allGatherRecvGM_[curAllGatherIdx],
                                                SendCountCheck(curAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            curAllGatherIdx++;
        }

        if (numReN != 0U) { // 余数下发
            all2allHandleId_[curAll2allIdx] =
                hccl_.template AlltoAll<false>(all2allSendGM_[curAll2allIdx], all2allRecvGM_[curAll2allIdx],
                                               SendCountCheck(curAll2allIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            curAll2allIdx++;

            allGatherHandleId_[curAllGatherIdx] =
                hccl_.template AllGather<false>(allGatherSendGM_[curAllGatherIdx], allGatherRecvGM_[curAllGatherIdx],
                                                SendCountCheck(curAllGatherIdx), AscendC::HCCL_DATA_TYPE_INT8, 0);
            curAllGatherIdx++;
        }
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantCommInt8<XType, WType, YType, MmType, CoreType, commMode>::InnerProcess(
    MmType &mmOp, uint32_t tileCnt, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling, uint32_t isAdd,
    uint32_t needUbBuffer, uint32_t curPadM, bool isTailFlag)
{
    const uint64_t aGmOffset = CalcShapeOffset(sizeof(XType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.Ka);
    const uint64_t cGmOffset = CalcShapeOffset(sizeof(YType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.N);
    const uint64_t tempOffset = CalcShapeOffset(sizeof(int8_t), curPadM, mmTiling->matmulTiling.N);

    for (uint32_t i = 0U; i < tileCnt; ++i) {
        tPipe_->Reset();
        mmOp.Init(aGM_, bGM_, biasGM_, dequantScaleGM_, pertokenGM_, cGM_, workspaceGM_, mmTiling, tPipe_);
        mmOp.Process();
        SyncAll<false>();
        if (isAdd) {
            MatmulAllReduceAddX3Kernel<YType>(cGM_, addGM_, cGmOffset / sizeof(YType), tilingData_->param.addX3UbCnt,
                                              tPipe_);
            addGM_ += cGmOffset;
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
                                                                   commQuantScale2GM_, curPadM,
                                                                   mmTiling->matmulTiling.N, tPipe_, hccl_);
                all2allOutGM_ += curPadM * mmTiling->matmulTiling.N * sizeof(int8_t);
            }
            SyncAll();
            if (notifyFlag_) {
                hccl_.Commit(allGatherHandleId_[all2allWaitIdx_]);
                all2allWaitIdx_++;
            }
        }
        isSendTileFlag_ = true;
        aGM_ += aGmOffset;
        cGM_ += cGmOffset;          // 偏原始大小
        all2allInGM_ += tempOffset; // 偏移 padM*N 大小
    }
}

template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, int commMode>
__aicore__ inline void MatmulAllReduceQuantCommInt8<XType, WType, YType, MmType, CoreType, commMode>::Process()
{
    auto &&tilingParam = tilingData_->param;

    // all2all allgather Prepare + InnerProcess
    PrepareInit();
    MmType opTile;
    InnerProcess(opTile, tilingParam.tileCnt, &tilingData_->tilematmulTiling, tilingParam.isAdd,
                 tilingParam.needUbBuffer, tilePadM_, false);
    if (tilingParam.tailM != 0U) {
        MmType opTail;
        InnerProcess(opTail, tilingParam.tailCnt, &tilingData_->tailmatmulTiling, tilingParam.isAdd,
                     tilingParam.needUbBuffer, tailPadM_, true);
    }

    // 最后一次的 all2all 任务没有 wait，以及最后一次的 allgather 任务没有下发
    if (g_coreType == AscendC::AIV) {
        if (notifyFlag_) {
            hccl_.Wait(all2allHandleId_[all2allWaitIdx_]);
        }
        SyncAll();
        uint32_t lastN = tilingData_->tilematmulTiling.matmulTiling.N;
        uint32_t padM = tilePadM_;
        if (tilingParam.tailM != 0U) {
            lastN = tilingData_->tailmatmulTiling.matmulTiling.N;
            padM = tailPadM_;
        }
        MatmulAllReduceQuantReduceSumInt8<YType, commMode>(all2allOutGM_, commQuantScale1GM_, commQuantScale2GM_, padM,
                                                           lastN, tPipe_, hccl_);
        SyncAll();
        if (notifyFlag_) {
            hccl_.Commit(allGatherHandleId_[all2allWaitIdx_]);
            all2allWaitIdx_++;
        }

        // wait所有的allGather任务 + dequant
        const uint64_t outGmTileOffset =
            tilingData_->tilematmulTiling.matmulTiling.M * tilingData_->tilematmulTiling.matmulTiling.N * sizeof(YType);
        const uint64_t outGmTailOffset =
            tilingData_->tailmatmulTiling.matmulTiling.M * tilingData_->tailmatmulTiling.matmulTiling.N * sizeof(YType);
        for (uint32_t i = 0U; i < (tilingParam.tileCnt + tilingParam.tailCnt); ++i) { // 尾块偏移
            if (notifyFlag_) {
                hccl_.Wait(allGatherHandleId_[i]);
            }
            SyncAll();
            if (i < tilingParam.tileCnt) {
                MatmulAllReduceDequantPerchannelCommInt8<YType>(allGatherOutGM_, commQuantScale2GM_, outGM_, tPipe_,
                                                                tilingData_->tilematmulTiling.matmulTiling.N,
                                                                tilingData_->tilematmulTiling.matmulTiling.M);
                outGM_ += outGmTileOffset;
                allGatherOutGM_ += tilePadDataCnt_ * sizeof(int8_t);
                SyncAll();
            } else {
                MatmulAllReduceDequantPerchannelCommInt8<YType>(allGatherOutGM_, commQuantScale2GM_, outGM_, tPipe_,
                                                                tilingData_->tailmatmulTiling.matmulTiling.N,
                                                                tilingData_->tailmatmulTiling.matmulTiling.M);
                outGM_ += outGmTailOffset;
                allGatherOutGM_ += tailPadDataCnt_ * sizeof(int8_t);
                SyncAll();
            }
        }
        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }
}

#define INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(templateClass, coreType, commMode, scaleType, ...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM}; \
        using OpType = templateClass<DTYPE_X1, DTYPE_X2, scaleType, DTYPE_BIAS, DTYPE_Y, X1_FORMAT, X2_FORMAT, \
                                     Y_FORMAT, __VA_ARGS__>; \
        MatmulAllReduceQuantCommInt8<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType, commMode> op; \
        op.Init(aGM, bGM, biasGM, addGM, dequantGM, pertokenGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS, \
                &tilingData, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
    } while (0)

} // namespace MatmulAllReduceImpl
#endif // MAtMUL_ALL_REDUCE_QUANT_COMM_INT8_H
