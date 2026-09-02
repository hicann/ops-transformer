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
 * \file matmul_all_reduce_weight_quant_310.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H
#define MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#else
#include "rac_server.h"
#endif
#include "mm_allreduce.h"
#include "../../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_weight_nz_performance.h"
#include "../weight_quant_matmul_all_reduce_tiling_data.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using namespace Mc2WeightQuantBatchMatmulV2;
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
class MatmulAllReduceWeightQuant310 {
public:
    __aicore__ inline MatmulAllReduceWeightQuant310() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
                                GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData *tilingData, TPipe *tPipe,
                                HcclServer *hcclServer);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(uint32_t tileCnt, Mc2WeightQuantBatchMatmulV2NzTilingData *mmTiling,
                                        uint32_t shift, int32_t coreNum);
    Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData *tilingData_;
    HcclServer *hcclServer_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR antiquantScaleGM_;
    GM_ADDR antiquantOffsetGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR cGM_;
    bool notifyFlag_{false};
    int32_t coreNum_{0};
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulAllReduceWeightQuant310<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                              hasAntiQuantOffset>::InnerProcess(uint32_t tileCnt,
                                                                Mc2WeightQuantBatchMatmulV2NzTilingData *mmTiling,
                                                                uint32_t shift, int32_t coreNum)
{
    const uint64_t aOffset = CalcShapeOffset(sizeof(xType), mmTiling->mSize, mmTiling->kSize);
    const uint64_t cOffset = CalcShapeOffset(sizeof(yType), mmTiling->mSize, mmTiling->nSize);
    if (GetBlockIdx() < coreNum) {
        Mc2WeightQuantBatchMatmulV2WeightNzPerformanceKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                             antiQuantType, hasAntiQuantOffset>
            op;
        for (uint32_t i = 1U; i <= tileCnt; ++i) {
            if (i == 1U) {
                tPipe_->Reset();
                op.Init(aGM_, bGM_, antiquantScaleGM_, antiquantOffsetGM_, nullptr, nullptr, biasGM_, cGM_,
                        workspaceGM_, mmTiling, tPipe_);
            } else {
                op.UpdateGlobalAddr(aGM_, bGM_, antiquantScaleGM_, antiquantOffsetGM_, nullptr, nullptr, biasGM_, cGM_,
                                    workspaceGM_);
            }
            op.Process();
            hcclServer_->TurnNotifyRun(block_idx, coreNum, i + shift);
            aGM_ += aOffset;
            cGM_ += cOffset;
        }
    } else {
        for (uint32_t i = 1U; i <= tileCnt; ++i) {
            aGM_ += aOffset;
            cGM_ += cOffset;
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulAllReduceWeightQuant310<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR biasGM, GM_ADDR cGM,
    GM_ADDR workspaceGM, Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData *tilingData, TPipe *tPipe,
    HcclServer *hcclServer)
{
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + tilingData->msg.notifyOff);
    TBuf<TPosition::VECCALC> tmpBuf;
    tPipe->InitBuffer(tmpBuf, 256);
    auto &&cfg = tilingData->param;
    auto &&tiling = tilingData->tilematmulTiling.matmulTiling;
    hcclServer->Init(workspaceMsg, (tilingData->msg).debugMode, tmpBuf);
    workspaceGM += cfg.nd2NzWorkLen;
    workspaceGM += cfg.biasLen;
    GM_ADDR softSyncAddr = workspaceGM;
    int32_t tileNum = tilingData->tilematmulTiling.cubeBlockDimN * tilingData->tilematmulTiling.cubeBlockDimM;
    int32_t tailNum = tilingData->tailmatmulTiling.cubeBlockDimN * tilingData->tailmatmulTiling.cubeBlockDimM;
    coreNum_ = tileNum > tailNum ? tileNum : tailNum;
    hcclServer->InitSoftSync(softSyncAddr, coreNum_, tmpBuf);
    workspaceGM += AC_MAX_AIV * DEFAULT_BLK_NUM;

    if (tilingData->msg.useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN &&
        context->config.determinism != 1) {
        cGM_ = (__gm__ uint8_t *)(context->windowsIn[context->rankId]);
    } else {
        cGM_ = cGM;
    }
    tilingData_ = tilingData;
    hcclServer_ = hcclServer;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    antiquantScaleGM_ = antiquantScaleGM;
    antiquantOffsetGM_ = antiquantOffsetGM;
    workspaceGM_ = workspaceGM;
    if (GetBlockIdx() == 0 && (g_coreType == AIC || g_coreType == MIX)) {
        notifyFlag_ = true;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulAllReduceWeightQuant310<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                     hasAntiQuantOffset>::Process()
{
    auto &&mc2Tiling = tilingData_->param;
    int32_t tileNum = tilingData_->tilematmulTiling.cubeBlockDimN * tilingData_->tilematmulTiling.cubeBlockDimM;
    int32_t tailNum = tilingData_->tailmatmulTiling.cubeBlockDimN * tilingData_->tailmatmulTiling.cubeBlockDimM;
    InnerProcess(mc2Tiling.tileCnt, &tilingData_->tilematmulTiling, 0U, tileNum);
    if (mc2Tiling.tailM != 0U) {
        InnerProcess(mc2Tiling.tailCnt, &tilingData_->tailmatmulTiling, mc2Tiling.tileCnt, tailNum);
    }
    if (notifyFlag_) {
        hcclServer_->TurnWait(mc2Tiling.tileCnt + mc2Tiling.tailCnt);
    }
}

__aicore__ inline void MatMulEmptyTensorBrcBias(GM_ADDR biasGM, GM_ADDR cGM,
                                                Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData *tilingData,
                                                TBuf<TPosition::VECCALC> &tmpBuf)
{
    // 搬运biase对齐部分
    int32_t cSizeHalf = (tilingData->param.rankN * tilingData->param.rankM) * sizeof(DTYPE_X1) / sizeof(DTYPE_Y);
    GlobalTensor<DTYPE_Y> cGlobalHalf;
    cGlobalHalf.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(cGM), cSizeHalf);
    LocalTensor<DTYPE_Y> bias = tmpBuf.Get<DTYPE_Y>();
    GlobalTensor<DTYPE_Y> biasGlobal;
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(biasGM));
    TBuffAddr buffAddr;
    buffAddr.logicPos = (uint8_t)QuePosition::VECCALC;
    uint32_t eleCntWq = 32 / sizeof(DTYPE_Y);
    uint32_t alSizeWq = (tilingData->param.rankN / eleCntWq) * eleCntWq;
    if (alSizeWq > 0) {
        DataCopy(bias, biasGlobal, alSizeWq);
        event_t eventIDWq = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDWq);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDWq);
        for (uint32_t i = 0; i < tilingData->param.rankM; ++i) {
            uint32_t offsetDstWq = i * tilingData->param.rankN;
            DataCopy(cGlobalHalf[offsetDstWq], bias, alSizeWq);
        }
    }
    if (tilingData->param.rankN % eleCntWq) {
        // 搬运biase非对齐部分
        event_t eventIDWq = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDWq);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDWq);
        DataCopy(bias, biasGlobal[tilingData->param.rankN - eleCntWq], eleCntWq);
        eventIDWq = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDWq);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDWq);
        for (uint32_t i = 0; i < tilingData->param.rankM; ++i) {
            uint32_t offsetDstWq = (i + 1) * tilingData->param.rankN - eleCntWq;
            DataCopy(cGlobalHalf[offsetDstWq], bias, eleCntWq);
        }
    }
}

__aicore__ inline void WeightQuantEmptyTensorKernel(GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                    Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData *tilingData,
                                                    HcclServer *hcclServer)
{
    TBuf<TPosition::VECCALC> tmpBuf;
    GetTPipePtr()->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
    // 初始化hccl
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + (tilingData->msg).notifyOff);
    hcclServer->Init(workspaceMsg, (tilingData->msg).debugMode, tmpBuf);
    if (tilingData->tilematmulTiling.matmulTiling.usedCoreNum > 1) {
        hcclServer->InitSoftSync(workspaceGM, tilingData->tilematmulTiling.matmulTiling.usedCoreNum, tmpBuf);
    }
    // 初始化输出tensor
    int32_t cSize = (tilingData->param.rankN * tilingData->param.rankM) * sizeof(DTYPE_X1) / sizeof(int32_t);
    GlobalTensor<int32_t> cGlobal;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cGM), cSize);
    if (block_idx == 0) {
        InitOutput<int32_t>(cGlobal, cSize, (int32_t)0);
        if (tilingData->tilematmulTiling.matmulTiling.isBias) {
            MatMulEmptyTensorBrcBias(biasGM, cGM, tilingData, tmpBuf);
        }
    }
    // 通知AICPU计算结束
    if (block_idx == 0) {
        hcclServer->TurnNotifyRun(block_idx, tilingData->tilematmulTiling.matmulTiling.usedCoreNum, 1);
        // 通过一个核轮询并清除数据，防止多核之间写后读依赖
        hcclServer->TurnWait(tilingData->param.tileCnt + tilingData->param.tailCnt);
    }
}

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(templateClass, ...) \
    do { \
        GET_TILING_DATA_MEMBER(Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData, msg, msg, tilingGM); \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) { \
            GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData, tilingData, tilingGM); \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_Y, __VA_ARGS__> op; \
            op.Init(aGM, bGM, antiquantScaleGM, antiquantOffsetGM, biasGM, cGM, userWS, &tilingData, &tPipe, \
                    &hcclServer); \
            op.Process(); \
        } \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_WEIGHT_QUANT_310_H
