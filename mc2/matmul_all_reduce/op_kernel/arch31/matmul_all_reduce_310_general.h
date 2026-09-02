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
 * \file matmul_all_reduce_310_general.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_310_GENERAL_H
#define MATMUL_ALL_REDUCE_310_GENERAL_H

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

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <class A_TYPE, class B_TYPE, class BIAS_TYPE, class C_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
class MatmulAllReduce310General {
public:
    __aicore__ inline MatmulAllReduce310General() {}
    __aicore__ inline void Init(GM_ADDR workspaceGM, Mc2Tiling::RCSTiling *cfg, Mc2Tiling::Mc2Msg *msg,
                                AscendC::tiling::TCubeTiling *tiling, HcclServer *hcclServer);
    __aicore__ inline void Process(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR dequantGM,
                                   GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR cGM,
                                   GM_ADDR workspaceGM, Mc2Tiling::RCSTiling *cfg, Mc2Tiling::Mc2Msg *msg,
                                   AscendC::tiling::TCubeTiling *tiling, AscendC::tiling::TCubeTiling *tailTiling,
                                   Mc2Tiling::Mc2L2cacheTilePara *tileL2cacheTiling,
                                   Mc2Tiling::Mc2L2cacheTilePara *tailL2cacheTiling, TPipe *tPipe,
                                   HcclServer *hcclServer);

private:
    using A_T = typename A_TYPE::T;
    LocalTensor<uint8_t> mmFormatUb_;
};

template <class A_TYPE, class B_TYPE, class BIAS_TYPE, class C_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulAllReduce310General<A_TYPE, B_TYPE, BIAS_TYPE, C_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                          hasAntiQuantOffset>::Init(GM_ADDR workspaceGM, Mc2Tiling::RCSTiling *cfg,
                                                    Mc2Tiling::Mc2Msg *msg, AscendC::tiling::TCubeTiling *tiling,
                                                    HcclServer *hcclServer)
{
    TBuf<TPosition::VECCALC> tmpBuf;
    GetTPipePtr()->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + msg->notifyOff);
    hcclServer->Init(workspaceMsg, msg->debugMode, tmpBuf);

    workspaceGM += cfg->nd2NzWorkLen;
    workspaceGM += cfg->biasLen;
    GM_ADDR softSyncAddr = workspaceGM;
    hcclServer->InitSoftSync(softSyncAddr, tiling->usedCoreNum, tmpBuf);
    workspaceGM += AC_MAX_AIV * DEFAULT_BLK_NUM;

    int32_t bufferRatio = 2;
    mmFormatUb_ = tmpBuf.Get<uint8_t>(tiling->transLength * bufferRatio);
}

template <class A_TYPE, class B_TYPE, class BIAS_TYPE, class C_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulAllReduce310General<A_TYPE, B_TYPE, BIAS_TYPE, C_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                          hasAntiQuantOffset>::Process(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR dequantGM,
                                                       GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM, GM_ADDR cGM,
                                                       GM_ADDR workspaceGM, Mc2Tiling::RCSTiling *cfg,
                                                       Mc2Tiling::Mc2Msg *msg, AscendC::tiling::TCubeTiling *tiling,
                                                       AscendC::tiling::TCubeTiling *tailTiling,
                                                       Mc2Tiling::Mc2L2cacheTilePara *tileL2cacheTiling,
                                                       Mc2Tiling::Mc2L2cacheTilePara *tailL2cacheTiling, TPipe *tPipe,
                                                       HcclServer *hcclServer)
{
    MatMulKernel_AllReduce<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, false, false, WeightQuant, antiQuantType,
                           hasAntiQuantOffset>(aGM, bGM, cGM, biasGM, dequantGM, *tiling, *cfg, *tileL2cacheTiling,
                                               hcclServer, cfg->tileCnt, (cfg->tailM ? false : true), false,
                                               mmFormatUb_, antiquantScaleGM, antiquantOffsetGM);
    if (cfg->tailM) { // 存在尾块
        aGM = GetTailA(aGM, *tiling, cfg->tileCnt);
        cGM = GetTailC(cGM, *tiling, cfg->tileCnt);
        MatMulKernel_AllReduce<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, false, false, WeightQuant, antiQuantType,
                               hasAntiQuantOffset>(aGM, bGM, cGM, biasGM, dequantGM, *tailTiling, *cfg,
                                                   *tailL2cacheTiling, hcclServer, cfg->tailCnt, true, true,
                                                   mmFormatUb_, antiquantScaleGM, antiquantOffsetGM);
    }
}

__aicore__ inline void MatMulEmptyTensorBrcBias(GM_ADDR biasGM, GM_ADDR cGM,
                                                Mc2Tiling::MatmulAllReduceTilingData *tilingData,
                                                TBuf<TPosition::VECCALC> &tmpBuf)
{
    // 搬运biase对齐部分
    int32_t cSizeHalf = (tilingData->param.rankN * tilingData->param.rankM) * sizeof(DTYPE_X1) / sizeof(DTYPE_Y);
    GlobalTensor<DTYPE_Y> cGlobalHalf;
    cGlobalHalf.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(cGM), cSizeHalf);
    LocalTensor<DTYPE_Y> bias = tmpBuf.Get<DTYPE_Y>();
    GlobalTensor<DTYPE_Y> biasGlobal;
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(biasGM));
    uint32_t eleCntGen = 32 / sizeof(DTYPE_Y);
    uint32_t alSizeGen = (tilingData->param.rankN / eleCntGen) * eleCntGen;
    if (alSizeGen > 0) {
        DataCopy(bias, biasGlobal, alSizeGen);
        event_t eventIDGen = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDGen);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDGen);
        for (uint32_t i = 0; i < tilingData->param.rankM; ++i) {
            uint32_t offsetDstGen = i * tilingData->param.rankN;
            DataCopy(cGlobalHalf[offsetDstGen], bias, alSizeGen);
        }
    }
    if (tilingData->param.rankN % eleCntGen) {
        // 搬运biase非对齐部分
        event_t eventIDGen = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDGen);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDGen);
        DataCopy(bias, biasGlobal[tilingData->param.rankN - eleCntGen], eleCntGen);
        eventIDGen = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDGen);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDGen);
        for (uint32_t i = 0; i < tilingData->param.rankM; ++i) {
            uint32_t offsetDstGen = (i + 1) * tilingData->param.rankN - eleCntGen;
            DataCopy(cGlobalHalf[offsetDstGen], bias, eleCntGen);
        }
    }
}

__aicore__ inline void MatMulEmptyTensorKernel(GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                               Mc2Tiling::MatmulAllReduceTilingData *tilingData, HcclServer *hcclServer)
{
    TBuf<TPosition::VECCALC> tmpBuf;
    GetTPipePtr()->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
    // 初始化hccl
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + (tilingData->msg).notifyOff);
    hcclServer->Init(workspaceMsg, (tilingData->msg).debugMode, tmpBuf);
    if (tilingData->matmulTiling.usedCoreNum > 1) {
        hcclServer->InitSoftSync(workspaceGM, tilingData->matmulTiling.usedCoreNum, tmpBuf);
    }
    // 初始化输出tensor
    int32_t cSize = (tilingData->param.rankN * tilingData->param.rankM) * sizeof(DTYPE_X1) / sizeof(int32_t);
    GlobalTensor<int32_t> cGlobal;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cGM), cSize);
    if (block_idx == 0) {
        InitOutput<int32_t>(cGlobal, cSize, (int32_t)0);
        if (tilingData->matmulTiling.isBias) {
            MatMulEmptyTensorBrcBias(biasGM, cGM, tilingData, tmpBuf);
        }
    }
    // 通知AICPU计算结束
    if (block_idx == 0) {
        hcclServer->TurnNotifyRun(block_idx, tilingData->matmulTiling.usedCoreNum, 1);
        // 通过一个核轮询并清除数据，防止多核之间写后读依赖
        hcclServer->TurnWait(tilingData->param.tileCnt + tilingData->param.tailCnt);
    }
}

#define INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(templateClass, ...) \
    do { \
        GET_TILING_DATA_MEMBER(Mc2Tiling::MatmulAllReduceTilingData, matmulTiling, tiling, tilingGM); \
        GET_TILING_DATA_MEMBER(Mc2Tiling::MatmulAllReduceTilingData, tailTiling, tailTiling, tilingGM); \
        GET_TILING_DATA_MEMBER(Mc2Tiling::MatmulAllReduceTilingData, msg, msg, tilingGM); \
        GET_TILING_DATA_MEMBER(Mc2Tiling::MatmulAllReduceTilingData, tailL2cacheTiling, tailL2cacheTiling, tilingGM); \
        GET_TILING_DATA_MEMBER(Mc2Tiling::MatmulAllReduceTilingData, tileL2cacheTiling, tileL2cacheTiling, tilingGM); \
        GET_TILING_DATA_MEMBER(Mc2Tiling::MatmulAllReduceTilingData, param, cfg, tilingGM); \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) { \
            templateClass<aType, bType, biasType, cType, __VA_ARGS__> op; \
            op.Init(workspaceGM, &cfg, &msg, &tiling, &hcclServer); \
            op.Process(aGM, bGM, biasGM, dequantGM, antiquantScaleGM, antiquantOffsetGM, cGM, workspaceGM, &cfg, &msg, \
                       &tiling, &tailTiling, &tileL2cacheTiling, &tailL2cacheTiling, &tPipe, &hcclServer); \
        } \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_310_GENERAL_H
