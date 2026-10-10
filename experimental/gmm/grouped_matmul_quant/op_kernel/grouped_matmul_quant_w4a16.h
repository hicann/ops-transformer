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
 * \file grouped_matmul_quant_w4a16.h
 * \brief
 */
#ifndef _ASCENDC_GROUPED_MATMUL_QUANT_W4A16_H_
#define _ASCENDC_GROUPED_MATMUL_QUANT_W4A16_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SYNC_PING_AIV_AIC_FLAG = 1;
constexpr uint32_t SYNC_PONG_AIV_AIC_FLAG = 2;
constexpr uint32_t SYNC_PING_AIC_AIV_FLAG = 3;
constexpr uint32_t SYNC_PONG_AIC_AIV_FLAG = 4;
constexpr uint32_t SYNC_CLEAR_AIV_AIC_FLAG = 11;
constexpr uint32_t SYNC_CLEAR_AIC_AIV_FLAG = 12;
constexpr uint32_t SYNC_CLEAR_AIC_FLAG = 13;
constexpr uint32_t SYNC_CLEAR_AIV_FLAG = 14;
constexpr uint32_t GROUP_MAX = 256;
constexpr uint32_t MY_BUFFER_NUM = 8;
constexpr float SPLITK_THRES = 1.12;

template <typename T>
class GroupedMatmulQuantW4A16 {
public:
    TPipe pipe;

    __aicore__ inline GroupedMatmulQuantW4A16() {}
    __aicore__ inline void Process();
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR weight_offset,
                                GM_ADDR group_list, GM_ADDR y, GM_ADDR usrworkspace,
                                const GroupedMatmulQuantTilingData* __restrict tilingData);

protected:
    __aicore__ inline void InitWorkspaceTensors(GM_ADDR usrWorkspace);
    __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale,
                                             GM_ADDR weight_offset, GM_ADDR group_list, GM_ADDR y);
    __aicore__ inline void InitLocalTensors();
    __aicore__ inline void ClearOutTensor(uint32_t loop, uint32_t baseN, uint32_t tailN, uint32_t tailCoreNum);
    __aicore__ inline void CastOutTensor(uint32_t loop, uint32_t baseN, uint32_t tailN, uint32_t tailCoreNum);
    __aicore__ inline void InitEventId();
    __aicore__ inline void TilingInKernel();
    __aicore__ inline void AntiQuantWeight(uint32_t task_id, uint32_t ping_pong, bool endKFlag, uint32_t baseN,
                                           uint32_t baseK, uint32_t offsetN, uint32_t offsetK);
    __aicore__ inline void ComputeMatMul(uint32_t task_id, uint32_t ping_pong, bool startKFlag, bool endKFlag,
                                         uint32_t realM, uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                         uint32_t offsetM, uint32_t offsetN, uint32_t offsetK);
    // GlobalTensor
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm0, yGm;
    GlobalTensor<int32_t> wGm0, wGm;
    GlobalTensor<T> wScaleGm0, wScaleGm;
    GlobalTensor<half> wOffsetGm0, wOffsetGm;
    GlobalTensor<int64_t> groupListGm;

    GlobalTensor<T> workspaceWGm[2][MY_BUFFER_NUM];
    GlobalTensor<float> workspaceCGm0, workspaceCGm;
    // LocalTensor
    TBuf<TPosition::VECCALC> UbBuf;
    TBuf<TPosition::TSCM> L1Buf;
    TBuf<TPosition::A2> L0ABuf;
    TBuf<TPosition::B2> L0BBuf;
    TBuf<TPosition::CO1> L0CBuf;
    // for antiquant
    LocalTensor<int4b_t> ubWInt4;
    LocalTensor<half> ubWHalf;
    LocalTensor<float> ubWFloat;
    LocalTensor<T> ubWT;
    LocalTensor<T> ubWScaleT;
    LocalTensor<half> ubWOffsetHalf;
    LocalTensor<float> ubWScaleFloat;
    // for matmul
    LocalTensor<T> L1MatA[2];
    LocalTensor<T> L1MatB[2];
    LocalTensor<T> L0MatA[2];
    LocalTensor<T> L0MatB[2];
    LocalTensor<float> L0MatC;
    //
    LocalTensor<float> clearUb;
    LocalTensor<float> castUbFloat[2];
    LocalTensor<T> castUbBF16[2];
    const GroupedMatmulQuantTilingData* __restrict tilingData;

    int32_t block_id;
    int32_t subblock_id;

    int32_t taskNum = 0;
    int32_t taskOffset = 0;

    event_t eventIdVToMTE3;
    event_t eventIdMTE3ToV[2];
    event_t eventIdVToMTE2[3];
    event_t eventIdMTE2ToV[3];
    event_t eventIdMTE3ToMTE2;

    event_t eventIdMTE1ToMTE2[2];
    event_t eventIdMTE2ToMTE1;
    event_t eventIdMTE1ToM;
    event_t eventIdMToMTE1[2];
    event_t eventIdMToFIX;
    event_t eventIdFIXToM;

    float deqScalar = 1.0;
    bool splitK = false;

    int32_t baseKNum;
    int32_t baseNNum;
    int32_t realM[GROUP_MAX];
    int32_t baseMNum[GROUP_MAX];
};
#endif
