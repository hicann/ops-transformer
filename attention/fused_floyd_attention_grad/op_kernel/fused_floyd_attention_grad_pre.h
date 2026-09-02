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
 * \file fused_floyd_attention_grad_pre.h
 * \brief
 */

#ifndef fused_floyd_attention_grad_PRE_KERNEL_H_
#define fused_floyd_attention_grad_PRE_KERNEL_H_

#include "kernel_operator.h"

template <typename T1, typename T2, typename TILING_TYPE, const bool INIT_OUTPUT = true>
class FusedFloydAttentionGradPre {
public:
    __aicore__ inline FusedFloydAttentionGradPre(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dk_1,
                                __gm__ uint8_t *dv_1, __gm__ uint8_t *workspace, const TILING_TYPE *ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dvGm, dv1Gm, dk1WorkSpaceGm, dv1WorkSpaceGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm;

    const TILING_TYPE *TilingData;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;

    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t kvPreBlockFactor;
    uint32_t kvPreBlockTotal;
    uint32_t kvPreBlockTail;
    uint32_t k1v1PreBlockFactor;
    uint32_t k1v1PreBlockTotal;
    uint32_t k1v1PreBlockTail;

    int64_t qSizeAlign;
    int64_t kvSizeAlign;
    int64_t k1v1SizeAlign;

    int64_t initdqSize;
    int64_t dqOffset;
    int64_t initdkSize;
    int64_t dkvOffset;
    int64_t initdk1v1Size;
    int64_t dk1v1Offset;

    uint32_t maskUsedCoreNum;
    uint32_t maskUBProcessNum;
    uint32_t maskTailUBProcessNum;
    uint32_t maskUBLoop;

    DataCopyParams copyParams;
    BinaryRepeatParams repParams;
    half padValue{1.0};
};

template <typename T1, typename T2, typename TILING_TYPE, const bool INIT_OUTPUT>
__aicore__ inline void FusedFloydAttentionGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dk_1, __gm__ uint8_t *dv_1,
    __gm__ uint8_t *workspace, const TILING_TYPE *orgTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    TilingData = orgTilingData;
    pipe = pipe_in;

    maskUsedCoreNum = TilingData->preTilingData.maskCoreNum;

    dvGm.SetGlobalBuffer((__gm__ float *)dv);
    dv1Gm.SetGlobalBuffer((__gm__ float *)dv_1);

    if constexpr (INIT_OUTPUT) {
        // tiling_data
        qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
        qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
        qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
        qSizeAlign = TilingData->postTilingData.qSizeAlign;
        kvPreBlockFactor = TilingData->preTilingData.kvPreBlockFactor;
        kvPreBlockTotal = TilingData->preTilingData.kvPreBlockTotal;
        kvPreBlockTail = TilingData->preTilingData.kvPreBlockTail;
        kvSizeAlign = TilingData->postTilingData.kvSizeAlign;

        k1v1PreBlockFactor = TilingData->preTilingData.k1v1PreBlockFactor;
        k1v1PreBlockTotal = TilingData->preTilingData.k1v1PreBlockTotal;
        k1v1PreBlockTail = TilingData->preTilingData.k1v1PreBlockTail;
        k1v1SizeAlign = TilingData->postTilingData.k1v1SizeAlign;

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      TilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      TilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      TilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
        dk1WorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                       TilingData->postTilingData.dk1WorkSpaceOffset / sizeof(float));
        dv1WorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                       TilingData->postTilingData.dv1WorkSpaceOffset / sizeof(float));

        initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
        initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
        dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
        initdk1v1Size = cBlockIdx == k1v1PreBlockTotal - 1 ? k1v1PreBlockTail : k1v1PreBlockFactor;
        dk1v1Offset = ((int64_t)cBlockIdx) * k1v1PreBlockFactor;
    }
}

template <typename T1, typename T2, typename TILING_TYPE, const bool INIT_OUTPUT>
__aicore__ inline void FusedFloydAttentionGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT>::Process()
{
    // process clear dq dk dv workspace
    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.qPreBlockTotal) {
        if constexpr (INIT_OUTPUT) {
            InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        }
    }

    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.kvPreBlockTotal) {
        if constexpr (INIT_OUTPUT) {
            InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
            InitOutput<float>(dk1WorkSpaceGm[dk1v1Offset], initdk1v1Size, 0);
            if constexpr (IsSameType<T1, float>::value) {
                InitOutput<float>(dvGm[dkvOffset], initdkSize, 0);
                InitOutput<float>(dv1Gm[dk1v1Offset], initdk1v1Size, 0);
            } else {
                InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
                InitOutput<float>(dv1WorkSpaceGm[dk1v1Offset], initdk1v1Size, 0);
            }
        }
    }
}

template <typename T1, typename T2, typename TILING_TYPE, const bool INIT_OUTPUT>
__aicore__ inline void FusedFloydAttentionGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT>::SyncALLCores()
{
    SyncAll();
}

#endif // fused_floyd_attention_grad_PRE_KERNEL_H_
