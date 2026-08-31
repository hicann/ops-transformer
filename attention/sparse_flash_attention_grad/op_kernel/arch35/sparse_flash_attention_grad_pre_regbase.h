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
 * \file sparse_flash_attention_grad_pre_regbase.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_GRAD_PRE_KERNEL_REGBASE_H_
#define SPARSE_FLASH_ATTENTION_GRAD_PRE_KERNEL_REGBASE_H_
#include "kernel_operator.h"

using namespace AscendC;

template <typename T1, typename T2, const uint32_t IS_TND = 0, bool IS_SINKS = false, bool IS_DETER = false,
          bool KV_MERGE = false>
class FlashAttentionGradPreRegbase {
public:
    __aicore__ inline FlashAttentionGradPreRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dSinks,
                                __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *workspace,
                                const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase *ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    GlobalTensor<T1> dqGm, dkGm, dvGm;
    GlobalTensor<float> dSinksGm, dSinkWorkSpaceGm;

    const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase *TilingData;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t S1S2_TND = 3;

    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t qPostBlockTotal;
    uint32_t kPreBlockFactor;
    uint32_t kPreBlockTotal;
    uint32_t kPreBlockTail;
    uint32_t kPostBlockTotal;
    uint32_t vPreBlockFactor;
    uint32_t vPreBlockTotal;
    uint32_t vPreBlockTail;
    uint32_t vPostBlockTotal;

    uint64_t initdqSize;
    uint64_t dqOffset;
    uint64_t initdkSize;
    uint64_t dkOffset;
    uint64_t initdvSize;
    uint64_t dvOffset;

    bool isDropBoolMode;
    uint32_t maskUsedCoreNum;
    uint32_t maskUBProcessNum;
    uint32_t maskTailUBProcessNum;
    uint32_t maskUBLoop;

    DataCopyParams copyParams;
    DataCopyPadParams padParams;
    BinaryRepeatParams repParams;
    half padValue{1.0};
};

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dSinks,
    __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *workspace,
    const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase *orgTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    TilingData = orgTilingData;
    pipe = pipe_in;

    qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
    qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
    qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
    qPostBlockTotal = TilingData->postTilingData.qPostBlockTotal;
    kPreBlockFactor = TilingData->preTilingData.kPreBlockFactor;
    kPreBlockTotal = TilingData->preTilingData.kPreBlockTotal;
    kPreBlockTail = TilingData->preTilingData.kPreBlockTail;
    kPostBlockTotal = TilingData->postTilingData.kPostBlockTotal;
    vPreBlockFactor = TilingData->preTilingData.vPreBlockFactor;
    vPreBlockTotal = TilingData->preTilingData.vPreBlockTotal;
    vPreBlockTail = TilingData->preTilingData.vPreBlockTail;
    vPostBlockTotal = TilingData->postTilingData.vPostBlockTotal;

    dqGm.SetGlobalBuffer((__gm__ T1 *)dq);
    dkGm.SetGlobalBuffer((__gm__ T1 *)dk);
    if constexpr (!KV_MERGE) {
        dvGm.SetGlobalBuffer((__gm__ T1 *)dv);
    }

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));

    if constexpr (IS_SINKS) {
        dSinksGm.SetGlobalBuffer((__gm__ float *)dSinks);
        dSinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                         TilingData->postTilingData.dSinkWorkSpaceOffset / sizeof(T2));
    }

    initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
    dqOffset = ((uint64_t)cBlockIdx) * qPreBlockFactor;
    initdkSize = cBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
    dkOffset = ((uint64_t)cBlockIdx) * kPreBlockFactor;
    initdvSize = cBlockIdx == vPreBlockTotal - 1 ? vPreBlockTail : vPreBlockFactor;
    dvOffset = ((uint64_t)cBlockIdx) * vPreBlockFactor;
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::Process()
{
    // process
    if (g_coreType == AIV) {
        // clear dq dk dv workspace
        InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        InitOutput<float>(dkWorkSpaceGm[dkOffset], initdkSize, 0);
        if constexpr (!KV_MERGE) {
            InitOutput<float>(dvWorkSpaceGm[dvOffset], initdvSize, 0);
        }
        // 单核清零，避免冗余；dSinksGm 0 初值为非 deter AtomicAdd 必需
        if constexpr (IS_SINKS) {
            if (GetBlockIdx() == 0) {
                InitOutput<float>(dSinksGm[0], TilingData->baseParams.g, 0);
                // dSinkWorkSpace: reduce 需初值 0；非 deter 不分配(len=0)，不清零避免越界
                if constexpr (IS_DETER) {
                    int64_t dSinkWsLen = TilingData->baseParams.usedCoreNum * optiling::sfag::MAX_N1;
                    InitOutput<float>(dSinkWorkSpaceGm[0], dSinkWsLen, 0);
                }
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::SyncALLCores()
{
    SyncAll<false>();
}
#endif // _SPARSE_FLASH_ATTENTION_GRAD_PRE_KERNEL_REGBASE_H_
