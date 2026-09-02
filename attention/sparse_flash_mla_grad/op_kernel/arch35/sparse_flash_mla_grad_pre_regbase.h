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
 * \file sparse_flash_mla_grad_pre_regbase.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_GRAD_PRE_KERNEL_REGBASE_H_
#define SPARSE_FLASH_MLA_GRAD_PRE_KERNEL_REGBASE_H_
#include "kernel_operator.h"

using namespace AscendC;

template <typename T1, typename T2, const uint32_t IS_TND = 0, const bool isOriKVExist = 0, const bool isCmpKVExist = 0,
          const bool IsOriKVSparse = 0, const bool IsCmpKVSparse = 0>
class SparseFlashMlaGradPreRegbase {
public:
    __aicore__ inline SparseFlashMlaGradPreRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *workspace, __gm__ uint8_t *dsinks, __gm__ uint8_t *ori_softmax_l1,
                                __gm__ uint8_t *cmp_softmax_l1,
                                const optiling::smlag::SparseFlashMlaGradTilingDataRegbase *ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dorikvWorkSpaceGm, dcmpkvWorkSpaceGm, dsinkWorkSpaceGm;
    GlobalTensor<float> dsinkGm, orisoftmaxl1Gm, cmpsoftmaxl1Gm;

    const optiling::smlag::SparseFlashMlaGradTilingDataRegbase *TilingData;

    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t qPostBlockTotal;
    uint32_t orikvPreBlockFactor;
    uint32_t orikvPreBlockTotal;
    uint32_t orikvPreBlockTail;
    uint32_t orikvPostBlockTotal;
    uint32_t cmpkvPreBlockFactor;
    uint32_t cmpkvPreBlockTotal;
    uint32_t cmpkvPreBlockTail;
    uint32_t cmpkvPostBlockTotal;
    uint32_t orisoftmaxl1PreBlockFactor;
    uint32_t orisoftmaxl1PreBlockTotal;
    uint32_t orisoftmaxl1PreBlockTail;
    uint32_t cmpsoftmaxl1PreBlockFactor;
    uint32_t cmpsoftmaxl1PreBlockTotal;
    uint32_t cmpsoftmaxl1PreBlockTail;

    uint64_t initdqSize;
    uint64_t dqOffset;
    uint64_t initdorikvSize;
    uint64_t dorikvOffset;
    uint64_t initdcmpkvSize;
    uint64_t dcmpkvOffset;
    uint64_t dsinkOffset;
    uint64_t initdsinkSize;
    uint64_t orisoftmaxl1Offset;
    uint64_t initorisoftmaxl1Size;
    uint64_t cmpsoftmaxl1Offset;
    uint64_t initcmpsoftmaxl1Size;
};

template <typename T1, typename T2, const uint32_t IS_TND, const bool isOriKVExist, const bool isCmpKVExist,
          const bool IsOriKVSparse, const bool IsCmpKVSparse>
__aicore__ inline void
SparseFlashMlaGradPreRegbase<T1, T2, IS_TND, isOriKVExist, isCmpKVExist, IsOriKVSparse, IsCmpKVSparse>::Init(
    __gm__ uint8_t *workspace, __gm__ uint8_t *dsinks, __gm__ uint8_t *ori_softmax_l1, __gm__ uint8_t *cmp_softmax_l1,
    const optiling::smlag::SparseFlashMlaGradTilingDataRegbase *orgTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    TilingData = orgTilingData;
    pipe = pipe_in;

    // tiling_data
    qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
    qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
    qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
    qPostBlockTotal = TilingData->postTilingData.qPostBlockTotal;
    orikvPreBlockFactor = TilingData->preTilingData.oriKVPreBlockFactor;
    orikvPreBlockTotal = TilingData->preTilingData.oriKVPreBlockTotal;
    orikvPreBlockTail = TilingData->preTilingData.oriKVPreBlockTail;
    orikvPostBlockTotal = TilingData->postTilingData.oriKVPostBlockTotal;
    cmpkvPreBlockFactor = TilingData->preTilingData.cmpKVPreBlockFactor;
    cmpkvPreBlockTotal = TilingData->preTilingData.cmpKVPreBlockTotal;
    cmpkvPreBlockTail = TilingData->preTilingData.cmpKVPreBlockTail;
    cmpkvPostBlockTotal = TilingData->postTilingData.cmpKVPostBlockTotal;
    orisoftmaxl1PreBlockFactor = TilingData->preTilingData.oriSoftmaxL1PreBlockFactor;
    orisoftmaxl1PreBlockTotal = TilingData->preTilingData.oriSoftmaxL1PreBlockTotal;
    orisoftmaxl1PreBlockTail = TilingData->preTilingData.oriSoftmaxL1PreBlockTail;
    cmpsoftmaxl1PreBlockFactor = TilingData->preTilingData.cmpSoftmaxL1PreBlockFactor;
    cmpsoftmaxl1PreBlockTotal = TilingData->preTilingData.cmpSoftmaxL1PreBlockTotal;
    cmpsoftmaxl1PreBlockTail = TilingData->preTilingData.cmpSoftmaxL1PreBlockTail;

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    if constexpr (isOriKVExist) {
        dorikvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          TilingData->postTilingData.dOriKVWorkSpaceOffset / sizeof(T2));
    }
    if constexpr (isCmpKVExist) {
        dcmpkvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          TilingData->postTilingData.dCmpKVWorkSpaceOffset / sizeof(T2));
    }
    if (TilingData->baseParams.isSink) {
        dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                         TilingData->baseParams.dSinkWorkSpaceOffset / sizeof(T2));
        dsinkGm.SetGlobalBuffer((__gm__ float *)dsinks);
    }
    if constexpr (IsOriKVSparse) {
        orisoftmaxl1Gm.SetGlobalBuffer((__gm__ float *)ori_softmax_l1);
    }
    if constexpr (IsCmpKVSparse) {
        cmpsoftmaxl1Gm.SetGlobalBuffer((__gm__ float *)cmp_softmax_l1);
    }

    initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
    dqOffset = ((uint64_t)cBlockIdx) * qPreBlockFactor;
    initdorikvSize = cBlockIdx == orikvPreBlockTotal - 1 ? orikvPreBlockTail : orikvPreBlockFactor;
    dorikvOffset = ((uint64_t)cBlockIdx) * orikvPreBlockFactor;
    initdcmpkvSize = cBlockIdx == cmpkvPreBlockTotal - 1 ? cmpkvPreBlockTail : cmpkvPreBlockFactor;
    dcmpkvOffset = ((uint64_t)cBlockIdx) * cmpkvPreBlockFactor;
    dsinkOffset = ((uint64_t)cBlockIdx) * TilingData->baseParams.g;
    initdsinkSize = TilingData->baseParams.g;
    initorisoftmaxl1Size =
        cBlockIdx == orisoftmaxl1PreBlockTotal - 1 ? orisoftmaxl1PreBlockTail : orisoftmaxl1PreBlockFactor;
    orisoftmaxl1Offset = ((uint64_t)cBlockIdx) * orisoftmaxl1PreBlockFactor;
    initcmpsoftmaxl1Size =
        cBlockIdx == cmpsoftmaxl1PreBlockTotal - 1 ? cmpsoftmaxl1PreBlockTail : cmpsoftmaxl1PreBlockFactor;
    cmpsoftmaxl1Offset = ((uint64_t)cBlockIdx) * cmpsoftmaxl1PreBlockFactor;
}

template <typename T1, typename T2, const uint32_t IS_TND, const bool isOriKVExist, const bool isCmpKVExist,
          const bool IsOriKVSparse, const bool IsCmpKVSparse>
__aicore__ inline void
SparseFlashMlaGradPreRegbase<T1, T2, IS_TND, isOriKVExist, isCmpKVExist, IsOriKVSparse, IsCmpKVSparse>::Process()
{
    // process
    if (g_coreType == AIV) {
        // clear dq dk dv workspace
        InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        if constexpr (isOriKVExist) {
            InitOutput<float>(dorikvWorkSpaceGm[dorikvOffset], initdorikvSize, 0);
        }
        if constexpr (isCmpKVExist) {
            InitOutput<float>(dcmpkvWorkSpaceGm[dcmpkvOffset], initdcmpkvSize, 0);
        }
        if (TilingData->baseParams.isSink) {
            InitOutput<float>(dsinkWorkSpaceGm[dsinkOffset], initdsinkSize, 0);
            InitOutput<float>(dsinkGm, initdsinkSize, 0);
        }
        if constexpr (IsOriKVSparse) {
            InitOutput<float>(orisoftmaxl1Gm[orisoftmaxl1Offset], initorisoftmaxl1Size, 0);
        }
        if constexpr (IsCmpKVSparse) {
            InitOutput<float>(cmpsoftmaxl1Gm[cmpsoftmaxl1Offset], initcmpsoftmaxl1Size, 0);
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_TND, const bool isOriKVExist, const bool isCmpKVExist,
          const bool IsOriKVSparse, const bool IsCmpKVSparse>
__aicore__ inline void
SparseFlashMlaGradPreRegbase<T1, T2, IS_TND, isOriKVExist, isCmpKVExist, IsOriKVSparse, IsCmpKVSparse>::SyncALLCores()
{
    SyncAll<false>();
}
#endif // _SPARSE_FLASH_MLA_GRAD_PRE_KERNEL_REGBASE_H_
