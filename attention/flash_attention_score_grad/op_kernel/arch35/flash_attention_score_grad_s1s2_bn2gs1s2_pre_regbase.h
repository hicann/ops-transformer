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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_pre_regbase.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_REGBASE_H_
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

#define FAG_PRE_CLASS_TEMPLATE \
    template <typename T1, typename T2, const uint8_t DETER_SPARSE_TYPE = 0, const uint32_t IS_TND = 0, \
              const uint8_t SPLIT_AXIS = 0, const bool IS_TND_SWIZZLE = 0, const bool IS_ATTEN_MASK = 0>
#define FAG_PRE_FUNCTION_TEMPLATE \
    template <typename T1, typename T2, const uint8_t DETER_SPARSE_TYPE, const uint32_t IS_TND, \
              const uint8_t SPLIT_AXIS, const bool IS_TND_SWIZZLE, const bool IS_ATTEN_MASK>
#define FAG_PRE_FUNCTION_PARAMS_TEMPLATE T1, T2, DETER_SPARSE_TYPE, IS_TND, SPLIT_AXIS, IS_TND_SWIZZLE, IS_ATTEN_MASK

FAG_PRE_CLASS_TEMPLATE
class FlashAttentionScoreGradS1S2BNGS1S2PreRegbase {
public:
    __aicore__ inline FlashAttentionScoreGradS1S2BNGS1S2PreRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                                __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *drop_mask, __gm__ uint8_t *workspace,
                                FagTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dsinkWorkSpaceGm;
    GlobalTensor<T1> dqGm, dkGm, dvGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm;
    GlobalTensor<uint8_t> drop_maskGm;

    FagTilingType tilingData;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t S1S2_TND = 3;

    uint32_t cBlockIdx;
    // query
    uint64_t ubBaseSize;
    uint64_t qPreBlockFactor;
    uint64_t qPreBlockTotal;
    uint64_t qPreBlockTail;
    uint64_t qPostBlockTotal;
    uint64_t kPreBlockFactor;
    uint64_t kPreBlockTotal;
    uint64_t kPreBlockTail;
    uint64_t kPostBlockTotal;
    uint64_t vPreBlockFactor;
    uint64_t vPreBlockTotal;
    uint64_t vPreBlockTail;
    uint64_t vPostBlockTotal;
    uint32_t isSink = 0;
    uint64_t sinkPreBlockFactor = 0;
    uint64_t sinkPreBlockTotal = 0;
    uint64_t sinkPreBlockTail = 0;

    uint64_t initdqSize;
    uint64_t dqOffset;
    uint64_t initdkSize;
    uint64_t dkOffset;
    uint64_t initdvSize;
    uint64_t dvOffset;
    uint64_t initdsinkSize = 0;
    uint64_t dsinkOffset = 0;

    bool isDropBoolMode;
    uint64_t maskUsedCoreNum;
    uint64_t maskUBProcessNum;
    uint64_t maskTailUBProcessNum;
    uint64_t maskUBLoop;

    DataCopyParams copyParams;
    DataCopyPadParams padParams;
    BinaryRepeatParams repParams;
    half padValue{1.0};
};

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *actual_seq_kvlen,
    __gm__ uint8_t *drop_mask, __gm__ uint8_t *workspace, FagTilingType orgTilingData, TPipe *pipeIn)
{
    cBlockIdx = GetBlockIdx();

    tilingData = orgTilingData;
    pipe = pipeIn;

    // tiling_data
    qPreBlockFactor = tilingData->preTilingData.qPreBlockFactor;
    qPreBlockTotal = tilingData->preTilingData.qPreBlockTotal;
    qPreBlockTail = tilingData->preTilingData.qPreBlockTail;
    qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
    kPreBlockFactor = tilingData->preTilingData.kPreBlockFactor;
    kPreBlockTotal = tilingData->preTilingData.kPreBlockTotal;
    kPreBlockTail = tilingData->preTilingData.kPreBlockTail;
    kPostBlockTotal = tilingData->postTilingData.kPostBlockTotal;
    vPreBlockFactor = tilingData->preTilingData.vPreBlockFactor;
    vPreBlockTotal = tilingData->preTilingData.vPreBlockTotal;
    vPreBlockTail = tilingData->preTilingData.vPreBlockTail;
    vPostBlockTotal = tilingData->postTilingData.vPostBlockTotal;
    isSink = tilingData->s1s2BNGS1S2BaseParams.sinkOptional;
    if (unlikely(isSink)) {
        sinkPreBlockFactor = tilingData->preTilingData.sinkPreBlockFactor;
        sinkPreBlockTotal = tilingData->preTilingData.sinkPreBlockTotal;
        sinkPreBlockTail = tilingData->preTilingData.sinkPreBlockTail;
        dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                         tilingData->postTilingData.dsinkWorkSpaceOffset / sizeof(float));
        initdsinkSize = cBlockIdx == sinkPreBlockTotal - 1 ? sinkPreBlockTail : sinkPreBlockFactor;
        dsinkOffset = ((uint64_t)cBlockIdx) * sinkPreBlockFactor;
    }

    maskUsedCoreNum = tilingData->preTilingData.maskCoreNum;

    drop_maskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    dqGm.SetGlobalBuffer((__gm__ T1 *)dq);
    dkGm.SetGlobalBuffer((__gm__ T1 *)dk);
    dvGm.SetGlobalBuffer((__gm__ T1 *)dv);

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));

    initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
    dqOffset = ((uint64_t)cBlockIdx) * qPreBlockFactor;
    initdkSize = cBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
    dkOffset = ((uint64_t)cBlockIdx) * kPreBlockFactor;
    initdvSize = cBlockIdx == vPreBlockTotal - 1 ? vPreBlockTail : vPreBlockFactor;
    dvOffset = ((uint64_t)cBlockIdx) * vPreBlockFactor;

    // dropMask params init
    isDropBoolMode = tilingData->preTilingData.dropoutIsDivisibleBy8 == 0;
    if constexpr (IS_TND) {
        if (!isDropBoolMode && tilingData->s1s2BNGS1S2BaseParams.keepProb < 1) {
            isDropBoolMode = (((__gm__ int64_t *)actual_seq_kvlen)[0] % 8 != 0);
            for (uint32_t i = 0; i + 1 < tilingData->s1s2BNGS1S2BaseParams.b; i++) {
                const int64_t seqS2iplus1 =
                    ((__gm__ int64_t *)actual_seq_kvlen)[i + 1] - ((__gm__ int64_t *)actual_seq_kvlen)[i];
                isDropBoolMode = (isDropBoolMode || (seqS2iplus1 % 8 != 0));
            }
        }
    }
    if (tilingData->s1s2BNGS1S2BaseParams.dropMaskOuter && isDropBoolMode) {
        maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + tilingData->postTilingData.dropMaskGmOffset);

        pipe->InitBuffer(helpQue, 1, HELP_LEN);
        pipe->InitBuffer(inputQue, 1, tilingData->preTilingData.inputBufferLen);
        pipe->InitBuffer(castQue, 1, tilingData->preTilingData.castBufferLen);
        pipe->InitBuffer(outQue, 1, tilingData->preTilingData.outputBufferLen);

        // reset params
        repParams.src0BlkStride = 1;
        repParams.src0RepStride = 0;
        repParams.src1BlkStride = 0;
        repParams.src1RepStride = 0;
        repParams.dstBlkStride = 1;
        repParams.dstRepStride = NUMBER_8;

        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
    }
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    // process
    if (g_coreType == AIV && cBlockIdx < tilingData->preTilingData.maskCoreNum) {
        // clear dq dk dv workspace
        if constexpr (IsSameType<T1, float>::value) {
            InitOutput<T1>(dqGm[dqOffset], initdqSize, 0);
            InitOutput<T1>(dkGm[dkOffset], initdkSize, 0);
            InitOutput<T1>(dvGm[dvOffset], initdvSize, 0);
        } else if constexpr (SPLIT_AXIS != 1) {
            InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
            if constexpr (SPLIT_AXIS == 0) {
                InitOutput<float>(dkWorkSpaceGm[dkOffset], initdkSize, 0);
                InitOutput<float>(dvWorkSpaceGm[dvOffset], initdvSize, 0);
            } else if constexpr (SPLIT_AXIS == 5) {
                if (tilingData->preTilingData.sValueZeroUnderTND ||
                    (IS_DETER_NEW(DETER_SPARSE_TYPE) && tilingData->preTilingData.hasInvalidCol)) {
                    // BN2S2针对TND中有S为0的场景 或 newDeter与无效列叠加场景，增加gm清零
                    InitOutput<T1>(dkGm[dkOffset], initdkSize, 0);
                    InitOutput<T1>(dvGm[dvOffset], initdvSize, 0);
                }
            }
        }

        if (unlikely(isSink && (IS_ATTEN_MASK || IS_TND))) {
            InitOutput<float>(dsinkWorkSpaceGm[dsinkOffset], initdsinkSize, 0);
        }
        if (!(tilingData->s1s2BNGS1S2BaseParams.dropMaskOuter && isDropBoolMode)) {
            return;
        }

        maskUBLoop = tilingData->preTilingData.maskSingleCoreLoop;
        maskTailUBProcessNum = tilingData->preTilingData.maskLastLoopNum;
        if (unlikely(cBlockIdx == maskUsedCoreNum - 1)) {
            maskUBLoop = tilingData->preTilingData.maskTailCoreLoop;
            maskTailUBProcessNum = tilingData->preTilingData.maskTailCoreLastLoopNum;
        }

        // malloc tensor filled by 1.0
        auto helpTensor = helpQue.AllocTensor<half>();
        Duplicate<half>(helpTensor, padValue, HELP_LEN / sizeof(half));

        uint64_t outputAddr = cBlockIdx * tilingData->preTilingData.maskSingleCoreNum;
        uint64_t inputAddr = cBlockIdx * tilingData->preTilingData.maskSingleCoreNum / BIT8;

        // process
        for (uint64_t idx = 0; idx < maskUBLoop; idx++) {
            maskUBProcessNum = tilingData->preTilingData.singleUBProcessNum;
            uint64_t outputOffset = idx * maskUBProcessNum;
            uint64_t inputOffset = idx * maskUBProcessNum / BIT8;
            if (unlikely(idx == maskUBLoop - 1)) {
                maskUBProcessNum = maskTailUBProcessNum;
            }

            // copyIn
            auto inputTensor = inputQue.AllocTensor<uint8_t>();
            copyParams.blockLen = maskUBProcessNum / BIT8;
            DataCopyPad(inputTensor, drop_maskGm[inputAddr + inputOffset], copyParams, padParams);
            inputQue.EnQue(inputTensor);
            inputQue.DeQue<uint8_t>();

            // select
            auto castTensor = castQue.AllocTensor<half>();
            uint8_t selectRepeat = (maskUBProcessNum + B16_VECTOR_MASK - 1) / B16_VECTOR_MASK;
            Select(castTensor, inputTensor, helpTensor, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, B16_VECTOR_MASK,
                   selectRepeat, repParams);
            inputQue.FreeTensor(inputTensor);

            // cast
            auto outputTensor = outQue.AllocTensor<uint8_t>();
            Cast(outputTensor, castTensor, RoundMode::CAST_ROUND, maskUBProcessNum);
            castQue.FreeTensor(castTensor);

            // copyOut
            outQue.EnQue(outputTensor);
            outQue.DeQue<uint8_t>();
            copyParams.blockLen = maskUBProcessNum;
            DataCopyPad(maskWorkSpaceGm[outputAddr + outputOffset], outputTensor, copyParams);
            outQue.FreeTensor(outputTensor);
        }
        helpQue.FreeTensor(helpTensor);
    }
}

FAG_PRE_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<FAG_PRE_FUNCTION_PARAMS_TEMPLATE>::SyncALLCores()
{
    SyncAll<false>();
}
#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_REGBASE_H_
