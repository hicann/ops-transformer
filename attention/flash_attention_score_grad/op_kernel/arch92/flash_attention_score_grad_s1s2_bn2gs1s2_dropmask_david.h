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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_dropmask_david.h
 * \brief
 */
#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_DROPMASK_KERNEL_DAVID_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_DROPMASK_KERNEL_DAVID_H_
#include "kernel_operator.h"

using namespace AscendC;

#define FAG_DROP_CLASS_TEMPLATE \
    template <typename T1, typename T2, const uint32_t IS_TND = 0, const uint8_t SPLIT_AXIS = 0, \
              const uint8_t DETER_SPARSE_TYPE = 0>

#define FAG_DROP_FUNCTION_TEMPLATE \
    template <typename T1, typename T2, const uint32_t IS_TND, const uint8_t SPLIT_AXIS, \
              const uint8_t DETER_SPARSE_TYPE>

#define FAG_DROP_FUNCTION_PARAMS_TEMPLATE T1, T2, IS_TND, SPLIT_AXIS, DETER_SPARSE_TYPE

FAG_DROP_CLASS_TEMPLATE
class FlashAttentionScoreGradS1S2BNGS1S2DropmaskDavid {
public:
    __aicore__ inline FlashAttentionScoreGradS1S2BNGS1S2DropmaskDavid(){};
    __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv,
                                __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen,
                                __gm__ uint8_t* drop_mask, __gm__ uint8_t* workspace, FagTilingType ordTilingData,
                                TPipe* pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    GlobalTensor<T1> dqGm, dkGm, dvGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm;
    GlobalTensor<uint8_t> drop_maskGm;

    FagTilingType TilingData;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t S1S2_TND = 3;

    uint32_t cBlockIdx;

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

FAG_DROP_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2DropmaskDavid<FAG_DROP_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* actual_seq_qlen,
    __gm__ uint8_t* actual_seq_kvlen, __gm__ uint8_t* drop_mask, __gm__ uint8_t* workspace, FagTilingType orgTilingData,
    TPipe* pipe_in)
{
    cBlockIdx = GetBlockIdx();

    TilingData = orgTilingData;
    pipe = pipe_in;

    // tiling_data
    maskUsedCoreNum = TilingData->preTilingData.maskCoreNum;

    drop_maskGm.SetGlobalBuffer((__gm__ uint8_t*)drop_mask);

    // dropMask params init
    isDropBoolMode = TilingData->preTilingData.dropoutIsDivisibleBy8 == 0;
    if constexpr (IS_TND) {
        if (!isDropBoolMode && TilingData->s1s2BNGS1S2BaseParams.keepProb < 1) {
            isDropBoolMode = (((__gm__ int64_t*)actual_seq_kvlen)[0] % 8 != 0);
            for (uint32_t i = 0; i + 1 < TilingData->s1s2BNGS1S2BaseParams.b; i++) {
                const int64_t seqS2iplus1 =
                    ((__gm__ int64_t*)actual_seq_kvlen)[i + 1] - ((__gm__ int64_t*)actual_seq_kvlen)[i];
                isDropBoolMode = (isDropBoolMode || (seqS2iplus1 % 8 != 0));
            }
        }
    }
    if (TilingData->s1s2BNGS1S2BaseParams.dropMaskOuter && isDropBoolMode) {
        maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace + TilingData->postTilingData.dropMaskGmOffset);

        pipe->InitBuffer(helpQue, 1, HELP_LEN);
        pipe->InitBuffer(inputQue, 1, TilingData->preTilingData.inputBufferLen);
        pipe->InitBuffer(castQue, 1, TilingData->preTilingData.castBufferLen);
        pipe->InitBuffer(outQue, 1, TilingData->preTilingData.outputBufferLen);

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

FAG_DROP_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2DropmaskDavid<FAG_DROP_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    // process
    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.maskCoreNum) {
        if (!(TilingData->s1s2BNGS1S2BaseParams.dropMaskOuter && isDropBoolMode)) {
            return;
        }

        maskUBLoop = TilingData->preTilingData.maskSingleCoreLoop;
        maskTailUBProcessNum = TilingData->preTilingData.maskLastLoopNum;
        if (unlikely(cBlockIdx == maskUsedCoreNum - 1)) {
            maskUBLoop = TilingData->preTilingData.maskTailCoreLoop;
            maskTailUBProcessNum = TilingData->preTilingData.maskTailCoreLastLoopNum;
        }

        // malloc tensor filled by 1.0
        auto helpTensor = helpQue.AllocTensor<half>();
        Duplicate<half>(helpTensor, padValue, HELP_LEN / sizeof(half));

        uint64_t outputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum;
        uint64_t inputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum / BIT8;

        // process
        for (uint64_t idx = 0; idx < maskUBLoop; idx++) {
            maskUBProcessNum = TilingData->preTilingData.singleUBProcessNum;
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

FAG_DROP_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradS1S2BNGS1S2DropmaskDavid<FAG_DROP_FUNCTION_PARAMS_TEMPLATE>::SyncALLCores()
{
    SyncAll();
}
#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_DROPMASK_KERNEL_DAVID_H_
