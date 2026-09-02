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
 * \file kv_rms_norm_rope_cache_b16_b1sd.h
 * \brief
 */
#ifndef _KV_RMS_NORM_ROPE_CACHE_B16_B1SD_H_
#define _KV_RMS_NORM_ROPE_CACHE_B16_B1SD_H_
#include "kv_rms_norm_rope_cache_comm.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;

template <bool isPagedAttention, typename KV_DTYPE>
class KernelKvRmsNormRopeCacheB16B1SD
    : public KernelKvRmsNormRopeCacheCutBS<isPagedAttention, KV_DTYPE, KV_DTYPE, KV_DTYPE> {
public:
    __aicore__ inline KernelKvRmsNormRopeCacheB16B1SD(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheCutBS<isPagedAttention, KV_DTYPE, KV_DTYPE, KV_DTYPE>(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache,
                                GM_ADDR v_cache, GM_ADDR optional_v)
    {
        methodMode = optional_v == nullptr ? 0 : 1;
        this->InitSharedData(methodMode);
        int64_t currentBlockFactor = this->tilingData_->blockFactor;
        if (GetBlockIdx() == (this->tilingData_->numBlocks - 1)) {
            currentBlockFactor =
                this->tilingData_->batchSize * this->tilingData_->seqLength * this->tilingData_->numHead -
                (this->tilingData_->numBlocks - 1) * this->tilingData_->blockFactor;
        }

        this->ubFactor = this->tilingData_->ubFactor;
        this->ubLoop = currentBlockFactor / this->ubFactor;
        this->ubTail = currentBlockFactor - this->ubLoop * this->ubFactor;

        // init global memory
        if (methodMode == METHOD_V1) {
            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE *)kv + GetBlockIdx() * this->tilingData_->blockFactor *
                                                                   (this->RMS_NORM_LENGTH + this->ROPE_LENGTH));
        } else {
            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE *)kv +
                                       GetBlockIdx() * this->tilingData_->blockFactor * this->RMS_NORM_LENGTH);
            this->vGm.SetGlobalBuffer((__gm__ KV_DTYPE *)optional_v +
                                      GetBlockIdx() * this->tilingData_->blockFactor * this->V_LENGTH);
        }
        this->gammaGm.SetGlobalBuffer((__gm__ KV_DTYPE *)gamma);
        this->cosGm.SetGlobalBuffer((__gm__ KV_DTYPE *)cos +
                                    GetBlockIdx() * this->tilingData_->blockFactor * this->ROPE_LENGTH);
        this->sinGm.SetGlobalBuffer((__gm__ KV_DTYPE *)sin +
                                    GetBlockIdx() * this->tilingData_->blockFactor * this->ROPE_LENGTH);
        if (this->tilingData_->numHead == METHOD_V2) {
            this->indexGm.SetGlobalBuffer((__gm__ int64_t *)index + GetBlockIdx() * this->tilingData_->blockFactor);
        } else {
            this->indexGm.SetGlobalBuffer((__gm__ int64_t *)index);
        }
        this->kCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE *)k_cache);
        this->vCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE *)v_cache);

        // init pipe
        this->pipe_->InitBuffer(this->inQueueGamma, 2, this->RMS_NORM_LENGTH * sizeof(float)); // 2*512*4/1024=4
        this->pipe_->InitBuffer(this->inQueueX, 2,
                                this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)); // 2*16*512*2/1024=32
        this->pipe_->InitBuffer(this->outQueue, 2,
                                this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)); // 2*16*512*2/1024=32
        this->pipe_->InitBuffer(this->wsBuffer,
                                3 * this->ubFactor * this->RMS_NORM_LENGTH * sizeof(float)); // 3*16*512*4/1024=96
        this->pipe_->InitBuffer(inQueueIndex, 1, this->ubFactor * sizeof(int64_t));          // 16*8/1024=0.125
        if (methodMode == METHOD_V2) {
            this->pipe_->InitBuffer(this->inQueueV, 2,
                                    this->ubFactor * this->V_LENGTH * sizeof(KV_DTYPE)); // 2*40*128*2/1024=20
        }
    }

    __aicore__ inline void Process()
    {
        if (methodMode == METHOD_V1) {
            ProcessV1();
        } else {
            ProcessV2();
        }
    }

    __aicore__ inline void ProcessV1()
    {
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyPadExtParams<int64_t> padParamsInt64{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockCount = 1;
        // CopyIn gamma: [this->RMS_NORM_LENGTH]
        LocalTensor<float> gammaLocalFp32 = this->inQueueGamma.template AllocTensor<float>();
        LocalTensor<KV_DTYPE> gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
        copyParamsContinguous.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
        DataCopyPad(gammaLocal, this->gammaGm, copyParamsContinguous, padParams);
        this->inQueueGamma.EnQue(gammaLocalFp32);
        gammaLocalFp32 = this->inQueueGamma.template DeQue<float>();
        gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
        Cast(gammaLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->RMS_NORM_LENGTH);

        LocalTensor<float> workspaceBuffer = this->wsBuffer.template Get<float>();
        for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
            int64_t kvGlobalMemoryOffset = loopIdx * this->ubFactor * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH);
            int64_t freqGlobalMemoryOffset = loopIdx * this->ubFactor * this->ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = loopIdx * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + loopIdx * this->ubFactor;

            LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubFactor * this->ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];
            // CopyIn x/cos/sin [this->ubFactor, this->ROPE_LENGTH]
            DataCopyExtParams copyParams{
                /* blockCount */ static_cast<uint16_t>(this->ubFactor),
                /* blockLen (Byte) */ static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                /* srcStride */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                /* dstStride */ 0,
                /* rsv */ 0};
            DataCopyPad(ropeLocal, this->kvGm[kvGlobalMemoryOffset + this->RMS_NORM_LENGTH], copyParams, padParams);
            copyParamsContinguous.blockLen = this->ubFactor * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueX.EnQue(ropeLocal);
            ropeLocal = this->inQueueX.template DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[this->ubFactor * this->ROPE_LENGTH];
            sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];

            // CopyIn index [this->ubFactor,]
            LocalTensor<int64_t> indexLocal = inQueueIndex.template AllocTensor<int64_t>();
            copyParamsContinguous.blockLen = this->ubFactor * sizeof(int64_t);
            DataCopyPad(indexLocal, this->indexGm[indexGlobalMemoryOffset], copyParamsContinguous, padParamsInt64);
            inQueueIndex.EnQue(indexLocal);
            indexLocal = inQueueIndex.template DeQue<int64_t>();

            // Calc: RoPE
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RoPE<KV_DTYPE, true>(outLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer, this->ubFactor,
                                 this->ROPE_LENGTH);
            this->inQueueX.FreeTensor(ropeLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache
            ScatterUpdate<KV_DTYPE, isPagedAttention>(this->kCacheGm, outLocal, indexLocal, startIdx, this->ubFactor,
                                                      this->ROPE_LENGTH);
            this->outQueue.FreeTensor(outLocal);

            // CopyIn x: [this->ubFactor, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            copyParams.blockCount = static_cast<uint16_t>(this->ubFactor);
            copyParams.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
            copyParams.srcStride = this->ROPE_LENGTH * sizeof(KV_DTYPE);
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RmsNorm<KV_DTYPE>(outLocal, xLocal, gammaLocalFp32, workspaceBuffer, this->ubFactor, this->RMS_NORM_LENGTH);
            this->inQueueX.FreeTensor(xLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update vCache
            ScatterUpdate<KV_DTYPE, isPagedAttention>(this->vCacheGm, outLocal, indexLocal, startIdx, this->ubFactor,
                                                      this->RMS_NORM_LENGTH);
            this->outQueue.FreeTensor(outLocal);
            inQueueIndex.FreeTensor(indexLocal);
        }
        if (this->ubTail > 0) {
            int64_t kvGlobalMemoryOffset = this->ubLoop * this->ubFactor * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH);
            int64_t freqGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = this->ubLoop * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + this->ubLoop * this->ubFactor;

            LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];
            // CopyIn x/cos/sin [this->ubTail, this->ROPE_LENGTH]
            DataCopyExtParams copyParams{
                /* blockCount */ static_cast<uint16_t>(this->ubTail),
                /* blockLen (Byte) */ static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                /* srcStride */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                /* dstStride */ 0,
                /* rsv */ 0};
            DataCopyPad(ropeLocal, this->kvGm[kvGlobalMemoryOffset + this->RMS_NORM_LENGTH], copyParams, padParams);
            copyParamsContinguous.blockLen = this->ubTail * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueX.EnQue(ropeLocal);
            ropeLocal = this->inQueueX.template DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
            sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];

            // CopyIn index [this->ubTail,]
            LocalTensor<int64_t> indexLocal = inQueueIndex.template AllocTensor<int64_t>();
            copyParamsContinguous.blockLen = this->ubTail * sizeof(int64_t);
            DataCopyPad(indexLocal, this->indexGm[indexGlobalMemoryOffset], copyParamsContinguous, padParamsInt64);
            inQueueIndex.EnQue(indexLocal);
            indexLocal = inQueueIndex.template DeQue<int64_t>();

            // Calc: RoPE
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RoPE<KV_DTYPE, true>(outLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer, this->ubTail,
                                 this->ROPE_LENGTH);
            this->inQueueX.FreeTensor(ropeLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache
            ScatterUpdate<KV_DTYPE, isPagedAttention>(this->kCacheGm, outLocal, indexLocal, startIdx, this->ubTail,
                                                      this->ROPE_LENGTH);
            this->outQueue.FreeTensor(outLocal);

            // CopyIn x: [this->ubTail, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            copyParams.blockCount = static_cast<uint16_t>(this->ubTail);
            copyParams.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
            copyParams.srcStride = this->ROPE_LENGTH * sizeof(KV_DTYPE);
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RmsNorm<KV_DTYPE>(outLocal, xLocal, gammaLocalFp32, workspaceBuffer, this->ubTail, this->RMS_NORM_LENGTH);
            this->inQueueX.FreeTensor(xLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update vCache
            ScatterUpdate<KV_DTYPE, isPagedAttention>(this->vCacheGm, outLocal, indexLocal, startIdx, this->ubTail,
                                                      this->RMS_NORM_LENGTH);
            this->outQueue.FreeTensor(outLocal);
            inQueueIndex.FreeTensor(indexLocal);
        }
        this->inQueueGamma.FreeTensor(gammaLocalFp32);
    }

    __aicore__ inline void ProcessV2()
    {
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyPadExtParams<int64_t> padParamsInt64{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockCount = 1;
        // CopyIn gamma: [this->RMS_NORM_LENGTH]
        LocalTensor<float> gammaLocalFp32 = this->inQueueGamma.template AllocTensor<float>();
        LocalTensor<KV_DTYPE> gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
        copyParamsContinguous.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
        DataCopyPad(gammaLocal, this->gammaGm, copyParamsContinguous, padParams);
        this->inQueueGamma.EnQue(gammaLocalFp32);
        gammaLocalFp32 = this->inQueueGamma.template DeQue<float>();
        gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
        Cast(gammaLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->RMS_NORM_LENGTH);

        LocalTensor<float> workspaceBuffer = this->wsBuffer.template Get<float>();
        for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
            int64_t kvGlobalMemoryOffset = loopIdx * this->ubFactor * this->RMS_NORM_LENGTH;
            int64_t vGlobalMemoryOffset = loopIdx * this->ubFactor * this->V_LENGTH;
            int64_t freqGlobalMemoryOffset = loopIdx * this->ubFactor * this->ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = loopIdx * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + loopIdx * this->ubFactor;

            // CopyIn index [this->ubFactor,]
            LocalTensor<int64_t> indexLocal = inQueueIndex.template AllocTensor<int64_t>();
            copyParamsContinguous.blockLen = this->ubFactor * sizeof(int64_t);
            DataCopyPad(indexLocal, this->indexGm[indexGlobalMemoryOffset], copyParamsContinguous, padParamsInt64);
            inQueueIndex.EnQue(indexLocal);
            indexLocal = inQueueIndex.template DeQue<int64_t>();

            // CopyIn x [this->ubFactor, this->RMS_NORM_LENGTH]
            LocalTensor<KV_DTYPE> rmsNormLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            DataCopyExtParams copyParams{
                /* blockCount */ static_cast<uint16_t>(this->ubFactor),
                /* blockLen (Byte) */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                /* srcStride */ 0,
                /* dstStride */ 0,
                /* rsv */ 0};
            DataCopyPad(rmsNormLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(rmsNormLocal);
            rmsNormLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // CopyIn sin/cos: [this->ubFactor, this->ROPE_LENGTH]
            LocalTensor<KV_DTYPE> cosLocal = this->inQueueV.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];
            copyParamsContinguous.blockLen = this->ubFactor * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueV.EnQue(cosLocal);
            cosLocal = this->inQueueV.template DeQue<KV_DTYPE>();
            sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];

            // Calc: RmsNorm and RoPE
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RmsNormAndRoPE<KV_DTYPE, true>(outLocal, rmsNormLocal, gammaLocalFp32, cosLocal, sinLocal, workspaceBuffer,
                                           this->ubFactor, this->RMS_NORM_LENGTH, this->ROPE_LENGTH);
            this->inQueueV.FreeTensor(cosLocal);
            this->inQueueX.FreeTensor(rmsNormLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache
            if (this->tilingData_->numHead == METHOD_V2) {
                ScatterUpdate<KV_DTYPE, isPagedAttention>(this->kCacheGm, outLocal, indexLocal, startIdx,
                                                          this->ubFactor, this->RMS_NORM_LENGTH);
            } else {
                ScatterUpdateV2<KV_DTYPE, isPagedAttention>(this->kCacheGm, outLocal, startIdx, this->ubFactor,
                                                            this->RMS_NORM_LENGTH);
            }
            this->outQueue.FreeTensor(outLocal);

            // CopyIn v
            LocalTensor<KV_DTYPE> vLocal = this->inQueueV.template AllocTensor<KV_DTYPE>();
            copyParams.blockCount = static_cast<uint16_t>(this->ubFactor);
            copyParams.blockLen = static_cast<uint32_t>(this->V_LENGTH * sizeof(KV_DTYPE));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(vLocal, this->vGm[vGlobalMemoryOffset], copyParams, padParams);
            SetWaitFlag<HardEvent::V_V>(HardEvent::V_V);
            this->inQueueV.EnQue(vLocal);
            vLocal = this->inQueueV.template DeQue<KV_DTYPE>();

            // Scatter Update vCache
            if (this->tilingData_->numHead == METHOD_V2) {
                ScatterUpdate<KV_DTYPE, isPagedAttention>(this->vCacheGm, vLocal, indexLocal, startIdx, this->ubFactor,
                                                          this->V_LENGTH);
            } else {
                ScatterUpdateV2<KV_DTYPE, isPagedAttention>(this->vCacheGm, vLocal, startIdx, this->ubFactor,
                                                            this->V_LENGTH);
            }
            this->inQueueV.FreeTensor(vLocal);
            inQueueIndex.FreeTensor(indexLocal);
        }
        if (this->ubTail > 0) {
            int64_t kvGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->RMS_NORM_LENGTH;
            int64_t vGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->V_LENGTH;
            int64_t freqGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = this->ubLoop * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + this->ubLoop * this->ubFactor;

            // CopyIn index [this->ubTail,]
            LocalTensor<int64_t> indexLocal = inQueueIndex.template AllocTensor<int64_t>();
            copyParamsContinguous.blockLen = this->ubTail * sizeof(int64_t);
            DataCopyPad(indexLocal, this->indexGm[indexGlobalMemoryOffset], copyParamsContinguous, padParamsInt64);
            inQueueIndex.EnQue(indexLocal);
            indexLocal = inQueueIndex.template DeQue<int64_t>();

            // CopyIn x [this->ubTail, this->RMS_NORM_LENGTH]
            LocalTensor<KV_DTYPE> rmsNormLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            DataCopyExtParams copyParams{
                /* blockCount */ static_cast<uint16_t>(this->ubTail),
                /* blockLen (Byte) */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                /* srcStride */ 0,
                /* dstStride */ 0,
                /* rsv */ 0};
            DataCopyPad(rmsNormLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(rmsNormLocal);
            rmsNormLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // CopyIn sin/cos: [this->ubTail, this->ROPE_LENGTH]
            LocalTensor<KV_DTYPE> cosLocal = this->inQueueV.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];
            copyParamsContinguous.blockLen = this->ubTail * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueV.EnQue(cosLocal);
            cosLocal = this->inQueueV.template DeQue<KV_DTYPE>();
            sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];

            // Calc: RmsNorm and RoPE
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RmsNormAndRoPE<KV_DTYPE, true>(outLocal, rmsNormLocal, gammaLocalFp32, cosLocal, sinLocal, workspaceBuffer,
                                           this->ubTail, this->RMS_NORM_LENGTH, this->ROPE_LENGTH);
            this->inQueueV.FreeTensor(cosLocal);
            this->inQueueX.FreeTensor(rmsNormLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache
            if (this->tilingData_->numHead == METHOD_V2) {
                ScatterUpdate<KV_DTYPE, isPagedAttention>(this->kCacheGm, outLocal, indexLocal, startIdx, this->ubTail,
                                                          this->RMS_NORM_LENGTH);
            } else {
                ScatterUpdateV2<KV_DTYPE, isPagedAttention>(this->kCacheGm, outLocal, startIdx, this->ubTail,
                                                            this->RMS_NORM_LENGTH);
            }
            this->outQueue.FreeTensor(outLocal);

            // CopyIn v
            LocalTensor<KV_DTYPE> vLocal = this->inQueueV.template AllocTensor<KV_DTYPE>();
            copyParams.blockCount = static_cast<uint16_t>(this->ubTail);
            copyParams.blockLen = static_cast<uint32_t>(this->V_LENGTH * sizeof(KV_DTYPE));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(vLocal, this->vGm[vGlobalMemoryOffset], copyParams, padParams);
            SetWaitFlag<HardEvent::V_V>(HardEvent::V_V);
            this->inQueueV.EnQue(vLocal);
            vLocal = this->inQueueV.template DeQue<KV_DTYPE>();

            // Scatter Update vCache
            if (this->tilingData_->numHead == METHOD_V2) {
                ScatterUpdate<KV_DTYPE, isPagedAttention>(this->vCacheGm, vLocal, indexLocal, startIdx, this->ubTail,
                                                          this->V_LENGTH);
            } else {
                ScatterUpdateV2<KV_DTYPE, isPagedAttention>(this->vCacheGm, vLocal, startIdx, this->ubTail,
                                                            this->V_LENGTH);
            }
            this->inQueueV.FreeTensor(vLocal);
            inQueueIndex.FreeTensor(indexLocal);
        }
        this->inQueueGamma.FreeTensor(gammaLocalFp32);
    }

    template <typename T, bool isElementWise = true>
    __aicore__ inline void RmsNormAndRoPE(const LocalTensor<T> &outLocal, const LocalTensor<T> &xLocal,
                                          const LocalTensor<float> &gammaLocal, const LocalTensor<T> &cosLocal,
                                          const LocalTensor<T> &sinLocal, const LocalTensor<float> &wsLocal,
                                          int64_t rows, int64_t headSize, int64_t ropeLen)
    {
        constexpr static int64_t NUM_ONE = 1;
        constexpr static int64_t NUM_TWO = 2;
        constexpr static int64_t NUM_THREE = 3;
        constexpr static int64_t NUM_FOUR = 4;
        constexpr static int64_t NUM_SIX = 6;
        constexpr static int64_t NUM_EIGHT = 8;
        constexpr static int64_t NUM_TWELVE = 12;
        constexpr static int64_t NUM_TWENTY_FOUR = 24;
        constexpr static int64_t NUM_FOURTY_EIGHT = 48;
        constexpr static int64_t NUM_SIXTY_FOUR = 64;
        constexpr static int64_t NUM_ONE_HUNDRED_TWENTY_EIGHT = 128;
        constexpr static int64_t BLOCK_SIZE = 32;

        int64_t xLocalFp32Offset = 0;                  // [0, rows * headSize]
        int64_t xSquareLocalOffset = rows * headSize;  // [rows * headSize, rows * headSize * 2]
        int64_t xSumLocalOffset = rows * headSize * 2; // [rows * headSize * 2, rows * headSize * 3]
        LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
        LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
        LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];

        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, rows * headSize);
        PipeBarrier<PIPE_V>();
        Mul(xSquareLocal, xLocalFp32, xLocalFp32, rows * headSize);
        PipeBarrier<PIPE_V>();
        int64_t repeatTimes = rows * (headSize >> 1) / NUM_FOURTY_EIGHT;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_FOURTY_EIGHT], NUM_FOURTY_EIGHT, repeatTimes,
            {1, 1, 1, NUM_SIX, NUM_TWELVE, NUM_TWELVE}); // 96
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_FOURTY_EIGHT], NUM_FOURTY_EIGHT, repeatTimes,
            {1, 1, 1, NUM_SIX, NUM_TWELVE, NUM_TWELVE}); // 48
        PipeBarrier<PIPE_V>();

        WholeReduceSum(xSumLocal, xSquareLocal, NUM_FOURTY_EIGHT, rows, 1, 1, NUM_SIX);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, this->tilingData_->reciprocal, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, this->tilingData_->epsilon, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, rows);
        PipeBarrier<PIPE_V>();

        // Brcb
        int64_t block_count = (rows + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 / xSquareLocal
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Div(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], xSquareLocal[rowId * NUM_EIGHT],
                NUM_SIXTY_FOUR, (headSize / NUM_SIXTY_FOUR), {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 * xSquareLocal [rows, RMS_NORM_LENGTH] * [RMS_NORM_LENGTH]
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Mul(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], gammaLocal, headSize);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (isElementWise) {
            /**
             * cosLocalFp32 : [rows * ropeLen * 0, rows * ropeLen * 1]
             * sinLocalFp32 : [rows * ropeLen * 1, rows * ropeLen * 2]
             * y0           : [rows * ropeLen * 2, rows * ropeLen * 3]
             * y1           : [rows * ropeLen * 3, rows * ropeLen * 4]
             * realLocalFp32: [rows * ropeLen * 4, rows * ropeLen * 5]
             * imagLocalFp32: [rows * ropeLen * 5, rows * ropeLen * 6]
             */

            xLocalFp32Offset = rows * headSize;
            int64_t cosLocalFp32Offset = xLocalFp32Offset + rows * ropeLen * 0;
            int64_t sinLocalFp32Offset = xLocalFp32Offset + rows * ropeLen * 1;
            int64_t y0Offset = xLocalFp32Offset + rows * ropeLen * 2;
            int64_t y1Offset = xLocalFp32Offset + rows * ropeLen * 3;
            int64_t realLocalFp32Offset = xLocalFp32Offset + rows * ropeLen * 4;
            int64_t imagLocalFp32Offset = xLocalFp32Offset + rows * ropeLen * 5;

            LocalTensor<float> cosLocalFp32 = wsLocal[cosLocalFp32Offset];
            LocalTensor<float> sinLocalFp32 = wsLocal[sinLocalFp32Offset];
            LocalTensor<float> y0 = wsLocal[y0Offset];
            LocalTensor<float> y1 = wsLocal[y1Offset];
            LocalTensor<float> realLocalFp32 = wsLocal[realLocalFp32Offset];
            LocalTensor<float> imagLocalFp32 = wsLocal[imagLocalFp32Offset];

            // step #1: cast cosLocal and sinLocal to fp32
            Cast(cosLocalFp32, cosLocal, RoundMode::CAST_NONE, rows * ropeLen);
            Cast(sinLocalFp32, sinLocal, RoundMode::CAST_NONE, rows * ropeLen);
            PipeBarrier<PIPE_V>();

            // step #2: Gather out real and imag
            uint64_t rsvdCnt = 0;
            GatherMask(realLocalFp32, xLocalFp32, NUM_ONE, true, ropeLen,
                       {1, static_cast<uint16_t>(rows), NUM_TWENTY_FOUR, 0}, rsvdCnt);
            GatherMask(imagLocalFp32, xLocalFp32, NUM_TWO, true, ropeLen,
                       {1, static_cast<uint16_t>(rows), NUM_TWENTY_FOUR, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();

            // step #4: y0 = (realLocalFp32, imagLocalFp32) * cosLocalFp32
            Mul(y0, realLocalFp32, cosLocalFp32, /*mask*/ (ropeLen >> 1), /*repeat*/ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y0[(ropeLen >> 1)], imagLocalFp32, cosLocalFp32[(ropeLen >> 1)], /*mask*/ (ropeLen >> 1),
                /*repeat*/ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #5: y1 = (-imagLocalFp32, realLocalFp32) * sinLocalFp32
            Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, rows * (ropeLen >> 1));
            PipeBarrier<PIPE_V>();
            Mul(y1, imagLocalFp32, sinLocalFp32, /*mask*/ (ropeLen >> 1), /*repeat*/ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y1[(ropeLen >> 1)], realLocalFp32, sinLocalFp32[(ropeLen >> 1)], /*mask*/ (ropeLen >> 1),
                /*repeat*/ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #6: y0 = y0 + y1
            Add(y0, y0, y1, rows * ropeLen);
            PipeBarrier<PIPE_V>();

            struct DataCopyParams copyInParams(rows, 0, 0, 0);
            copyInParams.blockLen = ropeLen * sizeof(float) / BLOCK_SIZE;
            copyInParams.dstStride = static_cast<uint16_t>(NUM_ONE_HUNDRED_TWENTY_EIGHT * sizeof(float) / BLOCK_SIZE);
            DataCopy(xLocalFp32, y0, copyInParams);
            SetWaitFlag<HardEvent::V_V>(HardEvent::V_V);
        }

        // Cast xLocalFp32 to outLocal
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(outLocal, xLocalFp32, RoundMode::CAST_RINT, rows * headSize);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(outLocal, xLocalFp32, RoundMode::CAST_NONE, rows * headSize);
        }
    }

    template <typename T, bool isPA = false>
    __aicore__ inline void ScatterUpdate(const GlobalTensor<T> &dst, const LocalTensor<T> &outLocal,
                                         const LocalTensor<int64_t> &indexLocal, int64_t startIdx, int64_t rows,
                                         int64_t headSize)
    {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
        for (int64_t i = 0; i < rows; i++) {
            int64_t offset = indexLocal.GetValue(i);
            SToMTE3Sync();
            if (offset >= 0) {
                if constexpr (isPA) {
                    int64_t gmOffset = headSize * offset;
                    int64_t ubOffset = headSize * i;
                    DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
                } else {
                    int64_t batchId = (startIdx + i) / this->tilingData_->seqLength;
                    int64_t gmOffset = batchId * this->tilingData_->cacheLength * headSize + offset * headSize;
                    int64_t ubOffset = i * headSize;
                    DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T, bool isPA = false>
    __aicore__ inline void ScatterUpdateV2(const GlobalTensor<T> &dst, const LocalTensor<T> &outLocal, int64_t startIdx,
                                           int64_t rows, int64_t headSize)
    {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = i * headSize;
            int64_t tokensPerBatch = this->tilingData_->seqLength * this->tilingData_->numHead;
            int64_t batchId = tokenId / tokensPerBatch;
            int64_t tokenInBatch = tokenId % tokensPerBatch;                                 // token in batch
            int64_t headIdx = tokenInBatch / this->tilingData_->seqLength;                   // numHead
            int64_t seqIdx = tokenInBatch % this->tilingData_->seqLength;                    // seq
            int64_t offset = this->indexGm(batchId * this->tilingData_->seqLength + seqIdx); //[batch, seq]
            SToMTE3Sync();
            if (offset >= 0) {
                if constexpr (isPA) {
                    int64_t gmOffset = offset * headSize * this->tilingData_->numHead + headIdx * headSize;
                    SToMTE3Sync();
                    DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
                } else {
                    int64_t gmOffset =
                        batchId * this->tilingData_->numHead * this->tilingData_->cacheLength * headSize +
                        headIdx * this->tilingData_->cacheLength * headSize + offset * headSize;
                    DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T, bool isElementWise = true>
    __aicore__ inline void RoPE(const LocalTensor<T> &outLocal, const LocalTensor<T> &xLocal,
                                const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal,
                                const LocalTensor<float> &wsLocal, int64_t rows, int64_t headSize)
    {
        constexpr static int64_t NUM_ONE = 1;
        constexpr static int64_t NUM_TWO = 2;
        constexpr static int64_t NUM_FOUR = 4;
        constexpr static int64_t NUM_EIGHT = 8;
        if constexpr (isElementWise) {
            /**
             * cosLocalFp32 : [rows * headSize * 0, rows * headSize * 1]
             * sinLocalFp32 : [rows * headSize * 1, rows * headSize * 2]
             * y0           : [rows * headSize * 2, rows * headSize * 3]
             * y1           : [rows * headSize * 3, rows * headSize * 4]
             * realLocalFp32: [rows * headSize * 4, rows * headSize * 5]
             * imagLocalFp32: [rows * headSize * 5, rows * headSize * 6]
             * realLocal    : [rows * headSize * 6, rows * headSize * 7]
             * imagLocal    : [rows * headSize * 7, rows * headSize * 8]
             */
            int64_t cosLocalFp32Offset = rows * headSize * 0;
            int64_t sinLocalFp32Offset = rows * headSize * 1;
            int64_t y0Offset = rows * headSize * 2;
            int64_t y1Offset = rows * headSize * 3;
            int64_t realLocalFp32Offset = rows * headSize * 4;
            int64_t imagLocalFp32Offset = rows * headSize * 5;
            int64_t realLocalOffset = rows * headSize * 6;
            int64_t imagLocalOffset = rows * headSize * 7;

            LocalTensor<float> cosLocalFp32 = wsLocal[cosLocalFp32Offset];
            LocalTensor<float> sinLocalFp32 = wsLocal[sinLocalFp32Offset];
            LocalTensor<float> y0 = wsLocal[y0Offset];
            LocalTensor<float> y1 = wsLocal[y1Offset];
            LocalTensor<float> realLocalFp32 = wsLocal[realLocalFp32Offset];
            LocalTensor<float> imagLocalFp32 = wsLocal[imagLocalFp32Offset];
            LocalTensor<T> realLocal = wsLocal[realLocalOffset].template ReinterpretCast<T>();
            LocalTensor<T> imagLocal = wsLocal[imagLocalOffset].template ReinterpretCast<T>();

            // step #1: cast cosLocal and sinLocal to fp32
            Cast(cosLocalFp32, cosLocal, RoundMode::CAST_NONE, rows * headSize);
            Cast(sinLocalFp32, sinLocal, RoundMode::CAST_NONE, rows * headSize);
            PipeBarrier<PIPE_V>();

            // step #2: Gather out real and imag
            uint64_t rsvdCnt = 0;
            GatherMask(realLocal, xLocal, NUM_ONE, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            GatherMask(imagLocal, xLocal, NUM_TWO, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();

            // step #3: Cast realLocal and imagLocal to Fp32
            Cast(realLocalFp32, realLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            Cast(imagLocalFp32, imagLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();

            // step #4: y0 = (realLocalFp32, imagLocalFp32) * cosLocalFp32
            Mul(y0, realLocalFp32, cosLocalFp32, /*mask*/ (headSize >> 1), /*repeat*/ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y0[(headSize >> 1)], imagLocalFp32, cosLocalFp32[(headSize >> 1)], /*mask*/ (headSize >> 1),
                /*repeat*/ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #5: y1 = (-imagLocalFp32, realLocalFp32) * sinLocalFp32
            Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();
            Mul(y1, imagLocalFp32, sinLocalFp32, /*mask*/ (headSize >> 1), /*repeat*/ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y1[(headSize >> 1)], realLocalFp32, sinLocalFp32[(headSize >> 1)], /*mask*/ (headSize >> 1),
                /*repeat*/ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #6: y0 = y0 + y1
            Add(y0, y0, y1, rows * headSize);
            PipeBarrier<PIPE_V>();

            // step #7: outLocal = Cast(y0, T)
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, y0, RoundMode::CAST_RINT, rows * headSize);
            } else if constexpr (std::is_same<T, half>::value) {
                Cast(outLocal, y0, RoundMode::CAST_NONE, rows * headSize);
            }
        }
    }

    template <typename T>
    __aicore__ inline void RmsNorm(const LocalTensor<T> &outLocal, const LocalTensor<T> &xLocal,
                                   const LocalTensor<float> &gammaLocal, const LocalTensor<float> &wsLocal,
                                   int64_t rows, int64_t headSize)
    {
        constexpr static int64_t NUM_EIGHT = 8;
        constexpr static int64_t NUM_SIXTEEN = 16;
        constexpr static int64_t NUM_SIXTY_FOUR = 64;
        int64_t xLocalFp32Offset = 0;                  // [0, rows * headSize]
        int64_t xSquareLocalOffset = rows * headSize;  // [rows * headSize, rows * headSize * 2]
        int64_t xSumLocalOffset = rows * headSize * 2; // [rows * headSize * 2, rows * headSize * 3]
        LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
        LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
        LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];

        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, rows * headSize);
        PipeBarrier<PIPE_V>();
        Mul(xSquareLocal, xLocalFp32, xLocalFp32, rows * headSize);
        PipeBarrier<PIPE_V>();
        int64_t repeatTimes = rows * (headSize >> 1) / NUM_SIXTY_FOUR;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 256
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 128
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 64
        PipeBarrier<PIPE_V>();

        WholeReduceSum(xSumLocal, xSquareLocal, NUM_SIXTY_FOUR, rows, 1, 1, NUM_EIGHT);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, this->tilingData_->reciprocal, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, this->tilingData_->epsilon, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, rows);
        PipeBarrier<PIPE_V>();

        // Brcb
        int64_t block_count = (rows + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 / xSquareLocal
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Div(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], xSquareLocal[rowId * NUM_EIGHT],
                NUM_SIXTY_FOUR, (headSize / NUM_SIXTY_FOUR), {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 * xSquareLocal [rows, this->RMS_NORM_LENGTH] * [this->RMS_NORM_LENGTH]
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Mul(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], gammaLocal, headSize);
        }
        PipeBarrier<PIPE_V>();

        // Cast xLocalFp32 to outLocal
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(outLocal, xLocalFp32, RoundMode::CAST_RINT, rows * headSize);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(outLocal, xLocalFp32, RoundMode::CAST_NONE, rows * headSize);
        }
    }

    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

private:
    TQue<QuePosition::VECIN, 1> inQueueIndex;
    int64_t methodMode = 0;
};
} // namespace KvRmsNormRopeCache

#endif // _KV_RMS_NORM_ROPE_CACHE_B16_B1SD_H_
