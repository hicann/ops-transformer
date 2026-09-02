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
 * \file kv_rms_norm_rope_cache_b16_pa_blk_nz.h
 * \brief
 */
#ifndef _KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_H_
#define _KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_H_
#include "kv_rms_norm_rope_cache_comm.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;
constexpr int64_t D1_SIZE4_PA_BLK_NZ = 4;
constexpr int64_t D1_SIZE8_PA_BLK_NZ = 8;
constexpr int64_t D1_SIZE12_PA_BLK_NZ = 12;
constexpr int64_t D1_SIZE32_PA_BLK_NZ = 32;
constexpr int64_t D0_SIZE16_PA_BLK_NZ = 16;

template <bool isPagedAttention, typename KV_DTYPE>
class KernelKvRmsNormRopeCacheB16PABLKNZ
    : public KernelKvRmsNormRopeCacheCutBSScatter<isPagedAttention, KV_DTYPE, KV_DTYPE, KV_DTYPE> {
public:
    __aicore__ inline KernelKvRmsNormRopeCacheB16PABLKNZ(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheCutBSScatter<isPagedAttention, KV_DTYPE, KV_DTYPE, KV_DTYPE>(pipe, tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR v,
        GM_ADDR optional_k_rope,
        GM_ADDR optional_c_kv) // RMS_NORM_LENGTH=[512, 192]  ROPE_LENGTH=[64, 64]     V_LENGTH=[512, 128]
    {
        methodMode = v == nullptr ? 0 : 1;
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
        if (methodMode == 0) {
            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE *)kv +
                                       GetBlockIdx() * this->tilingData_->blockFactor *
                                           (this->RMS_NORM_LENGTH + this->ROPE_LENGTH)); // blockFactor * (512 + 64)
        } else {
            rmsNormOutputPosition = this->ubFactor * this->RMS_NORM_LENGTH * NUM_TWO;
            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE *)kv + GetBlockIdx() * this->tilingData_->blockFactor *
                                                                   this->RMS_NORM_LENGTH); // blockFactor * 192
            this->vGm.SetGlobalBuffer((__gm__ KV_DTYPE *)v + GetBlockIdx() * this->tilingData_->blockFactor *
                                                                 this->V_LENGTH); // blockFactor * 128
        }
        this->gammaGm.SetGlobalBuffer((__gm__ KV_DTYPE *)gamma); // 512 * sizeof(half)
        this->cosGm.SetGlobalBuffer((__gm__ KV_DTYPE *)cos + GetBlockIdx() * this->tilingData_->blockFactor *
                                                                 this->ROPE_LENGTH); // blockFactor * ROPE_LENGTH
        this->sinGm.SetGlobalBuffer((__gm__ KV_DTYPE *)sin + GetBlockIdx() * this->tilingData_->blockFactor *
                                                                 this->ROPE_LENGTH); // blockFactor * ROPE_LENGTH
        this->indexGm.SetGlobalBuffer((__gm__ int64_t *)index);                      //
        this->kCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE *)k_cache);
        this->vCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE *)v_cache);
        if (this->tilingData_->isOutputKv) {
            this->kCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE *)optional_k_rope);
            this->vCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE *)optional_c_kv);
        }
        // init pipe, for v2, the buffer is <= 1.5 + 3.75*ubFactor
        this->pipe_->InitBuffer(this->inQueueGamma, KvRmsNormRopeCache::GAMMA_BUFFER_COUNT,
                                this->RMS_NORM_LENGTH * sizeof(float)); // [2*512*4/1024=4, 2*192*4/1024=1.5]
        this->pipe_->InitBuffer(
            this->inQueueX, KvRmsNormRopeCache::X_BUFFER_COUNT,
            this->ubFactor * this->RMS_NORM_LENGTH *
                sizeof(KV_DTYPE)); // [2*ubFactor*512*2/1024=2*ubFactor, 2*ubFactor*192*2/1024=0.75*ubFactor]
        this->pipe_->InitBuffer(
            this->outQueue, KvRmsNormRopeCache::OUT_BUFFER_COUNT,
            this->ubFactor * this->RMS_NORM_LENGTH *
                sizeof(KV_DTYPE)); // [2*ubFactor*512*2/1024=2*ubFactor, 2*ubFactor*192*2/1024=0.75*ubFactor]
        this->pipe_->InitBuffer(
            this->wsBuffer,
            KvRmsNormRopeCache::WORKSPACE_BUFFER_MULTIPLIER * this->ubFactor * this->RMS_NORM_LENGTH *
                sizeof(float)); // [3*ubFactor*512*4/1024=6*ubFactor, 3*ubFactor*192*4/1024=2.25*ubFactor]
    }

    __aicore__ inline void Process()
    {
        if (methodMode == 1) {
            ProcessV2();
            return;
        }
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

            // CopyIn x: [this->ubFactor, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            DataCopyExtParams copyParams{
                /* blockCount */ static_cast<uint16_t>(this->ubFactor),
                /* blockLen (Byte) */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                /* srcStride */ static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                /* dstStride */ 0,
                /* rsv */ 0};

            DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RmsNorm<KV_DTYPE>(outLocal, xLocal, gammaLocalFp32, workspaceBuffer, this->ubFactor, this->RMS_NORM_LENGTH);
            this->inQueueX.FreeTensor(xLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update vCache
            ScatterUpdateNZ<KV_DTYPE, isPagedAttention, D1_SIZE32_PA_BLK_NZ, D0_SIZE16_PA_BLK_NZ>(
                this->vCacheGm, this->vCacheGmNd, outLocal, startIdx, this->ubFactor, this->RMS_NORM_LENGTH);
            this->outQueue.FreeTensor(outLocal);

            LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubFactor * this->ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];

            // CopyIn x/cos/sin [this->ubFactor, this->ROPE_LENGTH]
            copyParams.blockCount = static_cast<uint16_t>(this->ubFactor);
            copyParams.blockLen = static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE));
            copyParams.srcStride = static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE));
            copyParams.dstStride = 0;
            DataCopyPad(ropeLocal, this->kvGm[kvGlobalMemoryOffset + this->RMS_NORM_LENGTH], copyParams, padParams);

            copyParamsContinguous.blockLen = this->ubFactor * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueX.EnQue(ropeLocal);
            ropeLocal = this->inQueueX.template DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[this->ubFactor * this->ROPE_LENGTH];
            sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];

            // Calc: RoPE
            outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RoPE<KV_DTYPE, true>(outLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer, this->ubFactor,
                                 this->ROPE_LENGTH);
            this->inQueueX.FreeTensor(ropeLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache
            ScatterUpdateNZ<KV_DTYPE, isPagedAttention, D1_SIZE4_PA_BLK_NZ, D0_SIZE16_PA_BLK_NZ>(
                this->kCacheGm, this->kCacheGmNd, outLocal, startIdx, this->ubFactor, this->ROPE_LENGTH);
            this->outQueue.FreeTensor(outLocal);
        }
        if (this->ubTail > 0) {
            int64_t kvGlobalMemoryOffset = this->ubLoop * this->ubFactor * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH);
            int64_t freqGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = this->ubLoop * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + this->ubLoop * this->ubFactor;

            // CopyIn x: [this->ubTail, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            DataCopyExtParams copyParams{
                /* blockCount */ static_cast<uint16_t>(this->ubTail),
                /* blockLen (Byte) */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                /* srcStride */ static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                /* dstStride */ 0,
                /* rsv */ 0};
            DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RmsNorm<KV_DTYPE>(outLocal, xLocal, gammaLocalFp32, workspaceBuffer, this->ubTail, this->RMS_NORM_LENGTH);
            this->inQueueX.FreeTensor(xLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update vCache
            ScatterUpdateNZ<KV_DTYPE, isPagedAttention, D1_SIZE32_PA_BLK_NZ, D0_SIZE16_PA_BLK_NZ>(
                this->vCacheGm, this->vCacheGmNd, outLocal, startIdx, this->ubTail, this->RMS_NORM_LENGTH);
            this->outQueue.FreeTensor(outLocal);

            LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];

            // CopyIn x/cos/sin [this->ubTail, this->ROPE_LENGTH]
            copyParams.blockCount = static_cast<uint16_t>(this->ubTail);
            copyParams.blockLen = static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE));
            copyParams.srcStride = static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE));
            copyParams.dstStride = 0;
            DataCopyPad(ropeLocal, this->kvGm[kvGlobalMemoryOffset + this->RMS_NORM_LENGTH], copyParams, padParams);
            copyParamsContinguous.blockLen = this->ubTail * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueX.EnQue(ropeLocal);
            ropeLocal = this->inQueueX.template DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
            sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];

            // Calc: RoPE
            outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RoPE<KV_DTYPE, true>(outLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer, this->ubTail,
                                 this->ROPE_LENGTH);
            this->inQueueX.FreeTensor(ropeLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache
            ScatterUpdateNZ<KV_DTYPE, isPagedAttention, D1_SIZE4_PA_BLK_NZ, D0_SIZE16_PA_BLK_NZ>(
                this->kCacheGm, this->kCacheGmNd, outLocal, startIdx, this->ubTail, this->ROPE_LENGTH);
            this->outQueue.FreeTensor(outLocal);
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
        PipeBarrier<PIPE_V>();

        LocalTensor<float> workspaceBuffer = this->wsBuffer.template Get<float>();

        for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
            int64_t kvGlobalMemoryOffset = loopIdx * this->ubFactor * this->RMS_NORM_LENGTH;
            int64_t vGlobalMemoryOffset = loopIdx * this->ubFactor * this->V_LENGTH;
            int64_t freqGlobalMemoryOffset = loopIdx * this->ubFactor * this->ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = loopIdx * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + loopIdx * this->ubFactor;

            // Scatter Update vCache
            LocalTensor<KV_DTYPE> vLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            DataCopyExtParams copyParams{/* blockCount */ static_cast<uint16_t>(this->ubFactor),
                                         /* blockLen (Byte) */ static_cast<uint32_t>(this->V_LENGTH * sizeof(KV_DTYPE)),
                                         /* srcStride */ 0,
                                         /* dstStride */ 0,
                                         /* rsv */ 0};
            DataCopyPad(vLocal, this->vGm[vGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(vLocal);
            vLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            ScatterUpdateNZV2<KV_DTYPE, isPagedAttention, D1_SIZE8_PA_BLK_NZ, D0_SIZE16_PA_BLK_NZ>(
                this->vCacheGm, this->vCacheGmNd, vLocal, startIdx, this->ubFactor, this->V_LENGTH);
            this->inQueueX.FreeTensor(vLocal);

            // CopyIn x: [this->ubFactor, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            copyParams.blockLen = static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE));
            DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            RmsNormV2<KV_DTYPE>(xLocal, gammaLocalFp32, workspaceBuffer, this->ubFactor, this->RMS_NORM_LENGTH);
            this->inQueueX.FreeTensor(xLocal);

            LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubFactor * this->ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];

            // CopyIn x/cos/sin [this->ubFactor, this->ROPE_LENGTH]
            copyParamsContinguous.blockLen = this->ubFactor * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueX.EnQue(ropeLocal);
            ropeLocal = this->inQueueX.template DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[this->ubFactor * this->ROPE_LENGTH];
            sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];

            // Calc: RoPE
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RoPEV2<KV_DTYPE, true>(outLocal, cosLocal, sinLocal, workspaceBuffer, this->ubFactor, this->ROPE_LENGTH);
            this->inQueueX.FreeTensor(ropeLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache
            ScatterUpdateNZV2<KV_DTYPE, isPagedAttention, D1_SIZE12_PA_BLK_NZ, D0_SIZE16_PA_BLK_NZ>(
                this->kCacheGm, this->kCacheGmNd, outLocal, startIdx, this->ubFactor, this->RMS_NORM_LENGTH);
            this->outQueue.FreeTensor(outLocal);
        }
        if (this->ubTail > 0) {
            int64_t kvGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->RMS_NORM_LENGTH;
            int64_t vGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->V_LENGTH;
            int64_t freqGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = this->ubLoop * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + this->ubLoop * this->ubFactor;

            // CopyIn x: [this->ubTail, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            DataCopyExtParams copyParams{
                /* blockCount */ static_cast<uint16_t>(this->ubTail),
                /* blockLen (Byte) */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                /* srcStride */ 0,
                /* dstStride */ 0,
                /* rsv */ 0};
            DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            RmsNormV2<KV_DTYPE>(xLocal, gammaLocalFp32, workspaceBuffer, this->ubTail, this->RMS_NORM_LENGTH);
            this->inQueueX.FreeTensor(xLocal);

            // Scatter Update vCache
            LocalTensor<KV_DTYPE> vLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            copyParams.blockLen = static_cast<uint32_t>(this->V_LENGTH * sizeof(KV_DTYPE));
            DataCopyPad(vLocal, this->vGm[vGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(vLocal);
            vLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            ScatterUpdateNZV2<KV_DTYPE, isPagedAttention, D1_SIZE8_PA_BLK_NZ, D0_SIZE16_PA_BLK_NZ>(
                this->vCacheGm, this->vCacheGmNd, vLocal, startIdx, this->ubTail, this->V_LENGTH);
            this->inQueueX.FreeTensor(vLocal);

            LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];

            // CopyIn x/cos/sin [this->ubTail, this->ROPE_LENGTH]
            copyParamsContinguous.blockLen = this->ubTail * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueX.EnQue(ropeLocal);
            ropeLocal = this->inQueueX.template DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
            sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];

            // Calc: RoPE
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RoPEV2<KV_DTYPE, true>(outLocal, cosLocal, sinLocal, workspaceBuffer, this->ubTail, this->ROPE_LENGTH);
            this->inQueueX.FreeTensor(ropeLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache
            ScatterUpdateNZV2<KV_DTYPE, isPagedAttention, D1_SIZE12_PA_BLK_NZ, D0_SIZE16_PA_BLK_NZ>(
                this->kCacheGm, this->kCacheGmNd, outLocal, startIdx, this->ubTail, this->RMS_NORM_LENGTH);
            this->outQueue.FreeTensor(outLocal);
        }
        this->inQueueGamma.FreeTensor(gammaLocalFp32);
    }

    template <typename T, bool isElementWise = true>
    __aicore__ inline void RoPE(const LocalTensor<T> &outLocal, const LocalTensor<T> &xLocal,
                                const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal,
                                const LocalTensor<float> &wsLocal, int64_t rows, int64_t headSize)
    {
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

    template <typename T, bool isElementWise = true>
    __aicore__ inline void RoPEV2(const LocalTensor<T> &outLocal, const LocalTensor<T> &cosLocal,
                                  const LocalTensor<T> &sinLocal, const LocalTensor<float> &wsLocal, int64_t rows,
                                  int64_t headSize)
    {
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

            LocalTensor<float> cosLocalFp32 = wsLocal[cosLocalFp32Offset];
            LocalTensor<float> sinLocalFp32 = wsLocal[sinLocalFp32Offset];
            LocalTensor<float> y0 = wsLocal[y0Offset];
            LocalTensor<float> y1 = wsLocal[y1Offset];
            LocalTensor<float> realLocalFp32 = wsLocal[realLocalFp32Offset];
            LocalTensor<float> imagLocalFp32 = wsLocal[imagLocalFp32Offset];

            // step #1: cast cosLocal and sinLocal to fp32
            Cast(cosLocalFp32, cosLocal, RoundMode::CAST_NONE, rows * headSize);
            Cast(sinLocalFp32, sinLocal, RoundMode::CAST_NONE, rows * headSize);
            PipeBarrier<PIPE_V>();

            CopyRepeatParams copyRepeatParams{/* dstStride */ static_cast<uint16_t>(1),
                                              /* srcStride */ static_cast<uint16_t>(1),
                                              /* dstRepeatSize */ NUM_EIGHT,
                                              /* srcRepeatSize */ NUM_TWENTY_FOUR};
            Copy(y0, wsLocal[this->rmsNormOutputPosition], NUM_SIXTY_FOUR, rows, copyRepeatParams);
            PipeBarrier<PIPE_V>();

            // step #2: Gather out real and imag
            uint64_t rsvdCnt = 0;
            GatherMask(realLocalFp32, y0, NUM_ONE, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            GatherMask(imagLocalFp32, y0, NUM_TWO, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
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

            // y0 -> wsLocal[this->rmsNormOutputPosition]
            copyRepeatParams.dstRepeatSize = NUM_TWENTY_FOUR;
            copyRepeatParams.srcRepeatSize = NUM_EIGHT;
            Copy(wsLocal[this->rmsNormOutputPosition], y0, NUM_SIXTY_FOUR, rows, copyRepeatParams);
            PipeBarrier<PIPE_V>();

            // step #7: outLocal = Cast(y0, T)
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, wsLocal[this->rmsNormOutputPosition], RoundMode::CAST_RINT,
                     rows * this->RMS_NORM_LENGTH);
            } else if constexpr (std::is_same<T, half>::value) {
                Cast(outLocal, wsLocal[this->rmsNormOutputPosition], RoundMode::CAST_NONE,
                     rows * this->RMS_NORM_LENGTH);
            }
        }
    }

    template <typename T, bool isPA = false, int64_t D1, int64_t D0>
    __aicore__ inline void ScatterUpdateNZ(const GlobalTensor<T> &dst, const GlobalTensor<T> &dstNd,
                                           const LocalTensor<T> &outLocal, int64_t startIdx, int64_t rows,
                                           int64_t headSize)
    {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParamsNz{static_cast<uint16_t>(D1), static_cast<uint32_t>(D0 * sizeof(T)), 0,
                                       static_cast<uint32_t>(this->tilingData_->blockSize * D0 * sizeof(T)) -
                                           static_cast<uint32_t>(D0 * sizeof(T)),
                                       0};
        int64_t indexPageLength =
            (this->tilingData_->seqLength + this->tilingData_->blockSize - 1) / this->tilingData_->blockSize;
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = headSize * i;
            if (this->tilingData_->isOutputKv) {
                DataCopyExtParams copyParamsNd{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
                int64_t gmOffsetNd = tokenId * headSize;
                DataCopyPad(dstNd[gmOffsetNd], outLocal[ubOffset], copyParamsNd);
            }

            int64_t batchId = tokenId / this->tilingData_->seqLength;
            int64_t tokenIdInCurrentBatch = tokenId % this->tilingData_->seqLength;
            int64_t indexPageId = tokenIdInCurrentBatch / this->tilingData_->blockSize;
            int64_t indexOffset = batchId * indexPageLength + indexPageId;
            int64_t pageOffset = this->indexGm(indexOffset);
            int64_t tokenOffsetInCurrentPage = tokenIdInCurrentBatch % this->tilingData_->blockSize;
            int64_t pageId = pageOffset / this->tilingData_->blockSize;

            if (pageOffset >= 0) {
                int64_t gmOffsetNz = pageId * D1 * this->tilingData_->blockSize * D0 + tokenOffsetInCurrentPage * D0;
                SToMTE3Sync();
                DataCopyPad(dst[gmOffsetNz], outLocal[ubOffset], copyParamsNz);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T, bool isPA = false, int64_t D1, int64_t D0>
    __aicore__ inline void ScatterUpdateNZV2(const GlobalTensor<T> &dst, const GlobalTensor<T> &dstNd,
                                             const LocalTensor<T> &outLocal, int64_t startIdx, int64_t rows,
                                             int64_t headSize)
    {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParamsNz{static_cast<uint16_t>(D1), static_cast<uint32_t>(D0 * sizeof(T)), 0,
                                       static_cast<uint32_t>(this->tilingData_->blockSize * D0 * sizeof(T)) -
                                           static_cast<uint32_t>(D0 * sizeof(T)),
                                       0};
        int64_t indexPageLength =
            (this->tilingData_->seqLength + this->tilingData_->blockSize - 1) / this->tilingData_->blockSize;
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = headSize * i;
            if (this->tilingData_->isOutputKv) {
                DataCopyExtParams copyParamsNd{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
                int64_t gmOffsetNd = tokenId * headSize;
                DataCopyPad(dstNd[gmOffsetNd], outLocal[ubOffset], copyParamsNd);
            }

            int64_t batchId = tokenId / (this->tilingData_->seqLength * this->tilingData_->numHead);
            int64_t tokenIdInCurrentBatch =
                tokenId % (this->tilingData_->seqLength * this->tilingData_->numHead) % this->tilingData_->seqLength;
            int64_t indexPageId = tokenIdInCurrentBatch / this->tilingData_->blockSize;
            int64_t indexOffset = batchId * indexPageLength + indexPageId;
            int64_t pageOffset = this->indexGm(indexOffset);
            int64_t tokenOffsetInCurrentPage = tokenIdInCurrentBatch % this->tilingData_->blockSize;
            int64_t pageId = pageOffset / this->tilingData_->blockSize;
            int64_t curHead =
                tokenId % (this->tilingData_->seqLength * this->tilingData_->numHead) / this->tilingData_->seqLength;

            if (pageOffset >= 0) {
                int64_t gmOffsetNz = pageId * D1 * this->tilingData_->blockSize * D0 * this->tilingData_->numHead +
                                     curHead * this->tilingData_->blockSize * D1 * D0 + tokenOffsetInCurrentPage * D0;
                SToMTE3Sync();
                DataCopyPad(dst[gmOffsetNz], outLocal[ubOffset], copyParamsNz);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T>
    __aicore__ inline void RmsNorm(const LocalTensor<T> &outLocal, const LocalTensor<T> &xLocal,
                                   const LocalTensor<float> &gammaLocal, const LocalTensor<float> &wsLocal,
                                   int64_t rows, int64_t headSize)
    {
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
                NUM_SIXTY_FOUR, (headSize / NUM_SIXTY_FOUR),
                {1, 1, 0, static_cast<uint8_t>(headSize / NUM_SIXTY_FOUR),
                 static_cast<uint8_t>(headSize / NUM_SIXTY_FOUR), 0});
        }
        PipeBarrier<PIPE_V>();

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

    template <typename T>
    __aicore__ inline void RmsNormV2(const LocalTensor<T> &xLocal, const LocalTensor<float> &gammaLocal,
                                     const LocalTensor<float> &wsLocal, int64_t rows, int64_t headSize)
    {
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

        int64_t repeatTimes = rows * (headSize >> 1) / NUM_TWENTY_FOUR;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_TWENTY_FOUR], NUM_TWENTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_THREE, NUM_SIX, NUM_SIX}); // 96
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_TWENTY_FOUR], NUM_TWENTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_THREE, NUM_SIX, NUM_SIX}); // 48
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_TWENTY_FOUR], NUM_TWENTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_THREE, NUM_SIX, NUM_SIX}); // 24
        PipeBarrier<PIPE_V>();

        WholeReduceSum(xSumLocal, xSquareLocal, NUM_TWENTY_FOUR, rows, 1, 1, NUM_THREE);
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
                NUM_SIXTY_FOUR, (headSize / NUM_SIXTY_FOUR),
                {1, 1, 0, static_cast<uint8_t>(headSize / NUM_TWENTY_FOUR),
                 static_cast<uint8_t>(headSize / NUM_TWENTY_FOUR), 0});
        }
        PipeBarrier<PIPE_V>();

        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Mul(wsLocal[this->rmsNormOutputPosition + rowId * headSize], xLocalFp32[rowId * headSize], gammaLocal,
                headSize);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

private:
    constexpr static int64_t NUM_ONE = 1;
    constexpr static int64_t NUM_TWO = 2;
    constexpr static int64_t NUM_THREE = 3;
    constexpr static int64_t NUM_FOUR = 4;
    constexpr static int64_t NUM_SIX = 6;
    constexpr static int64_t NUM_EIGHT = 8;
    constexpr static int64_t NUM_SIXTEEN = 16;
    constexpr static int64_t NUM_TWENTY_FOUR = 24;
    constexpr static int64_t NUM_SIXTY_FOUR = 64;
    int64_t methodMode = 0;

    int64_t rmsNormOutputPosition = 0; // used for RmsNorm output in wsBuffer
    // this->indexGm used as indexGmNz in this template.
};
} // namespace KvRmsNormRopeCache

#endif // _KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_H_
