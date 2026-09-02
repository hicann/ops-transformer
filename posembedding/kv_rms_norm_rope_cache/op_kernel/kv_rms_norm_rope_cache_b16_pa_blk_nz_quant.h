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
 * \file kv_rms_norm_rope_cache_b16_pa_blk_nz_quant.h
 * \brief
 */
#ifndef _KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_QUANT_H_
#define _KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_QUANT_H_
#include "kv_rms_norm_rope_cache_comm.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;
constexpr int64_t D1_SIZE2_PA_BLK_NZ_QUANT = 2;
constexpr int64_t D1_SIZE4_PA_BLK_NZ_QUANT = 4;
constexpr int64_t D1_SIZE6_PA_BLK_NZ_QUANT = 6;
constexpr int64_t D1_SIZE16_PA_BLK_NZ_QUANT = 16;
constexpr int64_t D0_SIZE32_PA_BLK_NZ_QUANT = 32;

template <bool isPagedAttention, typename KV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelKvRmsNormRopeCacheQuantB16PABLKNZ
    : public KernelKvRmsNormRopeCacheCutBSQuant<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE> {
public:
    __aicore__ inline KernelKvRmsNormRopeCacheQuantB16PABLKNZ(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheCutBSQuant<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache,
                                GM_ADDR v_cache, GM_ADDR optional_k_rope, GM_ADDR optional_c_kv, GM_ADDR kpe_scale,
                                GM_ADDR ckv_scale, GM_ADDR kpe_offset, GM_ADDR ckv_offset, GM_ADDR v)
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
            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE *)kv + GetBlockIdx() * this->tilingData_->blockFactor *
                                                                   (this->RMS_NORM_LENGTH + this->ROPE_LENGTH));
        } else {
            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE *)kv +
                                       GetBlockIdx() * this->tilingData_->blockFactor * this->RMS_NORM_LENGTH);
            this->vGm.SetGlobalBuffer((__gm__ KV_DTYPE *)v +
                                      GetBlockIdx() * this->tilingData_->blockFactor * this->V_LENGTH);
            if (this->tilingData_->numHead > 1) {
                int64_t startPosIdx = GetBlockIdx() * this->tilingData_->blockFactor;
                this->batchIdx = startPosIdx / (this->tilingData_->numHead * this->tilingData_->seqLength);
                this->numHeadIdx =
                    (startPosIdx - this->batchIdx * this->tilingData_->numHead * this->tilingData_->seqLength) /
                    (this->tilingData_->seqLength);
                this->seqIdx =
                    (startPosIdx - this->batchIdx * this->tilingData_->numHead * this->tilingData_->seqLength) %
                    (this->tilingData_->seqLength);
            }
        }

        this->gammaGm.SetGlobalBuffer((__gm__ KV_DTYPE *)gamma);
        this->cosGm.SetGlobalBuffer((__gm__ KV_DTYPE *)cos +
                                    GetBlockIdx() * this->tilingData_->blockFactor * this->ROPE_LENGTH);
        this->sinGm.SetGlobalBuffer((__gm__ KV_DTYPE *)sin +
                                    GetBlockIdx() * this->tilingData_->blockFactor * this->ROPE_LENGTH);
        this->indexGm.SetGlobalBuffer((__gm__ int64_t *)index);
        this->kCacheGm.SetGlobalBuffer((__gm__ K_DTYPE *)k_cache);
        this->vCacheGm.SetGlobalBuffer((__gm__ V_DTYPE *)v_cache);

        if (this->tilingData_->isOutputKv) {
            this->kCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE *)optional_k_rope);
            this->vCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE *)optional_c_kv);
        }

        this->kRopeScaleGm.SetGlobalBuffer((__gm__ float *)kpe_scale);
        this->cKvScaleGm.SetGlobalBuffer((__gm__ float *)ckv_scale);

        if (kpe_offset != nullptr && this->tilingData_->isKQuant) {
            this->kRopeOffsetGm.SetGlobalBuffer((__gm__ float *)kpe_offset);
            isAsymmetricKQuant = true;
        }
        if (ckv_offset != nullptr && this->tilingData_->isVQuant) {
            this->cKvOffsetGm.SetGlobalBuffer((__gm__ float *)ckv_offset);
            isAsymmetricVQuant = true;
        }

        // init pipe
        if (methodMode == 0) {
            this->pipe_->InitBuffer(
                this->inQueueGamma, 2,
                (2 * this->RMS_NORM_LENGTH + this->ROPE_LENGTH) * sizeof(float)); // 2*(2*512+64)*4/1024=8.5
            this->pipe_->InitBuffer(this->inQueueX, 2,
                                    this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)); // 2*16*512*2/1024=32
            this->pipe_->InitBuffer(
                this->outQueue, 2,
                this->ubFactor * this->RMS_NORM_LENGTH * (sizeof(KV_DTYPE) + sizeof(int8_t))); // 2*16*512*3/1024=48
            this->pipe_->InitBuffer(this->wsBuffer,
                                    3 * this->ubFactor * this->RMS_NORM_LENGTH * sizeof(float)); // 3*16*512*4/1024=96
        } else {
            // ubFactor = 16 -> 32 可以放大到32
            this->pipe_->InitBuffer(
                this->inQueueGamma, 2,
                (this->RMS_NORM_LENGTH + this->tilingData_->numHead * (this->V_LENGTH + this->RMS_NORM_LENGTH)) *
                    sizeof(float)); // 2* {192 + N * (192+128)}*4/1024 <= 21.5 gma rms v_len
            this->pipe_->InitBuffer(this->inQueueOffset, 2,
                                    this->tilingData_->numHead * (this->RMS_NORM_LENGTH + this->V_LENGTH) *
                                        sizeof(float)); // 2* N * (192+128)*4/1024 <= 20
            this->pipe_->InitBuffer(this->inQueueX, 2,
                                    this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)); // 2*16*192*2/1024=12
            this->pipe_->InitBuffer(this->inQueueV, 2,
                                    this->ubFactor * this->V_LENGTH * sizeof(KV_DTYPE)); // 2*16*128*2/1024=8
            this->pipe_->InitBuffer(
                this->outQueue, 2,
                this->ubFactor * this->RMS_NORM_LENGTH * (sizeof(KV_DTYPE) + sizeof(int8_t))); // 2*16*192*3/1024=18
            this->pipe_->InitBuffer(this->wsBuffer,
                                    3 * this->ubFactor * this->RMS_NORM_LENGTH * sizeof(float)); // 3*16*192*4/1024=36
        }
    }

    __aicore__ inline void Process()
    {
        if (methodMode == 0) {
            DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
            DataCopyPadExtParams<float> padParamsFp32{false, 0, 0, 0};
            DataCopyPadExtParams<int64_t> padParamsInt64{false, 0, 0, 0};
            DataCopyExtParams copyParamsContinguous;
            copyParamsContinguous.blockCount = 1;
            // CopyIn gamma: [RMS_NORM_LENGTH]
            LocalTensor<float> gammaLocalFp32 = this->inQueueGamma.template AllocTensor<float>();
            LocalTensor<KV_DTYPE> gammaLocal =
                gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
            LocalTensor<float> localCkvScale = gammaLocalFp32[this->RMS_NORM_LENGTH];
            LocalTensor<float> localKpeScale = gammaLocalFp32[2 * this->RMS_NORM_LENGTH];
            copyParamsContinguous.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(gammaLocal, this->gammaGm, copyParamsContinguous, padParams);
            copyParamsContinguous.blockLen = this->RMS_NORM_LENGTH * sizeof(float);
            if (this->tilingData_->isVQuant) {
                DataCopyPad(localCkvScale, this->cKvScaleGm, copyParamsContinguous, padParamsFp32);
            }
            copyParamsContinguous.blockLen = this->ROPE_LENGTH * sizeof(float);
            if (this->tilingData_->isKQuant) {
                DataCopyPad(localKpeScale, this->kRopeScaleGm, copyParamsContinguous, padParamsFp32);
            }
            this->inQueueGamma.EnQue(gammaLocalFp32);
            gammaLocalFp32 = this->inQueueGamma.template DeQue<float>();
            gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
            localCkvScale = gammaLocalFp32[this->RMS_NORM_LENGTH];
            localKpeScale = gammaLocalFp32[2 * this->RMS_NORM_LENGTH];
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
                // CopyIn x/cos/sin [this->ubFactor, ROPE_LENGTH]
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

                // Calc: RoPE
                LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
                LocalTensor<int8_t> quantOutLocal =
                    outLocal.template ReinterpretCast<int8_t>()[this->ubFactor * this->ROPE_LENGTH * sizeof(KV_DTYPE)];
                RoPEWithQuant<KV_DTYPE, true>(outLocal, quantOutLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer,
                                              localKpeScale, this->ubFactor, this->ROPE_LENGTH);
                this->inQueueX.FreeTensor(ropeLocal);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<KV_DTYPE>();
                // Scatter Update kCache
                if (this->tilingData_->isKQuant) {
                    ScatterUpdateNZWithQuant<int8_t, KV_DTYPE, isPagedAttention, D1_SIZE2_PA_BLK_NZ_QUANT,
                                             D0_SIZE32_PA_BLK_NZ_QUANT>((GlobalTensor<int8_t> &)this->kCacheGm,
                                                                        this->kCacheGmNd, outLocal, quantOutLocal,
                                                                        startIdx, this->ubFactor, this->ROPE_LENGTH);
                } else {
                    LocalTensor<K_DTYPE> outputLocal = outLocal.template ReinterpretCast<K_DTYPE>();
                    ScatterUpdateNZWithQuant<K_DTYPE, KV_DTYPE, isPagedAttention, D1_SIZE4_PA_BLK_NZ,
                                             D0_SIZE16_PA_BLK_NZ>(this->kCacheGm, this->kCacheGmNd, outLocal,
                                                                  outputLocal, startIdx, this->ubFactor,
                                                                  this->ROPE_LENGTH);
                }
                this->outQueue.FreeTensor(outLocal);

                // CopyIn x: [this->ubFactor, RmsLength]
                LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
                copyParams.blockCount = static_cast<uint16_t>(this->ubFactor);
                copyParams.blockLen = static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE));
                copyParams.srcStride = static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE));
                copyParams.dstStride = 0;
                DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
                this->inQueueX.EnQue(xLocal);
                xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

                // Calc: RmsNorm
                outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
                quantOutLocal =
                    outLocal
                        .template ReinterpretCast<int8_t>()[this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)];
                RmsNormWithQuant<KV_DTYPE>(outLocal, quantOutLocal, xLocal, gammaLocalFp32, workspaceBuffer,
                                           localCkvScale, this->ubFactor, this->RMS_NORM_LENGTH);
                this->inQueueX.FreeTensor(xLocal);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<KV_DTYPE>();

                // Scatter Update vCache
                if (this->tilingData_->isVQuant) {
                    ScatterUpdateNZWithQuant<int8_t, KV_DTYPE, isPagedAttention, D1_SIZE16_PA_BLK_NZ_QUANT,
                                             D0_SIZE32_PA_BLK_NZ_QUANT>(
                        (GlobalTensor<int8_t> &)this->vCacheGm, this->vCacheGmNd, outLocal, quantOutLocal, startIdx,
                        this->ubFactor, this->RMS_NORM_LENGTH);
                } else {
                    LocalTensor<V_DTYPE> outputLocal = outLocal.template ReinterpretCast<V_DTYPE>();
                    ScatterUpdateNZWithQuant<V_DTYPE, KV_DTYPE, isPagedAttention, D1_SIZE32_PA_BLK_NZ,
                                             D0_SIZE16_PA_BLK_NZ>(this->vCacheGm, this->vCacheGmNd, outLocal,
                                                                  outputLocal, startIdx, this->ubFactor,
                                                                  this->RMS_NORM_LENGTH);
                }
                this->outQueue.FreeTensor(outLocal);
            }
            if (this->ubTail > 0) {
                int64_t kvGlobalMemoryOffset =
                    this->ubLoop * this->ubFactor * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH);
                int64_t freqGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->ROPE_LENGTH;
                int64_t indexGlobalMemoryOffset = this->ubLoop * this->ubFactor;
                int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + this->ubLoop * this->ubFactor;

                LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
                LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
                LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];
                // CopyIn x/cos/sin [this->ubTail, ROPE_LENGTH]
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

                // Calc: RoPE
                LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
                LocalTensor<int8_t> quantOutLocal =
                    outLocal.template ReinterpretCast<int8_t>()[this->ubTail * this->ROPE_LENGTH * sizeof(KV_DTYPE)];
                RoPEWithQuant<KV_DTYPE, true>(outLocal, quantOutLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer,
                                              localKpeScale, this->ubTail, this->ROPE_LENGTH);
                this->inQueueX.FreeTensor(ropeLocal);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<KV_DTYPE>();
                // Scatter Update kCache
                if (this->tilingData_->isKQuant) {
                    ScatterUpdateNZWithQuant<int8_t, KV_DTYPE, isPagedAttention, D1_SIZE2_PA_BLK_NZ_QUANT,
                                             D0_SIZE32_PA_BLK_NZ_QUANT>((GlobalTensor<int8_t> &)this->kCacheGm,
                                                                        this->kCacheGmNd, outLocal, quantOutLocal,
                                                                        startIdx, this->ubTail, this->ROPE_LENGTH);
                } else {
                    LocalTensor<K_DTYPE> outputLocal = outLocal.template ReinterpretCast<K_DTYPE>();
                    ScatterUpdateNZWithQuant<K_DTYPE, KV_DTYPE, isPagedAttention, D1_SIZE4_PA_BLK_NZ,
                                             D0_SIZE16_PA_BLK_NZ>(this->kCacheGm, this->kCacheGmNd, outLocal,
                                                                  outputLocal, startIdx, this->ubTail,
                                                                  this->ROPE_LENGTH);
                }
                this->outQueue.FreeTensor(outLocal);

                // CopyIn x: [this->ubTail, RmsLength]
                LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
                copyParams.blockCount = static_cast<uint16_t>(this->ubTail);
                copyParams.blockLen = static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE));
                copyParams.srcStride = static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE));
                copyParams.dstStride = 0;
                DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
                this->inQueueX.EnQue(xLocal);
                xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

                // Calc: RmsNorm
                outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
                quantOutLocal =
                    outLocal
                        .template ReinterpretCast<int8_t>()[this->ubTail * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)];
                RmsNormWithQuant<KV_DTYPE>(outLocal, quantOutLocal, xLocal, gammaLocalFp32, workspaceBuffer,
                                           localCkvScale, this->ubTail, this->RMS_NORM_LENGTH);
                this->inQueueX.FreeTensor(xLocal);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<KV_DTYPE>();

                // Scatter Update vCache
                if (this->tilingData_->isVQuant) {
                    ScatterUpdateNZWithQuant<int8_t, KV_DTYPE, isPagedAttention, D1_SIZE16_PA_BLK_NZ_QUANT,
                                             D0_SIZE32_PA_BLK_NZ_QUANT>((GlobalTensor<int8_t> &)this->vCacheGm,
                                                                        this->vCacheGmNd, outLocal, quantOutLocal,
                                                                        startIdx, this->ubTail, this->RMS_NORM_LENGTH);
                } else {
                    LocalTensor<V_DTYPE> outputLocal = outLocal.template ReinterpretCast<V_DTYPE>();
                    ScatterUpdateNZWithQuant<V_DTYPE, KV_DTYPE, isPagedAttention, D1_SIZE32_PA_BLK_NZ,
                                             D0_SIZE16_PA_BLK_NZ>(this->vCacheGm, this->vCacheGmNd, outLocal,
                                                                  outputLocal, startIdx, this->ubTail,
                                                                  this->RMS_NORM_LENGTH);
                }
                this->outQueue.FreeTensor(outLocal);
            }
            this->inQueueGamma.FreeTensor(gammaLocalFp32);
        } else {
            DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
            DataCopyPadExtParams<float> padParamsFp32{false, 0, 0, 0};
            DataCopyPadExtParams<int64_t> padParamsInt64{false, 0, 0, 0};
            DataCopyExtParams copyParamsContinguous;
            copyParamsContinguous.blockCount = 1;
            // CopyIn gamma, scale, offset
            LocalTensor<float> gammaLocalFp32 = this->inQueueGamma.template AllocTensor<float>();
            LocalTensor<KV_DTYPE> gammaLocal =
                gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
            LocalTensor<float> localKvScale = gammaLocalFp32[this->RMS_NORM_LENGTH];
            LocalTensor<float> localVScale =
                gammaLocalFp32[this->RMS_NORM_LENGTH + this->RMS_NORM_LENGTH * this->tilingData_->numHead];
            LocalTensor<float> localKvOffset = this->inQueueOffset.template AllocTensor<float>();
            LocalTensor<float> localVOffset = localKvOffset[this->tilingData_->numHead * this->RMS_NORM_LENGTH];

            copyParamsContinguous.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(gammaLocal, this->gammaGm, copyParamsContinguous, padParams);
            copyParamsContinguous.blockLen = this->tilingData_->numHead * this->RMS_NORM_LENGTH * sizeof(float);
            if (this->tilingData_->isKQuant) {
                DataCopyPad(localKvScale, this->kRopeScaleGm, copyParamsContinguous, padParamsFp32);
                if (isAsymmetricKQuant) {
                    DataCopyPad(localKvOffset, this->kRopeOffsetGm, copyParamsContinguous, padParamsFp32);
                }
            }
            copyParamsContinguous.blockLen = this->tilingData_->numHead * this->V_LENGTH * sizeof(float);
            if (this->tilingData_->isVQuant) {
                DataCopyPad(localVScale, this->cKvScaleGm, copyParamsContinguous, padParamsFp32);
                if (isAsymmetricVQuant) {
                    DataCopyPad(localVOffset, this->cKvOffsetGm, copyParamsContinguous, padParamsFp32);
                }
            }
            this->inQueueGamma.EnQue(gammaLocalFp32);
            gammaLocalFp32 = this->inQueueGamma.template DeQue<float>();
            gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
            localKvScale = gammaLocalFp32[this->RMS_NORM_LENGTH];
            localVScale = gammaLocalFp32[this->RMS_NORM_LENGTH + this->RMS_NORM_LENGTH * this->tilingData_->numHead];

            this->inQueueOffset.EnQue(localKvOffset);
            localKvOffset = this->inQueueOffset.template DeQue<float>();
            localVOffset = localKvOffset[this->tilingData_->numHead * this->RMS_NORM_LENGTH];

            Cast(gammaLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->RMS_NORM_LENGTH);

            LocalTensor<float> workspaceBuffer = this->wsBuffer.template Get<float>();
            for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
                int64_t kvGlobalMemoryOffset = loopIdx * this->ubFactor * this->RMS_NORM_LENGTH;
                int64_t freqGlobalMemoryOffset = loopIdx * this->ubFactor * this->ROPE_LENGTH;
                int64_t indexGlobalMemoryOffset = loopIdx * this->ubFactor;
                int64_t vGlobalMemoryOffset = loopIdx * this->ubFactor * this->V_LENGTH;
                int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + loopIdx * this->ubFactor;

                LocalTensor<KV_DTYPE> rmsNormLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
                // CopyIn kv [this->ubFactor, RMS_NORM_LENGTH]
                DataCopyExtParams copyParams{
                    /* blockCount */ static_cast<uint16_t>(this->ubFactor),
                    /* blockLen (Byte) */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                    /* srcStride */ 0,
                    /* dstStride */ 0,
                    /* rsv */ 0};
                DataCopyPad(rmsNormLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
                this->inQueueX.EnQue(rmsNormLocal);
                rmsNormLocal = this->inQueueX.template DeQue<KV_DTYPE>();

                // CopyIn : cos/sin
                copyParamsContinguous.blockLen = this->ubFactor * this->ROPE_LENGTH * sizeof(KV_DTYPE);
                LocalTensor<KV_DTYPE> cosLocal = this->inQueueV.template AllocTensor<KV_DTYPE>();
                LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];
                DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
                DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
                this->inQueueV.EnQue(cosLocal);
                cosLocal = this->inQueueV.template DeQue<KV_DTYPE>();
                sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];
                LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
                LocalTensor<int8_t> quantOutLocal =
                    outLocal
                        .template ReinterpretCast<int8_t>()[this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)];
                // Calc: RmsNorm and Rope and KQuant
                int64_t bIdxTemp = this->batchIdx;
                int64_t nIdxTemp = this->numHeadIdx;
                int64_t sIdxTemp = this->seqIdx;
                RmsNormAndRopeV2WithQuant<KV_DTYPE, true>(rmsNormLocal, outLocal, quantOutLocal, gammaLocalFp32,
                                                          cosLocal, sinLocal, localKvScale, localKvOffset,
                                                          isAsymmetricKQuant, workspaceBuffer, this->ubFactor,
                                                          this->RMS_NORM_LENGTH, this->ROPE_LENGTH);
                this->batchIdx = this->tilingData_->isVQuant ? bIdxTemp : this->batchIdx;
                this->numHeadIdx = this->tilingData_->isVQuant ? nIdxTemp : this->numHeadIdx;
                this->seqIdx = this->tilingData_->isVQuant ? sIdxTemp : this->seqIdx;
                this->inQueueV.FreeTensor(cosLocal);
                this->inQueueX.FreeTensor(rmsNormLocal);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<KV_DTYPE>();

                // Scatter Update kCache
                if (this->tilingData_->isKQuant) {
                    ScatterUpdateNZWithQuant<int8_t, KV_DTYPE, isPagedAttention, D1_SIZE6_PA_BLK_NZ_QUANT,
                                             D0_SIZE32_PA_BLK_NZ_QUANT>(
                        (GlobalTensor<int8_t> &)this->kCacheGm, this->kCacheGmNd, outLocal, quantOutLocal, startIdx,
                        this->ubFactor, this->RMS_NORM_LENGTH);
                } else {
                    LocalTensor<K_DTYPE> outputLocal = outLocal.template ReinterpretCast<K_DTYPE>();
                    ScatterUpdateNZWithQuant<K_DTYPE, KV_DTYPE, isPagedAttention, D1_SIZE12_PA_BLK_NZ,
                                             D0_SIZE16_PA_BLK_NZ>(this->kCacheGm, this->kCacheGmNd, outLocal,
                                                                  outputLocal, startIdx, this->ubFactor,
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
                this->inQueueV.EnQue(vLocal);
                vLocal = this->inQueueV.template DeQue<KV_DTYPE>();

                // Calc: VQuant
                outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
                quantOutLocal =
                    outLocal.template ReinterpretCast<int8_t>()[this->ubFactor * this->V_LENGTH * sizeof(KV_DTYPE)];
                DataCopy(outLocal, vLocal, this->ubFactor * this->V_LENGTH);
                SetWaitFlag<HardEvent::V_V>(HardEvent::V_V);
                if (this->tilingData_->isVQuant) {
                    Cast(workspaceBuffer, vLocal, RoundMode::CAST_NONE, this->ubFactor * this->V_LENGTH);
                    PipeBarrier<PIPE_V>();
                    Quant<KV_DTYPE>(quantOutLocal, workspaceBuffer, localVScale, localVOffset, isAsymmetricVQuant,
                                    this->V_LENGTH, this->ubFactor);
                }
                this->inQueueV.FreeTensor(vLocal);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<KV_DTYPE>();

                // Scatter Update Vcache
                if (this->tilingData_->isVQuant) {
                    ScatterUpdateNZWithQuant<int8_t, KV_DTYPE, isPagedAttention, D1_SIZE4_PA_BLK_NZ_QUANT,
                                             D0_SIZE32_PA_BLK_NZ_QUANT>((GlobalTensor<int8_t> &)this->vCacheGm,
                                                                        this->vCacheGmNd, outLocal, quantOutLocal,
                                                                        startIdx, this->ubFactor, this->V_LENGTH);
                } else {
                    LocalTensor<V_DTYPE> outputLocal = outLocal.template ReinterpretCast<V_DTYPE>();
                    ScatterUpdateNZWithQuant<V_DTYPE, KV_DTYPE, isPagedAttention, D1_SIZE8_PA_BLK_NZ,
                                             D0_SIZE16_PA_BLK_NZ>(this->vCacheGm, this->vCacheGmNd, outLocal,
                                                                  outputLocal, startIdx, this->ubFactor,
                                                                  this->V_LENGTH);
                }
                this->outQueue.FreeTensor(outLocal);
            }
            if (this->ubTail > 0) {
                int64_t kvGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->RMS_NORM_LENGTH;
                int64_t freqGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->ROPE_LENGTH;
                int64_t indexGlobalMemoryOffset = this->ubLoop * this->ubFactor;
                int64_t vGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->V_LENGTH;
                int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + this->ubLoop * this->ubFactor;

                LocalTensor<KV_DTYPE> rmsNormLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
                // CopyIn kv [this->ubTail, RMS_NORM_LENGTH]
                DataCopyExtParams copyParams{
                    /* blockCount */ static_cast<uint16_t>(this->ubTail),
                    /* blockLen (Byte) */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                    /* srcStride */ 0,
                    /* dstStride */ 0,
                    /* rsv */ 0};
                DataCopyPad(rmsNormLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
                this->inQueueX.EnQue(rmsNormLocal);
                rmsNormLocal = this->inQueueX.template DeQue<KV_DTYPE>();

                // CopyIn : cos/sin
                copyParamsContinguous.blockLen = this->ubTail * this->ROPE_LENGTH * sizeof(KV_DTYPE);
                LocalTensor<KV_DTYPE> cosLocal = this->inQueueV.template AllocTensor<KV_DTYPE>();
                LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];
                DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
                DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
                this->inQueueV.EnQue(cosLocal);
                cosLocal = this->inQueueV.template DeQue<KV_DTYPE>();
                sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];
                LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
                LocalTensor<int8_t> quantOutLocal =
                    outLocal
                        .template ReinterpretCast<int8_t>()[this->ubTail * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)];

                // Calc : RmsNorm and Rope and KQuant
                int64_t bIdxTemp = this->batchIdx;
                int64_t nIdxTemp = this->numHeadIdx;
                int64_t sIdxTemp = this->seqIdx;
                RmsNormAndRopeV2WithQuant<KV_DTYPE, true>(rmsNormLocal, outLocal, quantOutLocal, gammaLocalFp32,
                                                          cosLocal, sinLocal, localKvScale, localKvOffset,
                                                          isAsymmetricKQuant, workspaceBuffer, this->ubTail,
                                                          this->RMS_NORM_LENGTH, this->ROPE_LENGTH);
                this->batchIdx = this->tilingData_->isVQuant ? bIdxTemp : this->batchIdx;
                this->numHeadIdx = this->tilingData_->isVQuant ? nIdxTemp : this->numHeadIdx;
                this->seqIdx = this->tilingData_->isVQuant ? sIdxTemp : this->seqIdx;
                this->inQueueV.FreeTensor(cosLocal);
                this->inQueueX.FreeTensor(rmsNormLocal);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<KV_DTYPE>();

                // Scatter Update kCache
                if (this->tilingData_->isKQuant) {
                    ScatterUpdateNZWithQuant<int8_t, KV_DTYPE, isPagedAttention, D1_SIZE6_PA_BLK_NZ_QUANT,
                                             D0_SIZE32_PA_BLK_NZ_QUANT>((GlobalTensor<int8_t> &)this->kCacheGm,
                                                                        this->kCacheGmNd, outLocal, quantOutLocal,
                                                                        startIdx, this->ubTail, this->RMS_NORM_LENGTH);
                } else {
                    LocalTensor<K_DTYPE> outputLocal = outLocal.template ReinterpretCast<K_DTYPE>();
                    ScatterUpdateNZWithQuant<K_DTYPE, KV_DTYPE, isPagedAttention, D1_SIZE12_PA_BLK_NZ,
                                             D0_SIZE16_PA_BLK_NZ>(this->kCacheGm, this->kCacheGmNd, outLocal,
                                                                  outputLocal, startIdx, this->ubTail,
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
                this->inQueueV.EnQue(vLocal);
                vLocal = this->inQueueV.template DeQue<KV_DTYPE>();

                // Calc: VQuant
                outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
                quantOutLocal =
                    outLocal.template ReinterpretCast<int8_t>()[this->ubTail * this->V_LENGTH * sizeof(KV_DTYPE)];
                DataCopy(outLocal, vLocal, this->ubTail * this->V_LENGTH);
                SetWaitFlag<HardEvent::V_V>(HardEvent::V_V);
                if (this->tilingData_->isVQuant) {
                    Cast(workspaceBuffer, vLocal, RoundMode::CAST_NONE, this->ubTail * this->V_LENGTH);
                    PipeBarrier<PIPE_V>();
                    Quant<KV_DTYPE>(quantOutLocal, workspaceBuffer, localVScale, localVOffset, isAsymmetricVQuant,
                                    this->V_LENGTH, this->ubTail);
                }
                this->inQueueV.FreeTensor(vLocal);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<KV_DTYPE>();

                // Scatter Update Vcache
                if (this->tilingData_->isVQuant) {
                    ScatterUpdateNZWithQuant<int8_t, KV_DTYPE, isPagedAttention, D1_SIZE4_PA_BLK_NZ_QUANT,
                                             D0_SIZE32_PA_BLK_NZ_QUANT>((GlobalTensor<int8_t> &)this->vCacheGm,
                                                                        this->vCacheGmNd, outLocal, quantOutLocal,
                                                                        startIdx, this->ubTail, this->V_LENGTH);
                } else {
                    LocalTensor<V_DTYPE> outputLocal = outLocal.template ReinterpretCast<V_DTYPE>();
                    ScatterUpdateNZWithQuant<V_DTYPE, KV_DTYPE, isPagedAttention, D1_SIZE8_PA_BLK_NZ,
                                             D0_SIZE16_PA_BLK_NZ>(this->vCacheGm, this->vCacheGmNd, outLocal,
                                                                  outputLocal, startIdx, this->ubTail, this->V_LENGTH);
                }
                this->outQueue.FreeTensor(outLocal);
            }
            this->inQueueGamma.FreeTensor(gammaLocalFp32);
            this->inQueueOffset.FreeTensor(localKvOffset);
        }
    }

    template <typename T, bool isElementWise = true>
    __aicore__ inline void RoPEWithQuant(const LocalTensor<T> &outLocal, const LocalTensor<int8_t> &quantOutLocal,
                                         const LocalTensor<T> &xLocal, const LocalTensor<T> &cosLocal,
                                         const LocalTensor<T> &sinLocal, const LocalTensor<float> &wsLocal,
                                         const LocalTensor<float> &scaleLocal, int64_t rows, int64_t headSize)
    {
        constexpr static int64_t NUM_ONE = 1;
        constexpr static int64_t NUM_TWO = 2;
        constexpr static int64_t NUM_THREE = 3;
        constexpr static int64_t NUM_FOUR = 4;
        constexpr static int64_t NUM_EIGHT = 8;
        constexpr static int64_t NUM_SIXTY_FOUR = 64;
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

            // step #6.1: y1 = y0 * scaleLocal
            if (this->tilingData_->isKQuant) {
                int64_t totalMask = headSize;
                while (totalMask > 0) {
                    // issue a instruction
                    int64_t offset = headSize - totalMask;
                    Mul(y1[offset], y0[offset], scaleLocal[offset],
                        /*mask*/ totalMask >= NUM_SIXTY_FOUR ? NUM_SIXTY_FOUR : totalMask,
                        /*repeat*/ rows,
                        {1, 1, 1, static_cast<uint8_t>(headSize / NUM_EIGHT),
                         static_cast<uint8_t>(headSize / NUM_EIGHT), 0});
                    totalMask -= NUM_SIXTY_FOUR;
                }
                PipeBarrier<PIPE_V>();

                // step #6.2: quantOutLocal = RoundFloat2Int8(y1)
                RoundFloat2Int8(quantOutLocal, y1, rows * headSize);
            }

            // step #7: outLocal = Cast(y0, T)
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, y0, RoundMode::CAST_RINT, rows * headSize);
            } else if constexpr (std::is_same<T, half>::value) {
                Cast(outLocal, y0, RoundMode::CAST_NONE, rows * headSize);
            }
        }
    }

    template <typename T>
    __aicore__ inline void Quant(const LocalTensor<int8_t> &quantOutLocal, LocalTensor<float> &srcTensor,
                                 const LocalTensor<float> &scaleLocal, const LocalTensor<float> &offsetLocal,
                                 bool ifAsymmetricQuant, int64_t headSize, int64_t rows)
    {
        if (this->tilingData_->numHead == 1 ||
            (this->tilingData_->numHead > 1 && (this->seqIdx + rows) < this->tilingData_->seqLength)) {
            int64_t totalMask = headSize;
            while (totalMask > 0) {
                // issue a instruction
                int64_t offset4Quant = this->numHeadIdx * headSize + headSize - totalMask;
                int64_t offset = headSize - totalMask;
                Mul(srcTensor[offset], srcTensor[offset], scaleLocal[offset4Quant],
                    /*mask*/ totalMask >= NUM_SIXTY_FOUR ? NUM_SIXTY_FOUR : totalMask,
                    /*repeat*/ rows,
                    {1, 1, 1, static_cast<uint8_t>(headSize / NUM_EIGHT), static_cast<uint8_t>(headSize / NUM_EIGHT),
                     0});
                if (ifAsymmetricQuant) {
                    PipeBarrier<PIPE_V>();
                    Add(srcTensor[offset], srcTensor[offset], offsetLocal[offset4Quant],
                        totalMask >= NUM_SIXTY_FOUR ? NUM_SIXTY_FOUR : totalMask, rows,
                        {1, 1, 1, static_cast<uint8_t>(headSize / NUM_EIGHT),
                         static_cast<uint8_t>(headSize / NUM_EIGHT), 0});
                }
                totalMask -= NUM_SIXTY_FOUR;
            }
            this->seqIdx += rows;
        } else {
            for (int i = 0; i < rows; ++i) {
                if (this->seqIdx >= this->tilingData_->seqLength) {
                    this->numHeadIdx++;
                    this->seqIdx -= this->tilingData_->seqLength;
                    if (this->numHeadIdx >= this->tilingData_->numHead) {
                        this->numHeadIdx -= this->tilingData_->numHead;
                        this->batchIdx++;
                    }
                }
                int64_t offset4Quant = this->numHeadIdx * headSize;
                int64_t srcOffset = i * headSize;
                int64_t repeat = headSize / NUM_SIXTY_FOUR;
                SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
                Mul(srcTensor[srcOffset], srcTensor[srcOffset], scaleLocal[offset4Quant], NUM_SIXTY_FOUR, repeat,
                    {1, 1, 1, NUM_EIGHT, NUM_EIGHT, NUM_EIGHT});
                if (ifAsymmetricQuant) {
                    PipeBarrier<PIPE_V>();
                    Add(srcTensor[srcOffset], srcTensor[srcOffset], offsetLocal[offset4Quant], NUM_SIXTY_FOUR, repeat,
                        {1, 1, 1, NUM_EIGHT, NUM_EIGHT, NUM_EIGHT});
                }
                this->seqIdx++;
            }
        }
        PipeBarrier<PIPE_V>();
        RoundFloat2Int8(quantOutLocal, srcTensor, rows * headSize);
    }

    __aicore__ inline void RoundFloat2Int8(const LocalTensor<int8_t> &dstTensor, const LocalTensor<float> &srcTensor,
                                           int32_t size)
    {
        Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
    }

    template <typename T1, typename T2, bool isPA = false, int64_t D1, int64_t D0>
    __aicore__ inline void ScatterUpdateNZWithQuant(const GlobalTensor<T1> &dst, const GlobalTensor<T2> &dstNd,
                                                    const LocalTensor<T2> &outLocal,
                                                    const LocalTensor<T1> &quantOutLocal, int64_t startIdx,
                                                    int64_t rows, int64_t headSize)
    {
        DataCopyExtParams copyParamsNz{static_cast<uint16_t>(D1),              // 2
                                       static_cast<uint32_t>(D0 * sizeof(T1)), // 32
                                       0,
                                       static_cast<uint32_t>(this->tilingData_->blockSize * D0 * sizeof(T1)) -
                                           static_cast<uint32_t>(D0 * sizeof(T1)), // 256*32*1-32*1
                                       0};
        int64_t indexPageLength =
            (this->tilingData_->seqLength + this->tilingData_->blockSize - 1) / this->tilingData_->blockSize; // 1
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t batchId = tokenId / (this->tilingData_->seqLength * this->tilingData_->numHead);
            int64_t numHeadIdx_ = (tokenId - batchId * this->tilingData_->seqLength * this->tilingData_->numHead) /
                                  this->tilingData_->seqLength;
            int64_t tokenIdInCurrentBatch = tokenId % (this->tilingData_->seqLength * this->tilingData_->numHead) %
                                            this->tilingData_->seqLength; // 当前token所在seqIdx(跨numHead)
            int64_t indexPageId = tokenIdInCurrentBatch / this->tilingData_->blockSize;
            int64_t indexOffset = batchId * indexPageLength + indexPageId;
            int64_t pageOffset = this->indexGm(indexOffset); // index.shape=[batchsize * ceil(seq_len / block_size)]
                                                             // 同一个token的不同numHead共享同一份index值
            SToMTE3Sync();
            int64_t tokenOffsetInCurrentPage =
                tokenIdInCurrentBatch % this->tilingData_->blockSize;   // 当前token所在blockoffset
            int64_t pageId = pageOffset / this->tilingData_->blockSize; // blkNum
            int64_t ubOffset = headSize * i;
            if (pageOffset >= 0) {
                // [BlockNum, BlockSize, N, headSize]
                int64_t gmOffsetNz = pageId * D1 * this->tilingData_->blockSize * D0 * this->tilingData_->numHead +
                                     tokenOffsetInCurrentPage * D0 +
                                     this->tilingData_->blockSize * numHeadIdx_ * D1 * D0;
                DataCopyPad(dst[gmOffsetNz], quantOutLocal[ubOffset], copyParamsNz);
            }
            if (this->tilingData_->isOutputKv) {
                DataCopyExtParams copyParamsNd{1, static_cast<uint32_t>(headSize * sizeof(T2)), 0, 0, 0};
                int64_t gmOffsetNd = tokenId * headSize;
                DataCopyPad(dstNd[gmOffsetNd], outLocal[ubOffset], copyParamsNd);
            }
        }
    }

    template <typename T>
    __aicore__ inline void RmsNormWithQuant(const LocalTensor<T> &outLocal, const LocalTensor<int8_t> &quantOutLocal,
                                            const LocalTensor<T> &xLocal, const LocalTensor<float> &gammaLocal,
                                            const LocalTensor<float> &wsLocal, const LocalTensor<float> &scaleLocal,
                                            int64_t rows, int64_t headSize)
    {
        constexpr static int64_t NUM_EIGHT = 8;
        constexpr static int64_t NUM_SIXTEEN = 16;
        constexpr static int64_t NUM_SIXTY_FOUR = 64;
        int64_t xSquareLocalOffset = 0;                // [0, rows * headSize]
        int64_t xLocalFp32Offset = rows * headSize;    // [rows * headSize, rows * headSize * 2]
        int64_t xSumLocalOffset = rows * headSize * 2; // [rows * headSize * 2, rows * headSize * 3]
        LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
        LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
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
        if (this->tilingData_->isVQuant) {
            PipeBarrier<PIPE_V>();
            for (int64_t rowId = 0; rowId < rows; rowId++) {
                Mul(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], scaleLocal, headSize);
            }
            PipeBarrier<PIPE_V>();
            RoundFloat2Int8(quantOutLocal, xLocalFp32, rows * headSize);
        }
    }

    template <typename T, bool isElementWise = true>
    __aicore__ inline void RmsNormAndRopeV2WithQuant(
        LocalTensor<T> &xLocal, const LocalTensor<T> &outLocal, const LocalTensor<int8_t> &quantOutLocal,
        const LocalTensor<float> &gammaLocal, const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal,
        const LocalTensor<float> &scaleLocal, const LocalTensor<float> &offsetLocal, bool ifAsymmetricQuant,
        LocalTensor<float> &wsLocal, int64_t rows, int64_t headSize, int64_t ropeLen)
    {
        constexpr static int64_t NUM_ZERO = 0;
        constexpr static int64_t NUM_ONE = 1;
        constexpr static int64_t NUM_TWO = 2;
        constexpr static int64_t NUM_THREE = 3;
        constexpr static int64_t NUM_FOUR = 4;
        constexpr static int64_t NUM_FIVE = 5;
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
        /*Brcb每次只能取8个数去填充，每个数复制成8个相同值（8个相同值是因为一个数复制到一个block里，一个block针对float32是8个数），
        所以迭代次数为rows/8，以遍历所有xSumLocal数
        {1, NUM_EIGHT}是固定写法，1表示单次迭代内，目的操作数不同block间是连续的，8表示不同迭代间间隔8个block*/
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
            int64_t cosLocalFp32Offset = xLocalFp32Offset + rows * ropeLen * NUM_ZERO;
            int64_t sinLocalFp32Offset = xLocalFp32Offset + rows * ropeLen * NUM_ONE;
            int64_t y0Offset = xLocalFp32Offset + rows * ropeLen * NUM_TWO;
            int64_t y1Offset = xLocalFp32Offset + rows * ropeLen * NUM_THREE;
            int64_t realLocalFp32Offset = xLocalFp32Offset + rows * ropeLen * NUM_FOUR;
            int64_t imagLocalFp32Offset = xLocalFp32Offset + rows * ropeLen * NUM_FIVE;

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
        // do KQuant
        if (this->tilingData_->isKQuant) {
            PipeBarrier<PIPE_V>();
            Quant<T>(quantOutLocal, xLocalFp32, scaleLocal, offsetLocal, ifAsymmetricQuant, headSize, rows);
        }
    }

    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

private:
    // indexGm used as indexGmNz in this template.
    bool isAsymmetricKQuant = false;
    bool isAsymmetricVQuant = false;
    int64_t methodMode = 0;
};
} // namespace KvRmsNormRopeCache

#endif // _KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_QUANT_H_
