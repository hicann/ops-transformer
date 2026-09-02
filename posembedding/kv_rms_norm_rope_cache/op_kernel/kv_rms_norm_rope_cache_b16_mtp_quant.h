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
 * \file kv_rms_norm_rope_cache_b16_mtp_quant.h
 * \brief
 */
#ifndef _KV_RMS_NORM_ROPE_CACHE_B16_MTP_QUANT_H_
#define _KV_RMS_NORM_ROPE_CACHE_B16_MTP_QUANT_H_
#include "kv_rms_norm_rope_cache_comm.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;

template <bool isPagedAttention, typename KV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KvRmsNormRopeCacheB16MTPQUANT
    : public KernelKvRmsNormRopeCacheMTPQuant<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE> {
public:
    __aicore__ inline KvRmsNormRopeCacheB16MTPQUANT(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheMTPQuant<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache,
                                GM_ADDR v_cache, GM_ADDR optional_k_rope, GM_ADDR optional_c_kv, GM_ADDR k_rope_scale,
                                GM_ADDR c_kv_scale, GM_ADDR k_rope_offset, GM_ADDR c_kv_offset, GM_ADDR v)
    {
        methodMode = v == nullptr ? 0 : 1;
        this->InitSharedData(methodMode);
        if (methodMode == 1) {
            // init global memory
            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE *)kv +
                                       GetBlockIdx() * this->tilingData_->rowsPerBlock * this->tilingData_->numHead *
                                           this->tilingData_->seqLength * this->RMS_NORM_LENGTH);
            this->vGm.SetGlobalBuffer((__gm__ KV_DTYPE *)v + GetBlockIdx() * this->tilingData_->rowsPerBlock *
                                                                 this->tilingData_->numHead *
                                                                 this->tilingData_->seqLength * this->V_LENGTH);
            this->gammaGm.SetGlobalBuffer((__gm__ KV_DTYPE *)gamma);
            this->cosGm.SetGlobalBuffer((__gm__ KV_DTYPE *)cos + GetBlockIdx() * this->tilingData_->rowsPerBlock *
                                                                     this->tilingData_->numHead * this->ROPE_LENGTH);
            this->sinGm.SetGlobalBuffer((__gm__ KV_DTYPE *)sin + GetBlockIdx() * this->tilingData_->rowsPerBlock *
                                                                     this->tilingData_->numHead * this->ROPE_LENGTH);
            if constexpr (isPagedAttention) {
                this->indexGm.SetGlobalBuffer((__gm__ int64_t *)index);
                this->kCacheGm.SetGlobalBuffer((__gm__ K_DTYPE *)k_cache);
                this->vCacheGm.SetGlobalBuffer((__gm__ V_DTYPE *)v_cache);
            } else {
                this->indexGm.SetGlobalBuffer((__gm__ int64_t *)index + GetBlockIdx() *
                                                                            this->tilingData_->rowsPerBlock *
                                                                            this->tilingData_->seqLength);
                this->kCacheGm.SetGlobalBuffer((__gm__ K_DTYPE *)k_cache +
                                               GetBlockIdx() * this->tilingData_->rowsPerBlock *
                                                   this->tilingData_->numHead * this->tilingData_->cacheLength *
                                                   this->RMS_NORM_LENGTH);
                this->vCacheGm.SetGlobalBuffer((__gm__ V_DTYPE *)v_cache +
                                               GetBlockIdx() * this->tilingData_->rowsPerBlock *
                                                   this->tilingData_->numHead * this->tilingData_->cacheLength *
                                                   this->V_LENGTH);
            }
            if (this->tilingData_->isOutputKv) {
                this->kCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE *)optional_k_rope);
                this->vCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE *)optional_c_kv);
            }
            isKOffset = k_rope_offset == nullptr ? 0 : 1;
            isVOffset = c_kv_offset == nullptr ? 0 : 1;
            if (this->tilingData_->isKQuant) {
                this->kRopeScaleGm.SetGlobalBuffer((__gm__ float *)k_rope_scale);
                if (isKOffset == 1)
                    this->kRopeOffsetGm.SetGlobalBuffer((__gm__ float *)k_rope_offset);
            }
            if (this->tilingData_->isVQuant) {
                this->cKvScaleGm.SetGlobalBuffer((__gm__ float *)c_kv_scale);
                if (isVOffset == 1)
                    this->cKvOffsetGm.SetGlobalBuffer((__gm__ float *)c_kv_offset);
            }

            // init pipe
            this->pipe_->InitBuffer(this->inQueueX, 1,
                                    this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH * sizeof(float));
            this->pipe_->InitBuffer(this->inQueueV, 1,
                                    this->tilingData_->rowsPerBlock * this->V_LENGTH * sizeof(float));
            this->pipe_->InitBuffer(this->inQueueGamma, 1, this->RMS_NORM_LENGTH * sizeof(KV_DTYPE));
            this->pipe_->InitBuffer(this->CosSin, 1,
                                    4 * this->tilingData_->rowsPerBlock * this->ROPE_LENGTH * sizeof(KV_DTYPE));
            this->pipe_->InitBuffer(this->inQueueScaleOffset, 1,
                                    (2 * (this->RMS_NORM_LENGTH + this->V_LENGTH)) * sizeof(float));
            this->pipe_->InitBuffer(this->bufferXFp32,
                                    this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH * sizeof(float));
            this->pipe_->InitBuffer(this->bufferXSquare,
                                    2 * this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH * sizeof(float));
            this->pipe_->InitBuffer(this->bufferSum, ((this->tilingData_->rowsPerBlock + NUM_EIGHT - 1) / NUM_EIGHT) *
                                                         NUM_EIGHT * sizeof(float));
            this->pipe_->InitBuffer(this->outQueueV, 1,
                                    2 * this->tilingData_->rowsPerBlock * this->V_LENGTH * sizeof(KV_DTYPE));
            this->pipe_->InitBuffer(this->outQueueK, 1,
                                    2 * this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE));
            this->pipe_->InitBuffer(this->wsBuffer, this->tilingData_->rowsPerBlock * this->V_LENGTH * sizeof(float));
        } else {
            // exception
        }
    }

    __aicore__ inline void Process()
    {
        if (methodMode == 1) {
            for (int64_t n = 0; n < this->tilingData_->numHead; n++) {
                for (int64_t i = 0; i < this->tilingData_->seqLength; i++) {
                    // load quant scale and offset
                    LocalTensor<float> kScaleLocal = this->inQueueScaleOffset.template AllocTensor<float>();
                    LocalTensor<float> kOffsetLocal = kScaleLocal[this->RMS_NORM_LENGTH];
                    LocalTensor<float> vScaleLocal = kOffsetLocal[this->RMS_NORM_LENGTH];
                    LocalTensor<float> vOffsetLocal = vScaleLocal[this->V_LENGTH];
                    if (this->tilingData_->isKQuant) {
                        DataCopy(kScaleLocal, this->kRopeScaleGm[n * this->RMS_NORM_LENGTH], this->RMS_NORM_LENGTH);
                        if (isKOffset == 1) {
                            DataCopy(kOffsetLocal, this->kRopeOffsetGm[n * this->RMS_NORM_LENGTH],
                                     this->RMS_NORM_LENGTH);
                        }
                    }
                    if (this->tilingData_->isVQuant) {
                        DataCopy(vScaleLocal, this->cKvScaleGm[n * this->V_LENGTH], this->V_LENGTH);
                        if (isVOffset == 1) {
                            DataCopy(vOffsetLocal, this->cKvOffsetGm[n * this->V_LENGTH], this->V_LENGTH);
                        }
                    }
                    this->inQueueScaleOffset.EnQue(kScaleLocal);
                    kScaleLocal = this->inQueueScaleOffset.template DeQue<float>();
                    kOffsetLocal = kScaleLocal[this->RMS_NORM_LENGTH];
                    vScaleLocal = kOffsetLocal[this->RMS_NORM_LENGTH];
                    vOffsetLocal = vScaleLocal[this->V_LENGTH];

                    // Process K
                    LocalTensor<KV_DTYPE> xB16Local = this->inQueueX.template AllocTensor<KV_DTYPE>();
                    DataCopyExtParams copyXParams{
                        static_cast<uint16_t>(this->tilingData_->rowsPerBlock),
                        static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                        static_cast<uint32_t>(
                            ((this->tilingData_->seqLength - 1) * this->RMS_NORM_LENGTH +
                             (this->tilingData_->numHead - 1) * this->tilingData_->seqLength * this->RMS_NORM_LENGTH) *
                            sizeof(KV_DTYPE)),
                        0, 0};
                    DataCopyPadExtParams<KV_DTYPE> padXParams{false, 0, 0, 0};
                    int64_t nXOffset =
                        n * this->tilingData_->seqLength * this->RMS_NORM_LENGTH + i * this->RMS_NORM_LENGTH;
                    DataCopyPad(xB16Local, this->kvGm[nXOffset], copyXParams, padXParams);
                    this->inQueueX.EnQue(xB16Local);
                    xB16Local = this->inQueueX.template DeQue<KV_DTYPE>();

                    LocalTensor<float> xFp32Local = this->bufferXFp32.template Get<float>();
                    Cast(xFp32Local, xB16Local, RoundMode::CAST_NONE,
                         this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH);
                    PipeBarrier<PIPE_V>();
                    this->inQueueX.FreeTensor(xB16Local);

                    LocalTensor<KV_DTYPE> gammaLocal = this->inQueueGamma.template AllocTensor<KV_DTYPE>();
                    DataCopy(gammaLocal, this->gammaGm, this->RMS_NORM_LENGTH);
                    this->inQueueGamma.EnQue(gammaLocal);
                    gammaLocal = this->inQueueGamma.template DeQue<KV_DTYPE>();

                    RmsNormV2<KV_DTYPE>(xFp32Local, gammaLocal);
                    this->inQueueGamma.FreeTensor(gammaLocal);

                    LocalTensor<KV_DTYPE> cosLocal = this->CosSin.template AllocTensor<KV_DTYPE>();
                    LocalTensor<KV_DTYPE> sinLocal =
                        cosLocal[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH * NUM_TWO];
                    LocalTensor<KV_DTYPE> cosLocalB16 = cosLocal[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH];
                    LocalTensor<KV_DTYPE> sinLocalB16 = sinLocal[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH];

                    // load cosLocal, sinLocal [this->tilingData_->rowsPerBlock, this->ROPE_LENGTH]
                    LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template DeQue<KV_DTYPE>();
                    DataCopyExtParams copyCosSinParams{
                        static_cast<uint16_t>(this->tilingData_->rowsPerBlock),
                        static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                        static_cast<uint32_t>((this->tilingData_->numHead - 1) * this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                        0, 0};
                    DataCopyPad(cosLocalB16, this->cosGm[n * this->ROPE_LENGTH], copyCosSinParams, {false, 0, 0, 0});
                    DataCopyPad(sinLocalB16, this->sinGm[n * this->ROPE_LENGTH], copyCosSinParams, {false, 0, 0, 0});
                    this->CosSin.EnQue(cosLocal);
                    cosLocal = this->CosSin.template DeQue<KV_DTYPE>();
                    sinLocal = cosLocal[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH * NUM_TWO];
                    cosLocalB16 = cosLocal[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH];
                    sinLocalB16 = sinLocal[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH];

                    LocalTensor<KV_DTYPE> outLocalK = this->outQueueK.template AllocTensor<KV_DTYPE>();
                    LocalTensor<int8_t> outLocalKQuant =
                        outLocalK.template ReinterpretCast<int8_t>()[this->tilingData_->rowsPerBlock *
                                                                     this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)];
                    LocalTensor<float> workspaceBuffer = this->wsBuffer.template Get<float>();
                    RopeKV2<KV_DTYPE>(xLocal, xFp32Local, cosLocal, sinLocal, cosLocalB16, sinLocalB16, outLocalKQuant,
                                      kScaleLocal, kOffsetLocal, isKOffset, this->RMS_NORM_LENGTH);
                    this->CosSin.FreeTensor(cosLocal);
                    this->inQueueX.EnQue(xLocal);
                    xLocal = this->inQueueX.template DeQue<KV_DTYPE>();
                    this->outQueueK.EnQue(outLocalK);
                    outLocalK = this->outQueueK.template DeQue<KV_DTYPE>();
                    PipeBarrier<PIPE_V>();

                    DataCopy(outLocalK, xLocal, this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH);
                    SetWaitFlag<HardEvent::V_V>(HardEvent::V_V);
                    this->inQueueX.FreeTensor(xLocal);

                    // ScatterUpdate
                    if (this->tilingData_->isKQuant) {
                        ScatterUpdate<int8_t, KV_DTYPE>((GlobalTensor<int8_t> &)this->kCacheGm, this->kCacheGmNd,
                                                        outLocalK, outLocalKQuant, n, i, this->RMS_NORM_LENGTH);
                    } else {
                        LocalTensor<K_DTYPE> outputLocal = outLocalK.template ReinterpretCast<K_DTYPE>();
                        ScatterUpdate<K_DTYPE, KV_DTYPE>(this->kCacheGm, this->kCacheGmNd, outLocalK, outputLocal, n, i,
                                                         this->RMS_NORM_LENGTH);
                    }
                    this->outQueueK.FreeTensor(outLocalK);
                    PipeBarrier<PIPE_V>();

                    // Process V
                    LocalTensor<KV_DTYPE> yLocal = this->inQueueV.template AllocTensor<KV_DTYPE>();
                    DataCopyExtParams copyYParams{
                        static_cast<uint16_t>(this->tilingData_->rowsPerBlock),
                        static_cast<uint32_t>(this->V_LENGTH * sizeof(KV_DTYPE)),
                        static_cast<uint32_t>(
                            ((this->tilingData_->seqLength - 1) * this->V_LENGTH +
                             (this->tilingData_->numHead - 1) * this->tilingData_->seqLength * this->V_LENGTH) *
                            sizeof(KV_DTYPE)),
                        0, 0};
                    DataCopyPadExtParams<KV_DTYPE> padYParams{false, 0, 0, 0};
                    int64_t nYOffset = n * this->tilingData_->seqLength * this->V_LENGTH + i * this->V_LENGTH;
                    DataCopyPad(yLocal, vGm[nYOffset], copyYParams, padYParams);
                    this->inQueueV.EnQue(yLocal);
                    yLocal = this->inQueueV.template DeQue<KV_DTYPE>();

                    LocalTensor<KV_DTYPE> outLocalV = this->outQueueV.template AllocTensor<KV_DTYPE>();
                    LocalTensor<int8_t> outLocalVQuant =
                        outLocalV.template ReinterpretCast<int8_t>()[this->tilingData_->rowsPerBlock * this->V_LENGTH *
                                                                     sizeof(KV_DTYPE)];
                    DataCopy(outLocalV, yLocal, this->tilingData_->rowsPerBlock * this->V_LENGTH);
                    SetWaitFlag<HardEvent::V_V>(HardEvent::V_V);

                    if (this->tilingData_->isVQuant) {
                        Cast(workspaceBuffer, yLocal, RoundMode::CAST_NONE,
                             this->tilingData_->rowsPerBlock * this->V_LENGTH);
                        PipeBarrier<PIPE_V>();
                        Quant<KV_DTYPE>(outLocalVQuant, workspaceBuffer, vScaleLocal, vOffsetLocal, isVOffset,
                                        this->V_LENGTH);
                    }
                    this->inQueueV.FreeTensor(yLocal);
                    this->inQueueScaleOffset.FreeTensor(kScaleLocal);
                    this->outQueueV.EnQue(outLocalV);
                    outLocalV = this->outQueueV.template DeQue<KV_DTYPE>();
                    // ScatterUpdate
                    if (this->tilingData_->isVQuant) {
                        ScatterUpdate<int8_t, KV_DTYPE>((GlobalTensor<int8_t> &)this->vCacheGm, this->vCacheGmNd,
                                                        outLocalV, outLocalVQuant, n, i, this->V_LENGTH);
                    } else {
                        LocalTensor<V_DTYPE> outputLocal = outLocalV.template ReinterpretCast<V_DTYPE>();
                        ScatterUpdate<V_DTYPE, KV_DTYPE>(this->vCacheGm, this->vCacheGmNd, outLocalV, outputLocal, n, i,
                                                         this->V_LENGTH);
                    }
                    this->outQueueV.FreeTensor(outLocalV);
                }
            }
        } else {
            // exception
        }
    }

    template <typename T>
    __aicore__ inline void RopeKV2(const LocalTensor<T> &xLocal, const LocalTensor<float> &xFp32Local,
                                   const LocalTensor<T> &cosLocal, const LocalTensor<T> &sinLocal,
                                   const LocalTensor<T> &cosLocalB16, const LocalTensor<T> &sinLocalB16,
                                   const LocalTensor<int8_t> &outLocalKQuant, const LocalTensor<float> &scaleLocal,
                                   const LocalTensor<float> &offsetLocal, bool ifAsymmetricQuant, int64_t kvLength)
    {
        constexpr static int64_t BLOCK_SIZE = 32;
        // realLocal, imagLocal comes from odd/even elements in kLocal
        LocalTensor<KV_DTYPE> k = this->bufferXSquare.template Get<KV_DTYPE>();
        LocalTensor<KV_DTYPE> realLocal = k[0]; // 32 * sizeof(float)
        LocalTensor<KV_DTYPE> imagLocal =
            realLocal[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH_HALF * NUM_TWO]; // 32 * sizeof(float)
        LocalTensor<KV_DTYPE> buf_ =
            imagLocal[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH_HALF * NUM_TWO]; // 64 * sizeof(float)
        LocalTensor<KV_DTYPE> outLocal = buf_[this->tilingData_->rowsPerBlock * this->ROPE_LENGTH * NUM_TWO];

        LocalTensor<float> realLocalFp32 = realLocal.template ReinterpretCast<float>();
        LocalTensor<float> imagLocalFp32 = imagLocal.template ReinterpretCast<float>();
        LocalTensor<float> cosLocalFp32 = cosLocal.template ReinterpretCast<float>();
        LocalTensor<float> sinLocalFp32 = sinLocal.template ReinterpretCast<float>();
        LocalTensor<float> outLocalFp32 = outLocal.template ReinterpretCast<float>();
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMask(realLocalFp32, xFp32Local, 1, true, this->ROPE_LENGTH,
                   {1, static_cast<uint16_t>(this->tilingData_->rowsPerBlock), 2 * NUM_TWELVE, 0}, rsvdCnt);
        GatherMask(imagLocalFp32, xFp32Local, NUM_TWO, true, this->ROPE_LENGTH,
                   {1, static_cast<uint16_t>(this->tilingData_->rowsPerBlock), 2 * NUM_TWELVE, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();

        Cast(cosLocalFp32, cosLocalB16, RoundMode::CAST_NONE, this->tilingData_->rowsPerBlock * this->ROPE_LENGTH);
        Cast(sinLocalFp32, sinLocalB16, RoundMode::CAST_NONE, this->tilingData_->rowsPerBlock * this->ROPE_LENGTH);
        PipeBarrier<PIPE_V>();

        uint64_t mask[NUM_TWO] = {MASK_CONFIG, 0}; // mask this->ROPE_LENGTH_HALF Elements
        Mul(outLocalFp32, realLocalFp32, cosLocalFp32, mask, this->tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        Mul(outLocalFp32[this->ROPE_LENGTH_HALF], imagLocalFp32, cosLocalFp32[this->ROPE_LENGTH_HALF], mask,
            this->tilingData_->rowsPerBlock, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, this->tilingData_->rowsPerBlock * this->ROPE_LENGTH_HALF);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> bufFp32 = buf_.template ReinterpretCast<float>();

        Mul(bufFp32, imagLocalFp32, sinLocalFp32, mask, this->tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        Mul(bufFp32[this->ROPE_LENGTH_HALF], realLocalFp32, sinLocalFp32[this->ROPE_LENGTH_HALF], mask,
            this->tilingData_->rowsPerBlock, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        Add(outLocalFp32, outLocalFp32, bufFp32, this->tilingData_->rowsPerBlock * this->ROPE_LENGTH);
        PipeBarrier<PIPE_V>();

        struct DataCopyParams copyInParams(this->tilingData_->rowsPerBlock, 0, 0, 0);
        copyInParams.blockLen = this->ROPE_LENGTH * sizeof(float) / BLOCK_SIZE;
        copyInParams.dstStride = static_cast<uint16_t>(this->V_LENGTH * sizeof(float) / BLOCK_SIZE);
        DataCopy(xFp32Local, outLocalFp32, copyInParams);
        SetWaitFlag<HardEvent::V_V>(HardEvent::V_V);

        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(xLocal, outLocalFp32, RoundMode::CAST_RINT, this->ROPE_LENGTH, this->tilingData_->rowsPerBlock,
                 {1, 1, NUM_TWELVE, NUM_EIGHT});
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(xLocal, outLocalFp32, RoundMode::CAST_NONE, this->ROPE_LENGTH, this->tilingData_->rowsPerBlock,
                 {1, 1, NUM_TWELVE, NUM_EIGHT});
        }

        if (this->tilingData_->isKQuant) {
            PipeBarrier<PIPE_V>();
            Quant<KV_DTYPE>(outLocalKQuant, xFp32Local, scaleLocal, offsetLocal, ifAsymmetricQuant,
                            this->RMS_NORM_LENGTH);
        }
    }

    template <typename T>
    __aicore__ inline void RmsNormV2(const LocalTensor<float> &xFp32Local, const LocalTensor<T> &gammaLocal)
    {
        // Calc: xSquare = xFp32 * xFp32, xSquare shape is [this->tilingData_->rowsPerBlock, this->RMS_NORM_LENGTH]
        LocalTensor<float> xSquareLocal = this->bufferXSquare.template Get<float>();
        Mul(xSquareLocal, xFp32Local, xFp32Local, this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();

        // Calc: xSquare[this->tilingData_->rowsPerBlock, 192]
        //          --> [this->tilingData_->rowsPerBlock, 96]
        //          --> [this->tilingData_->rowsPerBlock, 48]
        uint64_t mask[NUM_TWO] = {UINT64_MAX, UINT64_MAX};
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_FORTY_EIGHT], NUM_FORTY_EIGHT,
            NUM_TWO * this->tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_SIX, NUM_TWELVE, NUM_TWELVE}); // [this->tilingData_->rowsPerBlock, 96]
        PipeBarrier<PIPE_V>();
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_FORTY_EIGHT], NUM_FORTY_EIGHT,
            this->tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_SIX, NUM_TWELVE, NUM_TWELVE}); // [this->tilingData_->rowsPerBlock, 48]
        PipeBarrier<PIPE_V>();

        // Calc: xSum = [this->tilingData_->rowsPerBlock, 1]
        LocalTensor<float> xSumLocal = this->bufferSum.template Get<float>();
        WholeReduceSum(xSumLocal, xSquareLocal, NUM_FORTY_EIGHT, this->tilingData_->rowsPerBlock, 1, 1, NUM_SIX);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, this->tilingData_->reciprocal, this->tilingData_->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, this->tilingData_->epsilon, this->tilingData_->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, this->tilingData_->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSquare[this->tilingData_->rowsPerBlock, 8] = brc(xSum[this->tilingData_->rowsPerBlock, 1])
        int64_t block_count = (this->tilingData_->rowsPerBlock + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        // Calc: xFp32Local = xFp32Local / xSquareLocal
        for (int64_t rowId = 0; rowId < this->tilingData_->rowsPerBlock; rowId++) {
            Div(xFp32Local[rowId * this->RMS_NORM_LENGTH], xFp32Local[rowId * this->RMS_NORM_LENGTH],
                xSquareLocal[rowId * NUM_EIGHT], mask, (this->RMS_NORM_LENGTH / NUM_SIXTY_FOUR),
                {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Cast gammaLocal to xSquareLocal (b16 -> fp32) [this->RMS_NORM_LENGTH]
        Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, this->RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();

        // Calc: xFp32Local = xFp32Local * xSquareLocal
        for (int64_t rowId = 0; rowId < this->tilingData_->rowsPerBlock; rowId++) {
            Mul(xFp32Local[rowId * this->RMS_NORM_LENGTH], xFp32Local[rowId * this->RMS_NORM_LENGTH], xSquareLocal,
                this->RMS_NORM_LENGTH);
        }
        PipeBarrier<PIPE_V>();

        // Cast xFp32 to outLocal
        LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(xLocal, xFp32Local, RoundMode::CAST_RINT, this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(xLocal, xFp32Local, RoundMode::CAST_NONE, this->tilingData_->rowsPerBlock * this->RMS_NORM_LENGTH);
        }
        this->inQueueX.EnQue(xLocal);
    }

    template <typename T1, typename T2>
    __aicore__ inline void ScatterUpdate(const GlobalTensor<T1> &dst, const GlobalTensor<T2> &dstNd,
                                         const LocalTensor<T2> &outLocal, const LocalTensor<T1> &quantLocal,
                                         int64_t nIdx, int64_t seqId, int64_t kvLength)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(kvLength * sizeof(T1)), 0, 0, 0};
        if (this->tilingData_->isOutputKv) {
            int64_t gmOffsetNd = GetBlockIdx() * this->tilingData_->rowsPerBlock * this->tilingData_->numHead *
                                     this->tilingData_->seqLength * kvLength +
                                 nIdx * this->tilingData_->seqLength * kvLength + seqId * kvLength;
            DataCopyExtParams copyParamsNd{
                static_cast<uint16_t>(this->tilingData_->rowsPerBlock), static_cast<uint32_t>(kvLength * sizeof(T2)), 0,
                static_cast<uint32_t>(((this->tilingData_->seqLength - 1) * kvLength +
                                       (this->tilingData_->numHead - 1) * this->tilingData_->seqLength * kvLength) *
                                      sizeof(T2)),
                0};
            DataCopyPad(dstNd[gmOffsetNd], outLocal, copyParamsNd);
        }
        for (int64_t i = 0; i < this->tilingData_->rowsPerBlock; ++i) {
            if constexpr (isPagedAttention) {
                int64_t tokenId = GetBlockIdx() * this->tilingData_->rowsPerBlock * this->tilingData_->seqLength +
                                  i * this->tilingData_->seqLength + seqId;
                int64_t offset = this->indexGm(tokenId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                int64_t dstOffset = offset * this->tilingData_->numHead * kvLength + nIdx * kvLength;
                DataCopyPad(dst[dstOffset], quantLocal[i * kvLength], copyParams);
            } else {
                int64_t offset = this->indexGm(i * this->tilingData_->seqLength + seqId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                int64_t dstOffset = i * this->tilingData_->numHead * this->tilingData_->cacheLength * kvLength +
                                    nIdx * this->tilingData_->cacheLength * kvLength + offset * kvLength;
                DataCopyPad(dst[dstOffset], quantLocal[i * kvLength], copyParams);
            }
        }
    }

    template <typename T>
    __aicore__ inline void Quant(const LocalTensor<int8_t> &quantOutLocal, const LocalTensor<float> &srcTensor,
                                 const LocalTensor<float> &scaleLocal, const LocalTensor<float> &offsetLocal,
                                 bool ifAsymmetricQuant, int64_t kvLength)
    {
        int64_t totalMask = kvLength;
        while (totalMask > 0) {
            int64_t offset = kvLength - totalMask;
            Mul(srcTensor[offset], srcTensor[offset], scaleLocal[offset],
                totalMask >= NUM_SIXTY_FOUR ? NUM_SIXTY_FOUR : totalMask, // mask
                this->tilingData_->rowsPerBlock,
                {1, 1, 1, // repeat
                 static_cast<uint8_t>(kvLength / NUM_EIGHT), static_cast<uint8_t>(kvLength / NUM_EIGHT), 0});
            if (ifAsymmetricQuant) {
                PipeBarrier<PIPE_V>();
                Add(srcTensor[offset], srcTensor[offset], offsetLocal[offset],
                    totalMask >= NUM_SIXTY_FOUR ? NUM_SIXTY_FOUR : totalMask, // mask
                    this->tilingData_->rowsPerBlock,
                    {1, 1, 1, // repeat
                     static_cast<uint8_t>(kvLength / NUM_EIGHT), static_cast<uint8_t>(kvLength / NUM_EIGHT), 0});
            }
            totalMask -= NUM_SIXTY_FOUR;
        }
        PipeBarrier<PIPE_V>();
        RoundFloat2Int8(quantOutLocal, srcTensor, this->tilingData_->rowsPerBlock * kvLength);
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

private:
    int64_t ROPE_LENGTH_HALF{32};
    int64_t methodMode = 0;
    int64_t isKOffset = 0;
    int64_t isVOffset = 0;
    constexpr static int64_t NUM_SIX = 6;
    constexpr static int64_t NUM_TWELVE = 12;
    constexpr static int64_t NUM_FORTY_EIGHT = 48;
    TQue<QuePosition::VECIN, 1> inQueueScaleOffset;
    TQue<QuePosition::VECIN, 1> CosSin;
    GlobalTensor<KV_DTYPE> vGm;
};
} // namespace KvRmsNormRopeCache

#endif // _KV_RMS_NORM_ROPE_CACHE_B16_MTP_QUANT_H_
