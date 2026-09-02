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
 * \file rope_with_sin_cos_cache_f_bf16.h
 * \brief rope_with_sin_cos_cache_f_bf16.h
 */

#ifndef ROPE_WITH_SIN_COS_CACHE_f_BF16_H
#define ROPE_WITH_SIN_COS_CACHE_f_BF16_H

#include "rope_with_sin_cos_cache_base.h"

namespace RopeWithSinCosCache {
using namespace AscendC;

template <typename T>
class RopeWithSinCosCacheFP16 : public RopeWithSinCosCacheBase<T> {
public:
    __aicore__ inline RopeWithSinCosCacheFP16(){};
    __aicore__ inline void Init(GM_ADDR position_id, GM_ADDR query_in, GM_ADDR key_in, GM_ADDR cos_sin_cache,
                                GM_ADDR query_out, GM_ADDR key_out, const RopeWithSinCosCacheTilingData &tiling_data,
                                TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void Compute(uint64_t index, uint64_t loopN);
    __aicore__ inline void ComputeAlongHeads(uint64_t indexToken, uint64_t indexHeads);
    __aicore__ inline void GetCosSinCache(LocalTensor<T> inQueueCosSinCacheBeforeCastLocal,
                                          LocalTensor<T> copyBuf0Local, LocalTensor<float> inCosSin,
                                          LocalTensor<float> CosSin, uint64_t offsetPos, uint64_t cosSinOffset,
                                          uint64_t localStartAddr, uint32_t (&rcShape_)[2], uint32_t (&dstShape_)[2]);

protected:
    static constexpr uint64_t BLOCK_SIZE = 32;
    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t ELE_NUM_FP32 = 8;
    static constexpr uint64_t ELE_NUM_FP16 = 16;
    static constexpr uint64_t MASK = 64;
    uint64_t blockOffset;
    uint16_t headBlockLen{0};
    uint16_t rotaryBlockLen{0};
    uint16_t calBlockLen{0};
    uint16_t calBlockLenFP16{0};
    uint64_t q_size;
    uint64_t k_size;
    uint64_t num_heads_max;

    GlobalTensor<uint64_t> position_id_GM;
    GlobalTensor<T> query_in_GM;
    GlobalTensor<T> key_in_GM;
    GlobalTensor<T> cos_sin_cache_GM;

    GlobalTensor<T> queryGM;
    GlobalTensor<T> keyGM;

    TBuf<TPosition::VECCALC> inQQue, inQueueCosSinCache;
    TBuf<TPosition::VECCALC> outQue;
    TBuf<TPosition::VECCALC> reverseBuf, negOneBuf, cosSinBuf, temp1, offsetBuf, inQueCalBuf;
    TBuf<TPosition::VECCALC> copyBuf0, copyBuf1, copyBuf2;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQQueBeforeCast, inQueueCosSinCacheBeforeCast;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueAfterCast;
};

template <typename T>
__aicore__ inline void RopeWithSinCosCacheFP16<T>::Init(GM_ADDR position_id, GM_ADDR query_in, GM_ADDR key_in,
                                                        GM_ADDR cos_sin_cache, GM_ADDR query_out, GM_ADDR key_out,
                                                        const RopeWithSinCosCacheTilingData &tiling_data, TPipe *pipe)
{
    this->InitData(tiling_data);
    headBlockLen = static_cast<uint16_t>(this->head_size / ELE_NUM_FP32);
    rotaryBlockLen = static_cast<uint16_t>(this->rotary_dim / ELE_NUM_FP32);
    calBlockLenFP16 = static_cast<uint16_t>(this->rotary_dim / 2 / ELE_NUM_FP16);
    calBlockLen = rotaryBlockLen / 2;

    if (this->blockIdx_ < this->front_core) {
        blockOffset = this->num_tokens_each_front_core * this->blockIdx_;
    } else {
        blockOffset = this->num_tokens_each_front_core * (this->front_core) +
                      (this->blockIdx_ - this->front_core) * this->num_tokens_each_tail_core;
    }

    position_id_GM.SetGlobalBuffer((__gm__ uint64_t *)position_id + blockOffset);
    query_in_GM.SetGlobalBuffer((__gm__ T *)query_in + blockOffset * this->q_leading_dimension);
    key_in_GM.SetGlobalBuffer((__gm__ T *)key_in + blockOffset * this->k_leading_dimension);
    cos_sin_cache_GM.SetGlobalBuffer((__gm__ T *)cos_sin_cache);

    queryGM.SetGlobalBuffer((__gm__ T *)query_out + blockOffset * this->num_q_heads * this->head_size);
    keyGM.SetGlobalBuffer((__gm__ T *)key_out + blockOffset * this->num_kv_heads * this->head_size);

    num_heads_max = (this->num_q_heads > this->num_kv_heads) ? this->num_q_heads : this->num_kv_heads;
    if (this->loop_for_one_token == 0) {
        pipe->InitBuffer(inQQue,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(float));
        pipe->InitBuffer(inQueueCosSinCache, this->rotary_dim * sizeof(float));
        pipe->InitBuffer(reverseBuf,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(float));
        pipe->InitBuffer(negOneBuf,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(float));
        pipe->InitBuffer(cosSinBuf,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(float));
        pipe->InitBuffer(inQueCalBuf,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(T));

        pipe->InitBuffer(inQQueBeforeCast, BUFFER_NUM,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->head_size * sizeof(T));
        pipe->InitBuffer(inQueueCosSinCacheBeforeCast, BUFFER_NUM, this->rotary_dim * sizeof(T));
        pipe->InitBuffer(outQueAfterCast, BUFFER_NUM,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->head_size * sizeof(T));

        pipe->InitBuffer(copyBuf0, this->rotary_dim / 2 * sizeof(T));
        pipe->InitBuffer(copyBuf1, this->rotary_dim / 2 * sizeof(T));
        pipe->InitBuffer(copyBuf2, this->rotary_dim / 2 * sizeof(T));

        if (this->is_neox_style == 0) {
            pipe->InitBuffer(offsetBuf, this->rotary_dim * sizeof(uint32_t));
            pipe->InitBuffer(
                temp1, this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(float));
        } else {
            pipe->InitBuffer(offsetBuf, 0 * sizeof(uint32_t));
            pipe->InitBuffer(temp1, 0 * sizeof(float));
        }
    } else {
        pipe->InitBuffer(inQQue, this->num_heads_each_loop * this->rotary_dim * sizeof(float));
        pipe->InitBuffer(inQueueCosSinCache, this->rotary_dim * sizeof(float));
        pipe->InitBuffer(reverseBuf, this->num_heads_each_loop * this->rotary_dim * sizeof(float));
        pipe->InitBuffer(negOneBuf, this->num_heads_each_loop * this->rotary_dim * sizeof(float));
        pipe->InitBuffer(cosSinBuf, this->num_heads_each_loop * this->rotary_dim * sizeof(float));
        pipe->InitBuffer(inQueCalBuf, this->num_heads_each_loop * this->rotary_dim * sizeof(T));

        pipe->InitBuffer(inQQueBeforeCast, BUFFER_NUM, this->num_heads_each_loop * this->head_size * sizeof(T));
        pipe->InitBuffer(inQueueCosSinCacheBeforeCast, BUFFER_NUM, this->rotary_dim * sizeof(T));
        pipe->InitBuffer(outQueAfterCast, BUFFER_NUM, this->num_heads_each_loop * this->head_size * sizeof(T));

        pipe->InitBuffer(copyBuf0, this->rotary_dim / 2 * sizeof(T));
        pipe->InitBuffer(copyBuf1, this->rotary_dim / 2 * sizeof(T));
        pipe->InitBuffer(copyBuf2, this->rotary_dim / 2 * sizeof(T));

        if (this->is_neox_style == 0) {
            pipe->InitBuffer(offsetBuf, this->rotary_dim * sizeof(uint32_t));
            pipe->InitBuffer(temp1, this->num_heads_each_loop * this->rotary_dim * sizeof(float));
        } else {
            pipe->InitBuffer(offsetBuf, 0 * sizeof(uint32_t));
            pipe->InitBuffer(temp1, 0 * sizeof(float));
        }
    }
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheFP16<T>::GetCosSinCache(LocalTensor<T> inQueueCosSinCacheBeforeCastLocal,
                                                                  LocalTensor<T> copyBuf0Local,
                                                                  LocalTensor<float> inCosSin,
                                                                  LocalTensor<float> cosSin, uint64_t offsetPos,
                                                                  uint64_t cosSinOffset, uint64_t localStartAddr,
                                                                  uint32_t (&srcShape_)[2], uint32_t (&dstShape_)[2])
{
    if (this->mrope_section0 > 0) {
        // dataCopy和Copy的组合方式解决起始地址对齐和搬运单元对齐的问题
        uint64_t pos0 = position_id_GM.GetValue(offsetPos);
        uint64_t pos1 = position_id_GM.GetValue(offsetPos + this->num_tokens);
        uint64_t pos2 = position_id_GM.GetValue(offsetPos + 2 * this->num_tokens);
        if (this->cacheMode == 0) {
            if (this->mrope_section3 == 0) {
                uint8_t padding2 = ((this->mrope_section0 + this->mrope_section1) * sizeof(T) % BLOCK_SIZE) / sizeof(T);
                this->VToMTE2Sync();
                DataCopyPad(
                    inQueueCosSinCacheBeforeCastLocal[this->mrope_section0 +
                                                      static_cast<uint16_t>(this->mrope_section1) - padding2],
                    cos_sin_cache_GM[pos2 * this->rotary_dim +
                                     static_cast<uint16_t>(this->mrope_section0 + this->mrope_section1) + cosSinOffset],
                    {1, static_cast<uint16_t>(this->mrope_section2 * sizeof(T)), 0, 0}, {true, padding2, 0, 0});
            } else if (this->mrope_section3 > 0) {
                // 第四段
                uint64_t pos3 = position_id_GM.GetValue(offsetPos + 3 * this->num_tokens);
                uint8_t padding3 =
                    ((this->mrope_section0 + this->mrope_section1 + this->mrope_section2) * sizeof(T) % BLOCK_SIZE) /
                    sizeof(T);
                this->VToMTE2Sync();
                DataCopyPad(inQueueCosSinCacheBeforeCastLocal[static_cast<uint16_t>(this->mrope_section0 +
                                                                                    this->mrope_section1 +
                                                                                    this->mrope_section2) -
                                                              padding3],
                            cos_sin_cache_GM[pos3 * this->rotary_dim +
                                             static_cast<uint16_t>(this->mrope_section0 + this->mrope_section1 +
                                                                   this->mrope_section2) +
                                             cosSinOffset],
                            {1, static_cast<uint16_t>(this->mrope_section3 * sizeof(T)), 0, 0}, {true, padding3, 0, 0});
                // 第三段
                uint8_t padding2 = ((this->mrope_section0 + this->mrope_section1) * sizeof(T) % BLOCK_SIZE) / sizeof(T);
                DataCopyPad(
                    copyBuf0Local,
                    cos_sin_cache_GM[pos2 * this->rotary_dim +
                                     static_cast<uint16_t>(this->mrope_section0 + this->mrope_section1) + cosSinOffset],
                    {1, static_cast<uint16_t>(this->mrope_section2 * sizeof(T)), 0, 0}, {true, padding2, 0, 0});
                this->MTE2ToVSync();
                Copy(inQueueCosSinCacheBeforeCastLocal[this->mrope_section0 + this->mrope_section1 - padding2],
                     copyBuf0Local, this->mrope_section2 + padding2, 1, {1, 1, 0, 0});
                this->VToMTE2Sync();
            }
            uint8_t padding1 = (this->mrope_section0 * sizeof(T) % BLOCK_SIZE) / sizeof(T);
            DataCopyPad(
                copyBuf0Local,
                cos_sin_cache_GM[pos1 * this->rotary_dim + static_cast<uint16_t>(this->mrope_section0) + cosSinOffset],
                {1, static_cast<uint16_t>(this->mrope_section1 * sizeof(T)), 0, 0}, {true, padding1, 0, 0});
            this->MTE2ToVSync();
            Copy(inQueueCosSinCacheBeforeCastLocal[this->mrope_section0 - padding1], copyBuf0Local,
                 this->mrope_section1 + padding1, 1, {1, 1, 0, 0});
            this->VToMTE2Sync();
            DataCopyPad(copyBuf0Local, cos_sin_cache_GM[pos0 * this->rotary_dim + cosSinOffset],
                        {1, static_cast<uint16_t>(this->mrope_section0 * sizeof(T)), 0, 0}, {false, 0, 0, 0});
            this->MTE2ToVSync();
            Copy(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, this->mrope_section0, 1, {1, 1, 0, 0});
        } else if (this->cacheMode == 1) {
            // 新的处理方式
            this->VToMTE2Sync();
            DataCopyPad(inQueueCosSinCacheBeforeCastLocal, cos_sin_cache_GM[pos0 * this->rotary_dim + cosSinOffset],
                        {1, static_cast<uint16_t>(this->rotary_dim / 2 * sizeof(T)), 0, 0}, {false, 0, 0, 0});
            float pos1Value = 0;
            float pos2Value = 0;
            uint32_t i = 0;
            for (i = 0; i < this->rotary_dim / 2; i++) {
                if (i < this->mrope_section1) {
                    pos1Value = cos_sin_cache_GM.GetValue(pos1 * this->rotary_dim + cosSinOffset + i * 3 + 1);
                    inQueueCosSinCacheBeforeCastLocal.SetValue(i * 3 + 1, pos1Value);
                }
                if (i < this->mrope_section2) {
                    pos2Value = cos_sin_cache_GM.GetValue(pos2 * this->rotary_dim + cosSinOffset + i * 3 + 2);
                    inQueueCosSinCacheBeforeCastLocal.SetValue(i * 3 + 2, pos2Value);
                }
                if (i > this->mrope_section1 && i > this->mrope_section2) {
                    break;
                }
            }
        }
    } else {
        uint64_t pos = position_id_GM.GetValue(offsetPos);
        this->VToMTE2Sync();
        DataCopy(inQueueCosSinCacheBeforeCastLocal, cos_sin_cache_GM[pos * this->rotary_dim + cosSinOffset],
                 {1, calBlockLenFP16, 0, 0});
        this->MTE2ToVSync();
    }
    Cast(inCosSin, inQueueCosSinCacheBeforeCastLocal, AscendC::RoundMode::CAST_NONE,
         static_cast<uint16_t>(this->rotary_dim));
    DataCopy(inCosSin[this->rotary_dim / 2], inCosSin, {1, calBlockLen, 0, 0});
    Broadcast<float, 2, 0, false>(cosSin[localStartAddr], inCosSin, dstShape_, srcShape_);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheFP16<T>::Compute(uint64_t index, uint64_t loopN)
{
    q_size = this->num_q_heads * this->head_size;
    k_size = this->num_kv_heads * this->head_size;
    uint64_t query_in_offset = index * this->num_tokens_each_loop_current_core * this->q_leading_dimension;
    uint64_t key_in_offset = index * this->num_tokens_each_loop_current_core * this->k_leading_dimension;
    uint64_t offset = index * this->num_tokens_each_loop_current_core * q_size;
    uint64_t offsetk = index * this->num_tokens_each_loop_current_core * k_size;

    uint32_t dstShape_[2] = {static_cast<uint32_t>(this->num_heads_max), static_cast<uint32_t>(this->rotary_dim)};
    uint64_t negOneRows = loopN * this->num_heads_max;
    uint32_t dstShape_4Negone_[2] = {static_cast<uint32_t>(negOneRows - 1), static_cast<uint32_t>(this->rotary_dim)};
    uint32_t srcShape_[2] = {1, static_cast<uint32_t>(this->rotary_dim)};

    LocalTensor<T> inQQueBeforeCastLocal = inQQueBeforeCast.AllocTensor<T>();
    LocalTensor<T> inQueueCosSinCacheBeforeCastLocal = inQueueCosSinCacheBeforeCast.AllocTensor<T>();
    LocalTensor<T> outQueAfterCastLocal = outQueAfterCast.AllocTensor<T>();

    LocalTensor<float> inLocal = inQQue.Get<float>();
    LocalTensor<float> inCosSin = inQueueCosSinCache.Get<float>();
    LocalTensor<float> reverseQ = reverseBuf.Get<float>();
    LocalTensor<float> negOne = negOneBuf.Get<float>();
    LocalTensor<float> cosSin = cosSinBuf.Get<float>();
    LocalTensor<T> inQueCalLocal = inQueCalBuf.Get<T>();
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    LocalTensor<float> temp1Local = temp1.Get<float>();

    LocalTensor<T> copyBuf0Local = copyBuf0.Get<T>();
    LocalTensor<T> copyBuf1Local = copyBuf1.Get<T>();
    LocalTensor<T> copyBuf2Local = copyBuf2.Get<T>();

    if (this->is_neox_style == 0) {
        for (uint32_t i = 0; i < this->rotary_dim / 2; i++) {
            offsetLocal.SetValue(i * 2, i * 4);
            offsetLocal.SetValue(i * 2 + 1, (this->rotary_dim / 2 + i) * 4);
        }
    }

    DataCopy(inQQueBeforeCastLocal, query_in_GM[query_in_offset],
             {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_q_heads * headBlockLen / 2),
              static_cast<uint16_t>(this->q_leading_dimension / ELE_NUM_FP16 - this->q_size / ELE_NUM_FP16), 0});
    this->MTE2ToVSync();
    DataCopy(inQueCalLocal, inQQueBeforeCastLocal,
             {static_cast<uint16_t>(loopN * this->num_q_heads), static_cast<uint16_t>(rotaryBlockLen / 2),
              static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2), 0});

    if (this->is_neox_style == 0) {
        Cast(temp1Local, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
             static_cast<uint16_t>(loopN * this->num_q_heads * this->rotary_dim));
        uint64_t rsv = 0;
        for (uint32_t i = 0; i < loopN * this->num_q_heads; i++) {
            GatherMask(inLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], static_cast<uint8_t>(1), true,
                       this->rotary_dim, {1, 1, 0, 0}, rsv);
            GatherMask(inLocal[i * this->rotary_dim + this->rotary_dim / 2], temp1Local[i * this->rotary_dim],
                       static_cast<uint8_t>(2), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
        }
    } else {
        Cast(inLocal, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
             static_cast<uint16_t>(loopN * this->num_q_heads * this->rotary_dim));
    }

    DataCopy(reverseQ, inLocal[this->rotary_dim / 2],
             {static_cast<uint16_t>(loopN * this->num_q_heads), calBlockLen, calBlockLen, calBlockLen});
    DataCopy(reverseQ[this->rotary_dim / 2], inLocal,
             {static_cast<uint16_t>(loopN * this->num_q_heads), calBlockLen, calBlockLen, calBlockLen});

    float One = 1.0;
    float None = -1.0;
    Duplicate<float>(negOne, None, this->rotary_dim / 2);
    Duplicate<float>(negOne[this->rotary_dim / 2], One, this->rotary_dim / 2);
    if (negOneRows > 1) {
        Broadcast<float, 2, 0, false>(negOne[this->rotary_dim], negOne, dstShape_4Negone_, srcShape_);
    }
    uint64_t localStartAddr = 0;
    uint64_t cosSinOffset = 0;
    for (uint32_t i = 0; i < loopN; ++i) {
        uint64_t offsetPos = this->num_tokens_each_loop_current_core * index + i;
        GetCosSinCache(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset,
                       localStartAddr, srcShape_, dstShape_);
        localStartAddr += this->num_q_heads * this->rotary_dim;
    }
    Mul(inLocal, cosSin, inLocal, loopN * this->num_q_heads * this->rotary_dim);
    Mul(reverseQ, negOne, reverseQ, loopN * this->num_q_heads * this->rotary_dim);
    localStartAddr = 0;
    cosSinOffset = this->rotary_dim / 2;
    for (uint32_t i = 0; i < loopN; ++i) {
        uint64_t offsetPos = this->num_tokens_each_loop_current_core * index + i;
        GetCosSinCache(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset,
                       localStartAddr, srcShape_, dstShape_);
        localStartAddr += this->num_q_heads * this->rotary_dim;
    }
    Mul(reverseQ, cosSin, reverseQ, loopN * this->num_q_heads * this->rotary_dim);
    Add(inLocal, reverseQ, inLocal, loopN * this->num_q_heads * this->rotary_dim);

    if (this->is_neox_style == 0) {
        for (uint32_t i = 0; i < loopN * this->num_q_heads; i++) {
            Gather(temp1Local[i * this->rotary_dim], inLocal[i * this->rotary_dim], offsetLocal, (uint32_t)0,
                   this->rotary_dim);
        }

        Cast(inQueCalLocal, temp1Local, AscendC::RoundMode::CAST_RINT,
             static_cast<uint16_t>(loopN * this->num_q_heads * this->rotary_dim));
    } else {
        Cast(inQueCalLocal, inLocal, AscendC::RoundMode::CAST_RINT,
             static_cast<uint16_t>(loopN * this->num_q_heads * this->rotary_dim));
    }

    if (this->head_size != this->rotary_dim) {
        DataCopy(outQueAfterCastLocal, inQueCalLocal,
                 {static_cast<uint16_t>(loopN * this->num_q_heads), static_cast<uint16_t>(rotaryBlockLen / 2), 0,
                  static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2)});
        DataCopy(outQueAfterCastLocal[this->rotary_dim], inQQueBeforeCastLocal[this->rotary_dim],
                 {static_cast<uint16_t>(loopN * this->num_q_heads),
                  static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2),
                  static_cast<uint16_t>(rotaryBlockLen / 2), static_cast<uint16_t>(rotaryBlockLen / 2)});
    } else {
        DataCopy(outQueAfterCastLocal, inQueCalLocal,
                 {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_q_heads * headBlockLen / 2), 0, 0});
    }
    this->VToMTE3Sync();
    DataCopy(queryGM[offset], outQueAfterCastLocal,
             {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_q_heads * headBlockLen / 2), 0, 0});

    // 处理key
    this->MTE3ToVSync();
    this->VToMTE2Sync();
    DataCopy(inQQueBeforeCastLocal, key_in_GM[key_in_offset],
             {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_kv_heads * headBlockLen / 2),
              static_cast<uint16_t>(this->k_leading_dimension / ELE_NUM_FP16 - this->k_size / ELE_NUM_FP16), 0});
    this->MTE2ToVSync();
    DataCopy(inQueCalLocal, inQQueBeforeCastLocal,
             {static_cast<uint16_t>(loopN * this->num_kv_heads), static_cast<uint16_t>(rotaryBlockLen / 2),
              static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2), 0});

    if (this->is_neox_style == 0) {
        Cast(temp1Local, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
             static_cast<uint16_t>(loopN * this->num_kv_heads * this->rotary_dim));
        uint64_t rsv = 0;
        for (uint32_t i = 0; i < loopN * this->num_kv_heads; i++) {
            GatherMask(inLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], static_cast<uint8_t>(1), true,
                       this->rotary_dim, {1, 1, 0, 0}, rsv);
            GatherMask(inLocal[i * this->rotary_dim + this->rotary_dim / 2], temp1Local[i * this->rotary_dim],
                       static_cast<uint8_t>(2), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
        }
    } else {
        Cast(inLocal, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
             static_cast<uint16_t>(loopN * this->num_kv_heads * this->rotary_dim));
    }

    DataCopy(reverseQ, inLocal[this->rotary_dim / 2],
             {static_cast<uint16_t>(loopN * this->num_kv_heads), calBlockLen, calBlockLen, calBlockLen});
    DataCopy(reverseQ[this->rotary_dim / 2], inLocal,
             {static_cast<uint16_t>(loopN * this->num_kv_heads), calBlockLen, calBlockLen, calBlockLen});

    localStartAddr = 0;
    cosSinOffset = 0;
    for (uint32_t i = 0; i < loopN; ++i) {
        uint64_t offsetPos = this->num_tokens_each_loop_current_core * index + i;
        GetCosSinCache(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset,
                       localStartAddr, srcShape_, dstShape_);
        localStartAddr += this->num_kv_heads * this->rotary_dim;
    }

    Mul(inLocal, cosSin, inLocal, loopN * this->num_kv_heads * this->rotary_dim);
    Mul(reverseQ, negOne, reverseQ, loopN * this->num_kv_heads * this->rotary_dim);
    localStartAddr = 0;
    cosSinOffset = this->rotary_dim / 2;
    for (uint32_t i = 0; i < loopN; ++i) {
        uint64_t offsetPos = this->num_tokens_each_loop_current_core * index + i;
        GetCosSinCache(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset,
                       localStartAddr, srcShape_, dstShape_);
        localStartAddr += this->num_kv_heads * this->rotary_dim;
    }

    Mul(reverseQ, cosSin, reverseQ, loopN * this->num_kv_heads * this->rotary_dim);
    Add(inLocal, reverseQ, inLocal, loopN * this->num_kv_heads * this->rotary_dim);

    if (this->is_neox_style == 0) {
        for (uint32_t i = 0; i < loopN * this->num_kv_heads; i++) {
            Gather(temp1Local[i * this->rotary_dim], inLocal[i * this->rotary_dim], offsetLocal, (uint32_t)0,
                   this->rotary_dim);
        }
        Cast(inQueCalLocal, temp1Local, AscendC::RoundMode::CAST_RINT,
             static_cast<uint16_t>(loopN * this->num_kv_heads * this->rotary_dim));
    } else {
        Cast(inQueCalLocal, inLocal, AscendC::RoundMode::CAST_RINT,
             static_cast<uint16_t>(loopN * this->num_kv_heads * this->rotary_dim));
    }

    if (this->head_size != this->rotary_dim) {
        DataCopy(outQueAfterCastLocal, inQueCalLocal,
                 {static_cast<uint16_t>(loopN * this->num_kv_heads), static_cast<uint16_t>(rotaryBlockLen / 2), 0,
                  static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2)});
        DataCopy(outQueAfterCastLocal[this->rotary_dim], inQQueBeforeCastLocal[this->rotary_dim],
                 {static_cast<uint16_t>(loopN * this->num_kv_heads),
                  static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2),
                  static_cast<uint16_t>(rotaryBlockLen / 2), static_cast<uint16_t>(rotaryBlockLen / 2)});
    } else {
        DataCopy(outQueAfterCastLocal, inQueCalLocal,
                 {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_kv_heads * headBlockLen / 2), 0, 0});
    }
    this->VToMTE3Sync();
    DataCopy(keyGM[offsetk], outQueAfterCastLocal,
             {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_kv_heads * headBlockLen / 2), 0, 0});

    inQQueBeforeCast.FreeTensor(inQQueBeforeCastLocal);
    inQueueCosSinCacheBeforeCast.FreeTensor(inQueueCosSinCacheBeforeCastLocal);
    outQueAfterCast.FreeTensor(outQueAfterCastLocal);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheFP16<T>::ComputeAlongHeads(uint64_t indexToken, uint64_t indexHeads)
{
    q_size = this->num_q_heads * this->head_size;
    k_size = this->num_kv_heads * this->head_size;
    uint64_t offsetQHead = indexHeads * this->num_qheads_each_loop * this->head_size;
    uint64_t offsetKHead = indexHeads * this->num_kheads_each_loop * this->head_size;
    uint64_t offset = indexToken * q_size + offsetQHead; // 非连续?
    uint64_t offsetk = indexToken * k_size + offsetKHead;
    uint64_t query_in_offset = indexToken * this->q_leading_dimension + offsetQHead;
    uint64_t key_in_offset = indexToken * this->k_leading_dimension + offsetKHead;

    uint32_t dstShape_[2] = {static_cast<uint32_t>(this->num_heads_each_loop), static_cast<uint32_t>(this->rotary_dim)};
    uint32_t dstShape_4Negone_[2] = {static_cast<uint32_t>(this->num_heads_each_loop - 1),
                                     static_cast<uint32_t>(this->rotary_dim)};
    uint32_t srcShape_[2] = {1, static_cast<uint32_t>(this->rotary_dim)};

    LocalTensor<T> inQQueBeforeCastLocal = inQQueBeforeCast.AllocTensor<T>();
    LocalTensor<T> inQueueCosSinCacheBeforeCastLocal = inQueueCosSinCacheBeforeCast.AllocTensor<T>();
    LocalTensor<T> outQueAfterCastLocal = outQueAfterCast.AllocTensor<T>();

    LocalTensor<float> inLocal = inQQue.Get<float>();
    LocalTensor<float> inCosSin = inQueueCosSinCache.Get<float>();
    LocalTensor<float> reverseQ = reverseBuf.Get<float>();
    LocalTensor<float> negOne = negOneBuf.Get<float>();
    LocalTensor<float> cosSin = cosSinBuf.Get<float>();
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    LocalTensor<float> temp1Local = temp1.Get<float>();
    LocalTensor<T> inQueCalLocal = inQueCalBuf.Get<T>();

    LocalTensor<T> copyBuf0Local = copyBuf0.Get<T>();
    LocalTensor<T> copyBuf1Local = copyBuf1.Get<T>();
    LocalTensor<T> copyBuf2Local = copyBuf2.Get<T>();

    if (this->is_neox_style == 0) {
        for (uint32_t i = 0; i < this->rotary_dim / 2; i++) {
            offsetLocal.SetValue(i * 2, i * 4);
            offsetLocal.SetValue(i * 2 + 1, (this->rotary_dim / 2 + i) * 4);
        }
    }

    if (indexHeads < this->loop_along_qheads) {
        uint64_t loopNQhead = (indexHeads == this->loop_along_qheads - 1 && this->num_qheads_last_loop != 0) ?
                                  this->num_qheads_last_loop :
                                  this->num_qheads_each_loop;
        DataCopy(inQQueBeforeCastLocal, query_in_GM[query_in_offset],
                 {1, static_cast<uint16_t>(loopNQhead * headBlockLen / 2), 0, 0});
        this->MTE2ToVSync();
        DataCopy(inQueCalLocal, inQQueBeforeCastLocal,
                 {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(rotaryBlockLen / 2),
                  static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2), 0});
        if (this->is_neox_style == 0) {
            Cast(temp1Local, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
                 static_cast<uint16_t>(loopNQhead * this->rotary_dim));

            uint64_t rsv = 0;
            for (uint32_t i = 0; i < loopNQhead; i++) {
                GatherMask(inLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], static_cast<uint8_t>(1),
                           true, this->rotary_dim, {1, 1, 0, 0}, rsv);
                GatherMask(inLocal[i * this->rotary_dim + this->rotary_dim / 2], temp1Local[i * this->rotary_dim],
                           static_cast<uint8_t>(2), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
            }
        } else {
            Cast(inLocal, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
                 static_cast<uint16_t>(loopNQhead * this->rotary_dim));
        }

        DataCopy(reverseQ, inLocal[this->rotary_dim / 2],
                 {static_cast<uint16_t>(loopNQhead), calBlockLen, calBlockLen, calBlockLen});
        DataCopy(reverseQ[this->rotary_dim / 2], inLocal,
                 {static_cast<uint16_t>(loopNQhead), calBlockLen, calBlockLen, calBlockLen});

        float One = 1.0;
        float None = -1.0;
        Duplicate<float>(negOne, None, this->rotary_dim / 2);
        Duplicate<float>(negOne[this->rotary_dim / 2], One, this->rotary_dim / 2);
        if (this->num_heads_each_loop > 1) {
            Broadcast<float, 2, 0, false>(negOne[this->rotary_dim], negOne, dstShape_4Negone_, srcShape_);
        }

        uint64_t offsetPos = indexToken;
        uint64_t cosSinOffset = 0;
        uint64_t localStartAddr = 0;
        GetCosSinCache(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset,
                       localStartAddr, srcShape_, dstShape_);
        Mul(inLocal, cosSin, inLocal, loopNQhead * this->rotary_dim);
        Mul(reverseQ, negOne, reverseQ, loopNQhead * this->rotary_dim);

        offsetPos = indexToken;
        cosSinOffset = this->rotary_dim / 2;
        localStartAddr = 0;
        GetCosSinCache(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset,
                       localStartAddr, srcShape_, dstShape_);
        Mul(reverseQ, cosSin, reverseQ, loopNQhead * this->rotary_dim);
        Add(inLocal, reverseQ, inLocal, loopNQhead * this->rotary_dim);

        if (this->is_neox_style == 0) {
            for (uint32_t i = 0; i < loopNQhead; i++) {
                Gather(temp1Local[i * this->rotary_dim], inLocal[i * this->rotary_dim], offsetLocal, (uint32_t)0,
                       this->rotary_dim);
            }
            Cast(inQueCalLocal, temp1Local, AscendC::RoundMode::CAST_RINT,
                 static_cast<uint16_t>(loopNQhead * this->rotary_dim));
        } else {
            Cast(inQueCalLocal, inLocal, AscendC::RoundMode::CAST_RINT,
                 static_cast<uint16_t>(loopNQhead * this->rotary_dim));
        }

        if (this->head_size != this->rotary_dim) {
            DataCopy(outQueAfterCastLocal, inQueCalLocal,
                     {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(rotaryBlockLen / 2), 0,
                      static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2)});
            DataCopy(outQueAfterCastLocal[this->rotary_dim], inQQueBeforeCastLocal[this->rotary_dim],
                     {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2),
                      static_cast<uint16_t>(rotaryBlockLen / 2), static_cast<uint16_t>(rotaryBlockLen / 2)});
        } else {
            DataCopy(outQueAfterCastLocal, inQueCalLocal,
                     {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(headBlockLen / 2), 0, 0});
        }
        this->VToMTE3Sync();
        DataCopy(queryGM[offset], outQueAfterCastLocal,
                 {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(headBlockLen / 2), 0, 0});
    }
    // key
    this->MTE3ToVSync();
    this->VToMTE2Sync();
    if (indexHeads < this->loop_along_kheads) {
        uint64_t loopNKhead = (indexHeads == this->loop_along_kheads - 1 && this->num_kheads_last_loop != 0) ?
                                  this->num_kheads_last_loop :
                                  this->num_kheads_each_loop;
        DataCopy(inQQueBeforeCastLocal, key_in_GM[key_in_offset],
                 {1, static_cast<uint16_t>(loopNKhead * headBlockLen / 2), 0, 0});
        this->MTE2ToVSync();
        DataCopy(inQueCalLocal, inQQueBeforeCastLocal,
                 {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(rotaryBlockLen / 2),
                  static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2), 0});
        if (this->is_neox_style == 0) {
            Cast(temp1Local, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
                 static_cast<uint16_t>(loopNKhead * this->rotary_dim));

            uint64_t rsv = 0;
            for (uint32_t i = 0; i < loopNKhead; i++) {
                GatherMask(inLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], static_cast<uint8_t>(1),
                           true, this->rotary_dim, {1, 1, 0, 0}, rsv);
                GatherMask(inLocal[i * this->rotary_dim + this->rotary_dim / 2], temp1Local[i * this->rotary_dim],
                           static_cast<uint8_t>(2), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
            }
        } else {
            Cast(inLocal, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
                 static_cast<uint16_t>(loopNKhead * this->rotary_dim));
        }

        DataCopy(reverseQ, inLocal[this->rotary_dim / 2],
                 {static_cast<uint16_t>(loopNKhead), calBlockLen, calBlockLen, calBlockLen});
        DataCopy(reverseQ[this->rotary_dim / 2], inLocal,
                 {static_cast<uint16_t>(loopNKhead), calBlockLen, calBlockLen, calBlockLen});

        uint64_t offsetPos = indexToken;
        uint64_t cosSinOffset = 0;
        uint64_t localStartAddr = 0;
        GetCosSinCache(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset,
                       localStartAddr, srcShape_, dstShape_);
        Mul(inLocal, cosSin, inLocal, loopNKhead * this->rotary_dim);
        Mul(reverseQ, negOne, reverseQ, loopNKhead * this->rotary_dim);

        offsetPos = indexToken;
        cosSinOffset = this->rotary_dim / 2;
        localStartAddr = 0;
        GetCosSinCache(inQueueCosSinCacheBeforeCastLocal, copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset,
                       localStartAddr, srcShape_, dstShape_);
        Mul(reverseQ, cosSin, reverseQ, loopNKhead * this->rotary_dim);
        Add(inLocal, reverseQ, inLocal, loopNKhead * this->rotary_dim);

        if (this->is_neox_style == 0) {
            for (uint32_t i = 0; i < loopNKhead; i++) {
                Gather(temp1Local[i * this->rotary_dim], inLocal[i * this->rotary_dim], offsetLocal, (uint32_t)0,
                       this->rotary_dim);
            }
            Cast(inQueCalLocal, temp1Local, AscendC::RoundMode::CAST_RINT,
                 static_cast<uint16_t>(loopNKhead * this->rotary_dim));
        } else {
            Cast(inQueCalLocal, inLocal, AscendC::RoundMode::CAST_RINT,
                 static_cast<uint16_t>(loopNKhead * this->rotary_dim));
        }

        if (this->head_size != this->rotary_dim) {
            DataCopy(outQueAfterCastLocal, inQueCalLocal,
                     {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(rotaryBlockLen / 2), 0,
                      static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2)});
            DataCopy(outQueAfterCastLocal[this->rotary_dim], inQQueBeforeCastLocal[this->rotary_dim],
                     {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2),
                      static_cast<uint16_t>(rotaryBlockLen / 2), static_cast<uint16_t>(rotaryBlockLen / 2)});
        } else {
            DataCopy(outQueAfterCastLocal, inQueCalLocal,
                     {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(headBlockLen / 2), 0, 0});
        }
        this->VToMTE3Sync();
        DataCopy(keyGM[offsetk], outQueAfterCastLocal,
                 {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(headBlockLen / 2), 0, 0});
    }

    inQQueBeforeCast.FreeTensor(inQQueBeforeCastLocal);
    inQueueCosSinCacheBeforeCast.FreeTensor(inQueueCosSinCacheBeforeCastLocal);
    outQueAfterCast.FreeTensor(outQueAfterCastLocal);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheFP16<T>::Process()
{
    if (this->loop_for_one_token == 0) {
        for (uint64_t n = 0; n < this->loop_time_current_core - 1; n++) {
            Compute(n, this->num_tokens_each_loop_current_core);
        }
        if (this->num_tokens_last_loop_current_core == 0) {
            Compute(this->loop_time_current_core - 1, this->num_tokens_each_loop_current_core);
        } else {
            Compute(this->loop_time_current_core - 1, this->num_tokens_last_loop_current_core);
        }
    } else {
        uint64_t loop_along_heads =
            this->loop_along_qheads > this->loop_along_kheads ? this->loop_along_qheads : this->loop_along_kheads;
        for (uint64_t n = 0; n < this->num_tokens_current_core; n++) {
            for (uint64_t m = 0; m < loop_along_heads; m++) {
                ComputeAlongHeads(n, m);
            }
        }
    }
}
} // namespace RopeWithSinCosCache

#endif
