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
 * \file rope_with_sin_cos_cache_fp32_arch35.h
 * \brief rope_with_sin_cos_cache_fp32.h
 */

#ifndef ROPE_WITH_SIN_COS_CACHE_ARCH35_FP32_H
#define ROPE_WITH_SIN_COS_CACHE_ARCH35_FP32_H

#include "rope_with_sin_cos_cache_base_arch35.h"

namespace RopeWithSinCosCache {
using namespace AscendC;

template <typename T>
class RopeWithSinCosCacheF32 : public RopeWithSinCosCacheBase<T> {
public:
    __aicore__ inline RopeWithSinCosCacheF32(){};
    __aicore__ inline void Init(GM_ADDR position_id, GM_ADDR query_in, GM_ADDR key_in, GM_ADDR cos_sin_cache,
                                GM_ADDR query_out, GM_ADDR key_out, const RopeWithSinCosCacheTilingData &tiling_data,
                                TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void Compute(uint64_t index, uint64_t loopN);
    __aicore__ inline void ComputeAlongHeads(uint64_t indexToken, uint64_t indexHeads);
    __aicore__ inline void GetCosSinCache(LocalTensor<T> copyBuf0Local, LocalTensor<float> inCosSin,
                                          LocalTensor<float> CosSin, uint64_t offsetPos, uint64_t cosSinOffset,
                                          uint64_t localStartAddr, uint32_t (&rcShape_)[2], uint32_t (&dstShape_)[2]);

protected:
    static constexpr uint64_t BLOCK_SIZE = 32;
    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t ELE_NUM_FP32 = 8;
    static constexpr uint64_t MASK = 64;
    static constexpr uint64_t MROPE_SECTION_BASE_SIZE = 8;
    uint64_t blockOffset;
    uint16_t headBlockLen{0};
    uint16_t rotaryBlockLen{0};
    uint16_t calBlockLen{0};
    uint64_t q_size;
    uint64_t k_size;
    uint64_t num_heads_max;

    GlobalTensor<uint64_t> position_id_GM;
    GlobalTensor<T> query_in_GM;
    GlobalTensor<T> key_in_GM;
    GlobalTensor<T> cos_sin_cache_GM;

    GlobalTensor<T> queryGM;
    GlobalTensor<T> keyGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQQue, inQueueCosSinCache;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> reverseBuf, negOneBuf, cosSinBuf, temp1, offsetBuf, inQueCalBuf;
    TBuf<QuePosition::VECCALC> copyBuf0;
};

template <typename T>
__aicore__ inline void RopeWithSinCosCacheF32<T>::Init(GM_ADDR position_id, GM_ADDR query_in, GM_ADDR key_in,
                                                       GM_ADDR cos_sin_cache, GM_ADDR query_out, GM_ADDR key_out,
                                                       const RopeWithSinCosCacheTilingData &tiling_data, TPipe *pipe)
{
    this->InitData(tiling_data);
    headBlockLen = static_cast<uint16_t>(this->head_size / ELE_NUM_FP32);
    rotaryBlockLen = static_cast<uint16_t>(this->rotary_dim / ELE_NUM_FP32);
    calBlockLen = rotaryBlockLen / 2;
    num_heads_max = (this->num_q_heads > this->num_kv_heads) ? this->num_q_heads : this->num_kv_heads;

    if (this->blockIdx_ < this->front_core) {
        blockOffset = this->num_tokens_each_front_core * this->blockIdx_;
    } else {
        blockOffset = this->num_tokens_each_front_core * (this->front_core) +
                      (this->blockIdx_ - this->front_core) * this->num_tokens_each_tail_core;
    }

    position_id_GM.SetGlobalBuffer((__gm__ uint64_t *)position_id + blockOffset);
    query_in_GM.SetGlobalBuffer((__gm__ T *)query_in + blockOffset * this->num_q_heads * this->head_size);
    key_in_GM.SetGlobalBuffer((__gm__ T *)key_in + blockOffset * this->num_kv_heads * this->head_size);
    cos_sin_cache_GM.SetGlobalBuffer((__gm__ T *)cos_sin_cache);

    queryGM.SetGlobalBuffer((__gm__ T *)query_out + blockOffset * this->num_q_heads * this->head_size);
    keyGM.SetGlobalBuffer((__gm__ T *)key_out + blockOffset * this->num_kv_heads * this->head_size);

    pipe->InitBuffer(copyBuf0, this->rotary_dim / 2 * sizeof(T));
    if (this->loop_for_one_token == 0) {
        pipe->InitBuffer(inQQue, BUFFER_NUM,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->head_size * sizeof(T));
        pipe->InitBuffer(inQueueCosSinCache, BUFFER_NUM,
                         this->num_tokens_each_loop_current_core * this->rotary_dim * sizeof(T));
        pipe->InitBuffer(outQue, BUFFER_NUM,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->head_size * sizeof(T));
        pipe->InitBuffer(reverseBuf,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(T));
        pipe->InitBuffer(negOneBuf,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(T));
        pipe->InitBuffer(cosSinBuf,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(T));
        pipe->InitBuffer(inQueCalBuf,
                         this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(T));

        if (this->is_neox_style == 0) {
            pipe->InitBuffer(temp1,
                             this->num_tokens_each_loop_current_core * num_heads_max * this->rotary_dim * sizeof(T));
            pipe->InitBuffer(offsetBuf, this->rotary_dim * sizeof(uint32_t));
        } else {
#ifndef __CCE_KT_TEST__
            pipe->InitBuffer(temp1, 0 * sizeof(T));
            pipe->InitBuffer(offsetBuf, 0 * sizeof(uint32_t));
#else
            pipe->InitBuffer(temp1, 1 * sizeof(T));
            pipe->InitBuffer(offsetBuf, 1 * sizeof(uint32_t));
#endif
        }
    } else {
        pipe->InitBuffer(inQQue, BUFFER_NUM, this->num_heads_each_loop * this->head_size * sizeof(T));
        pipe->InitBuffer(inQueueCosSinCache, BUFFER_NUM, this->rotary_dim * sizeof(T));
        pipe->InitBuffer(outQue, BUFFER_NUM, this->num_heads_each_loop * this->head_size * sizeof(T));
        pipe->InitBuffer(reverseBuf, this->num_heads_each_loop * this->rotary_dim * sizeof(T));
        pipe->InitBuffer(negOneBuf, this->num_heads_each_loop * this->rotary_dim * sizeof(T));
        pipe->InitBuffer(cosSinBuf, this->num_heads_each_loop * this->rotary_dim * sizeof(T));
        pipe->InitBuffer(inQueCalBuf, this->num_heads_each_loop * this->rotary_dim * sizeof(T));

        if (this->is_neox_style == 0) {
            pipe->InitBuffer(temp1, this->num_heads_each_loop * this->rotary_dim * sizeof(T));
            pipe->InitBuffer(offsetBuf, this->rotary_dim * sizeof(uint32_t));
        } else {
#ifndef __CCE_KT_TEST__
            pipe->InitBuffer(temp1, 0 * sizeof(T));
            pipe->InitBuffer(offsetBuf, 0 * sizeof(uint32_t));
#else
            pipe->InitBuffer(temp1, 1 * sizeof(T));
            pipe->InitBuffer(offsetBuf, 1 * sizeof(uint32_t));
#endif
        }
    }
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheF32<T>::GetCosSinCache(LocalTensor<T> copyBuf0Local,
                                                                 LocalTensor<float> inCosSin, LocalTensor<float> cosSin,
                                                                 uint64_t offsetPos, uint64_t cosSinOffset,
                                                                 uint64_t localStartAddr, uint32_t (&srcShape_)[2],
                                                                 uint32_t (&dstShape_)[2])
{
    if (this->mrope_section0 > 0) {
        uint64_t pos0 = position_id_GM.GetValue(offsetPos);
        uint64_t pos1 = position_id_GM.GetValue(offsetPos + this->num_tokens);
        uint64_t pos2 = position_id_GM.GetValue(offsetPos + 2 * this->num_tokens);

        uint8_t padding2 = ((this->mrope_section0 + this->mrope_section1) * sizeof(T) % BLOCK_SIZE) / sizeof(T);
        DataCopyPad(inCosSin[this->mrope_section0 + static_cast<uint16_t>(this->mrope_section1) - padding2],
                    cos_sin_cache_GM[pos2 * this->rotary_dim +
                                     static_cast<uint16_t>(this->mrope_section0 + this->mrope_section1) + cosSinOffset],
                    {1, static_cast<uint16_t>(this->mrope_section2 * sizeof(T)), 0, 0}, {true, padding2, 0, 0});
        PipeBarrier<PIPE_ALL>();

        // 如果mrope_section不是32B对齐，DataCopyPad接口会自动填充，同时为了兼容旧版16 24 24做法，避免该场景性能劣化
        uint8_t padding1 = (this->mrope_section0 * sizeof(T) % BLOCK_SIZE) / sizeof(T);
        if (this->mrope_section1 % MROPE_SECTION_BASE_SIZE == 0) {
            DataCopyPad(
                inCosSin[this->mrope_section0 - padding1],
                cos_sin_cache_GM[pos1 * this->rotary_dim + static_cast<uint16_t>(this->mrope_section0) + cosSinOffset],
                {1, static_cast<uint16_t>(this->mrope_section1 * sizeof(float)), 0, 0}, {true, padding1, 0, 0});
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopyPad(
                copyBuf0Local,
                cos_sin_cache_GM[pos1 * this->rotary_dim + static_cast<uint16_t>(this->mrope_section0) + cosSinOffset],
                {1, static_cast<uint16_t>(this->mrope_section1 * sizeof(T)), 0, 0}, {true, padding1, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Copy(inCosSin[this->mrope_section0 - padding1], copyBuf0Local, this->mrope_section1 + padding1, 1,
                 {1, 1, 0, 0});
        }
        if (this->mrope_section0 % MROPE_SECTION_BASE_SIZE == 0) {
            DataCopyPad(inCosSin, cos_sin_cache_GM[pos0 * this->rotary_dim + cosSinOffset],
                        {1, static_cast<uint16_t>(this->mrope_section0 * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopyPad(copyBuf0Local, cos_sin_cache_GM[pos0 * this->rotary_dim + cosSinOffset],
                        {1, static_cast<uint16_t>(this->mrope_section0 * sizeof(T)), 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Copy(inCosSin, copyBuf0Local, this->mrope_section0, 1, {1, 1, 0, 0});
        }
    } else {
        uint64_t pos = position_id_GM.GetValue(offsetPos);
        PipeBarrier<PIPE_ALL>();
        DataCopy(inCosSin, cos_sin_cache_GM[pos * this->rotary_dim + cosSinOffset], {1, calBlockLen, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

    DataCopy(inCosSin[this->rotary_dim / 2], inCosSin, {1, calBlockLen, 0, 0});
    PipeBarrier<PIPE_ALL>();
    Broadcast<float, 2, 0, false>(cosSin[localStartAddr], inCosSin, dstShape_, srcShape_);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheF32<T>::Compute(uint64_t index, uint64_t loopN)
{
    q_size = this->num_q_heads * this->head_size;
    k_size = this->num_kv_heads * this->head_size;
    uint64_t offset = index * this->num_tokens_each_loop_current_core * q_size;
    uint64_t offsetk = index * this->num_tokens_each_loop_current_core * k_size;
    uint64_t offsetCosSin = index * loopN * this->rotary_dim;

    uint32_t dstShape_[2] = {static_cast<uint32_t>(this->num_heads_max), static_cast<uint32_t>(this->rotary_dim)};
#ifndef __CCE_KT_TEST__
    uint32_t dstShape_4Negone_[2] = {static_cast<uint32_t>(loopN * this->num_heads_max),
                                     static_cast<uint32_t>(this->rotary_dim)};
#else
    uint32_t dstShape_4Negone_[2] = {static_cast<uint32_t>(loopN * (this->num_heads_max - 1)),
                                     static_cast<uint32_t>(this->rotary_dim)};
#endif
    uint32_t srcShape_[2] = {1, static_cast<uint32_t>(this->rotary_dim)};

    LocalTensor<T> inLocal = inQQue.AllocTensor<T>();
    LocalTensor<T> inCosSin = inQueueCosSinCache.AllocTensor<T>();
    LocalTensor<T> outLocal = outQue.AllocTensor<T>();
    LocalTensor<T> reverseQ = reverseBuf.Get<T>();
    LocalTensor<T> negOne = negOneBuf.Get<T>();
    LocalTensor<T> cosSin = cosSinBuf.Get<T>();
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    LocalTensor<T> temp1Local = temp1.Get<T>();
    LocalTensor<T> inQueCalLocal = inQueCalBuf.Get<T>();
    LocalTensor<T> copyBuf0Local = copyBuf0.Get<T>();

    if (this->is_neox_style == 0) {
        for (uint32_t i = 0; i < this->rotary_dim / 2; i++) {
            offsetLocal.SetValue(i * 2, i * 4);
            offsetLocal.SetValue(i * 2 + 1, (this->rotary_dim / 2 + i) * 4);
        }
    }
    DataCopy(inLocal, query_in_GM[offset],
             {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_q_heads * headBlockLen),
              static_cast<uint16_t>(this->q_leading_dimension / ELE_NUM_FP32 - this->q_size / ELE_NUM_FP32), 0});
    this->MTE2ToVSync();

    if (this->is_neox_style == 0) {
        DataCopy(temp1Local, inLocal,
                 {static_cast<uint16_t>(loopN * this->num_q_heads), static_cast<uint16_t>(rotaryBlockLen),
                  static_cast<uint16_t>(headBlockLen - rotaryBlockLen), 0});

        uint64_t rsv = 0;
        for (uint32_t i = 0; i < loopN * this->num_q_heads; i++) {
            GatherMask(inQueCalLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], static_cast<uint8_t>(1),
                       true, this->rotary_dim, {1, 1, 0, 0}, rsv);
            GatherMask(inQueCalLocal[i * this->rotary_dim + this->rotary_dim / 2], temp1Local[i * this->rotary_dim],
                       static_cast<uint8_t>(2), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
        }
    } else {
        DataCopy(inQueCalLocal, inLocal,
                 {static_cast<uint16_t>(loopN * this->num_q_heads), static_cast<uint16_t>(rotaryBlockLen),
                  static_cast<uint16_t>(headBlockLen - rotaryBlockLen), 0});
    }

    DataCopy(reverseQ, inQueCalLocal[this->rotary_dim / 2],
             {static_cast<uint16_t>(loopN * this->num_q_heads), calBlockLen, calBlockLen, calBlockLen});
    DataCopy(reverseQ[this->rotary_dim / 2], inQueCalLocal,
             {static_cast<uint16_t>(loopN * this->num_q_heads), calBlockLen, calBlockLen, calBlockLen});

    float One = 1.0;
    float None = -1.0;
    Duplicate<float>(negOne, None, this->rotary_dim / 2);
    Duplicate<float>(negOne[this->rotary_dim / 2], One, this->rotary_dim / 2);

    Broadcast<float, 2, 0, false>(negOne[this->rotary_dim], negOne, dstShape_4Negone_, srcShape_);
    uint64_t localStartAddr = 0;
    uint64_t cosSinOffset = 0;
    for (uint32_t i = 0; i < loopN; ++i) {
        PipeBarrier<PIPE_ALL>();
        uint64_t offsetPos = this->num_tokens_each_loop_current_core * index + i;
        GetCosSinCache(copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset, localStartAddr, srcShape_, dstShape_);
        localStartAddr += this->num_q_heads * this->rotary_dim;
    }
    Mul(inQueCalLocal, cosSin, inQueCalLocal, loopN * this->num_q_heads * this->rotary_dim);
    Mul(reverseQ, negOne, reverseQ, loopN * this->num_q_heads * this->rotary_dim);
    localStartAddr = 0;
    cosSinOffset = this->rotary_dim / 2;
    for (uint32_t i = 0; i < loopN; ++i) {
        PipeBarrier<PIPE_ALL>();
        uint64_t offsetPos = this->num_tokens_each_loop_current_core * index + i;
        GetCosSinCache(copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset, localStartAddr, srcShape_, dstShape_);
        localStartAddr += this->num_q_heads * this->rotary_dim;
    }
    Mul(reverseQ, cosSin, reverseQ, loopN * this->num_q_heads * this->rotary_dim);

    if (this->is_neox_style == 0) {
        Add(temp1Local, reverseQ, inQueCalLocal, loopN * this->num_q_heads * this->rotary_dim);
        for (uint32_t i = 0; i < loopN * this->num_q_heads; i++) {
            Gather(inQueCalLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], offsetLocal, (uint32_t)0,
                   this->rotary_dim);
        }
    } else {
        Add(inQueCalLocal, reverseQ, inQueCalLocal, loopN * this->num_q_heads * this->rotary_dim);
    }

    if (this->head_size != this->rotary_dim) {
        DataCopy(outLocal, inQueCalLocal,
                 {static_cast<uint16_t>(loopN * this->num_q_heads), static_cast<uint16_t>(rotaryBlockLen), 0,
                  static_cast<uint16_t>(headBlockLen - rotaryBlockLen)});
        DataCopy(
            outLocal[this->rotary_dim], inLocal[this->rotary_dim],
            {static_cast<uint16_t>(loopN * this->num_q_heads), static_cast<uint16_t>(headBlockLen - rotaryBlockLen),
             static_cast<uint16_t>(rotaryBlockLen), static_cast<uint16_t>(rotaryBlockLen)});
    } else {
        DataCopy(outLocal, inQueCalLocal,
                 {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_q_heads * headBlockLen), 0, 0});
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(queryGM[offset], outLocal,
             {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_q_heads * headBlockLen), 0, 0});
    PipeBarrier<PIPE_ALL>();
    // key的处理
    DataCopy(inLocal, key_in_GM[offsetk],
             {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_kv_heads * headBlockLen),
              static_cast<uint16_t>(this->k_leading_dimension / ELE_NUM_FP32 - this->k_size / ELE_NUM_FP32), 0});
    this->MTE2ToVSync();

    if (this->is_neox_style == 0) {
        DataCopy(temp1Local, inLocal,
                 {static_cast<uint16_t>(loopN * this->num_kv_heads), static_cast<uint16_t>(rotaryBlockLen),
                  static_cast<uint16_t>(headBlockLen - rotaryBlockLen), 0});
        uint64_t rsv = 0;
        for (uint32_t i = 0; i < loopN * this->num_kv_heads; i++) {
            GatherMask(inQueCalLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], static_cast<uint8_t>(1),
                       true, this->rotary_dim, {1, 1, 0, 0}, rsv);
            GatherMask(inQueCalLocal[i * this->rotary_dim + this->rotary_dim / 2], temp1Local[i * this->rotary_dim],
                       static_cast<uint8_t>(2), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
        }
    } else {
        DataCopy(inQueCalLocal, inLocal,
                 {static_cast<uint16_t>(loopN * this->num_kv_heads), static_cast<uint16_t>(rotaryBlockLen),
                  static_cast<uint16_t>(headBlockLen - rotaryBlockLen), 0});
    }
    DataCopy(reverseQ, inQueCalLocal[this->rotary_dim / 2],
             {static_cast<uint16_t>(loopN * this->num_kv_heads), calBlockLen, calBlockLen, calBlockLen});
    DataCopy(reverseQ[this->rotary_dim / 2], inQueCalLocal,
             {static_cast<uint16_t>(loopN * this->num_kv_heads), calBlockLen, calBlockLen, calBlockLen});

    localStartAddr = 0;
    cosSinOffset = 0;
    for (uint32_t i = 0; i < loopN; ++i) {
        PipeBarrier<PIPE_ALL>();
        uint64_t offsetPos = this->num_tokens_each_loop_current_core * index + i;
        GetCosSinCache(copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset, localStartAddr, srcShape_, dstShape_);
        localStartAddr += this->num_kv_heads * this->rotary_dim;
    }

    Mul(inQueCalLocal, cosSin, inQueCalLocal, loopN * this->num_kv_heads * this->rotary_dim);
    Mul(reverseQ, negOne, reverseQ, loopN * this->num_kv_heads * this->rotary_dim);

    localStartAddr = 0;
    cosSinOffset = this->rotary_dim / 2;
    for (uint32_t i = 0; i < loopN; ++i) {
        PipeBarrier<PIPE_ALL>();
        uint64_t offsetPos = this->num_tokens_each_loop_current_core * index + i;
        GetCosSinCache(copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset, localStartAddr, srcShape_, dstShape_);
        localStartAddr += this->num_kv_heads * this->rotary_dim;
    }

    Mul(reverseQ, cosSin, reverseQ, loopN * this->num_kv_heads * this->rotary_dim);

    if (this->is_neox_style == 0) {
        Add(temp1Local, reverseQ, inQueCalLocal, loopN * this->num_kv_heads * this->rotary_dim);

        for (uint32_t i = 0; i < loopN * this->num_kv_heads; i++) {
            Gather(inQueCalLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], offsetLocal, (uint32_t)0,
                   this->rotary_dim);
        }
    } else {
        Add(inQueCalLocal, reverseQ, inQueCalLocal, loopN * this->num_kv_heads * this->rotary_dim);
    }

    if (this->head_size != this->rotary_dim) {
        DataCopy(outLocal, inQueCalLocal,
                 {static_cast<uint16_t>(loopN * this->num_kv_heads), static_cast<uint16_t>(rotaryBlockLen), 0,
                  static_cast<uint16_t>(headBlockLen - rotaryBlockLen)});

        DataCopy(
            outLocal[this->rotary_dim], inLocal[this->rotary_dim],
            {static_cast<uint16_t>(loopN * this->num_kv_heads), static_cast<uint16_t>(headBlockLen - rotaryBlockLen),
             static_cast<uint16_t>(rotaryBlockLen), static_cast<uint16_t>(rotaryBlockLen)});
    } else {
        DataCopy(outLocal, inQueCalLocal,
                 {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_kv_heads * headBlockLen), 0, 0});
    }
    PipeBarrier<PIPE_ALL>();

    DataCopy(keyGM[offsetk], outLocal,
             {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->num_kv_heads * headBlockLen), 0, 0});
    PipeBarrier<PIPE_ALL>();

    inQueueCosSinCache.FreeTensor(inCosSin);
    inQQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheF32<T>::ComputeAlongHeads(uint64_t indexToken, uint64_t indexHeads)
{
    q_size = this->num_q_heads * this->head_size;
    k_size = this->num_kv_heads * this->head_size;
    uint64_t offsetQHead = indexHeads * this->num_qheads_each_loop * this->head_size;
    uint64_t offsetKHead = indexHeads * this->num_kheads_each_loop * this->head_size;
    uint64_t offset = indexToken * q_size + offsetQHead; // 非连续?
    uint64_t offsetk = indexToken * k_size + offsetKHead;

    uint32_t dstShape_[2] = {static_cast<uint32_t>(this->num_heads_each_loop), static_cast<uint32_t>(this->rotary_dim)};
    uint32_t dstShape_4Negone_[2] = {static_cast<uint32_t>(this->num_heads_each_loop),
                                     static_cast<uint32_t>(this->rotary_dim)};
    uint32_t srcShape_[2] = {1, static_cast<uint32_t>(this->rotary_dim)};

    LocalTensor<T> inLocal = inQQue.AllocTensor<T>();
    LocalTensor<T> inCosSin = inQueueCosSinCache.AllocTensor<T>();
    LocalTensor<T> outLocal = outQue.AllocTensor<T>();
    LocalTensor<T> reverseQ = reverseBuf.Get<T>();
    LocalTensor<T> negOne = negOneBuf.Get<T>();
    LocalTensor<T> cosSin = cosSinBuf.Get<T>();
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    LocalTensor<T> temp1Local = temp1.Get<T>();
    LocalTensor<T> inQueCalLocal = inQueCalBuf.Get<T>();
    LocalTensor<T> copyBuf0Local = copyBuf0.Get<T>();

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
        DataCopy(inLocal, query_in_GM[offset], {1, static_cast<uint16_t>(loopNQhead * headBlockLen), 0, 0});
        PipeBarrier<PIPE_ALL>();
        if (this->is_neox_style == 0) {
            PipeBarrier<PIPE_ALL>();
            DataCopy(temp1Local, inLocal,
                     {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(rotaryBlockLen),
                      static_cast<uint16_t>(headBlockLen - rotaryBlockLen), 0});
            PipeBarrier<PIPE_ALL>();

            uint64_t rsv = 0;
            for (uint32_t i = 0; i < loopNQhead; i++) {
                GatherMask(inQueCalLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim],
                           static_cast<uint8_t>(1), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
                PipeBarrier<PIPE_ALL>();
                GatherMask(inQueCalLocal[i * this->rotary_dim + this->rotary_dim / 2], temp1Local[i * this->rotary_dim],
                           static_cast<uint8_t>(2), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
                PipeBarrier<PIPE_ALL>();
            }
        } else {
            PipeBarrier<PIPE_ALL>();
            DataCopy(inQueCalLocal, inLocal,
                     {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(rotaryBlockLen),
                      static_cast<uint16_t>(headBlockLen - rotaryBlockLen), 0});
            PipeBarrier<PIPE_ALL>();
        }

        DataCopy(reverseQ, inQueCalLocal[this->rotary_dim / 2],
                 {static_cast<uint16_t>(loopNQhead), calBlockLen, calBlockLen, calBlockLen});
        PipeBarrier<PIPE_ALL>();
        DataCopy(reverseQ[this->rotary_dim / 2], inQueCalLocal,
                 {static_cast<uint16_t>(loopNQhead), calBlockLen, calBlockLen, calBlockLen});
        PipeBarrier<PIPE_ALL>();

        float One = 1.0;
        float None = -1.0;
        Duplicate<float>(negOne, None, this->rotary_dim / 2);
        Duplicate<float>(negOne[this->rotary_dim / 2], One, this->rotary_dim / 2);
        PipeBarrier<PIPE_ALL>();

        Broadcast<float, 2, 0, false>(negOne[this->rotary_dim], negOne, dstShape_4Negone_, srcShape_);
        PipeBarrier<PIPE_ALL>();

        uint64_t offsetPos = indexToken;
        uint64_t cosSinOffset = 0;
        uint64_t localStartAddr = 0;
        GetCosSinCache(copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset, localStartAddr, srcShape_, dstShape_);
        Mul(inQueCalLocal, cosSin, inQueCalLocal, loopNQhead * this->rotary_dim);
        PipeBarrier<PIPE_ALL>();
        Mul(reverseQ, negOne, reverseQ, loopNQhead * this->rotary_dim);
        PipeBarrier<PIPE_ALL>();
        offsetPos = indexToken;
        cosSinOffset = this->rotary_dim / 2;
        localStartAddr = 0;
        GetCosSinCache(copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset, localStartAddr, srcShape_, dstShape_);
        Mul(reverseQ, cosSin, reverseQ, loopNQhead * this->rotary_dim);
        PipeBarrier<PIPE_ALL>();

        if (this->is_neox_style == 0) {
            Add(temp1Local, reverseQ, inQueCalLocal, loopNQhead * this->rotary_dim);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < loopNQhead; i++) {
                Gather(inQueCalLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], offsetLocal, (uint32_t)0,
                       this->rotary_dim);
                PipeBarrier<PIPE_ALL>();
            }
        } else {
            Add(inQueCalLocal, reverseQ, inQueCalLocal, loopNQhead * this->rotary_dim);
            PipeBarrier<PIPE_ALL>();
        }

        if (this->head_size != this->rotary_dim) {
            DataCopy(outLocal, inQueCalLocal,
                     {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(rotaryBlockLen), 0,
                      static_cast<uint16_t>(headBlockLen - rotaryBlockLen)});
            PipeBarrier<PIPE_ALL>();
            DataCopy(outLocal[this->rotary_dim], inLocal[this->rotary_dim],
                     {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(headBlockLen - rotaryBlockLen),
                      static_cast<uint16_t>(rotaryBlockLen), static_cast<uint16_t>(rotaryBlockLen)});
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopy(outLocal, inQueCalLocal,
                     {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(headBlockLen), 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
        DataCopy(queryGM[offset], outLocal,
                 {static_cast<uint16_t>(loopNQhead), static_cast<uint16_t>(headBlockLen), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
    // key
    if (indexHeads < this->loop_along_kheads) {
        uint64_t loopNKhead = (indexHeads == this->loop_along_kheads - 1 && this->num_kheads_last_loop != 0) ?
                                  this->num_kheads_last_loop :
                                  this->num_kheads_each_loop;
        DataCopy(inLocal, key_in_GM[offsetk], {1, static_cast<uint16_t>(loopNKhead * headBlockLen), 0, 0});
        PipeBarrier<PIPE_ALL>();
        if (this->is_neox_style == 0) {
            PipeBarrier<PIPE_ALL>();
            DataCopy(temp1Local, inLocal,
                     {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(rotaryBlockLen),
                      static_cast<uint16_t>(headBlockLen - rotaryBlockLen), 0});
            PipeBarrier<PIPE_ALL>();

            uint64_t rsv = 0;
            for (uint32_t i = 0; i < loopNKhead; i++) {
                GatherMask(inQueCalLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim],
                           static_cast<uint8_t>(1), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
                PipeBarrier<PIPE_ALL>();
                GatherMask(inQueCalLocal[i * this->rotary_dim + this->rotary_dim / 2], temp1Local[i * this->rotary_dim],
                           static_cast<uint8_t>(2), true, this->rotary_dim, {1, 1, 0, 0}, rsv);
                PipeBarrier<PIPE_ALL>();
            }
        } else {
            PipeBarrier<PIPE_ALL>();
            DataCopy(inQueCalLocal, inLocal,
                     {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(rotaryBlockLen),
                      static_cast<uint16_t>(headBlockLen - rotaryBlockLen), 0});
            PipeBarrier<PIPE_ALL>();
        }

        DataCopy(reverseQ, inQueCalLocal[this->rotary_dim / 2],
                 {static_cast<uint16_t>(loopNKhead), calBlockLen, calBlockLen, calBlockLen});
        PipeBarrier<PIPE_ALL>();
        DataCopy(reverseQ[this->rotary_dim / 2], inQueCalLocal,
                 {static_cast<uint16_t>(loopNKhead), calBlockLen, calBlockLen, calBlockLen});
        PipeBarrier<PIPE_ALL>();

        uint64_t offsetPos = indexToken;
        uint64_t cosSinOffset = 0;
        uint64_t localStartAddr = 0;
        GetCosSinCache(copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset, localStartAddr, srcShape_, dstShape_);

        Mul(inQueCalLocal, cosSin, inQueCalLocal, loopNKhead * this->rotary_dim);
        PipeBarrier<PIPE_ALL>();
        Mul(reverseQ, negOne, reverseQ, loopNKhead * this->rotary_dim);
        PipeBarrier<PIPE_ALL>();
        offsetPos = indexToken;
        cosSinOffset = this->rotary_dim / 2;
        localStartAddr = 0;
        GetCosSinCache(copyBuf0Local, inCosSin, cosSin, offsetPos, cosSinOffset, localStartAddr, srcShape_, dstShape_);
        Mul(reverseQ, cosSin, reverseQ, loopNKhead * this->rotary_dim);
        PipeBarrier<PIPE_ALL>();

        if (this->is_neox_style == 0) {
            Add(temp1Local, reverseQ, inQueCalLocal, loopNKhead * this->rotary_dim);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < loopNKhead; i++) {
                Gather(inQueCalLocal[i * this->rotary_dim], temp1Local[i * this->rotary_dim], offsetLocal, (uint32_t)0,
                       this->rotary_dim);
                PipeBarrier<PIPE_ALL>();
            }
        } else {
            Add(inQueCalLocal, reverseQ, inQueCalLocal, loopNKhead * this->rotary_dim);
            PipeBarrier<PIPE_ALL>();
        }

        if (this->head_size != this->rotary_dim) {
            DataCopy(outLocal, inQueCalLocal,
                     {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(rotaryBlockLen), 0,
                      static_cast<uint16_t>(headBlockLen - rotaryBlockLen)});
            PipeBarrier<PIPE_ALL>();
            DataCopy(outLocal[this->rotary_dim], inLocal[this->rotary_dim],
                     {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(headBlockLen - rotaryBlockLen),
                      static_cast<uint16_t>(rotaryBlockLen), static_cast<uint16_t>(rotaryBlockLen)});
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopy(outLocal, inQueCalLocal,
                     {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(headBlockLen), 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
        DataCopy(keyGM[offsetk], outLocal,
                 {static_cast<uint16_t>(loopNKhead), static_cast<uint16_t>(headBlockLen), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

    inQueueCosSinCache.FreeTensor(inCosSin);
    inQQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void RopeWithSinCosCacheF32<T>::Process()
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
