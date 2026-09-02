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
 * \file moe_token_unpermute_with_ep.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_WITH_EP
#define MOE_TOKEN_UNPERMUTE_WITH_EP

#include "kernel_operator.h"
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
class KernelMoeTokenUnpermuteWithEp {
public:
    __aicore__ inline KernelMoeTokenUnpermuteWithEp() {}

    __aicore__ inline void Init(GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs,
                                GM_ADDR unpermuted_tokens,
                                const MoeTokenUnpermuteWithEpTilingData *__restrict tiling_data, TPipe *t_pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CalMultiOutToken(const int64_t out_offset, const int64_t out_tokens_number);
    __aicore__ inline void CalSingleOutToken(const int64_t start_token, const int64_t out_token_idx);
    __aicore__ inline void CalPartOutToken(const int64_t start_token, const int64_t h_index, const int64_t h_length,
                                           const int64_t out_token_index);
    __aicore__ inline void CopyTokenIn(const T2 in_token_index, const int64_t h_index, const int64_t h_length);
    __aicore__ inline void CalFirstToken(const float prob_value, const int64_t h_length);
    __aicore__ inline void CalToken(const float prob_value, const int64_t h_length);
    __aicore__ inline void CopyOut(const int64_t out_token_index, const int64_t h_index, const int64_t h_length);
    __aicore__ inline void CalPartOutProb(const int64_t start_token, const int64_t h_index, const int64_t h_length,
                                          const int64_t out_token_index);
    __aicore__ inline void CopyProbIn(const T2 in_prob_index, const int64_t h_index, const int64_t h_length);
    __aicore__ inline void CopyProbOut(const int64_t out_prob_index, const int64_t h_index, const int64_t h_length);

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> tokens_inque;
    TQue<QuePosition::VECIN, 1> indices_inque;
    TQue<QuePosition::VECIN, 1> probs_inque;
    TBuf<TPosition::VECCALC> temp_buffer0;
    TBuf<TPosition::VECCALC> temp_buffer1;
    TBuf<TPosition::VECCALC> temp_buffer2;
    TQue<QuePosition::VECOUT, 1> outque;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> probs_inoutque;
    GlobalTensor<T1> tokensGM;
    GlobalTensor<T1> outGM;
    GlobalTensor<T2> indicesGM;
    GlobalTensor<T3> probsGM;
    LocalTensor<T2> indicesLocal;
    LocalTensor<float> token_tensor0;
    LocalTensor<float> token_tensor1;
    LocalTensor<float> probs_tensor;
    DataCopyPadExtParams<T1> extParams1{false, 0, 0, 0};
    DataCopyPadExtParams<T2> extParams2{false, 0, 0, 0};
    DataCopyPadExtParams<T3> extParams3{false, 0, 0, 0};
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};

    event_t probsMte2ToS;
    event_t probsSToMte2;
    event_t probsMte2ToV;
    event_t probsVToMte2;
    event_t probsMte3ToV;
    event_t probsVToMte3;
    event_t probsVToS;
    event_t probsSToV;

    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t ALIGN_512 = 512;

    int64_t hidden_size;
    int64_t permuted_probs_grad_length;
    int64_t top_k;
    int64_t start;
    int64_t end;
    int64_t num_out_tokens;
    int64_t hidden_splited_length;
    int64_t hidden_splited_num;
    int64_t hidden_splited_remain;
    int64_t tokens_core_length;
    int64_t tokens_core_remain;
    int64_t tokens_splited_length;
    int64_t tokens_splited_num;
    int64_t tokens_splited_remain;
};

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::Init(
    GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens,
    const MoeTokenUnpermuteWithEpTilingData *__restrict tiling_data, TPipe *t_pipe)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->pipe = t_pipe;
    // row_input
    this->hidden_size = ISTOKEN ? tiling_data->hidden_size : 1;
    this->top_k = tiling_data->top_k;
    this->start = tiling_data->start;
    this->end = tiling_data->end;
    this->num_out_tokens = tiling_data->num_out_tokens;
    this->permuted_probs_grad_length = tiling_data->permuted_probs_grad_length;
    // hidden_tiling
    this->hidden_splited_length = tiling_data->hidden_splited_length;
    this->hidden_splited_num = tiling_data->hidden_splited_num;
    this->hidden_splited_remain = tiling_data->hidden_splited_remain;
    // token_tiling
    this->tokens_core_length = tiling_data->tokens_core_length;
    this->tokens_core_remain = tiling_data->tokens_core_remain;
    this->tokens_splited_length = tiling_data->tokens_splited_length;
    this->tokens_splited_num = tiling_data->tokens_splited_num;
    this->tokens_splited_remain = tiling_data->tokens_splited_remain;

    // 处理token_by_core尾块
    if (this->tokens_core_remain > 0 && GetBlockIdx() < this->tokens_core_remain) {
        this->tokens_core_length += 1;
        this->tokens_splited_remain += 1;
    }

    int64_t hidden_splited_length_align512 = (this->hidden_splited_length + ALIGN_512 - 1) & ~(ALIGN_512 - 1);

    int64_t block_length = this->tokens_core_length * this->top_k;
    int64_t block_splited_length = this->tokens_splited_length * this->top_k;

    int64_t block_offset;
    if (this->tokens_core_remain > 0) {
        if (GetBlockIdx() < this->tokens_core_remain) {
            block_offset = block_length * GetBlockIdx();
        } else {
            block_offset = (block_length + this->top_k) * this->tokens_core_remain +
                           block_length * (GetBlockIdx() - this->tokens_core_remain);
        }
    } else {
        block_offset = block_length * GetBlockIdx();
    }

    this->tokensGM.SetGlobalBuffer((__gm__ T1 *)permuted_tokens);
    this->indicesGM.SetGlobalBuffer((__gm__ T2 *)sorted_indices + block_offset, block_length);

    int64_t out_block_offset;
    if (this->tokens_core_remain > 0) {
        if (GetBlockIdx() < this->tokens_core_remain) {
            out_block_offset = this->tokens_core_length * GetBlockIdx() * hidden_size;
        } else {
            out_block_offset = (this->tokens_core_length + 1) * this->tokens_core_remain +
                               this->tokens_core_length * (GetBlockIdx() - this->tokens_core_remain);
            out_block_offset *= this->hidden_size;
        }
    } else {
        out_block_offset = this->tokens_core_length * GetBlockIdx() * hidden_size;
    }

    // 如果是probs(permute_with_ep_grad场景), 按照一个AIV分topk个probs处理
    if constexpr (!ISTOKEN) {
        out_block_offset = out_block_offset * this->top_k;
    }

    this->outGM.SetGlobalBuffer((__gm__ T1 *)unpermuted_tokens + out_block_offset,
                                this->tokens_core_length * this->hidden_size);
    if constexpr (ISTOKEN) {
        this->pipe->InitBuffer(tokens_inque, tiling_data->buffer_num, hidden_splited_length_align512 * sizeof(T1));
    } else {
        this->pipe->InitBuffer(probs_inoutque, 1, hidden_splited_length_align512 * sizeof(T1));
    }
    this->pipe->InitBuffer(indices_inque, 1, block_splited_length * (sizeof(T2)));
    this->pipe->InitBuffer(outque, 1, hidden_splited_length_align512 * sizeof(T1));

    if constexpr (!IsSameType<T1, float>::value) {
        this->pipe->InitBuffer(temp_buffer0, hidden_splited_length_align512 * sizeof(float) + 256);
        this->pipe->InitBuffer(temp_buffer1, hidden_splited_length_align512 * sizeof(float));
        this->token_tensor0 = this->temp_buffer0.template Get<float>();
        this->token_tensor1 = this->temp_buffer1.template Get<float>();
    }

    if constexpr (PROBS) {
        this->probsGM.SetGlobalBuffer((__gm__ T3 *)probs + block_offset, block_length);
        this->pipe->InitBuffer(probs_inque, 1, block_splited_length * (sizeof(T3)));
        if constexpr (!IsSameType<T3, float>::value) {
            this->pipe->InitBuffer(temp_buffer2, block_splited_length * sizeof(float));
            this->probs_tensor = this->temp_buffer2.template Get<float>();
        }
    }

    probsMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    probsSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    probsMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    probsVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    probsMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    probsVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    probsVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    probsSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
};

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::Process()
{
    for (int64_t i = 0; i < this->tokens_splited_num; ++i) {
        CalMultiOutToken(i * this->tokens_splited_length, this->tokens_splited_length);
    }
    // 处理tokens_num不能均匀分核数的尾块
    if (this->tokens_splited_remain > 0) {
        CalMultiOutToken(this->tokens_splited_num * this->tokens_splited_length, this->tokens_splited_remain);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CalMultiOutToken(
    const int64_t out_offset, const int64_t out_tokens_number)
{
    this->indicesLocal = this->indices_inque.template AllocTensor<T2>();
    int64_t in_offset = out_offset * this->top_k;
    this->copyParams.blockLen = out_tokens_number * this->top_k * sizeof(T2);
    DataCopyPad(this->indicesLocal, this->indicesGM[in_offset], this->copyParams, this->extParams2);
    this->indices_inque.template EnQue(this->indicesLocal);

    if constexpr (PROBS) {
        LocalTensor<T3> temp_probs_tensor = this->probs_inque.template AllocTensor<T3>();
        this->copyParams.blockLen = out_tokens_number * this->top_k * sizeof(T3);
        DataCopyPad(temp_probs_tensor, this->probsGM[in_offset], this->copyParams, this->extParams3);
        this->probs_inque.template EnQue(temp_probs_tensor);
        temp_probs_tensor = this->probs_inque.template DeQue<T3>();
        if constexpr (!IsSameType<T3, float>::value) {
            Cast(this->probs_tensor, temp_probs_tensor, RoundMode::CAST_NONE, out_tokens_number * this->top_k);
            this->probs_inque.FreeTensor(temp_probs_tensor);
            PipeBarrier<PIPE_V>();
        } else {
            this->probs_tensor = temp_probs_tensor;
        }
    }
    this->indicesLocal = this->indices_inque.template DeQue<T2>();

#pragma bisheng auto_sync parallel
    for (int64_t out_token_idx = 0; out_token_idx < out_tokens_number; ++out_token_idx) {
        CalSingleOutToken(out_token_idx * this->top_k, out_offset + out_token_idx);
    }
    // Free Tensor
    this->indices_inque.FreeTensor(this->indicesLocal);
    if constexpr (PROBS && IsSameType<T3, float>::value) {
        this->probs_inque.FreeTensor(this->probs_tensor);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CalSingleOutToken(
    const int64_t start_token, const int64_t out_token_idx)
{
    if constexpr (ISTOKEN) {
        for (int64_t h_index = 0; h_index < this->hidden_splited_num; ++h_index) {
            CalPartOutToken(start_token, h_index, this->hidden_splited_length, out_token_idx);
        }
        // 一次不能完整容纳完整的hidden_size, 处理尾块
        if (this->hidden_splited_remain > 0) {
            CalPartOutToken(start_token, this->hidden_splited_num, this->hidden_splited_remain, out_token_idx);
        }
    } else {
        CalPartOutProb(start_token, 0, this->permuted_probs_grad_length, out_token_idx);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CalPartOutToken(
    const int64_t start_token, const int64_t h_index, const int64_t h_length, const int64_t out_token_index)
{
    if constexpr (IsSameType<T1, float>::value) {
        this->token_tensor0 = this->outque.template AllocTensor<T1>();
    }
    int64_t end_token = start_token + this->top_k;
    bool needCopyIn = false;
    float prob_value = 0;

    SetFlag<HardEvent::MTE2_S>(probsMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(probsMte2ToS);
    SetFlag<HardEvent::V_S>(probsVToS);
    WaitFlag<HardEvent::V_S>(probsVToS);
    T2 acl_token_idx = this->indicesLocal.GetValue(start_token);
    SetFlag<HardEvent::S_MTE2>(probsSToMte2);
    WaitFlag<HardEvent::S_MTE2>(probsSToMte2);
    SetFlag<HardEvent::S_V>(probsSToV);
    WaitFlag<HardEvent::S_V>(probsSToV);

    if constexpr (PROBS) {
        prob_value = this->probs_tensor.GetValue(start_token);
        needCopyIn = acl_token_idx >= this->start && acl_token_idx < this->end && prob_value != 0;
    } else {
        needCopyIn = acl_token_idx >= this->start && acl_token_idx < this->end;
    }
    // 处理第一个Token数据
    if (needCopyIn) {
        CopyTokenIn(acl_token_idx, h_index, h_length);
        PipeBarrier<PIPE_V>();
        CalFirstToken(prob_value, h_length);
    } else {
        PipeBarrier<PIPE_V>();
        Duplicate(this->token_tensor0, static_cast<float>(0), h_length);
    }
    // 处理剩余的Token数据
    for (int64_t token_index = start_token + 1; token_index < end_token; ++token_index) {
        SetFlag<HardEvent::MTE2_S>(probsMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(probsMte2ToS);
        SetFlag<HardEvent::V_S>(probsVToS);
        WaitFlag<HardEvent::V_S>(probsVToS);
        acl_token_idx = this->indicesLocal.GetValue(token_index);
        SetFlag<HardEvent::S_MTE2>(probsSToMte2);
        WaitFlag<HardEvent::S_MTE2>(probsSToMte2);
        SetFlag<HardEvent::S_V>(probsSToV);
        WaitFlag<HardEvent::S_V>(probsSToV);
        if constexpr (PROBS) {
            prob_value = this->probs_tensor.GetValue(token_index);
            needCopyIn = acl_token_idx >= this->start && acl_token_idx < this->end && prob_value != 0;
        } else {
            needCopyIn = acl_token_idx >= this->start && acl_token_idx < this->end;
        }
        if (needCopyIn) {
            CopyTokenIn(acl_token_idx, h_index, h_length);
            PipeBarrier<PIPE_V>();
            CalToken(prob_value, h_length);
        }
    }
    // 输出计算结果
    CopyOut(out_token_index, h_index, h_length);
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CopyTokenIn(const T2 in_token_index,
                                                                                              const int64_t h_index,
                                                                                              const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = this->tokens_inque.template AllocTensor<T1>();
    int64_t offset = (in_token_index - this->start) * this->hidden_size + h_index * this->hidden_splited_length;

    if (likely((h_length * sizeof(T1)) % BLOCK_SIZE == 0)) {
        DataCopy(tokensLocal, this->tokensGM[offset], h_length);
    } else {
        this->copyParams.blockLen = h_length * sizeof(T1);
        DataCopyPad(tokensLocal, this->tokensGM[offset], this->copyParams, this->extParams1);
    }

    this->tokens_inque.template EnQue(tokensLocal);
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CalFirstToken(const float prob_value,
                                                                                                const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = this->tokens_inque.template DeQue<T1>();

    if constexpr (!IsSameType<T1, float>::value) {
        Cast(this->token_tensor0, tokensLocal, RoundMode::CAST_NONE, h_length);
    } else {
        uint64_t byteAlign32 = (h_length * sizeof(float) + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
        DataCopy(this->token_tensor0, tokensLocal, byteAlign32 / sizeof(float));
    }

    this->tokens_inque.FreeTensor(tokensLocal);

    if constexpr (PROBS) {
        PipeBarrier<PIPE_V>();
        Muls(this->token_tensor0, this->token_tensor0, prob_value, h_length);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CalToken(const float prob_value,
                                                                                           const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = this->tokens_inque.template DeQue<T1>();

    if constexpr (!IsSameType<T1, float>::value) {
        Cast(this->token_tensor1, tokensLocal, RoundMode::CAST_NONE, h_length);
        this->tokens_inque.FreeTensor(tokensLocal);
        if constexpr (PROBS) {
            PipeBarrier<PIPE_V>();
            Muls(this->token_tensor1, this->token_tensor1, prob_value, h_length);
        }
        PipeBarrier<PIPE_V>();
        Add(this->token_tensor0, this->token_tensor0, this->token_tensor1, h_length);
    } else {
        if constexpr (PROBS) {
            Muls(tokensLocal, tokensLocal, prob_value, h_length);
            PipeBarrier<PIPE_V>();
        }
        Add(this->token_tensor0, this->token_tensor0, tokensLocal, h_length);
        this->tokens_inque.FreeTensor(tokensLocal);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CopyOut(const int64_t out_token_index,
                                                                                          const int64_t h_index,
                                                                                          const int64_t h_length)
{
    LocalTensor<T1> temp_out_tensors;
    if constexpr (!IsSameType<T1, float>::value) {
        temp_out_tensors = this->outque.template AllocTensor<T1>();
        PipeBarrier<PIPE_V>();
        Cast(temp_out_tensors, this->token_tensor0, RoundMode::CAST_RINT, h_length);
    } else {
        temp_out_tensors = this->token_tensor0;
    }

    this->outque.template EnQue<T1>(temp_out_tensors);
    temp_out_tensors = this->outque.template DeQue<T1>();

    int64_t offset = out_token_index * this->hidden_size + h_index * this->hidden_splited_length;
    if (likely((h_length * sizeof(T1)) % BLOCK_SIZE == 0)) {
        DataCopy(this->outGM[offset], temp_out_tensors, h_length);
    } else {
        this->copyParams.blockLen = h_length * sizeof(T1);
        DataCopyPad(this->outGM[offset], temp_out_tensors, this->copyParams);
    }

    this->outque.FreeTensor(temp_out_tensors);
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CalPartOutProb(
    const int64_t start_prob, const int64_t h_index, const int64_t h_length, const int64_t out_prob_index)
{
    int64_t end_prob = start_prob + this->top_k;
    for (int64_t prob_index = start_prob; prob_index < end_prob; ++prob_index) {
        SetFlag<HardEvent::MTE2_S>(probsMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(probsMte2ToS);
        SetFlag<HardEvent::V_S>(probsVToS);
        WaitFlag<HardEvent::V_S>(probsVToS);
        T2 acl_prob_idx = this->indicesLocal.GetValue(prob_index);
        SetFlag<HardEvent::S_MTE2>(probsSToMte2);
        WaitFlag<HardEvent::S_MTE2>(probsSToMte2);
        SetFlag<HardEvent::S_V>(probsSToV);
        WaitFlag<HardEvent::S_V>(probsSToV);

        CopyProbIn(acl_prob_idx, h_index, h_length);
        CopyProbOut(out_prob_index * this->top_k + prob_index - start_prob, h_index, h_length);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CopyProbIn(const T2 in_prob_index,
                                                                                             const int64_t h_index,
                                                                                             const int64_t h_length)
{
    LocalTensor<T1> probsTmpLocal = this->probs_inoutque.template AllocTensor<T1>();
    bool needCopyIn = in_prob_index >= this->start && in_prob_index < this->end;
    if (needCopyIn) {
        int64_t offset = (in_prob_index - this->start) * this->hidden_size + h_index * this->hidden_splited_length;
        this->copyParams.blockLen = h_length * sizeof(T1);
        DataCopyPad(probsTmpLocal, this->tokensGM[offset], this->copyParams, this->extParams1);
    } else {
        SetFlag<HardEvent::MTE3_V>(probsMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(probsMte3ToV);
        SetFlag<HardEvent::MTE2_V>(probsMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(probsMte2ToV);
        Duplicate(probsTmpLocal, static_cast<T1>(0), h_length);
        SetFlag<HardEvent::V_MTE3>(probsVToMte3);
        WaitFlag<HardEvent::V_MTE3>(probsVToMte3);
        SetFlag<HardEvent::V_MTE2>(probsVToMte2);
        WaitFlag<HardEvent::V_MTE2>(probsVToMte2);
    }
    this->probs_inoutque.template EnQue<T1>(probsTmpLocal);
}

template <typename T1, typename T2, typename T3, bool PROBS, bool ISTOKEN>
__aicore__ inline void KernelMoeTokenUnpermuteWithEp<T1, T2, T3, PROBS, ISTOKEN>::CopyProbOut(
    const int64_t out_prob_index, const int64_t h_index, const int64_t h_length)
{
    LocalTensor<T1> probsTmpLocal = this->probs_inoutque.template DeQue<T1>();
    int64_t offset = out_prob_index * this->hidden_size + h_index * this->hidden_splited_length;
    this->copyParams.blockLen = h_length * sizeof(T1);
    DataCopyPad(this->outGM[offset], probsTmpLocal, this->copyParams);
    this->probs_inoutque.FreeTensor(probsTmpLocal);
}
#endif // MOE_TOKEN_UNPERMUTE_WITH_EP
