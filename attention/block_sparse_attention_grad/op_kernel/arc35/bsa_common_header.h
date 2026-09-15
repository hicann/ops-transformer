/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

namespace BSA_ARC35 {

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))

static constexpr uint32_t BSND = 0;
static constexpr uint32_t BNSD = 1;
static constexpr uint32_t TND = 2;

static constexpr uint32_t QK = 10;
static constexpr uint32_t DYV = 11;
static constexpr uint32_t DQ = 12;
static constexpr uint32_t DK = 13;
static constexpr uint32_t DV = 14;

static constexpr uint32_t C0_SIZE = 16;
static constexpr uint32_t FLAG_C1_V1 = 0;     // cube(qk) -> vec(softmax)
static constexpr uint32_t FLAG_C2_V2 = 1;     // cube(dyV) -> vec(softmaxGrad)
static constexpr uint32_t FLAG_V1_C3 = 2;     // vec(softmax) -> cube(dv)
static constexpr uint32_t FLAG_V2_C45 = 3;    // vec(softmaxGrad) -> cube(dq\dk)
static constexpr uint32_t FLAG_CUBE_POST = 4; // vec(softmaxGrad) -> cube(dq\dk)

struct ConstInfo {
    int32_t q_head_num{0};
    int32_t kv_head_num{0};
    int32_t block_x{0};
    int32_t block_y{0};
    int32_t head_dim{0};
    int32_t q_block_num{0};
    int32_t kv_block_num{0};
};

struct RunTimeInfo {
    int32_t taskId{0};
    int32_t bIdx{0};  // 当前计算的batch的idx
    int32_t n1Idx{0}; // 当前计算的q_head的idx
    int32_t n2Idx{0}; // 当前计算的kv_head的idx
    int32_t s1Idx{0}; // 当前计算的q_seq的起始idx
    int32_t s2Idx{0}; // 当前计算的kv_seq的起始idx
    int32_t last_q_seq_sum{0};
    int32_t last_kv_seq_sum{0};
    int32_t cur_q_seq_len{0};
    int32_t cur_kv_seq_len{0};
    int32_t need_compute{0};     // 是否存在任务需要计算
    int32_t need_copy_kv{0};     // 是否需要copy kv
    int32_t kv_ping_pong_idx{0}; // kv的ping pong idx
    int32_t is_singlekv_last{0};
    int64_t s1Len{0};         // 当前计算的q_seq的长度
    int64_t s2Len{0};         // 当前计算的kv_seq的长度
    int64_t s1LenAlign{0};    // 当前计算的q_seq的16对齐的长度
    int64_t s2LenAlign{0};    // 当前计算的kv_seq的16对齐的长度
    int64_t queryGmOffset{0}; // query gm offset
    int64_t keyGmOffset{0};   // key gm offset
    int64_t lseGmOffset{0};   // lse gm offset
    int64_t sftgGmOffset{0};  // softmaxGradFront gm offset
};

inline __aicore__ uint32_t max(const uint32_t a, const uint32_t b)
{
    return a > b ? a : b;
}

template <typename T>
inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    return (val + align - 1) / align * align;
}

__aicore__ inline int64_t GetSeqLen(int32_t i, __gm__ uint8_t *seq_Len)
{
    int64_t actualSeqlen;
    if (i == 0) {
        actualSeqlen = ((__gm__ int64_t *)seq_Len)[0];
    } else {
        actualSeqlen = ((__gm__ int64_t *)seq_Len)[i] - ((__gm__ int64_t *)seq_Len)[i - 1];
    }
    return actualSeqlen;
}

__aicore__ inline int64_t GetSeqTotalLen(int32_t i, __gm__ uint8_t *seq_Len)
{
    int64_t actualTotalSeqlen = ((__gm__ int64_t *)seq_Len)[i];
    return actualTotalSeqlen;
}

template <uint32_t INPUT_LAYOUT>
__aicore__ inline int64_t GetQKVGmOffset(int64_t lastBatchSum, int64_t current_seq_len, int64_t head_num,
                                         int64_t head_dim, int64_t batch_idx, int64_t seqlen_idx, int64_t n_idx)
{
    if constexpr (INPUT_LAYOUT == BSND) {
        return batch_idx * (current_seq_len * head_num * head_dim) + (seqlen_idx * head_num * head_dim) +
               (n_idx * head_dim);
    } else if (INPUT_LAYOUT == BNSD) {
        return batch_idx * (head_num * current_seq_len * head_dim) + (n_idx * current_seq_len * head_dim) +
               (seqlen_idx * head_dim);
    } else if constexpr (INPUT_LAYOUT == TND) {
        return lastBatchSum * (head_num * head_dim) + (seqlen_idx * head_num * head_dim) + (n_idx * head_dim);
    }
}

template <uint32_t INPUT_LAYOUT>
__aicore__ inline int64_t GetLseGmOffset(int64_t lastBatchSum, int64_t current_seq_len, int64_t head_num,
                                         int64_t batch_idx, int64_t seqlen_idx, int64_t n1Idx)
{
    if constexpr (INPUT_LAYOUT == TND) {
        // TN1
        return lastBatchSum * head_num + (seqlen_idx * head_num) + n1Idx;
    } else {
        // BNS1
        return batch_idx * (head_num * current_seq_len) + n1Idx * (current_seq_len) + seqlen_idx;
    }
}

template <uint32_t INPUT_LAYOUT>
__aicore__ inline int64_t GetSftgGmOffset(int64_t lastBatchSum, int64_t current_seq_len, int64_t head_num,
                                          int64_t batch_idx, int64_t seqlen_idx, int64_t n1Idx)
{
    if constexpr (INPUT_LAYOUT == TND) {
        // BNS8
        return lastBatchSum * head_num * 8 + n1Idx * current_seq_len * 8 + seqlen_idx * 8;
    } else {
        // BNS8
        return batch_idx * (head_num * current_seq_len) * 8 + n1Idx * (current_seq_len) * 8 + seqlen_idx * 8;
    }
}

__aicore__ inline bool IsValidBlock(const ConstInfo &const_info, const int64_t batch_idx, const int64_t n1_idx,
                                    const int64_t q_block_idx, const int64_t kv_block_idx,
                                    __gm__ uint8_t *block_sparse_mask)

{
    // The public mask layout is [B, N1, Q_BLOCK, KV_BLOCK].  The Arch35 aclnn
    // adapter transposes it to contiguous [B, N1, KV_BLOCK, Q_BLOCK] before
    // launching this kernel.
    int64_t offset = batch_idx * (const_info.q_head_num * const_info.q_block_num * const_info.kv_block_num) +
                     n1_idx * (const_info.q_block_num * const_info.kv_block_num) +
                     kv_block_idx * const_info.q_block_num + q_block_idx;
    bool is_valid = block_sparse_mask[offset];
    return is_valid;
}

__aicore__ inline int32_t GetBlockLen(int32_t start_idx, int32_t end_idx, int32_t max_len)
{
    return start_idx + max_len < end_idx ? max_len : (end_idx - start_idx);
}

} // namespace BSA_ARC35
