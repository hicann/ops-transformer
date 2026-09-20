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
 * \file gsag_common_header.h
 * \brief Flags / layout / RunTimeInfo for GenericBlockSparseAttentionGrad (design §3.5 / §4.1).
 */

#ifndef GSAG_ARC35_COMMON_HEADER_H
#define GSAG_ARC35_COMMON_HEADER_H

namespace GSAG_ARC35 {

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))

static constexpr uint32_t BSND = 0;
static constexpr uint32_t BNSD = 1;
static constexpr uint32_t TND = 2;

static constexpr uint32_t DK = 13;
static constexpr uint32_t DV = 14;

static constexpr uint32_t C0_SIZE = 16;
static constexpr uint32_t BLOCK_SIZE = 32; // Ascend DataCopy block unit (bytes)
static constexpr uint32_t FLAG_CUBE_POST = 4;
static constexpr uint32_t FLAG_C_GATHER_ARM = 5;
static constexpr uint32_t FLAG_C_MM345_DONE = 6;
static constexpr uint32_t FLAG_V_GATHER_C_PING = 7;
static constexpr uint32_t FLAG_V_GATHER_C_PONG = 9;
static constexpr uint32_t FLAG_C1_V1_PING = 0;
static constexpr uint32_t FLAG_C1_V1_PONG = 13;
static constexpr uint32_t FLAG_C2_V2_PING = 1;
static constexpr uint32_t FLAG_C2_V2_PONG = 14;
static constexpr uint32_t FLAG_V1_C3 = 2;
static constexpr uint32_t FLAG_V2_C45 = 3;
static constexpr uint32_t CROSS_CORE_SYNC_MODE_2 = 2;
static constexpr uint32_t CROSS_CORE_SYNC_MODE = 4;
static constexpr uint32_t AIV0_AIV1_FLAG_OFFSET = 16;

// Keep in sync with generic_block_sparse_attention_grad_metadata.h (standalone kernel compile).
static constexpr uint32_t AIC_CORE_MAX_NUM = 36;
static constexpr uint32_t GRAD_METADATA_SIZE = 8;
static constexpr uint32_t TASK_ENTRY_SIZE = 4;
static constexpr uint32_t TASK_LIST_OFFSET = GRAD_METADATA_SIZE + 2 * AIC_CORE_MAX_NUM;
static constexpr uint32_t CORE_TASK_START_OFFSET = GRAD_METADATA_SIZE;
static constexpr uint32_t CORE_TASK_END_OFFSET = GRAD_METADATA_SIZE + AIC_CORE_MAX_NUM;
static constexpr uint32_t TASK_B = 0;
static constexpr uint32_t TASK_N2 = 1;
static constexpr uint32_t TASK_J = 2;
static constexpr uint32_t TASK_G = 3;

struct ConstInfo {
    int32_t q_head_num{0};
    int32_t kv_head_num{0};
    int32_t block_x{0};
    int32_t block_y{0};
    int32_t head_dim{0};
    int32_t max_s1{0};
    int32_t num_j{0};
    int32_t group_size{0};
};

struct RunTimeInfo {
    int32_t taskId{0};
    int32_t bIdx{0};
    int32_t n1Idx{0}; // n1_cur = n2*G + g
    int32_t n2Idx{0};
    int32_t gIdx{0};
    int32_t kvBlockIdx{0}; // j
    int32_t s1Idx{0};
    int32_t s2Idx{0}; // s2_start = j * By
    int32_t last_q_seq_sum{0};
    int32_t last_kv_seq_sum{0};
    int32_t cur_q_seq_len{0};
    int32_t cur_kv_seq_len{0};
    int32_t need_compute{0};
    int32_t need_copy_kv{0};
    int32_t kv_ping_pong_idx{0};
    int32_t is_singlekv_last{0};
    int32_t mask_type{0};
    int64_t s1Len{0}; // m = count
    int64_t s2Len{0}; // n = kv block len
    int64_t s1LenAlign{0};
    int64_t s2LenAlign{0};
    int64_t queryGmOffset{0};
    int64_t keyGmOffset{0};
    int64_t lseGmOffset{0};
    int64_t sftgGmOffset{0};
    int64_t sparseIdxOffset{0}; // base of sparseBlockIdx[b,n2,j,0]
    int64_t sparseCount{0};
};

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

__aicore__ inline int64_t GetSeqLen(int32_t i, __gm__ uint8_t *seqLen)
{
    if (seqLen == nullptr) {
        return 0;
    }
    const __gm__ int64_t *cu = (__gm__ int64_t *)seqLen;
    return cu[i + 1] - cu[i];
}

__aicore__ inline int64_t GetSeqPrefix(int32_t i, __gm__ uint8_t *seqLen)
{
    if (i <= 0 || seqLen == nullptr) {
        return 0;
    }
    return ((__gm__ int64_t *)seqLen)[i];
}

__aicore__ inline int64_t GetSeqUsedLen(int32_t i, __gm__ uint8_t *seqused, __gm__ uint8_t *cuSeqLen)
{
    if (seqused != nullptr) {
        return static_cast<int64_t>(((__gm__ int32_t *)seqused)[i]);
    }
    return GetSeqLen(i, cuSeqLen);
}

template <uint32_t INPUT_LAYOUT>
__aicore__ inline int64_t GetQKVGmOffset(int64_t lastBatchSum, int64_t currentSeqLen, int64_t headNum, int64_t headDim,
                                         int64_t batchIdx, int64_t seqlenIdx, int64_t nIdx)
{
    if constexpr (INPUT_LAYOUT == BSND) {
        return batchIdx * (currentSeqLen * headNum * headDim) + (seqlenIdx * headNum * headDim) + (nIdx * headDim);
    } else if constexpr (INPUT_LAYOUT == BNSD) {
        return batchIdx * (headNum * currentSeqLen * headDim) + (nIdx * currentSeqLen * headDim) +
               (seqlenIdx * headDim);
    } else {
        return lastBatchSum * (headNum * headDim) + (seqlenIdx * headNum * headDim) + (nIdx * headDim);
    }
}

template <uint32_t INPUT_LAYOUT>
__aicore__ inline int64_t GetSftgGmOffset(int64_t lastBatchSum, int64_t currentSeqLen, int64_t headNum,
                                          int64_t batchIdx, int64_t seqlenIdx, int64_t n1Idx)
{
    if constexpr (INPUT_LAYOUT == TND) {
        return lastBatchSum * headNum * 8 + n1Idx * currentSeqLen * 8 + seqlenIdx * 8;
    }
    return batchIdx * (headNum * currentSeqLen) * 8 + n1Idx * currentSeqLen * 8 + seqlenIdx * 8;
}

} // namespace GSAG_ARC35

#endif
