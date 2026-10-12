/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "bsa_common_header.h"
using namespace AscendC;

namespace BSA_ARC35 {

template <typename BSA_TYPE>
class AddrComputeModule {
protected:
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;
    static constexpr bool INDEX_ENABLE = BSA_TYPE::index_enable;
    static_assert(DETERMINISTIC_ENABLE || INDEX_ENABLE,
                  "BSAG arch35: 非确定性路径必须使用 NonZero index 遍历，不能再读 blockSparseMask 判块");
    GM_ADDR actualQseqlen_;
    GM_ADDR actualKvseqlen_;
    GM_ADDR blockSparseMask_;
    GM_ADDR indexWorkspace_;
    GM_ADDR indexShapeWorkspace_;
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t bIdx_{0};            // 当前batch计算到的位置
    int32_t s1Idx_{0};           // 当前s1方向计算到的位置
    int32_t s2Idx_{0};           // 当前s2方向计算到的位置
    int32_t n1Idx_{0};           // 当前n1方向计算到的位置
    int32_t cur_q_seq_len_{0};   // 当前batch的q_seq_len
    int32_t cur_kv_seq_len_{0};  // 当前batch的kv_seq_len
    int32_t last_q_seq_sum_{0};  // 上一个batch的q_seq_len的累加和
    int32_t last_kv_seq_sum_{0}; // 上一个batch的kv_seq_len的累加和
    int32_t cube_core_idx_{0};   // 实际cube核的Idx
    int32_t cube_core_num_{0};   // cube核的数量
    int32_t base_m_{0};          // 每个cube核计算的s1方向的长度
    int32_t base_n_{0};          // 每个cube核计算的s2方向的长度
    int32_t q_block_num_{0};
    int32_t kv_block_num_{0};
    int32_t block_x_{0};
    int32_t block_y_{0};
    int32_t kv_ping_pong_idx_{0};
    int32_t max_q_seq_len_{0};
    int32_t max_kv_seq_len_{0};
    int32_t s1_outer_{0};
    int32_t s2_outer_{0};
    uint32_t deter_latin_r_{0};
    uint32_t deter_max_round_{0};
    uint32_t dkv_group_open_{0};
    ConstInfo const_info_;
    uint64_t loop_num_{0};
    int32_t block_global_idx{0};
    int32_t block_local_idx{0};
    uint64_t vaild_block_num{0};
    int32_t index_block_num_{16};

public:
    __aicore__ inline void Init(const TILING_CLASS* tilingData, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                GM_ADDR blockSparseMask, GM_ADDR workspace)
    {
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->cube_core_num_ = tilingData->cubeCoreNum;
        this->actualQseqlen_ = actualQseqlen;
        this->actualKvseqlen_ = actualKvseqlen;
        this->blockSparseMask_ = blockSparseMask;
        this->indexWorkspace_ = workspace + tilingData->indexWorkspaceOffset;
        this->indexShapeWorkspace_ = workspace + tilingData->indexShapeWorkspaceOffset;
        this->block_x_ = tilingData->BlockX;
        this->block_y_ = tilingData->BlockY;
        this->base_m_ = tilingData->baseM;
        this->base_n_ = tilingData->baseN;
        this->index_block_num_ = static_cast<int32_t>(tilingData->singleM / tilingData->baseM);
        if constexpr (INPUT_LAYOUT == TND) {
            UpdateSeqLenByBatch(bIdx_);
            max_q_seq_len_ = 0;
            max_kv_seq_len_ = 0;
            for (int32_t i = 0; i < batch_num_; i++) {
                int64_t q_seq_len = GetSeqLen(i, actualQseqlen_);
                int64_t kv_seq_len = GetSeqLen(i, actualKvseqlen_);
                max_q_seq_len_ = IMax(max_q_seq_len_, q_seq_len);
                max_kv_seq_len_ = IMax(max_kv_seq_len_, kv_seq_len);
            }
            q_block_num_ = CeilDiv(max_q_seq_len_, block_x_);
            kv_block_num_ = CeilDiv(max_kv_seq_len_, block_y_);
        } else {
            cur_q_seq_len_ = q_seq_len_;
            cur_kv_seq_len_ = kv_seq_len_;
            last_q_seq_sum_ = 0;
            last_kv_seq_sum_ = 0;
            max_q_seq_len_ = q_seq_len_;
            max_kv_seq_len_ = kv_seq_len_;
            q_block_num_ = CeilDiv(q_seq_len_, block_x_);
            kv_block_num_ = CeilDiv(kv_seq_len_, block_y_);
        }
        s1_outer_ = CeilDiv(max_q_seq_len_, base_m_);
        s2_outer_ = CeilDiv(max_kv_seq_len_, base_n_);
        if constexpr (DETERMINISTIC_ENABLE) {
            deter_max_round_ = CalcDeterMaxRound();
        }
        const_info_.q_head_num = q_head_num_;
        const_info_.kv_head_num = kv_head_num_;
        const_info_.block_x = block_x_;
        const_info_.block_y = block_y_;
        const_info_.head_dim = head_dim_;
        const_info_.q_block_num = q_block_num_;
        const_info_.kv_block_num = kv_block_num_;
        if ASCEND_IS_AIC {
            this->cube_core_idx_ = GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            this->cube_core_idx_ = GetBlockIdx() / 2;
        }
    }

    __aicore__ inline void GetRunTimeInfo(RunTimeInfo& runTimeInfo)
    {
        if constexpr (DETERMINISTIC_ENABLE) {
            GetRunTimeInfoDeter(runTimeInfo);
        } else {
            GetRunTimeInfoByIndex(runTimeInfo);
        }
    }

    __aicore__ inline uint32_t GetDeterMaxRound() const
    {
        return deter_max_round_;
    }

private:
    __aicore__ inline void GetRunTimeInfoByIndex(RunTimeInfo& runTimeInfo)
    {
        runTimeInfo.need_compute = 0;
        if (unlikely(loop_num_ == 0)) {
            this->vaild_block_num = ((__gm__ uint64_t*)indexShapeWorkspace_)[2];
            block_global_idx = cube_core_idx_;
            block_local_idx = 0;
        }
        while (true) {
            if (block_local_idx >= index_block_num_) {
                block_global_idx += cube_core_num_;
                block_local_idx = 0;
            }
            int64_t index_workspace_offset =
                static_cast<int64_t>(block_global_idx) * index_block_num_ + block_local_idx;
            if (index_workspace_offset >= static_cast<int64_t>(vaild_block_num)) {
                return;
            }
            block_local_idx++;
            loop_num_++;
            int32_t block_sparse_idx = ((__gm__ int32_t*)indexWorkspace_)[index_workspace_offset];

            int32_t b_idx;
            int32_t n1_idx;
            int32_t s1_block;
            int32_t s2_block;
            unravel_index(block_sparse_idx, b_idx, n1_idx, s1_block, s2_block);
            if (b_idx != bIdx_) {
                bIdx_ = b_idx;
                UpdateSeqLenByBatch(b_idx);
            }
            n1Idx_ = n1_idx;
            s1Idx_ = s1_block * base_m_;
            s2Idx_ = s2_block * base_n_;
            if (s1Idx_ >= cur_q_seq_len_ || s2Idx_ >= cur_kv_seq_len_) {
                // TND场景非法校验：s1Idx_ 或 s2Idx_ 超出当前 batch 的序列长度，却被SparseMask错误命中
                continue;
            }
            RunTimeInfoRecord(runTimeInfo, s1Idx_, s2Idx_, GetBlockLen(s1Idx_, cur_q_seq_len_, base_m_),
                              GetBlockLen(s2Idx_, cur_kv_seq_len_, base_n_));
            return;
        }
    }

    __aicore__ inline void RunTimeInfoRecord(RunTimeInfo& runTimeInfo, int32_t vaild_s1_idx, int32_t vaild_s2_idx,
                                             int32_t vaild_s1_len, int32_t vaild_s2_len)
    {
        int32_t n2Idx = n1Idx_ / q_group_;
        runTimeInfo.bIdx = bIdx_;
        runTimeInfo.last_q_seq_sum = last_q_seq_sum_;
        runTimeInfo.cur_q_seq_len = cur_q_seq_len_;
        runTimeInfo.s1Idx = vaild_s1_idx;
        runTimeInfo.s2Idx = vaild_s2_idx;
        runTimeInfo.n1Idx = n1Idx_;
        runTimeInfo.n2Idx = n2Idx;
        runTimeInfo.s1Len = vaild_s1_len;
        runTimeInfo.s2Len = vaild_s2_len;
        runTimeInfo.s1LenAlign = RoundUp(vaild_s1_len, 16);
        runTimeInfo.s2LenAlign = RoundUp(vaild_s2_len, 16);
        runTimeInfo.queryGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_, head_dim_,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.keyGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(last_kv_seq_sum_, cur_kv_seq_len_, kv_head_num_, head_dim_, runTimeInfo.bIdx,
                                         runTimeInfo.s2Idx, runTimeInfo.n2Idx);
        runTimeInfo.lseGmOffset =
            GetLseGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.sftgGmOffset =
            GetSftgGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_,
                                          runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.need_compute = 1;
    }

    __aicore__ inline uint32_t CalcDeterMaxRound()
    {
        int64_t m = s1_outer_;
        int64_t n = s2_outer_;
        int64_t k = cube_core_num_;
        if (m <= 0 || n <= 0 || k <= 0) {
            return 0;
        }
        if (q_group_ <= 1) {
            int64_t b = static_cast<int64_t>(batch_num_) * q_head_num_;
            k = IMin(k, b * m);
            return static_cast<uint32_t>(m * ICeil(b * n, k));
        }
        int64_t b = static_cast<int64_t>(batch_num_) * kv_head_num_;
        int64_t g = q_group_;
        k = IMin(IMin(k, b * g * m), b * n);
        int64_t R = IMax(IMax(ICeil(b * n * g, k), ICeil(n, m)), g);
        return static_cast<uint32_t>(R * m);
    }

    __aicore__ inline bool MapDenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r, int64_t& w,
                                         int64_t& x, int64_t& y)
    {
        k = IMin(k, b * m);
        if (j > k || j < 1 || r < 1) {
            return false;
        }
        int64_t p = (ICeil(r, m) - 1) * k + j;
        w = p % b;
        w = (w != 0) ? w : b;
        y = ICeil(p, b);
        int64_t y1 = y % m;
        y1 = (y1 != 0) ? y1 : m;
        int64_t r1 = r % m;
        r1 = (r1 != 0) ? r1 : m;
        x = y1 + r1 - 1;
        if (x > m) {
            x -= m;
        }
        return (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n);
    }

    __aicore__ inline bool MapGqaIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t core_id, int64_t round_id,
                                       int64_t g, int64_t& bn1, int64_t& x, int64_t& y)
    {
        k = IMin(IMin(k, b * g * m), b * n);
        int64_t R = IMax(IMax(ICeil(b * n * g, k), ICeil(n, m)), g);
        if (core_id < 1 || core_id > k || round_id < 1 || round_id > R * m) {
            return false;
        }
        int64_t ID = (core_id - 1) * R + ICeil(round_id, m);
        int64_t local_id = round_id % m;
        local_id = local_id != 0 ? local_id : m;
        if (ID > g * n * b) {
            return false;
        }
        int64_t N = b * g;
        int64_t b_id = ID % N;
        b_id = b_id != 0 ? b_id : N;
        b_id = ICeil(b_id, g);
        y = ICeil(ID, N);
        int64_t w = ID % g;
        w = w != 0 ? w : g;
        int64_t gcd = IGcd(N, R);
        int64_t t1 = R / gcd;
        int64_t t2 = N / gcd;
        int64_t t1_new = t1 * m;
        int64_t y1 = y % t1_new;
        y1 = y1 != 0 ? y1 : t1_new;
        int64_t offset = ICeil(y1, t1);
        if (t1_new < n) {
            int64_t n1 = (n % t1_new);
            n1 = n1 != 0 ? n1 : t1_new;
            if (y <= n - n1) {
                int64_t delta = ICeil(y, t1_new);
                ID += delta;
                if (ID > (delta - 1) * t2 * m * R + offset * t2 * R) {
                    ID -= t2 * R;
                }
                b_id = ID % N;
                b_id = b_id != 0 ? b_id : N;
                b_id = ICeil(b_id, g);
                w = ID % g;
                w = w != 0 ? w : g;
                y = ICeil(ID, N);
            }
        }
        x = local_id + offset - 1;
        if (x > m) {
            x -= m;
        }
        bn1 = w + (b_id - 1) * g;
        return (bn1 >= 1 && x >= 1 && x <= m && y >= 1 && y <= n);
    }

    __aicore__ inline bool LastValidInGroup(int64_t k, int64_t m, int64_t n, int64_t bflat, int64_t j, int64_t r,
                                            uint32_t latin_max)
    {
        if (m <= 0) {
            return true;
        }
        int64_t group_end = ICeil(r, m) * m;
        if (group_end > latin_max) {
            group_end = latin_max;
        }
        for (int64_t rr = r + 1; rr <= group_end; rr++) {
            int64_t w = 0;
            int64_t x = 0;
            int64_t y = 0;
            bool ok = false;
            if (q_group_ <= 1) {
                ok = MapDenseIndex(k, m, n, bflat, j, rr, w, x, y);
            } else {
                ok = MapGqaIndex(k, m, n, bflat, j, rr, q_group_, w, x, y);
            }
            if (!ok) {
                continue;
            }
            int64_t bn1 = w - 1;
            int32_t bIdx = static_cast<int32_t>(bn1 / q_head_num_);
            int32_t n1Idx = static_cast<int32_t>(bn1 % q_head_num_);
            int32_t s1Idx = static_cast<int32_t>((x - 1) * base_m_);
            int32_t s2Idx = static_cast<int32_t>((y - 1) * base_n_);
            int32_t cur_q = q_seq_len_;
            int32_t cur_kv = kv_seq_len_;
            if constexpr (INPUT_LAYOUT == TND) {
                if (bIdx < 0 || bIdx >= batch_num_) {
                    continue;
                }
                cur_q = GetSeqLen(bIdx, actualQseqlen_);
                cur_kv = GetSeqLen(bIdx, actualKvseqlen_);
            }
            if (s1Idx >= 0 && s2Idx >= 0 && s1Idx < cur_q && s2Idx < cur_kv &&
                IsValidBlock(const_info_, bIdx, n1Idx, s1Idx / block_x_, s2Idx / block_y_, blockSparseMask_)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void GetRunTimeInfoDeter(RunTimeInfo& runTimeInfo)
    {
        runTimeInfo.need_compute = 0;
        deter_latin_r_++;
        int64_t m = s1_outer_;
        int64_t n = s2_outer_;
        int64_t k = cube_core_num_;
        int64_t j = cube_core_idx_ + 1;
        int64_t r = static_cast<int64_t>(deter_latin_r_);
        if (deter_latin_r_ > deter_max_round_ || m <= 0 || n <= 0 || k <= 0) {
            return;
        }
        int64_t w = 0;
        int64_t x = 0;
        int64_t y = 0;
        bool ok = false;
        int64_t bflat = q_group_ <= 1 ? static_cast<int64_t>(batch_num_) * q_head_num_ :
                                        static_cast<int64_t>(batch_num_) * kv_head_num_;
        if (q_group_ <= 1) {
            ok = MapDenseIndex(k, m, n, bflat, j, r, w, x, y);
        } else {
            ok = MapGqaIndex(k, m, n, bflat, j, r, q_group_, w, x, y);
        }
        if (!ok) {
            return;
        }
        int64_t bn1 = w - 1;
        int32_t bIdx = static_cast<int32_t>(bn1 / q_head_num_);
        int32_t n1Idx = static_cast<int32_t>(bn1 % q_head_num_);
        int32_t n2Idx = n1Idx / q_group_;
        int32_t s1Idx = static_cast<int32_t>((x - 1) * base_m_);
        int32_t s2Idx = static_cast<int32_t>((y - 1) * base_n_);
        int32_t cur_q = q_seq_len_;
        int32_t cur_kv = kv_seq_len_;
        int32_t last_q = 0;
        int32_t last_kv = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            if (bIdx < 0 || bIdx >= batch_num_) {
                return;
            }
            cur_q = GetSeqLen(bIdx, actualQseqlen_);
            cur_kv = GetSeqLen(bIdx, actualKvseqlen_);
            last_q = bIdx > 0 ? GetSeqTotalLen(bIdx - 1, actualQseqlen_) : 0;
            last_kv = bIdx > 0 ? GetSeqTotalLen(bIdx - 1, actualKvseqlen_) : 0;
        } else if (bIdx < 0 || bIdx >= batch_num_) {
            return;
        }
        if (s1Idx >= cur_q || s2Idx >= cur_kv || s1Idx < 0 || s2Idx < 0) {
            return;
        }
        if (!IsValidBlock(const_info_, bIdx, n1Idx, s1Idx / block_x_, s2Idx / block_y_, blockSparseMask_)) {
            return;
        }
        int32_t s1Len = GetBlockLen(s1Idx, cur_q, base_m_);
        int32_t s2Len = GetBlockLen(s2Idx, cur_kv, base_n_);
        runTimeInfo.bIdx = bIdx;
        runTimeInfo.last_q_seq_sum = last_q;
        runTimeInfo.last_kv_seq_sum = last_kv;
        runTimeInfo.cur_q_seq_len = cur_q;
        runTimeInfo.cur_kv_seq_len = cur_kv;
        runTimeInfo.s1Idx = s1Idx;
        runTimeInfo.s2Idx = s2Idx;
        runTimeInfo.n1Idx = n1Idx;
        runTimeInfo.n2Idx = n2Idx;
        runTimeInfo.s1Len = s1Len;
        runTimeInfo.s2Len = s2Len;
        runTimeInfo.s1LenAlign = RoundUp(s1Len, 16);
        runTimeInfo.s2LenAlign = RoundUp(s2Len, 16);
        runTimeInfo.queryGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(last_q, cur_q, q_head_num_, head_dim_, bIdx, s1Idx, n1Idx);
        runTimeInfo.keyGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(last_kv, cur_kv, kv_head_num_, head_dim_, bIdx, s2Idx, n2Idx);
        runTimeInfo.lseGmOffset = GetLseGmOffset<INPUT_LAYOUT>(last_q, cur_q, q_head_num_, bIdx, s1Idx, n1Idx);
        runTimeInfo.sftgGmOffset = GetSftgGmOffset<INPUT_LAYOUT>(last_q, cur_q, q_head_num_, bIdx, s1Idx, n1Idx);
        runTimeInfo.need_compute = 1;
        if (!dkv_group_open_) {
            kv_ping_pong_idx_ = 1 - kv_ping_pong_idx_;
            runTimeInfo.need_copy_kv = 1;
            dkv_group_open_ = 1;
        } else {
            runTimeInfo.need_copy_kv = 0;
        }
        runTimeInfo.kv_ping_pong_idx = kv_ping_pong_idx_;
        bool last = LastValidInGroup(k, m, n, bflat, j, r, deter_max_round_);
        runTimeInfo.is_singlekv_last = last ? 1 : 0;
        if (last) {
            dkv_group_open_ = 0;
        }
    }

    __aicore__ inline void unravel_index(int32_t flat_index, int32_t& b, int32_t& n, int32_t& s1_block,
                                         int32_t& s2_block)
    {
        // 将 block index 展平下标还原为 (batch, q_head, q_block, kv_block)
        s1_block = flat_index % q_block_num_;
        int32_t remaining = flat_index / q_block_num_;
        s2_block = remaining % kv_block_num_;
        remaining = remaining / kv_block_num_;
        n = remaining % q_head_num_;
        b = remaining / q_head_num_;
    }

    __aicore__ inline void UpdateSeqLenByBatch(int32_t b_idx)
    {
        if constexpr (INPUT_LAYOUT != TND) {
            return;
        }
        cur_q_seq_len_ = GetSeqLen(b_idx, actualQseqlen_);
        cur_kv_seq_len_ = GetSeqLen(b_idx, actualKvseqlen_);
        last_q_seq_sum_ = b_idx > 0 ? GetSeqTotalLen(b_idx - 1, actualQseqlen_) : 0;
        last_kv_seq_sum_ = b_idx > 0 ? GetSeqTotalLen(b_idx - 1, actualKvseqlen_) : 0;
    }
};

} // namespace BSA_ARC35
