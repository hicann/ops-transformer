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

class SingleBlock {
    /*
     *  Single块内只遍历S1方向。
     */
public:
    __aicore__ inline void Reset(const ConstInfo &const_info, __gm__ uint8_t *block_sparse_mask, const int32_t b_idx,
                                 const int32_t n1_idx, const int32_t s1_idx, const int32_t s2_idx, const int32_t s1_len,
                                 int32_t &base_s1_start_idx, int32_t &base_s1_len)
    {
        /*
         * 功能：设置SingleBlock的起始索引和结束索引，同时更新base_s1_start_idx和base_s1_len。
         * base_s1_start_idx：表示下一个需要计算的base块的s1方向的起始索引。
         * base_s1_len：表示下一个需要计算的base块的s1方向的长度。
         * 默认起始位置必为有效块。
         */

        this->is_finish_ = false;
        this->s1_start_idx_ = s1_idx;
        this->s1_end_idx_ = s1_idx + s1_len;
        this->Update(const_info, block_sparse_mask, b_idx, n1_idx, s2_idx, base_s1_start_idx, base_s1_len);
    }

    __aicore__ inline void RecordRunTimeInfo(const RunTimeInfo &runTimeInfo)
    {
        this->runTimeInfoBak_.bIdx = runTimeInfo.bIdx;
        this->runTimeInfoBak_.last_q_seq_sum = runTimeInfo.last_q_seq_sum;
        this->runTimeInfoBak_.last_kv_seq_sum = runTimeInfo.last_kv_seq_sum;
        this->runTimeInfoBak_.cur_q_seq_len = runTimeInfo.cur_q_seq_len;
        this->runTimeInfoBak_.cur_kv_seq_len = runTimeInfo.cur_kv_seq_len;
        this->runTimeInfoBak_.s1Idx = runTimeInfo.s1Idx;
        this->runTimeInfoBak_.s2Idx = runTimeInfo.s2Idx;
        this->runTimeInfoBak_.n1Idx = runTimeInfo.n1Idx;
        this->runTimeInfoBak_.n2Idx = runTimeInfo.n2Idx;
        this->runTimeInfoBak_.s1Len = runTimeInfo.s1Len;
        this->runTimeInfoBak_.s2Len = runTimeInfo.s2Len;
        this->runTimeInfoBak_.s1LenAlign = runTimeInfo.s1LenAlign;
        this->runTimeInfoBak_.s2LenAlign = runTimeInfo.s2LenAlign;
        this->runTimeInfoBak_.kv_ping_pong_idx = runTimeInfo.kv_ping_pong_idx;
    }

    template <uint32_t INPUT_LAYOUT>
    __aicore__ inline void UpdateRunTimeInfo(const ConstInfo &const_info, __gm__ uint8_t *block_sparse_mask,
                                             RunTimeInfo &runTimeInfo)
    {
        int32_t base_s1_start_idx, base_s1_len;
        this->Update(const_info, block_sparse_mask, this->runTimeInfoBak_.bIdx, this->runTimeInfoBak_.n1Idx,
                     this->runTimeInfoBak_.s2Idx, base_s1_start_idx, base_s1_len);

        runTimeInfo.bIdx = this->runTimeInfoBak_.bIdx;
        runTimeInfo.last_q_seq_sum = this->runTimeInfoBak_.last_q_seq_sum;
        runTimeInfo.last_kv_seq_sum = this->runTimeInfoBak_.last_kv_seq_sum;
        runTimeInfo.cur_q_seq_len = this->runTimeInfoBak_.cur_q_seq_len;
        runTimeInfo.cur_kv_seq_len = this->runTimeInfoBak_.cur_kv_seq_len;
        runTimeInfo.s1Idx = base_s1_start_idx;
        runTimeInfo.s2Idx = this->runTimeInfoBak_.s2Idx;
        runTimeInfo.n1Idx = this->runTimeInfoBak_.n1Idx;
        runTimeInfo.n2Idx = this->runTimeInfoBak_.n2Idx;
        runTimeInfo.s1Len = base_s1_len;
        runTimeInfo.s2Len = this->runTimeInfoBak_.s2Len;
        runTimeInfo.s1LenAlign = RoundUp(base_s1_len, 16);
        runTimeInfo.s2LenAlign = this->runTimeInfoBak_.s2LenAlign;
        runTimeInfo.queryGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, const_info.q_head_num,
                                         const_info.head_dim, runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.keyGmOffset = GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_kv_seq_sum, runTimeInfo.cur_kv_seq_len,
                                                               const_info.kv_head_num, const_info.head_dim,
                                                               runTimeInfo.bIdx, runTimeInfo.s2Idx, runTimeInfo.n2Idx);
        runTimeInfo.lseGmOffset =
            GetLseGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, const_info.q_head_num,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.sftgGmOffset =
            GetSftgGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, const_info.q_head_num,
                                          runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.need_compute = 1;
        runTimeInfo.need_copy_kv = 0;
        runTimeInfo.is_singlekv_last = is_finish_;
        runTimeInfo.kv_ping_pong_idx = this->runTimeInfoBak_.kv_ping_pong_idx;
    }

    __aicore__ inline bool IsFinish()
    {
        return is_finish_;
    }

    __aicore__ inline void SetBaseBlock(uint32_t s1_base_size)
    {
        this->s1_base_size_ = s1_base_size;
    }

    __aicore__ inline uint32_t CountValidBase(const ConstInfo &const_info, __gm__ uint8_t *block_sparse_mask,
                                              const int32_t b_idx, const int32_t n1_idx, const int32_t s2_idx,
                                              const int32_t s1_idx, const int32_t s1_len)
    {
        int32_t base_s1_start_idx;
        int32_t base_s1_len;
        this->Reset(const_info, block_sparse_mask, b_idx, n1_idx, s1_idx, s2_idx, s1_len, base_s1_start_idx,
                    base_s1_len);
        uint32_t count = 1;
        while (!this->is_finish_) {
            this->Update(const_info, block_sparse_mask, b_idx, n1_idx, s2_idx, base_s1_start_idx, base_s1_len);
            count++;
        }
        return count;
    }

private:
    __aicore__ inline void Update(const ConstInfo &const_info, __gm__ uint8_t *block_sparse_mask, const int32_t b_idx,
                                  const int32_t n1_idx, const int32_t s2Idx, int32_t &base_s1_start_idx,
                                  int32_t &base_s1_len)
    {
        /*
         * 功能：返回有效base块的起始地址和长度。并判断是否遍历完所有的s1方向。
         */
        base_s1_start_idx = this->s1_start_idx_;
        base_s1_len = GetBlockLen(this->s1_start_idx_, this->s1_end_idx_, s1_base_size_);
        this->s1_start_idx_ += base_s1_len;

        if (this->s1_start_idx_ >= this->s1_end_idx_) {
            // 下一块遍历完，提前退出
            this->is_finish_ = true;
            return;
        }

        // 判断下一块是否是有效块
        int32_t q_block_idx = this->s1_start_idx_ / const_info.block_x;
        int32_t kv_block_idx = s2Idx / const_info.block_y;
        while (true) {
            if (IsValidBlock(const_info, b_idx, n1_idx, q_block_idx, kv_block_idx, block_sparse_mask)) {
                break;
            }
            int32_t tmp_s1_len = GetBlockLen(this->s1_start_idx_, this->s1_end_idx_, s1_base_size_);
            this->s1_start_idx_ += tmp_s1_len;
            q_block_idx = this->s1_start_idx_ / const_info.block_x;
            if (this->s1_start_idx_ >= this->s1_end_idx_) {
                break;
            }
        }

        if (this->s1_start_idx_ >= this->s1_end_idx_) {
            this->is_finish_ = true;
        }
    }

    int32_t s1_start_idx_{0};
    int32_t s1_end_idx_{0};
    int32_t s1_base_size_{0};
    bool is_finish_{true};
    RunTimeInfo runTimeInfoBak_;
};

template <typename BSA_TYPE>
class AddrComputeModule {
protected:
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;
    GM_ADDR actualQseqlen_;
    GM_ADDR actualKvseqlen_;
    GM_ADDR blockSparseMask_;
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t bIdx_{0};             // 当前batch计算到的位置
    int32_t s1Idx_{0};            // 当前s1方向计算到的位置
    int32_t s2Idx_{0};            // 当前s2方向计算到的位置
    int32_t n1Idx_{0};            // 当前n1方向计算到的位置
    int32_t cur_q_seq_len_{0};    // 当前batch的q_seq_len
    int32_t cur_kv_seq_len_{0};   // 当前batch的kv_seq_len
    int32_t last_q_seq_sum_{0};   // 上一个batch的q_seq_len的累加和
    int32_t last_kv_seq_sum_{0};  // 上一个batch的kv_seq_len的累加和
    int32_t cube_core_idx_{0};    // 实际cube核的Idx
    int32_t cube_core_num_{0};    // cube核的数量
    int32_t base_m_{0};           // 每个cube核计算的s1方向的长度
    int32_t base_n_{0};           // 每个cube核计算的s2方向的长度
    int32_t first_loop_{0};       // 当前循环次数(包括sparse跳过的计算)
    int32_t current_cube_idx_{0}; // 当前cube计算到的位置
    int32_t q_block_num_{0};
    int32_t kv_block_num_{0};
    int32_t block_x_{0};
    int32_t block_y_{0};
    int32_t single_m_{0};
    int32_t kv_ping_pong_idx_{0};
    int32_t max_q_seq_len_{0};
    int32_t max_kv_seq_len_{0};
    int32_t s1_outer_{0};
    int32_t s2_outer_{0};
    uint32_t deter_latin_r_{0};
    uint32_t deter_max_round_{0};
    uint32_t dkv_group_open_{0};
    SingleBlock single_block_;
    ConstInfo const_info_;

public:
    __aicore__ inline void Init(const TILING_CLASS *tilingData, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                GM_ADDR blockSparseMask)
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
        this->block_x_ = tilingData->BlockX;
        this->block_y_ = tilingData->BlockY;
        this->base_m_ = tilingData->baseM;
        this->base_n_ = tilingData->baseN;
        this->single_m_ = tilingData->singleM;
        single_block_.SetBaseBlock(this->base_m_);
        if constexpr (INPUT_LAYOUT == TND) {
            UpdateSeqLen();
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

    __aicore__ inline uint32_t GetDeterMaxRound() const
    {
        return deter_max_round_;
    }

    __aicore__ inline void GetRunTimeInfo(RunTimeInfo &runTimeInfo)
    {
        if constexpr (DETERMINISTIC_ENABLE) {
            GetRunTimeInfoDeter(runTimeInfo);
            return;
        }
        runTimeInfo.need_compute = 0;

        if (!single_block_.IsFinish()) {
            // 如果singleBlock未计算完，先计算singleBlock内的内容
            single_block_.UpdateRunTimeInfo<INPUT_LAYOUT>(const_info_, blockSparseMask_, runTimeInfo);
            return;
        }

        while (true) {
            if (InitStartIdx()) {
                break;
            }
            int32_t vaild_s1_idx;
            int32_t vaild_s1_len;
            int32_t s2_len = GetBlockLen(s2Idx_, cur_kv_seq_len_, base_n_);

            if (IsValidSingleBlock(vaild_s1_idx, vaild_s1_len)) {
                // 如果singleBlock内全是无效的base块，则不记录
                RunTimeInfoRecord(runTimeInfo, vaild_s1_idx, s2Idx_, vaild_s1_len, s2_len);
            }

            if (current_cube_idx_ && current_cube_idx_ % cube_core_num_ == 0) {
                // 所有cube核分配到任务，暂时跳出循环，等下次分配
                current_cube_idx_ = 0;
                break;
            }
        }
    }

protected:
    __aicore__ inline bool IsValidSingleBlock(int32_t &vaild_s1_idx, int32_t &vaild_s1_len)
    {
        /*
         * 功能：判断singleBlock内是否存在有效base块，并返回第一块有效base块的起始idx和长度
         */
        int32_t s1_len = GetBlockLen(s1Idx_, cur_q_seq_len_, single_m_);
        int32_t s1_end_idx = s1Idx_ + s1_len;
        int32_t block_y_idx = s2Idx_ / block_y_;
        bool find_vaild_block = false;

        for (int32_t s1_idx = s1Idx_; s1_idx < s1_end_idx; s1_idx += block_x_) {
            int32_t block_x_idx = s1_idx / block_x_;
            if (IsValidBlock(const_info_, bIdx_, n1Idx_, block_x_idx, block_y_idx, blockSparseMask_)) {
                find_vaild_block = true;
                vaild_s1_idx = s1_idx;
                vaild_s1_len = s1_end_idx - s1_idx;
            }
            if (find_vaild_block) {
                break;
            }
        }

        return find_vaild_block;
    }

    __aicore__ inline void UpdateSeqLen()
    {
        if constexpr (INPUT_LAYOUT != TND) {
            return;
        }
        cur_q_seq_len_ = GetSeqLen(bIdx_, actualQseqlen_);
        cur_kv_seq_len_ = GetSeqLen(bIdx_, actualKvseqlen_);
        while ((cur_q_seq_len_ == 0 || cur_kv_seq_len_ == 0) && bIdx_ < batch_num_ - 1) {
            bIdx_++;
            cur_q_seq_len_ = GetSeqLen(bIdx_, actualQseqlen_);
            cur_kv_seq_len_ = GetSeqLen(bIdx_, actualKvseqlen_);
        }
        last_q_seq_sum_ = bIdx_ > 0 ? GetSeqTotalLen(bIdx_ - 1, actualQseqlen_) : 0;
        last_kv_seq_sum_ = bIdx_ > 0 ? GetSeqTotalLen(bIdx_ - 1, actualKvseqlen_) : 0;
    }

    __aicore__ inline bool InitStartIdx()
    {
        // 遍历顺序 s1->s2->n1->batch
        int32_t recoderS1 = GetBlockLen(s1Idx_, cur_q_seq_len_, single_m_);
        int32_t recoderS2 = GetBlockLen(s2Idx_, cur_kv_seq_len_, base_n_);

        if (unlikely(first_loop_ == 0)) {
            first_loop_ = 1;
            return false;
        }

        if (s1Idx_ < cur_q_seq_len_ - recoderS1) {
            s1Idx_ += recoderS1;
            return false;
        }

        if (s2Idx_ < cur_kv_seq_len_ - recoderS2) {
            s1Idx_ = 0;
            s2Idx_ += recoderS2;
            return false;
        }

        if (n1Idx_ < q_head_num_ - 1) {
            s1Idx_ = 0;
            s2Idx_ = 0;
            n1Idx_++;
            return false;
        }

        if (bIdx_ < batch_num_ - 1) {
            s1Idx_ = 0;
            s2Idx_ = 0;
            n1Idx_ = 0;
            bIdx_++;
            if constexpr (INPUT_LAYOUT == TND) {
                UpdateSeqLen();
            }
            return false;
        }

        return true;
    }

private:
    __aicore__ inline void RunTimeInfoRecord(RunTimeInfo &runTimeInfo, int32_t vaild_s1_idx, int32_t vaild_s2_idx,
                                             int32_t vaild_s1_len, int32_t vaild_s2_len)
    {
        if (current_cube_idx_ % cube_core_num_ != cube_core_idx_) {
            current_cube_idx_++;
            return;
        }
        int32_t base_s1_idx;
        int32_t base_s1_len;
        single_block_.Reset(const_info_, blockSparseMask_, bIdx_, n1Idx_, vaild_s1_idx, vaild_s2_idx, vaild_s1_len,
                            base_s1_idx, base_s1_len);
        int32_t n2Idx = n1Idx_ / q_group_;

        runTimeInfo.bIdx = bIdx_;
        runTimeInfo.last_q_seq_sum = last_q_seq_sum_;
        runTimeInfo.last_kv_seq_sum = last_kv_seq_sum_;
        runTimeInfo.cur_q_seq_len = cur_q_seq_len_;
        runTimeInfo.cur_kv_seq_len = cur_kv_seq_len_;
        runTimeInfo.s1Idx = base_s1_idx;
        runTimeInfo.s2Idx = vaild_s2_idx;
        runTimeInfo.n1Idx = n1Idx_;
        runTimeInfo.n2Idx = n2Idx;
        runTimeInfo.s1Len = base_s1_len;
        runTimeInfo.s2Len = vaild_s2_len;
        runTimeInfo.s1LenAlign = RoundUp(base_s1_len, 16);
        runTimeInfo.s2LenAlign = RoundUp(vaild_s2_len, 16);
        runTimeInfo.queryGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_, head_dim_,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.keyGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_kv_seq_sum, runTimeInfo.cur_kv_seq_len, kv_head_num_,
                                         head_dim_, runTimeInfo.bIdx, runTimeInfo.s2Idx, runTimeInfo.n2Idx);
        runTimeInfo.lseGmOffset =
            GetLseGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.sftgGmOffset =
            GetSftgGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_,
                                          runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.need_compute = 1;
        runTimeInfo.need_copy_kv = 1;
        runTimeInfo.kv_ping_pong_idx = kv_ping_pong_idx_;
        runTimeInfo.is_singlekv_last = single_block_.IsFinish();
        single_block_.RecordRunTimeInfo(runTimeInfo);
        kv_ping_pong_idx_ = 1 - kv_ping_pong_idx_;
        current_cube_idx_++;
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

    __aicore__ inline bool MapDenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r, int64_t &w,
                                         int64_t &x, int64_t &y)
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
                                       int64_t g, int64_t &bn1, int64_t &x, int64_t &y)
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

    __aicore__ inline void GetRunTimeInfoDeter(RunTimeInfo &runTimeInfo)
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
};

template <typename BSA_TYPE>
class AddrComputeMaxTaskModule : public AddrComputeModule<BSA_TYPE> {
public:
    __aicore__ inline uint32_t PrecomputeMaxTaskNum()
    {
        if constexpr (BSA_TYPE::deterministic_enable) {
            return this->GetDeterMaxRound();
        }
        constexpr int32_t MAX_CUBE_CORES = 64;
        uint32_t counts[MAX_CUBE_CORES];
        int32_t core_num = this->cube_core_num_ < MAX_CUBE_CORES ? this->cube_core_num_ : MAX_CUBE_CORES;
        for (int32_t i = 0; i < core_num; i++) {
            counts[i] = 0;
        }

        int32_t assign_idx = 0;
        while (true) {
            if (this->InitStartIdx()) {
                break;
            }
            int32_t valid_s1_idx;
            int32_t valid_s1_len;
            if (this->IsValidSingleBlock(valid_s1_idx, valid_s1_len)) {
                uint32_t n_base = CountValidBaseInSingle(valid_s1_idx, valid_s1_len);
                int32_t core = assign_idx % this->cube_core_num_;
                if (core < MAX_CUBE_CORES) {
                    counts[core] += n_base;
                }
                assign_idx++;
            }
        }

        uint32_t max_task_num = 0;
        for (int32_t i = 0; i < core_num; i++) {
            if (counts[i] > max_task_num) {
                max_task_num = counts[i];
            }
        }
        return max_task_num;
    }

private:
    __aicore__ inline uint32_t CountValidBaseInSingle(int32_t valid_s1_idx, int32_t valid_s1_len)
    {
        return this->single_block_.CountValidBase(this->const_info_, this->blockSparseMask_, this->bIdx_, this->n1Idx_,
                                                  this->s2Idx_, valid_s1_idx, valid_s1_len);
    }
};
} // namespace BSA_ARC35
