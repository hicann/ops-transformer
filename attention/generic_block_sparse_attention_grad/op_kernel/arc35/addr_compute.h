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
 * \file addr_compute.h
 * \brief Metadata-driven task scheduling for GSAG (design §3.1.2 / §4.2).
 *        Atomic unit (b,n2,j) stays on one AIC; g=0..G-1 continuous.
 *        Each sparse J block (BlockY) is further tiled by baseM (S1) and baseN (S2<=128).
 *        Softmax uses global LSE (exp(S-LSE)); S2 sub-tiles match full-BlockY numerically.
 */

#pragma once
#include "common_header.h"
#include "../../../generic_block_sparse_attention_grad_metadata/op_kernel/generic_block_sparse_attention_grad_metadata.h"

using namespace AscendC;
using namespace optiling;

namespace GSAG_ARC35 {

template <typename GSAG_TYPE>
class AddrComputeModule {
    using TILING_CLASS = typename GSAG_TYPE::tiling_class;
    static constexpr uint32_t INPUT_LAYOUT = GSAG_TYPE::input_layout;

public:
    __aicore__ inline void Init(const TILING_CLASS *tilingData, __gm__ uint8_t *cuSeqQ, __gm__ uint8_t *cuSeqKv,
                                __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sparseBlockIdx,
                                __gm__ uint8_t *sparseBlockCount, __gm__ uint8_t *metadata)
    {
        tilingData_ = tilingData;
        cuSeqQ_ = cuSeqQ;
        cuSeqKv_ = cuSeqKv;
        sequsedQ_ = sequsedQ;
        sequsedKv_ = sequsedKv;
        sparseBlockIdx_ = sparseBlockIdx;
        sparseBlockCount_ = sparseBlockCount;
        metadata_ = reinterpret_cast<__gm__ int32_t *>(metadata);

        cubeCoreIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            cubeCoreIdx_ = GetBlockIdx() / 2;
        }

        constInfo_.q_head_num = static_cast<int32_t>(tilingData->qHeadNum);
        constInfo_.kv_head_num = static_cast<int32_t>(tilingData->kvHeadNum);
        constInfo_.block_x = static_cast<int32_t>(tilingData->BlockX);
        constInfo_.block_y = static_cast<int32_t>(tilingData->BlockY);
        constInfo_.head_dim = static_cast<int32_t>(tilingData->headDim);
        constInfo_.max_s1 = static_cast<int32_t>(tilingData->maxS1);
        constInfo_.num_j = static_cast<int32_t>(tilingData->numJ);
        constInfo_.group_size = static_cast<int32_t>(tilingData->qGroup);
        baseM_ = static_cast<int32_t>(tilingData->baseM);
        baseN_ = static_cast<int32_t>(tilingData->baseN);

        taskStart_ = static_cast<int32_t>(metadata_[CORE_TASK_START_OFFSET + cubeCoreIdx_]);
        taskEnd_ = static_cast<int32_t>(metadata_[CORE_TASK_END_OFFSET + cubeCoreIdx_]);
        cursor_ = taskStart_;
        lastB_ = -1;
        lastN2_ = -1;
        lastJ_ = -1;
        lastS2Offset_ = -1;
        kvPingPong_ = 0;
        s1Offset_ = 0;
        s1Remain_ = 0;
        s2Offset_ = 0;
        s2Remain_ = 0;
        curCount_ = 0;
        inMetaTask_ = false;
    }

    __aicore__ inline void GetRunTimeInfo(RunTimeInfo &info)
    {
        info = {};
        if (!inMetaTask_) {
            if (cursor_ >= taskEnd_) {
                info.need_compute = 0;
                return;
            }
            if (!LoadMetaTask()) {
                info.need_compute = 0;
                return;
            }
        }

        const int32_t n = s2Remain_ > baseN_ ? baseN_ : s2Remain_;
        const int32_t m = s1Remain_ > baseM_ ? baseM_ : s1Remain_;
        FillRunTimeInfo(info, m, n);

        // Inner: S1 for current S2 tile; outer: advance S2 within sparse J (ends on last S2 tile).
        s1Offset_ += m;
        s1Remain_ -= m;
        if (s1Remain_ <= 0) {
            s2Offset_ += n;
            s2Remain_ -= n;
            if (s2Remain_ <= 0) {
                inMetaTask_ = false;
                cursor_++;
            } else {
                s1Offset_ = 0;
                s1Remain_ = curCount_;
            }
        }
    }

private:
    __aicore__ inline bool LoadMetaTask()
    {
        while (cursor_ < taskEnd_) {
            const uint32_t base = TASK_LIST_OFFSET + static_cast<uint32_t>(cursor_) * TASK_ENTRY_SIZE;
            curB_ = static_cast<int32_t>(metadata_[base + TASK_B]);
            curN2_ = static_cast<int32_t>(metadata_[base + TASK_N2]);
            curJ_ = static_cast<int32_t>(metadata_[base + TASK_J]);
            curG_ = static_cast<int32_t>(metadata_[base + TASK_G]);

            if constexpr (INPUT_LAYOUT == TND) {
                curQPrefix_ = GetSeqPrefix(curB_, cuSeqQ_);
                curKvPrefix_ = GetSeqPrefix(curB_, cuSeqKv_);
                curActS1_ = GetSeqUsedLen(curB_, sequsedQ_, cuSeqQ_);
                curActS2_ = GetSeqUsedLen(curB_, sequsedKv_, cuSeqKv_);
            } else {
                curQPrefix_ = 0;
                curKvPrefix_ = 0;
                curActS1_ = tilingData_->qSeqLen;
                curActS2_ = tilingData_->kvSeqLen;
            }

            const int64_t countOffset =
                (static_cast<int64_t>(curB_) * constInfo_.kv_head_num + curN2_) * constInfo_.num_j + curJ_;
            curCount_ = reinterpret_cast<__gm__ int32_t *>(sparseBlockCount_)[countOffset];
            // Guard against corrupt / out-of-range counts (avoids GetValue into -1 pad → AIC 264).
            if (curCount_ < 0) {
                curCount_ = 0;
            } else if (curCount_ > constInfo_.max_s1) {
                curCount_ = constInfo_.max_s1;
            }
            curIdxBase_ = countOffset * static_cast<int64_t>(constInfo_.max_s1);

            // Sparse J span (BlockY); compute tiles further split by baseN (<=128).
            curBlockS2Start_ = curJ_ * constInfo_.block_y;
            curKvBlockLen_ = constInfo_.block_y;
            if (curBlockS2Start_ + curKvBlockLen_ > curActS2_) {
                curKvBlockLen_ = static_cast<int32_t>(curActS2_ - curBlockS2Start_);
            }
            if (curKvBlockLen_ < 0) {
                curKvBlockLen_ = 0;
            }

            if (curCount_ > 0 && curKvBlockLen_ > 0) {
                s2Offset_ = 0;
                s2Remain_ = curKvBlockLen_;
                s1Offset_ = 0;
                s1Remain_ = curCount_;
                inMetaTask_ = true;
                return true;
            }
            cursor_++;
        }
        return false;
    }

    __aicore__ inline void FillRunTimeInfo(RunTimeInfo &info, int32_t m, int32_t n)
    {
        const int32_t n1Cur = curN2_ * constInfo_.group_size + curG_;
        const int32_t tileS2Start = curBlockS2Start_ + s2Offset_;
        const bool sameKvTile = (curB_ == lastB_ && curN2_ == lastN2_ && curJ_ == lastJ_ && s2Offset_ == lastS2Offset_);
        if (!sameKvTile) {
            kvPingPong_ = 1 - kvPingPong_;
            lastB_ = curB_;
            lastN2_ = curN2_;
            lastJ_ = curJ_;
            lastS2Offset_ = s2Offset_;
        }

        info.taskId = cursor_;
        info.bIdx = curB_;
        info.n2Idx = curN2_;
        info.gIdx = curG_;
        info.n1Idx = n1Cur;
        info.kvBlockIdx = curJ_;
        info.s2Idx = tileS2Start;
        info.s1Idx = s1Offset_;
        info.last_q_seq_sum = static_cast<int32_t>(curQPrefix_);
        info.last_kv_seq_sum = static_cast<int32_t>(curKvPrefix_);
        info.cur_q_seq_len = static_cast<int32_t>(curActS1_);
        info.cur_kv_seq_len = static_cast<int32_t>(curActS2_);
        info.s1Len = m;
        info.s2Len = n;
        info.s1LenAlign = RoundUp(static_cast<int64_t>(m), static_cast<int64_t>(C0_SIZE));
        info.s2LenAlign = RoundUp(static_cast<int64_t>(n), static_cast<int64_t>(C0_SIZE));
        info.sparseIdxOffset = curIdxBase_ + s1Offset_;
        info.sparseCount = m;
        info.use_sparse_gather = 1;
        // First S1 of an S2 tile: reload this KV slice and init L0C (Fixpipe AtomicAdd across g).
        info.need_copy_kv = (s1Offset_ == 0) ? 1 : 0;
        info.kv_ping_pong_idx = kvPingPong_;
        info.mask_type = static_cast<int32_t>(tilingData_->maskType);
        info.keyGmOffset = GetQKVGmOffset<INPUT_LAYOUT>(curKvPrefix_, curActS2_, constInfo_.kv_head_num,
                                                        constInfo_.head_dim, curB_, tileS2Start, curN2_);
        info.lseGmOffset = GetLseGmOffset<INPUT_LAYOUT>(curQPrefix_, curActS1_, constInfo_.q_head_num, curB_, 0, n1Cur);
        info.sftgGmOffset =
            GetSftgGmOffset<INPUT_LAYOUT>(curQPrefix_, curActS1_, constInfo_.q_head_num, curB_, 0, n1Cur);
        info.queryGmOffset = 0;
        info.need_compute = 1;

        // Dump dK/dV after last S1 of current S2 tile. Meta-task ends only after last S2 of J.
        const bool lastS1OfS2 = (s1Remain_ - m) <= 0;
        info.is_singlekv_last = lastS1OfS2 ? 1 : 0;
    }

    const TILING_CLASS *tilingData_{nullptr};
    __gm__ uint8_t *cuSeqQ_{nullptr};
    __gm__ uint8_t *cuSeqKv_{nullptr};
    __gm__ uint8_t *sequsedQ_{nullptr};
    __gm__ uint8_t *sequsedKv_{nullptr};
    __gm__ uint8_t *sparseBlockIdx_{nullptr};
    __gm__ uint8_t *sparseBlockCount_{nullptr};
    __gm__ int32_t *metadata_{nullptr};
    ConstInfo constInfo_{};
    int32_t cubeCoreIdx_{0};
    int32_t taskStart_{0};
    int32_t taskEnd_{0};
    int32_t cursor_{0};
    int32_t lastB_{-1};
    int32_t lastN2_{-1};
    int32_t lastJ_{-1};
    int32_t lastS2Offset_{-1};
    int32_t kvPingPong_{0};
    int32_t baseM_{128};
    int32_t baseN_{128};

    bool inMetaTask_{false};
    int32_t curB_{0};
    int32_t curN2_{0};
    int32_t curJ_{0};
    int32_t curG_{0};
    int32_t curCount_{0};
    int64_t curIdxBase_{0};
    int32_t curBlockS2Start_{0};
    int32_t curKvBlockLen_{0};
    int64_t curQPrefix_{0};
    int64_t curKvPrefix_{0};
    int64_t curActS1_{0};
    int64_t curActS2_{0};
    int32_t s1Offset_{0};
    int32_t s1Remain_{0};
    int32_t s2Offset_{0};
    int32_t s2Remain_{0};
};

} // namespace GSAG_ARC35
