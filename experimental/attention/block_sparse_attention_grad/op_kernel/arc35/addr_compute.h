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
using namespace AscendC;

namespace BSA_ARC35 {

// ============================================================================
// 本文件实现 Block Sparse Attention 反向（BSAG）算子的地址计算与任务调度。
//
// 层级关系：
//   AddrComputeModule  -> 按遍历顺序 (s1 -> s2 -> n1 -> batch) 切出 SingleBlock
//     └── SingleBlock  -> 在单个 SingleBlock 内沿 S1 方向逐 BaseBlock 推进
//                          跳过稀疏 mask 标记为无效的 BaseBlock
//
// 块大小关系（典型配置，以 arc35 为例）：
//   single_m  >= base_m  >= block_x
// 即一个 SingleBlock 由多个 BaseBlock 组成，一个 BaseBlock 又覆盖若干 mask block。
// 只有 mask 中标记为有效的 BaseBlock 才会下发到 cube 核进行实际计算。
// ============================================================================

class SingleBlock {
    /*
     *  Single块内只遍历S1方向。
     *  职责：在给定 [s1_idx, s1_idx+s1_len) 区间内，按 base_m 步长推进，
     *        跳过稀疏 mask 标记为无效的 base 块，逐个产出有效 base 块的 (起始idx, 长度)。
     */
public:
    __aicore__ inline void Reset(const ConstInfo &const_info, __gm__ uint8_t *block_sparse_mask,
                                 __gm__ int32_t *per_block_size_mask, const int32_t b_idx, const int32_t n1_idx,
                                 const int32_t s1_idx, const int32_t s2_idx, const int32_t s1_len,
                                 int32_t &base_s1_start_idx, int32_t &base_s1_len)
    /*
     * 参数：
     *   const_info           - 常量上下文（head 数、block 尺寸等）
     *   block_sparse_mask    - GM 上的稀疏 mask 指针
     *   per_block_size_mask  - GM 上的 PerBlockSizeMask 指针（maskType=1 时非空，否则 nullptr）
     *   b_idx / n1_idx       - 当前 batch / query head 索引
     *   s1_idx               - SingleBlock 在 S1 方向的起点
     *   s2_idx               - 当前 BaseBlock 在 S2 方向的起点（整个 SingleBlock 内固定）
     *   s1_len               - SingleBlock 在 S1 方向的长度
     * 出参：
     *   base_s1_start_idx    - 第一个有效 base 块的 S1 起点
     *   base_s1_len          - 第一个有效 base 块的 S1 长度
     */
    {
        /*
         * 功能：设置SingleBlock的起始索引和结束索引，同时更新base_s1_start_idx和base_s1_len。
         * base_s1_start_idx：表示下一个需要计算的base块的s1方向的起始索引。
         * base_s1_len：表示下一个需要计算的base块的s1方向的长度。
         * 默认起始位置必为有效块。
         */

        this->is_finish_ = false;                         // 重置完成标志，准备开始遍历
        this->s1_start_idx_ = s1_idx;                     // 记录 S1 起点
        this->s1_end_idx_ = s1_idx + s1_len;              // 记录 S1 终点（exclusive）
        this->per_block_size_mask_ = per_block_size_mask; // 保存 PerBlockSizeMask 指针供 Update 使用
        // 立即推进一次，吐出首个有效 base 块信息
        this->Update(const_info, block_sparse_mask, b_idx, n1_idx, s2_idx, base_s1_start_idx, base_s1_len);
    }

    __aicore__ inline void RecordRunTimeInfo(const RunTimeInfo &runTimeInfo)
    /*
     * 把外部传入的 runTimeInfo 整体快照到 runTimeInfoBak_，
     * 后续 UpdateRunTimeInfo 在同一 SingleBlock 内续推时复用这份上下文。
     */
    {
        this->runTimeInfoBak_.bIdx = runTimeInfo.bIdx;                         // 当前 batch
        this->runTimeInfoBak_.last_q_seq_sum = runTimeInfo.last_q_seq_sum;     // TND 下前置 q 累加
        this->runTimeInfoBak_.last_kv_seq_sum = runTimeInfo.last_kv_seq_sum;   // TND 下前置 kv 累加
        this->runTimeInfoBak_.cur_q_seq_len = runTimeInfo.cur_q_seq_len;       // 当前 batch q 长度
        this->runTimeInfoBak_.cur_kv_seq_len = runTimeInfo.cur_kv_seq_len;     // 当前 batch kv 长度
        this->runTimeInfoBak_.s1Idx = runTimeInfo.s1Idx;                       // base 块 S1 起点
        this->runTimeInfoBak_.s2Idx = runTimeInfo.s2Idx;                       // base 块 S2 起点
        this->runTimeInfoBak_.n1Idx = runTimeInfo.n1Idx;                       // query head
        this->runTimeInfoBak_.n2Idx = runTimeInfo.n2Idx;                       // kv head（GQA）
        this->runTimeInfoBak_.s1Len = runTimeInfo.s1Len;                       // base 块 S1 长度
        this->runTimeInfoBak_.s2Len = runTimeInfo.s2Len;                       // base 块 S2 长度
        this->runTimeInfoBak_.s1LenAlign = runTimeInfo.s1LenAlign;             // 16 对齐后的 S1 长度
        this->runTimeInfoBak_.s2LenAlign = runTimeInfo.s2LenAlign;             // 16 对齐后的 S2 长度
        this->runTimeInfoBak_.kv_ping_pong_idx = runTimeInfo.kv_ping_pong_idx; // KV 乒乓索引
    }

    template <uint32_t INPUT_LAYOUT>
    __aicore__ inline void UpdateRunTimeInfo(const ConstInfo &const_info, __gm__ uint8_t *block_sparse_mask,
                                             RunTimeInfo &runTimeInfo)
    {
        // 在已记录的 SingleBlock 上下文基础上推进到下一个有效 base 块，
        // 并刷新所有 GM 偏移（Q/KV/Lse/Sftg）。供同一 SingleBlock 内多次取任务使用。
        int32_t base_s1_start_idx, base_s1_len;
        // 用备份的 batch/head/s2 上下文推进 SingleBlock，得到下一个有效 base 块
        this->Update(const_info, block_sparse_mask, this->runTimeInfoBak_.bIdx, this->runTimeInfoBak_.n1Idx,
                     this->runTimeInfoBak_.s2Idx, base_s1_start_idx, base_s1_len);

        // 回填批次/head 上下文（这些字段在 SingleBlock 内不变）
        runTimeInfo.bIdx = this->runTimeInfoBak_.bIdx;
        runTimeInfo.last_q_seq_sum = this->runTimeInfoBak_.last_q_seq_sum;
        runTimeInfo.last_kv_seq_sum = this->runTimeInfoBak_.last_kv_seq_sum;
        runTimeInfo.cur_q_seq_len = this->runTimeInfoBak_.cur_q_seq_len;
        runTimeInfo.cur_kv_seq_len = this->runTimeInfoBak_.cur_kv_seq_len;
        // 用新推进出的 base 块坐标覆盖
        runTimeInfo.s1Idx = base_s1_start_idx;
        runTimeInfo.s2Idx = this->runTimeInfoBak_.s2Idx; // S2 在同一 SingleBlock 内固定
        runTimeInfo.n1Idx = this->runTimeInfoBak_.n1Idx;
        runTimeInfo.n2Idx = this->runTimeInfoBak_.n2Idx;
        runTimeInfo.s1Len = base_s1_len;
        runTimeInfo.s2Len = this->runTimeInfoBak_.s2Len;
        runTimeInfo.s1LenAlign = RoundUp(base_s1_len, 16); // cube 指令要求 16 元素对齐
        runTimeInfo.s2LenAlign = this->runTimeInfoBak_.s2LenAlign;
        // 刷新各 GM 偏移（根据 layout 计算）
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
        runTimeInfo.need_compute = 1;              // 续推任务仍需计算
        runTimeInfo.need_copy_kv = 0;              // KV 已在首次 Reset 时搬运，无需重复
        runTimeInfo.is_singlekv_last = is_finish_; // 当前 SingleBlock 是否已吐完
        runTimeInfo.kv_ping_pong_idx = this->runTimeInfoBak_.kv_ping_pong_idx; // 复用同一乒乓槽
    }

    __aicore__ inline bool IsFinish()
    {
        return is_finish_; // 当前 SingleBlock 是否遍历完
    }

    __aicore__ inline void SetBaseBlock(uint32_t s1_base_size)
    {
        this->s1_base_size_ = s1_base_size; // 设置 base 块在 S1 方向的步长（= base_m）
    }

private:
    __aicore__ inline void Update(const ConstInfo &const_info, __gm__ uint8_t *block_sparse_mask, const int32_t b_idx,
                                  const int32_t n1_idx, const int32_t s2Idx, int32_t &base_s1_start_idx,
                                  int32_t &base_s1_len)
    /*
     * 参数：
     *   const_info        - 常量上下文（含 block_x/block_y 用于算 mask block 索引）
     *   block_sparse_mask - GM 上的稀疏 mask
     *   b_idx / n1_idx    - 当前 batch / query head
     *   s2Idx             - 当前 S2 起点（SingleBlock 内固定）
     * 出参：
     *   base_s1_start_idx - 本次吐出的有效 base 块 S1 起点
     *   base_s1_len       - 本次吐出的有效 base 块 S1 长度
     */
    {
        /*
         * 功能：返回有效base块的起始地址和长度。并判断是否遍历完所有的s1方向。
         *
         * 推进策略：
         *   1) 先吐出当前 base 块（s1_start_idx_ 处，长度按 base_m 对齐到剩余区间）。
         *   2) 推进 s1_start_idx_ 到下一个 base 块起点。
         *   3) 若已到达 s1_end_idx_，标记完成并返回。

         *   4) 否则用 while 循环连续跳过所有被稀疏 mask 标记为无效的 base 块，
         *      直到找到下一个有效块或到达区间末尾。
         */
        // --- 步骤 1：吐出当前 base 块 ---
        base_s1_start_idx = this->s1_start_idx_;                                          // 记录起点
        base_s1_len = GetBlockLen(this->s1_start_idx_, this->s1_end_idx_, s1_base_size_); // 算长度（末尾不足则截断）
        // 当 base 块跨越 mask block 边界时，截断到 mask 边界
        // 避免 blockShape 不能被 base_m 整除时（如 192），base 块跨越两个不同 mask 值的 block
        int32_t q_block_idx_cur = base_s1_start_idx / const_info.block_x;   // 起点所在的 mask block 索引
        int32_t mask_boundary = (q_block_idx_cur + 1) * const_info.block_x; // 当前 mask block 的右边界
        if (base_s1_start_idx + base_s1_len > mask_boundary) {
            base_s1_len = mask_boundary - base_s1_start_idx; // 截断到 mask 边界
        }
        // PerBlockSizeMask (maskType=1): 用 actualSizeX 截断到当前 mask block 的有效长度
        // actualSizeX < block_x 时，[actualSizeX, block_x) 为 padding，不参与计算
        if (const_info.has_per_block_size_mask && this->per_block_size_mask_ != nullptr) {
            int32_t actual_size_x =
                GetPerBlockSizeMaskValue(const_info, this->per_block_size_mask_, b_idx, n1_idx, q_block_idx_cur, 0);
            base_s1_len = ClipLenByActualSize(base_s1_start_idx, base_s1_len, const_info.block_x, actual_size_x);
        }
        this->s1_start_idx_ += base_s1_len; // 推进到下一个 base 块起点

        // --- 步骤 2：检查是否到达 SingleBlock 末尾 ---
        if (this->s1_start_idx_ >= this->s1_end_idx_) {
            // 下一块遍历完，提前退出
            this->is_finish_ = true;
            return;
        }

        // --- 步骤 3：计算下一个 base 块对应的 mask block 索引 ---
        int32_t q_block_idx = this->s1_start_idx_ / const_info.block_x; // Q 方向 mask block 索引
        int32_t kv_block_idx = s2Idx / const_info.block_y;              // KV 方向 mask block 索引

        // --- 步骤 4：连续跳过无效 base 块 ---
        // 连续跳过无效 base 块，直到命中有效块或越过 SingleBlock 边界
        // 无效判定：block_sparse_mask=0，或 PerBlockSizeMask 截断后长度<=0（actualSizeX=0 或起点在 padding 区）
        while (true) {
            // 查 mask：当前 (q_block_idx, kv_block_idx) 是否标记为计算
            if (IsValidBlock(const_info, b_idx, n1_idx, q_block_idx, kv_block_idx, block_sparse_mask)) {
                // block_sparse_mask 有效，但还需检查 PerBlockSizeMask 是否把整个 block 标为 padding
                if (const_info.has_per_block_size_mask && this->per_block_size_mask_ != nullptr) {
                    int32_t actual_size_x =
                        GetPerBlockSizeMaskValue(const_info, this->per_block_size_mask_, b_idx, n1_idx, q_block_idx, 0);
                    int32_t local_idx = this->s1_start_idx_ % const_info.block_x;
                    if (actual_size_x - local_idx <= 0) {
                        // actualSizeX=0 或起点已在 padding 区，视为无效块，继续跳过
                    } else {
                        break; // block_sparse_mask 有效且 PerBlockSizeMask 也有效，命中有效块
                    }
                } else {
                    break; // 无 PerBlockSizeMask，block_sparse_mask 有效即命中
                }
            }
            // 无效块：跳过
            int32_t tmp_s1_len = GetBlockLen(this->s1_start_idx_, this->s1_end_idx_, s1_base_size_);
            // 跳过时也需截断到 mask 边界，避免跳过属于下一个有效 mask block 的区间
            int32_t skip_q_block_idx = this->s1_start_idx_ / const_info.block_x;
            int32_t skip_mask_boundary = (skip_q_block_idx + 1) * const_info.block_x;
            if (this->s1_start_idx_ + tmp_s1_len > skip_mask_boundary) {
                tmp_s1_len = skip_mask_boundary - this->s1_start_idx_;
            }
            this->s1_start_idx_ += tmp_s1_len;                      // 推进到下一个 base 块
            q_block_idx = this->s1_start_idx_ / const_info.block_x; // 更新 mask block 索引
            if (this->s1_start_idx_ >= this->s1_end_idx_) {
                break; // 越过 SingleBlock 边界，退出
            }
        }

        // --- 步骤 5：跳过后再次检查是否到达末尾 ---
        if (this->s1_start_idx_ >= this->s1_end_idx_) {
            this->is_finish_ = true; // 所有 base 块已遍历完
        }
    }

    // ---- SingleBlock 状态 ----
    int32_t s1_start_idx_{0};                      // 下一个待处理的 base 块在 S1 方向的起点
    int32_t s1_end_idx_{0};                        // SingleBlock 在 S1 方向的终点（ exclusive ）
    int32_t s1_base_size_{0};                      // base 块在 S1 方向的步长（= base_m）
    bool is_finish_{true};                         // 当前 SingleBlock 是否已遍历完
    RunTimeInfo runTimeInfoBak_;                   // 上一次 Reset 时的上下文，供 UpdateRunTimeInfo 续推
    __gm__ int32_t *per_block_size_mask_{nullptr}; // PerBlockSizeMask GM 指针（maskType=1 时非空）
};

template <typename BSA_TYPE>
class AddrComputeModule {
    /*
     * 职责：整个 BSAG 任务的地址计算与任务下发调度器。
     *
     * 遍历顺序：s1 -> s2 -> n1 -> batch（见 InitStartIdx）。
     * 调度方式：按 SingleBlock 粒度产出任务，每个有效 SingleBlock 内部由 SingleBlock
     *           继续按 base_m 粒度产出有效 BaseBlock 任务。
     * 多核分配：current_cube_idx_ 在所有 cube 核之间轮询分配任务，
     *           每个核只接收 current_cube_idx_ % cube_core_num_ == cube_core_idx_ 的任务。
     *           每凑齐 cube_core_num_ 个任务就暂停下发，等下一轮 GetRunTimeInfo 调用。
     */
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    // ---- GM 输入地址 ----
    GM_ADDR actualQseqlen_;                // TND 布局下每个 batch 的实际 q 长度数组
    GM_ADDR actualKvseqlen_;               // TND 布局下每个 batch 的实际 kv 长度数组
    GM_ADDR blockSparseMask_;              // block 级稀疏 mask，1=计算 / 0=跳过
    GM_ADDR perBlockSizeMask_;             // maskType=1 时的 per-block actualSize mask，INT32 4D [B,N,maxBlockNum,2]
    bool has_per_block_size_mask_ = false; // 是否启用 PerBlockSizeMask（接口已通，实际计算暂未实现）
    int32_t max_block_num_ = 0;            // mask 最后一维的 block 数量

    // ---- 形状参数 ----
    int32_t batch_num_;   // batch 数量
    int32_t q_seq_len_;   // Q 序列长度（非 TND 用）
    int32_t kv_seq_len_;  // KV 序列长度（非 TND 用）
    int32_t q_group_;     // GQA group size，n2 = n1 / q_group
    int32_t q_head_num_;  // query head 数量
    int32_t kv_head_num_; // kv head 数量（GQA 下 < q_head_num_）
    int32_t head_dim_;    // 每个头的维度

    // ---- 遍历游标（外层）----
    int32_t bIdx_{0};  // 当前 batch
    int32_t s1Idx_{0}; // 当前 SingleBlock 在 S1 方向的起点
    int32_t s2Idx_{0}; // 当前 BaseBlock 在 S2 方向的起点
    int32_t n1Idx_{0}; // 当前 query head 索引

    // ---- TND 实际序列长度缓存 ----
    int32_t cur_q_seq_len_{0};   // 当前 batch 的 q 序列长度
    int32_t cur_kv_seq_len_{0};  // 当前 batch 的 kv 序列长度
    int32_t last_q_seq_sum_{0};  // TND 下前面 batch q 长度累加（用于 GM 偏移）
    int32_t last_kv_seq_sum_{0}; // TND 下前面 batch kv 长度累加

    // ---- 多核调度 ----
    int32_t cube_core_idx_{0};    // 本核在 cube 核集合中的索引
    int32_t cube_core_num_{0};    // cube 核总数
    int32_t current_cube_idx_{0}; // 全局轮询计数器，配合 cube_core_idx_ 做任务分配

    // ---- 块大小 ----
    int32_t base_m_{0};           // BaseBlock 在 S1 方向长度（实际下发给单核的最小粒度）
    int32_t base_n_{0};           // BaseBlock 在 S2 方向长度
    int32_t first_loop_{0};       // 首次调用 InitStartIdx 时的标志位
    int32_t q_block_num_{0};      // Q 方向 mask block 数 = ceil(q_seq_len / block_x)
    int32_t kv_block_num_{0};     // KV 方向 mask block 数
    int32_t block_x_{0};          // mask block 在 Q 方向尺寸
    int32_t block_y_{0};          // mask block 在 KV 方向尺寸
    int32_t single_m_{0};         // SingleBlock 在 S1 方向长度（>= base_m，一个 Single 含多个 Base）
    int32_t kv_ping_pong_idx_{0}; // KV 双缓冲乒乓索引，0/1 交替

    SingleBlock single_block_; // 内层 SingleBlock 推进器
    ConstInfo const_info_;     // 传递给 SingleBlock / IsValidBlock 的常量上下文

public:
    __aicore__ inline void Init(const TILING_CLASS *tilingData, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                GM_ADDR blockSparseMask, GM_ADDR perBlockSizeMask)
    /*
     * 初始化调度器：从 tilingData 拷贝形状/块大小参数，记录 GM 地址，
     * 计算 mask block 数量，填充 const_info_，并确定本核的 cube_core_idx_。
     * perBlockSizeMask: maskType=1 时的 per-block actualSize mask GM 地址（接口已通，实际计算暂未实现）。
     */
    {
        // ---- 从 tiling 数据拷贝形状参数 ----
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->cube_core_num_ = tilingData->cubeCoreNum;
        // ---- 记录 GM 输入地址 ----
        this->actualQseqlen_ = actualQseqlen;
        this->actualKvseqlen_ = actualKvseqlen;
        this->blockSparseMask_ = blockSparseMask;
        this->perBlockSizeMask_ = perBlockSizeMask;
        this->has_per_block_size_mask_ = (tilingData->hasPerBlockSizeMask != 0);
        this->max_block_num_ = tilingData->maxBlockNum;
        // ---- 从 tiling 数据拷贝块大小参数 ----
        this->block_x_ = tilingData->BlockX;
        this->block_y_ = tilingData->BlockY;
        this->base_m_ = tilingData->baseM;
        this->base_n_ = tilingData->baseN;
        this->single_m_ = tilingData->singleM;
        single_block_.SetBaseBlock(this->base_m_); // 把 base_m 传给 SingleBlock 推进器

        // ---- 根据布局计算 mask block 数量 ----
        if constexpr (INPUT_LAYOUT == TND) {
            // TND：变长序列，需要遍历所有 batch 取最大长度
            UpdateSeqLen();
            int32_t max_q_seq_len_ = 0;
            int32_t max_kv_seq_len_ = 0;
            for (int32_t i = 0; i < batch_num_; i++) {
                int64_t q_seq_len = GetSeqLen(i, actualQseqlen_);   // 读第 i 个 batch 的 q 长度
                int64_t kv_seq_len = GetSeqLen(i, actualKvseqlen_); // 读第 i 个 batch 的 kv 长度
                max_q_seq_len_ = max(max_q_seq_len_, q_seq_len);    // 取最大
                max_kv_seq_len_ = max(max_kv_seq_len_, kv_seq_len);
            }
            q_block_num_ = CeilDiv(max_q_seq_len_, block_x_);   // Q 方向 mask block 数
            kv_block_num_ = CeilDiv(max_kv_seq_len_, block_y_); // KV 方向 mask block 数
        } else {
            // 非 TND：定长序列，直接用 q_seq_len_ / kv_seq_len_
            cur_q_seq_len_ = q_seq_len_;
            cur_kv_seq_len_ = kv_seq_len_;
            last_q_seq_sum_ = 0; // 非 TND 无前置累加
            last_kv_seq_sum_ = 0;
            q_block_num_ = CeilDiv(q_seq_len_, block_x_);
            kv_block_num_ = CeilDiv(kv_seq_len_, block_y_);
        }

        // ---- 填充常量上下文（传给 SingleBlock / IsValidBlock）----
        const_info_.q_head_num = q_head_num_;
        const_info_.kv_head_num = kv_head_num_;
        const_info_.block_x = block_x_;
        const_info_.block_y = block_y_;
        const_info_.head_dim = head_dim_;
        const_info_.q_block_num = q_block_num_;
        const_info_.kv_block_num = kv_block_num_;
        const_info_.max_block_num = max_block_num_;
        const_info_.has_per_block_size_mask = has_per_block_size_mask_;

        // ---- 确定本核在 cube 核集合中的索引 ----
        if ASCEND_IS_AIC {
            this->cube_core_idx_ = GetBlockIdx(); // AIC：block idx 直接作为 cube idx
        }
        if ASCEND_IS_AIV {
            this->cube_core_idx_ = GetBlockIdx() / 2; // AIV：2 个 vector 核共享 1 个 cube，故除 2
        }
    }

    __aicore__ inline void GetRunTimeInfo(RunTimeInfo &runTimeInfo)
    {
        /*
         * 调度入口：每次调用产出下一个需要本核处理的任务到 runTimeInfo。
         *
         * 优先级：
         *   1) 如果当前 SingleBlock 还没遍历完，直接从里面取下一个有效 BaseBlock。
         *   2) 否则进入外层 while 循环，按 s1->s2->n1->batch 顺序找下一个有效 SingleBlock，
         *      通过 RunTimeInfoRecord 把它登记到本核任务，并初始化 SingleBlock 推进器。
         *
         * 多核负载均衡：
         *   current_cube_idx_ 是全局计数器，每产出一个候选任务自增一次；
         *   只有 current_cube_idx_ % cube_core_num_ == cube_core_idx_ 的任务才会真正下发给本核。
         *   每凑齐 cube_core_num_ 个任务就跳出，把控制权交还上层让其它核也来取任务。
         *
         * 注意：若 SingleBlock 内全为无效 base 块，IsValidSingleBlock 返回 false，
         *       不产生任务，current_cube_idx_ 也不自增，继续找下一个 SingleBlock。
         */
        runTimeInfo.need_compute = 0; // 默认本次不产生计算任务

        // --- 优先级 1：当前 SingleBlock 还有剩余 base 块 ---
        if (!single_block_.IsFinish()) {
            // 优先把当前 SingleBlock 内剩余的 base 块吐完（不进外层 while）
            single_block_.UpdateRunTimeInfo<INPUT_LAYOUT>(const_info_, blockSparseMask_, runTimeInfo);
            return;
        }

        // --- 优先级 2：外层 while 循环找下一个有效 SingleBlock ---
        while (true) {
            // 推进游标到下一个 SingleBlock；返回 true 表示整个任务空间遍历完毕
            if (InitStartIdx()) {
                break; // 所有 batch/head/s1/s2 遍历完，退出
            }
            int32_t vaild_s1_idx;
            int32_t vaild_s1_len;
            int32_t s2_len = GetBlockLen(s2Idx_, cur_kv_seq_len_, base_n_); // 当前 S2 方向 base 块长度
            // 当 base 块跨越 mask block 边界时，截断到 mask 边界（与 S1 方向同理）
            // 避免 blockShape 不能被 base_n 整除时（如 192），base 块跨越两个不同 mask 值的 block
            int32_t kv_block_idx_cur = s2Idx_ / block_y_;
            int32_t kv_mask_boundary = (kv_block_idx_cur + 1) * block_y_;
            if (s2Idx_ + s2_len > kv_mask_boundary) {
                s2_len = kv_mask_boundary - s2Idx_; // 截断到 mask 边界
            }
            // PerBlockSizeMask (maskType=1): 用 actualSizeY 截断到当前 KV mask block 的有效长度
            // actualSizeY < block_y 时，[actualSizeY, block_y) 为 padding，不参与计算
            if (has_per_block_size_mask_) {
                int32_t actual_size_y =
                    GetPerBlockSizeMaskValue(const_info_, reinterpret_cast<__gm__ int32_t *>(perBlockSizeMask_), bIdx_,
                                             n1Idx_, kv_block_idx_cur, 1);
                s2_len = ClipLenByActualSize(s2Idx_, s2_len, block_y_, actual_size_y);
            }

            // 检查当前 SingleBlock 内是否至少含一个有效 base 块
            // PerBlockSizeMask 截断后 s2_len 可能为 0（actualSizeY=0），此时不下发任务
            if (s2_len > 0 && IsValidSingleBlock(vaild_s1_idx, vaild_s1_len)) {
                // 当前 SingleBlock 至少含一个有效 base 块，登记为一次任务
                RunTimeInfoRecord(runTimeInfo, vaild_s1_idx, s2Idx_, vaild_s1_len, s2_len);
            }

            // 检查是否已为所有 cube 核各分配了一个任务
            if (current_cube_idx_ && current_cube_idx_ % cube_core_num_ == 0) {
                // 已为所有 cube 核各分配了一个任务，本批次结束
                current_cube_idx_ = 0; // 重置计数器，下一轮重新轮询
                break;
            }
        }
    }

private:
    __aicore__ inline bool IsValidSingleBlock(int32_t &vaild_s1_idx, int32_t &vaild_s1_len)
    {
        /*
         * 功能：判断singleBlock内是否存在有效base块，并返回第一块有效base块的起始idx和长度
         *
         * 实现：以 block_x 为步长在 S1 方向扫描整个 SingleBlock，
         *      一旦 mask 命中就记录 [s1_idx, s1_end_idx) 区间返回。
         *      （后续真正切 base_m 粒度的工作交给 SingleBlock.Update）
         */
        int32_t s1_len = GetBlockLen(s1Idx_, cur_q_seq_len_, single_m_); // 当前 SingleBlock 的 S1 长度
        int32_t s1_end_idx = s1Idx_ + s1_len;                            // SingleBlock 的 S1 终点
        int32_t block_y_idx = s2Idx_ / block_y_;                         // 当前 KV 方向 mask block 索引
        bool find_vaild_block = false;

        // 以 block_x 为步长扫描 SingleBlock 内所有 mask block
        for (int32_t s1_idx = s1Idx_; s1_idx < s1_end_idx; s1_idx += block_x_) {
            int32_t block_x_idx = s1_idx / block_x_; // 当前 Q 方向 mask block 索引
            // 查 mask：(b, n1, block_x_idx, block_y_idx) 是否标记为计算
            if (IsValidBlock(const_info_, bIdx_, n1Idx_, block_x_idx, block_y_idx, blockSparseMask_)) {
                // PerBlockSizeMask: actualSizeX=0 的 block 视为无效（无有效数据）
                if (has_per_block_size_mask_) {
                    int32_t actual_size_x =
                        GetPerBlockSizeMaskValue(const_info_, reinterpret_cast<__gm__ int32_t *>(perBlockSizeMask_),
                                                 bIdx_, n1Idx_, block_x_idx, 0);
                    if (actual_size_x <= 0) {
                        continue; // actualSizeX=0，跳过此 block 继续扫描
                    }
                }
                find_vaild_block = true;
                vaild_s1_idx = s1_idx;              // 记录第一个有效 mask block 的 S1 起点
                vaild_s1_len = s1_end_idx - s1_idx; // 长度延伸到 SingleBlock 末尾（后续由 base_m 细分）
            }
            if (find_vaild_block) {
                break; // 找到第一个有效块即可，无需继续扫描
            }
        }

        return find_vaild_block;
    }

    __aicore__ inline void UpdateSeqLen()
    {
        /*
         * 仅 TND 布局使用：从 GM 读取当前 batch 的实际 q/kv 长度，
         * 跳过长度为 0 的空 batch，并刷新 last_*_seq_sum_ 用于 GM 偏移计算。
         */
        if constexpr (INPUT_LAYOUT != TND) {
            return; // 非 TND 直接返回，序列长度在 Init 中已设置
        }
        cur_q_seq_len_ = GetSeqLen(bIdx_, actualQseqlen_);   // 读当前 batch 的 q 长度
        cur_kv_seq_len_ = GetSeqLen(bIdx_, actualKvseqlen_); // 读当前 batch 的 kv 长度
        // 跳过空 batch（q 或 kv 长度为 0）
        while ((cur_q_seq_len_ == 0 || cur_kv_seq_len_ == 0) && bIdx_ < batch_num_ - 1) {
            bIdx_++;
            cur_q_seq_len_ = GetSeqLen(bIdx_, actualQseqlen_);
            cur_kv_seq_len_ = GetSeqLen(bIdx_, actualKvseqlen_);
        }
        // 计算前置 batch 的累加长度（用于 TND 的 GM 偏移）
        last_q_seq_sum_ = bIdx_ > 0 ? GetSeqTotalLen(bIdx_ - 1, actualQseqlen_) : 0;
        last_kv_seq_sum_ = bIdx_ > 0 ? GetSeqTotalLen(bIdx_ - 1, actualKvseqlen_) : 0;
    }

    __aicore__ inline bool InitStartIdx()
    {
        /*
         * 推进外层游标到下一个 SingleBlock 起点。
         * 遍历顺序：s1 -> s2 -> n1 -> batch（最内层 s1 步长为 single_m）。
         * 返回值：true 表示所有任务遍历完毕，false 表示已移动到新的有效起点。
         *
         * 注意：首次调用（first_loop_==0）只记录当前步长，不立即推进，
         *       避免跳过最开头的 [0, *) 任务。
         */
        // 遍历顺序 s1->s2->n1->batch
        int32_t recoderS1 = GetBlockLen(s1Idx_, cur_q_seq_len_, single_m_); // 当前 SingleBlock 的 S1 长度
        int32_t recoderS2 = GetBlockLen(s2Idx_, cur_kv_seq_len_, base_n_);  // 当前 BaseBlock 的 S2 长度
        // S2 方向也需截断到 mask 边界，与 GetRunTimeInfo 中的截断保持一致
        // 否则 s2Idx_ 推进时会跨过 mask 边界，漏算属于下一个 mask block 的区间
        int32_t init_kv_block_idx = s2Idx_ / block_y_;
        int32_t init_kv_mask_boundary = (init_kv_block_idx + 1) * block_y_;
        if (s2Idx_ + recoderS2 > init_kv_mask_boundary) {
            recoderS2 = init_kv_mask_boundary - s2Idx_;
        }

        // 首次调用：只记录步长，不推进（避免跳过 [0,*) 任务）
        if (unlikely(first_loop_ == 0)) {
            first_loop_ = 1;
            return false;
        }

        // 1) S1 方向还有剩余 SingleBlock
        if (s1Idx_ < cur_q_seq_len_ - recoderS1) {
            s1Idx_ += recoderS1; // 推进到下一个 SingleBlock
            return false;
        }

        // 2) S2 方向还有剩余 BaseBlock
        if (s2Idx_ < cur_kv_seq_len_ - recoderS2) {
            s1Idx_ = 0;          // S1 重置到开头
            s2Idx_ += recoderS2; // 推进到下一个 BaseBlock
            return false;
        }

        // 3) 切到下一个 query head
        if (n1Idx_ < q_head_num_ - 1) {
            s1Idx_ = 0; // S1 重置
            s2Idx_ = 0; // S2 重置
            n1Idx_++;   // 推进到下一个 head
            return false;
        }

        // 4) 切到下一个 batch（TND 下还要刷新序列长度）
        if (bIdx_ < batch_num_ - 1) {
            s1Idx_ = 0;
            s2Idx_ = 0;
            n1Idx_ = 0;
            bIdx_++; // 推进到下一个 batch
            if constexpr (INPUT_LAYOUT == TND) {
                UpdateSeqLen(); // TND：刷新 cur_*_seq_len_ 和 last_*_seq_sum_
            }
            return false;
        }

        return true; // 所有遍历完成
    }

    __aicore__ inline void RunTimeInfoRecord(RunTimeInfo &runTimeInfo, int32_t vaild_s1_idx, int32_t vaild_s2_idx,
                                             int32_t vaild_s1_len, int32_t vaild_s2_len)
    {
        /*
         * 把一个有效 SingleBlock 登记为本核任务，并初始化 SingleBlock 推进器。
         *
         * 多核分配：current_cube_idx_ 全局计数；只有轮到本核
         *           (current_cube_idx_ % cube_core_num_ == cube_core_idx_) 才真正登记，
         *           否则只自增计数器直接返回（该任务由其它核处理）。
         *
         * KV 乒乓：每登记一个任务翻转一次 kv_ping_pong_idx_ (0/1 交替)，
         *          配合 need_copy_kv=1 实现搬运与计算的双缓冲隐藏延迟。
         *          is_singlekv_last 标记当前 SingleBlock 是否已吐完最后一个 base 块，
         *          上层据此判断何时可以切换 KV 缓冲。
         */
        // 不属于本核的任务，仅推进全局计数
        if (current_cube_idx_ % cube_core_num_ != cube_core_idx_) {
            current_cube_idx_++; // 该任务由其它核处理，只自增计数器
            return;
        }
        // --- 本核任务：重置 SingleBlock 推进器 ---
        // 重置 SingleBlock 推进器，并立即取出第一个有效 base 块的信息
        int32_t base_s1_idx;
        int32_t base_s1_len;
        single_block_.Reset(const_info_, blockSparseMask_, reinterpret_cast<__gm__ int32_t *>(perBlockSizeMask_), bIdx_,
                            n1Idx_, vaild_s1_idx, vaild_s2_idx, vaild_s1_len, base_s1_idx, base_s1_len);
        int32_t n2Idx = n1Idx_ / q_group_; // GQA: kv head = q head / group

        // --- 填充 runTimeInfo 的批次/head/序列上下文 ---
        runTimeInfo.bIdx = bIdx_;
        runTimeInfo.last_q_seq_sum = last_q_seq_sum_;
        runTimeInfo.last_kv_seq_sum = last_kv_seq_sum_;
        runTimeInfo.cur_q_seq_len = cur_q_seq_len_;
        runTimeInfo.cur_kv_seq_len = cur_kv_seq_len_;
        // --- 填充本次 base 块坐标 ---
        runTimeInfo.s1Idx = base_s1_idx;
        runTimeInfo.s2Idx = vaild_s2_idx;
        runTimeInfo.n1Idx = n1Idx_;
        runTimeInfo.n2Idx = n2Idx;
        runTimeInfo.s1Len = base_s1_len;
        runTimeInfo.s2Len = vaild_s2_len;
        runTimeInfo.s1LenAlign = RoundUp(base_s1_len, 16); // 16 元素对齐，适配 cube 指令要求
        runTimeInfo.s2LenAlign = RoundUp(vaild_s2_len, 16);
        // --- 计算各 GM 偏移 ---
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
        // --- 设置控制标志 ---
        runTimeInfo.need_compute = 1;                            // 本次任务需要真正计算
        runTimeInfo.need_copy_kv = 1;                            // 触发 KV 搬运（首次进入 SingleBlock）
        runTimeInfo.kv_ping_pong_idx = kv_ping_pong_idx_;        // 当前乒乓槽
        runTimeInfo.is_singlekv_last = single_block_.IsFinish(); // SingleBlock 是否只含一个 base 块
        single_block_.RecordRunTimeInfo(runTimeInfo);            // 备份上下文供后续 UpdateRunTimeInfo 续推
        kv_ping_pong_idx_ = 1 - kv_ping_pong_idx_;               // 翻转乒乓缓冲（下个任务用另一个槽）
        current_cube_idx_++;                                     // 本核任务登记完成，推进全局计数
    }
};
} // namespace BSA_ARC35
